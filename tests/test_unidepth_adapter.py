"""CPU adapter contracts with tiny checkpoint tensors; no real UniDepth weights."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.inference import model_name, prediction_depths, build_inference_model
import models.unidepth_adapter as adapter
from tools.prepare_unidepth import import_archive


class FakeUni(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.last_input = None

    def infer(self, rgb, camera=None, normalize=True):
        self.last_input = rgb.clone()
        self.last_camera, self.last_normalize = camera, normalize
        self.had_grad = torch.is_grad_enabled()
        return {'depth': 2 + self.scale * rgb.mean(1, keepdim=True) / 255.0}


def fixture(root):
    package = root / 'models/unidepth'
    source = package / 'models/unidepthv2/unidepthv2.py'
    source.parent.mkdir(parents=True)
    source.write_text('# official-source stand-in, used only by adapter unit tests\n')
    weights = root / 'weights'
    weights.mkdir()
    config = dict(model=dict(name='UniDepthV2', pixel_encoder=dict(name='dinov2_vitl14', pretrained=None)))
    (weights / 'config.json').write_text(json.dumps(config))
    torch.save(FakeUni(config).state_dict(), weights / 'pytorch_model.bin')
    mod = types.ModuleType('unidepth')
    mod.__path__ = [str(package)]
    models = types.ModuleType('unidepth.models')
    models.UniDepthV2 = FakeUni
    return weights, {'unidepth': mod, 'unidepth.models': models}


class UniDepthTests(unittest.TestCase):
    def test_float_rgb_scaling_native_depth_and_frozen_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); weights, modules = fixture(root)
            with patch.dict(sys.modules, modules), patch.object(adapter, '__file__', str(root/'models/unidepth_adapter.py')):
                model = adapter.UniDepthV2Adapter(weights, resolution_level=7)
                rgb = torch.full((1,3,4,6), .00001, requires_grad=True)
                out = model(rgb)
                torch.testing.assert_close(model.net.last_input, rgb * 255)
                self.assertGreater(float(model.net.last_input.min()), 0)  # never quantized to uint8 zero
                torch.testing.assert_close(out['metric_depth'], 2 + rgb.mean(1,keepdim=True))
                torch.testing.assert_close(out['raw_relative'] * out['metric_depth'], torch.ones((1,1,4,6)))
                self.assertIsNone(model.net.last_camera)
                self.assertTrue(model.net.last_normalize)
                self.assertFalse(model.net.had_grad)
                self.assertFalse(out['metric_depth'].requires_grad)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                model.train(True)
                self.assertFalse(model.training)
                self.assertFalse(model.net.training)
                self.assertEqual(model.net.resolution_level, 7)
                self.assertEqual(model.inference_metadata['weights_sha256'], adapter.sha256(weights/'pytorch_model.bin'))

    def test_native_metric_depth_is_not_floored_as_dav2_inverse(self):
        depth = torch.tensor([[[[2000., 4000.]]]])
        output = dict(raw_relative=depth.reciprocal(), metric_depth=depth)
        _, actual = prediction_depths(output, {'name': 'unidepth_v2_vitl14', 'inverse_epsilon': .001})
        torch.testing.assert_close(actual, depth)
        _, dav2 = prediction_depths(output, {'name':'dav2_vitl','inverse_epsilon':.001})
        torch.testing.assert_close(dav2, torch.full_like(depth,1000.))

    def test_dav2_zero_and_positive_outputs_preserve_historical_conversion(self):
        q = torch.tensor([[[[0., .0001, .5, 2.]]]])
        _, depth = prediction_depths({'raw_relative':q}, {})
        torch.testing.assert_close(depth, q.clamp_min(.001).reciprocal(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model_name({'name':'marigold-not-implemented'})

    def test_missing_and_wrong_checkpoint_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError): adapter.checkpoint_files(root)
            weights, modules = fixture(root)
            for pretrained in ['', False, 'backbone.pth']:
                cfg=json.loads((weights/'config.json').read_text())
                cfg['model']['pixel_encoder']['pretrained']=pretrained
                (weights/'config.json').write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError,'null'):
                    adapter.UniDepthV2Adapter(weights)
            with self.assertRaises(ValueError): adapter.UniDepthV2Adapter(weights,resolution_level=10)

    def test_bad_input_and_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); weights,modules=fixture(root)
            with patch.dict(sys.modules,modules),patch.object(adapter,'__file__',str(root/'models/unidepth_adapter.py')):
                model=adapter.UniDepthV2Adapter(weights)
                for value in [-.1,1.1,float('nan')]:
                    with self.assertRaises(ValueError): model(torch.full((1,3,4,6),value))
                for value in [0.,-1.,float('nan')]:
                    with patch.object(model.net,'infer',return_value={'depth':torch.full((1,1,4,6),value)}):
                        with self.assertRaises(ValueError): model(torch.ones(1,3,4,6))
                with patch.object(model.net,'infer',return_value={'depth':torch.ones(1,1,2,3)}):
                    with self.assertRaises(ValueError): model(torch.ones(1,3,4,6))

    def test_strict_weight_loading_and_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weights,modules=fixture(root)
            with patch.dict(sys.modules,modules),patch.object(adapter,'__file__',str(root/'models/unidepth_adapter.py')):
                a=adapter.UniDepthV2Adapter(weights).inference_metadata
                source=root/'models/unidepth/models/unidepthv2/unidepthv2.py'
                source.write_text('# changed source\n')
                b=adapter.UniDepthV2Adapter(weights).inference_metadata
                self.assertNotEqual(a['source_sha256'],b['source_sha256'])
                torch.save({'unexpected':torch.ones(())},weights/'pytorch_model.bin')
                with self.assertRaises(RuntimeError): adapter.UniDepthV2Adapter(weights)

    def test_source_import_rejects_collision_and_traversal(self):
        def archive(files):
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w') as z:
                for name,value in files.items(): z.writestr(name,value)
            return buffer.getvalue()
        content={'upstream/unidepth/models/unidepthv2/unidepthv2.py':'# source',
                 'upstream/LICENSE':'license','upstream/weights.bin':'not imported'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            receipt=import_archive(archive(content),root)
            self.assertEqual(len(receipt['files']),2)
            self.assertFalse((root/'weights.bin').exists())
            content['upstream/unidepth/models/unidepthv2/unidepthv2.py']='# changed'
            with self.assertRaises(FileExistsError): import_archive(archive(content),root)
            with self.assertRaises(ValueError): import_archive(archive({'upstream/../escape.py':'bad'}),root)

    def test_severity_pipeline_evaluates_native_depth(self):
        from tests.test_sweep_execution import fixture as sweep_fixture, arguments
        from diagnose_severity import run
        from analysis.sweep_store import read_checkpoint
        import yaml

        class MetricModel(torch.nn.Module):
            def forward(self,rgb):
                depth=2000 + rgb.mean(1,keepdim=True)*1000
                return dict(raw_relative=depth.reciprocal(),metric_depth=depth)

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,_=sweep_fixture(root)
            cfg=yaml.safe_load(config.read_text());cfg['model']['name']='unidepth_v2_vitl14'
            config.write_text(yaml.safe_dump(cfg))
            args=arguments(config,root/'run',['--conditions','LN','LNG'])
            with contextlib.redirect_stdout(io.StringIO()): run(args,MetricModel())
            contract=json.loads((root/'run/contract.json').read_text())
            for sample in contract['samples']:
                clean=[r for r in read_checkpoint(root/'run',sample['id'])['rows'] if r['condition']=='clean']
                # GT is finer than the prediction grid, so a few nearest-resize
                # ties are legitimate. The erroneous floor would give .5.
                self.assertTrue(all(r['gt_ordinal_error']<.01 for r in clean))
            # Flooring reciprocal depth at .001 would collapse all these depths to
            # 1000 and incorrectly produce 50% tied-pair error instead of zero.
            self.assertIn('native metric',contract['model_identity']['output'])


if __name__ == '__main__':
    unittest.main()
