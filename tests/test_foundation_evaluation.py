"""Real dataset/evaluator round trips with tiny frozen stand-in models."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import yaml
import test as entry
from datasets import make_dataset
from evaluation.metrics import METRIC_NAMES
from evaluation.protocol import eval_pair, prepare_gt


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.ones(()))
        self.inference_metadata={'test_only':True}

    def forward(self,x):
        h,w=x.shape[-2:]
        depth=torch.linspace(1200.,6000.,h*w,device=x.device).reshape(1,1,h,w)*self.scale
        q=depth.reciprocal()
        return {'raw_relative':q,'metric_depth':depth,
                'relative_depth':q.clamp_min(.001).reciprocal()}


def fixture(root,dataset,model='unidepth_v2_vitl14'):
    data=dict(name=dataset,root=str(root/'data'),input_height=4,input_width=6,gt_space='raw')
    if dataset=='robotcar':
        data.update(day_package='day',night_package='night',crop_width=12,crop_height=8,gt_suffix='_depth')
        for split,names in [('night',['b','a']),('day',['dayonly'])]:
            package=root/'data'/split
            (package/'rgb').mkdir(parents=True);(package/'depth').mkdir()
            (package/'test_split.txt').write_text('\n'.join(names))
            for name in names:
                cv2.imwrite(str(package/'rgb'/(name+'.png')),np.full((12,16,3),70,np.uint8))
                gt=np.linspace(.2,45.,12*16).reshape(12,16).astype(np.float32)
                if name=='a':gt[::2]=0  # different valid-pixel counts, still image-balanced
                np.save(package/'depth'/(name+'_depth.npy'),gt)
        bounds=(.1,40.)
    else:
        base=root/'data';(base/'test/color').mkdir(parents=True);(base/'test/gt').mkdir()
        (base/'splits').mkdir()
        (base/'splits/night_test_split.txt').write_text('b\na\n')
        (base/'splits/day_test_split.txt').write_text('dayonly\n')
        for name in ['a','b','dayonly']:
            cv2.imwrite(str(base/'test/color'/(name+'.png')),np.full((12,16,3),70,np.uint8))
            gt=np.linspace(.2,65.,12*16).reshape(12,16).astype(np.float32)
            if name=='a':gt[::2]=0
            np.save(base/'test/gt'/(name+'.npy'),gt)
        bounds=(1e-5,60.)
    weights=root/'tiny_weights.pth';torch.save(TinyModel().state_dict(),weights)
    cfg=dict(data=data,model=dict(name=model,weights=str(weights),inverse_epsilon=.001),
             eval=dict(split='day',align='median',min_depth=bounds[0],max_depth=bounds[1],output=str(root/'unused')))
    path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    return path,cfg


class FoundationEvaluationTests(unittest.TestCase):
    def test_unidepth_native_depth_and_existing_evaluator_for_both_night_splits(self):
        for dataset in ['robotcar','nuscenes']:
            with self.subTest(dataset=dataset),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);config,cfg=fixture(root,dataset);out=root/'uni_night'
                ds=make_dataset(cfg,'night',False)
                gt_before={str(r['gt']):Path(r['gt']).read_bytes() for r in ds.records}
                with patch.object(entry,'build_inference_model',return_value=TinyModel()),contextlib.redirect_stdout(io.StringIO()):
                    result=entry.run(config,out,split_override='night',evaluate=True,device='cpu')
                depth=np.load(out/'predictions.npy')
                expected=torch.linspace(1200.,6000.,24).reshape(4,6).numpy()
                np.testing.assert_array_equal(depth,np.stack([expected,expected]))
                inverse=np.load(out/'raw_inverse_depth.npy')
                np.testing.assert_allclose(inverse*depth,1.,rtol=1e-7)
                index=json.loads((out/'index.json').read_text())
                self.assertEqual([r['id'] for r in index],['a','b'])
                expected_metrics=[]
                for i,rec in enumerate(index):
                    self.assertEqual(rec['rgb_sha256'],hashlib.sha256(Path(rec['rgb_path']).read_bytes()).hexdigest())
                    self.assertEqual(Path(rec['gt_path']).read_bytes(),gt_before[rec['gt_path']])
                    gt=prepare_gt(np.load(rec['gt_path']),rec['geometry'])
                    expected_metrics.append(eval_pair(depth[i],gt,cfg['eval']['min_depth'],cfg['eval']['max_depth']))
                for key in METRIC_NAMES:
                    self.assertAlmostEqual(result[key],np.mean([r[key] for r in expected_metrics]),places=12)
                saved=json.loads((out/'metrics.json').read_text())
                self.assertEqual(saved['summary'],result)
                self.assertEqual(len(saved['per_image']),2)
                contract=json.loads((out/'prediction_contract.json').read_text())
                self.assertEqual(contract['evaluated_split'],'night')
                self.assertTrue(contract['full_split'])
                self.assertIsNone(contract['epsilon'])
                self.assertIn('native UniDepth',contract['predictions.npy'])
                self.assertEqual(contract['evaluation_protocol']['max_depth'],cfg['eval']['max_depth'])

    def test_dav2_preserves_historical_depth_and_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,cfg=fixture(root,'robotcar','dav2_vitl');out=root/'dav_night'
            net=TinyModel();checkpoint=root/'finetuned.pth'
            torch.save({'model':{'scale':torch.tensor(2.)}},checkpoint)
            with patch('models.depth_net.DepthNet',return_value=net),contextlib.redirect_stdout(io.StringIO()):
                entry.run(config,out,checkpoint=checkpoint,split_override='night',evaluate=True,device='cpu')
            expected=net(torch.ones(1,3,4,6))['relative_depth'][0,0].detach().numpy()
            np.testing.assert_array_equal(np.load(out/'predictions.npy')[0],expected)
            self.assertFalse(net.training)
            self.assertFalse(net.scale.requires_grad)
            contract=json.loads((out/'prediction_contract.json').read_text())
            self.assertEqual(contract['checkpoint_sha256'],entry.file_sha256(checkpoint))

    def test_uni_rejects_dav_checkpoint_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,cfg=fixture(root,'nuscenes');out=root/'existing';out.mkdir()
            marker=out/'metrics.json';marker.write_text('existing baseline')
            with self.assertRaisesRegex(ValueError,'not a DAV2'):
                entry.run(config,root/'other',checkpoint='wrong.pth',split_override='night')
            with self.assertRaisesRegex(ValueError,'not empty'):
                entry.run(config,out,split_override='night')
            self.assertEqual(marker.read_text(),'existing baseline')

    def test_prediction_only_uses_no_gt_values_and_marks_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,cfg=fixture(root,'robotcar');out=root/'pred'
            # Corrupt GT bytes are never opened by the prediction-only path.
            for path in (root/'data/night/depth').glob('*.npy'):path.write_bytes(b'not read during inference')
            with patch.object(entry,'build_inference_model',return_value=TinyModel()),contextlib.redirect_stdout(io.StringIO()):
                entry.run(config,out,split_override='night',limit=1,device='cpu')
            self.assertFalse((out/'metrics.json').exists())
            contract=json.loads((out/'prediction_contract.json').read_text())
            self.assertFalse(contract['full_split'])
            self.assertEqual(contract['num_images'],1)


if __name__=='__main__':
    unittest.main()
