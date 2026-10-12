"""Additional GT/FOV and Flare7K regression checks; no DAV2 weights required."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from evaluation.protocol import eval_pair,prepare_gt
from analysis.degradation_v2.pipeline import list_flare_templates,sample_parameters,degrade


class ContractTests(unittest.TestCase):
    def test_invalid_predictions_are_not_excluded_from_metric(self):
        gt=np.array([[1.,2.],[3.,4.]])
        with self.assertRaisesRegex(ValueError,'nonfinite/nonpositive'):
            eval_pair(np.array([[0.,2.],[3.,4.]]),gt,0.1,40)
        with self.assertRaisesRegex(ValueError,'nonfinite/nonpositive'):
            eval_pair(np.array([[np.nan,2.],[3.,4.]]),gt,0.1,40)
        with self.assertRaisesRegex(ValueError,'valid_mask'):
            eval_pair(gt,gt,0.1,40,valid_mask=np.ones((1,2)))
        # A broken prediction at an invalid GT pixel is not assessed.
        result=eval_pair(np.array([[0.,2.],[3.,4.]]),
                         np.array([[0.,2.],[3.,4.]]),0.1,40)
        self.assertAlmostEqual(result['abs_rel'],0.)

    def test_gt_strict_raw_cropped_and_overlay(self):
        from tools.check_eval_data import overlay_pair
        with tempfile.TemporaryDirectory() as tmp:
            raw=np.zeros((8,12,3),dtype=np.uint8)
            raw[:]=[50,100,150]
            path=Path(tmp)/'rgb.png';cv2.imwrite(str(path),raw)
            geom=dict(raw_hw=[8,12],crop_xywh=[2,2,8,4],input_hw=[4,8],gt_space='raw')
            gt=np.arange(96,dtype=float).reshape(8,12)+1
            aligned=prepare_gt(gt,geom)
            np.testing.assert_array_equal(aligned,gt[2:6,2:10])
            marked,roi=overlay_pair(path,aligned,geom,0.1,120)
            self.assertEqual(marked.shape,(8,12,3))
            self.assertEqual(roi.shape,(4,8,3))
            self.assertTrue(np.any(roi!=raw[2:6,2:10]))
            with self.assertRaises(ValueError):prepare_gt(aligned,geom)
            cropped=dict(geom,gt_space='cropped')
            np.testing.assert_array_equal(prepare_gt(aligned,cropped),aligned)

    def test_flare_bank_shape_seed_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'shape.png'
            flare=np.zeros((32,32,3),dtype=np.uint8)
            cv2.line(flare,(0,15),(31,15),(255,255,255),2)
            cv2.imwrite(str(path),flare)
            bank=list_flare_templates(tmp)
            self.assertEqual(len(bank),1)
            params=sample_parameters(19,{'source_count':(1,1),'u':(.5,.5),
                'v':(.5,.5),'radius':(.2,.2),'amplitude':(.7,.7)},bank)
            self.assertEqual(params,sample_parameters(19,{'source_count':(1,1),
                'u':(.5,.5),'v':(.5,.5),'radius':(.2,.2),'amplitude':(.7,.7)},bank))
            self.assertEqual(params['sources'][0]['template'],str(path.resolve()))
            x=torch.full((1,3,56,84),.25)
            images={c:degrade(x,c,params)[0] for c in ('G','LG','NG','LNG')}
            for image in images.values():
                self.assertEqual(image.shape,x.shape)
                self.assertTrue(torch.isfinite(image).all())
            self.assertTrue(torch.equal(images['G'],degrade(x,'G',params)[0]))
            fallback=sample_parameters(19)
            self.assertNotIn('template',fallback['sources'][0])
            self.assertFalse(torch.equal(images['G'],degrade(x,'G',fallback)[0]))
            with self.assertRaises(FileNotFoundError):
                list_flare_templates(str(Path(tmp)/'absent'))


if __name__=='__main__':unittest.main()
