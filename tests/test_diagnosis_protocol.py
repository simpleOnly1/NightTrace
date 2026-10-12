"""Regression checks for scientific failure modes; no DAV2 weights required."""
import csv
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
from torch import nn
from torch.nn import functional as F
from analysis.sampling import select_samples,save_selection,stable_seed
from analysis.degradation_v2.pipeline import CONDITIONS,degrade,sample_parameters,to_linear,to_srgb
from analysis.relations import affine_error,ordinal_error
from analysis.spread import spatial_drift
from analysis.statistics import factorial_effects,group_estimate
from datasets.common import FrameDataset,read_K
from evaluation.protocol import prepare_gt,evaluate_records


class ProtocolTests(unittest.TestCase):
    def test_crop_is_coordinate_transform_not_resize(self):
        gt=np.arange(8*10,dtype=float).reshape(8,10)+1
        geometry=dict(raw_hw=[8,10],crop_xywh=[2,2,6,4],input_hw=[4,6],gt_space='raw')
        expected=gt[2:6,2:8]
        np.testing.assert_array_equal(prepare_gt(gt,geometry),expected)
        with self.assertRaises(ValueError):prepare_gt(expected,geometry)
        cropped=dict(geometry,gt_space='cropped')
        np.testing.assert_array_equal(prepare_gt(expected,cropped),expected)
        with self.assertRaises(ValueError):prepare_gt(gt,cropped)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);np.save(path/'gt.npy',gt)
            np.save(path/'predictions.npy',expected[None]*3)
            index=[dict(id='1',gt_path=str(path/'gt.npy'),geometry=geometry,dataset='robotcar')]
            (path/'index.json').write_text(json.dumps(index))
            result=evaluate_records(path,min_depth=.1,max_depth=100,dataset='robotcar')
            self.assertAlmostEqual(result['abs_rel'],0)
            del index[0]['geometry'];(path/'index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError,'geometry'):evaluate_records(path)
            np.save(path/'K.npy',[[10.,0.,5.],[0.,10.,4.],[0.,0.,1.]])
            K=read_K(path/'K.npy',(8,10),(2,3),crop=(6,4))
            np.testing.assert_allclose(K.numpy(),[[5.,0.,1.5],[0.,5.,1.],[0.,0.,1.]])

    def test_reverse_order_cannot_fit_as_perfect(self):
        t=torch.tensor([1.,2.,3.]);p=t.flip(0)
        self.assertGreater(affine_error(p,t),.5)
        self.assertEqual(ordinal_error(p,t),1)
        self.assertLess(affine_error(2*t+3,t),1e-6)

    def test_paired_parameters_condition_order_and_noise_sequence(self):
        x=torch.linspace(.01,.9,3*28*42).reshape(1,3,28,42)
        params=sample_parameters(123)
        before=torch.random.get_rng_state().clone()
        a={c:degrade(x,c,params)[0] for c in CONDITIONS}
        b={c:degrade(x,c,params)[0] for c in reversed(CONDITIONS)}
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        for c in CONDITIONS:
            self.assertTrue(torch.equal(a[c],b[c]));self.assertTrue(torch.isfinite(a[c]).all())
        self.assertTrue(torch.equal(a['clean'],x))
        self.assertFalse(torch.equal(a['LNG'],a['LG']))
        expected=to_srgb(to_linear(x)*params['exposure'])
        torch.testing.assert_close(a['L'],expected)
        self.assertNotEqual(sample_parameters(124)['sources'],params['sources'])
        # Noise-free G is fed into Poisson before clipping, not added afterwards.
        from analysis.degradation_v2.pipeline import flare_field
        signal=to_linear(x)*params['exposure']+flare_field(x,params['sources'])
        gen=torch.Generator().manual_seed(params['seed'])
        noisy=torch.poisson(signal*params['shot_peak'],generator=gen)/params['shot_peak']
        read_gen=torch.Generator().manual_seed(params['seed'] ^ 0x5DEECE66D)
        noisy+=torch.randn(signal.shape,generator=read_gen)*params['read_sigma']
        torch.testing.assert_close(a['LNG'],to_srgb(noisy.clamp(0,1)).clamp(0,1))
        with self.assertRaises(ValueError):sample_parameters(0,{'radius':(-1,0)})

    def test_patch_support_preserves_one_changed_corner_pixel(self):
        c=torch.randn(1,4,5);d=c.clone();d[:,0,:]*=-1
        mask=torch.zeros(1,1,28,28);mask[:,:,13,13]=1
        result=spatial_drift([c],[d],(2,2),mask)
        affected=next(r for r in result if r['band']=='affected')
        self.assertEqual(affected['n'],1);self.assertAlmostEqual(affected['drift'],2,places=5)
        with self.assertRaises(ValueError):spatial_drift([c],[d],(2,2),torch.ones(1,1,56,56))

    def test_preprocessed_compact_support_and_patch_negative_control(self):
        from models.depth_net import preprocess
        x=torch.rand(1,3,28,56)
        params=sample_parameters(4,{'source_count':(1,1),'u':(.4,.4),'v':(.5,.5),'radius':(.02,.02)})
        y,field=degrade(x,'G',params,compact=True)
        self.assertTrue(torch.equal(y.expand_as(x)[field.expand_as(x)==0],x[field.expand_as(x)==0]))
        a=preprocess(x,56);b=preprocess(y,56);changed=(a-b).abs().amax(1,keepdim=True)>0
        proj=nn.Conv2d(3,8,14,stride=14)
        ca=proj(a).flatten(2).transpose(1,2);cb=proj(b).flatten(2).transpose(1,2)
        rows=spatial_drift([ca],[cb],(4,8),changed)
        for r in rows:
            if r['band']!='affected':self.assertLess(abs(r['drift']),1e-6)

    def make_data(self,tmp):
        root=Path(tmp);pkg=root/'day';(pkg/'rgb').mkdir(parents=True)
        rng=np.random.default_rng(8);tokens=[]
        for i in range(30):
            token=str(1418132400000000+i*2_000_000);tokens.append(token)
            image=rng.integers(0,255,size=(28,42,3),dtype=np.uint8)
            if i==29:image=cv2.imread(str(pkg/'rgb'/(tokens[0]+'.png')))
            cv2.imwrite(str(pkg/'rgb'/(token+'.png')),image)
        (pkg/'train_split.txt').write_text('\n'.join(tokens))
        return dict(data=dict(name='robotcar',root=str(root),day_package='day',night_package='day',
                             crop_width=42,crop_height=28,input_height=28,input_width=42),
                    diagnosis=dict(block_seconds=10,min_gap_seconds=1,duplicate_mae=.001),
                    model=dict(weights='unused',input_size=28))

    def test_group_sampling_repeatable_nested_and_no_duplicate_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.make_data(tmp)
            a,audit=select_samples(cfg,6,3);b,_=select_samples(cfg,12,3)
            self.assertEqual([r['id'] for r in a],[r['id'] for r in b[:6]])
            self.assertEqual(len({r['group'] for r in a}),6)
            full,report=select_samples(cfg,0,3)
            self.assertEqual(len(full),29)
            self.assertEqual(sum(r['duplicate_rejected'] for r in report),1)
            save_selection(Path(tmp)/'out',a,audit)
            again,_=select_samples(cfg,1,99,Path(tmp)/'out/samples.txt')
            self.assertEqual([r['id'] for r in a],[r['id'] for r in again])
            self.assertEqual(stable_seed(2,a[0]['id']),stable_seed(2,a[0]['id']))

    def test_nuscenes_single_frames_and_split_membership(self):
        from analysis.sampling import candidates
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'splits').mkdir()
            (root/'splits/day_train_split.txt').write_text('clip1\nclip2\n')
            for clip in ('clip1','clip2','excluded'):
                base=root/'sequences'/clip;base.mkdir(parents=True)
                (base/'file_list.txt').write_text('frame.png\n')
                cv2.imwrite(str(base/'frame.png'),np.ones((8,12,3),dtype=np.uint8))
            cfg=dict(data=dict(name='nuscenes',root=str(root),input_height=8,input_width=12))
            result=candidates(cfg)
            self.assertEqual(len(result),2)  # no PoseNet/triplet requirement
            self.assertEqual(len({r['id'] for r in result}),2)
            self.assertEqual(len({r['group'] for r in result}),2)
            self.assertFalse(any('excluded' in r['id'] for r in result))

    def test_interactions_and_cluster_unit(self):
        v={'clean':0,'L':1,'N':2,'G':3,'LN':7,'LG':4,'NG':5,'LNG':10}
        rows=[dict(id='x',group='g',repeat=0,condition=c,depth_drift=x) for c,x in v.items()]
        result=factorial_effects(rows)[0]
        self.assertEqual(result['LN'],4);self.assertEqual(result['LNG'],0)
        # More repeats of one image cannot change group weight or inflate N.
        data=[dict(id='x',group='a',v=1)]*20+[dict(id='y',group='b',v=3)]
        est=group_estimate(data,'v');self.assertEqual(est['mean'],2)
        self.assertEqual(est['images'],2);self.assertEqual(est['groups'],2)

    def test_entrypoints_with_small_fake_model(self):
        import diagnose,diagnose_spread,yaml
        class FakeModel(nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__();self.proj=nn.Conv2d(3,8,14,stride=14)
                from types import SimpleNamespace
                self.net=SimpleNamespace(pretrained=SimpleNamespace(patch_embed=self.patch_embed))
            def patch_embed(self,x):return self.proj(x).flatten(2).transpose(1,2)
            def extract(self,x):
                f=self.patch_embed(x)
                return x,[(f*(i+1),f.mean(1)) for i in range(4)]
            def decode(self,x,features,target_size):
                vals=sum(f[0].mean(-1) for f in features).reshape(1,1,2,3)
                return F.interpolate(vals,size=target_size,mode='bilinear',align_corners=False)
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.make_data(tmp);p=Path(tmp);config=p/'config.yaml';config.write_text(yaml.safe_dump(cfg))
            args=['cmd','--config',str(config),'--limit','2','--repeats','1','--out',str(p/'diag')]
            with patch.object(sys,'argv',args),patch.object(diagnose,'DepthModel',FakeModel):diagnose.main()
            with (p/'diag/per_image.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),16);self.assertEqual(len({r['parameter_seed'] for r in rows}),2)
            self.assertTrue((p/'diag/interactions_summary.csv').is_file())
            args[-1]=str(p/'spread')
            with patch.object(sys,'argv',args),patch.object(diagnose_spread,'DepthModel',FakeModel):diagnose_spread.main()
            self.assertTrue((p/'spread/per_image.csv').is_file())

if __name__=='__main__':unittest.main()
