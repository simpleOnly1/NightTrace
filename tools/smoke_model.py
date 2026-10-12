"""One-image *real* ViT-L checkpoint test; unlike fake adapter unit test.

No ground truth required. Reports raw q distribution before any training/eval.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from datasets import make_dataset
from models.depth_net import DepthNet
from utils.config import load_config,project_path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    a=p.parse_args()
    c=load_config(a.config);m=c['model'];e=c['eval']
    frame=make_dataset(c,split=e['split'],training=False)[0]
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=DepthNet(str(project_path(m['weights'])),m.get('input_size',518),
                 True,False,m.get('inverse_epsilon',1e-3)).to(dev).eval()
    with torch.no_grad():out=net(frame['image'][None].to(dev),return_features=True)
    q=out['raw_relative'].detach().float().cpu().numpy()
    d=out['relative_depth'].detach().float().cpu().numpy()
    print('sample',frame['id'],'rgb',tuple(frame['image'].shape))
    for name,x in [('DAV2 raw relative inverse depth q',q),('1/max(q,eps) relative depth',d)]:
        print(name, 'shape',x.shape,'min',float(x.min()),'median',float(np.median(x)),
              'p99',float(np.percentile(x,99)),'max',float(x.max()),
              'zeros',float(np.mean(x<=0)))
    print('DPT taps',len(out['features']),[tuple(x[0].shape) for x in out['features']])
    assert np.isfinite(d).all() and np.isfinite(q).all()
    print('PASS: real model forward is finite; manually inspect near/far ordering')
if __name__=='__main__':main()
