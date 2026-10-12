"""Fail-fast audit for STEPS sample ordering / matched LiDAR GT.

This intentionally never opens GT during training; only check/test stage.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse, pathlib
import numpy as np
from datasets import make_dataset
from utils.config import load_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    cfg=load_config(a.config)
    d=make_dataset(cfg,split=cfg['eval']['split'],training=False)
    if not len(d):raise ValueError('Empty evaluation set')
    n=min(len(d),a.limit) if a.limit else len(d)
    seen=set()
    for i in range(n):
        x=d[i]
        if x['id'] in seen:raise RuntimeError('Duplicate ID: '+x['id'])
        seen.add(x['id'])
        path=pathlib.Path(x['gt_path'])
        if not path.is_file():raise FileNotFoundError('Missing GT: '+str(path))
        gt=np.squeeze(np.load(str(path)))
        if gt.ndim!=2:raise ValueError('Bad GT shape: '+str(gt.shape))
        if i<3: print(i,x['id'],'RGB',tuple(x['image'].shape),'GT',gt.shape)
    print('PASS: %d/%d ordered test samples have unique IDs and matched GT'% (n,len(d)))
    print('NOTE: final benchmark parity still requires checking the official STEPS RobotCar GT crop/mask')

if __name__=='__main__':main()
