"""Single source of test-time DAV2 predictions for both datasets.

Writes STEPS predictions.npy *as predicted depth*, not DAV2's raw inverse depth.
Writes raw_inverse_depth.npy for independent baseline/representation diagnosis.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils.config import load_config,project_path
from datasets import make_dataset
from models.depth_net import DepthNet
from visualization.depth import save_relative_disp

def run(config,output,checkpoint=None,vis=False,limit=0,split_override=None):
    cfg=load_config(config);m=cfg['model'];ev=cfg['eval']
    ds=make_dataset(cfg,split=split_override or ev['split'],training=False)
    out=project_path(output or ev['output']);out.mkdir(parents=True,exist_ok=True)
    model=DepthNet(weights=str(project_path(m['weights'])),
                   input_size=m.get('input_size',518),
                   freeze_encoder=m.get('encoder_frozen',True),
                   train_dpt=m.get('train_dpt',True),
                   inverse_epsilon=m.get('inverse_epsilon',1e-3))
    if checkpoint:
        sd=torch.load(str(project_path(checkpoint)),map_location='cpu')
        sd=sd.get('model',sd)
        model.load_state_dict(sd, strict=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    depth=[];raw=[];index=[]
    count=min(len(ds),limit) if limit else len(ds)
    with torch.no_grad():
        for i in tqdm(range(count),desc='predict'):
            rec=ds[i]
            x=rec['image'][None].to(device)
            res=model(x)
            d=res['relative_depth'][0,0].cpu().numpy().astype(np.float32)
            q=res['raw_relative'][0,0].cpu().numpy().astype(np.float32)
            depth.append(d);raw.append(q)
            index.append({'id':rec['id'],'gt_path':rec['gt_path'],
                          'geometry':rec['geometry'],'dataset':rec['dataset']})
            if vis:
                save_relative_disp(q,out/'vis'/(str(rec['id'])+'.png'),95)
    if not depth:raise ValueError('Empty test dataset')
    if len(set(x.shape for x in depth))!=1:
        raise ValueError('STEPS predictions.npy requires same HxW for all samples')
    np.save(str(out/'predictions.npy'),np.stack(depth),allow_pickle=False)
    np.save(str(out/'raw_inverse_depth.npy'),np.stack(raw),allow_pickle=False)
    (out/'index.json').write_text(json.dumps(index,indent=2))
    (out/'prediction_contract.json').write_text(json.dumps({
        'predictions.npy':'relative_depth = 1/max(DAV2_raw_inverse_depth, epsilon)',
        'raw_inverse_depth.npy':'unmodified DAV2 DPT output',
        'epsilon':m.get('inverse_epsilon',1e-3),
        'alignment':'none; median scaling done only by evaluator',
        'ordering':'sorted by STEPS test split, stored in index.json',
        'evaluated_split':split_override or ev['split']},indent=2))
    print('Saved',len(depth),'predictions into',out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--out')
    p.add_argument('--vis',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();run(a.config,a.out,a.checkpoint,a.vis,a.limit)
if __name__=='__main__':main()
