import argparse,csv
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils.config import load_config,project_path
from datasets.common import read_tokens,image_file,rgb
from models.depth_model import DepthModel
from analysis.degradation import corrupt
from analysis.spread import spatial_drift


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--limit',type=int,default=128)
    ap.add_argument('--out',default='runs/spread_robotcar')
    ap.add_argument('--kind',choices=['colored_glare_only','white_glare_only'],default='colored_glare_only')
    a=ap.parse_args();c=load_config(a.config);d=c['data'];m=c['model']
    if d['name']!='robotcar':raise ValueError('Spread demo currently uses RobotCar daytime split')
    base=project_path(d['root'])/d['day_package']
    names=read_tokens(base/'train_split.txt')
    files=[image_file(base/'rgb',Path(x).name) for x in names[:a.limit]]
    model=DepthModel(str(project_path(m['weights'])),input_size=m.get('input_size',518),freeze_encoder=True,train_dpt=False,inverse_epsilon=m.get('inverse_epsilon',1e-3))
    dev='cuda' if torch.cuda.is_available() else 'cpu';model.to(dev).eval()
    rows=[]
    with torch.no_grad():
        for path in tqdm(files):
            x,_=rgb(path,(d['input_height'],d['input_width']))
            x=x[None].to(dev);dark,mask=corrupt(x,a.kind)
            inp,clean_feat=model.extract(x)
            _,bad_feat=model.extract(dark)
            # exactly unchanged in glare_only outside the luminous mask
            pixel_changed=(dark-x).abs().amax(dim=1,keepdim=True)>1e-6
            sample=spatial_drift(clean_feat,bad_feat,
                                 (inp.shape[-2]//14,inp.shape[-1]//14),pixel_changed)
            for r in sample:r.update(id=Path(path).stem,kind=a.kind)
            rows.extend(sample)
    out=project_path(a.out);out.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('No changed pixels or spatial bands')
    with (out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['kind','layer','band','median_drift','images'])
        for layer in ['F1','F2','F3','F4']:
            for band in ['affected','1-2tokens','3-5tokens','6+tokens']:
                group=[x['drift'] for x in rows if x['layer']==layer and x['band']==band]
                if group:writer.writerow([a.kind,layer,band,np.median(group),len(group)])
    print('Spatial-drift summary:',out/'summary.csv')

if __name__=='__main__':main()
