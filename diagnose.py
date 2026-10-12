"""Frozen DAV2 diagnostic: condition x layer relations and DPT oracle swap.

NO depth GT or pose used. Reports synthetic consistency, NOT LiDAR accuracy.
"""
import argparse,csv
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils.config import load_config,project_path
from datasets.common import read_tokens,image_file,rgb
from models.depth_model import DepthModel
from analysis.degradation_v2.pipeline import nighttime_degradation
from analysis.relations import relation_vector,spearman,affine_error
from analysis.rgb_statistics import (
    channel_saturation_ratio,
    channel_diversity
)
from visualization.qualitative import save_panel


CONDITIONS=(
    'dark_mild',
    'dark_medium',
    'dark_extreme',

    'dark_noise_mild',
    'dark_noise_medium',
    'dark_noise_extreme',

    'dark_white_flare_near',
    'dark_white_flare_medium',
    'dark_white_flare_far',

    'dark_red_flare_near',
    'dark_red_flare_far'
)

def generate_night(img, condition):

    mask=torch.zeros_like(img[:,0:1])

    if condition=='dark_mild':
        out,mask=nighttime_degradation(
            img,
            illumination='mild',
            noise=None
        )

    elif condition=='dark_medium':
        out,mask=nighttime_degradation(
            img,
            illumination='medium',
            noise=None
        )

    elif condition=='dark_extreme':
        out,mask=nighttime_degradation(
            img,
            illumination='extreme',
            noise=None
        )


    elif condition=='dark_noise_mild':
        out,mask=nighttime_degradation(
            img,
            illumination='medium',
            noise='mild'
        )


    elif condition=='dark_noise_medium':
        out,mask=nighttime_degradation(
            img,
            illumination='medium',
            noise='medium'
        )


    elif condition=='dark_noise_extreme':
        out,mask=nighttime_degradation(
            img,
            illumination='extreme',
            noise='extreme'
        )


    elif condition.startswith('dark_white_flare'):

        distance=condition.split('_')[-1]

        out,mask=nighttime_degradation(
            img,
            illumination='medium',
            noise='medium',
            flare_distance=distance,
            flare_color='white'
        )


    elif condition.startswith('dark_red_flare'):

        distance=condition.split('_')[-1]

        out,mask=nighttime_degradation(
            img,
            illumination='medium',
            noise='medium',
            flare_distance=distance,
            flare_color='red'
        )


    else:
        raise ValueError(condition)


    return out,mask

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    p.add_argument('--limit',type=int,default=128)
    p.add_argument('--out',default='runs/diagnose')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save_vis', action='store_true')
    p.add_argument('--num_vis_per_condition', type=int, default=3)
    a=p.parse_args()
    torch.manual_seed(a.seed)
    cfg=load_config(a.config);d=cfg['data'];m=cfg['model']
    if d['name']=='robotcar':
        package=project_path(d['root'])/d['day_package']
        paths=[image_file(package/'rgb',Path(s).name) for s in read_tokens(package/'train_split.txt')]
    else:
        from datasets import make_dataset
        ds=make_dataset(cfg,'day',training=True)
        paths=[ds.records[i]['target'] for i in range(len(ds))]
    if not paths:raise ValueError('No diagnosis images')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=DepthModel(str(project_path(m['weights'])),input_size=m.get('input_size',518),freeze_encoder=True,train_dpt=False,inverse_epsilon=m.get('inverse_epsilon',1e-3)).to(device).eval()
    out=project_path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    vis_count = {c: 0 for c in a.conditions}
    with torch.no_grad():
        for path in tqdm(paths[:a.limit],desc='Diagnosis'):
            clean,_=rgb(path,(d['input_height'],d['input_width']))
            clean=clean.unsqueeze(0).to(device)
            xc,fc=net.extract(clean)
            ph,pw=xc.shape[-2]//14,xc.shape[-1]//14
            dc=net.decode(xc,fc,clean.shape[-2:]);rc=[relation_vector(f,(ph,pw))[0].cpu().numpy() for f in fc]
            for cond in a.conditions:
                dark,mask = generate_night(clean,cond)
                sat = channel_saturation_ratio(dark)
                cdi = channel_diversity(dark)
                xd,fd=net.extract(dark)
                dr=net.decode(xd,fd,clean.shape[-2:])
                e_dark=affine_error(dr,dc)
                lrr=[spearman(rc[i],relation_vector(f,(ph,pw))[0].cpu().numpy()) for i,f in enumerate(fd)]
                # Cumulative DPT feature swaps; patch + class tokens are kept consistent.
                swap4 = net.decode(xd, list(fd[:3]) + [fc[3]], clean.shape[-2:])
                swap34 = net.decode(xd, list(fd[:2]) + list(fc[2:]), clean.shape[-2:])

                e_swap4 = affine_error(swap4, dc)
                e_swap34 = affine_error(swap34, dc)

                rows.append(dict(id=Path(path).stem,condition=cond,
                  R_sat=sat["R_sat"],G_sat=sat["G_sat"],B_sat=sat["B_sat"],
                  RGB_sat=sat["RGB_sat"],CDI=cdi,LRR_F1=lrr[0],LRR_F2=lrr[1],
                  LRR_F3=lrr[2],LRR_F4=lrr[3],depth_drift=e_dark,
                  recovery_F4=(e_dark-e_swap4)/(e_dark+1e-8),
                  recovery_F34=(e_dark-e_swap34)/(e_dark+1e-8),
                  glare_area=float((mask > 0.25).float().mean()) if mask is not None else 0.0))
                
                if a.save_vis and vis_count[cond] < a.num_vis_per_condition:
                    vis_dir = out / 'qualitative' / cond / Path(path).stem

                    degraded_err = (dr - dc).abs()
                    swap34_err = (swap34 - dc).abs()

                    save_panel(
                        save_dir=vis_dir,
                        clean_rgb=clean,
                        degraded_rgb=dark,
                        clean_depth=dc,
                        degraded_depth=dr,
                        swap4_depth=swap4,
                        swap34_depth=swap34,
                        degraded_err=degraded_err,
                        swap34_err=swap34_err,
                        title=f'{cond} | {Path(path).stem}'
                    )

                    vis_count[cond] += 1
                
    
    with (out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'summary.csv').open('w',newline='') as f:
        keys=list(rows[0])[2:]
        writer=csv.writer(f);writer.writerow(['condition','n']+keys)
        for cond in a.conditions:
            selection=[r for r in rows if r['condition']==cond]
            if selection:writer.writerow([cond,len(selection)]+[np.nanmedian([r[key] for r in selection]) for key in keys])
    print('Saved diagnosis:',out,'images:',min(len(paths),a.limit),'conditions:',len(a.conditions))

if __name__=='__main__':main()
