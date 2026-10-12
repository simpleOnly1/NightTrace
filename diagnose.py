"""Frozen DAV2: paired approximate L/N/G, relation drift and DPT oracle swaps.

Outputs compare RAW RELATIVE INVERSE DEPTH to clean predictions, not GT accuracy.
Run test.py/evaluate.py separately for unfiltered real-night GT evaluation.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
from utils.config import load_config, project_path
from datasets.common import rgb
from models.depth_model import DepthModel
from analysis.sampling import select_samples, save_selection, stable_seed
from analysis.degradation_v2.pipeline import CONDITIONS, sample_parameters, degrade
from analysis.relations import relation_vector, spearman, affine_fit, affine_error, ordinal_error
from analysis.statistics import save_statistics, write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    p.add_argument('--limit',type=int,default=512,help='Mechanism subset budget; 0 = all eligible')
    p.add_argument('--out',default='runs/diagnose')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--sample-list',help='Reuse samples.txt exactly, within the allowed day split')
    p.add_argument('--prepare-only',action='store_true',help='Audit/select RGB without loading DAV2 weights')
    p.add_argument('--save_vis',action='store_true')
    p.add_argument('--num_vis_per_condition',type=int,default=3)
    a=p.parse_args()
    if a.repeats<1:raise ValueError('repeats must be positive')
    if len(set(a.conditions))!=len(a.conditions):raise ValueError('Duplicate conditions')
    cfg=load_config(a.config);d=cfg['data'];m=cfg['model']
    out=project_path(a.out);out.mkdir(parents=True,exist_ok=True)
    selected,audit=select_samples(cfg,a.limit,a.seed,a.sample_list)
    save_selection(out,selected,audit)
    (out/'run.yaml').write_text(yaml.safe_dump(dict(config=cfg,arguments=vars(a),
        estimand='group-balanced synthetic consistency; not real-world GT accuracy'),sort_keys=False))
    if a.prepare_only:
        print('Selected',len(selected),'images; inspect',out/'sampling_coverage.csv');return
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=DepthModel(str(project_path(m['weights'])),input_size=m.get('input_size',518),
        freeze_encoder=True,train_dpt=False,inverse_epsilon=m.get('inverse_epsilon',1e-3)).to(device).eval()
    crop=(d.get('crop_width',1152),d.get('crop_height',640)) if d['name']=='robotcar' else None
    rows=[];parameters=[];vis_count={c:0 for c in a.conditions};previous=None
    swaps={**{f'F{i+1}':(i,) for i in range(4)},'F34':(2,3)}
    with torch.no_grad(), (out/'per_image.csv').open('w',newline='') as stream:
        writer=None
        for sample in tqdm(selected,desc='Paired diagnosis'):
            clean,_=rgb(sample['path'],(d['input_height'],d['input_width']),crop=crop)
            clean=clean[None].to(device)
            xc,fc=net.extract(clean);hw=(xc.shape[-2]//14,xc.shape[-1]//14)
            dc=net.decode(xc,fc,clean.shape[-2:])
            rc=[relation_vector(f,hw)[0].cpu().numpy() for f in fc]
            clean_sham=affine_error(net.decode(xc,list(fc),clean.shape[-2:]),dc)
            for repeat in range(a.repeats):
                seed=stable_seed(a.seed,sample['id'],repeat)
                params=sample_parameters(seed,cfg.get('diagnosis',{}).get('degradation_ranges'))
                for i,source in enumerate(params['sources']):
                    parameters.append(dict(id=sample['id'],repeat=repeat,parameter_seed=seed,
                        exposure=params['exposure'],shot_peak=params['shot_peak'],read_sigma=params['read_sigma'],
                        source=i,**source))
                for cond in a.conditions:
                    dark,flare=degrade(clean,cond,params)
                    if cond=='clean':xd,fd,dr=xc,fc,dc
                    else:
                        xd,fd=net.extract(dark);dr=net.decode(xd,fd,clean.shape[-2:])
                    fit=affine_fit(dr,dc);e_dark=fit['error']
                    lrr=[spearman(rc[i],relation_vector(f,hw)[0].cpu().numpy()) for i,f in enumerate(fd)]
                    row=dict(id=sample['id'],group=sample['group'],repeat=repeat,condition=cond,parameter_seed=seed,
                        **{f'LRR_F{i+1}':v for i,v in enumerate(lrr)},depth_drift=e_dark,
                        ordinal_error=ordinal_error(dr,dc,seed=stable_seed(a.seed,sample['id'])),
                        affine_scale=fit['scale'],affine_shift=fit['shift'],
                        underexposed_area=float((dark.mean(1)<.05).float().mean()),
                        saturated_area=float((dark.amax(1)>=.99).float().mean()),
                        glare_area=float((flare>.01).float().mean()),clean_sham_error=clean_sham)
                    swapped={}
                    for name,indices in swaps.items():
                        mixed=[fc[i] if i in indices else fd[i] for i in range(4)]
                        q=net.decode(xd,mixed,clean.shape[-2:]);err=affine_error(q,dc)
                        row['delta_'+name]=e_dark-err
                        row['recovery_'+name]=(e_dark-err)/e_dark if e_dark>1e-6 else float('nan')
                        if name in ('F4','F34'):swapped[name]=q
                    # Sham and wrong-image decoder controls; neither localizes encoder damage.
                    row['degraded_sham_error']=affine_error(net.decode(xd,list(fd),clean.shape[-2:]),dr)
                    if previous is not None:
                        wrong=net.decode(xd,list(fd[:3])+[previous[3]],clean.shape[-2:])
                        row['delta_wrong_F4']=e_dark-affine_error(wrong,dc)
                    else:row['delta_wrong_F4']=float('nan')
                    if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);stream.flush();rows.append(row)
                    if a.save_vis and vis_count[cond]<a.num_vis_per_condition:
                        from visualization.qualitative import save_panel
                        save_panel(out/'qualitative'/cond/f'{seed}',clean,dark,dc,dr,swapped['F4'],swapped['F34'],
                                   (dr-dc).abs(),(swapped['F34']-dc).abs(),title=f'{cond} | {sample["id"]}')
                        vis_count[cond]+=1
            previous=fc
            write_csv(out/'parameters.csv',parameters)
    save_statistics(out,rows,a.seed)
    print('Saved',len(selected),'base images x',a.repeats,'repeats x',len(a.conditions),'conditions into',out)

if __name__=='__main__':main()
