"""Compact G-only input probe, separate from approximate night distribution."""
import argparse
import torch
import yaml
from tqdm import tqdm
from utils.config import load_config,project_path
from datasets.common import rgb
from models.depth_model import DepthModel
from analysis.degradation_v2.pipeline import sample_parameters,degrade
from analysis.sampling import select_samples,save_selection,stable_seed
from analysis.spread import spatial_drift
from analysis.statistics import write_csv,group_estimate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--limit',type=int,default=512)
    p.add_argument('--out',default='runs/spread');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sample-list');p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    if a.repeats<1:raise ValueError('repeats must be positive')
    cfg=load_config(a.config);d=cfg['data'];m=cfg['model']
    samples,audit=select_samples(cfg,a.limit,a.seed,a.sample_list)
    out=project_path(a.out);out.mkdir(parents=True,exist_ok=True);save_selection(out,samples,audit)
    (out/'run.yaml').write_text(yaml.safe_dump(dict(config=cfg,arguments=vars(a),probe='compact G only')))
    model=DepthModel(str(project_path(m['weights'])),input_size=m.get('input_size',518),freeze_encoder=True,train_dpt=False)
    dev='cuda' if torch.cuda.is_available() else 'cpu';model.to(dev).eval()
    crop=(d.get('crop_width',1152),d.get('crop_height',640)) if d['name']=='robotcar' else None
    rows=[];parameters=[]
    with torch.no_grad():
        for sample in tqdm(samples,desc='Local propagation probe'):
            x,_=rgb(sample['path'],(d['input_height'],d['input_width']),crop=crop);x=x[None].to(dev)
            inp,clean_feat=model.extract(x)
            # Patch projection before attention is the preprocessing/support negative control.
            patch_clean=model.net.pretrained.patch_embed(inp)
            for repeat in range(a.repeats):
                seed=stable_seed(a.seed,sample['id'],repeat)
                # Compact, in-frame localized probe: do not call this natural night synthesis.
                params=sample_parameters(seed,{'source_count':(1,1),'radius':(.02,.06),'u':(.1,.9),'v':(.1,.9)})
                dark,_=degrade(x,'G',params,compact=True)
                bad_inp,bad_feat=model.extract(dark)
                changed=(bad_inp-inp).abs().amax(1,keepdim=True)>0
                patch_bad=model.net.pretrained.patch_embed(bad_inp)
                result=spatial_drift([patch_clean]+list(clean_feat),[patch_bad]+list(bad_feat),
                                    (inp.shape[-2]//14,inp.shape[-1]//14),changed,
                                    layer_names=['patch_embed','F1','F2','F3','F4'])
                for row in result:row.update(id=sample['id'],group=sample['group'],repeat=repeat,parameter_seed=seed)
                rows.extend(result)
                parameters.append(dict(id=sample['id'],repeat=repeat,parameter_seed=seed,**params['sources'][0]))
            write_csv(out/'per_image.csv',rows)
    write_csv(out/'parameters.csv',parameters)
    summary=[]
    for layer,band in sorted({(r['layer'],r['band']) for r in rows}):
        group=[r for r in rows if r['layer']==layer and r['band']==band]
        summary.append(dict(layer=layer,band=band,**group_estimate(group,'drift',a.seed)))
    write_csv(out/'summary.csv',summary)
    print('Saved propagation probe into',out)

if __name__=='__main__':main()
