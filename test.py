"""Frozen DAV2/UniDepth predictions and optional shared seven-metric evaluation.

predictions.npy always stores depth: reciprocal-clamped DAV2 q or native UniDepth
metric depth. Median alignment happens only in the existing evaluator.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils.config import load_config,project_path
from datasets import make_dataset
from models.inference import build_inference_model, model_name, prediction_depths, DAV2
from visualization.depth import save_relative_disp

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def run(config,output,checkpoint=None,vis=False,limit=0,split_override=None,
        evaluate=False,device='auto'):
    cfg=load_config(config);m=cfg['model'];ev=cfg['eval']
    name=model_name(m)
    split=split_override or ev['split']
    if limit<0 or split not in ('day','night'):
        raise ValueError('Use a nonnegative limit and day/night test split')
    if evaluate and ev.get('align','median')!='median':
        raise ValueError('Baseline --evaluate requires median alignment; use evaluate.py for separate diagnostic protocols')
    if checkpoint and name!=DAV2:
        raise ValueError('UniDepth uses its local checkpoint directory from model.weights, not a DAV2 --checkpoint')
    ds=make_dataset(cfg,split=split,training=False)
    count=min(len(ds),limit) if limit else len(ds)
    if not count:
        raise ValueError('Empty test dataset')
    if evaluate:
        for rec in ds.records[:count]:
            if not Path(rec['gt']).is_file():
                raise FileNotFoundError(rec['gt'])
    out=project_path(output or ev['output'])
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a new baseline directory: '+str(out))
    if name==DAV2:
        # Keep the historical DAV2/checkpoint construction contract.
        from models.depth_net import DepthNet
        model=DepthNet(weights=str(project_path(m['weights'])),
                       input_size=m.get('input_size',518),
                       freeze_encoder=m.get('encoder_frozen',True),
                       train_dpt=m.get('train_dpt',True),
                       inverse_epsilon=m.get('inverse_epsilon',1e-3))
    else:
        model=build_inference_model(m)
    if checkpoint:
        sd=torch.load(str(project_path(checkpoint)),map_location='cpu')
        sd=sd.get('model',sd)
        model.load_state_dict(sd, strict=True)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    model.to(device).eval().requires_grad_(False)
    out.mkdir(parents=True,exist_ok=True)
    depth=[];raw=[];index=[]
    with torch.no_grad():
        for i in tqdm(range(count),desc='predict'):
            rec=ds[i]
            x=rec['image'][None].to(device)
            res=model(x)
            inverse,predicted_depth=prediction_depths(res,m)
            d=predicted_depth[0,0].cpu().numpy().astype(np.float32)
            q=inverse[0,0].cpu().numpy().astype(np.float32)
            depth.append(d);raw.append(q)
            rgb_path=Path(ds.records[i]['target'])
            index.append({'id':rec['id'],'gt_path':rec['gt_path'],
                          'rgb_path':str(rgb_path),'rgb_sha256':file_sha256(rgb_path),
                          'geometry':rec['geometry'],'dataset':rec['dataset']})
            if vis:
                save_relative_disp(q,out/'vis'/(str(rec['id'])+'.png'),95)
    if len({r['id'] for r in index})!=len(index):
        raise ValueError('Test split contains duplicate sample IDs')
    if len(set(x.shape for x in depth))!=1:
        raise ValueError('STEPS predictions.npy requires same HxW for all samples')
    np.save(str(out/'predictions.npy'),np.stack(depth),allow_pickle=False)
    np.save(str(out/'raw_inverse_depth.npy'),np.stack(raw),allow_pickle=False)
    (out/'index.json').write_text(json.dumps(index,indent=2))
    data=cfg['data'];root=project_path(data['root'])
    split_file=(root/data.get('split_dir','splits')/(split+'_test_split.txt')
                if data['name']=='nuscenes' else
                root/data[split+'_package']/'test_split.txt')
    source_files=('test.py','models/inference.py','models/depth_net.py',
                  'models/unidepth_adapter.py','models/unidepth_env.py',
                  'datasets/common.py','datasets/robotcar.py','datasets/nuscenes.py',
                  'evaluation/protocol.py','evaluation/metrics.py')
    code_root=Path(__file__).resolve().parent
    protocol=dict(alignment=ev.get('align','median'),min_depth=ev['min_depth'],max_depth=ev['max_depth'],
                  gt_mask='finite GT strictly greater than min_depth and less than max_depth',
                  geometry='explicit RGB crop and raw/cropped GT provenance; never resize GT',
                  prediction_resize='nearest to aligned GT before median scaling',
                  clipping='prediction after median scaling',aggregation='equal mean of per-image metrics',
                  metrics=['abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'])
    contract={
        'model':name,'config':cfg,'device':str(device),'num_images':len(index),
        'full_split':count==len(ds),'split_file':str(split_file),
        'split_sha256':file_sha256(split_file),
        'predictions.npy':('relative_depth = 1/max(DAV2_raw_inverse_depth, epsilon)'
                           if name==DAV2 else 'native UniDepth metric z-depth, without alignment or clipping'),
        'raw_inverse_depth.npy':('unmodified DAV2 DPT output' if name==DAV2 else 'reciprocal of native UniDepth depth; visualization only'),
        'epsilon':m.get('inverse_epsilon',1e-3) if name==DAV2 else None,
        'alignment':'none; median scaling done only by evaluator',
        'ordering':'sorted by STEPS test split, stored in index.json',
        'evaluated_split':split,'evaluation_protocol':protocol,
        'model_identity':getattr(model,'inference_metadata',None),
        'checkpoint_sha256':file_sha256(project_path(checkpoint)) if checkpoint else None,
        'base_weights_sha256':(file_sha256(project_path(m['weights']))
                               if name==DAV2 and project_path(m['weights']).is_file() else None),
        'code_sha256':{p:file_sha256(code_root/p) for p in source_files}}
    (out/'prediction_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    print('Saved',len(depth),'predictions into',out)
    if evaluate:
        from evaluation.protocol import evaluate_records
        summary=evaluate_records(out,align='median',min_depth=ev['min_depth'],
                                 max_depth=ev['max_depth'],dataset=data['name'])
        print('Seven depth metrics (per-image median aligned):')
        print(' '.join(protocol['metrics']))
        print(' '.join('{:.6f}'.format(summary[k]) for k in protocol['metrics']))
        return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--out')
    p.add_argument('--vis',action='store_true')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--split',choices=['day','night'])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--evaluate',action='store_true',help='Evaluate saved depth with the existing median-aligned seven-metric protocol')
    a=p.parse_args();run(a.config,a.out,a.checkpoint,a.vis,a.limit,a.split,a.evaluate,a.device)
if __name__=='__main__':main()

