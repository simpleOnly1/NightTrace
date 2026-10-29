"""Audit RGB/GT field-of-view provenance before publishing depth metrics.

A shape match is necessary but NOT proof of LiDAR projection/calibration.
The optional overlays are for human inspection only. The evaluator never
resizes/interpolates GT to hide a mismatch.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import cv2
import numpy as np
from datasets import make_dataset
from evaluation.protocol import prepare_gt
from utils.config import load_config,project_path


def overlay_pair(rgb_path, gt_aligned, geometry, min_depth, max_depth):
    """Return original-RGB crop-window image and pixel-aligned sparse-GT overlay."""
    raw=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
    if raw is None:raise FileNotFoundError(rgb_path)
    rh,rw=geometry['raw_hw']
    if tuple(raw.shape[:2])!=(rh,rw):
        raise ValueError('Recorded original RGB dimensions changed: '+str(rgb_path))
    x,y,w,h=map(int,geometry['crop_xywh'])
    if gt_aligned.shape!=(h,w):raise ValueError('GT and crop FOV disagree')
    window=raw.copy()
    cv2.rectangle(window,(x,y),(x+w-1,y+h-1),(255,255,255),2)
    roi=raw[y:y+h,x:x+w].copy()
    valid=np.isfinite(gt_aligned)&(gt_aligned>min_depth)&(gt_aligned<max_depth)
    colors=np.zeros_like(roi)
    if valid.any():
        scaled=np.clip((gt_aligned-min_depth)/max(max_depth-min_depth,1e-12),0,1)
        colors=cv2.applyColorMap((255*scaled).astype(np.uint8),cv2.COLORMAP_TURBO)
        roi[valid]=np.rint(.50*roi[valid]+.50*colors[valid]).astype(np.uint8)
    return window,roi


def audit(cfg, out, limit=0, overlays=0):
    if limit<0 or overlays<0:raise ValueError('limit/overlays must be nonnegative')
    ds=make_dataset(cfg,split=cfg['eval']['split'],training=False)
    if not len(ds):raise ValueError('Empty evaluation set')
    n=min(len(ds),limit) if limit else len(ds)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    vis_ids=set(np.linspace(0,n-1,min(n,overlays),dtype=np.int64).tolist()) if overlays else set()
    seen=set();rows=[]
    min_depth=float(cfg['eval']['min_depth']);max_depth=float(cfg['eval']['max_depth'])
    for i in range(n):
        rec=ds[i]
        if rec['id'] in seen:raise ValueError('Duplicate evaluation ID: '+rec['id'])
        seen.add(rec['id'])
        path=Path(rec['gt_path'])
        if not path.is_file():raise FileNotFoundError('Missing GT: '+str(path))
        gt=np.load(str(path),allow_pickle=False)
        try:
            aligned=prepare_gt(gt,rec['geometry'])
        except ValueError as e:
            g=rec['geometry'];x,y,w,h=g['crop_xywh']
            hint=('GT shape suggests an already-cropped source; independently verify '
                  'provenance, then set data.gt_space: cropped' if gt.shape==(h,w)
                  else 'Check original RGB size, GT projection source and crop metadata')
            raise ValueError('GT FOV invalid for '+str(rec['id'])+': '+str(e)+'; '+hint) from e
        valid=np.isfinite(aligned)&(aligned>min_depth)&(aligned<max_depth)
        if not valid.any():raise ValueError('No valid GT pixels in evaluation range for '+rec['id'])
        g=rec['geometry']
        rows.append(dict(index=i,id=rec['id'],gt_space=g['gt_space'],
            raw_height=g['raw_hw'][0],raw_width=g['raw_hw'][1],
            crop_x=g['crop_xywh'][0],crop_y=g['crop_xywh'][1],
            crop_width=g['crop_xywh'][2],crop_height=g['crop_xywh'][3],
            gt_raw_height=gt.shape[0],gt_raw_width=gt.shape[1],
            aligned_height=aligned.shape[0],aligned_width=aligned.shape[1],
            valid_gt_pixels=int(valid.sum()),valid_gt_fraction=float(valid.mean()),
            depth_min=float(np.min(aligned[valid])),depth_median=float(np.median(aligned[valid])),
            depth_max=float(np.max(aligned[valid]))))
        if i in vis_ids:
            raw,roi=overlay_pair(ds.records[i]['target'],aligned,g,min_depth,max_depth)
            stem='%04d_%s'%(i,str(rec['id']).replace('/','_'))
            cv2.imwrite(str(out/(stem+'_raw_crop_window.png')),raw)
            cv2.imwrite(str(out/(stem+'_aligned_sparse_GT.png')),roi)
        if i<3:
            print('Sample',i,rec['id'],'raw',g['raw_hw'],'crop',g['crop_xywh'],
                  'GT',gt.shape,'aligned',aligned.shape,'space',g['gt_space'])
    with (out/'audit.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'checked':n,'available':len(ds),'dataset':cfg['data']['name'],
             'gt_space':cfg['data'].get('gt_space','raw'),
             'eval_depth_range_m':[min_depth,max_depth],
             'valid_gt_fraction_mean':float(np.mean([r['valid_gt_fraction'] for r in rows])),
             'overlays':len(vis_ids),
             'projection_provenance_verified':False,
             'warning':'Pixel shapes passed; calibration/temporal alignment still require human inspection.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('PASS: strict GT shape/FOV checks:',n,'records. Audit:',out)
    print('IMPORTANT: Review overlays against camera/LiDAR projection provenance; shape alone is NOT proof.')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--split',choices=['day','night'],help='Override eval.split for GT auditing')
    p.add_argument('--limit',type=int,default=0,help='0 checks the full official test split')
    p.add_argument('--overlays',type=int,default=0,help='Number of evenly spread sample overlays')
    p.add_argument('--out',default='runs/gt_geometry_audit')
    a=p.parse_args()
    cfg=load_config(a.config)
    if a.split:cfg['eval']['split']=a.split
    audit(cfg,project_path(a.out),a.limit,a.overlays)


if __name__=='__main__':main()

