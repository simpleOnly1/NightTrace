"""STEPS-style relative-depth protocol.

Predictions.npy ALWAYS stores predicted positive depth maps (not raw DAV2 q).
For DAV2-L, raw q is inverse-depth-like; 1/max(q,eps) produces relative depth.
No min-max normalization, no GT during inference. Median scaling at evaluation.

Preserves the STEPS evaluation calculations in NightTrace native interfaces.
The official STEPS implementation may be verified using tools/check_parity.py.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from .metrics import METRIC_NAMES, metrics


def eval_pair(depth, gt, min_depth=1e-5, max_depth=60.0, align='median', valid_mask=None):
    gt=np.squeeze(np.asarray(gt,dtype=np.float64))
    pred=np.squeeze(np.asarray(depth,dtype=np.float64))
    if pred.ndim != 2 or gt.ndim!=2:
        raise ValueError('Both pred and GT must be 2D HxW')
    if pred.shape!=gt.shape:
        # STEPS: prediction resized to GT by nearest, never vice versa.
        pred=cv2.resize(pred,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
    valid=(gt>min_depth)&(gt<max_depth)&np.isfinite(gt)&np.isfinite(pred)&(pred>0)
    if valid_mask is not None:
        valid &= valid_mask.astype(bool)
    if not valid.any():
        raise ValueError('Zero evaluable ground-truth pixels')
    pv=pred[valid].copy()
    gv=gt[valid].copy()
    if align=='median':
        den=np.median(pv)
        if den<=0:raise ValueError('Nonpositive prediction median')
        a=np.median(gv)/den;b=0.0
    elif align=='scale_shift':
        a,b=np.linalg.lstsq(np.stack([pv,np.ones_like(pv)],axis=1),gv,rcond=None)[0]
    elif align=='none':a,b=1.0,0.0
    else:raise ValueError(align)
    pv=np.clip(a*pv+b,min_depth,max_depth)
    result=metrics(gv,pv)
    result.update({'scale':float(a),'shift':float(b),'valid_pixels':int(valid.sum())})
    return result


def evaluate_records(pred_dir, out_file=None, align='median', min_depth=1e-5,
                     max_depth=60.0):
    pred_dir=Path(pred_dir)
    index=json.loads((pred_dir/'index.json').read_text())
    pred=np.load(str(pred_dir/'predictions.npy'),allow_pickle=False)
    if len(index)!=len(pred):
        raise ValueError('Prediction/index lengths differ: %d vs %d'%(len(pred),len(index)))
    rows=[]
    for i,rec in enumerate(index):
        gt_path=Path(rec['gt_path'])
        if not gt_path.is_file():raise FileNotFoundError(gt_path)
        gt=np.load(str(gt_path))
        r=eval_pair(pred[i],gt,min_depth,max_depth,align)
        r['id']=rec['id'];rows.append(r)
    summary={k:float(np.mean([r[k] for r in rows])) for k in METRIC_NAMES}
    summary['num_images']=len(rows)
    summary['alignment']=align
    output=Path(out_file) if out_file else pred_dir/'metrics.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'summary':summary,'per_image':rows},indent=2))
    return summary

# Old signature kept for compatibility with scripts that already import this.
def run_evaluation(pred_dir,out=None,alignment='median',kind='relative_depth',
                   min_depth=1e-5,max_depth=60.0,allow_resize=False):
    if kind!='relative_depth':
        raise ValueError('predictions.npy must store depth; kind must be relative_depth')
    return evaluate_records(pred_dir,out,alignment,min_depth,max_depth)


def align_gt_geometry(gt, desired, allow_resize=False):
    """Legacy dataset-test helper. Not used by the official eval protocol."""
    gt=np.squeeze(np.asarray(gt))
    h,w=desired
    if gt.shape==(h,w):return gt
    if gt.ndim==2 and gt.shape[0]>=h and gt.shape[1]>=w:
        y=(gt.shape[0]-h)//2;x=(gt.shape[1]-w)//2
        return gt[y:y+h,x:x+w]
    if allow_resize:
        return cv2.resize(gt,(w,h),interpolation=cv2.INTER_NEAREST)
    raise ValueError('GT size mismatch')
