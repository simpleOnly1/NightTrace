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
    # A corrupt prediction must not make a difficult valid-GT pixel disappear.
    # STEPS masks by GT only; excluding invalid predictions silently biases results.
    valid=(gt>min_depth)&(gt<max_depth)&np.isfinite(gt)
    if valid_mask is not None:
        valid_mask=np.asarray(valid_mask)
        if valid_mask.shape!=gt.shape:
            raise ValueError('valid_mask must match the aligned GT grid')
        valid &= valid_mask.astype(bool)
    if not valid.any():
        raise ValueError('Zero evaluable ground-truth pixels')
    invalid_pred=valid & (~np.isfinite(pred) | (pred<=0))
    if invalid_pred.any():
        raise ValueError('%d nonfinite/nonpositive predictions at valid GT pixels' % int(invalid_pred.sum()))
    pv=pred[valid].copy()
    gv=gt[valid].copy()
    if align=='median':
        den=np.median(pv)
        if den<=0:raise ValueError('Nonpositive prediction median')
        a=np.median(gv)/den;b=0.0
    elif align=='scale_shift':
        a,b=np.linalg.lstsq(np.stack([pv,np.ones_like(pv)],axis=1),gv,rcond=None)[0]
        a=max(float(a),1e-12);b=float(np.mean(gv)-a*np.mean(pv))
    elif align=='none':a,b=1.0,0.0
    else:raise ValueError(align)
    pv=np.clip(a*pv+b,min_depth,max_depth)
    result=metrics(gv,pv)
    result.update({'scale':float(a),'shift':float(b),'valid_pixels':int(valid.sum())})
    return result


def evaluate_records(pred_dir, out_file=None, align='median', min_depth=1e-5,
                     max_depth=60.0, dataset=None):
    pred_dir=Path(pred_dir)
    index=json.loads((pred_dir/'index.json').read_text())
    pred=np.load(str(pred_dir/'predictions.npy'),allow_pickle=False)
    if len(index)!=len(pred):
        raise ValueError('Prediction/index lengths differ: %d vs %d'%(len(pred),len(index)))
    if not index:raise ValueError('Empty prediction index')
    rows=[]
    for i,rec in enumerate(index):
        gt_path=Path(rec['gt_path'])
        if not gt_path.is_file():raise FileNotFoundError(gt_path)
        if dataset and rec.get('dataset')!=dataset:
            raise ValueError('Prediction dataset does not match evaluator; regenerate predictions')
        geometry=rec.get('geometry')
        if geometry is None:
            raise ValueError('Missing RGB/GT geometry in index.json; regenerate predictions with test.py')
        if list(pred[i].shape[-2:]) != geometry['input_hw']:
            raise ValueError('Prediction resolution disagrees with recorded geometry')
        gt=prepare_gt(np.load(str(gt_path)),geometry)
        r=eval_pair(pred[i],gt,min_depth,max_depth,align)
        r['id']=rec['id'];rows.append(r)
    summary={k:float(np.mean([r[k] for r in rows])) for k in METRIC_NAMES}
    summary['num_images']=len(rows)
    summary['alignment']=align
    summary['min_depth']=min_depth;summary['max_depth']=max_depth
    summary['geometry']='explicit raw/cropped GT; prediction resize only after FOV alignment'
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


def prepare_gt(gt, geometry):
    """Align fields of view using explicit provenance; never infer crop from size.

    raw: GT was projected into the original RGB grid.
    cropped: GT already covers exactly crop_xywh (user must verify provenance).
    """
    gt=np.asarray(gt)
    if gt.ndim!=2:raise ValueError('GT must be a 2D array')
    rh,rw=map(int,geometry['raw_hw'])
    x,y,w,h=map(int,geometry['crop_xywh'])
    if min(x,y)<0 or min(w,h)<=0 or x+w>rw or y+h>rh:
        raise ValueError('Invalid RGB crop geometry')
    space=geometry['gt_space']
    if space=='raw':
        if gt.shape!=(rh,rw):
            raise ValueError('Raw GT shape %s != raw RGB %s; verify gt_space, do not resize GT' % (gt.shape,(rh,rw)))
        return gt[y:y+h,x:x+w]
    if space=='cropped':
        if gt.shape!=(h,w):
            raise ValueError('Precropped GT does not match recorded RGB crop')
        return gt
    raise ValueError('gt_space must be raw or cropped')
