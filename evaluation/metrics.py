"""Seven STEPS depth measures (adapted from STEPS eval_nuscenes.py).

Only the seven mathematical measures are copied/maintained. Old Lightning/MMCV
imports are not needed by metric evaluation. MIT-licensed STEPS credited in
ATTRIBUTION.md.
"""
import numpy as np

METRIC_NAMES = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')


def compute_metrics(pred, gt):
    """STEPS argument order: predicted depth first, ground-truth depth second."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse,
            'rmse_log': rmse_log, 'a1': a1, 'a2': a2, 'a3': a3}


def metrics(gt, pred):
    return {k: float(v) for k, v in compute_metrics(pred, gt).items()}


def relative_to_depth(q, kind='relative_inverse_depth', eps=1e-3):
    q = np.asarray(q)
    if kind in ('relative_inverse_depth', 'relative_disparity'):
        return 1.0 / np.maximum(q, eps)
    if kind == 'relative_depth':
        return np.maximum(q, eps)
    raise ValueError(kind)


def evaluate_single(relative, gt, alignment='median', kind='relative_disparity',
                    min_depth=0.1, max_depth=100.0):
    from .protocol import eval_pair
    depth = relative_to_depth(relative, kind)
    result = eval_pair(depth, gt, min_depth, max_depth, alignment)
    valid = (np.asarray(gt) > min_depth) & (np.asarray(gt) < max_depth)
    return result, depth, valid
