"""Independent GT ordering and descriptive, sampled threshold crossings."""
from collections import defaultdict
import cv2
import numpy as np
from evaluation.protocol import eval_pair
from analysis.statistics import group_estimate, write_csv, factorial_effects


def gt_pairs(gt, min_depth, max_depth, seed=0, pairs=8192, relative_gap=.1):
    """Fixed global pairs on valid GT; this is not a boundary/contact annotation.

    Exclude near-ties by |d1-d2|/min(d1,d2) > relative_gap. Deduplicate
    unordered pairs so repeated draws cannot inflate minimum support.
    Pixel pairs are NOT independent scenes.
    """
    if pairs < 1 or not np.isfinite(relative_gap) or relative_gap <= 0:
        raise ValueError('Positive pair count and relative gap required')
    g = np.asarray(gt, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.isfinite(g) & (g > min_depth) & (g < max_depth))
    if len(valid) < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0)
    rng = np.random.default_rng(seed)
    i = rng.choice(valid, pairs); j = rng.choice(valid, pairs)
    good = np.abs(g[i]-g[j]) > relative_gap*np.minimum(g[i], g[j])
    unique = np.unique(np.sort(np.stack([i[good], j[good]], axis=1), axis=1), axis=0)
    i, j = unique[:, 0], unique[:, 1]
    return i, j, np.sign(g[i]-g[j])


def evaluate_gt(depth, gt, pairs, min_depth, max_depth, min_pairs=128):
    depth = np.asarray(depth, dtype=np.float64)
    metrics = eval_pair(depth, gt, min_depth, max_depth, align='median')
    # Keep ordering BEFORE per-condition scale/clipping. Invalid predictions
    # at valid GT pixels have already failed eval_pair, never dropped here.
    if depth.shape != gt.shape:
        depth = cv2.resize(depth, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    i, j, signs = pairs
    delta = depth.ravel()[i]-depth.ravel()[j]
    error = np.where(delta == 0, .5, (np.sign(delta) != signs).astype(float))
    return {**{'gt_'+k: v for k, v in metrics.items()},
            'gt_ordinal_error': float(error.mean()) if len(i) >= min_pairs else float('nan'),
            'gt_ordinal_pairs': len(i)}


def first_crossing(levels, values, threshold):
    """No monotonicity assumption and no interpolated/physical critical point."""
    levels = np.asarray(levels, dtype=float); values = np.asarray(values, dtype=float)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Failure threshold must be positive and finite')
    if len(levels) != len(values) or np.any(np.diff(levels) <= 0):
        raise ValueError('Crossing requires aligned, strictly increasing doses')
    if not len(values) or not np.isfinite(values).all():
        return dict(status='insufficient_data', lower=float('nan'), upper=float('nan'), recrossed=False)
    hit = np.flatnonzero(values >= threshold)
    if not len(hit):
        return dict(status='not_observed', lower=float(levels[-1]), upper=float('nan'), recrossed=False)
    k = int(hit[0])
    return dict(status='left_censored' if k == 0 else 'bracketed',
                lower=float(levels[k-1]) if k else float('nan'), upper=float(levels[k]),
                recrossed=bool(np.any(values[k+1:] < threshold)))


def save_sweep_statistics(out, rows, seed, threshold=.1):
    metadata = {'id', 'group', 'repeat', 'condition', 'severity', 'parameter_seed'}
    numeric = sorted({k for r in rows for k, v in r.items()
                      if k not in metadata and isinstance(v, (int, float))})
    summary = []
    levels = sorted({r['severity'] for r in rows if r['severity'] > 0})
    conditions = list(dict.fromkeys(r['condition'] for r in rows if r['condition'] != 'clean'))
    for condition, severity in dict.fromkeys((r['condition'], r['severity']) for r in rows):
        subset = [r for r in rows if r['condition'] == condition and r['severity'] == severity]
        for key in numeric:
            subset_key = [r for r in subset if key in r]
            if subset_key:
                summary.append(dict(condition=condition, severity=severity, metric=key,
                                    **group_estimate(subset_key, key, seed)))
    write_csv(out/'summary.csv', summary)
    metric = 'delta_gt_ordinal_error'
    if not any(metric in r for r in rows):
        return  # No GT => no file purporting to estimate failure.
    paired = defaultdict(list)
    for r in rows:
        if r['condition'] != 'clean':
            paired[(r['id'], r['group'], r['repeat'], r['condition'])].append(r)
    boundaries = []
    for (sid, group, repeat, condition), subset in paired.items():
        subset.sort(key=lambda r: r['severity'])
        valid_baseline = np.isfinite(subset[0][metric])
        boundaries.append(dict(id=sid, group=group, repeat=repeat, condition=condition,
            metric=metric, threshold=threshold,
            **first_crossing([0.] + [r['severity'] for r in subset],
                             [0. if valid_baseline else float('nan')] + [r[metric] for r in subset], threshold)))
    write_csv(out/'boundaries_per_image.csv', boundaries)
    population = []
    for condition in conditions:
        estimates = [next(r for r in summary if r['condition'] == condition and
                         r['severity'] == s and r['metric'] == metric) for s in levels]
        # Missing GT must not silently change the cohort between dose levels.
        for measure in ('mean', 'low'):
            values = [r[measure] for r in estimates]
            population.append(dict(condition=condition, metric=metric, threshold=threshold,
                rule='group_mean' if measure == 'mean' else 'pointwise_95pct_lower_bound',
                **first_crossing([0.] + levels, [0.] + values, threshold)))
    write_csv(out/'boundaries_group.csv', population)
    interactions = []
    for severity in levels:
        paired_rows = [r for r in rows if r['severity'] == severity or r['condition'] == 'clean']
        for key in ('gt_ordinal_error', 'gt_abs_rel'):
            for effect in factorial_effects(paired_rows, key):
                interactions.append(dict(severity=severity, metric=key, **effect))
    write_csv(out/'interactions_per_image.csv', interactions)
    effects_summary = []
    for severity, metric in dict.fromkeys((r['severity'], r['metric']) for r in interactions):
        subset = [r for r in interactions if r['severity'] == severity and r['metric'] == metric]
        for effect in ('L', 'N', 'G', 'LN', 'LG', 'NG', 'LNG'):
            effects_summary.append(dict(severity=severity, metric=metric, effect=effect,
                                        **group_estimate(subset, effect, seed)))
    write_csv(out/'interactions_summary.csv', effects_summary)
