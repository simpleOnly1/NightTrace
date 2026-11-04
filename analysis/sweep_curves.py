"""Descriptive regularized linear splines and paired group-bootstrap refits.

No imposed sigmoid/monotonicity, no extrapolation, no universal threshold claim.
Failure probability averages repeat indicators within images, then within groups.
"""
import numpy as np
from .statistics import write_csv


def smoother(levels, penalty):
    x = np.asarray(levels, dtype=float)
    if len(x) < 4 or x[0] != 0 or np.any(np.diff(x) <= 0):
        raise ValueError('Curve fitting needs S0 and at least three distinct positive doses')
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('penalty must be finite and nonnegative')
    d = np.zeros((len(x)-2, len(x)))
    for i in range(len(x)-2):
        a, b = x[i+1]-x[i], x[i+2]-x[i+1]
        d[i, i:i+3] = [1/a, -1/a-1/b, 1/b]
    # Baseline delta and failure rate are identically zero; pin S0 exactly.
    weights = np.zeros((len(x), len(x)))
    weights[1:, 1:] = np.linalg.solve(np.eye(len(x)-1)+penalty*d[:, 1:].T@d[:, 1:], np.eye(len(x)-1))
    return weights


def crossing(x, y, threshold):
    hits = np.flatnonzero(np.asarray(y) >= threshold)
    if not len(hits):
        return float('nan'), False
    j = int(hits[0]); value = float(x[j])
    if j and y[j] != y[j-1]:
        value = float(x[j-1]+(threshold-y[j-1])*(x[j]-x[j-1])/(y[j]-y[j-1]))
    return value, bool(np.any(np.asarray(y)[j+1:] < threshold))


def interval(values):
    values = np.asarray(values); finite = values[np.isfinite(values)]
    # These are explicitly CONDITIONAL on an observed crossing. Missing crossings
    # are reported separately; they are never quietly replaced with S4.
    q = np.quantile(finite, [.025, .975]) if len(finite) else [float('nan')]*2
    return dict(bootstrap_observed_fraction=float(len(finite)/len(values)),
                conditional_p025=float(q[0]), conditional_p975=float(q[1]))


def fit_group_curves(out, rows, contract, n_boot=500, penalty=.1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.array([0.]+contract['levels'])
    if n_boot < 20:
        raise ValueError('At least 20 bootstrap draws required (500+ recommended)')
    if len(x) < 4:
        write_csv(out/'curve_status.csv', [dict(status='insufficient_doses', required='S0 + at least 3 positive doses')])
        return
    weights = smoother(x, penalty)
    grid = np.linspace(0, x[-1], 201)
    basis = np.stack([np.interp(grid, x, np.eye(len(x))[i]) for i in range(len(x))], axis=1)
    conditions = contract['conditions']
    curves = []; boundaries = []; shifts = []; statuses = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, metric in zip(axes, ('delta_gt_ordinal_error', 'gt_failure_rate')):
        lookup = {(r['group'], r['condition'], r['severity']): r['value'] for r in rows if r['metric'] == metric}
        all_groups = sorted({r['group'] for r in rows if r['metric'] == metric})
        groups = [g for g in all_groups if all(np.isfinite(lookup.get((g, c, s), np.nan))
                  for c in conditions for s in contract['levels'])]
        statuses.append(dict(metric=metric, groups=len(groups), excluded_incomplete_groups=len(all_groups)-len(groups),
                             status='ok' if len(groups) >= 2 else 'insufficient_groups'))
        if len(groups) < 2:
            ax.set_title(metric+' (insufficient groups)'); continue
        data = np.array([[[0.]+[lookup[g, c, s] for s in contract['levels']] for c in conditions] for g in groups])
        rng = np.random.default_rng(contract['seed'])
        # Same resampled groups across all conditions: retain pairing for shifts.
        boot_means = np.stack([data[rng.integers(len(groups), size=len(groups))].mean(0) for _ in range(n_boot)])
        estimate = data.mean(0)@weights.T@basis.T
        boot = boot_means@weights.T@basis.T
        if metric == 'gt_failure_rate':
            estimate = np.clip(estimate, 0, 1); boot = np.clip(boot, 0, 1)
        low, high = np.quantile(boot, [.025, .975], axis=0)
        thresholds = [contract['failure_delta']] if metric == 'delta_gt_ordinal_error' else [.1, .9]
        for ci, c in enumerate(conditions):
            for j, s in enumerate(grid):
                curves.append(dict(metric=metric, condition=c, severity=float(s), mean=float(estimate[ci, j]),
                    low=float(low[ci, j]), high=float(high[ci, j]), groups=len(groups), penalty=penalty))
            line, = ax.plot(grid, estimate[ci], label=c)
            ax.scatter(x, data.mean(0)[ci], s=13, color=line.get_color())
            ax.fill_between(grid, low[ci], high[ci], alpha=.10, color=line.get_color())
        for threshold in thresholds:
            crossings = []
            points = []
            for ci, c in enumerate(conditions):
                point, recrossed = crossing(grid, estimate[ci], threshold)
                draws = np.array([crossing(grid, y, threshold)[0] for y in boot[:, ci]])
                crossings.append(draws); points.append(point)
                boundaries.append(dict(metric=metric, condition=c, threshold=threshold,
                    crossing=point, status='observed' if np.isfinite(point) else 'not_observed', recrossed=recrossed,
                    groups=len(groups), **interval(draws)))
            if metric == 'delta_gt_ordinal_error':
                for joint_name in ('LN', 'LG', 'NG', 'LNG'):
                    if joint_name not in conditions: continue
                    joint = conditions.index(joint_name)
                    for c in joint_name:
                        if c not in conditions: continue
                        ci = conditions.index(c)
                        shifts.append(dict(single=c, joint=joint_name, metric=metric, threshold=threshold,
                            earlier_by=float(points[ci]-points[joint]),
                            **interval(crossings[ci]-crossings[joint])))
            ax.axhline(threshold, color='gray', linestyle='--', linewidth=.7)
        ax.set_title(metric); ax.set_xlabel('Experimental severity; no extrapolation')
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Descriptive linear-spline fits; pointwise group-bootstrap 95% intervals')
    fig.savefig(out/'fitted_response_curves.png', dpi=160); plt.close(fig)
    write_csv(out/'curve_status.csv', statuses)
    write_csv(out/'fitted_curves.csv', curves)
    write_csv(out/'fitted_boundaries.csv', boundaries)
    write_csv(out/'joint_boundary_shifts.csv', shifts)
