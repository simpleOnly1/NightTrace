"""Stream image checkpoints; aggregate repeats -> images -> groups, not pixels."""
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import numpy as np
from .sweep_store import CsvSink, collect_shards, atomic_json
from .statistics import group_estimate, write_csv, factorial_effects
from .failure_boundary import first_crossing


def summarize_runs(paths, out, allow_existing=False, curves=True, n_boot=500, penalty=.1):
    if n_boot < 20 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Require bootstrap >= 20 and a finite nonnegative penalty')
    contract, records = collect_shards(paths)
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not allow_existing:
        raise ValueError('Analysis output is not empty; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    has_gt = contract['source'] == 'day-test' and not contract['render_only']
    expected = {(r, c, s) for r in range(contract['repeats'])
                for c, s in [('clean', 0.)] + [(c, s) for c in contract['conditions'] for s in contract['levels']]}
    sample_groups = {s['id']: s['group'] for s in contract['samples']}
    acc = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    effect_acc = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    excluded = {'id', 'group', 'repeat', 'condition', 'severity', 'parameter_seed'}
    count = 0; templates = {}; input_hashes = []
    with ExitStack() as stack:
        def sink(name, optional=()):
            result = CsvSink(out/name, optional); stack.callback(result.close); return result
        table = sink('per_image.csv')
        params = sink('parameters.csv', ('template', 'reflective_template'))
        warnings = sink('audit_warnings.csv')
        bounds = sink('boundaries_per_image.csv') if has_gt else None
        effects = sink('interactions_per_image.csv') if has_gt else None
        journal = stack.enter_context((out/'parameters.jsonl').open('w', encoding='utf-8'))
        for sid, path in records:
            payload = json.loads(path.read_text(encoding='utf-8')); rows = payload['rows']
            keys = [(r['repeat'], r['condition'], r['severity']) for r in rows]
            if payload.get('id') != sid or not payload.get('complete') or len(keys) != len(expected) or set(keys) != expected:
                raise ValueError('Incomplete/duplicate checkpoint rows for '+sid)
            if any(r['id'] != sid or r['group'] != sample_groups[sid] for r in rows):
                raise ValueError('Checkpoint IDs/groups differ from contract')
            if has_gt:
                for r in rows:
                    delta = r['delta_gt_ordinal_error']
                    r['gt_failure_rate'] = float(delta >= contract['failure_delta']) if np.isfinite(delta) else float('nan')
            table.write(rows); params.write(payload['parameters']); warnings.write(payload['warnings'])
            for record in payload['parameter_records']:
                journal.write(json.dumps(record)+'\n')
            input_hashes.append(payload['input_hash'])
            for name, digest in payload['template_hashes'].items():
                if name in templates and templates[name] != digest:
                    raise ValueError('Template changed between images/shards: '+name)
                templates[name] = digest
            values = defaultdict(list)
            for r in rows:
                for k, v in r.items():
                    if k not in excluded and isinstance(v, (int, float)) and np.isfinite(v):
                        values[(r['condition'], r['severity'], k)].append(v)
            for key, vals in values.items():
                cell = acc[key][sample_groups[sid]]; cell[0] += float(np.mean(vals)); cell[1] += 1
            # Preserve all-missing cells so undefined GT never becomes an absent condition.
            for r in rows:
                for k, v in r.items():
                    if k not in excluded and isinstance(v, (int, float)):
                        acc[(r['condition'], r['severity'], k)]
            if has_gt:
                for repeat in range(contract['repeats']):
                    for cond in contract['conditions']:
                        subset = sorted([r for r in rows if r['repeat'] == repeat and r['condition'] == cond], key=lambda r: r['severity'])
                        vals = [r['delta_gt_ordinal_error'] for r in subset]
                        bounds.write([dict(id=sid, group=sample_groups[sid], repeat=repeat, condition=cond,
                            metric='delta_gt_ordinal_error', threshold=contract['failure_delta'],
                            **first_crossing([0.] + contract['levels'], [0.] + vals, contract['failure_delta']))])
                image_effects = defaultdict(list)
                for severity in contract['levels']:
                    subset = [r for r in rows if r['condition'] == 'clean' or r['severity'] == severity]
                    for metric in ('gt_ordinal_error', 'gt_abs_rel'):
                        for effect in factorial_effects(subset, metric):
                            effects.write([dict(severity=severity, metric=metric, **effect)])
                            for key in ('L', 'N', 'G', 'LN', 'LG', 'NG', 'LNG'):
                                if np.isfinite(effect[key]):
                                    image_effects[(severity, metric, key)].append(effect[key])
                for key, vals in image_effects.items():
                    cell = effect_acc[key][sample_groups[sid]]; cell[0] += float(np.mean(vals)); cell[1] += 1
            count += len(rows)
    summary = []; group_rows = []
    for (condition, severity, metric), groups in sorted(acc.items()):
        compact = [dict(group=g, id=g, value=s/n) for g, (s, n) in groups.items() if n]
        estimate = group_estimate(compact, 'value', contract['seed'], n_boot)
        estimate['images'] = sum(n for _, n in groups.values())
        summary.append(dict(condition=condition, severity=severity, metric=metric, **estimate))
        for g, (total, n) in groups.items():
            group_rows.append(dict(group=g, condition=condition, severity=severity, metric=metric, value=total/n, images=n))
    write_csv(out/'summary.csv', summary); write_csv(out/'group_curves.csv', group_rows)
    interaction_summary = []
    for (severity, metric, effect), groups in sorted(effect_acc.items()):
        compact = [dict(group=g, id=g, value=s/n) for g, (s, n) in groups.items()]
        estimate = group_estimate(compact, 'value', contract['seed'], n_boot)
        estimate['images'] = sum(n for _, n in groups.values())
        interaction_summary.append(dict(severity=severity, metric=metric, effect=effect, **estimate))
    write_csv(out/'interactions_summary.csv', interaction_summary)
    if has_gt:
        group_bounds = []
        for c in contract['conditions']:
            subset = [next(r for r in summary if r['condition'] == c and r['severity'] == s and
                           r['metric'] == 'delta_gt_ordinal_error') for s in contract['levels']]
            for key in ('mean', 'low'):
                group_bounds.append(dict(condition=c, metric='delta_gt_ordinal_error', threshold=contract['failure_delta'],
                    rule='group_mean' if key == 'mean' else 'pointwise_95pct_lower_bound',
                    **first_crossing([0.]+contract['levels'], [0.]+[r[key] for r in subset], contract['failure_delta'])))
        write_csv(out/'boundaries_group.csv', group_bounds)
    from visualization.severity import save_response_curves
    save_response_curves(out)
    if curves and has_gt:
        from .sweep_curves import fit_group_curves
        fit_group_curves(out, group_rows, contract, n_boot, penalty)
    atomic_json(out/'template_sha256.json', templates)
    atomic_json(out/'input_sha256.json', input_hashes)
    analysis_code = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ('sweep_summary.py', 'sweep_curves.py', 'sweep_store.py',
                                  'statistics.py', 'failure_boundary.py')}
    atomic_json(out/'analysis.json', dict(status='complete', images=len(records), rows=count,
        contract=contract, bootstrap=n_boot, penalty=penalty,
        analysis_code_sha256=analysis_code, fitted_curves_requested=bool(curves and has_gt),
        fit_status_file='curve_status.csv' if curves and has_gt else None))
    print('Summarized', len(records), 'images and', count, 'rows into', out)
    return summary
