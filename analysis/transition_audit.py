"""Post-hoc transition checks and explicitly outcome-selected pilot manifests.

Standard library only. Never changes synthesis, fits a curve, or drops bad rows.
Threshold sensitivity is descriptive, not independent confirmation.
"""
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


FOCUS = ('LN', 'LNG')
CONDITIONS = {'clean', 'L', 'N', 'G', 'LN', 'LG', 'NG', 'LNG'}


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'id', 'group', 'repeat', 'condition', 'severity',
                'gt_ordinal_error', 'delta_gt_ordinal_error'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Need nonempty per_image.csv with GT ordering and delta fields')
    keys = set()
    groups = {}
    for r in rows:
        for field in ('severity', 'gt_ordinal_error', 'delta_gt_ordinal_error'):
            r[field] = float(r[field])
            if not math.isfinite(r[field]):
                raise ValueError('Missing/nonfinite GT or dose; no silent deletion: '+r['id'])
        r['repeat'] = int(r['repeat'])
        if (not r['id'] or not r['group'] or r['repeat'] < 0 or
                not 0 <= r['gt_ordinal_error'] <= 1 or
                not -1 <= r['delta_gt_ordinal_error'] <= 1 or
                not 0 <= r['severity'] <= 4 or r['condition'] not in CONDITIONS or
                ((r['condition'] == 'clean') != (r['severity'] == 0))):
            raise ValueError('Invalid row: '+str(r))
        key = (r['id'], r['repeat'], r['condition'], r['severity'])
        if key in keys:
            raise ValueError('Duplicate observation: '+str(key))
        keys.add(key)
        if r['id'] in groups and groups[r['id']] != r['group']:
            raise ValueError('An image has multiple groups: '+r['id'])
        groups[r['id']] = r['group']
    levels = sorted({r['severity'] for r in rows if r['severity'] > 0})
    conditions = sorted({r['condition'] for r in rows} - {'clean'})
    repeats = sorted({r['repeat'] for r in rows})
    if not levels or not set(FOCUS).issubset(conditions):
        raise ValueError('Audit requires both LN and LNG and positive doses')
    expected = {(rep, c, s) for rep in repeats for c in conditions for s in levels}
    expected |= {(rep, 'clean', 0.) for rep in repeats}
    by_image = defaultdict(dict)
    for r in rows:
        by_image[r['id']][(r['repeat'], r['condition'], r['severity'])] = r
    for sid, observations in by_image.items():
        if set(observations) != expected:
            raise ValueError('Incomplete paired dose/repeat grid: '+sid)
        for (rep, c, s), r in observations.items():
            base = observations[(rep, 'clean', 0.)]
            residual = r['gt_ordinal_error'] - base['gt_ordinal_error'] - r['delta_gt_ordinal_error']
            if abs(residual) > 1e-8:
                raise ValueError('Delta disagrees with paired clean: '+sid)
            if ('parameter_seed' in r and
                    r['parameter_seed'] != base['parameter_seed']):
                raise ValueError('Parameter seed changes within image/repeat: '+sid)
    return rows, levels, conditions, repeats, groups


def crossing(levels, values, threshold):
    """Observed brackets only; a later fall below threshold is retained."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Positive finite threshold required')
    for i, value in enumerate(values):
        if value >= threshold:
            return dict(status='bracketed', lower=levels[i-1] if i else None,
                        upper=levels[i], recrossed=any(v < threshold for v in values[i+1:]))
    return dict(status='not_observed', lower=levels[-1], upper=None, recrossed=False)


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def replay_check(rows, reference, tolerance):
    """Compare all common cells, including clean, before interpreting new repeats."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Replay tolerance must be finite and nonnegative')
    old, old_levels, old_conditions, old_repeats, old_groups = read_rows(reference)
    def key(r):
        return r['id'], r['repeat'], r['condition'], r['severity']
    lookup = {key(r): r for r in old}
    candidate_ids = {r['id'] for r in rows}
    if candidate_ids - old_groups.keys():
        raise ValueError('Replay candidate contains IDs outside reference')
    matches = [r for r in rows if key(r) in lookup]
    if not any(r['condition'] != 'clean' for r in matches):
        raise ValueError('No overlapping degraded observations for replay')
    differences = []
    max_error = 0.
    for r in matches:
        prev = lookup[key(r)]
        error = max(abs(r[f]-prev[f]) for f in ('gt_ordinal_error', 'delta_gt_ordinal_error'))
        max_error = max(max_error, error)
        metadata_changed = [f for f in ('group', 'parameter_seed', 'gt_ordinal_pairs')
                            if f in r and f in prev and r[f] != prev[f]]
        if error > tolerance or metadata_changed:
            differences.append(dict(id=r['id'], repeat=r['repeat'], condition=r['condition'],
                                    severity=r['severity'], max_abs_error=error,
                                    metadata_changed=';'.join(metadata_changed)))
    return dict(status='pass' if not differences else 'mismatch', compared_rows=len(matches),
                max_abs_error=max_error, tolerance=tolerance,
                reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
                scope='Overlapping output metrics only; not a weights/input hash attestation'), differences


def audit(csv_path, out, thresholds=(.05, .10, .15), select_per_group=4,
          selection_seed=42, reference=None, replay_tolerance=1e-6, min_gap_seconds=1.):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty/new output directory')
    if select_per_group < 0 or not thresholds or any(not math.isfinite(t) or t <= 0 for t in thresholds):
        raise ValueError('Nonnegative selection size and finite positive thresholds required')
    if not math.isfinite(min_gap_seconds) or min_gap_seconds < 0:
        raise ValueError('Nonnegative finite min-gap-seconds required')
    thresholds = sorted(set(thresholds))
    rows, positive, conditions, repeats, image_groups = read_rows(csv_path)
    levels = [0.] + positive
    groups = sorted(set(image_groups.values()))
    # One value per image/repeat/condition/dose. Clean is paired, never pooled.
    raw = {(r['id'], r['repeat'], r['condition'], r['severity']): r['delta_gt_ordinal_error']
           for r in rows}
    per_image = {}
    individual = []
    for sid in sorted(image_groups):
        for c in FOCUS:
            curves = [[raw[(sid, rep, 'clean', 0.)]] + [raw[(sid, rep, c, s)] for s in positive]
                      for rep in repeats]
            averaged = [mean(v) for v in zip(*curves)]
            per_image[(sid, c)] = averaged
            for threshold in thresholds:
                for rep, curve in list(zip(repeats, curves)) + [('mean', averaged)]:
                    individual.append(dict(id=sid, group=image_groups[sid], condition=c,
                                           repeat=rep, threshold=threshold,
                                           **crossing(levels, curve, threshold)))
    aggregate = []
    group_curves = {}
    for c in FOCUS:
        for g in groups:
            group_curves[(g, c)] = [mean(v) for v in zip(*[
                per_image[(sid, c)] for sid in image_groups if image_groups[sid] == g])]
        scenarios = [('all_groups', 'all', groups)]
        scenarios += [('one_group', g, [g]) for g in groups]
        if len(groups) > 1:
            scenarios += [('leave_one_group_out', g, [h for h in groups if h != g]) for g in groups]
        for scope, label, kept in scenarios:
            curve = [mean(v) for v in zip(*[group_curves[(g, c)] for g in kept])]
            for threshold in thresholds:
                aggregate.append(dict(scope=scope, group=label, condition=c,
                                      repeat='mean', threshold=threshold, groups=len(kept),
                                      **crossing(levels, curve, threshold)))
        for rep in repeats:
            curve = [0.] + [mean([mean([raw[(sid, rep, c, s)] for sid in image_groups
                                       if image_groups[sid] == g]) for g in groups]) for s in positive]
            for threshold in thresholds:
                aggregate.append(dict(scope='one_repeat', group='all', condition=c,
                                      repeat=rep, threshold=threshold, groups=len(groups),
                                      **crossing(levels, curve, threshold)))
    # Select by fixed exploratory .10 rule, independent of sensitivity thresholds.
    profiles = []
    for sid in sorted(image_groups):
        ln = crossing(levels, per_image[(sid, 'LN')], .1)
        lng = crossing(levels, per_image[(sid, 'LNG')], .1)
        signatures = []
        for c in FOCUS:
            signs = []
            for rep in repeats:
                b = crossing(levels, [0.] + [raw[(sid, rep, c, s)] for s in positive], .1)
                signs.append((b['status'], b['upper'], b['recrossed']))
            signatures.append(len(set(signs)) > 1)
        profiles.append(dict(id=sid, group=image_groups[sid],
                             LN_recross=ln['recrossed'], repeat_disagreement=any(signatures),
                             LNG_sustained=lng['status'] == 'bracketed' and not lng['recrossed'],
                             LNG_not_observed=lng['status'] == 'not_observed',
                             LN_upper=ln['upper'], LNG_upper=lng['upper']))
    chosen = []
    coverage = []
    def timestamp(sid):
        found = re.findall(r'(?<!\d)(\d{16})(?!\d)', Path(sid).stem)
        return int(found[-1])/1e6 if found else None

    def spaced(candidate):
        stamp = timestamp(candidate['id'])
        if stamp is None:
            return True
        return all(Path(candidate['id']).parent != Path(other['id']).parent or
                   timestamp(other['id']) is None or
                   abs(stamp-timestamp(other['id'])) >= min_gap_seconds for other in chosen)

    for group in groups:
        candidates = [r for r in profiles if r['group'] == group]
        ranked = sorted(candidates, key=lambda r: hashlib.sha256(
            ('%s|%s' % (selection_seed, r['id'])).encode()).hexdigest())
        picked = set()
        # Reserve scarce non-crossing controls first, then avoid adjacent frames.
        for category in ('LNG_not_observed', 'LN_recross', 'repeat_disagreement', 'LNG_sustained'):
            eligible = [r for r in ranked if r[category] and r['id'] not in picked and spaced(r)]
            coverage.append(dict(group=group, category=category,
                                 available=sum(bool(r[category]) for r in candidates),
                                 eligible_at_selection=len(eligible)))
            if len(picked) < select_per_group and eligible:
                r = eligible[0]
                chosen.append(dict(**r, selected_for=category))
                picked.add(r['id'])
        for r in ranked:
            if len(picked) >= select_per_group:
                break
            if r['id'] not in picked and spaced(r):
                chosen.append(dict(**r, selected_for='group_coverage'))
                picked.add(r['id'])
    replay, differences = (replay_check(rows, reference, replay_tolerance)
                           if reference else (None, []))
    report = dict(schema=1, source=str(Path(csv_path).resolve()),
                  source_sha256=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                  rows=len(rows), images=len(image_groups), groups=len(groups),
                  group_sizes={g: sum(v == g for v in image_groups.values()) for g in groups},
                  repeats=repeats, levels=levels, conditions=conditions, thresholds=thresholds,
                  selection_threshold=.1, selection_seed=selection_seed,
                  min_gap_seconds=min_gap_seconds,
                  timestamped_images=sum(timestamp(sid) is not None for sid in image_groups),
                  selected=len(chosen), replay=replay,
                  limitations=['Outcome-selected exploratory pilot; not population inference.',
                               'Time groups are not guaranteed independent scenes.',
                               'Repeat varies flare geometry/template AND noise; not noise-only.',
                               'CSV consistency checks do not verify GT projection or run provenance.'])
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'boundary_sensitivity.csv', aggregate)
    write_csv(out/'per_image_boundaries.csv', individual)
    write_csv(out/'case_profiles.csv', profiles)
    write_csv(out/'selection_coverage.csv', coverage)
    write_csv(out/'selected_cases.csv', chosen)
    write_csv(out/'replay_differences.csv', differences)
    if chosen:
        (out/'samples.txt').write_text('\n'.join(r['id'] for r in chosen)+'\n', encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# 过渡区复核', '',
             '%s 张基图，%s 个组，%s 个重复；选出 %s 张探索样本。' %
             (len(image_groups), len(groups), len(repeats), len(chosen)), '',
             '## 实测均值首次越阈（无拟合）', '',
             '| 条件 | 增量阈值 | 下界 | 上界 | 后续回落 |',
             '|---|---:|---:|---:|---|']
    for r in aggregate:
        if r['scope'] == 'all_groups':
            lines.append('| {condition} | {threshold:.0%} | {lower} | {upper} | {recrossed} |'.format(**r))
    lines += ['', '阈值 10% 指错误率增加 10 个百分点。None 表示上界未观测到，不能填 S4。',
              '逐组、逐重复和留一组结果见 boundary_sensitivity.csv。', '',
              '## 解释范围', '',
              '清单按 .10 阈值的回落、重复不一致、持续越阈、未越阈分层选择；',
              '优先保留未越阈对照，同组同类按固定哈希顺序取样；可解析时间戳时，同序列至少间隔 %s 秒。' % min_gap_seconds,
              '缺类如实记录后按组补足；间隔不满足则少选，不强行补数。不能用此小样本估计总体失效率。',
              '不同重复同时改变眩光模板/位置和噪声，不是纯噪声重复。',
              '新增样本复测前必须核对原 run.yaml 的 seed、profile、权重、GT 范围和素材库。']
    if replay:
        lines += ['', '重叠结果复现检查：'+json.dumps(replay, ensure_ascii=False),
                  '若 mismatch，先核对数据/权重/参数/软件环境，不解释新增曲线。']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report
