"""Reproducible input dose audit and optional independent-GT failure sweep.

Start with --render-only. Default: 1 clean + 7 conditions x 4 doses = 29
forward passes per image/repeat (clean inference is cached across repeats).
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from datasets.common import rgb, original_size
from analysis.sampling import select_samples, save_selection, stable_seed
from analysis.degradation_v2.pipeline import CONDITIONS, list_flare_templates
from analysis.degradation_v2.severity import (DEFAULT_PROFILE, DEFAULT_GEOMETRY,
    VERSION, validate_profile, sample_plan, at_severity, render, audit_input)
from analysis.failure_boundary import gt_pairs, evaluate_gt
from analysis.relations import affine_error, ordinal_error
from evaluation.protocol import prepare_gt
from utils.config import load_config, project_path
from analysis.sweep_store import atomic_json, initialize, read_checkpoint, checkpoint_path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluation_candidates(cfg):
    """Use ONLY the official daytime test split, with recording/block proxies.

    nuScenes test filenames identify recordings, not scene tokens; group by
    recording conservatively rather than pretend every frame is independent.
    """
    from datasets import make_dataset
    ds = make_dataset(cfg, 'day', training=False)
    d = cfg['data']; root = project_path(d['root'])
    block = cfg.get('diagnosis', {}).get('block_seconds', 30)
    if block <= 0:
        raise ValueError('block_seconds must be positive')
    rows = []; ranks = defaultdict(int)
    for rec in ds.records:
        path = Path(rec['target']); sid = path.relative_to(root).as_posix()
        stamps = re.findall(r'(?<!\d)(\d{16})(?!\d)', path.stem)
        stamp = int(stamps[-1])/1e6 if stamps else None
        seq = (path.parent.relative_to(root).as_posix() if d['name'] == 'robotcar'
               else path.stem.split('__CAM_')[0] if '__CAM_' in path.stem else 'unknown_recording')
        group = seq
        if d['name'] == 'robotcar' and stamp is not None:
            group += f'/block_{int(stamp//block)}'
        rows.append(dict(path=path, id=sid, sequence=seq, group=group,
                         timestamp=stamp, frame_rank=ranks[seq], gt_path=str(rec['gt'])))
        ranks[seq] += 1
    return rows


def load_gt(sample, cfg):
    d = cfg['data']; h, w = original_size(sample['path'])
    cw, ch = (d.get('crop_width', 1152), d.get('crop_height', 640)) if d['name'] == 'robotcar' else (w, h)
    geometry = dict(raw_hw=[h, w], crop_xywh=[(w-cw)//2, (h-ch)//2, cw, ch],
                    input_hw=[d['input_height'], d['input_width']], gt_space=d.get('gt_space', 'raw'))
    return prepare_gt(np.load(sample['gt_path'], allow_pickle=False), geometry), geometry


def run(args, net=None):
    if args.repeats < 1 or args.num_vis < 0:
        raise ValueError('Positive repeats and nonnegative num-vis required')
    if args.levels is not None and args.level_step is not None:
        raise ValueError('Choose --levels or --level-step, not both')
    if args.level_step is not None:
        if not np.isfinite(args.level_step) or not .001 <= args.level_step <= 4:
            raise ValueError('level-step must be finite in [.001,4]')
        levels = sorted(set([round(float(s), 8) for s in np.arange(args.level_step, 4, args.level_step)] + [4.]))
    else:
        levels = sorted(args.levels if args.levels is not None else [1., 2., 3., 4.])
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Require 0 <= shard-index < num-shards')
    if args.resume and args.prepare_only:
        raise ValueError('prepare-only outputs are selection manifests, not resumable inference runs')
    if len(set(levels)) != len(levels) or not levels or not all(np.isfinite(s) and 0 < s <= 4 for s in levels):
        raise ValueError('Use distinct levels in (0,4]; S0 clean is automatic')
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError('Duplicate conditions')
    if not np.isfinite(args.failure_delta) or args.failure_delta <= 0:
        raise ValueError('failure-delta must be positive and finite')
    if args.pairs < 1 or args.min_pairs < 1 or args.min_pairs > args.pairs or not np.isfinite(args.pair_gap) or args.pair_gap <= 0:
        raise ValueError('Invalid GT pair policy')
    cfg = load_config(args.config); d = cfg['data']; m = cfg['model']
    settings = yaml.safe_load(Path(args.profile).read_text(encoding='utf-8')) if args.profile else {}
    settings = settings or {}
    if set(settings) - {'profile', 'geometry'}:
        raise ValueError('Severity YAML permits only profile and geometry')
    profile = validate_profile(settings.get('profile', DEFAULT_PROFILE))
    geometry_ranges = settings.get('geometry', DEFAULT_GEOMETRY)
    # Validate geometry even in prepare-only mode.
    sample_plan(args.seed, geometry=geometry_ranges)
    out = project_path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Output directory is not empty; choose a new run directory')
    candidate_rows = evaluation_candidates(cfg) if args.source == 'day-test' else None
    if args.all_split:
        if args.source != 'day-test' or args.sample_list or args.limit != 0:
            raise ValueError('--all-split requires day-test, --limit 0 and no sample-list')
        selected = candidate_rows
        audit = [dict(group=g, candidates=sum(r['group'] == g for r in selected),
                      selected=sum(r['group'] == g for r in selected), time_rejected=0, duplicate_rejected=0)
                 for g in sorted({r['group'] for r in selected})]
    else:
        selected, audit = select_samples(cfg, args.limit, args.seed, args.sample_list, candidate_rows)
    if not selected or len({s['id'] for s in selected}) != len(selected):
        raise ValueError('Selection must contain nonempty, unique image IDs')
    if args.groups_csv:
        import csv
        with Path(args.groups_csv).open(encoding='utf-8-sig', newline='') as f:
            groups = list(csv.DictReader(f))
        mapping = {r['id']: r['group'] for r in groups}
        if len(mapping) != len(groups) or any(not mapping.get(r['id']) for r in selected):
            raise ValueError('groups-csv needs unique IDs and a nonempty group for every selected image')
        for r in selected:
            r['group'] = mapping[r['id']]
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_selection(out, selected, audit)
    all_selected = selected
    selected = [s for i, s in enumerate(all_selected) if i % args.num_shards == args.shard_index]
    ev = cfg['eval']; minimum, maximum = ev['min_depth'], ev['max_depth']
    gt_audit = []
    if args.source == 'day-test':
        # Fail before costly inference if a selected GT file/FOV is invalid.
        for sample in selected:
            gt, geometry = load_gt(sample, cfg)
            valid = np.isfinite(gt) & (gt > minimum) & (gt < maximum)
            if not valid.any():
                raise ValueError('No evaluable GT for '+sample['id'])
            gt_audit.append(dict(id=sample['id'], gt_path=sample['gt_path'],
                                 valid_pixels=int(valid.sum()), **geometry))
        if not args.resume:
            (out/'gt_geometry.json').write_text(json.dumps(gt_audit, indent=2), encoding='utf-8')
    flare_dir = args.flare_dir or cfg.get('diagnosis', {}).get('flare_dir')
    templates = [] if args.prepare_only or args.analytic_flare or not any('G' in c for c in args.conditions) else list_flare_templates(project_path(flare_dir) if flare_dir else None)
    metadata = dict(version=VERSION, config=cfg, arguments=vars(args), profile=profile,
        geometry=geometry_ranges, levels=levels, model_frozen=True,
        baseline='S0 unmodified RGB', gt_reference=(args.source == 'day-test'),
        ordering='global valid-GT pairs; no semantic boundary/contact annotations',
        grouping='explicit group map' if args.groups_csv else 'recording/time-block proxy, not guaranteed independent scenes',
        flare_backend='Flare7K++ templates' if templates else 'analytic (or unused)',
        flare_template_count=len(templates),
        noise_coupling='same seeds; separate read stream; Poisson marginals, not matched photon events',
        software=dict(torch=str(torch.__version__), numpy=str(np.__version__)),
        status='prepared' if args.prepare_only else 'running')
    if not args.resume:
        (out/'run.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    if args.prepare_only:
        print('Selected', len(selected), 'images into', out); return []
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.render_only:
        net = None
    if net is None and not args.render_only:
        from models.depth_model import DepthModel
        net = DepthModel(str(project_path(m['weights'])), input_size=m.get('input_size', 518),
            freeze_encoder=True, train_dpt=False, inverse_epsilon=m.get('inverse_epsilon', 1e-3))
    if net is not None:
        net = net.to(device).eval(); net.requires_grad_(False)
    crop = (d.get('crop_width', 1152), d.get('crop_height', 640)) if d['name'] == 'robotcar' else None
    # Hash the actual inputs and implementation, not only their path names.
    metadata['code_sha256'] = {p: file_sha256(Path(__file__).parent/p) for p in (
        'diagnose_severity.py', 'analysis/degradation_v2/pipeline.py',
        'analysis/degradation_v2/severity.py', 'analysis/failure_boundary.py',
        'analysis/sweep_store.py', 'analysis/sampling.py', 'datasets/common.py',
        'datasets/robotcar.py', 'datasets/nuscenes.py',
        'models/depth_net.py', 'evaluation/protocol.py')}
    weights = project_path(m['weights'])
    metadata['weights_sha256'] = file_sha256(weights) if net is not None and weights.is_file() else None
    contract = dict(schema=2, config=cfg, profile=profile, geometry=geometry_ranges,
        samples=[{k: str(v) if isinstance(v, Path) else v for k, v in s.items()} for s in all_selected],
        source=args.source, seed=args.seed, levels=levels, conditions=args.conditions,
        repeats=args.repeats, pairs=args.pairs, pair_gap=args.pair_gap, min_pairs=args.min_pairs,
        failure_delta=args.failure_delta, render_only=args.render_only, device=device,
        software=metadata['software'], code_sha256=metadata['code_sha256'], weights_sha256=metadata['weights_sha256'],
        template_bank_sha256=hashlib.sha256('\n'.join(templates).encode()).hexdigest(),
        num_shards=args.num_shards, shard_index=args.shard_index)
    initialize(out, contract, args.resume)
    (out/'run.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    with torch.no_grad():
        for index, sample in enumerate(selected):
            input_hash = dict(id=sample['id'], rgb_sha256=file_sha256(sample['path']),
                gt_sha256=file_sha256(sample['gt_path']) if args.source == 'day-test' else None)
            previous = read_checkpoint(out, sample['id']) if args.resume else None
            if previous is not None:
                if previous['input_hash'] != input_hash:
                    raise ValueError('Input changed since checkpoint: '+sample['id'])
                for path, digest in previous['template_hashes'].items():
                    if file_sha256(path) != digest:
                        raise ValueError('Flare template changed since checkpoint: '+path)
                print(f'[{index+1}/{len(selected)}] verified; skipped {sample["id"]}', flush=True)
                continue
            rows = []; parameters = []; hashes = {}; warnings = []; parameter_records = []
            clean, _ = rgb(sample['path'], (d['input_height'], d['input_width']), crop=crop)
            clean = clean[None].to(device)
            qc = net(clean)['raw_relative'] if net is not None else None
            gt = pairs = None
            if args.source == 'day-test' and net is not None:
                gt, _ = load_gt(sample, cfg)
                pairs = gt_pairs(gt, minimum, maximum, stable_seed(args.seed, sample['id']), args.pairs, args.pair_gap)
            base_gt = {}
            if gt is not None:
                base_gt = evaluate_gt(qc.clamp_min(m.get('inverse_epsilon', 1e-3)).reciprocal().cpu().numpy()[0, 0],
                                      gt, pairs, minimum, maximum, args.min_pairs)
            for repeat in range(args.repeats):
                seed = stable_seed(args.seed, sample['id'], repeat)
                plan = sample_plan(seed, templates, geometry_ranges)
                for source in plan['sources']:
                    for key in ('template', 'reflective_template'):
                        if key in source and source[key] not in hashes:
                            hashes[source[key]] = hashlib.sha256(Path(source[key]).read_bytes()).hexdigest()
                frames = {}; predictions = {}; vis_rows = {}
                schedule = [('clean', 0.)] + [(c, s) for c in args.conditions for s in levels]
                for cond, severity in schedule:
                    params = at_severity(plan, severity, profile)
                    dark, flare = render(clean, cond, params)
                    row = dict(id=sample['id'], group=sample['group'], repeat=repeat, condition=cond,
                               severity=severity, parameter_seed=seed, **audit_input(clean, dark, cond, params, flare))
                    # Serialize one parameter record per dose, not duplicate per condition.
                    if cond in ('clean', args.conditions[0]):
                        record = dict(id=sample['id'], repeat=repeat, **params)
                        parameter_records.append(record)
                        for i, source in enumerate(params['sources']):
                            parameters.append(dict(id=sample['id'], repeat=repeat, severity=severity,
                                parameter_seed=seed, exposure=params['exposure'], shot_peak=params['shot_peak'],
                                read_sigma=params['read_sigma'], glare_gain=params['glare_gain'], source=i, **source))
                    if net is not None:
                        q = qc if cond == 'clean' else net(dark)['raw_relative']
                        if not torch.isfinite(q).all() or (q < 0).any():
                            raise ValueError('Invalid model inverse depth for '+sample['id'])
                        scale = (qc-qc.median()).abs().median().clamp_min(1e-6)
                        row.update(depth_drift=affine_error(q, qc),
                            raw_q_mae=float((q-qc).abs().mean()),
                            fixed_scale_q_mae=float((q-qc).abs().mean()/scale),
                            clean_ordinal_disagreement=ordinal_error(q, qc, stable_seed(args.seed, sample['id'])))
                        if gt is not None:
                            scores = base_gt if cond == 'clean' else evaluate_gt(
                                q.clamp_min(m.get('inverse_epsilon', 1e-3)).reciprocal().cpu().numpy()[0, 0],
                                gt, pairs, minimum, maximum, args.min_pairs)
                            row.update(scores)
                            for key in ('gt_ordinal_error', 'gt_abs_rel', 'gt_a1'):
                                row['delta_'+key] = scores[key]-base_gt[key]
                        if args.save_vis and index < args.num_vis and repeat == 0:
                            predictions[(cond, severity)] = q.detach().cpu()
                    rows.append(row)
                    if severity == max(levels):
                        flags = []
                        if 'G' in cond and row['glare_visible_area'] < .01:
                            flags.append('glare_affects_less_than_1pct')
                        if cond == 'L' and row['luma_p50'] > .1:
                            flags.append('low_light_median_above_0.1')
                        if 'N' in cond and row['noise_rgb_rms'] < .02:
                            flags.append('noise_rgb_rms_below_0.02')
                        if row['saturated_area'] > .95 or row['black_area'] > .95:
                            flags.append('over_95pct_clipped_or_black')
                        for flag in flags:
                            warnings.append(dict(id=sample['id'], repeat=repeat, condition=cond,
                                severity=severity, flag=flag, action='inspect; retained without resampling'))
                    if args.save_vis and index < args.num_vis and repeat == 0:
                        frames[(cond, severity)] = dark.detach().cpu(); vis_rows[(cond, severity)] = row
                if frames:
                    from visualization.severity import save_sweep_panels
                    save_sweep_panels(out/'qualitative'/str(seed), frames, predictions, vis_rows,
                                      args.conditions, levels, sample['id'])
            atomic_json(checkpoint_path(out, sample['id']), dict(id=sample['id'], complete=True,
                rows=rows, parameters=parameters, parameter_records=parameter_records,
                input_hash=input_hash, template_hashes=hashes, warnings=warnings))
            print(f'[{index+1}/{len(selected)}] {sample["id"]}', flush=True)
    metadata['status'] = 'inference_complete'
    (out/'run.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    if not args.defer_summary and args.num_shards == 1:
        from analysis.sweep_summary import summarize_runs
        summarize_runs([out], out, allow_existing=True, curves=False)
        metadata['status'] = 'complete'
        (out/'run.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
        # Preserve the small-run/test API; large runs must use --defer-summary.
        return [r for s in selected for r in read_checkpoint(out, s['id'])['rows']]
    print('Checkpoints complete; summarize all shards with tools/summarize_sweep.py')
    return []


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--source', choices=['day-train', 'day-test'], default='day-train', help='day-test enables independent GT')
    p.add_argument('--profile', help='YAML with profile (S0..S4) and optional geometry ranges')
    p.add_argument('--levels', nargs='+', type=float, help='Explicit dose points; default 1 2 3 4')
    p.add_argument('--level-step', type=float, help='Dense grid: .25 gives 16 nonzero doses up to S4')
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS[1:], default=list(CONDITIONS[1:]))
    p.add_argument('--limit', type=int, default=16)
    p.add_argument('--all-split', action='store_true', help='Full unfiltered day-test; requires --limit 0')
    p.add_argument('--resume', action='store_true', help='Verify contract and hashes, skip completed images')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--defer-summary', action='store_true', help='Bound inference memory; merge checkpoints separately')
    p.add_argument('--sample-list', help='Exact IDs within the chosen source split; not truncated by limit')
    p.add_argument('--groups-csv', help='Optional id,group mapping; apply true scene groups to statistics')
    p.add_argument('--out', default='runs/severity')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--flare-dir')
    p.add_argument('--analytic-flare', action='store_true', help='Explicit analytic backend, ignores configured asset bank')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--render-only', action='store_true', help='Input audit without weights/inference; no failure claims')
    p.add_argument('--save-vis', action='store_true')
    p.add_argument('--num-vis', type=int, default=4, help='First selected base images, first repeat only')
    p.add_argument('--pairs', type=int, default=8192)
    p.add_argument('--min-pairs', type=int, default=128)
    p.add_argument('--pair-gap', type=float, default=.1)
    p.add_argument('--failure-delta', type=float, default=.1, help='GT ordering error increase; .1 = 10 percentage points, exploratory default')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
