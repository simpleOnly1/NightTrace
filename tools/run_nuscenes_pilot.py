"""Run the DAV2/nuScenes entry checks in small, explicit stages (Python 3.8+)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ['L', 'N', 'G', 'LN', 'LG', 'NG', 'LNG']


def project_path(value, repo=REPO):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def validate_config(cfg):
    if cfg.get('data', {}).get('name') != 'nuscenes':
        raise ValueError('This pilot requires data.name: nuscenes')
    if cfg.get('model', {}).get('name') != 'dav2_vitl':
        raise ValueError('This pilot only implements DAV2-L; other models need adapters')


def require_audit(base):
    path = base / 'gt_audit' / 'summary.json'
    if not path.is_file():
        raise ValueError('Run --stage audit first: missing ' + str(path))
    result = json.loads(path.read_text(encoding='utf-8'))
    checked, available = result.get('checked', 0), result.get('available', 0)
    if result.get('dataset') != 'nuscenes' or checked <= 0 or checked != available:
        raise ValueError('Need a complete, nonempty nuScenes GT audit before the pilot')
    return result


def build_commands(stage, config, base, limit=8, resume=False, analysis_out=None,
                   repo=REPO, python=None):
    if stage not in ('audit', 'pilot', 'summarize'):
        raise ValueError('Unknown stage: ' + stage)
    if not 1 <= limit <= 16:
        raise ValueError('Pilot limit must be 1..16; full sweeps use diagnose_severity.py')
    if resume and stage != 'pilot':
        raise ValueError('--resume is only used with --stage pilot')
    python = python or sys.executable
    summary = [python, str(repo / 'tools/summarize_sweep.py'), '--runs',
               str(base / 'sweep'), '--no-fit', '--bootstrap', '1000', '--out',
               str(analysis_out or base / 'analysis')]
    if stage == 'audit':
        return [('gt_audit', [python, str(repo / 'tools/check_eval_data.py'),
                '--config', str(config), '--split', 'day', '--overlays', '8',
                '--out', str(base / 'gt_audit')])]
    if stage == 'summarize':
        return [('summarize', summary)]
    inference = [python, str(repo / 'diagnose_severity.py'), '--config', str(config),
                 '--source', 'day-test', '--profile', str(repo / 'config/severity.yaml'),
                 '--limit', str(limit), '--conditions'] + CONDITIONS + [
                 '--levels', '1', '2', '3', '4', '--repeats', '1', '--seed', '42',
                 '--pairs', '8192', '--min-pairs', '128', '--pair-gap', '0.1',
                 '--failure-delta', '0.10', '--save-vis', '--num-vis', '2',
                 '--defer-summary', '--out', str(base / 'sweep')]
    if resume:
        inference.append('--resume')
    return [('inference', inference), ('summarize', summary)]


def execute(commands, repo, status_path, runner=None):
    runner = runner or subprocess.run
    state = {'steps': [], 'status': 'running'}
    status_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        status_path.write_text(json.dumps(state, indent=2), encoding='utf-8')

    save()
    for name, command in commands:
        step = {'name': name, 'command': command, 'status': 'running'}
        state['steps'].append(step)
        save()
        print('\nRunning ' + name, flush=True)
        try:
            runner(command, cwd=str(repo), check=True)
        except (subprocess.CalledProcessError, OSError, KeyboardInterrupt) as exc:
            step.update(status='failed', error=str(exc))
            state['status'] = 'failed'
            save()
            raise
        step['status'] = 'completed'
        save()
    state['status'] = 'completed'
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['audit', 'pilot', 'summarize'])
    parser.add_argument('--config', default='config/nuscenes.yaml')
    parser.add_argument('--out', default='runs/ns_dav2_pilot')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--analysis-out', help='Fresh analysis directory for re-analysis')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running/writing')
    args = parser.parse_args(argv)
    config = project_path(args.config)
    base = project_path(args.out)
    analysis_out = project_path(args.analysis_out) if args.analysis_out else base / 'analysis'
    try:
        if not config.is_file():
            raise ValueError('Missing config: ' + str(config))
        import yaml
        cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
        if not isinstance(cfg, dict):
            raise ValueError('Config must be a YAML mapping')
        validate_config(cfg)
        commands = build_commands(args.stage, config, base, args.limit, args.resume, analysis_out)
        print(json.dumps({'stage': args.stage,
                          'maximum_forwards': args.limit * 29 if args.stage == 'pilot' else 0,
                          'commands': commands}, indent=2), flush=True)
        if args.dry_run:
            return 0
        if args.stage == 'pilot':
            require_audit(base)
            print('Audit checked shapes/FOV. Review GT overlays; this is not a calibration attestation.', flush=True)
        if args.stage in ('pilot', 'summarize') and analysis_out.exists() and any(analysis_out.iterdir()):
            raise ValueError('Analysis directory is not empty; use a fresh --analysis-out')
        execute(commands, REPO, base / (args.stage + '_status.json'))
        if args.stage == 'audit':
            print('Next: inspect gt_audit/*aligned_sparse_GT.png, then run --stage pilot.', flush=True)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError, ImportError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
