"""Check the existing environment; optionally install extras into .deps only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unidepth_env import (activate_extras, check_base_runtime,
                                 extras_path, protected_versions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true',
                        help='Install pinned extras in this project, with --no-deps; never update torch/numpy')
    parser.add_argument('--out', help='Save the environment check JSON')
    args = parser.parse_args()
    before = protected_versions()
    report = dict(python=sys.version, executable=sys.executable,
                  protected_before=before, extras_directory=str(extras_path()))
    try:
        check_base_runtime()
        if args.install:
            command = [sys.executable, '-m', 'pip', 'install', '--no-deps',
                       '--only-binary=:all:', '--upgrade', '--target', str(extras_path()),
                       '-r', str(ROOT/'requirements-unidepth.txt')]
            subprocess.run(command, check=True)
        activate_extras()
        package_root = ROOT / 'models'
        sys.path.insert(0, str(package_root))
        from unidepth.models import UniDepthV2
        import unidepth
        if {Path(p).resolve() for p in unidepth.__path__} != {(package_root/'unidepth').resolve()}:
            raise RuntimeError('A different UniDepth package was imported')
        report.update(ok=True, model_class=UniDepthV2.__name__,
                      protected_after=protected_versions())
        if before != report['protected_after']:
            raise RuntimeError('Protected package identity changed; check project extras')
    except Exception as error:
        report.update(ok=False, error=type(error).__name__+': '+str(error))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
