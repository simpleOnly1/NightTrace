"""Merge complete severity checkpoints and fit descriptive GT response curves."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from analysis.sweep_summary import summarize_runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='Every shard of one experiment, or one unsharded run')
    p.add_argument('--out', required=True, help='New analysis directory')
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--penalty', type=float, default=.1, help='Fixed descriptive smoothing penalty; do not tune to desired boundary')
    p.add_argument('--no-fit', action='store_true', help='Measured points/statistics only')
    a = p.parse_args()
    if a.bootstrap < 20:
        p.error('--bootstrap must be >=20')
    summarize_runs(a.runs, Path(a.out), curves=not a.no_fit, n_boot=a.bootstrap, penalty=a.penalty)


if __name__ == '__main__':
    main()
