"""CPU-only post-hoc LN/LNG audit and small exploratory sample selection."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.transition_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='Existing analysis/per_image.csv')
    parser.add_argument('--out', required=True, help='New output directory')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[.05, .1, .15])
    parser.add_argument('--select-per-group', type=int, default=4, help='0 disables sample list')
    parser.add_argument('--selection-seed', type=int, default=42, help='Selection only, not synthesis seed')
    parser.add_argument('--reference', help='Original per_image.csv to compare overlapping observations')
    parser.add_argument('--replay-tolerance', type=float, default=1e-6)
    parser.add_argument('--min-gap-seconds', type=float, default=1., help='Spacing when IDs contain microsecond timestamps')
    args = parser.parse_args()
    try:
        result = audit(args.csv, args.out, args.thresholds, args.select_per_group,
                       args.selection_seed, args.reference, args.replay_tolerance, args.min_gap_seconds)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['replay'] and result['replay']['status'] != 'pass':
        parser.exit(1, 'Replay mismatch; inspect replay_differences.csv before interpreting new doses.\n')


if __name__ == '__main__':
    main()
