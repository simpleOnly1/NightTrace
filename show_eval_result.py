"""STEPS-style seven-column terminal report for NightTrace metrics.json."""
import argparse
import json
from pathlib import Path
from evaluation.metrics import METRIC_NAMES


def format_table(metrics):
    width = 12
    line = '+' + '+'.join('-' * width for _ in METRIC_NAMES) + '+'
    head = '|' + '|'.join(('%s' % k).center(width) for k in METRIC_NAMES) + '|'
    vals = '|' + '|'.join(('%0.3f' % metrics[k]).center(width) for k in METRIC_NAMES) + '|'
    return '\n'.join([line, head, line, vals, line])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--worst', type=int, default=0)
    a = p.parse_args()
    report = json.loads(Path(a.file).read_text())
    summary = report.get('summary', report)
    print(format_table(summary))
    print('images:', summary.get('num_images', '?'),
          'scale alignment:', summary.get('alignment', '?'))
    if a.worst and 'per_image' in report:
        for item in sorted(report['per_image'], key=lambda x: x['abs_rel'], reverse=True)[:a.worst]:
            print('high AbsRel:', item['id'], '%.5f' % item['abs_rel'])


if __name__ == '__main__':
    main()
