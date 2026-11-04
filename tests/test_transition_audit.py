import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.transition_audit import audit, crossing, read_rows


def fixture():
    rows = []
    # Unequal group sizes must NOT turn into an equal-image group estimate.
    for sid, group, peak in [('a', 'g1', .20), ('b', 'g2', .04), ('c', 'g2', .04)]:
        for repeat in (0, 1):
            rows.append(dict(id=sid, group=group, repeat=repeat, condition='clean', severity=0,
                             gt_ordinal_error=.1, delta_gt_ordinal_error=0, parameter_seed=repeat))
            for condition in ('LN', 'LNG'):
                for level in (1, 2, 3):
                    delta = peak if level == 2 else 0 if level == 1 else peak/4
                    if condition == 'LNG' and level == 3:
                        delta = peak
                    rows.append(dict(id=sid, group=group, repeat=repeat, condition=condition,
                                     severity=level, gt_ordinal_error=.1+delta,
                                     delta_gt_ordinal_error=delta, parameter_seed=repeat))
    return rows


def save(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TransitionAuditTests(unittest.TestCase):
    def test_brackets_and_censoring(self):
        self.assertEqual(crossing([0, 1, 2, 3], [0, .02, .12, .04], .1),
                         dict(status='bracketed', lower=1, upper=2, recrossed=True))
        self.assertIsNone(crossing([0, 1], [0, .01], .1)['upper'])

    def test_group_weighting_selection_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); save(root/'in.csv', fixture())
            result = audit(root/'in.csv', root/'out', select_per_group=2)
            self.assertEqual(result['images'], 3)
            self.assertEqual(result['selected'], 3)  # no duplicate padding
            with (root/'out/boundary_sensitivity.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            b = next(r for r in rows if r['scope'] == 'all_groups' and
                     r['condition'] == 'LN' and float(r['threshold']) == .1)
            self.assertEqual(b['upper'], '2.0')  # group mean .12, image mean .0933
            self.assertEqual(b['recrossed'], 'True')
            audit(root/'in.csv', root/'again', select_per_group=2)
            self.assertEqual((root/'out/samples.txt').read_bytes(), (root/'again/samples.txt').read_bytes())
            with self.assertRaisesRegex(ValueError, 'empty/new'):
                audit(root/'in.csv', root/'out')

    def test_bad_grids_and_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'in.csv'
            cases = []
            rows = fixture(); cases.append(rows + [rows[0]])
            cases.append(fixture()[:-1])
            for field, value in [('delta_gt_ordinal_error', float('nan')),
                                 ('delta_gt_ordinal_error', .9), ('group', 'other'),
                                 ('parameter_seed', 999)]:
                rows = fixture(); rows[1][field] = value; cases.append(rows)
            for rows in cases:
                save(path, rows)
                with self.assertRaises(ValueError):
                    read_rows(path)

    def test_missing_stratum_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); save(root/'in.csv', fixture())
            audit(root/'in.csv', root/'out')
            with (root/'out/selection_coverage.csv').open() as stream:
                coverage = list(csv.DictReader(stream))
            self.assertTrue(any(r['category'] == 'repeat_disagreement' and r['available'] == '0'
                                for r in coverage))

    def test_replay_checks_metrics_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original = fixture(); save(root/'old.csv', original)
            save(root/'new.csv', original)
            r = audit(root/'new.csv', root/'pass', reference=root/'old.csv', select_per_group=0)
            self.assertEqual(r['replay']['status'], 'pass')
            self.assertEqual(r['replay']['compared_rows'], len(original))
            self.assertFalse((root/'pass/samples.txt').exists())
            changed = copy.deepcopy(original)
            changed[1]['delta_gt_ordinal_error'] += .001
            changed[1]['gt_ordinal_error'] += .001
            save(root/'new.csv', changed)
            r = audit(root/'new.csv', root/'fail', reference=root/'old.csv')
            self.assertEqual(r['replay']['status'], 'mismatch')
            changed = copy.deepcopy(original)
            for row in changed:
                row['parameter_seed'] += 10
            save(root/'new.csv', changed)
            r = audit(root/'new.csv', root/'metadata', reference=root/'old.csv')
            self.assertEqual(r['replay']['status'], 'mismatch')

    def test_nan_threshold_rejected(self):
        with self.assertRaises(ValueError):
            audit('unused.csv', 'unused', thresholds=[float('nan')])

    def test_temporal_spacing_does_not_pad(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = fixture()
            names = dict(a='seq/1418134530000000.png', b='seq/1418134530100000.png',
                         c='seq/1418134530200000.png')
            for row in rows:
                row['id'] = names[row['id']]; row['group'] = 'one_group'
            save(root/'in.csv', rows)
            result = audit(root/'in.csv', root/'out')
            self.assertEqual(result['selected'], 1)


if __name__ == '__main__':
    unittest.main()
