"""Real checkpoint/merge round trips with tiny tensors and a frozen fake model."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import yaml
from diagnose_severity import parser, run
from analysis.sweep_summary import summarize_runs
from analysis.sweep_curves import smoother, crossing, interval, fit_group_curves


class CountingModel(torch.nn.Module):
    def __init__(self, fail_at=None):
        super().__init__(); self.calls = 0; self.fail_at = fail_at

    def forward(self, x):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('simulated interruption')
        return {'raw_relative': x.mean(1, keepdim=True)+.01}


def fixture(root):
    package = root/'data'/'day'
    (package/'rgb').mkdir(parents=True); (package/'depth').mkdir()
    ids = ['1418133156135993', '1418133216135993']
    image = np.tile(np.linspace(20, 220, 48, dtype=np.uint8)[None, :, None], (32, 1, 3))
    for sid in ids:
        cv2.imwrite(str(package/'rgb'/(sid+'.png')), image)
        np.save(package/'depth'/(sid+'_depth.npy'), np.tile(np.linspace(2, 25, 48)[None, :], (32, 1)))
    for split in ('train', 'test'):
        (package/(split+'_split.txt')).write_text('\n'.join(ids))
    cfg = dict(data=dict(name='robotcar', root=str(root/'data'), day_package='day',
        input_height=16, input_width=24, crop_width=40, crop_height=24, gt_space='raw'),
        model=dict(weights='fake-not-needed', inverse_epsilon=.001),
        eval=dict(min_depth=.1, max_depth=40), diagnosis=dict(duplicate_mae=1, min_gap_seconds=1000))
    config = root/'config.yaml'; config.write_text(yaml.safe_dump(cfg))
    return config, package


def arguments(config, out, extra=()):
    return parser().parse_args(['--config', str(config), '--source', 'day-test', '--all-split', '--limit', '0',
        '--device', 'cpu', '--analytic-flare', '--out', str(out), '--levels', '1', '4',
        '--defer-summary', *extra])


class ExecutionTests(unittest.TestCase):
    def test_resume_skips_only_complete_images_and_checks_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, package = fixture(root); out = root/'run'
            args = arguments(config, out)
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                run(args, CountingModel(fail_at=16))
            self.assertEqual(len(list((out/'checkpoints').glob('*.json'))), 1)
            args.resume = True; model = CountingModel(); run(args, model)
            self.assertEqual(model.calls, 15)
            model = CountingModel(); run(args, model); self.assertEqual(model.calls, 0)
            args.failure_delta = .2
            with self.assertRaisesRegex(ValueError, 'contract mismatch'): run(args, CountingModel())
            args.failure_delta = .1
            image = package/'rgb'/'1418133156135993.png'
            cv2.imwrite(str(image), np.zeros((32, 48, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, 'Input changed'): run(args, CountingModel())

    def test_shards_equal_unsharded_and_reject_partial_or_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, _ = fixture(root)
            run(arguments(config, root/'whole'), CountingModel())
            shards = [root/'part0', root/'part1']
            run(arguments(config, shards[0], ['--num-shards', '2', '--shard-index', '0']), CountingModel())
            with self.assertRaisesRegex(ValueError, 'every shard'):
                summarize_runs(shards[:1], root/'bad')
            run(arguments(config, shards[1], ['--num-shards', '2', '--shard-index', '1']), CountingModel())
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize_runs([shards[0], shards[0]], root/'bad')
            summarize_runs([root/'whole'], root/'whole_summary', curves=False, n_boot=20)
            summarize_runs(shards, root/'merged', curves=False, n_boot=20)
            self.assertEqual((root/'whole_summary'/'per_image.csv').read_text(),
                             (root/'merged'/'per_image.csv').read_text())
            self.assertEqual((root/'whole_summary'/'summary.csv').read_text(),
                             (root/'merged'/'summary.csv').read_text())
            report = json.loads((root/'merged'/'analysis.json').read_text())
            self.assertEqual(report['images'], 2)  # all-split bypasses thumbnail/time filtering
            self.assertEqual(report['rows'], 30)

    def test_dense_grid_and_full_conditions_default(self):
        args = parser().parse_args(['--config', 'unused'])
        self.assertEqual(args.conditions, ['L', 'N', 'G', 'LN', 'LG', 'NG', 'LNG'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, _ = fixture(root)
            args = arguments(config, root/'dense'); args.levels = None; args.level_step = .25
            args.render_only = True
            run(args)
            contract = json.loads((root/'dense'/'contract.json').read_text())
            self.assertEqual(contract['levels'], [.25*i for i in range(1, 17)])
            model = CountingModel(); args.resume = True; run(args, model)
            self.assertEqual(model.calls, 0)

    def test_curve_math_censoring_pairing_and_sparse_groups(self):
        x = np.arange(5.)
        np.testing.assert_allclose(smoother(x, .1)@x, x, atol=1e-12)
        self.assertEqual(crossing(x, [0, .05, .15, .05, .3], .1), (1.5, True))
        self.assertTrue(np.isnan(crossing(x, np.zeros(5), .1)[0]))
        censored = interval([1., 2., np.nan, np.nan])
        self.assertEqual(censored['bootstrap_observed_fraction'], .5)
        self.assertTrue(np.isnan(interval([np.nan])['conditional_p025']))
        contract = dict(levels=[1., 2., 3., 4.], conditions=['L', 'N', 'G', 'LN', 'LG', 'NG', 'LNG'], seed=1, failure_delta=.1)
        rows = []
        for g in ('a', 'b', 'c'):
            for c in contract['conditions']:
                for s in contract['levels']:
                    rows.append(dict(group=g, condition=c, severity=s, metric='delta_gt_ordinal_error',
                                     value=s*(.08 if len(c) > 1 else .04)))
                    rows.append(dict(group=g, condition=c, severity=s, metric='gt_failure_rate', value=s/4))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); fit_group_curves(out, rows, contract, n_boot=20, penalty=.1)
            import csv
            with (out/'joint_boundary_shifts.csv').open() as f:
                shifts = list(csv.DictReader(f))
            self.assertEqual(len(shifts), 9)
            self.assertTrue(all(abs(float(r['earlier_by'])-1.25) < 1e-9 for r in shifts))
            self.assertTrue(all(float(r['bootstrap_observed_fraction']) == 1 for r in shifts))
            self.assertTrue((out/'fitted_response_curves.png').is_file())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fit_group_curves(out, [r for r in rows if r['group'] == 'a'], contract, n_boot=20)
            self.assertIn('insufficient_groups', (out/'curve_status.csv').read_text())
            self.assertFalse((out/'fitted_boundaries.csv').exists())


if __name__ == '__main__':
    unittest.main()
