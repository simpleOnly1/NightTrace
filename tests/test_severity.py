"""CPU scientific contracts and CLI integration; no DAV2 checkpoint required."""
import csv
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import yaml
from analysis.degradation_v2.pipeline import CONDITIONS, to_linear
from analysis.degradation_v2.severity import (DEFAULT_PROFILE, sample_plan,
    at_severity, render, audit_input, validate_profile)
from analysis.failure_boundary import gt_pairs, evaluate_gt, first_crossing
from diagnose_severity import parser, run


class FakeDepth(torch.nn.Module):
    def forward(self, x):
        return {'raw_relative': x.mean(1, keepdim=True)+.01}


class SeverityTests(unittest.TestCase):
    def test_identity_reproducibility_and_fixed_geometry(self):
        x = torch.linspace(.01, .9, 3*32*48).reshape(1, 3, 32, 48)
        plan = sample_plan(17); old = deepcopy(plan)
        rng = torch.random.get_rng_state().clone()
        for c in CONDITIONS:
            self.assertTrue(torch.equal(render(x, c, at_severity(plan, 0))[0], x))
        for s in (1, 2, 2.5, 3, 4):
            p = at_severity(plan, s)
            for a, b in zip(p['sources'], plan['sources']):
                self.assertEqual({k: v for k, v in a.items() if k != 'amplitude'},
                                 {k: v for k, v in b.items() if k != 'amplitude'})
            forward = {c: render(x, c, p)[0] for c in CONDITIONS}
            reverse = {c: render(x, c, p)[0] for c in reversed(CONDITIONS)}
            for c in CONDITIONS:
                self.assertTrue(torch.equal(forward[c], reverse[c]))
        self.assertEqual(plan, old)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_observed_dose_and_noise_audit(self):
        x = torch.full((1, 3, 160, 240), .4)
        p = sample_plan(8)
        light = [float(render(x, 'L', at_severity(p, s))[0].mean()) for s in range(5)]
        self.assertTrue(all(a > b for a, b in zip(light, light[1:])))
        glare = [float(render(x, 'G', at_severity(p, s))[1].mean()) for s in range(5)]
        self.assertTrue(all(a < b for a, b in zip(glare, glare[1:])))
        noise = []
        for s in (1, 2, 3, 4):
            ps = at_severity(p, s); y, f = render(x, 'N', ps)
            audit = audit_input(x, y, 'N', ps, f)
            self.assertEqual(audit['glare_rgb_mae'], 0)
            self.assertAlmostEqual(audit['noise_rgb_rms'], float((y-x).square().mean().sqrt()), places=6)
            noise.append(audit['noise_rgb_rms'])
        self.assertTrue(all(a < b for a, b in zip(noise, noise[1:])))
        ps = at_severity(p, 1); ps['read_sigma'] = 0
        y, _ = render(x, 'N', ps)
        # Unclipped middle-gray region: Poisson mean and variance in linear RGB.
        z = to_linear(y); expected = float(to_linear(x).mean())
        self.assertAlmostEqual(float(z.mean()), expected, delta=.001)
        self.assertAlmostEqual(float(z.var()), expected/ps['shot_peak'], delta=.00002)

    def test_flare_template_is_fixed_and_strong_dose_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
            field = np.exp(-(xx**2+yy**2)/.05)
            path = Path(tmp)/'flare.png'
            cv2.imwrite(str(path), (field*255).astype(np.uint8))
            plan = sample_plan(7, [str(path)], {'u': [.5, .5], 'v': [.5, .5], 'source_count': [1, 1]})
            x = torch.full((1, 3, 96, 144), .4)
            areas = []
            for s in (1, 2, 3, 4):
                p = at_severity(plan, s); y, f = render(x, 'G', p)
                self.assertEqual(p['sources'][0]['template'], str(path))
                areas.append(audit_input(x, y, 'G', p, f)['glare_visible_area'])
            self.assertGreater(areas[-1], .05)
            self.assertGreater(areas[-1], areas[0])

    def test_gt_ordering_not_clean_reference_and_ties(self):
        gt = np.linspace(1, 30, 48*32).reshape(32, 48)
        pairs = gt_pairs(gt, .1, 40, seed=7)
        correct = evaluate_gt(gt*5, gt, pairs, .1, 40)
        inverse = evaluate_gt(31-gt, gt, pairs, .1, 40)
        constant = evaluate_gt(np.ones_like(gt), gt, pairs, .1, 40)
        self.assertEqual(correct['gt_ordinal_error'], 0)
        self.assertAlmostEqual(correct['gt_abs_rel'], 0)
        self.assertEqual(inverse['gt_ordinal_error'], 1)
        self.assertEqual(constant['gt_ordinal_error'], .5)
        broken = gt.copy(); broken[0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_gt(broken, gt, pairs, .1, 40)
        sparse = gt_pairs(np.ones_like(gt), .1, 40)
        self.assertTrue(np.isnan(evaluate_gt(gt, np.ones_like(gt), sparse, .1, 40)['gt_ordinal_error']))
        tiny_gt = np.array([[1., 10.], [0., 0.]])
        tiny_pairs = gt_pairs(tiny_gt, .1, 40, pairs=8192)
        self.assertEqual(len(tiny_pairs[0]), 1)
        self.assertTrue(np.isnan(evaluate_gt(np.ones_like(tiny_gt), tiny_gt, tiny_pairs, .1, 40)['gt_ordinal_error']))

    def test_crossing_reports_recrossing_and_censoring(self):
        r = first_crossing([0, 1, 2, 3, 4], [0, .02, .14, .06, .3], .1)
        self.assertEqual((r['lower'], r['upper']), (1, 2)); self.assertTrue(r['recrossed'])
        self.assertEqual(first_crossing([0, 1], [0, .02], .1)['status'], 'not_observed')
        self.assertEqual(first_crossing([1, 2], [.2, .3], .1)['status'], 'left_censored')
        self.assertEqual(first_crossing([0, 1], [0, float('nan')], .1)['status'], 'insufficient_data')
        for bad in (float('nan'), -1, 5):
            with self.assertRaises(ValueError): at_severity(sample_plan(1), bad)
        profile = deepcopy(DEFAULT_PROFILE); profile['exposure'][2] = .9
        with self.assertRaises(ValueError): validate_profile(profile)

    def test_end_to_end_gt_and_render_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = root/'data'/'day'
            (package/'rgb').mkdir(parents=True); (package/'depth').mkdir()
            ids = ['1418133156135993', '1418133216135993']
            image = np.tile(np.linspace(20, 220, 48, dtype=np.uint8)[None, :, None], (32, 1, 3))
            for sid in ids:
                cv2.imwrite(str(package/'rgb'/(sid+'.png')), image)
                gt = np.tile(np.linspace(2, 25, 48)[None, :], (32, 1))
                np.save(package/'depth'/(sid+'_depth.npy'), gt)
            for split in ('train', 'test'):
                (package/(split+'_split.txt')).write_text('\n'.join(ids))
            cfg = dict(data=dict(name='robotcar', root=str(root/'data'), day_package='day',
                input_height=16, input_width=24, crop_width=40, crop_height=24, gt_space='raw'),
                model=dict(weights='not-needed', inverse_epsilon=.001),
                eval=dict(min_depth=.1, max_depth=40), diagnosis=dict(duplicate_mae=0))
            config = root/'config.yaml'; config.write_text(yaml.safe_dump(cfg))
            args = parser().parse_args(['--config', str(config), '--source', 'day-test', '--device', 'cpu',
                '--analytic-flare', '--out', str(root/'gt'), '--limit', '2', '--levels', '1', '4',
                '--conditions', *CONDITIONS[1:], '--save-vis', '--num-vis', '1', '--repeats', '2'])
            rows = run(args, FakeDepth())
            self.assertEqual(len(rows), 2*2*(1+7*2))
            for file in ('boundaries_per_image.csv', 'boundaries_group.csv', 'interactions_summary.csv',
                         'response_curves.png', 'gt_geometry.json', 'parameters.jsonl'):
                self.assertTrue((root/'gt'/file).is_file(), file)
            # Each image uses identical GT pair counts in every condition/repeat.
            for sid in {r['id'] for r in rows}:
                self.assertEqual(len({r['gt_ordinal_pairs'] for r in rows if r['id'] == sid}), 1)
            self.assertEqual(yaml.safe_load((root/'gt'/'run.yaml').read_text())['status'], 'complete')
            with self.assertRaisesRegex(ValueError, 'not empty'): run(args, FakeDepth())
            args.out = str(root/'render'); args.source = 'day-train'; args.render_only = True
            args.save_vis = False; args.repeats = 1
            records = run(args)
            self.assertFalse(any('gt_ordinal_error' in r for r in records))
            self.assertFalse((root/'render'/'boundaries_per_image.csv').exists())
            # Missing GT must fail, never fall back to the clean prediction.
            (package/'depth'/(ids[0]+'_depth.npy')).unlink()
            args.out = str(root/'missing'); args.source = 'day-test'
            with self.assertRaises(FileNotFoundError): run(args, FakeDepth())


if __name__ == '__main__':
    unittest.main()
