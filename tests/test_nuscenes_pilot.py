"""Check pilot budgets, separation of stages, and stopping on failed subprocesses."""
import json
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.run_nuscenes_pilot import build_commands, execute, main, require_audit, validate_config


class NuscenesPilotTests(unittest.TestCase):
    def test_pilot_pairs_all_conditions_and_defers_summary(self):
        commands = build_commands('pilot', Path('/cfg'), Path('/runs'), python='python')
        self.assertEqual([name for name, _ in commands], ['inference', 'summarize'])
        cmd = commands[0][1]
        self.assertEqual(cmd[cmd.index('--conditions') + 1:cmd.index('--levels')],
                         ['L', 'N', 'G', 'LN', 'LG', 'NG', 'LNG'])
        self.assertEqual(cmd[cmd.index('--levels') + 1:cmd.index('--repeats')], ['1','2','3','4'])
        self.assertEqual(cmd[cmd.index('--source') + 1], 'day-test')
        self.assertEqual(cmd[cmd.index('--pairs') + 1], '8192')
        self.assertIn('--defer-summary', cmd)
        self.assertIn('--no-fit', commands[1][1])
        self.assertNotIn('--resume', cmd)
        resumed = build_commands('pilot', Path('/cfg'), Path('/runs'), resume=True)
        self.assertIn('--resume', resumed[0][1])

    def test_dry_run_does_not_create_output_or_start_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.yaml'
            config.write_text('test fixture')
            out = root / 'not_created'
            yaml = SimpleNamespace(safe_load=lambda text: {
                'data': {'name': 'nuscenes'}, 'model': {'name': 'dav2_vitl'}})
            with patch.dict(sys.modules, {'yaml': yaml}), patch('subprocess.run') as run:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    code = main(['--stage', 'pilot', '--config', str(config),
                                 '--out', str(out), '--dry-run'])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())
            run.assert_not_called()
            plan = json.loads(output.getvalue())
            command = plan['commands'][0][1]
            conditions = command[command.index('--conditions')+1:command.index('--levels')]
            levels = command[command.index('--levels')+1:command.index('--repeats')]
            images = int(command[command.index('--limit')+1])
            repeats = int(command[command.index('--repeats')+1])
            self.assertEqual(plan['maximum_forwards'], images * (1 + len(conditions)*len(levels)*repeats))

    def test_stages_and_pilot_bounds(self):
        audit = build_commands('audit', Path('/cfg'), Path('/runs'))
        self.assertEqual([n for n, _ in audit], ['gt_audit'])
        self.assertNotIn('--limit', audit[0][1])  # audit the full test split
        self.assertEqual([n for n, _ in build_commands('summarize', Path('/cfg'), Path('/runs'))], ['summarize'])
        for count in [0, -1, 17]:
            with self.assertRaises(ValueError):
                build_commands('pilot', Path('/cfg'), Path('/runs'), limit=count)
        with self.assertRaises(ValueError):
            build_commands('audit', Path('/cfg'), Path('/runs'), resume=True)

    def test_wrong_dataset_or_unsupported_model_rejected(self):
        validate_config({'data': {'name': 'nuscenes'}, 'model': {'name': 'dav2_vitl'}})
        for cfg in [{'data': {'name': 'robotcar'}, 'model': {'name': 'dav2_vitl'}},
                    {'data': {'name': 'nuscenes'}, 'model': {'name': 'unidepthv2'}}]:
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_full_nonempty_audit_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises(ValueError):
                require_audit(base)
            (base / 'gt_audit').mkdir()
            p = base / 'gt_audit/summary.json'
            for data in [{'dataset':'nuscenes','checked':0,'available':0},
                         {'dataset':'nuscenes','checked':8,'available':500},
                         {'dataset':'robotcar','checked':8,'available':8}]:
                p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    require_audit(base)
            p.write_text(json.dumps({'dataset':'nuscenes','checked':500,'available':500}))
            self.assertEqual(require_audit(base)['checked'], 500)

    def test_failure_stops_summary_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'status.json'
            calls = []
            def fail(command, **kwargs):
                calls.append(command)
                raise subprocess.CalledProcessError(2, command)
            with self.assertRaises(subprocess.CalledProcessError):
                execute([('inference', ['infer']), ('summarize', ['summary'])], Path(tmp), status, fail)
            self.assertEqual(calls, [['infer']])
            data = json.loads(status.read_text())
            self.assertEqual(data['status'], 'failed')
            self.assertEqual(data['steps'][0]['status'], 'failed')

    def test_success_records_completed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'status.json'
            calls = []
            execute([('inference', ['infer']), ('summarize', ['summary'])], Path(tmp), status,
                    lambda command, **kwargs: calls.append(command))
            self.assertEqual(calls, [['infer'], ['summary']])
            data = json.loads(status.read_text())
            self.assertEqual(data['status'], 'completed')
            self.assertTrue(all(s['status'] == 'completed' for s in data['steps']))


if __name__ == '__main__':
    unittest.main()
