"""Exercise the in-container command and its exit-code boundary."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from uuid import UUID, uuid4
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_inputs import create_run


COMMAND = Path(__file__).resolve().parents[1] / 'scripts/perf-then-eval'


class PerfThenEval(unittest.TestCase):
    def test_order_arguments_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = [root/'perf config.json', root/'suite config.json']
            for config in configs:
                config.write_text('{}')
            log = root/'calls'
            for command, code in [('perf-suite', 'PERF_CODE'), ('eval-suite', 'EVAL_CODE')]:
                tool = root/command
                tool.write_text(f'#!/bin/sh\nprintf "%s\\n" {command} "$@" >> "$LOG"\nexit "${code}"\n')
                tool.chmod(0o755)
            env = dict(os.environ, PATH=tmp+':'+os.environ['PATH'], LOG=str(log))
            custom = root/'output with spaces'
            ids = set()
            for first, second, output in [(0, 0, None), (0, 0, custom), (1, 0, custom),
                                          (130, 0, None), (0, 7, custom)]:
                base = output or Path('/tmp/vllm-suite')
                options = ['--output', str(output)] if output else []
                with self.subTest(first=first, second=second):
                    log.write_text('')
                    result = subprocess.run(['sh', COMMAND, *options, *configs], env=dict(env, PERF_CODE=str(first),
                                                                          EVAL_CODE=str(second)))
                    self.assertEqual(result.returncode, first or second)
                    calls = log.read_text().splitlines()
                    run_id = calls[calls.index('--group-id') + 1]
                    self.assertEqual(UUID(run_id).version, 4)
                    self.assertNotIn(run_id, ids)
                    ids.add(run_id)
                    shared = ['--output', str(base), '--group-id', run_id]
                    perf = ['perf-suite', '--config', str(configs[0]), *shared, '--stage', 'all']
                    accuracy = ['eval-suite', '--config', str(configs[1]), *shared]
                    self.assertEqual(calls, perf if first else perf + accuracy)

    def test_group_layout_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            run_id = uuid4()
            group = base/str(run_id)
            perf = create_run(base, [], group_id=run_id, phase='perf')
            self.assertEqual(perf, group/'perf')
            self.assertEqual(list(group.iterdir()), [perf])
            accuracy = create_run(base, [], group_id=run_id, phase='accuracy')
            self.assertEqual(accuracy, group/'accuracy')
            self.assertEqual(set(group.iterdir()), {perf, accuracy})
            for phase in ('perf', 'accuracy'):
                with self.assertRaises(FileExistsError):
                    create_run(base, [], group_id=run_id, phase=phase)
