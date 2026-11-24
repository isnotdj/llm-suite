import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from eval_concurrency import records, write_records


class AccuracyConcurrencyTest(unittest.TestCase):
    def test_refill_while_first_blocked_and_publish_in_order(self):
        third = threading.Event()
        def predict(index):
            if index == 0:
                self.assertTrue(third.wait(3))
            if index == 2:
                third.set()
            return dict(id=index)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for workers in (1, 4):
                write_records(root/str(workers), lambda i: dict(id=i), range(10), workers, 10)
            self.assertEqual((root/'1').read_bytes(), (root/'4').read_bytes())
            write_records(root/'samples.jsonl', predict, range(10), 2, 10)
            self.assertEqual(list(records(root/'samples.jsonl')), list(records(root/'1')))
            self.assertFalse((root/'samples.partial.jsonl').exists())

    def test_failure_drains_already_started_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.jsonl'
            def predict(i):
                if i == 0:
                    raise ValueError('invalid answer')
                self.assertEqual(i, 1)
                for _ in range(300):
                    if path.with_suffix('.failure.json').exists():
                        return dict(id=i)
                    time.sleep(.01)
                self.fail('failure was not recorded promptly')
            with self.assertRaises(ValueError):
                write_records(path, predict, range(8), 2, 8)
            self.assertFalse(path.exists())
            self.assertEqual(list(records(path.with_suffix('.partial.jsonl'))),
                             [dict(ordinal=1, record=dict(id=1))])

    def test_input_failure_drains_workers_before_propagating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.jsonl'
            def inputs():
                yield 0
                yield 1
                raise ValueError('invalid frozen input')

            with self.assertRaisesRegex(ValueError, 'invalid frozen input'):
                write_records(path, lambda i: dict(id=i), inputs(), 4, 3)
            self.assertFalse(path.exists())
            saved = list(records(path.with_suffix('.partial.jsonl')))
            self.assertEqual(sorted((r['ordinal'], r['record']['id']) for r in saved),
                             [(0, 0), (1, 1)])
            failure = json.loads(path.with_suffix('.failure.json').read_text())
            self.assertEqual((failure['phase'], failure['ordinal']), ('input', 2))

    def test_interrupt_exits_before_worker_timeout(self):
        program = """
import sys, time
import run_suite
from run_journal import RunJournal
from pathlib import Path
from eval_concurrency import write_records
p = Path(sys.argv[1])
def slow(i):
    if i == 0:
        return dict(id=0)
    p.with_suffix('.started').touch()
    time.sleep(60)
def run():
    journal = run_suite.ACTIVE_RUN = RunJournal(p.parent, ['ceval'], ['ceval'])
    role = dict(api_key_env='TEST_ONLY_KEY')
    run_suite.ACTIVE_CONFIG = dict(target=role, judge=role)
    journal.ready({})
    journal.start('ceval')
    if sys.argv[2] == 'full':
        def full_disk():
            raise OSError(28, 'No space left on device')
        journal.flush = full_disk
    if sys.argv[2] == 'cleanup':
        p.with_suffix('.ordered.tmp').mkdir()
    write_records(p, slow, [0, 1], 1, 2)
run_suite._main = run
sys.exit(run_suite.cli())
"""
        for sig, code, disk in ((signal.SIGTERM, 143, 'ok'), (signal.SIGINT, 130, 'ok'),
                                (signal.SIGTERM, 143, 'full'), (signal.SIGINT, 130, 'full'),
                                (signal.SIGTERM, 143, 'cleanup'), (signal.SIGINT, 130, 'cleanup')):
            with self.subTest(sig=sig), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'samples.jsonl'
                with subprocess.Popen([sys.executable, '-c', program, str(path), disk], stderr=subprocess.PIPE, text=True,
                                      env=dict(os.environ, PYTHONPATH=str(SCRIPTS))) as process:
                    try:
                        for _ in range(300):
                            if path.with_suffix('.started').exists():
                                break
                            time.sleep(.01)
                        self.assertTrue(path.with_suffix('.started').exists())
                        process.send_signal(sig)
                        _, errors = process.communicate(timeout=3)
                        self.assertEqual(process.returncode, code)
                    finally:
                        if process.poll() is None:
                            process.kill()
                report = json.loads((path.parent/'summary.json').read_text())
                if disk == 'full':
                    self.assertIn('summary.json', errors)
                    self.assertIn('No space left', errors)
                else:
                    self.assertEqual(report['overall_status'], 'interrupted')
                if disk == 'cleanup':
                    self.assertIn('Could not clean up', errors)
                self.assertFalse(path.exists())
                self.assertEqual(list(records(path.with_suffix('.partial.jsonl'))),
                                 [dict(ordinal=0, record=dict(id=0))])
