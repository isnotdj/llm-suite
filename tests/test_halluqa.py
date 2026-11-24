import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
import os

spec = importlib.util.spec_from_file_location('halluqa', Path(__file__).parents[1] / 'scripts/halluqa_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class StatusError(Exception):
    def __init__(self, status):
        self.status_code = status

class TransportError(Exception):
    pass

class Tests(unittest.TestCase):
    def setUp(self):
        self.original = sys.modules.get('openai')
        sys.modules['openai'] = types.SimpleNamespace(APIError=(StatusError, TransportError), APIStatusError=StatusError,
            APIConnectionError=TransportError, APITimeoutError=TransportError)

    def tearDown(self):
        if self.original is None:
            del sys.modules['openai']
        else:
            sys.modules['openai'] = self.original

    def client(self, function):
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=function)))

    def test_auth_error_not_retried(self):
        calls = []
        def create(**kw):
            calls.append(kw)
            raise StatusError(401)
        with self.assertRaises(StatusError):
            module.request(self.client(create), {}, sleep=lambda _: self.fail('must not sleep'))
        self.assertEqual(len(calls), 1)

    def test_transient_error_is_bounded(self):
        calls, waits = [], []
        def create(**kw):
            calls.append(kw)
            raise StatusError(503)
        with self.assertRaises(StatusError):
            module.request(self.client(create), {}, sleep=waits.append)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(waits), 2)
        self.assertTrue(all(t <= 8 for t in waits))

    def test_judge_exit_policy(self):
        def denied(**kwargs):
            raise StatusError(401)
        failures = [(lambda: module.judge(self.client(denied), {}), 1)]
        for exc, code in [(module.JudgeFailure('invalid votes'), 1),
                          (OSError('disk full'), os.EX_IOERR),
                          (RuntimeError('broken scorer'), os.EX_SOFTWARE),
                          (ValueError('broken scorer'), os.EX_SOFTWARE)]:
            failures.append((exc, code))
        for failure, expected in failures:
            with self.subTest(failure=failure), patch.object(module, 'main', side_effect=failure), patch.object(module.traceback, 'print_exc'):
                self.assertEqual(module.cli(), expected)

    def test_short_vote_batch_is_rejected(self):
        calls = []
        def create(**kw):
            calls.append(kw)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='是'))])
        with self.assertRaises(ValueError):
            module.judge(self.client(create), {})
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(x['n'] == 5 for x in calls))

    def test_malformed_batches_retry_then_recover_or_fail_the_dataset(self):
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content='否'), finish_reason='stop')
        valid = types.SimpleNamespace(choices=[choice] * 5)
        for choices in (None, {}, [None] * 5, [types.SimpleNamespace(message=None)] * 5):
            malformed = types.SimpleNamespace(choices=choices)
            with self.subTest(choices=choices):
                create = Mock(side_effect=[malformed, valid])
                self.assertEqual(module.judge(self.client(create), {}), ['否'] * 5)
                create = Mock(return_value=malformed)
                with self.assertRaises(module.JudgeFailure):
                    module.judge(self.client(create), {})
                self.assertEqual(create.call_count, 5)

    def test_length_terminated_votes_are_scored(self):
        def create(**kw):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content='是'), finish_reason='length') for _ in range(5)])
        self.assertEqual(module.judge(self.client(create), {}), ['是'] * 5)

    def test_official_votes_preserved(self):
        votes = ['是', '是', '是', '否', '否']
        def create(**kw):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=v), finish_reason='stop') for v in votes])
        self.assertEqual(module.judge(self.client(create), {}), votes)

if __name__ == '__main__':
    unittest.main()
