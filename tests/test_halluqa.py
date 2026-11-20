import importlib.util
from pathlib import Path
import sys
import types
import unittest

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
        sys.modules['openai'] = types.SimpleNamespace(APIStatusError=StatusError,
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

    def test_short_vote_batch_is_rejected(self):
        calls = []
        def create(**kw):
            calls.append(kw)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='是'))])
        with self.assertRaises(ValueError):
            module.judge(self.client(create), {})
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(x['n'] == 5 for x in calls))

    def test_truncated_votes_are_rejected(self):
        def create(**kw):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content='是'), finish_reason='length') for _ in range(5)])
        with self.assertRaises(ValueError):
            module.judge(self.client(create), {})

    def test_official_votes_preserved(self):
        votes = ['是', '是', '是', '否', '否']
        def create(**kw):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=v), finish_reason='stop') for v in votes])
        self.assertEqual(module.judge(self.client(create), {}), votes)

if __name__ == '__main__':
    unittest.main()
