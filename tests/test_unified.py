import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_unified as suite
from perf_bench_serve import connection_argv


class UnifiedTests(unittest.TestCase):
    model = {'name':'default', 'base_url':'http://localhost:8000/v1'}

    def test_stage_defaults_and_verbatim_argv(self):
        conf, data = suite.performance({}, self.model)
        self.assertEqual(conf['stage'], 'all')
        self.assertEqual(conf['requests_per_concurrency'], 10)
        self.assertEqual(conf['overload_concurrency'], [192,256])
        self.assertEqual([p['name'] for p in data['profiles']], ['4k','16k','64k'])
        tokens = ['--seed=1','--seed','42','--header','A=x','B=y']
        conf, data = suite.performance(dict(stage='single',bench=tokens), self.model)
        self.assertEqual(data['bench'][-len(tokens):], tokens)
        self.assertNotIn('A=x', str(suite.redact(dict(execution=dict(perf=conf)))))
        with self.assertRaises(ValueError):
            suite.performance(dict(stage='single', profiles=[]), self.model)

    def test_failure_barrier_and_unique_directories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = root/'map.json'
            samples = root/'ceval'
            samples.mkdir()
            mapping.write_text(json.dumps(dict(version=1, datasets=dict(ceval=str(samples)))))
            config = root/'.vllm.conf'
            config.write_text(json.dumps(dict(model=self.model, execution=dict(
                perf={},accuracy=dict(dataset_map='map.json')))))
            args = argparse.Namespace(output=root/'out', config=config)
            with patch('compass_runner.load_configs'), patch.object(suite,'invoke',return_value=1) as invoke:
                self.assertEqual(suite.execute(args),1)
                self.assertEqual(suite.execute(args),1)
                self.assertEqual(invoke.call_count,2)
                self.assertEqual(invoke.call_args.args[0][0],'perf-suite')
            runs = list(args.output.iterdir())
            self.assertEqual(len(runs),2)
            for run in runs:
                state = json.loads((run/'summary.json').read_text())
                self.assertEqual(state['phases'], dict(perf='failed',accuracy='not_started'))
                self.assertEqual(state['run_id'], run.name)

    def test_missing_judge_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'accuracy.judge'):
            suite.accuracy(dict(datasets=['followbench']), self.model, Path.cwd())

    def test_seed_and_partial_slo_keep_defaults_on_custom_prefix(self):
        for suffix in ('', '/v1', '/gateway/', '/gateway/v1'):
            with self.subTest(suffix=suffix):
                url = 'http://localhost:8000' + suffix
                conf, payload = suite.performance(dict(bench=['--seed','7'], slo={'ttft_ms':500}),
                                                  dict(self.model, base_url=url))
                args = connection_argv(payload['model_api_url'], payload['bench'])
                self.assertEqual(args[-2:], ['--seed','7'])
                self.assertEqual(args[args.index('--backend') + 1], 'openai-chat')
                self.assertEqual(args[args.index('--base-url')+1] + args[args.index('--endpoint')+1],
                                 url.rstrip('/') + '/chat/completions')
                self.assertEqual(conf['slo'], {'ttft_ms':500, 'tpot_ms':50})

    def test_duplicate_yaml_keys_rejected_with_line(self):
        for text, line in [('execution: {}\nexecution: {}',2),
                           ('execution:\n  perf: {}\n  perf: {}',3)]:
            with self.assertRaisesRegex(ValueError, f'Duplicate YAML key .*line {line}'):
                suite.yaml.load(text, Loader=suite.ConfigLoader)

    def test_accuracy_uses_launch_snapshot_after_map_and_alias_rotate(self):
        from dataset_mapping import read_mapping
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            a, b, alias = (root/k for k in ('v1','v2','current'))
            a.mkdir()
            b.mkdir()
            alias.symlink_to(a)
            mapping = root/'map.json'
            mapping.write_text(json.dumps(dict(version=1, datasets={'ceval':'current'})))
            config = root/'.vllm.conf'
            config.write_text(json.dumps(dict(model=self.model, execution=dict(
                perf={}, accuracy=dict(dataset_map='map.json')))))
            def invoke(cmd, env):
                if cmd[0] == 'perf-suite':
                    alias.unlink()
                    alias.symlink_to(b)
                    mapping.write_text(json.dumps(dict(version=1, datasets={'ceval':'v2'})))
                else:
                    data = json.loads(Path(cmd[cmd.index('--config')+1]).read_text())
                    path = Path(data['dataset_map'])
                    self.assertEqual(path.parent.name, cmd[cmd.index('--group-id')+1])
                    self.assertEqual(read_mapping(path), {'ceval':a})
                return 0
            with patch('compass_runner.load_configs'), patch.object(suite,'invoke',side_effect=invoke):
                self.assertEqual(suite.execute(argparse.Namespace(output=root/'out', config=config)), 0)

    def test_connection_and_detail_flags_are_resolved_per_workload(self):
        conf, data = suite.performance(dict(profiles=[
            dict(name='standard'), dict(name='other', bench=['--port=8123', '--save-detailed'])
        ]), dict(self.model, base_url='http://example/gateway/v1'))
        common = data['bench']
        default = connection_argv(data['model_api_url'], common)
        other = connection_argv(data['model_api_url'], common + conf['profiles'][1]['bench'])
        self.assertIn('--base-url', default)
        self.assertNotIn('--save-detailed', default)
        self.assertNotIn('--base-url', other)
        self.assertIn('/gateway/v1/chat/completions', other)
        self.assertIn('--save-detailed', other)
        with self.assertRaisesRegex(ValueError, 'either --base-url'):
            connection_argv(data['model_api_url'], common + ['--base-url=http://a', '--host', 'b'])
