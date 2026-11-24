import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from perf_bench_serve import parse_options
from run_perf import main


def register(parser):
    parser.add_argument('--model', required=True)
    parser.add_argument('--request-rate', type=float)
    parser.add_argument('--burstiness', type=float, default=1.0)
    parser.add_argument('--header', nargs='*')
    parser.add_argument('--save-detailed', action='store_true')
    parser.add_argument('--goodput', nargs='+')
    parser.add_argument('--limits', type=json.loads)
    parser.add_argument('--save-result', action='store_true')
    parser.add_argument('--append-result', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result-dir')
    parser.add_argument('--result-filename')


class NativeBenchmarkTests(unittest.TestCase):
    def test_single_export_cannot_target_previous_run(self):
        from perf_bench_serve import prepare_argv_config, run_native
        def register_single(parser):
            register(parser)
            for name in ('dataset-path', 'tokenizer', 'lora-modules'):
                parser.add_argument('--' + name)
        benchmarks = ModuleType('vllm.benchmarks')
        benchmarks.serve = SimpleNamespace(add_cli_args=register_single)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'vllm.benchmarks': benchmarks}), \
                patch('perf_bench_serve.version', return_value='0.11.0'):
            previous = Path(tmp).resolve()/'previous'/'perf'
            current = previous.parents[1]/'current'/'perf'
            previous.mkdir(parents=True)
            current.mkdir(parents=True)
            report = previous/'result.json'
            report.write_text('original')
            (current/'alias').symlink_to(previous)
            for directory in (str(previous), '../../previous/perf', 'alias'):
                raw = prepare_argv_config({'bench':['--model','default','--save-result',
                    '--result-filename','result.json','--result-dir',directory]})
                with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, 'inside this run'):
                    run_native(raw, current, 'all', False, {})
                self.assertEqual(report.read_text(), 'original')

    def test_options_use_cli_types_and_defaults(self):
        parser = argparse.ArgumentParser()
        register(parser)
        args = parse_options(parser, dict(model='--literal', request_rate='inf',
                             header=[], goodput=['ttft:20'], limits={'image': 2},
                             save_detailed=False))
        self.assertEqual(args.model, '--literal')
        self.assertEqual(args.request_rate, float('inf'))
        self.assertFalse(args.save_detailed)
        self.assertEqual(args.limits, {'image': 2})
        self.assertEqual(args.goodput, ['ttft:20'])
        self.assertEqual(args.header, [])
        for invalid in ({'typo': 1}, {'header': 'a=b'}, {'header': ['--save-detailed']},
                        {'save_detailed': 1}, {'request_rate': 2, 'request-rate': 3}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_options(parser, {'model': 'fixture', **invalid})

    def test_native_entry_preserves_upstream_report_and_validates_offline(self):
        benchmarks = ModuleType('vllm.benchmarks')
        serve = ModuleType('serve')
        benchmarks.serve = serve
        serve.add_cli_args = register
        table = '============= Serving Benchmark Result =============\nP99 TTFT (ms): 12.34\n' + '=' * 50 + '\n'
        calls = []

        serve.get_samples = lambda: [1]
        async def send(**kwargs):
            return SimpleNamespace(success=True)
        serve.ASYNC_REQUEST_FUNCS = {'chat': send}

        def benchmark(args):
            calls.append(args)
            serve.get_samples()
            asyncio.run(serve.ASYNC_REQUEST_FUNCS['chat'](pbar=None))
            print(args)
            print(table, end='')
            return {'p99_ttft_ms': 12.34, 'completed': 1}

        serve.main = benchmark
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'vllm.benchmarks': benchmarks}), \
                patch('perf_bench_serve.version', return_value='0.11.0'):
            root = Path(folder)
            config = root / 'perf.json'
            config.write_text(json.dumps({'vllm_bench_serve': {
                'model': 'fixture', 'request-rate': 2, 'save_detailed': True,
                'header': ['custom=private']}}))
            def run(output, *extra):
                return main(['--config', str(config), '--output', str(root/output), *extra])
            self.assertEqual(run('check', '--validate-only'), 0)
            self.assertFalse(calls)
            self.assertEqual(run('run'), 0)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].save_detailed)
            self.assertEqual(calls[0].request_rate, 2)
            self.assertEqual((next((root/'run').iterdir())/'vllm-report.txt').read_text(), table)
            self.assertEqual(json.loads((next((root/'run').iterdir())/'vllm-result.json').read_text()), {'p99_ttft_ms': 12.34, 'completed': 1})
            self.assertEqual(run('wrong-stage', '--stage', 'soak'), 1)
            config.write_text('{"seed": 42, "vllm_bench_serve": {"model": "fixture"}}')
            self.assertEqual(run('mixed'), 1)

    def test_failure_matrix(self):
        table = '====== Serving Benchmark Result ======\n' + '='*50 + '\n'
        for case, code, status, want_report in (
                ('all',0,'complete',True), ('partial',1,'incomplete',True),
                ('none',1,'failed',True), ('empty',1,'failed',False),
                ('readiness',1,'failed',False), ('export',1,'failed',True), ('single_export',1,'failed',True),
                ('before_end',130,'interrupted',False), ('after_end',130,'interrupted',True)):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                serve = ModuleType('serve')
                def parser(p):
                    register(p)
                    for flag in ('--dataset-path','--label','--backend','--ramp-up-strategy','--max-concurrency','--tokenizer','--lora-modules'):
                        p.add_argument(flag)
                    p.add_argument('--num-prompts', type=int, default=200)
                serve.add_cli_args = parser
                serve.get_samples = lambda: [] if case == 'empty' else [0,1,2]
                async def send(*, success, **kwargs):
                    return SimpleNamespace(success=success)
                serve.ASYNC_REQUEST_FUNCS = {'chat':send}
                def benchmark(args):
                    samples = serve.get_samples()
                    fn = serve.ASYNC_REQUEST_FUNCS['chat']
                    asyncio.run(fn(success=True))  # readiness is never measured
                    if case == 'readiness':
                        raise ValueError('endpoint not ready')
                    successful = 0
                    for i in samples:
                        ok = case != 'none' and (case != 'partial' or i > 0)
                        successful += asyncio.run(fn(success=ok,pbar=None)).success
                    if case == 'before_end':
                        print(table.splitlines()[0])
                        raise KeyboardInterrupt()
                    print(table,end='')
                    if case == 'export':
                        (next((root/'run').iterdir())/'exports').write_text('simulated output conflict')
                    if case == 'single_export':
                        (Path(args.result_dir)/'result.json').touch()
                    if case == 'after_end':
                        self.assertTrue((next((root/'run').iterdir())/'vllm-report.txt').is_file())
                        raise KeyboardInterrupt()
                    return {'date':'stamp','completed':successful}
                serve.main = benchmark
                benchmarks = ModuleType('vllm.benchmarks')
                benchmarks.serve = serve
                config = root/'config.json'
                config.write_text(json.dumps({'vllm_bench_serve': {
                    'model':'fixture','save_result':case=='export','result_dir':'metrics',
                    'header':['test=secret-value']}}))
                if case == 'single_export':
                    config.write_text(json.dumps({'bench':['--model','fixture','--save-result',
                        '--result-dir','metrics','--result-filename','result.json']}))
                with patch.dict(sys.modules, {'vllm.benchmarks':benchmarks}), patch('perf_bench_serve.version',return_value='0.11.0'):
                    self.assertEqual(main(['--config',str(config),'--output',str(root/'run')]),code)
                out = next((root/'run').iterdir())
                summary = json.loads((out/'summary.json').read_text())
                self.assertEqual(summary['overall_status'],status)
                self.assertEqual((out/'vllm-report.txt').exists(),want_report)
                self.assertNotIn('secret-value',(out/'setup.json').read_text())
                self.assertEqual(summary['requests']['selected'],0 if case=='empty' else 3)
                self.assertEqual(summary['requests']['dispatched'],0 if case in ('empty','readiness') else 3)
                if case == 'single_export':
                    self.assertEqual(summary['measurement_status'], 'complete')
                if case in ('export','single_export'):
                    self.assertEqual(summary['traffic_status'], 'complete')
                    self.assertEqual(summary['artifacts']['export'],'failed')
                    self.assertTrue((out/'vllm-result.json').is_file())

    def test_export_paths_and_inputs_are_separate(self):
        from run_inputs import create_output
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root/'nas'
            inputs.mkdir()
            config = inputs/'launch.json'
            original = b'{"mode":"fixture"}'
            config.write_bytes(original)
            alias = root/'alias'
            alias.symlink_to(inputs, target_is_directory=True)
            for dest in (config,inputs/'run',alias/'run'):
                with self.assertRaises(ValueError):
                    create_output(dest.resolve(), [config, inputs])
            self.assertEqual(config.read_bytes(),original)
            self.assertEqual(list(inputs.iterdir()),[config])
            output = root/'outputs/run'
            create_output(output,[config,inputs])
            with self.assertRaises(FileExistsError):
                create_output(output,[config,inputs])

    def test_config_overwrite_is_rejected_before_requests(self):
        benchmarks = ModuleType('vllm.benchmarks')
        serve = ModuleType('serve')
        benchmarks.serve = serve
        serve.add_cli_args = register
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{'vllm.benchmarks':benchmarks}), patch('perf_bench_serve.version',return_value='0.11.0'):
            root = Path(directory)
            config = root/'launch.json'
            for i,(dest,filename) in enumerate(((str(root),config.name),('../',config.name),('nested',str(config)),('nested','../../launch.json'))):
                with self.subTest(dest=dest,filename=filename):
                    raw = json.dumps({'vllm_bench_serve':{'model':'fixture','save_result':True,'result_dir':dest,'result_filename':filename}})
                    config.write_text(raw)
                    output = root/f'run-{i}'
                    self.assertEqual(main(['--config',str(config),'--output',str(output)]),1)
                    self.assertEqual(config.read_text(),raw)
                    self.assertFalse((next(output.iterdir())/'exports').exists())

    def test_export_creates_only_new_files(self):
        from perf_bench_serve import export_result
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',SAVE_TO_PYTORCH_BENCHMARK_FORMAT=''):
            path = Path(directory)/'result.json'
            export_result(None,path,{'completed':1},{})
            with self.assertRaises(FileExistsError):
                export_result(None,path,{'completed':2},{})
            self.assertEqual(json.loads(path.read_text()),{'completed':1})

    def test_timestamp_runs_are_unique_and_network_storage_rejected(self):
        from concurrent.futures import ThreadPoolExecutor
        from run_inputs import create_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'results'
            with ThreadPoolExecutor(max_workers=8) as pool:
                runs = list(pool.map(lambda _: create_run(root, []), range(20)))
            self.assertEqual(len(set(runs)), 20)
            self.assertTrue(all(p.parent == root.resolve() and p.is_dir() for p in runs))
            with patch('run_inputs.sys.platform', 'linux'), patch('pathlib.Path.read_bytes',
                    return_value=b'1 0 0:1 / / rw - nfs server:/share rw\n'):
                with self.assertRaisesRegex(ValueError, 'local storage'):
                    create_run(root/'nas-output', [])
            self.assertFalse((root/'nas-output').exists())

    def test_macos_checks_volume_type_before_creating_run(self):
        from run_inputs import create_run
        with tempfile.TemporaryDirectory() as directory, patch('run_inputs.sys.platform', 'darwin'):
            volume = Path(directory).resolve()/'volume with spaces'
            volume.mkdir()
            alias = volume.parent/'alias'
            alias.symlink_to(volume, target_is_directory=True)
            for fs_type in ('smbfs', 'nfs', 'apfs'):
                with self.subTest(fs_type=fs_type):
                    payload = {'storage-system-information': {'filesystem': [{'type': fs_type}]}}
                    output = alias/'new'/fs_type
                    with patch('subprocess.check_output', return_value=json.dumps(payload).encode()) as probe:
                        if fs_type == 'apfs':
                            self.assertTrue(create_run(output, []).is_dir())
                        else:
                            with self.assertRaisesRegex(ValueError, fs_type):
                                create_run(output, [])
                            self.assertFalse(output.parent.exists())
                        probe.assert_called_once_with(['/bin/df', '--libxo', 'json', '-Y', str(volume)])

            with patch('subprocess.check_output', return_value=b'{}'), self.assertRaisesRegex(ValueError, 'filesystem type'):
                create_run(alias/'unidentified', [])
            self.assertFalse((alias/'unidentified').exists())

    def test_output_inside_a_second_bind_mount_of_the_input(self):
        from run_inputs import create_output
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            nas, alias = base/'nas', base/'writable'
            nas.mkdir()
            alias.mkdir()
            mounts = [(nas,b'0:42',Path('/share/model')), (alias,b'0:42',Path('/share/model/subdir'))]
            with patch('run_inputs.filesystem_locations',return_value=mounts):
                with self.assertRaisesRegex(ValueError,'overlaps input'):
                    create_output(alias/'new',[nas])
            self.assertFalse((alias/'new').exists())
