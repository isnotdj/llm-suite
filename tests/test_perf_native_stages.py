"""Exercise the public CLI and count separate calls to vllm.benchmarks.serve.main."""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_perf import main
from test_perf_bench_serve import register


class NativeStages(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.file = self.root/'plan.json'
        tokenizer = self.root/'tokenizer'
        tokenizer.mkdir()
        (tokenizer/'tokenizer_config.json').write_text('{}')
        self.cfg = dict(vllm_bench_serve=dict(model='fixture', tokenizer=str(tokenizer), dataset_name='random'),
                        profiles=[dict(name='tiny')], concurrency=[1, 2],
                        overload_concurrency=[3], requests_per_concurrency=2, soak_requests=7)
        self.calls = []
        self.short_sample = False
        self.interrupt = False
        self.soak_tpot = 1
        self.normal_tpot = 1
        self.output_length = 2
        serve = ModuleType('serve')

        def parser(p):
            register(p)
            p.add_argument('--metadata', nargs='*')
            p.add_argument('--dataset-path')
            p.add_argument('--backend', default='openai')
            for key in ('dataset-name', 'metric-percentiles', 'percentile-metrics', 'tokenizer'):
                p.add_argument('--'+key)
            for key in ('num-prompts', 'max-concurrency', 'ready-check-timeout-sec', 'prefix-repetition-num-prefixes'):
                p.add_argument('--'+key, type=int)

        async def send(**kwargs):
            return SimpleNamespace(success=True)

        def benchmark(args):
            if self.calls:
                previous = self.output()/self.calls[-1][0]
                self.assertTrue((previous/'summary.json').is_file())
                self.assertTrue((previous/'vllm-result.json').is_file())
            folder = Path(args.result_dir).parent
            phase = folder.name
            self.calls.append((folder.parent.name+'/'+phase, args.max_concurrency, args.num_prompts))
            self.assertEqual(args.request_rate, float('inf'))
            self.assertEqual(args.ready_check_timeout_sec, 0)
            if self.interrupt and phase == 'overload-c3-r2':
                signal.raise_signal(signal.SIGTERM)
            count = args.num_prompts - int(self.short_sample and phase != 'connectivity')
            samples = serve.get_samples(count)
            result = asyncio.run(serve.benchmark(samples))
            if not args.save_detailed:
                result.pop('output_lens')
            print('============= Serving Benchmark Result =============')
            print(phase)
            print('='*50)
            return dict(result, completed=count, p95_ttft_ms=1, p95_tpot_ms=self.soak_tpot if phase == 'soak' else self.normal_tpot,
                        output_throughput=args.max_concurrency*100)

        async def upstream_benchmark(samples):
            for _ in samples:
                await serve.ASYNC_REQUEST_FUNCS['chat'](pbar=None)
            return {'output_lens': [self.output_length] * len(samples)}

        serve.benchmark = upstream_benchmark
        serve.add_cli_args = parser
        serve.get_samples = lambda n: list(range(n))
        serve.ASYNC_REQUEST_FUNCS = {'chat': send}
        serve.main = benchmark
        benchmarks = ModuleType('vllm.benchmarks')
        benchmarks.serve = serve
        self.enterContext(patch.dict(sys.modules, {'vllm.benchmarks':benchmarks}))
        self.enterContext(patch('perf_bench_serve.version', return_value='0.11.0'))

    def output(self):
        return next((self.root/'out').iterdir())

    def launch(self, *args):
        self.file.write_text(json.dumps(self.cfg))
        return main(['--config',str(self.file),'--output',str(self.root/'out'),*args])

    def test_official_calls_and_artifacts(self):
        self.assertEqual(self.launch(), 0)
        self.assertEqual(self.calls, [
            ('tiny/connectivity',1,1), ('tiny/scan-c1',1,2), ('tiny/scan-c2',2,4),
            ('tiny/overload-c3-r1',3,3), ('tiny/overload-c3-r2',3,3),
            ('tiny/overload-c3-r3',3,3), ('tiny/soak',2,7)])
        reports = json.loads((self.output()/'vllm-reports.json').read_text())
        self.assertEqual([r['phase'] for r in reports], [p.split('/')[1] for p, _, _ in self.calls])
        aggregate = (self.output()/'vllm-report.txt').read_text()
        for phase, _, _ in self.calls:
            report = (self.output()/phase/'vllm-report.txt').read_text()
            self.assertIn(phase.split('/')[1], report)
            self.assertIn(report, aggregate)
        self.assertEqual(json.loads((self.output()/'summary.json').read_text())['profiles']['tiny']['sweet_concurrency'],2)

    def test_short_dataset_stops_before_dispatch(self):
        self.short_sample = True
        self.assertEqual(self.launch('--stage','overload'),1)
        self.assertEqual(len(self.calls),2)
        child = json.loads((self.output()/'tiny/overload-c3-r1/summary.json').read_text())
        self.assertEqual(child['requests']['dispatched'],0)

    def test_interrupt_preserves_previous_round(self):
        self.interrupt = True
        self.assertEqual(self.launch('--stage','overload'),130)
        self.assertTrue((self.output()/'tiny/overload-c3-r1/vllm-report.txt').is_file())
        self.assertFalse((self.output()/'tiny/overload-c3-r3').exists())
        child = json.loads((self.output()/'tiny/overload-c3-r2/summary.json').read_text())
        index = json.loads((self.output()/'vllm-reports.json').read_text())
        self.assertEqual([r['phase'] for r in index], ['connectivity', 'overload-c3-r1'])
        self.assertEqual(child['traffic_status'], 'interrupted')
        self.assertEqual(child['overall_status'], 'interrupted')

    def test_validate_only_never_benchmarks(self):
        self.assertEqual(self.launch('--validate-only'),0)
        self.assertFalse(self.calls)

    def test_invalid_later_profile_before_traffic(self):
        self.cfg['profiles'].append(dict(name='other', vllm_bench_serve={'typo': 1}))
        self.assertEqual(self.launch(),1)
        self.assertFalse(self.calls)

    def test_one_token_preserves_success_and_has_no_tpot_coverage(self):
        self.output_length = 1
        self.assertEqual(self.launch('--stage', 'soak'), 1)
        state = json.loads((self.output()/'summary.json').read_text())['profiles']['tiny']
        self.assertEqual(state['soak_status'], 'skipped_no_sweet_spot')
        self.assertIsNone(state['sweet_concurrency'])
        self.assertTrue(all(p['status'] == 'complete' and p['requests']['tpot_samples'] == 0
                            for p in state['runs'] if p['phase'] != 'soak'))

    def test_all_profiles_reach_barrier_before_next_stage(self):
        self.cfg['profiles'].append(dict(name='other'))
        self.assertEqual(self.launch(), 0)
        phases = [path.split('/')[1].split('-')[0] for path, _, _ in self.calls]
        index = json.loads((self.output()/'vllm-reports.json').read_text())
        self.assertEqual([r['profile']+'/'+r['phase'] for r in index], [p for p, _, _ in self.calls])
        self.assertEqual(phases, ['connectivity']*2 + ['scan']*4 + ['overload']*6 + ['soak']*2)

    def test_missing_local_tokenizer_before_first_probe(self):
        self.cfg['profiles'].append(dict(name='other', vllm_bench_serve={'tokenizer': str(self.root/'missing')}))
        self.assertEqual(self.launch(), 1)
        self.assertFalse(self.calls)

    def test_invalid_percentile_before_first_probe(self):
        self.cfg['vllm_bench_serve']['metric_percentiles'] = '101'
        self.assertEqual(self.launch('--validate-only'), 1)
        self.assertFalse(self.calls)

    def test_prefix_counts_checked_before_connectivity(self):
        self.cfg['vllm_bench_serve'].update(dataset_name='prefix_repetition', prefix_repetition_num_prefixes=2)
        self.assertEqual(self.launch('--stage', 'overload'), 1)
        self.assertFalse(self.calls)

    def test_prefix_probe_uses_one_prefix_and_scan_keeps_configured_value(self):
        self.cfg['vllm_bench_serve'].update(dataset_name='prefix_repetition', prefix_repetition_num_prefixes=2)
        self.assertEqual(self.launch('--stage', 'scan'), 0)
        for label, expected in [('connectivity', 1), ('scan-c1', 2), ('scan-c2', 2)]:
            setup = json.loads((self.output()/'tiny'/label/'setup.json').read_text())
            self.assertEqual(setup['options']['prefix_repetition_num_prefixes'], expected)

    def test_metadata_preflight(self):
        self.cfg['vllm_bench_serve']['metadata'] = ['version']
        self.assertEqual(self.launch('--validate-only'), 1)
        self.assertFalse(self.calls)

    def test_export_failure_keeps_both_artifacts_and_measurement(self):
        self.cfg['vllm_bench_serve'].update(save_result=True, result_filename='native.json')
        with patch('perf_bench_serve.export_result', side_effect=OSError('export storage failure')):
            self.assertEqual(self.launch(), 1)
        root = self.output()
        run = json.loads((root/'summary.json').read_text())['profiles']['tiny']['runs'][0]
        self.assertEqual((run['measurement_status'], run['export_status']), ('complete', 'failed'))
        self.assertTrue((root/run['report']).is_file())
        self.assertTrue((root/run['result']).is_file())
        self.assertEqual(len(self.calls), 1)
        self.check_rebuild(root)

    def check_rebuild(self, root):
        names = ('summary.json', 'vllm-report.txt', 'vllm-reports.json')
        expected = {n: (root/n).read_bytes() for n in names}
        for n in names:
            (root/n).unlink()
        with patch('perf_bench_serve.run_native', side_effect=AssertionError('unexpected traffic')):
            self.assertEqual(main(['--report-only','--output',str(root)]), 0)
        self.assertEqual(expected, {n: (root/n).read_bytes() for n in names})

    def test_freeze_deduplicates_and_preserves_original_data(self):
        self.cfg['profiles'].append(dict(name='other'))
        data = self.root/'samples.jsonl'
        data.write_text('{"prompt":"A"}\n')
        self.cfg['vllm_bench_serve'].update(dataset_name='custom', dataset_path=str(data))
        self.assertEqual(self.launch('--stage','scan'), 0)
        data.write_text('{"prompt":"B"}\n')
        root = self.output()
        inputs = json.loads((root/'setup.json').read_text())['inputs']
        self.assertEqual(len(inputs['resources']), 2)
        self.assertEqual(inputs['profiles']['tiny'], inputs['profiles']['other'])
        self.assertEqual((root/inputs['profiles']['tiny']['dataset_path']).read_text(), '{"prompt":"A"}\n')
        self.check_rebuild(root)

    def kill_and_rebuild(self, where):
        self.cfg['vllm_bench_serve'].update(save_result=True, result_filename='native.json')
        import perf_native_stages
        write = perf_native_stages.atomic_json

        def kill(*args):
            os.kill(os.getpid(), signal.SIGKILL)

        def plan_then_kill(path, value):
            if path.name == 'selection.json' and where == 'before_selection':
                kill()
            write(path, value)
            if ((where == 'before_first' and value.get('schema') == 'native_stages_v4')
                    or (where == 'after_selection' and path.name == 'selection.json')):
                kill()

        original = sys.modules['vllm.benchmarks'].serve.benchmark
        async def killed_after_response(*args):
            await original(*args)
            kill()

        hooks = {'before_selection': ('perf_native_stages.atomic_json', plan_then_kill),
                 'after_selection': ('perf_native_stages.atomic_json', plan_then_kill),
                 'before_first': ('perf_native_stages.atomic_json', plan_then_kill),
                 'export': ('perf_bench_serve.export_result', kill),
                 'publish': ('perf_native_stages.publish_reports', kill)}
        pid = os.fork()
        if pid == 0:
            try:
                hook = (patch.object(sys.modules['vllm.benchmarks'].serve, 'benchmark', killed_after_response)
                        if where == 'after_response' else patch(*hooks[where]))
                with hook:
                    self.launch('--stage', 'soak' if where.endswith('_selection') else 'connectivity')
            finally:
                os._exit(99)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), -signal.SIGKILL)
        root = self.output()
        self.assertEqual(main(['--report-only', '--output', str(root)]), 0)
        point = json.loads((root/'summary.json').read_text())['profiles']['tiny']['runs'][0]
        self.check_rebuild(root)
        return point

    def test_sigkill_after_plan_before_first_round(self):
        point = self.kill_and_rebuild('before_first')
        self.assertEqual(point['status'], 'not_started_or_unrecorded')
        self.assertNotIn('report', point)

    def test_sigkill_after_measurement_before_export(self):
        point = self.kill_and_rebuild('export')
        self.assertEqual((point['status'], point['measurement_status'], point['export_status']),
                         ('unconfirmed', 'complete', 'unconfirmed'))
        self.assertIn('report', point)
        self.assertIn('result', point)

    def test_sigkill_after_stage_before_root_report(self):
        point = self.kill_and_rebuild('publish')
        self.assertEqual((point['status'], point['measurement_status'], point['export_status']),
                         ('complete', 'complete', 'complete'))

    def test_source_mutation_during_copy_stops_before_traffic(self):
        import shutil
        copyfile = shutil.copyfile
        def change(src, dst):
            copyfile(src, dst)
            Path(src).write_text('{"changed":true}')
        with patch('perf_stage_inputs.shutil.copyfile', change):
            self.assertEqual(self.launch(), 1)
        self.assertFalse(self.calls)

    def test_declared_storage_snapshot_uses_resolved_version_path(self):
        tokenizer = Path(self.cfg['vllm_bench_serve']['tokenizer']).resolve()
        self.cfg['immutable_snapshots'] = [str(tokenizer)]
        self.assertEqual(self.launch('--stage', 'connectivity'), 0)
        plan = json.loads((self.output()/'setup.json').read_text())
        resource, = plan['inputs']['resources']
        self.assertEqual((resource['storage'], resource['path']), ('immutable_snapshot', str(tokenizer)))

    def test_external_media_must_be_pinned_before_traffic(self):
        data = self.root/'external.json'
        data.write_text(json.dumps([{'image':'https://example.invalid/image.jpg'}]))
        self.cfg['vllm_bench_serve'].update(dataset_name='sharegpt', dataset_path=str(data))
        self.assertEqual(self.launch(), 1)
        self.assertFalse(self.calls)

    def test_measurement_commit_failure_keeps_files_without_claiming_success(self):
        from run_journal import atomic_json
        def fail(path, data):
            if path.name == 'summary.json' and data.get('measurement_status') == 'complete':
                raise OSError('stage commit error')
            atomic_json(path, data)
        with patch('perf_bench_serve.atomic_json', fail):
            self.assertEqual(self.launch(), 1)
        root = self.output()
        summary = json.loads((root/'summary.json').read_text())
        point = summary['profiles']['tiny']['runs'][0]
        self.assertEqual(point['measurement_status'], 'unconfirmed')
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((root/point['result']).is_file())
        self.assertTrue((root/point['report']).is_file())
        self.check_rebuild(root)

    def test_rotate_tokenizer_alias_between_profiles(self):
        import shutil
        copy = shutil.copyfile
        first = Path(self.cfg['vllm_bench_serve']['tokenizer'])
        second = self.root/'second'
        second.mkdir()
        (second/'tokenizer_config.json').write_text('{"version":2}')
        alias = self.root/'latest'
        alias.symlink_to(first)
        self.cfg['vllm_bench_serve']['tokenizer'] = str(alias)
        self.cfg['profiles'].append(dict(name='other'))
        def rotate(src, dst):
            copy(src, dst)
            alias.unlink()
            alias.symlink_to(second)
        with patch('perf_stage_inputs.shutil.copyfile', rotate):
            self.assertEqual(self.launch('--stage','connectivity'), 0)
        root = self.output()
        inputs = json.loads((root/'setup.json').read_text())['inputs']
        self.assertEqual(len(inputs['resources']), 1)
        for binding in inputs['profiles'].values():
            self.assertEqual((root/binding['tokenizer']/'tokenizer_config.json').read_text(), '{}')

    def test_caught_interrupt_keeps_observed_response(self):
        serve = sys.modules['vllm.benchmarks'].serve
        original = serve.benchmark
        async def stop(samples):
            await original(samples)
            raise KeyboardInterrupt
        with patch.object(serve, 'benchmark', stop):
            self.assertEqual(self.launch('--stage','connectivity'), 130)
        root = self.output()
        point = json.loads((root/'summary.json').read_text())['profiles']['tiny']['runs'][0]
        self.assertEqual(point['request_count_status'], 'partial')
        self.assertEqual((point['requests']['dispatched'], point['requests']['successful']), (1, 1))
        self.check_rebuild(root)

    def test_killed_after_response_has_unknown_dispatch_and_success(self):
        point = self.kill_and_rebuild('after_response')
        self.assertEqual(point['request_count_status'], 'unknown')
        self.assertEqual(point['requests'], dict(selected=1, dispatched=None, successful=None))

    def test_rebuild_detects_changed_and_truncated_artifacts(self):
        self.assertEqual(self.launch('--stage','connectivity'), 0)
        root = self.output()
        folder = root/'tiny/connectivity'
        result = folder/'vllm-result.json'
        result.write_text('{')
        report = folder/'vllm-report.txt'
        report.write_bytes(report.read_bytes().replace(b'connectivity', b'XXXXXXXXXXXX'))
        self.assertEqual(main(['--report-only','--output',str(root)]), 0)
        summary = json.loads((root/'summary.json').read_text())
        self.assertEqual(summary['overall_status'], 'failed')
        point = summary['profiles']['tiny']['runs'][0]
        self.assertEqual(point['measurement_status'], 'complete')
        self.assertEqual(point['artifact_integrity'], dict(report='corrupt', result='corrupt'))
        self.assertNotIn('XXXXXXXXXXXX', (root/'vllm-report.txt').read_text())
        self.check_rebuild(root)

    def test_stage_directory_error_survives_rebuild(self):
        mkdir = Path.mkdir
        def fail(path, *args, **kwargs):
            if path.name == 'scan-c1':
                raise OSError('stage directory unavailable')
            return mkdir(path, *args, **kwargs)
        with patch.object(Path, 'mkdir', fail):
            self.assertEqual(self.launch('--stage','scan'), 1)
        root = self.output()
        summary = json.loads((root/'summary.json').read_text())
        self.assertEqual(summary['overall_status'], 'failed')
        error = summary['run_error']
        self.assertEqual((error['profile'], error['phase'], error['operation'], error['detail']),
                         ('tiny','scan-c1','create_stage_directory','stage directory unavailable'))
        self.check_rebuild(root)

    def test_signature_required_for_new_measurements_but_optional_for_legacy_runs(self):
        self.assertEqual(self.launch('--stage','connectivity'), 0)
        root = self.output()
        child = root/'tiny/connectivity/summary.json'
        data = json.loads(child.read_text())
        del data['artifact_signatures']
        child.write_text(json.dumps(data))
        for schema, expected, integrity in [('native_stages_v2','failed','missing_signature'),
                                            ('native_stages_v1','complete','unverified')]:
            setup = root/'setup.json'
            plan = json.loads(setup.read_text())
            plan['schema'] = schema
            setup.write_text(json.dumps(plan))
            self.assertEqual(main(['--report-only','--output',str(root)]), 0)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['overall_status'], expected)
            self.assertEqual(summary['profiles']['tiny']['runs'][0]['artifact_integrity']['report'], integrity)

    def rebuild_state(self):
        root = self.output()
        self.assertEqual(main(['--report-only', '--output', str(root)]), 0)
        self.check_rebuild(root)
        return json.loads((root/'summary.json').read_text())

    def test_swapped_scan_rounds_excluded(self):
        self.assertEqual(self.launch(), 0)
        folder = self.output()/'tiny'
        (folder/'scan-c1').rename(folder/'temp')
        (folder/'scan-c2').rename(folder/'scan-c1')
        (folder/'temp').rename(folder/'scan-c2')
        summary = self.rebuild_state()
        self.assertEqual(summary['overall_status'], 'failed')
        scans = [r for r in summary['profiles']['tiny']['runs'] if r['phase'].startswith('scan-')]
        self.assertTrue(all(r['status'] == 'round_mismatch' and 'report' not in r for r in scans))
        self.assertNotIn('[tiny/scan-', (self.output()/'vllm-report.txt').read_text())

    def test_same_round_from_other_run_is_excluded(self):
        import shutil
        self.assertEqual(self.launch('--stage','connectivity'), 0)
        root = self.output()
        other = self.root/'other'
        self.assertEqual(main(['--config',str(self.file),'--output',str(other),'--stage','connectivity']), 0)
        original = root/'tiny/connectivity'
        original.rename(root/'original')
        shutil.copytree(next(other.iterdir())/'tiny/connectivity', original)
        self.assertEqual(self.rebuild_state()['profiles']['tiny']['runs'][0]['status'], 'round_mismatch')
        self.assertEqual(json.loads((root/'vllm-reports.json').read_text()), [])

    def test_damaged_scan_never_changes_selection(self):
        self.assertEqual(self.launch(), 0)
        root = self.output()
        (root/'tiny/scan-c2/vllm-result.json').write_text('{')
        summary = self.rebuild_state()
        state = summary['profiles']['tiny']
        self.assertEqual(summary['overall_status'], 'failed')
        self.assertEqual((state['selection_status'],state['sweet_concurrency']), ('committed',2))
        self.assertEqual(state['runs'][-1]['concurrency'], 2)
        self.assertEqual(state['selection_evidence'], 'unconfirmed')

    def test_sigkill_before_selection_is_unconfirmed(self):
        self.kill_and_rebuild('before_selection')
        state = self.rebuild_state()['profiles']['tiny']
        self.assertEqual(state['selection_status'], 'unconfirmed')
        self.assertIsNone(state['sweet_concurrency'])
        self.assertFalse((self.output()/'tiny/soak').exists())

    def test_sigkill_after_selection_preserves_decision(self):
        self.kill_and_rebuild('after_selection')
        state = self.rebuild_state()['profiles']['tiny']
        self.assertEqual((state['selection_status'], state['sweet_concurrency']), ('committed',2))
        self.assertFalse((self.output()/'tiny/soak').exists())

    def test_edited_selection_blocks_soak_without_dispatch(self):
        import perf_native_stages
        commit = perf_native_stages.commit_selection
        def damage(root, plan, name):
            commit(root, plan, name)
            path = root/name/'selection.json'
            data = json.loads(path.read_text())
            data['sweet_concurrency'] = 1
            path.write_text(json.dumps(data))
        with patch.object(perf_native_stages, 'commit_selection', damage):
            self.assertEqual(self.launch('--stage','soak'), 1)
        self.assertTrue(all(not phase.endswith('/soak') for phase, _, _ in self.calls))
        self.assertEqual(self.rebuild_state()['profiles']['tiny']['selection_status'], 'invalid')

    def test_legacy_soak_preserves_actual_concurrency_without_inferred_decision(self):
        self.assertEqual(self.launch(), 0)
        root = self.output()
        plan = json.loads((root/'setup.json').read_text())
        plan['schema'] = 'native_stages_v2'
        (root/'setup.json').write_text(json.dumps(plan))
        (root/'tiny/selection.json').unlink()
        state = self.rebuild_state()['profiles']['tiny']
        self.assertEqual(state['selection_status'], 'not_recorded')
        self.assertIsNone(state['sweet_concurrency'])
        self.assertEqual(state['runs'][-1]['concurrency'], 2)

    def test_wrong_committed_count_and_soak_concurrency_are_excluded(self):
        self.assertEqual(self.launch(), 0)
        folder = self.output()/'tiny'
        for phase in ('scan-c1','soak'):
            path = folder/phase/'summary.json'
            record = json.loads(path.read_text())
            if phase == 'soak':
                record['concurrency'] = 1
            else:
                record['requests']['dispatched'] += 1
            path.write_text(json.dumps(record))
        state = self.rebuild_state()['profiles']['tiny']
        for point in state['runs']:
            if point['phase'] in ('scan-c1','soak'):
                self.assertEqual(point['status'], 'round_mismatch')
                self.assertNotIn('report', point)

    def test_official_result_controls_soak_after_summary_edit(self):
        self.soak_tpot = 200
        self.assertEqual(self.launch('--stage', 'soak'), 1)
        path = self.output()/'tiny/soak/summary.json'
        record = json.loads(path.read_text())
        record['metrics']['p95_tpot_ms'] = 1
        path.write_text(json.dumps(record))
        state = self.rebuild_state()['profiles']['tiny']
        self.assertEqual(state['soak_status'], 'failed_acceptance')
        self.assertEqual(state['runs'][-1]['metrics']['p95_tpot_ms'], 200)
        for value in (None, 1):
            if value is None:
                del record['requests']['tpot_samples']
            else:
                record['requests']['tpot_samples'] = value
            path.write_text(json.dumps(record))
            self.assertEqual(self.rebuild_state()['profiles']['tiny']['soak_status'], 'invalid_stage_record')

    def test_missing_soak_plan_entry_and_directory_cannot_pass(self):
        import shutil
        self.assertEqual(self.launch(), 0)
        path = self.output()/'setup.json'
        plan = json.loads(path.read_text())
        plan['rounds'].pop()
        path.write_text(json.dumps(plan))
        shutil.rmtree(self.output()/'tiny/soak')
        summary = self.rebuild_state()
        self.assertEqual(summary['overall_status'], 'failed')
        self.assertEqual(summary['profiles']['tiny']['soak_status'], 'not_started_or_unrecorded')

    def test_foreign_run_error_is_reported_and_ignored(self):
        self.assertEqual(self.launch('--stage', 'connectivity'), 0)
        (self.output()/'run-error.json').write_text(json.dumps(dict(run_id='another-run', status='failed')))
        state = self.rebuild_state()
        self.assertEqual(state['overall_status'], 'complete')
        self.assertIn('warnings', state)
        self.assertNotIn('run_error', state)

    def test_signal_during_final_report_persisted(self):
        from perf_native_stages import publish_reports
        attempts = []
        def interrupted(root):
            attempts.append(root)
            if len(attempts) == 1:
                signal.raise_signal(signal.SIGTERM)
            return publish_reports(root)
        with patch('perf_native_stages.publish_reports', interrupted):
            self.assertEqual(self.launch('--stage', 'connectivity'), 130)
        state = self.rebuild_state()
        self.assertEqual(state['overall_status'], 'interrupted')
        plan = json.loads((self.output()/'setup.json').read_text())
        self.assertEqual(state['run_error']['run_id'], plan['run_id'])
        self.assertEqual(state['run_error']['operation'], 'publish_reports')

    def test_static_input_constraints_apply_to_later_profiles(self):
        for invalid in (dict(dataset_name='sharegpt'), dict(trust_remote_code=True)):
            self.cfg['profiles'] = [dict(name='first'), dict(name='second', vllm_bench_serve=invalid)]
            for flags in (['--validate-only'], []):
                self.assertEqual(self.launch(*flags), 1)
                self.assertFalse(self.calls)

    def test_negative_official_tpot_keeps_traffic_and_excludes_scan(self):
        self.normal_tpot = -0.0001
        self.assertEqual(self.launch('--stage', 'soak'), 1)
        state = self.rebuild_state()['profiles']['tiny']
        self.assertEqual(state['runs'][0]['status'], 'complete')
        self.assertEqual(state['runs'][0]['metrics']['p95_tpot_ms'], self.normal_tpot)
        for scan in state['runs'][1:-1]:
            self.assertEqual(scan['status'], 'complete')
            self.assertEqual(scan['acceptance'], 'unusable_p95_tpot_ms')
        self.assertEqual(state['soak_status'], 'skipped_no_sweet_spot')

    def test_soak_with_unusable_tpot_preserves_successful_requests(self):
        self.soak_tpot = -0.0001
        self.assertEqual(self.launch('--stage', 'soak'), 1)
        state = self.rebuild_state()['profiles']['tiny']
        self.assertEqual(state['soak_status'], 'failed_acceptance')
        soak = state['runs'][-1]
        self.assertEqual((soak['measurement_status'], soak['requests']['successful'],
                          soak['acceptance']), ('complete', 7, 'unusable_p95_tpot_ms'))

    def test_random_mm_requires_chat_before_any_probe(self):
        self.cfg['profiles'].append(dict(name='bad', vllm_bench_serve=dict(dataset_name='random-mm')))
        for flags in (['--validate-only'], []):
            self.assertEqual(self.launch(*flags), 1)
            self.assertFalse(self.calls)
        self.cfg['profiles'][1]['vllm_bench_serve']['backend'] = 'openai-chat'
        self.assertEqual(self.launch('--validate-only'), 0)

    def test_invalid_leaf_does_not_hide_healthy_rounds(self):
        self.assertEqual(self.launch(), 0)
        path = self.output()/'tiny/scan-c1/summary.json'
        original = json.loads(path.read_text())
        for key, bad in [('artifacts', None), ('errors', [42]), ('overall_status', [])]:
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(original, **{key: bad})))
                summary = self.rebuild_state()
                self.assertEqual(summary['overall_status'], 'failed')
                points = summary['profiles']['tiny']['runs']
                self.assertEqual(points[1]['status'], 'invalid_stage_record')
                self.assertTrue(all(p['status'] == 'complete' for p in points if p != points[1]))

    def test_export_integrity_after_archive_move_change_and_loss(self):
        self.cfg['vllm_bench_serve'].update(save_result=True, result_dir='nested', result_filename='raw.json')
        self.assertEqual(self.launch('--stage', 'connectivity'), 0)
        root = self.root/'moved'
        self.output().rename(root)
        path = root/'tiny/connectivity/exports/nested/raw.json'
        for expected, overall in [('verified', 'complete'), ('corrupt', 'failed'), ('missing', 'failed')]:
            if expected == 'corrupt':
                path.write_text('{}')
            elif expected == 'missing':
                path.unlink()
            self.assertEqual(main(['--report-only', '--output', str(root)]), 0)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['overall_status'], overall)
            point = summary['profiles']['tiny']['runs'][0]
            self.assertEqual((point['status'], point['export_status'], point['export_integrity']),
                             ('complete', 'complete', expected))

    def test_late_profile_invalid_burstiness_prevents_all_traffic(self):
        opts = dict(burstiness=0)
        self.cfg['profiles'].append(dict(name='later', vllm_bench_serve=opts))
        for value in (0, -1):
            opts['burstiness'] = value
            for flags in (['--validate-only'], []):
                self.assertEqual(self.launch(*flags), 1)
                self.assertFalse(self.calls)
        opts['burstiness'] = 0.5
        self.assertEqual(self.launch('--validate-only'), 0)

    def check_soak_export_failure(self, expected_acceptance, expected_soak):
        from perf_bench_serve import export_result
        self.cfg['vllm_bench_serve'].update(save_result=True, result_filename='raw.json')

        def fail_soak(serve, path, result, options):
            if path.parent.parent.name == 'soak':
                raise OSError('soak export unavailable')
            return export_result(serve, path, result, options)

        with patch('perf_bench_serve.export_result', fail_soak):
            self.assertEqual(self.launch('--stage', 'soak'), 1)
        initial = json.loads((self.output()/'summary.json').read_text())
        self.assertEqual(self.rebuild_state(), initial)
        state = initial['profiles']['tiny']
        soak = state['runs'][-1]
        self.assertEqual(initial['overall_status'], 'failed')
        self.assertEqual((soak['measurement_status'], soak['requests']['successful']), ('complete', 7))
        self.assertEqual(soak['export_status'], 'failed')
        self.assertEqual((soak['acceptance'], state['soak_status']), (expected_acceptance, expected_soak))

    def test_soak_pass_survives_failed_export(self):
        self.check_soak_export_failure('passed', 'complete')

    def test_soak_slo_failure_survives_failed_export(self):
        self.soak_tpot = 1000
        self.check_soak_export_failure('slo_exceeded', 'failed_acceptance')
