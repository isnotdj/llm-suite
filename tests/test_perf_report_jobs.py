"""Reporting remains recoverable and errors are safe before persistence."""
import asyncio
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
try:
    import aiohttp
    import transformers
except ImportError:
    aiohttp=None


@unittest.skipUnless(aiohttp,'Requires performance dependencies')
class ReportJobs(unittest.IsolatedAsyncioTestCase):
    def fixture(self, root):
        from perf_vendor.endpoint_request_func import RequestFuncOutput
        folder=root/'4k';folder.mkdir()
        native=RequestFuncOutput(success=True,generated_text='OK',output_tokens=2,
                                 prompt_len=10,ttft=.01,latency=.02,itl=[.01],start_time=1)
        row=dict(sequence=0,native_output=asdict(native),native_output_origin='vllm',native_input_tokens=10)
        (folder/'scan-c1.requests.jsonl').write_text(json.dumps(row)+'\n')
        result=dict(concurrency=1,duration=1,metrics={'requests':1},requests_file='scan-c1.requests.jsonl')
        (folder/'scan-c1.json').write_text(json.dumps(result))
        summary=dict(overall_status='interrupted',profiles={'4k':{'scan':[result]}},errors=[])
        config=dict(target={'model':'fixture','api_key_env':'REPORT_TEST_KEY'},tokenizer='/unused')
        (root/'config.json').write_text(json.dumps(config))
        (root/'summary.json').write_text(json.dumps(summary))
        report=('4k','scan-c1',folder,result,1,None)
        return config,summary,report

    async def test_rebuild_refuses_an_aliased_versions_directory(self):
        from run_perf import main
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, inputs = base/'run', base/'inputs'
            root.mkdir()
            inputs.mkdir(mode=0o700)
            self.fixture(root)
            (root/'.vllm-reports').symlink_to(inputs, target_is_directory=True)
            before = inputs.stat().st_mode
            self.assertEqual(main(['--report-only','--output',str(root)]),1)
            self.assertEqual(inputs.stat().st_mode,before)
            self.assertEqual(list(inputs.iterdir()),[])

    async def test_legacy_run_inside_input_is_rejected_before_lock_or_journal(self):
        from run_perf import main
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            root = source/'legacy-run'
            root.mkdir()
            config,_,_ = self.fixture(root)
            config['tokenizer'] = str(source)
            (root/'config.json').write_text(json.dumps(config))
            before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(main(['--report-only','--output',str(root)]),1)
            after = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(after,before)

    async def test_historical_permission_repair_detaches_shared_file(self):
        from perf_report_jobs import _limit_access
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, historical, restricted = root/'original', root/'historical', root/'restricted'
            original.write_text('archived content')
            original.chmod(0o644)
            os.link(original,historical)
            restricted.touch(mode=0o600)
            _limit_access(historical,restricted)
            self.assertEqual(original.stat().st_mode & 0o777,0o644)
            self.assertEqual(historical.stat().st_mode & 0o777,0o600)
            self.assertFalse(original.samefile(historical))
            self.assertEqual(historical.read_bytes(),original.read_bytes())

    async def test_interruption_persists_jobs_without_generating_reports(self):
        from run_perf import execute
        for interruption in (asyncio.CancelledError,KeyboardInterrupt):
            with self.subTest(interruption=interruption),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);config,summary,report=self.fixture(root)
                async def interrupted(c,tokenizer,p,r,s,stage,reports):
                    reports.append(report)
                    raise interruption()
                with patch('run_perf.execute_stages',side_effect=interrupted),patch('perf_native_report.write_report') as writer:
                    with self.assertRaises(interruption): await execute(config,None,{},root,summary,'all')
                writer.assert_not_called()
                self.assertEqual(summary['reporting_status'],'pending')
                jobs=json.loads((root/'report-jobs.json').read_text())['jobs']
                self.assertEqual(jobs[0]['result_file'],'4k/scan-c1.json')
                self.assertEqual(jobs[0]['status'],'pending')
                self.assertEqual(json.loads((root/'summary.json').read_text())['reporting_status'],'pending')

    async def test_error_redacted_in_summary_and_job_manifest(self):
        from run_perf import execute
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,REPORT_TEST_KEY='dummy-report-credential'):
            root=Path(directory);config,summary,report=self.fixture(root)
            (root/'vllm-report.txt').write_text('previous complete report')
            (root/'vllm-reports.json').write_text('[]')
            async def completed(c,tokenizer,p,r,s,stage,reports): reports.append(report)
            with patch('run_perf.execute_stages',side_effect=completed),patch('perf_native_report.write_report',side_effect=RuntimeError('failed: dummy-report-credential')):
                with self.assertRaises(RuntimeError): await execute(config,None,{},root,summary,'all')
            for name in ('summary.json','report-jobs.json'):
                text=(root/name).read_text()
                self.assertNotIn('dummy-report-credential',text)
                self.assertIn('[REDACTED]',text)
            self.assertEqual((root/'vllm-report.txt').read_text(),'previous complete report')

    async def test_offline_retry_recovers_unqueued_stage_and_is_idempotent(self):
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,REPORT_TEST_KEY=''):
            root=Path(directory);self.fixture(root)
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=None),patch('perf_transport.Transport.__call__',side_effect=AssertionError('No API requests allowed')),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--report-only','--output',str(root)]),0)
                first=(root/'vllm-report.txt').read_text()
                self.assertEqual(main(['--report-only','--output',str(root)]),0)
            self.assertEqual((root/'vllm-report.txt').read_text(),first)
            self.assertEqual(first.count('Serving Benchmark Result'),1)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['overall_status'],'interrupted')
            self.assertEqual(summary['reporting_status'],'complete')
            self.assertEqual(len(summary['vllm_reports']),1)
            self.assertEqual(json.loads((root/'report-jobs.json').read_text())['jobs'][0]['status'],'complete')

    async def test_cli_failure_is_redacted_and_can_be_retried(self):
        from run_perf import main
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,REPORT_TEST_KEY='dummy-report-credential'):
            root=Path(directory);self.fixture(root)
            saved=json.loads((root/'summary.json').read_text())
            saved['errors']=[dict(phase='reporting',detail='old failure: dummy-report-credential')]
            (root/'summary.json').write_text(json.dumps(saved))
            output=io.StringIO()
            with patch('perf_native_report.write_report',side_effect=RuntimeError('dummy-report-credential')),redirect_stderr(output):
                self.assertEqual(main(['--report-only','--output',str(root)]),1)
            self.assertNotIn('dummy-report-credential',output.getvalue())
            self.assertNotIn('dummy-report-credential',(root/'summary.json').read_text())
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=None),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--report-only','--output',str(root)]),0)

    async def test_failed_retry_preserves_existing_report_and_index(self):
        from perf_native_report import write_report
        from perf_report_jobs import generate_reports, save_jobs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,summary,report=self.fixture(root)
            second=dict(report[3],concurrency=2)
            (root/'4k/scan-c2.json').write_text(json.dumps(second))
            manifest=save_jobs(root,discover=True)
            old_index=[dict(profile='4k',phase='previous')]
            (root/'vllm-reports.json').write_text(json.dumps(old_index))
            (root/'vllm-report.txt').write_text('previous complete report')
            count=0
            def fail_second(*args,**kwargs):
                nonlocal count
                count+=1
                if count==2: raise RuntimeError('transient report failure')
                return write_report(*args,**kwargs)
            with patch('perf_native_report.write_report',side_effect=fail_second),redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError): generate_reports(root,config,summary,None,manifest)
            self.assertEqual(count,2)
            self.assertEqual((root/'vllm-report.txt').read_text(),'previous complete report')
            self.assertEqual(json.loads((root/'vllm-reports.json').read_text()),old_index)
            self.assertEqual(summary['vllm_reports'],old_index)

    async def test_interrupt_before_and_after_version_switch(self):
        import perf_report_jobs as jobs
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            config,summary,_=self.fixture(root)
            manifest=jobs.save_jobs(root,discover=True)
            def generate():
                jobs.generate_reports(root,config,summary,None,manifest)
            generate()
            published=root/'vllm-reports.json'
            old=published.read_bytes()
            entry=json.loads(old)[0]
            originals={root/entry[key]:(root/entry[key]).read_bytes() for key in ('text','json')}
            old_text=(root/'vllm-report.txt').read_bytes()
            dumps=json.dumps
            def interrupt_stream(value,*args,**kwargs):
                if value=='input_lens':
                    raise KeyboardInterrupt()
                return dumps(value,*args,**kwargs)
            with patch('perf_native_report.json.dumps',side_effect=interrupt_stream):
                with self.assertRaises(KeyboardInterrupt):
                    generate()
            self.assertEqual(published.read_bytes(),old)
            self.assertEqual((root/'vllm-report.txt').read_bytes(),old_text)
            self.assertEqual(len(list((root/'.vllm-reports').glob('run-*'))),1)

            link=jobs._link
            def interrupt_after_switch(path,target):
                link(path,target)
                if path.name=='current':
                    raise KeyboardInterrupt()
            with patch.object(jobs,'_link',side_effect=interrupt_after_switch):
                with self.assertRaises(KeyboardInterrupt):
                    generate()
            new=published.read_bytes()
            self.assertNotEqual(new,old)
            new_entry=json.loads(new)[0]
            for key in ('text','json'):
                versioned=root/new_entry[key]
                self.assertTrue(versioned.samefile(root/'4k'/versioned.name))
                self.assertEqual(originals[root/entry[key]],(root/entry[key]).read_bytes())
            self.assertEqual(json.loads((root/new_entry['json']).read_bytes())['generated_texts'],['OK'])
            generate()
            self.assertEqual(summary['reporting_status'],'complete')

    async def test_legacy_index_and_reports_survive_failed_migration(self):
        import perf_report_jobs as jobs
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            config,summary,_=self.fixture(root)
            manifest=jobs.save_jobs(root,discover=True)
            entry=dict(profile='4k',phase='scan-c1',text='4k/scan-c1.vllm.txt',json='4k/scan-c1.vllm.json')
            (root/entry['text']).write_text('legacy text')
            (root/entry['json']).write_text('{"legacy": true}')
            published=root/'vllm-reports.json'
            published.write_text(json.dumps([entry]))
            link=jobs._link
            def interrupt_before_index(path,target):
                if path==published.resolve():
                    raise KeyboardInterrupt()
                link(path,target)
            with patch.object(jobs,'_link',side_effect=interrupt_before_index):
                with self.assertRaises(KeyboardInterrupt):
                    jobs.generate_reports(root,config,summary,None,manifest)
            self.assertEqual(json.loads(published.read_text()),[entry])
            self.assertEqual((root/entry['text']).read_text(),'legacy text')
            self.assertEqual(json.loads((root/entry['json']).read_text()),{'legacy':True})
            jobs.generate_reports(root,config,summary,None,manifest)
            new=json.loads(published.read_text())[0]
            self.assertTrue((root/new['json']).samefile(root/entry['json']))

    async def test_busy_output_rejects_offline_rebuild_before_journal_changes(self):
        from perf_report_jobs import output_lock, rebuild_reports
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            previous=(root/'summary.json').read_bytes()
            with output_lock(root),self.assertRaisesRegex(RuntimeError,'already in use'):
                rebuild_reports(root)
            self.assertEqual((root/'summary.json').read_bytes(),previous)
            self.assertFalse((root/'report-jobs.json').exists())

    async def test_every_legacy_publication_step_is_recoverable(self):
        import perf_report_jobs as jobs
        # Bootstrap current, wire the four public files, then commit current.
        for boundary in range(1,7):
            with self.subTest(boundary=boundary),tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
                root=Path(directory).resolve()
                config,summary,_=self.fixture(root)
                manifest=jobs.save_jobs(root,discover=True)
                entry=dict(profile='4k',phase='scan-c1',text='4k/scan-c1.vllm.txt',json='4k/scan-c1.vllm.json')
                public=root/'vllm-reports.json'
                public.write_text(json.dumps([entry]))
                (root/'vllm-report.txt').write_text('old aggregate')
                for key in ('text','json'):
                    (root/entry[key]).write_text('old '+key)
                link=jobs._link
                steps=[]
                def interrupt(path,target):
                    link(path,target)
                    steps.append(path)
                    if len(steps)==boundary:
                        raise KeyboardInterrupt()
                with patch.object(jobs,'_link',side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        jobs.generate_reports(root,config,summary,None,manifest)
                indexed=json.loads(public.read_text())[0]
                for key in ('text','json'):
                    data=(root/indexed[key]).read_bytes()
                    self.assertEqual(data,(root/entry[key]).read_bytes())
                    self.assertEqual(data==('old '+key).encode(),boundary<6)
                aggregate=(root/'vllm-report.txt').read_text()
                self.assertEqual(aggregate=='old aggregate',boundary<6)
                jobs.generate_reports(root,config,summary,None,manifest)
                self.assertEqual(summary['reporting_status'],'complete')
                if indexed['text'].startswith('.vllm-reports/legacy-'):
                    self.assertEqual((root/indexed['text']).read_text(),'old text')

    async def test_published_permissions_follow_output_and_request_permissions(self):
        import perf_report_jobs as jobs
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            previous=os.umask(0o022)
            try:
                root.chmod(0o755)
                config,summary,_=self.fixture(root)
                requests=root/'4k/scan-c1.requests.jsonl'
                requests.chmod(0o640)
                (root/'4k').chmod(0o750)
                manifest=jobs.save_jobs(root,discover=True)
                jobs.generate_reports(root,config,summary,None,manifest)
                index=root/'vllm-reports.json'
                generation=index.resolve().parent
                self.assertEqual(generation.stat().st_mode & 0o777,0o755)
                self.assertEqual(index.stat().st_mode & 0o777,0o644)
                self.assertEqual((root/'vllm-report.txt').stat().st_mode & 0o777,0o644)
                for key in ('text','json'):
                    stage=root/json.loads(index.read_text())[0][key]
                    self.assertEqual(stage.parent.stat().st_mode & 0o777,0o750)
                    self.assertEqual(stage.stat().st_mode & 0o777,0o640)
            finally:
                os.umask(previous)

    @unittest.skipUnless(os.geteuid()==0,'Requires root to switch to an unprivileged reader')
    async def test_unprivileged_host_user_can_read_container_reports(self):
        import subprocess
        import perf_report_jobs as jobs
        with tempfile.TemporaryDirectory(dir='/tmp') as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            root.chmod(0o755)
            previous=os.umask(0o022)
            try:
                config,summary,_=self.fixture(root)
                folder=root/'4k'
                requests=folder/'scan-c1.requests.jsonl'
                os.chown(folder,-1,65533)
                folder.chmod(0o750)
                os.chown(requests,-1,65533)
                requests.chmod(0o640)
                jobs.generate_reports(root,config,summary,None,jobs.save_jobs(root,discover=True))
            finally:
                os.umask(previous)
            def read(path,groups=()):
                if sys.platform.startswith('linux'):
                    membership = '--groups='+','.join(map(str,groups)) if groups else '--clear-groups'
                    return subprocess.check_output(['setpriv','--reuid=65534','--regid=65534',membership,
                                                    '/bin/cat',path],stderr=subprocess.PIPE)
                return subprocess.check_output(['/bin/cat',path],user=65534,group=65534,
                                               extra_groups=groups,stderr=subprocess.PIPE)
            entry=json.loads(read(root/'vllm-reports.json'))[0]
            self.assertIn(b'Serving Benchmark Result',read(root/'vllm-report.txt'))
            self.assertIn(b'Serving Benchmark Result',read(root/entry['text'],[65533]))
            self.assertIn('generated_texts',json.loads(read(root/entry['json'],[65533])))
            with self.assertRaises(subprocess.CalledProcessError):
                read(root/entry['json'])

    async def test_rebuild_preserves_private_metadata_and_request_group(self):
        import perf_report_jobs as jobs
        groups = [gid for gid in os.getgroups() if gid != os.getegid()]
        if not groups and os.geteuid() != 0:
            self.skipTest('Requires a second assignable group')
        group = groups[0] if groups else 65534
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            config,summary,_=self.fixture(root)
            stage=root/'4k/scan-c1.json'
            requests=stage.with_suffix('.requests.jsonl')
            manifest=jobs.save_jobs(root,discover=True)
            private=[root/'summary.json',root/'report-jobs.json',stage]
            for path in [stage.parent,requests,*private]:
                os.chown(path,-1,group)
            for path in private:
                path.chmod(0o600)
            stage.parent.chmod(0o750)
            requests.chmod(0o640)
            previous=os.umask(0o022)
            try:
                historical={}
                for attempt in range(2):
                    jobs.generate_reports(root,config,summary,None,manifest)
                    for path in [*private,root/'vllm-report.txt',root/'vllm-reports.json']:
                        info=path.stat()
                        self.assertEqual((info.st_uid,info.st_gid,info.st_mode & 0o777),
                                         (os.geteuid(),group,0o600),path)
                    report=summary['vllm_reports'][0]
                    for key in ('text','json'):
                        output=root/report[key]
                        self.assertEqual(output.parent.stat().st_gid,group)
                        self.assertEqual(output.stat().st_gid,group)
                        self.assertEqual(output.stat().st_mode & 0o777,0o640)
                        # Simulate an already published pre-fix version.
                        if attempt==0:
                            os.chown(output,-1,os.getegid())
                            historical[output]=output.read_bytes()
                for path,contents in historical.items():
                    self.assertEqual(path.stat().st_gid,group)
                    self.assertEqual(path.read_bytes(),contents)
            finally:
                os.umask(previous)

    async def test_atomic_replace_permission_failure_leaves_private_original(self):
        from run_journal import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            path.write_text('{"previous":true}')
            path.chmod(0o600)
            def fail(temporary,source,mode):
                self.assertEqual(temporary.stat().st_mode & 0o077,0)
                raise PermissionError('ownership not assignable')
            with patch('run_journal.apply_access',side_effect=fail):
                with self.assertRaises(PermissionError):
                    atomic_json(path,{'new':True})
            self.assertEqual(path.read_text(),'{"previous":true}')
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            self.assertEqual(list(path.parent.iterdir()),[path])

    async def test_sealed_log_rejects_damage_and_rebuilds_without_nas(self):
        import hashlib
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            self.fixture(root)
            path=root/'4k/scan-c1.requests.jsonl'
            row=json.loads(path.read_bytes())
            row['native_output']['output_tokens']=0
            row.update(native_report_tokens=2,native_report_token_source='tokenizer')
            lines=[json.dumps(dict(row,sequence=n)).encode()+b'\n' for n in (1,0)]
            original=b''.join(lines)
            path.write_bytes(original)
            stage=root/'4k/scan-c1.json'
            data=json.loads(stage.read_bytes())
            data.update(metrics={'requests':2},report_contract=dict(version=1,token_count_version=1,
                        requests_sha256=hashlib.sha256(original).hexdigest()))
            stage.write_text(json.dumps(data))
            cmd=['--report-only','--output',str(root)]
            with patch('transformers.AutoTokenizer.from_pretrained',side_effect=AssertionError('unexpected NAS access')):
                self.assertEqual(main(cmd),0)
                published=root/'vllm-reports.json'
                before=published.read_bytes()
                old={root/json.loads(before)[0][k] for k in ('text','json')}
                files={f:f.read_bytes() for f in old}
                for damaged in (lines[1],lines[0]+lines[0],original.replace(b'OK',b'NO'),
                                original.replace(b'"sequence": 0',b'"sequence": 2')):
                    path.write_bytes(damaged)
                    self.assertEqual(main(cmd),1)
                    self.assertEqual(published.read_bytes(),before)
                    self.assertTrue(all(f.read_bytes()==b for f,b in files.items()))
                path.write_bytes(original)
                self.assertEqual(main(cmd),0)
                report=json.loads((root/'4k/scan-c1.vllm.json').read_bytes())
                self.assertEqual(report['total_output_tokens'],4)
                self.assertTrue(report['vllm_suite_metadata']['sha256_verified'])

    async def test_legacy_fallback_requires_matching_fingerprint(self):
        from perf_resources import tokenizer_hashes
        from run_perf import main
        from test_perf import Tokenizer
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            config,_,_=self.fixture(root)
            tok=root/'tokenizer'
            tok.mkdir()
            vocab=tok/'vocab.txt'
            vocab.write_text('OK')
            config['tokenizer']=str(tok)
            (root/'config.json').write_text(json.dumps(config))
            path=root/'4k/scan-c1.requests.jsonl'
            row=json.loads(path.read_bytes())
            row['native_output']['output_tokens']=0
            path.write_text(json.dumps(row)+'\n')
            cmd=['--report-only','--output',str(root)]
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()) as loader:
                self.assertEqual(main(cmd),1)
                loader.assert_not_called()
                (root/'setup.json').write_text(json.dumps({'tokenizer_hashes':tokenizer_hashes(tok,Tokenizer())}))
                self.assertEqual(main(cmd),0)
                loader.assert_called_once()
                report=json.loads((root/'4k/scan-c1.vllm.json').read_bytes())
                self.assertFalse(report['vllm_suite_metadata']['sha256_verified'])
                self.assertTrue(report['vllm_suite_metadata']['sequence_verified'])
                vocab.write_text('changed')
                previous=(root/'vllm-reports.json').read_bytes()
                self.assertEqual(main(cmd),1)
                self.assertEqual((root/'vllm-reports.json').read_bytes(),previous)

    @unittest.skipUnless(sys.platform.startswith('linux') and os.geteuid()==0,
                         'Requires Linux with independent reader identities')
    async def test_private_inputs_and_all_public_report_paths(self):
        import subprocess
        from perf_report_jobs import generate_reports,save_jobs
        from run_inputs import restrict_run_readers,freeze_tasks
        with tempfile.TemporaryDirectory(dir='/tmp') as directory,redirect_stdout(io.StringIO()):
            base=Path(directory)
            base.chmod(0o755)
            source=base/'dataset.jsonl'
            source.write_text('question')
            os.chown(source,0,65533)
            source.chmod(0o640)
            root=base/'output'
            root.mkdir(mode=0o755)
            restrict_run_readers(root,[source])
            config,summary,_=self.fixture(root)
            task=dict(config={'abbr':'tiny'},ids=[1],prompts=[['question']],references=['reference'])
            freeze_tasks(root,'accuracy',[task])
            manifest=save_jobs(root,discover=True)
            generate_reports(root,config,summary,None,manifest)
            index=summary['vllm_reports'][0]
            files=[Path(task['input_path']),root/'4k/scan-c1.requests.jsonl',root/'vllm-reports.json',
                   root/'vllm-report.txt',root/'4k/scan-c1.vllm.json',root/index['json'],root/index['text']]
            for file in files:
                for group,allowed in ((65533,True),(65534,False)):
                    result=subprocess.run(['setpriv','--reuid=65534',f'--regid={group}','--clear-groups','/bin/cat',file],
                                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    self.assertEqual(result.returncode==0,allowed,(file,group))
            second=base/'other.jsonl'
            second.write_text('private')
            os.chown(second,0,65532)
            second.chmod(0o640)
            restrict_run_readers(root,[source,second])
            self.assertEqual(root.stat().st_mode & 0o777,0o700)

    async def test_request_log_without_stage_record_is_never_published(self):
        from run_journal import atomic_json
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            self.fixture(root)
            stage=root/'4k/scan-c1.json'
            completion=json.loads(stage.read_bytes())
            stage.unlink()  # Simulate interruption after log close, before stage commit.
            cmd=['--report-only','--output',str(root)]
            self.assertEqual(main(cmd),1)
            self.assertFalse((root/'vllm-reports.json').exists())
            atomic_json(stage,completion)  # Crash before queue insertion.
            self.assertEqual(main(cmd),0)
            jobs=json.loads((root/'report-jobs.json').read_bytes())['jobs']
            self.assertEqual([(j['phase'],j['status']) for j in jobs],[('scan-c1','complete')])

    async def test_new_runs_and_new_rows_cannot_downgrade_to_legacy(self):
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            self.fixture(root)
            cmd=['--report-only','--output',str(root)]
            self.assertEqual(main(cmd),0)
            index=root/'vllm-reports.json'
            published=index.read_bytes()
            stage=root/'4k/scan-c1.json'
            requests=stage.with_suffix('.requests.jsonl')
            row=json.loads(requests.read_text())
            row['native_output']['generated_text']='NO'
            row.update(native_report_tokens=2,native_report_token_source='usage')
            requests.write_text(json.dumps(row)+'\n')
            self.assertEqual(main(cmd),1)
            self.assertEqual(index.read_bytes(),published)
            # The run-level contract also protects empty stages.
            data=json.loads(stage.read_text())
            data['metrics']['requests']=0
            stage.write_text(json.dumps(data))
            requests.write_text('')
            summary=root/'summary.json'
            summary.write_text(json.dumps(dict(json.loads(summary.read_text()),report_contract_version=1)))
            self.assertEqual(main(cmd),1)
            self.assertEqual(index.read_bytes(),published)
            self.assertEqual(json.loads((root/'4k/scan-c1.vllm.json').read_text())['generated_texts'],['OK'])

    async def test_corrupt_stage_records_failure_before_and_after_queue_insertion(self):
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            self.fixture(root)
            cmd=['--report-only','--output',str(root)]
            self.assertEqual(main(cmd),0)
            index=root/'vllm-reports.json'
            previous=index.read_bytes()
            stage=root/'4k/scan-c1.json'
            good=stage.read_bytes()
            stage.write_text('{')
            manifest=root/'report-jobs.json'
            for queued in (True,False):
                if not queued:
                    manifest.unlink()
                self.assertEqual(main(cmd),1)
                self.assertEqual(index.read_bytes(),previous)
                summary=json.loads((root/'summary.json').read_text())
                job,=json.loads(manifest.read_text())['jobs']
                self.assertEqual(summary['reporting_status'],'failed')
                self.assertEqual(job['status'],'failed')
                self.assertIn('4k/scan-c1.json',job['error'])
                self.assertIn('4k/scan-c1.json',summary['errors'][-1]['detail'])
            stage.write_bytes(good)
            self.assertEqual(main(cmd),0)

    async def test_versioned_fast_tokenizer_file_must_have_recorded_digest(self):
        from tokenizers import Tokenizer,models,pre_tokenizers
        from transformers import AutoTokenizer,PreTrainedTokenizerFast
        from perf_resources import tokenizer_hashes
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            config,_,_=self.fixture(root)
            tok=root/'tokenizer'
            raw=Tokenizer(models.WordLevel({'[UNK]':0,'OK':1,'O':2,'K':3},unk_token='[UNK]'))
            raw.pre_tokenizer=pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]').save_pretrained(tok)
            filename='tokenizer.4.0.0.json'
            raw.save(str(tok/filename))
            cfg=tok/'tokenizer_config.json'
            cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()),fast_tokenizer_files=[filename])))
            config['tokenizer']=str(tok)
            (root/'config.json').write_text(json.dumps(config))
            path=root/'4k/scan-c1.requests.jsonl'
            row=json.loads(path.read_text())
            row['native_output']['output_tokens']=0
            path.write_text(json.dumps(row)+'\n')
            hashes=tokenizer_hashes(tok,AutoTokenizer.from_pretrained(tok,local_files_only=True))
            self.assertIn(filename,hashes)
            setup=root/'setup.json'
            setup.write_text(json.dumps({'tokenizer_hashes':hashes}))
            cmd=['--report-only','--output',str(root)]
            self.assertEqual(main(cmd),0)
            index=root/'vllm-reports.json'
            published=index.read_bytes()
            raw.pre_tokenizer=pre_tokenizers.Split('',behavior='isolated')
            raw.save(str(tok/filename))
            self.assertEqual(main(cmd),1)
            del hashes[filename]
            setup.write_text(json.dumps({'tokenizer_hashes':hashes}))
            self.assertEqual(main(cmd),1)
            self.assertEqual(index.read_bytes(),published)
            self.assertEqual(json.loads((root/'4k/scan-c1.vllm.json').read_text())['total_output_tokens'],1)

    async def test_unqueued_published_stage_cannot_disappear(self):
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            self.fixture(root)
            folder=root/'4k'
            stage=folder/'soak.json'
            result=json.loads((folder/'scan-c1.json').read_bytes())
            result['requests_file']='soak.requests.jsonl'
            stage.write_text(json.dumps(result))
            (folder/result['requests_file']).write_bytes((folder/'scan-c1.requests.jsonl').read_bytes())
            cmd=['--report-only','--output',str(root)]
            self.assertEqual(main(cmd),0)
            index=root/'vllm-reports.json'
            before=index.read_bytes()
            payloads={root/e[k]:(root/e[k]).read_bytes() for e in json.loads(before) for k in ('text','json')}
            stage.unlink()
            (root/'report-jobs.json').unlink()
            summary=root/'summary.json'
            previous=json.loads(summary.read_bytes())
            previous.pop('vllm_reports')
            summary.write_text(json.dumps(previous))
            self.assertEqual(main(cmd),1)
            self.assertEqual(index.read_bytes(),before)
            self.assertTrue(all(path.read_bytes()==value for path,value in payloads.items()))
            failed=json.loads(summary.read_bytes())
            self.assertEqual(failed['reporting_status'],'failed')
            self.assertIn('4k/soak.json',failed['errors'][-1]['detail'])
            stage.write_text(json.dumps(result))
            self.assertEqual(main(cmd),0)
            self.assertEqual({e['phase'] for e in json.loads(index.read_bytes())},{'scan-c1','soak'})

    async def test_explicit_backend_override_must_be_verified(self):
        from tokenizers import Tokenizer,models,pre_tokenizers
        from transformers import AutoTokenizer,PreTrainedTokenizerFast
        from perf_resources import tokenizer_hashes
        from run_perf import main
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            config,_,_=self.fixture(root)
            tok=root/'tokenizer'
            raw=Tokenizer(models.WordLevel({'[UNK]':0,'OK':1,'O':2,'K':3},unk_token='[UNK]'))
            raw.pre_tokenizer=pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]').save_pretrained(tok)
            override=tok/'active.json'
            raw.save(str(override))
            cfg=tok/'tokenizer_config.json'
            cfg.write_text(json.dumps(dict(json.loads(cfg.read_bytes()),tokenizer_file=str(override))))
            config['tokenizer']=str(tok)
            (root/'config.json').write_text(json.dumps(config))
            path=root/'4k/scan-c1.requests.jsonl'
            row=json.loads(path.read_bytes())
            row['native_output']['output_tokens']=0
            path.write_text(json.dumps(row)+'\n')
            hashes=tokenizer_hashes(tok,AutoTokenizer.from_pretrained(tok,local_files_only=True))
            self.assertIn('active.json',hashes)
            setup=root/'setup.json'
            setup.write_text(json.dumps({'tokenizer_hashes':hashes}))
            cmd=['--report-only','--output',str(root)]
            self.assertEqual(main(cmd),0)
            index=root/'vllm-reports.json'
            published=index.read_bytes()
            raw.pre_tokenizer=pre_tokenizers.Split('',behavior='isolated')
            raw.save(str(override))
            self.assertEqual(main(cmd),1)
            del hashes['active.json']
            setup.write_text(json.dumps({'tokenizer_hashes':hashes}))
            self.assertEqual(main(cmd),1)
            self.assertEqual(index.read_bytes(),published)
            self.assertEqual(json.loads((root/'4k/scan-c1.vllm.json').read_bytes())['total_output_tokens'],1)
