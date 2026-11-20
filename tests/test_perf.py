import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from perf_config import normalize
from perf_data import prepare, messages_for
from perf_metrics import metrics, select


def config():
    return normalize(dict(target=dict(base_url='http://localhost/v1',model='test',api_key_env='PERF_TEST_KEY'),
        dataset={'name':'random'},tokenizer='/unused',context_window=1000,max_generation_tokens=100,
        profiles=[dict(name='tiny',input_tokens=100,output_target=None)],sample_count=4,min_samples=2))


class Tokenizer:
    vocab_size=128
    all_special_ids=[]
    def decode(self,ids,**kwargs): return ''.join(chr(65+i%26) for i in ids)
    def encode(self,text,**kwargs): return list(text)
    def apply_chat_template(self,messages,**kwargs): return list(''.join(m['content'] for m in messages))+[0]*3


class Contracts(unittest.TestCase):
    def test_defaults_and_slo_override(self):
        c=config()
        self.assertEqual(c['profiles'][0]['slo'],{'ttft_ms':2000,'tpot_ms':50})
        c['profiles'][0]['slo']={'ttft_ms':3000}
        self.assertEqual(normalize(c)['profiles'][0]['slo']['tpot_ms'],50)
    def test_tokenizer_config_is_a_local_directory_path(self):
        cfg=config()
        cfg['tokenizer']='/nas/modelscope/models/Qwen--demo/snapshots/v1'
        self.assertEqual(normalize(cfg)['tokenizer'],cfg['tokenizer'])
        cfg['tokenizer']={'modelscope_id':'Qwen/demo','cache_dir':'/nas/modelscope'}
        with self.assertRaisesRegex(ValueError,'directory path'):
            normalize(cfg)

    def test_template_change_after_preparation(self):
        from unittest.mock import patch
        from run_perf import main
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root/'template.jinja'
            template.write_text('original')
            cfg = config()
            (root/'tokenizer').mkdir()
            cfg.update(tokenizer=str(root/'tokenizer'), chat_template=str(template))
            config_path = root/'config.json'
            config_path.write_text(json.dumps(cfg))

            def prepare_and_replace(config, loaded, output):
                self.assertIs(loaded, tokenizer)
                self.assertEqual(loaded.chat_template, 'original')
                data = prepare(config, loaded, output)
                template.write_text('replacement')
                return data

            async def probe(transport, sample):
                self.assertIs(transport.tokenizer, tokenizer)
                self.assertEqual(transport.tokenizer.chat_template, 'original')
                return {'success': True, 'input_token_delta': None}

            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer) as loader, \
                    patch('run_perf.prepare', side_effect=prepare_and_replace), \
                    patch('perf_transport.Transport.__call__', probe), \
                    patch.dict(os.environ, PERF_TEST_KEY='fixture'):
                code = main(['--config', str(config_path), '--output', str(root/'out'),
                             '--stage', 'connectivity'])
            self.assertEqual(code, 0)
            loader.assert_called_once()
            setup = json.loads((next((root/'out').iterdir())/'setup.json').read_text())
            self.assertEqual(setup['chat_template'], 'original')

    def test_fixed_config_fields_and_seed(self):
        for field, value, message in [
                ('timing', {'scan_request': 2}, 'timing.scan_request'),
                ('slo', {'tttf_ms': 1}, 'slo.tttf_ms'),
                ('seed', None, 'seed'), ('seed', True, 'seed'),
                ('seed', 3.5, 'seed'), ('seed', '42', 'seed')]:
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError,message):
                normalize({**config(),field:value})
        c = config()
        c['target']['extra_body'] = {'chat_template': 'conflicting'}
        with self.assertRaisesRegex(ValueError, 'target.extra_body.chat_template'):
            normalize(c)

    def test_dataset_snapshot_matches_sampled_bytes(self):
        from perf_resources import freeze_dataset, digest
        from perf_data import records
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'nas.jsonl'
            source.write_text('{"prompt": "original"}\n')
            expected = digest(source)
            frozen, checksum = freeze_dataset(source,root)
            source.write_text('{"prompt": "updated"}\n')
            self.assertEqual(list(records(frozen)), [{'prompt': 'original'}])
            self.assertEqual(checksum,expected)
            self.assertEqual(digest(frozen),expected)

    def test_changed_dataset_rejects_snapshot(self):
        from perf_resources import freeze_dataset
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'nas.jsonl'
            source.write_text('original')
            original_stat = Path.stat
            def change_before_final_check(path, *args, **kwargs):
                if path == source:
                    source.write_text('replacement bytes')
                return original_stat(path, *args, **kwargs)
            with patch.object(Path,'stat',change_before_final_check), self.assertRaisesRegex(ValueError,'changed'):
                freeze_dataset(source,root)
            self.assertEqual(list((root/'inputs/source').iterdir()), [])

    def test_protected_fields(self):
        for key in ('ignore_eos','max_tokens','min_tokens','stop'):
            c=config();c['target']['extra_body']={key:1}
            with self.assertRaises(ValueError): normalize(c)
    def test_random_length_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            x,_=prepare(config(),Tokenizer(),Path(a)); y,_=prepare(config(),Tokenizer(),Path(b))
            self.assertEqual(x['tiny']['sha256'],y['tiny']['sha256'])
            self.assertTrue(all(90<=s['input_tokens']<=110 for s in x['tiny']['samples']))
    def test_context_rejected(self):
        c=config();c['context_window']=150
        with tempfile.TemporaryDirectory() as a, self.assertRaises(ValueError): prepare(c,Tokenizer(),Path(a))
    def test_sharegpt_no_answer_leak(self):
        row={'conversations':[{'from':'human','value':'Q'},{'from':'gpt','value':'ANSWER'}]}
        self.assertEqual(messages_for(row,'sharegpt'),[{'role':'user','content':'Q'}])
    def test_insufficient_samples(self):
        with tempfile.TemporaryDirectory() as a:
            root=Path(a);path=root/'data.jsonl';path.write_text('{"prompt":"short"}\n')
            c=config();c['dataset']={'name':'longcodebench','path':str(path)}
            prepared,unavailable=prepare(c,Tokenizer(),root)
            self.assertFalse(prepared);self.assertIn('tiny',unavailable)
    def test_disk_metrics_equivalence(self):
        from perf_metrics import MetricStore
        rows=[dict(sample_id=str(i%3),success=i%4!=0,truncated=i==4,ttft=.1*i,
            tpot=None if i==0 else .01*i,latency=.2*i,input_tokens=100,output_tokens=i,
            token_source='usage',http_status=200) for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            store=MetricStore(Path(directory)/'metrics.sqlite')
            for row in rows: store.append(row)
            self.assertEqual(store.summary(10),metrics(rows,10))
            store.close()

    def test_single_choice_only(self):
        c=config();c['target']['extra_body']={'n':2}
        with self.assertRaisesRegex(ValueError,'n=1'): normalize(c)
        c['target']['extra_body']={'n':1}
        normalize(c)

    def test_template_kwargs_forwarded(self):
        from perf_data import token_count
        class Template:
            def apply_chat_template(self,messages,**kwargs):
                return [0]* (100 if kwargs.get('enable_thinking') else 10)
        self.assertEqual(token_count(Template(),[],{'enable_thinking':True}),100)
        self.assertEqual(token_count(Template(),[],{'enable_thinking':False}),10)

    def test_bottleneck_not_selected(self):
        m=metrics([dict(sample_id='1',success=True,truncated=False,ttft=.1,tpot=.01,
            latency=.2,input_tokens=100,output_tokens=10,http_status=200)],1)
        points=[dict(concurrency=n,metrics=m,sufficient=True,client_bottleneck_suspected=True) for n in (1,2)]
        self.assertEqual(select(points,config()['slo']),dict(sweet_concurrency=None,saturation_concurrency=None))

    def test_overload_rejections_fail_acceptance(self):
        from perf_metrics import overload_verdict
        r=dict(sample_id='1',success=True,truncated=False,ttft=.1,tpot=.01,
            latency=.2,input_tokens=100,output_tokens=10,http_status=200)
        result=dict(metrics=metrics([r,dict(r,success=False,http_status=429)],1),
            sufficient=True,recovered=True,aborted=None,client_bottleneck_suspected=False)
        verdict=overload_verdict(result)
        self.assertFalse(verdict['passed']);self.assertTrue(verdict['rejection_observed'])
        result['metrics']=metrics([r],1)
        self.assertTrue(overload_verdict(result)['passed'])
        result['recovered']=False
        self.assertFalse(overload_verdict(result)['passed'])

    def test_stream_timeout_is_guard_failure(self):
        from perf_engine import guard_failure
        self.assertTrue(guard_failure(dict(success=False,http_status=200,error_kind='timeout')))
        self.assertFalse(guard_failure(dict(success=False,http_status=429,error_kind='response')))
        self.assertFalse(guard_failure(dict(success=True,http_status=200,error_kind=None)))

    def test_input_truncation_rejected(self):
        c=config();c['target']['extra_body']={'truncate_prompt_tokens':100}
        with self.assertRaises(ValueError): normalize(c)

    def test_mismatch_excludes_results(self):
        from perf_metrics import qualifies, overload_verdict
        row=dict(sample_id='1',success=True,truncated=False,ttft=.1,tpot=.01,latency=.2,
            input_tokens=10,output_tokens=10,http_status=200,input_token_delta=-90)
        m=metrics([row],1)
        self.assertFalse(qualifies(m,config()['slo']))
        points=[dict(concurrency=n,metrics=m,sufficient=True) for n in (1,2)]
        self.assertEqual(select(points,config()['slo']),dict(sweet_concurrency=None,saturation_concurrency=None))
        self.assertFalse(overload_verdict(dict(metrics=m,sufficient=True,recovered=True))['passed'])
        from perf_metrics import MetricStore
        with tempfile.TemporaryDirectory() as d:
            store=MetricStore(Path(d)/'m.sqlite');store.append(row)
            self.assertEqual(store.summary(1),m);store.close()

    def test_tokenizer_snapshot_excludes_weights(self):
        from perf_resources import tokenizer_hashes, digest
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'tokenizer.json').write_text('{}')
            (root/'model.safetensors').write_text('must not read weights')
            with patch.object(Path,'read_bytes',side_effect=AssertionError('unbounded read')):
                result=tokenizer_hashes(root,Tokenizer())
                self.assertEqual(result,{'tokenizer.json':digest(root/'tokenizer.json')})

    def test_streaming_reservoir_is_bounded_and_repeatable(self):
        import tracemalloc
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'data.jsonl'
            with path.open('w') as out:
                for i in range(20000): out.write(json.dumps({'id':str(i),'prompt':'x'*97})+'\n')
            c=config();c['dataset']={'name':'longcodebench','path':str(path)}
            (root/'one').mkdir();(root/'two').mkdir()
            tracemalloc.start()
            with patch.object(Path,'read_text',side_effect=AssertionError('whole dataset read')):
                first,_=prepare(c,Tokenizer(),root/'one')
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            second,_=prepare(c,Tokenizer(),root/'two')
            self.assertLess(peak,2000000)
            self.assertEqual(first['tiny']['sha256'],second['tiny']['sha256'])
            self.assertEqual(len(first['tiny']['samples']),4)
            self.assertTrue(any(int(r['id'])>100 for r in first['tiny']['samples']))

    def test_selection(self):
        def point(n,throughput,ttft=.1):
            row=dict(success=True,truncated=False,input_tokens=100,output_tokens=10,ttft=ttft,tpot=.01,latency=.2,sample_id='1',http_status=200)
            m=metrics([row],1);m['output_tokens_per_second']=throughput
            return dict(concurrency=n,metrics=m,sufficient=True)
        points=[point(1,10),point(2,20),point(4,20.5)]
        self.assertEqual(select(points,config()['slo']),dict(sweet_concurrency=2,saturation_concurrency=2))
        self.assertIsNone(select([point(1,10),point(2,20)],config()['slo'])['saturation_concurrency'])
        self.assertIsNone(select([point(1,10,3)],config()['slo'])['sweet_concurrency'])

    def test_saturation_requires_last_two_scan_points_healthy(self):
        row=dict(success=True,truncated=False,input_tokens=100,output_tokens=10,
                 ttft=.1,tpot=.01,latency=.2,sample_id='1',http_status=200)
        def point(n, **changes):
            return dict(concurrency=n,metrics=metrics([row],1),sufficient=True,**changes)
        for points in (
            [point(1),point(2),point(4,client_bottleneck_suspected=True)],
            [point(1),point(2,client_bottleneck_suspected=True),point(4)],
        ):
            with self.subTest(points=points):
                self.assertIsNone(select(points,config()['slo'])['saturation_concurrency'])


try:
    import aiohttp
except ImportError:
    aiohttp=None


@unittest.skipUnless(aiohttp,'Run in performance venv for streaming integration tests')
class Streaming(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web
        self.web=web
        self.mode='normal';self.payload=None
        async def handler(request):
            self.payload=await request.json()
            if self.mode=='secret_error': return web.Response(status=503,reason='test-secret')
            if self.mode=='error': return web.Response(status=503)
            if self.mode=='timeout':
                await asyncio.sleep(.2)
                return web.Response(text='late')
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            async def event(data):
                payload=('data: '+json.dumps(data,ensure_ascii=False)+'\n\n').encode()
                # Deliberately split inside UTF8 and SSE fields.
                for byte in payload: await response.write(bytes([byte]))
            await event({'choices':[{'delta':{'role':'assistant'},'finish_reason':None}]})
            if self.mode=='stream_timeout':
                await asyncio.sleep(.2)
                return response
            await asyncio.sleep(.025)
            if self.mode=='reasoning_no_usage':
                await event({'choices':[{'delta':{'reasoning_content':'think'},'finish_reason':None}]})
            await event({'choices':[{'delta':{'content':'你好 abc'},'finish_reason':None}]})
            if self.mode=='idle': await asyncio.sleep(.2); return response
            if self.mode=='continuous':
                for _ in range(12):
                    await asyncio.sleep(.01)
                    if request.transport is None or request.transport.is_closing(): return response
                    await event({'choices':[{'delta':{'content':'x'},'finish_reason':None}]})
            await asyncio.sleep(.01)
            if self.mode!='incomplete':
                if self.mode!='done_without_finish':
                    await event({'choices':[{'delta':{},'finish_reason':'length' if self.mode=='length' else 'stop'}]})
                if self.mode not in ('no_usage','reasoning_no_usage'): await event({'choices':[],'usage':getattr(self,'usage_override',{'prompt_tokens':len(''.join(m['content'] for m in self.payload['messages']))+3,'completion_tokens':9})})
                await response.write(b'data: [DONE]\n\n')
                if hasattr(self,'done_gate'):
                    await self.done_gate.wait()
                    return response
            await response.write_eof();return response
        app=web.Application();app.router.add_post('/v1/chat/completions',handler)
        self.runner=web.AppRunner(app);await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.port=site._server.sockets[0].getsockname()[1]
        os.environ['PERF_TEST_KEY']='test-secret'
    async def asyncTearDown(self): await self.runner.cleanup()
    async def request(self,timeout=1,timing=None):
        from perf_transport import Transport
        c=config();c['target']['base_url']=f'http://127.0.0.1:{self.port}/v1'
        if timing: c['timing'].update(timing)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            return await Transport(c,Tokenizer(),session)({'id':'0','input_tokens':10,'messages':[{'role':'user','content':'test'}]})
    async def test_streaming_json_array(self):
        from perf_data import records
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.json'
            path.write_text('[{"prompt":"你好"},{"prompt":"second"}]')
            with patch.object(Path,'read_text',side_effect=AssertionError('whole file read')):
                self.assertEqual(list(records(path)),[{'prompt':'你好'},{'prompt':'second'}])
            path.write_text('{"item":[]}')
            with self.assertRaises(ValueError): list(records(path))
            path.write_text('[{"prompt":"broken"}')
            with self.assertRaises(ValueError): list(records(path))

    async def test_request_uses_loaded_template(self):
        from perf_transport import Transport
        cfg = config()
        cfg['chat_template'] = '/replaced/on/nas'
        cfg['target']['base_url'] = f'http://127.0.0.1:{self.port}/v1'
        cfg['target']['extra_body'] = {'chat_template_kwargs':{'enable_thinking':False}}
        tokenizer = Tokenizer()
        tokenizer.chat_template = 'frozen template'
        async with aiohttp.ClientSession() as session:
            transport = Transport(cfg,tokenizer,session)
            await transport({'id':'0','input_tokens':7,'messages':[{'role':'user','content':'test'}]})
        self.assertEqual(self.payload['chat_template'], 'frozen template')
        self.assertEqual(self.payload['chat_template_kwargs'], {'enable_thinking':False})

    async def test_natural_output_and_true_ttft(self):
        row=await self.request()
        self.assertTrue(row['success']);self.assertEqual(row['output_tokens'],9)
        self.assertGreater(row['ttft'],.02)
        self.assertGreater(row['ttft'],row['native_ttft'])
        self.assertEqual(self.payload['max_completion_tokens'],100)
        self.assertNotIn('ignore_eos',self.payload)
    async def test_invalid_usage_is_request_failure(self):
        bad = [[], 'bad', {'completion_tokens':-3}, {'completion_tokens':'3'},
               {'completion_tokens':True}, {'completion_tokens':1.5},
               {'completion_tokens':None}, {'completion_tokens':0},
               {'completion_tokens':101}, {'prompt_tokens':10**30},
               {'prompt_tokens':7,'completion_tokens':9,'total_tokens':99}]
        for usage in bad:
            with self.subTest(usage=usage):
                self.usage_override=usage
                row=await self.request()
                self.assertFalse(row['success'])
                self.assertEqual(row['error_kind'],'protocol')
                self.assertEqual(row['token_source'],'invalid')
                self.assertEqual(row['output_tokens'],0)
                self.assertEqual(metrics([row],1)['output_tokens_per_second'],0)
        del self.usage_override
        self.assertTrue((await self.request())['success'])

    async def test_missing_usage_field_estimates(self):
        self.usage_override={'prompt_tokens':7}
        row=await self.request()
        self.assertTrue(row['success'])
        self.assertEqual(row['token_source'],'estimated')
        self.assertGreater(row['output_tokens'],0)

    async def test_usage_validation_failure_does_not_stop_load(self):
        from perf_engine import load
        from perf_transport import Transport
        c=config();c['target']['base_url']=f'http://127.0.0.1:{self.port}/v1'
        c['timing']['window']=.04
        self.usage_override={'completion_tokens':-3}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
            transport=Transport(c,Tokenizer(),session)
            with tempfile.TemporaryDirectory() as d:
                result=await load(transport,[{'id':'1','input_tokens':7,'messages':[{'role':'user','content':'test'}]}],
                    1,.12,.12,0,c,Path(d),'bad-usage',c['slo'])
                self.assertGreater(result['metrics']['requests'],1)
                self.assertEqual(result['metrics']['completed'],0)
                self.assertEqual(result['metrics']['output_tokens_per_second'],0)

    async def test_done_finishes_without_http_eof(self):
        self.done_gate=asyncio.Event()
        try:
            # The server cannot send EOF until after the client has returned.
            row=await asyncio.wait_for(self.request(timeout=2),timeout=1)
            self.assertTrue(row['success'])
            self.assertEqual(row['finish_reason'],'stop')
            self.assertEqual(row['output_tokens'],9)
            self.assertIsNone(row['error_kind'])
            self.assertFalse(self.done_gate.is_set())
            self.assertLess(row['latency'],1)
        finally:
            self.done_gate.set()

    async def test_done_does_not_bypass_finish_validation(self):
        self.done_gate=asyncio.Event();self.mode='done_without_finish'
        try:
            row=await asyncio.wait_for(self.request(timeout=2),timeout=1)
            self.assertFalse(row['success'])
            self.assertIsNone(row['finish_reason'])
            self.assertEqual(row['error_kind'],'response')
        finally:
            self.done_gate.set()

    async def test_usage_missing(self):
        self.mode='no_usage';row=await self.request()
        self.assertTrue(row['success']);self.assertEqual(row['token_source'],'estimated')
        self.assertEqual(row['native_report_token_source'],'tokenizer')
        self.assertEqual(row['native_report_tokens'],len(Tokenizer().encode(row['native_output']['generated_text'])))

    async def test_native_fallback_timing_and_reasoning(self):
        from unittest.mock import patch
        calls=[]
        def encode(tokenizer,text,**kwargs):
            calls.append((time.perf_counter(),text))
            return list(text)
        with patch.object(Tokenizer,'encode',encode):
            self.mode='reasoning_no_usage'
            row=await self.request()
            self.assertEqual([text for _,text in calls],['think你好 abc','你好 abc'])
            self.assertLessEqual(row['end'],calls[0][0])
            self.assertEqual(row['native_report_tokens'],len('你好 abc'))
            self.assertEqual(row['output_tokens'],len('think你好 abc'))
            calls.clear()
            self.mode='no_usage'
            await self.request()
            self.assertEqual(len(calls),1)

    async def test_native_request_output_preserved(self):
        row=await self.request()
        native=row['native_output']
        self.assertEqual(row['native_output_origin'],'vllm')
        self.assertEqual(native['generated_text'],'你好 abc')
        self.assertTrue(native['success'])
        self.assertEqual(native['output_tokens'],9)
        self.assertIsInstance(native['itl'],list)
        self.assertEqual(native['ttft'],row['native_ttft'])
        self.assertEqual(native['latency'],row['native_latency'])
        self.assertGreater(row['ttft'],native['ttft'])

    async def test_native_error_is_redacted(self):
        self.mode='secret_error';row=await self.request()
        self.assertFalse(row['native_output']['success'])
        self.assertEqual(row['native_output']['error'],'[REDACTED]')
        self.assertNotIn('test-secret',json.dumps(row))

    async def test_adapter_deadline_is_marked(self):
        self.mode='timeout'
        row=await self.request(timing={'first_token_timeout':.01})
        self.assertEqual(row['native_output_origin'],'adapter_timeout')
        self.assertFalse(row['native_output']['success'])
        self.assertEqual(row['native_output']['error'],'first_token_timeout')
    async def test_truncated_and_incomplete(self):
        for mode in ('length','incomplete','error'):
            self.mode=mode;row=await self.request()
            self.assertFalse(row['success'])
            self.assertEqual(row['truncated'],mode=='length')
    async def test_timeout(self):
        self.mode='timeout';row=await self.request(.01)
        self.assertFalse(row['success']);self.assertIsNone(row['http_status'])
    async def test_timeout_after_http_200(self):
        self.mode='stream_timeout'
        row=await self.request(.1)
        self.assertFalse(row['success'])
        self.assertEqual(row['http_status'],200)
        self.assertEqual(row['error_kind'],'timeout')

    async def test_stream_progress_and_three_timeouts(self):
        self.mode='continuous'
        row=await self.request(timing={'first_token_timeout':.2,'stream_idle_timeout':.08,'timeout':1})
        self.assertTrue(row['success'])
        self.assertGreater(row['latency'],.08)
        self.mode='stream_timeout'
        row=await self.request(timing={'first_token_timeout':.04})
        self.assertEqual(row['timeout_phase'],'first_token_timeout')
        self.assertEqual(row['error_kind'],'timeout')
        self.mode='idle'
        row=await self.request(timing={'stream_idle_timeout':.04})
        self.assertEqual(row['timeout_phase'],'stream_idle_timeout')
        self.mode='continuous'
        row=await self.request(timing={'first_token_timeout':.2,'stream_idle_timeout':.1,'timeout':.08})
        self.assertEqual(row['timeout_phase'],'request_timeout')

    async def test_long_healthy_requests_do_not_abort(self):
        from perf_engine import load
        async def slow(sample):
            start=time.perf_counter();await asyncio.sleep(.1)
            return dict(sample_id='1',start=start,end=time.perf_counter(),success=True,truncated=False,
                ttft=.001,tpot=.001,latency=.1,input_tokens=100,output_tokens=100,http_status=200)
        c=config();c['timing'].update(window=.01,no_success=.04)
        with tempfile.TemporaryDirectory() as d:
            result=await load(slow,[{}],2,.15,.15,0,c,Path(d),'healthy',c['slo'],guard=True)
            self.assertIsNone(result['aborted']);self.assertTrue(result['sufficient'])

    async def test_no_success_guard_after_initial_success(self):
        from perf_engine import load
        count=0
        async def transport(sample):
            nonlocal count
            count+=1
            start=time.perf_counter()
            await asyncio.sleep(.005)
            return dict(sample_id='1',start=start,end=time.perf_counter(),success=count==1,
                truncated=False,ttft=.001,tpot=.001,latency=.005,input_tokens=10,
                output_tokens=4,http_status=200 if count==1 else 503)
        c=config();c['timing'].update(window=.01,no_success=.04)
        with tempfile.TemporaryDirectory() as d:
            result=await load(transport,[{}],1,.2,.2,0,c,Path(d),'degraded',c['slo'],guard=True)
            self.assertEqual(result['aborted'],'no_success')

    async def test_client_lag_includes_stats_and_storage(self):
        asyncio.get_running_loop().slow_callback_duration=1
        from perf_engine import load
        from perf_metrics import MetricStore
        from unittest.mock import patch
        async def transport(sample):
            start=time.perf_counter();await asyncio.sleep(.001)
            return dict(sample_id='1',start=start,end=time.perf_counter(),success=True,truncated=False,
                ttft=.001,tpot=.001,latency=.002,input_tokens=10,output_tokens=4,http_status=200)
        def slow_metrics(rows,seconds):
            time.sleep(.15)
            return metrics(rows,seconds)
        original=MetricStore.append
        def slow_append(store,row):
            time.sleep(.15)
            return original(store,row)
        c=config();c['timing']['window']=.01
        for target,replacement in [('perf_engine.metrics',slow_metrics),('perf_metrics.MetricStore.append',slow_append)]:
            with self.subTest(target=target),tempfile.TemporaryDirectory() as d,patch(target,replacement):
                result=await load(transport,[{}],1,.2,.2,0,c,Path(d),'blocked',c['slo'])
                self.assertTrue(result['client_bottleneck_suspected'])
                self.assertGreater(max(w['event_loop_lag_max'] for w in result['windows']),.1)
                self.assertIsNone(select([result],c['slo'])['sweet_concurrency'])

    async def test_bounded_workers_and_drain(self):
        from perf_engine import load
        active=0;peak=0;count=0
        async def transport(sample):
            nonlocal active,peak,count
            start=time.perf_counter();active+=1;peak=max(peak,active)
            await asyncio.sleep(.04);active-=1;count+=1
            return dict(sample_id='1',start=start,end=time.perf_counter(),success=True,truncated=False,
                ttft=.01,tpot=.01,latency=.04,input_tokens=10,output_tokens=4,http_status=200)
        c=config();c['timing']['window']=.03
        with tempfile.TemporaryDirectory() as a:
            r=await load(transport,[{}],3,.08,.08,0,c,Path(a),'test',c['slo'])
            self.assertEqual(peak,3);self.assertEqual(active,0)
            self.assertEqual(r['metrics']['requests'],count)
            self.assertGreater(r['drain_seconds'],0)
            self.assertGreater(len(r['windows']),1)
    async def test_full_orchestration(self):
        from unittest.mock import patch
        from run_perf import execute
        c=config();c['target']['base_url']=f'http://127.0.0.1:{self.port}/v1'
        c['concurrency']=[1,2]
        c['timing'].update(warmup=.05,scan=.08,scan_max=.3,scan_requests=2,
            overload=.05,recovery=.5,recovery_window=.05,soak=.08,window=.05)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            tokenizer=Tokenizer();data,_=prepare(c,tokenizer,root)
            summary={'profiles':{}}
            from perf_native_report import write_report
            def after_traffic(*args,**kwargs):
                self.assertTrue((root/'tiny/soak.requests.jsonl').exists())
                return write_report(*args,**kwargs)
            with patch('perf_native_report.write_report',side_effect=after_traffic):
                await execute(c,tokenizer,data,root,summary,'all')
            state=summary['profiles']['tiny']
            self.assertEqual(len(state['scan']),2)
            self.assertIsNotNone(state['sweet_concurrency'])
            self.assertIn('soak',state)
            self.assertEqual(json.loads((root/'summary.json').read_text()), summary)
            self.assertTrue((root/'tiny/soak.requests.jsonl').exists())
            self.assertTrue((root/'vllm-report.txt').is_file())
            reports=json.loads((root/'vllm-reports.json').read_text())
            self.assertTrue(any(r['phase']=='soak' for r in reports))
            report=json.loads((root/'tiny/soak.vllm.json').read_text())
            self.assertIn('mean_ttft_ms',report)
            self.assertIn('generated_texts',report)
            rows=[json.loads(line) for line in (root/'tiny/soak.requests.jsonl').read_text().splitlines()]
            rows.sort(key=lambda r:r['sequence'])
            self.assertEqual(len(report['ttfts']),len(rows))
            self.assertEqual(report['generated_texts'],[r['native_output']['generated_text'] for r in rows])

    async def test_overload_recovery_sequence(self):
        from unittest.mock import patch
        from run_perf import execute
        c=config();c['target']['base_url']=f'http://127.0.0.1:{self.port}/v1'
        c['concurrency']=[1,2];c['timing'].update(recovery=3,recovery_window=1)
        calls=[]
        async def controlled_load(transport,samples,concurrency,duration,maximum,minimum,config,folder,label,slo,guard=False):
            folder.mkdir(parents=True,exist_ok=True);calls.append((label,concurrency))
            row=dict(sample_id='1',success=True,truncated=False,ttft=.1,tpot=.01,latency=.2,
                input_tokens=100,output_tokens=10,http_status=200)
            m=metrics([row],1)
            return dict(concurrency=concurrency,metrics=m,sufficient=True,aborted=None,
                windows=[],dispatch_seconds=duration,slo_pass=True,client_bottleneck_suspected=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=Tokenizer();data,_=prepare(c,tokenizer,root)
            summary={'profiles':{}}
            with patch('perf_engine.load',side_effect=controlled_load):
                await execute(c,tokenizer,data,root,summary,'all')
            state=summary['profiles']['tiny']
            self.assertEqual(state['saturation_concurrency'],1)
            self.assertEqual([r['concurrency'] for r in state['overload']],[2,3])
            self.assertTrue(all(r['recovered'] for r in state['overload']))
            saved=json.loads((root/'tiny/overload-c2.json').read_text())
            self.assertTrue(saved['acceptance']['passed'])
            self.assertEqual(sum('recovery' in label for label,c in calls),6)
            self.assertEqual(state['soak_status'],'complete')

    async def test_recovery_total_deadline(self):
        from unittest.mock import patch
        from run_perf import execute
        c=config();c['target']['base_url']=f'http://127.0.0.1:{self.port}/v1'
        c['concurrency']=[1,2];c['timing'].update(recovery=.05,recovery_window=.01)
        cancelled=[]
        async def slow_recovery(t,s,n,d,maximum,minimum,config,folder,label,slo,guard=False):
            folder.mkdir(parents=True,exist_ok=True)
            if 'recovery' in label:
                try: await asyncio.sleep(10)
                finally: cancelled.append(label)
            row=dict(sample_id='1',success=True,truncated=False,ttft=.1,tpot=.01,latency=.2,
                input_tokens=100,output_tokens=10,http_status=200)
            return dict(concurrency=n,metrics=metrics([row],1),sufficient=True,aborted=None,
                windows=[],dispatch_seconds=d,slo_pass=True,client_bottleneck_suspected=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=Tokenizer();data,_=prepare(c,tokenizer,root);summary={'profiles':{}}
            started=time.perf_counter()
            with patch('perf_engine.load',side_effect=slow_recovery):
                with self.assertRaisesRegex(ValueError,'did not recover'):
                    await execute(c,tokenizer,data,root,summary,'overload')
            self.assertLess(time.perf_counter()-started,1)
            self.assertEqual(len(cancelled),1)
            self.assertEqual(summary['profiles']['tiny']['overload_status'],'failed_acceptance')
            self.assertFalse(summary['profiles']['tiny']['overload'][0]['acceptance']['passed'])

    async def test_cancellation_preserves_logs(self):
        from perf_engine import load
        active=0
        async def transport(sample):
            nonlocal active
            active+=1
            try: await asyncio.sleep(10)
            finally: active-=1
        c=config();c['timing']['window']=.02
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            task=asyncio.create_task(load(transport,[{}],2,10,10,0,c,root,'interrupt',c['slo']))
            await asyncio.sleep(.03);task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(active,0)
            self.assertTrue((root/'interrupt.requests.jsonl').exists())
            self.assertTrue((root/'interrupt.windows.jsonl').exists())

    async def test_no_success_guard(self):
        from perf_engine import load
        async def transport(sample):
            start=time.perf_counter();await asyncio.sleep(.01)
            return dict(sample_id='1',start=start,end=time.perf_counter(),success=False,truncated=False,
                ttft=None,tpot=None,latency=.01,input_tokens=10,output_tokens=0,http_status=503)
        c=config();c['timing'].update(window=.02,no_success=.04)
        with tempfile.TemporaryDirectory() as a:
            r=await load(transport,[{}],2,1,1,0,c,Path(a),'test',c['slo'],guard=True)
            self.assertEqual(r['aborted'],'no_success')


if __name__=='__main__': unittest.main()
