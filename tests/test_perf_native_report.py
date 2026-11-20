"""Compare disk aggregation and report against the untouched upstream source."""
import ast
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipUnless(np is not None,'Requires performance dependencies')
class NativeReport(unittest.TestCase):
    def oracle(self, rows, duration, tokenizer):
        from perf_native_report import PERCENTILES, upstream_report
        from perf_vendor.endpoint_request_func import RequestFuncOutput
        namespace,_=upstream_report()
        source=Path(__file__).parents[1]/'scripts/perf_vendor/serve.py'
        tree=ast.parse(source.read_text())
        calculator=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_metrics')
        benchmark=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='benchmark')
        segment=lambda n:ast.get_source_segment(source.read_text(),n) or ''
        first=next(i for i,n in enumerate(benchmark.body) if 'Serving Benchmark Result' in segment(n))
        last=next(i for i in range(first,len(benchmark.body)) if segment(benchmark.body[i])=='print("=" * 50)')
        outputs=[RequestFuncOutput(**r['native_output']) for r in rows]
        inputs=[SimpleNamespace(prompt_len=r['native_input_tokens']) for r in rows]
        env={**namespace,'np':np,'warnings':warnings,'SampleRequest':SimpleNamespace,
             'RequestFuncOutput':RequestFuncOutput,'PreTrainedTokenizerBase':object,
             'TERM_PLOTLIB_AVAILABLE':False,'MILLISECONDS_TO_SECONDS_CONVERSION':1000}
        exec(compile(ast.Module(body=[calculator],type_ignores=[]),str(source),'exec'),env)
        with redirect_stdout(io.StringIO()):
            metrics,lengths=env['calculate_metrics'](inputs,outputs,duration,tokenizer,list(PERCENTILES),{})
        env.update(metrics=metrics,outputs=outputs,actual_output_lens=lengths,
                   benchmark_duration=duration,max_concurrency=4,request_rate=float('inf'),
                   goodput_config_dict={},rps_change_events=[],
                   selected_percentile_metrics=['ttft','tpot','itl','e2el'],task_type=env['TaskType'].GENERATION)
        text=io.StringIO()
        with redirect_stdout(text):
            exec(compile(ast.Module(body=benchmark.body[first:last+1],type_ignores=[]),str(source),'exec'),env)
        return env['result'],text.getvalue()

    def rows(self):
        from perf_vendor.endpoint_request_func import RequestFuncOutput
        values=[
            RequestFuncOutput(success=True,generated_text='hello',output_tokens=5,prompt_len=100,
                              ttft=.1,latency=.5,itl=[.1,.1,.1,.1],start_time=100),
            RequestFuncOutput(success=True,generated_text='你好',output_tokens=0,prompt_len=200,
                              ttft=.2,latency=.6,itl=[.4],start_time=100.3),
            RequestFuncOutput(success=True,generated_text='x',output_tokens=1,prompt_len=300,
                              ttft=.3,latency=.3,itl=[],start_time=101),
            RequestFuncOutput(success=False,error='503',prompt_len=400,start_time=101.5),
        ]
        return [dict(sequence=i,native_output=asdict(v),native_output_origin='vllm',native_input_tokens=v.prompt_len)
                for i,v in enumerate(values)]

    def test_json_and_text_match_upstream(self):
        from perf_native_report import write_report
        class Tokenizer:
            def encode(self,text,**kwargs): return list(text)
            def __call__(self,text,**kwargs): return SimpleNamespace(input_ids=self.encode(text))
        for rows in (self.rows(),list(reversed(self.rows())),[],[self.rows()[-1]]):
            with self.subTest(count=len(rows)), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);requests=root/'requests.jsonl'
                requests.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
                expected,text=self.oracle(sorted(rows,key=lambda r:r['sequence']),2,Tokenizer())
                artifacts=write_report(requests,root/'scan-c4',2,4,Tokenizer())
                actual=json.loads((root/artifacts['json']).read_text())
                actual.pop('vllm_suite_metadata')
                self.assertEqual(actual.keys(),expected.keys())
                for key,value in expected.items():
                    if isinstance(value,float): self.assertAlmostEqual(actual[key],value,places=10,msg=key)
                    else: self.assertEqual(actual[key],value,key)
                self.assertEqual((root/artifacts['text']).read_text(),text)

    def test_native_success_is_not_replaced_by_suite_verdict(self):
        from perf_native_report import write_report
        rows=self.rows()[:1];rows[0]['success']=False
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'requests.jsonl'
            source.write_text(json.dumps(rows[0])+'\n')
            paths=write_report(source,root/'scan',1,1,None)
            self.assertEqual(json.loads((root/paths['json']).read_text())['completed'],1)

    def test_adapter_timeouts_identified(self):
        from perf_native_report import write_report
        row=self.rows()[-1];row['native_output_origin']='adapter_timeout'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'requests.jsonl';source.write_text(json.dumps(row)+'\n')
            paths=write_report(source,root/'scan',1,1,None)
            result=json.loads((root/paths['json']).read_text())
            self.assertEqual(result['vllm_suite_metadata']['adapter_timeout_outputs'],1)
            self.assertEqual(result['completed'],0)

    def test_large_texts_are_streamed(self):
        import tracemalloc
        from perf_native_report import upstream_report, write_report
        upstream_report()  # exclude one-time parsing/imports
        row=self.rows()[0];row['native_output']['generated_text']='x'*16384
        row.pop('sequence')
        encoded=json.dumps(row)+'\n'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'requests.jsonl'
            with source.open('w') as out:
                for _ in range(600): out.write(encoded)
            tracemalloc.start()
            try:
                paths=write_report(source,root/'scan',1,1,None)
                _,peak=tracemalloc.get_traced_memory()
            finally: tracemalloc.stop()
            self.assertLess(peak,3_000_000)
            self.assertGreater((root/paths['json']).stat().st_size,9_000_000)
