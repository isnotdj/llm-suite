"""vLLM v0.11.0 serving report, aggregated from disk after load stops.

The upstream report statements are executed verbatim. SQLite implements the
upstream calculate_metrics definitions without retaining all request texts/ITLs
in Python memory. Equivalence tests run the original calculator as the oracle.
"""
import ast
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile

from perf_resources import digest

REPORT_CONTRACT_VERSION = 2
PERCENTILES = (50, 95, 99)


@lru_cache(maxsize=1)
def upstream_report():
    vendor = Path(__file__).parent/'perf_vendor'
    provenance = json.loads((vendor/'PROVENANCE.json').read_text())['report']
    source = vendor/'serve.py'
    if digest(source) != provenance['sha256']:
        raise ValueError('Pinned vLLM report source hash mismatch')
    source_text = source.read_text()
    tree = ast.parse(source_text)
    classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('BenchmarkMetrics','TaskType')]
    benchmark = next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='benchmark')
    def text(node): return ast.get_source_segment(source_text,node) or ''
    start = next(i for i,n in enumerate(benchmark.body) if 'Serving Benchmark Result' in text(n))
    end = next(i for i in range(start,len(benchmark.body)) if text(benchmark.body[i])=='print("=" * 50)')
    # Omit only result-array materialization; those arrays are streamed to JSON.
    statements = [n for n in benchmark.body[start:end+1] if not any(
        isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in x.targets)
        for x in ast.walk(n))]
    namespace = dict(dataclass=dataclass, Enum=Enum)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace, compile(ast.Module(body=statements,type_ignores=[]),str(source),'exec')


def _distribution(db, key):
    count, mean = db.execute('SELECT COUNT(*), AVG(value) FROM values_ WHERE metric=?',(key,)).fetchone()
    if not count:
        return dict(mean=0.,std=0.,median=0.,percentiles=[(p,0.) for p in PERCENTILES])
    variance = db.execute('SELECT AVG((value-?)*(value-?)) FROM values_ WHERE metric=?',
                          (mean,mean,key)).fetchone()[0]
    def quantile(p):
        pos=(count-1)*p/100
        values=[r[0] for r in db.execute('SELECT value FROM values_ WHERE metric=? ORDER BY value LIMIT 2 OFFSET ?',
                                       (key,math.floor(pos)))]
        return (values[0]+(values[-1]-values[0])*(pos-math.floor(pos)))*1000
    return dict(mean=mean*1000,std=math.sqrt(max(0,variance))*1000,median=quantile(50),
                percentiles=[(p,quantile(p)) for p in PERCENTILES])


def _scheduler(stage, phase):
    if stage is None:
        return 'unknown'
    if 'request_limit' not in stage:
        return 'vllm-suite duration-based closed loop'
    if phase.startswith(('overload-', 'burst-')) and stage['request_limit'] == stage['concurrency']:
        return 'single-pulse'
    return 'fixed-request-count'


def write_report(requests, prefix, duration, concurrency, tokenizer, config=None, *, stage=None):
    """Return artifact names. No HTTP calls; duration includes request drain."""
    requests, prefix = Path(requests), Path(prefix)
    if duration <= 0: raise ValueError('Report duration must be positive')
    namespace, formatting = upstream_report()
    with tempfile.TemporaryDirectory(prefix='vllm-report-',dir=prefix.parent) as temporary:
        db = sqlite3.connect(Path(temporary)/'report.sqlite')
        try:
            db.execute('PRAGMA temp_store=FILE')
            db.execute('PRAGMA cache_size=-2048')
            db.execute('CREATE TABLE outputs (sequence INTEGER PRIMARY KEY, native TEXT, actual_len INTEGER)')
            db.execute('CREATE TABLE values_ (metric TEXT, value REAL)')
            completed=total_input=total_output=synthetic=0
            origin=None
            max_end=0.
            expected = stage['metrics']['requests'] if stage is not None else None
            contract = stage.get('report_contract') if stage is not None else None
            if stage is not None and (type(expected) is not int or expected < 0):
                raise ValueError(f'{requests}: invalid expected request count')
            profile, phase = prefix.parent.name, prefix.name
            version = contract['version'] if contract is not None else None
            require_identity = version == REPORT_CONTRACT_VERSION
            if contract is not None:
                if (type(version) is not int or version not in (1, REPORT_CONTRACT_VERSION)
                        or type(contract['token_count_version']) is not int or contract['token_count_version'] != 1):
                    raise ValueError(f'{requests}: unsupported report contract')
                if require_identity and (contract.get('profile'), contract.get('phase')) != (profile, phase):
                    raise ValueError(f'{requests}: stage identity conflicts with {profile}/{phase}')
            checksum = hashlib.sha256()
            observed = 0
            with requests.open('rb') as source:
                for index,line in enumerate(source):
                    if contract is not None:
                        checksum.update(line)
                    row=json.loads(line)
                    if stage is not None:
                        if (require_identity or 'phase' in row) and row.get('phase') != phase:
                            raise ValueError(f'{requests}: phase mismatch at row {index}')
                        if (require_identity or 'profile' in row) and row.get('profile') != profile:
                            raise ValueError(f'{requests}: profile mismatch at row {index}')
                    if stage is not None and contract is None and (
                        'native_report_tokens' in row or 'native_report_token_source' in row
                    ):
                        raise ValueError(f'{requests}: new-format request without stage report_contract')
                    sequence=row['sequence'] if stage is not None else row.get('sequence',index)
                    if expected is not None and (type(sequence) is not int or not 0 <= sequence < expected):
                        raise ValueError(f'{requests}: sequence {sequence!r} outside expected range [0,{expected})')
                    observed += 1
                    native=row['native_output']
                    synthetic += row['native_output_origin'] != 'vllm'
                    length=0
                    if contract is not None:
                        length=row['native_report_tokens']
                        source_kind = row['native_report_token_source']
                        expected_kind = ('failed' if not native['success'] else
                                         'usage' if native['output_tokens'] else 'tokenizer')
                        if (type(length) is not int or length < 0 or source_kind != expected_kind
                                or (source_kind == 'failed' and length != 0)
                                or (source_kind == 'usage' and length != native['output_tokens'])):
                            raise ValueError(f'{requests}: invalid native token count at sequence {sequence}')
                    if native['success']:
                        if contract is None:
                            length=native['output_tokens'] or len(tokenizer.encode(native['generated_text'],add_special_tokens=False))
                        completed+=1
                        total_input+=row['native_input_tokens']
                        total_output+=length
                        start=native['start_time']
                        origin=start if origin is None else min(origin,start)
                        max_end=max(max_end,start+native['latency'])
                        values=[('ttft',native['ttft']),('e2el',native['latency'])]
                        if length>1: values.append(('tpot',(native['latency']-native['ttft'])/(length-1)))
                        values.extend(('itl',v) for v in native['itl'])
                        db.executemany('INSERT INTO values_ VALUES (?,?)',values)
                    try:
                        db.execute('INSERT INTO outputs VALUES (?,?,?)',(sequence,json.dumps(native,ensure_ascii=False),length))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f'{requests}: duplicate sequence {sequence}') from exc
            if expected is not None and observed != expected:
                raise ValueError(f'{requests}: expected {expected} requests, found {observed}')
            if contract is not None and checksum.hexdigest() != contract['requests_sha256']:
                raise ValueError(f'{requests}: request log SHA-256 mismatch')
            db.execute('CREATE INDEX value_order ON values_ (metric,value)')
            # Buckets grow with elapsed seconds, never with request/token count.
            db.execute('CREATE TABLE buckets (second INTEGER PRIMARY KEY, tokens INTEGER, active INTEGER)')
            def bucket(second, tokens=0, active=0):
                if second < 0 or second >= int(math.ceil(max_end-origin))+1: return
                db.execute('INSERT INTO buckets VALUES (?,?,?) ON CONFLICT(second) DO UPDATE SET tokens=tokens+excluded.tokens, active=active+excluded.active',
                           (second,tokens,active))
            for (encoded,) in db.execute('SELECT native FROM outputs'):
                native=json.loads(encoded)
                if not native['success']: continue
                start=native['start_time']
                timestamp=start+native['ttft']
                bucket(int(timestamp-origin),tokens=1)
                for interval in native['itl']:
                    timestamp+=interval
                    bucket(int(timestamp-origin),tokens=1)
                for second in range(int(start-origin),int(start+native['latency']-origin)+1):
                    bucket(second,active=1)
            peak_tokens,peak_active=db.execute('SELECT COALESCE(MAX(tokens),0), COALESCE(MAX(active),0) FROM buckets').fetchone()
            fields=dict(completed=completed,total_input=total_input,total_output=total_output,
                        request_throughput=completed/duration,request_goodput=0.,
                        output_throughput=total_output/duration,total_token_throughput=(total_input+total_output)/duration,
                        max_output_tokens_per_s=float(peak_tokens),max_concurrent_requests=peak_active)
            for key in ('ttft','tpot','itl','e2el'):
                stats=_distribution(db,key)
                for name in ('mean','median','std'): fields[f'{name}_{key}_ms']=stats[name]
                fields[f'percentiles_{key}_ms']=stats['percentiles']
            metrics=namespace['BenchmarkMetrics'](**fields)
            result=dict(duration=duration,completed=completed,total_input_tokens=total_input,
                        total_output_tokens=total_output,request_throughput=metrics.request_throughput,
                        request_goodput=None,output_throughput=metrics.output_throughput,
                        total_token_throughput=metrics.total_token_throughput,
                        max_output_tokens_per_s=metrics.max_output_tokens_per_s,
                        max_concurrent_requests=metrics.max_concurrent_requests)
            environment={**namespace,'metrics':metrics,'benchmark_duration':duration,
                         'max_concurrency':concurrency,'request_rate':float('inf'),
                         'goodput_config_dict':{},'rps_change_events':[],
                         'selected_percentile_metrics':['ttft','tpot','itl','e2el'],
                         'task_type':namespace['TaskType'].GENERATION,'result':result}
            output=io.StringIO()
            with redirect_stdout(output): exec(formatting,environment)
            text_path=prefix.with_name(prefix.name+'.vllm.txt')
            text_path.write_text(output.getvalue())
            json_path=prefix.with_name(prefix.name+'.vllm.json')
            metadata=dict(request_count_verified=expected is not None,sequence_verified=expected is not None,
                          sha256_verified=contract is not None,vllm_version='v0.11.0',scheduler=_scheduler(stage, prefix.name),
                          source_requests=requests.name,adapter_timeout_outputs=synthetic,
                          percentiles=list(PERCENTILES),native_success_semantics=True,
                          request_order='dispatch sequence')
            result['vllm_suite_metadata']=metadata
            if config is not None:
                result.update(date=datetime.now().strftime('%Y%m%d-%H%M%S'),
                              endpoint_type='openai-chat',backend='openai-chat',label=prefix.name,
                              model_id=config['target']['model'],tokenizer_id=config['tokenizer'],
                              num_prompts=db.execute('SELECT COUNT(*) FROM outputs').fetchone()[0],
                              request_rate='inf',burstiness=1.,max_concurrency=concurrency)
            # Preserve upstream's result arrays without loading them all at once.
            with json_path.open('w') as out:
                out.write(json.dumps(result,ensure_ascii=False,allow_nan=False)[:-1])
                for field,key in [('input_lens','prompt_len'),('output_lens',None),('ttfts','ttft'),
                                  ('itls','itl'),('generated_texts','generated_text'),('errors','error')]:
                    out.write(',\n'+json.dumps(field)+':[')
                    separator=''
                    for encoded,length in db.execute('SELECT native,actual_len FROM outputs ORDER BY sequence'):
                        value=json.loads(encoded)[key] if key is not None else length
                        out.write(separator+json.dumps(value,ensure_ascii=False,allow_nan=False))
                        separator=','
                    out.write(']')
                out.write('}\n')
            return dict(text=text_path.name,json=json_path.name)
        finally:
            db.close()
