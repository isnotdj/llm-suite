"""Run a real FastAPI child process and production performance CLI on localhost."""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def make_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    vocab = {'[UNK]': 0, **{chr(i): i - 31 for i in range(32, 127)}}
    raw = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    raw.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ':' + message['content'] + ';' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}"
    tokenizer.save_pretrained(path)
    return tokenizer


async def faults(config, tokenizer, root):
    import aiohttp
    from perf_transport import Transport
    from perf_data import token_count
    from perf_engine import load
    from perf_metrics import overload_verdict
    sample = dict(id='fault', messages=[dict(role='user', content='hello')])
    sample['input_tokens'] = token_count(tokenizer, sample['messages'])
    results = {}
    async with aiohttp.ClientSession() as session:
        for mode in ('normal', '429', '503', 'first_timeout', 'idle_timeout', 'length',
                     'incomplete', 'done_without_finish', 'invalid_usage', 'no_usage', 'mismatch', 'done_open'):
            c = copy.deepcopy(config)
            c['target']['extra_body'] = {'mock_mode': mode}
            c['timing'].update(first_token_timeout=.5, stream_idle_timeout=.5, timeout=1)
            row = await Transport(c, tokenizer, session)(sample)
            if mode in ('normal', 'no_usage', 'mismatch', 'done_open'):
                assert row['success'], (mode, row)
            else:
                assert not row['success'], (mode, row)
            if mode in ('429', '503'): assert row['http_status'] == int(mode)
            if mode in ('first_timeout', 'idle_timeout'): assert row['error_kind'] == 'timeout'
            if mode == 'length': assert row['truncated']
            if mode == 'invalid_usage': assert row['token_source'] == 'invalid'
            if mode == 'no_usage': assert row['token_source'] == 'estimated'
            if mode == 'mismatch': assert row['input_token_delta'] == 1
            if mode == 'done_open': assert row['latency'] < 1
            results[mode] = row
            # Allow the ASGI disconnect watcher to release the previous slot.
            await asyncio.sleep(.05)
        c['target']['extra_body'] = {'mock_mode': '503'}
        c['timing']['no_success'] = .3
        result = await load(Transport(c, tokenizer, session), [sample], 2, 2, 2, 0,
                            c, root / 'fault-load', '503', c['slo'], guard=True)
        assert result['aborted'] == 'no_success', result
        assert not overload_verdict(dict(result, recovered=True))['passed']
        results['503_guard'] = result
    write(root / 'faults.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tokenizer = make_tokenizer(root / 'tokenizer')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, MOCK_TOKENIZER=str(root / 'tokenizer'), PERF_MOCK_KEY='local-fixture', TOKENIZERS_PARALLELISM='false')
    os.environ['PERF_MOCK_KEY'] = env['PERF_MOCK_KEY']
    base = f'http://127.0.0.1:{port}'
    from perf_config import normalize
    config = normalize(dict(target=dict(base_url=base+'/v1', model='mock', api_key_env='PERF_MOCK_KEY'),
        tokenizer=str(root/'tokenizer'), context_window=131072, max_generation_tokens=8192,
        dataset={'name':'random'}, sample_count=4, min_samples=2,
        profiles=[dict(name=name,input_tokens=n,output_target=out) for name,n,out in
                  [('4k',4096,1536),('16k',16384,512),('64k',65536,1536)]],
        concurrency=[1,2,4], overload_cap=12,
        slo=dict(ttft_ms=5000,tpot_ms=100),
        timing=dict(warmup=.3,scan=4,scan_max=6,scan_requests=10,overload=2,
                    recovery=10,recovery_window=.6,soak=3,window=.5,timeout=10,no_success=3)))
    outcomes = {}
    with (root/'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable,'-m','uvicorn','server:app','--app-dir',str(Path(__file__).parent),
                                   '--host','127.0.0.1','--port',str(port),'--log-level','warning'],env=env,stdout=log,stderr=log)
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(base+'/health',timeout=1): break
                except OSError:
                    if server.poll() is not None: raise RuntimeError('Mock exited; see server.log')
                    time.sleep(.1)
            else: raise RuntimeError('Mock startup timeout')
            print(f'Mock ready: {base}; artifacts: {root}',flush=True)
            for source in ('random','sharegpt'):
                c=copy.deepcopy(config)
                if source=='sharegpt':
                    data=root/'sharegpt.jsonl'
                    with data.open('w') as out:
                        for p in c['profiles']:
                            for i in range(4):
                                out.write(json.dumps({'id':f"{p['name']}-{i}",'conversations':[
                                    {'from':'human','value':'x'*(p['input_tokens']-150+i*10)},
                                    {'from':'gpt','value':'ANSWER_MUST_NOT_LEAK'}]})+'\n')
                    c['dataset']={'name':'sharegpt','path':str(data)}
                cfg=root/f'{source}.json';write(cfg,c)
                with (root/f'{source}.log').open('w') as output:
                    run=subprocess.run([sys.executable,str(REPO/'scripts/run_perf.py'),'--config',str(cfg),
                                        '--output',str(root/source)],env=env,stdout=output,stderr=output,timeout=240)
                run_dir=next((root/source).iterdir())
                summary=json.loads((run_dir/'summary.json').read_text())
                reports=json.loads((run_dir/'vllm-reports.json').read_text())
                assert summary['reporting_status']=='complete' and reports
                assert 'Serving Benchmark Result' in (run_dir/'vllm-report.txt').read_text()
                for report in reports:
                    native=json.loads((run_dir/report['json']).read_text())
                    request_file=run_dir/report['profile']/native['vllm_suite_metadata']['source_requests']
                    rows=[json.loads(line) for line in request_file.read_text().splitlines()]
                    rows.sort(key=lambda r:r['sequence'])
                    assert native['completed']==sum(r['native_output']['success'] for r in rows)
                    assert native['generated_texts']==[r['native_output']['generated_text'] for r in rows]
                    assert len(native['itls'])==len(rows)
                    assert (run_dir/report['text']).is_file()
                outcomes[source]={'exit_code':run.returncode,'status':summary['overall_status']}
                assert run.returncode==0, (source,summary)
                for name,state in summary['profiles'].items():
                    assert state['overload_status']=='complete', (name,state)
                    assert state['soak_status']=='complete', (name,state)
                    assert state['overload'] and all(r['recovered'] and r['acceptance']['passed'] for r in state['overload'])
                print(f'{source}: complete (all profiles, overload/recovery/soak)',flush=True)
            asyncio.run(faults(config,tokenizer,root))
            with urllib.request.urlopen(base+'/health') as response: stats=json.load(response)
            assert stats['peak_active']==1 and stats['peak_waiting']>0, stats
            outcomes.update(faults='passed',server=stats)
            write(root/'verification.json',outcomes)
            print(json.dumps(outcomes,indent=2),flush=True)
        finally:
            server.terminate()
            try: server.wait(timeout=5)
            except subprocess.TimeoutExpired: server.kill();server.wait()

if __name__=='__main__': main()
