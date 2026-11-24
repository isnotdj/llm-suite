"""Run a real FastAPI child process and production performance CLI on localhost."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    make_tokenizer(root / 'tokenizer')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, MOCK_TOKENIZER=str(root / 'tokenizer'), TOKENIZERS_PARALLELISM='false')
    base = f'http://127.0.0.1:{port}'
    config = json.loads((REPO/'perf.example.json').read_text())
    config['vllm_bench_serve'].update(base_url=base, tokenizer=str(root/'tokenizer'))
    config.update(profiles=[{'name':'tiny', 'vllm_bench_serve':{'random_input_len':64,'random_output_len':16}}],
                  concurrency=[1, 2], requests_per_concurrency=2, overload_concurrency=[3], soak_requests=7,
                  slo={'ttft_ms':5000, 'tpot_ms':1000})
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
            cfg = root/'perf.json'
            write(cfg, config)
            env['OPENAI_API_KEY'] = 'unused'
            subprocess.run([sys.executable, str(REPO/'scripts/run_perf.py'), '--config', str(cfg),
                            '--output',str(root/'outputs')], env=env, check=True, timeout=300)
            run = next((root/'outputs').iterdir())
            state = json.loads((run/'summary.json').read_text())['profiles']['tiny']
            measured = [point['expected'] for point in state['runs']]
            assert measured == [1, 2, 4, 3, 3, 3, 7], measured
            for point in state['runs']:
                result = json.loads((run/point['result']).read_text())
                assert result['completed'] == point['expected']
                assert len(result['generated_texts']) == point['expected']
                assert 'Serving Benchmark Result' in (run/point['report']).read_text()
            with urllib.request.urlopen(base+'/health') as response:
                stats = json.load(response)
            assert stats['requests'] == sum(measured), stats
            write(root/'verification.json', dict(counts=measured, server=stats))
        finally:
            server.terminate()
            try: server.wait(timeout=5)
            except subprocess.TimeoutExpired: server.kill();server.wait()

if __name__=='__main__': main()
