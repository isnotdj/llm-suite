"""Container smoke checks: no datasets downloaded, no remote LLM calls."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

assert sys.version_info[:2] == (3, 12)
assert 'VERSION_ID="24.04"' in Path('/etc/os-release').read_text()
for lock in (Path('/opt/evaluation') / name for name in
             ('requirements.build.lock', 'requirements.cpu.lock', 'requirements.lock')):
    for line in lock.read_text().splitlines():
        if not line or line.startswith('#'): continue
        name, version = line.split('==')
        assert importlib.metadata.version(name) == version, (name, version)
versions = {}
for name in ('opencompass', 'openai', 'datasets', 'torch', 'nltk', 'tiktoken', 'matplotlib'):
    importlib.import_module(name)
    versions[name] = importlib.metadata.version(name)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
for command in ('eval-suite', 'dataset-path', 'curl', 'dig', 'ip', 'jq'):
    assert shutil.which(command), command
subprocess.run(['eval-suite', '--help'], check=True, stdout=subprocess.DEVNULL)
# The performance client has its own interpreter and dependency versions.
subprocess.run(['/opt/perf-venv/bin/python', '-c', '''
import importlib.metadata
from pathlib import Path
for line in Path('/opt/evaluation/requirements.perf.lock').read_text().splitlines():
    if not line or line.startswith('#'): continue
    name, expected = line.split('==')
    assert importlib.metadata.version(name) == expected, (name, expected)
import aiohttp, transformers, ijson
import sys
sys.path.insert(0, '/opt/evaluation/scripts')
from perf_native_report import upstream_report
upstream_report()
'''], check=True)
subprocess.run(['/opt/perf-venv/bin/python', '-m', 'pip', 'check'], check=True)
subprocess.run(['perf-suite', '--help'], check=True, stdout=subprocess.DEVNULL)
import nltk
for path in ('tokenizers/punkt_tab', 'corpora/stopwords', 'taggers/averaged_perceptron_tagger_eng'):
    nltk.data.find(path)
import tiktoken
for name in ('cl100k_base', 'o200k_base'):
    assert tiktoken.get_encoding(name).encode('离线 smoke check')
from halluqa_eval import official_prompt
prompt = official_prompt(Path('/opt/benchmarks/HalluQA/calculate_metrics.py'))
samples = json.loads(Path('/opt/benchmarks/HalluQA/HalluQA.json').read_text())
item = samples[0]
_, messages = prompt({'question_id':item['question_id'], 'response':'测试'}, {item['question_id']:item})
assert messages
for code in ('code','code_zh'):
    subprocess.run([sys.executable, f'/opt/benchmarks/FollowBench-api/{code}/llm_eval.py', '--help'], check=True, stdout=subprocess.DEVNULL)


from evaluation_contracts import follow_groups
for file in Path('/opt/benchmarks/FollowBench').glob('data*/*_constraints.json'):
    follow_groups(file.name.split('_')[0], json.loads(file.read_text()))

import compass_plugins
from compass_runner import SUPPORTED
assert len(SUPPORTED) == 9
assert compass_plugins.cmrc_module().calc_em_score(['北京'], '北京') == 1

assert compass_plugins.cmrc_module().calc_f1_score(['北京 A'], '北京 A') == 1
print(json.dumps({'status':'passed', 'architecture':platform.machine(), 'python':platform.python_version(), 'versions':versions}, indent=2))
