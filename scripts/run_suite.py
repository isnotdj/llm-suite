#!/opt/venv/bin/python
"""Sequential remote-only evaluation suite with a JSON result and setup snapshot."""
from eval_concurrency import write_records
import argparse
import ast
import csv
import json
import os
from uuid import UUID
from pathlib import Path
import re
import subprocess
import sys
import math
import traceback
import signal
import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError, distributions
from urllib.parse import urlsplit
from dataset_mapping import read_mapping, resolve
from halluqa_eval import request
from run_journal import atomic_json

SCRIPTS = Path(__file__).resolve().parent
ACTIVE_RUN = None
ACTIVE_CONFIG = {}


class SuiteInterrupted(KeyboardInterrupt):
    pass


def terminate_handler(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SuiteInterrupted('SIGTERM')


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


from compass_runner import SUPPORTED, selected as selected_stages
STAGE_ORDER = SUPPORTED


def validate_config(config):
    for key in ('concurrency', 'judge_concurrency'):
        value = config.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    for role in ('target', 'judge'):
        item = config[role]
        url = urlsplit(item['base_url'])
        if url.scheme not in ('http','https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError(f'{role}.base_url must be an HTTP(S) endpoint without credentials/query')
        for key, default in [('timeout',60),('max_tokens',2048)]:
            value = item.get(key,default)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{role}.{key} must be positive and finite')
        temp = item.get('temperature',0)
        if isinstance(temp,bool) or not isinstance(temp,(int,float)) or not math.isfinite(temp) or temp < 0:
            raise ValueError('temperature must be nonnegative and finite')
        if not isinstance(item.get('max_tokens',2048),int): raise ValueError('max_tokens must be integer')
    if not isinstance(config.get('continue_on_error', True), bool):
        raise ValueError('continue_on_error must be boolean')
    options = config.get('options', {})
    if not isinstance(options,dict) or set(options)-set(SUPPORTED):
        raise ValueError('Unknown dataset in options')
    if 'language' in options.get('followbench', {}):
        raise ValueError('Set language in top-level followbench.language')
    if config.get('followbench',{}).get('language','en') not in ('en','zh'):
        raise ValueError('FollowBench language must be en or zh')


def preflight(config, mappings, on_resolve=None):
    validate_config(config)
    pinned = {}
    for name in selected_stages(config):
        key = 'followbench_'+config.get('followbench',{}).get('language','en') if name=='followbench' else name
        if on_resolve is not None: on_resolve(name)
        pinned[key] = resolve(key,mappings)
    return pinned


def failure_details(exc, config):
    detail = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for role in ('target', 'judge'):
        key = os.environ.get(config[role]['api_key_env'])
        if key:
            detail = detail.replace(key, '[REDACTED]')
    return detail


def client(config):
    from openai import OpenAI
    return OpenAI(base_url=config['base_url'], api_key=os.environ[config['api_key_env']],
                  timeout=config.get('timeout', 60), max_retries=0)


def generate(api, config, prompt):
    return generate_messages(api, config, [{'role':'user','content':prompt}])


def generate_messages(api, config, messages):
    return generate_choice(api, config, messages).message.content


def generate_choice(api, config, messages):
    response = request(api, dict(model=config['model'], messages=messages,
                                temperature=config.get('temperature', 0),
                                max_tokens=config.get('max_tokens', 2048)))
    if len(response.choices) != 1 or response.choices[0].finish_reason not in ('stop', 'length'):
        raise ValueError('Generation incomplete or rejected; inspect token limit/service')
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError('Missing generated text')
    return response.choices[0]


def stop_helper(process):
    # Helpers own a process group; descendants can outlive its leader.
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(cmd, cwd, env=None, *, score_report=None):
    with (cwd / 'commands.log').open('a') as log:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
        except BaseException as exc:
            stop_helper(process)
            if score_report is not None and isinstance(exc, KeyboardInterrupt):
                try:
                    if score_report.exists():
                        report = json.loads(score_report.read_text())
                        if report['status'] == 'running':
                            report['status'] = 'interrupted'
                            atomic_json(score_report, report)
                except (OSError, ValueError, KeyError) as report_error:
                    # Preserve the interruption so continue_on_error cannot start more work.
                    detail = f'Could not finalize {score_report}: {report_error}'
                    exc.add_note(detail)
                    print(detail, file=sys.stderr)
            raise
        if code == 1:
            raise subprocess.CalledProcessError(code, cmd)
        if code == os.EX_IOERR:
            raise OSError(f'Scorer I/O failure; see {cwd / "commands.log"}')
        if code:
            raise OSError(f'Scorer exited with {code}; see {cwd / "commands.log"}')


def votes(text, level):
    matches = re.findall(r'\[[^\[\]]*\]', text)
    if not matches:
        raise ValueError('Judge must return a YES/NO list')
    value = ast.literal_eval(matches[-1])
    if not isinstance(value, list) or len(value) != level or any(v not in ('YES','NO') for v in value):
        raise ValueError('Invalid judge vote list')
    return value


def follow(config, folder, mappings, predictions):
    language = config.get('followbench', {}).get('language', 'en')
    if language not in ('en','zh'):
        raise ValueError('FollowBench language must be en or zh')
    code = Path('/opt/benchmarks/FollowBench') / ('code' if language == 'en' else 'code_zh')
    data = Path(resolve('followbench_' + language, mappings))
    sys.path.insert(0, str(code))
    from gpt4_based_evaluation import acquire_discriminative_eval_input
    judge = client(config['judge'])
    responses = folder / 'responses'
    (responses / 'target').mkdir(parents=True)
    categories = ['content','situation','style','format','mixed','example']
    def relative_file(category):
        return Path('target') / f'{category}_constraint.jsonl' if language == 'en' else Path(f'target_{category}_constraint.jsonl')
    provided = iter(predictions)
    for category in categories:
        samples = json.loads((data / f'{category}_constraints.json').read_text())
        with (responses / relative_file(category)).open('w') as out:
            for sample in samples:
                if sample['level'] == 0:
                    continue
                text = next(provided)
                out.write(json.dumps({('prompt' if language == 'en' else 'prompt_new'):sample['instruction'],'choices':[{'message':{'content':text}}]}, ensure_ascii=False)+'\n')
                out.flush()
    metadata, inputs, outputs = folder / 'judge_metadata', folder / 'judge_inputs', folder / 'judge_outputs'
    (outputs / 'target').mkdir(parents=True)
    def judge_line(line):
        item = json.loads(line)
        for attempt in range(3):
            text = generate(judge, {'max_tokens': 1024, **config['judge']}, item['prompt_new'])
            try:
                canonical = repr(votes(text, item['level']))
                break
            except (ValueError, SyntaxError):
                if attempt == 2:
                    raise
        return {'prompt_new':item['prompt_new'],'choices':[{'message':{'content':canonical}}]}

    for category in categories[:-1]:
        acquire_discriminative_eval_input(str(data), str(responses), category, 'target', str(metadata), str(inputs))
        lines = (metadata / relative_file(category)).read_text().splitlines()
        write_records(outputs / relative_file(category), judge_line, lines,
                      config.get('judge_concurrency', 1), len(lines))
    results = folder / 'evaluation_result'
    run([sys.executable,str(SCRIPTS/'followbench_score.py'),str(code/'eval.py'),'--model_paths','target','--data_path',str(data),
         '--api_output_path',str(responses),'--data_gpt4_discriminative_eval_input_path',str(metadata),
         '--gpt4_discriminative_eval_output_path',str(outputs),'--evaluation_result_path',str(results)], folder)
    expected = {'example_rule_based.csv','example_csl.csv'} | {f'{category}_{metric}.csv' for category in categories[:-1] for metric in ('hsr','ssr','csl')}
    if {p.name for p in results.glob('*.csv')} != expected:
        raise ValueError('FollowBench report coverage is incomplete')
    rows = []
    for path in sorted(results.glob('*.csv')):
        with path.open() as source:
            reader = csv.DictReader(source)
            records = list(reader)
            columns = {'CSL'} if path.stem.endswith('_csl') else {f'level {i}' for i in range(1,6)}
            if len(records) != 1 or records[0].get('model_name') != 'target' or set(records[0]) != columns | {'model_name'}:
                raise ValueError('FollowBench report fields/count are invalid')
            for record in records:
                for key, value in record.items():
                    if key != 'model_name':
                        rows.append(['FollowBench', language + '/' + path.stem, key, value, str(path.relative_to(folder.parent))])
    if not rows:
        raise ValueError('No FollowBench CSV metrics found')
    return rows


def hallu(config, folder, mappings, predictions):
    path = Path(resolve('halluqa', mappings, '/opt/benchmarks/HalluQA/HalluQA.json'))
    if path.is_dir():
        path = path / 'HalluQA.json'
    samples = json.loads(path.read_text())
    if len(samples) != len(predictions):
        raise ValueError('HalluQA prediction coverage mismatch')
    answers = []
    provided = iter(predictions)
    for sample in samples:
        answer = next(provided)
        answers.append({'question_id':sample['question_id'],'response':answer})
    save(folder/'responses.json', answers)
    save(folder/'score-dataset-map.json', {'version':1,'root':'/','datasets':mappings})
    env = dict(os.environ, DATASET_MAP=str(folder/'score-dataset-map.json'), OPENAI_API_KEY=os.environ[config['judge']['api_key_env']],
               OPENAI_BASE_URL=config['judge']['base_url'], JUDGE_MODEL=config['judge']['model'])
    run([sys.executable,str(SCRIPTS/'halluqa_eval.py'),'--responses',str(folder/'responses.json'),
         '--concurrency',str(config.get('judge_concurrency', 1)),
         '--output-dir',str(folder/'score'),'--timeout',str(config['judge'].get('timeout',60))], folder, env,
        score_report=folder/'score/run.json')
    result = json.loads((folder/'score/run.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('HalluQA incomplete')
    return [['HalluQA','full','non_hallucination_rate (%)',result['non_hallucination_rate'],'halluqa/score/run.json']]


def setup_snapshot(config, pinned):
    packages = dict(sorted((d.metadata['Name'],d.version) for d in distributions() if d.metadata['Name']))
    dependency_locks = {p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                               'requirements':p.read_text().splitlines()}
                        for p in sorted(SCRIPTS.parent.glob('requirements*.lock'))}
    roles = {}
    for role in ('target', 'judge'):
        item = config[role]
        roles[role] = {key: item[key] for key in ('base_url', 'model', 'api_key_env')}
        roles[role].update(temperature=item.get('temperature', 0),
                          max_tokens=item.get('max_tokens', 2048 if role == 'target' else 1024),
                          timeout=item.get('timeout', 60))
    commits = {}
    for name in ('HalluQA', 'FollowBench'):
        repo = Path('/opt/benchmarks') / name
        if (repo/'REVISION').is_file():
            commits[name] = (repo/'REVISION').read_text().strip()
        elif (repo/'.git').exists():
            commits[name] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, timeout=5).strip()
    return {
        'network_policy': {'resources': 'local_only', 'api_allowlist': False},
        'benchmark_commits': {**commits, 'OpenCompass':__import__('compass_runner').REVISION, 'CMRC2018':'c0eb1b6ba219847457e6af3180da722bbeb656af', 'FinSafetyBench':'8dcdf791ac2be782d72f31d4f8e8ad44be54b6f5'},
        'protocols': {'logiqa-zh':'v1 Chinese test, zero-shot generation, strict A/B/C/D', 'cmrc2018':'official v5 EM/F1, Python3 Unicode port', 'longbench':'v1 raw chat prompts; no truncation', 'finsafetybench':'direct prompts, official English judge prompt; ASR score==10; invalid judge output fails'},
        'created_at': datetime.now(timezone.utc).isoformat(),
        'stages': selected_stages(config),
        **roles,
        'concurrency': config.get('concurrency', 1),
        'judge_concurrency': config.get('judge_concurrency', 1),
        'opencompass': {'revision': __import__('compass_runner').REVISION, 'options': config.get('options', {}), 'transport': 'OpenAI chat completions; full input; stop and length accepted'},
        'followbench': {'language': config.get('followbench', {}).get('language', 'en'),
                        'judge_invalid_attempts': 3} if 'followbench' in selected_stages(config) else None,
        'halluqa': {'votes': 5, 'temperature': 0, 'top_p': 0.5, 'max_tokens': 10,
                    'judge_invalid_attempts': 5} if 'halluqa' in selected_stages(config) else None,
        'finsafetybench': {'language':config.get('options',{}).get('finsafetybench',{}).get('language','en'),
                          'mode':'direct', 'asr_rating':10, 'judge_invalid_attempts':3}
                          if 'finsafetybench' in selected_stages(config) else None,
        'continue_on_error': config.get('continue_on_error', True),
        'dataset_map_config': config.get('dataset_map') or os.environ.get('DATASET_MAP'),
        'dataset_paths': pinned,
        'python': platform.python_version(), 'packages': packages, 'dependency_locks': dependency_locks,
        'runner_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(SCRIPTS.glob('*.py'))},
        'cache_paths': {key: os.environ.get(key) for key in
                        ('MODELSCOPE_CACHE','HF_HOME','COMPASS_DATA_CACHE','DATASET_WORK_CACHE')},
        'transport': {'custom_api_attempts': 3, 'sdk_retries': 0,
                      'note': 'All datasets use the same bounded transport' },
    }


def _main():
    global ACTIVE_RUN, ACTIVE_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='Load data and prepare prompts without API calls')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group-id', type=UUID, help=argparse.SUPPRESS)
    opts = parser.parse_args()
    config = json.loads(opts.config.read_text())
    ACTIVE_CONFIG = config
    stages = selected_stages(config)
    if 'stages' in config or 'evalscope' in config:
        parser.error('Legacy stages/evalscope configuration is unsupported; use top-level datasets and options')
    for role in ('target','judge'):
        for key in ('base_url','model','api_key_env'):
            if not config[role].get(key):
                parser.error(f'Missing {role}.{key}')
        if not opts.validate_only and not os.environ.get(config[role]['api_key_env']):
            parser.error(f"Missing environment variable {config[role]['api_key_env']}")
        if 'api_key' in config[role]:
            parser.error('Do not put API keys in config; use api_key_env')
    root = opts.output.resolve()
    from run_inputs import create_run
    mapping_file = config.get('dataset_map') or os.environ.get('DATASET_MAP')
    mappings = read_mapping(mapping_file)
    selected_keys = [('followbench_'+config.get('followbench',{}).get('language','en')
                      if name == 'followbench' else name) for name in stages]
    root = create_run(root, [opts.config,mapping_file,*(mappings.get(key) for key in selected_keys)],
                      group_id=opts.group_id, phase='accuracy')
    from run_journal import RunJournal
    ACTIVE_RUN = journal = RunJournal(root, stages, STAGE_ORDER)
    validate_config(config)
    setup = setup_snapshot(config, {})
    setup.update(tasks={}, snapshot_complete=False)
    journal.checkpoint('mapping', setup=setup)
    # Pin all paths once for the entire sequence.
    if not mappings:
        raise ValueError('Suite requires dataset_map or DATASET_MAP')
    pinned = preflight(config, mappings, on_resolve=lambda name: journal.checkpoint('path_resolution', name))
    from run_inputs import restrict_run_readers
    restrict_run_readers(root,[mappings[key] for key in pinned] + list(pinned.values()))
    setup['dataset_paths'] = pinned
    journal.checkpoint('preparation', setup=setup)
    from offline_runtime import configure
    configure([config[role]['base_url'] for role in ('target', 'judge')])
    save(root/'dataset-map.json', {'version':1,'root':'/','datasets':pinned})
    os.environ['DATASET_MAP'] = str(root/'dataset-map.json')
    save(root/'config.json',config)
    from compass_runner import prepare, execute
    from run_inputs import freeze_tasks
    prepared = {}
    for name in stages:
        options = dict(config.get('options', {}).get(name, {}))
        if name == 'followbench': options.update(config.get('followbench', {}))
        key = 'followbench_'+options.get('language','en') if name=='followbench' else name
        journal.checkpoint('preparation', name)
        prepared[name] = freeze_tasks(root, name, prepare(name, pinned[key], options))
        setup['tasks'][name] = [{'subset':t['config']['abbr'], 'count':t['count'],
            'sample_ids':t['ids'], 'config':json.loads(json.dumps(t['config'], default=lambda x: x.__module__+'.'+x.__name__)),
            'input_file':str(Path(t['input_path']).relative_to(root)), 'input_sha256':t['input_sha256']}
            for t in prepared[name]]
        journal.checkpoint('preparation', name, setup=setup)
    setup['snapshot_complete'] = True
    journal.checkpoint('preparation', setup=setup)
    journal.ready(setup, validate_only=opts.validate_only)
    if opts.validate_only:
        save(root/'validation.json', {'status':'passed', 'datasets':stages, 'api_calls':0})
        return 0
    for name in stages:
        journal.checkpoint('execution', name)
        folder = root/name
        folder.mkdir()
        journal.start(name)
        try:
            result = execute(name,prepared[name],config,folder,pinned)
        except OSError:
            # Artifact integrity is unknown. Never advance to the next dataset.
            raise
        except Exception as exc:
            partial = folder/'metrics.partial.json'
            result = json.loads(partial.read_text()) if partial.exists() else []
            detail = failure_details(exc, config)
            (folder/'error.txt').write_text(detail)
            journal.error(exc, detail)
            journal.record(name, result, error=exc)
            print(f'{name} failed: {type(exc).__name__}; see stage artifacts',file=sys.stderr)
            if not config.get('continue_on_error',True):
                break
        else:
            journal.record(name, result)
    journal.finish()
    return int(journal.data['overall_status'] != 'complete')


def finalize_failure(exc, interrupted):
    journal = ACTIVE_RUN
    detail = failure_details(exc, ACTIVE_CONFIG) if journal is not None else str(exc)
    if isinstance(exc, OSError):
        print(detail, file=sys.stderr, flush=True)
    if journal is None:
        return
    name = journal.data['active_dataset']
    if name and journal.data['status'][name] == 'running':
        partial = journal.root/name/'metrics.partial.json'
        try:
            journal.merge_metrics(name, json.loads(partial.read_text()))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as err:
            print(f'Could not restore {partial}: {err}', file=sys.stderr, flush=True)
    try:
        journal.error(exc, detail, flush=False)
        journal.terminate(interrupted=interrupted)
    except OSError as err:
        print(f'Could not finalize {journal.root / "summary.json"}: {err}', file=sys.stderr, flush=True)


def main():
    global ACTIVE_RUN
    ACTIVE_RUN = None
    previous = signal.signal(signal.SIGTERM, terminate_handler)
    try:
        return _main()
    except KeyboardInterrupt as exc:
        finalize_failure(exc, interrupted=True)
        return 143 if isinstance(exc, SuiteInterrupted) else 130
    except Exception as exc:
        finalize_failure(exc, interrupted=False)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def cli():
    try:
        code = main()
    except OSError:
        # main already reported the error to stderr.
        return 1
    if code in (130, 143):
        # The journal and partial files are closed. Interpreter shutdown would
        # otherwise join request threads that may still be blocked in socket I/O.
        os._exit(code)
    return code


if __name__ == '__main__':
    # Scorers import run_suite during execution and share this run's journal.
    sys.modules['run_suite'] = sys.modules[__name__]
    sys.exit(cli())
