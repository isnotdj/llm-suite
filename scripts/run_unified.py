#!/opt/venv/bin/python
"""Run the configured remote benchmarks and evaluations in one local UUID directory."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

from run_inputs import check_output, require_local_output
from run_journal import atomic_json

PROJECT = Path(__file__).resolve().parents[1]
STAGES = ('all', 'connectivity', 'scan', 'overload', 'soak', 'single')


class ConfigLoader(yaml.SafeLoader):
    def compose_mapping_node(self, anchor):
        node = super().compose_mapping_node(anchor)
        # Check keys before SafeLoader expands YAML merge keys.
        seen = set()
        for key, _ in node.value:
            if key.tag not in ('tag:yaml.org,2002:str', 'tag:yaml.org,2002:merge'):
                raise ValueError(f'Expected a string key at line {key.start_mark.line + 1}')
            name = key.value
            if name in seen:
                raise ValueError(f'Duplicate YAML key {name!r} at line {key.start_mark.line + 1}')
            seen.add(name)
        return node


def block(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'{name} requires a mapping with fields from {sorted(allowed)}')
    return dict(value)


def role(value, judge=False):
    defaults = dict(temperature=0, timeout=60, max_tokens=1024 if judge else 2048)
    fields = {'name', 'base_url', 'api_key_env', *defaults}
    if not judge:
        fields.add('tokenizer')
    conf = defaults | block(value, fields, 'judge' if judge else 'model')
    for key in ('name', 'base_url', *(k for k in ('api_key_env', 'tokenizer') if k in conf)):
        if not isinstance(conf.get(key), str) or not conf[key].strip():
            raise ValueError(f'Missing or invalid model field {key}')
    if 'api_key_env' in conf and not os.environ.get(conf['api_key_env']):
        raise ValueError(f'Missing environment variable {conf["api_key_env"]}')
    return conf


def argv(value):
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
        raise ValueError('bench requires an array of strings (quote numbers)')
    return value


def performance(value, model):
    defaults = json.loads((PROJECT/'perf.example.json').read_text())
    profiles = [dict(name=p['name'], bench=[arg for k,v in p['vllm_bench_serve'].items()
               for arg in ('--'+k.replace('_','-'), str(v))]) for p in defaults.pop('profiles')]
    defaults.pop('vllm_bench_serve')
    conf = block(value, {*defaults, 'bench', 'profiles', 'stage', 'immutable_snapshots'}, 'perf')
    stage = conf.get('stage', 'all')
    if stage not in STAGES:
        raise ValueError(f'Unknown perf stage {stage}')
    if stage == 'single' and conf.keys() - {'stage', 'bench'}:
        raise ValueError('single accepts only stage and bench')
    if stage != 'single':
        conf['slo'] = defaults['slo'] | block(conf.get('slo', {}), defaults['slo'], 'perf.slo')
        conf = defaults | dict(immutable_snapshots=[], profiles=profiles) | conf
        if not isinstance(conf['profiles'], list) or not conf['profiles']:
            raise ValueError('perf.profiles requires a nonempty list')
        conf['profiles'] = [block(p, ('name','bench'), 'profile') for p in conf['profiles']]
        for p in conf['profiles']:
            p['bench'] = argv(p.get('bench', []))
            if not isinstance(p.get('name'), str):
                raise ValueError('Every profile requires a name')
    conf['stage'] = stage
    default_args = ['--backend', 'openai-chat', '--dataset-name', 'random', '--seed', '42']
    conf['bench'] = default_args + argv(conf.get('bench', []))
    prefix = ['--model', model['name']]
    if model.get('tokenizer'):
        prefix += ['--tokenizer', model['tokenizer']]
    payload = {k:v for k,v in conf.items() if k != 'stage'}
    payload['model_api_url'] = model['base_url']
    payload['bench'] = prefix + conf['bench']
    return conf, payload


def bench_inputs(payload):
    # Protect every named input before the first directory is created. The
    # native worker subsequently validates the effective argv with vLLM.
    paths = list(payload.get('immutable_snapshots', []))
    for part in [payload, *payload.get('profiles', [])]:
        flag = ''
        for token in part['bench']:
            if token.startswith('--'):
                flag, sep, value = token.partition('=')
                if not sep:
                    continue
            else:
                value = token
            if flag in ('--tokenizer', '--model', '--dataset-path', '--lora-modules'):
                paths.append(value)
                if flag != '--lora-modules':
                    flag = ''
    return paths


def accuracy(value, model, config_dir, *, before_perf=False):
    from dataset_mapping import read_mapping
    from run_suite import preflight
    defaults = dict(datasets=['ceval'], concurrency=1, judge_concurrency=1, continue_on_error=True,
                    followbench={'language':'en'}, options={}, dataset_map=str(PROJECT/'dataset-map.default.json'))
    conf = defaults | block(value, {*defaults, 'judge'}, 'accuracy')
    mapping = config_dir / conf['dataset_map']
    conf['dataset_map'] = str(mapping.resolve(strict=True))
    if 'judge' in conf:
        conf['judge'] = role(conf['judge'], True)
    elif isinstance(conf['datasets'], list) and set(conf['datasets']) & {'finsafetybench', 'halluqa', 'followbench'}:
        raise ValueError('Selected dataset requires execution.accuracy.judge')
    def api(item):
        return {**{k:v for k,v in item.items() if k not in ('name','tokenizer')}, 'model':item['name'],
                'api_key_env':item.get('api_key_env','OPENAI_API_KEY')}
    conf['followbench'] = block(conf['followbench'], ('language',), 'accuracy.followbench')
    if not isinstance(conf['options'], dict) or any(not isinstance(v, dict) for v in conf['options'].values()):
        raise ValueError('accuracy.options requires a mapping of dataset options')
    if conf['options'].get('finsafetybench', {}).get('language','both') not in ('en','zh','both'):
        raise ValueError('FinSafetyBench language must be en, zh or both')
    payload = dict(conf, target=api(model), judge=api(conf.get('judge', model)))
    pinned = preflight(payload, read_mapping(mapping))
    if before_perf:
        from compass_runner import load_configs
        for name in conf['datasets']:
            opts = dict(conf['options'].get(name, {}))
            key = name
            if name == 'followbench':
                opts.update(conf['followbench'])
                key += '_' + opts.get('language','en')
            load_configs(name, pinned[key], opts)
    return conf, payload, pinned


def redact(conf):
    safe = deepcopy(conf)
    perf = safe['execution'].get('perf', {})
    for part in [perf, *perf.get('profiles', [])]:
        sensitive = False
        for i, token in enumerate(part.get('bench', [])):
            if token.startswith('--'):
                flag, eq, _ = token.partition('=')
                sensitive = flag in ('--header','--metadata','--base-url','--endpoint')
                if sensitive and eq:
                    part['bench'][i] = flag+'=[REDACTED]'
            elif sensitive:
                part['bench'][i] = '[REDACTED]'
    return safe


def invoke(command, env):
    proc = subprocess.Popen(command, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise


def execute(args):
    from run_suite import validate_config
    source = args.config.resolve(strict=True)
    cfg = block(yaml.load(source.read_text(), Loader=ConfigLoader), ('model','execution'), 'config')
    model = role(cfg.get('model'))
    # Apply the same endpoint and generation checks even for perf-only runs.
    validate_config(dict(target=model, judge=model))
    tasks = block(cfg.get('execution'), ('perf','accuracy'), 'execution')
    if not tasks:
        raise ValueError('Enable execution.perf or execution.accuracy')
    effective = dict(model=model, execution={})
    payloads = {}
    inputs = [source, model.get('tokenizer')]
    if 'perf' in tasks:
        effective['execution']['perf'], payloads['perf'] = performance(tasks['perf'], model)
        inputs.extend(bench_inputs(payloads['perf']))
    if 'accuracy' in tasks:
        conf, payloads['accuracy'], pinned = accuracy(tasks['accuracy'], model, source.parent, before_perf='perf' in tasks)
        effective['execution']['accuracy'] = conf
        inputs.extend([conf['dataset_map'], *pinned.values()])
    root = args.output.resolve()/str(uuid4())
    require_local_output(root)
    check_output(root, inputs)
    root.mkdir(parents=True, mode=0o700)
    print(f'Run directory: {root}', flush=True)
    state = dict(run_id=root.name, overall_status='running', phases=dict.fromkeys(payloads, 'not_started'))
    def save():
        atomic_json(root/'summary.json', state)
    save()
    try:
        if 'accuracy' in payloads:
            snapshot = root/'dataset-map.json'
            atomic_json(snapshot, dict(version=1, root='/', datasets=pinned))
            payloads['accuracy']['dataset_map'] = str(snapshot)
            effective['execution']['accuracy']['dataset_map'] = str(snapshot)
        (root/'effective-config.yaml').write_text(yaml.safe_dump(redact(effective), sort_keys=False, allow_unicode=True))
        with TemporaryDirectory() as tmp:
            for name, data in payloads.items():
                path = Path(tmp)/(name+'.json')
                path.write_text(json.dumps(data))
                cmd = ['perf-suite' if name=='perf' else 'eval-suite','--config',str(path),
                       '--output',str(root.parent),'--group-id',root.name]
                env = os.environ.copy()
                if name == 'perf':
                    env['VLLM_SUITE_INPUTS'] = json.dumps([str(p) for p in inputs if p])
                    cmd += ['--existing-group']
                    stage = effective['execution'][name]['stage']
                    if stage != 'single':
                        cmd += ['--stage',stage]
                    key = env.get(model.get('api_key_env','OPENAI_API_KEY'))
                    if key:
                        env['OPENAI_API_KEY'] = key
                else:
                    env['OPENAI_API_KEY'] = env.get('OPENAI_API_KEY') or 'unused'
                state['phases'][name] = 'running'
                save()
                code = invoke(cmd, env)
                status = 'complete' if code == 0 else 'interrupted' if code in (130,143,-2,-15) else 'failed'
                state['phases'][name] = status
                if code:
                    state['overall_status'] = status
                    return code if code > 0 else 128-code
        state['overall_status'] = 'complete'
        return 0
    finally:
        if state['overall_status'] == 'running':
            status = 'interrupted' if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 'failed'
            state['overall_status'] = status
            for phase in payloads:
                if state['phases'][phase] == 'running':
                    state['phases'][phase] = status
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('.vllm.conf'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/vllm-suite'))
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        return execute(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, TypeError, KeyError, OSError, yaml.YAMLError) as exc:
        print(f'vllm-suite: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
