"""Configuration normalization shared by CLI and tests."""
import copy
import math
import re
from urllib.parse import urlsplit
from perf_data import SOURCES

DEFAULTS = dict(seed=42, sample_count=200, min_samples=20,
    concurrency=[1,2,4,8,16,32,64,128], overload_cap=256,
    slo={'ttft_ms':2000, 'tpot_ms':50},
    timing={'warmup':30, 'scan':120, 'scan_max':600, 'scan_requests':200,
            'overload':180, 'recovery':300, 'recovery_window':30,
            'soak':7200, 'window':60, 'timeout':600, 'no_success':30},
    profiles=[{'name':'4k','input_tokens':4096,'output_target':1536},
              {'name':'16k','input_tokens':16384,'output_target':512},
              {'name':'64k','input_tokens':65536,'output_target':1536}])


def positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0 or (integer and not isinstance(value,int)):
        raise ValueError(f'{name} must be positive' + (' integer' if integer else ''))


def check_keys(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(f'{path} must be an object')
    for key in value.keys() - allowed:
        raise ValueError(f'Unknown configuration field: {path}.{key}')


def normalize(raw):
    known = DEFAULTS.keys() | {'target','dataset','tokenizer','chat_template','context_window','max_generation_tokens'}
    unknown = raw.keys() - known
    if unknown:
        raise ValueError(f'Unknown suite fields: {sorted(unknown)}. Put native options in vllm_bench_serve.')
    check_keys(raw.get('target'), {'base_url','model','api_key_env','extra_body','temperature'}, 'target')
    check_keys(raw.get('dataset'), {'name','path'}, 'dataset')
    check_keys(raw.get('timing', {}), DEFAULTS['timing'].keys() | {'first_token_timeout','stream_idle_timeout'}, 'timing')
    check_keys(raw.get('slo', {}), DEFAULTS['slo'].keys(), 'slo')
    for i, profile in enumerate(raw.get('profiles', [])):
        path = f'profiles[{i}]'
        check_keys(profile, {'name','input_tokens','output_target','variation','slo'}, path)
        check_keys(profile.get('slo', {}), DEFAULTS['slo'].keys(), path+'.slo')
    if 'seed' in raw and type(raw['seed']) is not int:
        raise ValueError('seed must be a JSON integer')
    c = copy.deepcopy(DEFAULTS)
    c.update(copy.deepcopy(raw))
    for key in ('slo','timing'):
        c[key] = {**DEFAULTS[key], **raw.get(key,{})}
    for key in ('context_window','max_generation_tokens','sample_count','min_samples','overload_cap'):
        positive(c[key], key, True)
    c['timing'].setdefault('first_token_timeout', c['timing']['timeout'])
    c['timing'].setdefault('stream_idle_timeout', c['timing']['no_success'])
    if c['min_samples'] > c['sample_count']: raise ValueError('min_samples > sample_count')
    if c['dataset']['name'] not in SOURCES: raise ValueError('Unsupported dataset')
    if c['dataset']['name'] != 'random' and not c['dataset'].get('path'): raise ValueError('Dataset path required')
    tokenizer = c.get('tokenizer')
    if not isinstance(tokenizer,str) or not tokenizer.strip():
        raise ValueError('tokenizer must be a local model directory path')
    t = c['target']
    url = urlsplit(t['base_url'])
    if url.scheme not in ('http','https') or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError('Invalid API base_url')
    for key in ('model','api_key_env'):
        if not isinstance(t.get(key),str) or not t[key]: raise ValueError(f'target.{key} required')
    extra = t.get('extra_body',{})
    if isinstance(extra,dict) and 'chat_template' in extra:
        raise ValueError('target.extra_body.chat_template conflicts with the suite chat_template')
    forbidden = {'messages','model','stream','stream_options','max_tokens','max_completion_tokens','ignore_eos','min_tokens','stop','stop_token_ids','truncate_prompt_tokens','truncation','max_input_tokens'}
    if not isinstance(extra,dict) or forbidden & extra.keys(): raise ValueError('extra_body overrides protected generation fields')
    if extra.get('n', 1) != 1 or isinstance(extra.get('n',1), bool):
        raise ValueError('Only n=1 is supported')
    if not isinstance(extra.get('chat_template_kwargs', {}), dict):
        raise ValueError('chat_template_kwargs must be an object')
    for reserved in ('chat_template','tokenize','add_generation_prompt'):
        if reserved in extra.get('chat_template_kwargs',{}):
            raise ValueError(f'target.extra_body.chat_template_kwargs.{reserved} overrides template rendering')
    temperature = t.get('temperature',0)
    if not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature < 0: raise ValueError('Invalid temperature')
    if not c['concurrency'] or c['concurrency'] != sorted(set(c['concurrency'])): raise ValueError('concurrency must be increasing and unique')
    for n in c['concurrency']: positive(n,'concurrency',True)
    for k,v in c['timing'].items(): positive(v,k, k=='scan_requests')
    if c['timing']['scan_max'] < c['timing']['scan']: raise ValueError('scan_max < scan')
    seen = set()
    for p in c['profiles']:
        if p['name'] == 'inputs': raise ValueError('Profile name inputs is reserved for frozen samples')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',p['name']) or p['name'] in seen: raise ValueError('Invalid/duplicate profile name')
        seen.add(p['name'])
        positive(p['input_tokens'],'input_tokens',True)
        if p.get('output_target') is not None: positive(p['output_target'],'output_target',True)
        p.setdefault('variation',0.1)
        if not 0 <= p['variation'] < 1: raise ValueError('variation must be in [0,1)')
        p['slo'] = {**c['slo'], **p.get('slo',{})}
        for k in ('ttft_ms','tpot_ms'): positive(p['slo'][k],k)
    if not seen: raise ValueError('profiles must be nonempty')
    return c
