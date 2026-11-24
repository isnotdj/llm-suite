"""Sequence independent vLLM bench serve calls. vLLM owns sampling, traffic and metrics."""
import asyncio
import hashlib
import json
import math
import os
import re
from tempfile import NamedTemporaryFile
from pathlib import Path
from uuid import uuid4

from perf_bench_serve import count_checksum, result_metrics, run_native
from perf_stage_inputs import freeze_inputs, validate_stage_input_options
from run_inputs import restrict_run_readers
from run_journal import atomic_json


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def levels(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{name} must be a nonempty array')
    for value in values:
        positive(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f'{name} contains duplicate levels')
    return values


def options(raw):
    if not isinstance(raw, dict):
        raise ValueError('vllm_bench_serve must be an object')
    result = {key.replace('-', '_'): value for key, value in raw.items()}
    if len(result) != len(raw):
        raise ValueError('Duplicate native option after normalizing hyphens')
    controlled = {'num_prompts', 'max_concurrency', 'request_rate', 'ready_check_timeout_sec',
                  'ramp_up_strategy', 'ramp_up_start_rps', 'ramp_up_end_rps', 'profile'}
    if controlled & result.keys():
        raise ValueError(f'Stage orchestration controls {sorted(controlled & result.keys())}')
    return result


def native_inputs(raw):
    """Include per-profile input overrides when checking output isolation."""
    inputs = list(raw['immutable_snapshots']) if isinstance(raw.get('immutable_snapshots'), list) else []
    blocks = [raw.get('vllm_bench_serve')]
    profiles = raw.get('profiles', [])
    if isinstance(profiles, list):
        blocks.extend(p.get('vllm_bench_serve') for p in profiles if isinstance(p, dict))
    for block in blocks:
        if isinstance(block, dict):
            for key, value in block.items():
                if key.replace('-', '_') in ('model', 'tokenizer', 'dataset_path'):
                    inputs.append(value)
                if key.replace('-', '_') == 'lora_modules' and isinstance(value, list):
                    inputs.extend(value)
    return inputs


def workloads(raw):
    allowed = {'vllm_bench_serve', 'profiles', 'concurrency', 'requests_per_concurrency',
               'overload_concurrency', 'soak_requests', 'slo', 'immutable_snapshots'}
    if raw.keys() - allowed:
        raise ValueError(f'Unknown native stage fields: {sorted(raw.keys() - allowed)}')
    base = options(raw['vllm_bench_serve'])
    profiles = raw['profiles']
    if not isinstance(profiles, list) or not profiles:
        raise ValueError('profiles must be a nonempty array')
    result = {}
    seen = set()
    for profile in profiles:
        if not isinstance(profile, dict) or profile.keys() - {'name', 'vllm_bench_serve'}:
            raise ValueError('A profile accepts name and vllm_bench_serve')
        name = profile.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name.casefold() in seen | {'inputs'}:
            raise ValueError(f'Invalid or duplicate profile name: {name!r}')
        seen.add(name.casefold())
        opts = {**base, **options(profile.get('vllm_bench_serve', {}))}
        if not opts.get('model') or not opts.get('dataset_name'):
            raise ValueError(f'{name}: set native model and dataset_name')
        # Request the official metrics used for sweet-spot selection.
        for key, default, required in [('metric_percentiles', '99', ('95',)),
                                       ('percentile_metrics', 'ttft,tpot,itl', ('ttft','tpot'))]:
            value = opts.get(key, default)
            if not isinstance(value, str):
                raise ValueError(f'{key} must be a comma-separated string')
            values = [v.strip() for v in value.split(',')]
            if key == 'metric_percentiles' and not all(0 <= float(v) <= 100 for v in values):
                raise ValueError(f'{name}: metric_percentiles must be within [0, 100]')
            if key == 'percentile_metrics' and set(values) - {'ttft', 'tpot', 'itl', 'e2el'}:
                raise ValueError(f'{name}: invalid percentile_metrics')
            opts[key] = ','.join(dict.fromkeys([*values, *required]))
        if (str(opts.get('endpoint', '')).endswith('/v1/embeddings')
                or 'openai-embeddings' in (opts.get('backend'), opts.get('endpoint_type'))
                or int(opts.get('random_batch_size', 1)) != 1):
            raise ValueError(f'{name}: staged benchmarks require generation requests; use single-run mode for embeddings')
        result[name] = opts
    return result


def assess_point(point, slo):
    """Explain whether a completed scan or soak has usable evidence and meets its SLO."""
    if not point.get('requests', {}).get('tpot_samples'):
        return 'no_tpot_samples'
    metrics = point.get('metrics', {})
    keys = ['p95_ttft_ms', 'p95_tpot_ms']
    if point['phase'].startswith('scan-'):
        keys.append('output_throughput')
    for key in keys:
        value = metrics.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            return f'unusable_{key}'
    if metrics['p95_ttft_ms'] > slo['ttft_ms'] or metrics['p95_tpot_ms'] > slo['tpot_ms']:
        return 'slo_exceeded'
    return 'passed'


def planned_rounds(profiles, stage, scan, factor, overload, soak):
    rounds = []

    def add(name, phase, concurrency, count):
        rounds.append(dict(profile=name, phase=phase, concurrency=concurrency, expected=count,
                           sequence=len(rounds) + 1))

    for name in profiles:
        add(name, 'connectivity', 1, 1)
    if stage in ('all', 'scan', 'soak'):
        for name in profiles:
            for c in scan:
                add(name, f'scan-c{c}', c, c * factor)
    if stage in ('all', 'overload'):
        for name in profiles:
            for c in overload:
                for n in range(1, 4):
                    add(name, f'overload-c{c}-r{n}', c, c)
    if stage in ('all', 'soak'):
        for name in profiles:
            add(name, 'soak', None, soak)

    return rounds


def run_stages(raw, root, stage, validate_only, summary):
    summary.update(operation='validate_config', run_id=uuid4().hex)
    profiles = workloads(raw)
    scan = levels(raw.get('concurrency', [1, 2, 4, 8, 16, 32, 64, 128]), 'concurrency')
    factor = positive(raw.get('requests_per_concurrency', 10), 'requests_per_concurrency')
    overload = raw.get('overload_concurrency')
    if overload is not None or stage in ('all', 'overload'):
        levels(overload, 'overload_concurrency')
    soak = positive(raw.get('soak_requests', 1000), 'soak_requests')
    slo = raw.get('slo', {'ttft_ms': 2000, 'tpot_ms': 50})
    if not isinstance(slo, dict) or set(slo) != {'ttft_ms', 'tpot_ms'} or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in slo.values()):
        raise ValueError('slo requires positive ttft_ms and tpot_ms')
    sources = [Path(opts['dataset_path']) for opts in profiles.values()
               if opts.get('dataset_path') and Path(opts['dataset_path']).exists()]
    restrict_run_readers(root, sources)
    summary.update(mode='native_stages', profiles={}, overall_status='preparing',
                   stage=stage, concurrency=scan, overload_concurrency=overload,
                   requests_per_concurrency=factor, soak_requests=soak, slo=slo)
    rounds = planned_rounds(profiles, stage, scan, factor, overload, soak)
    counts = {r['expected'] for r in rounds if r['phase'] != 'connectivity'}

    for name, opts in profiles.items():
        validate_stage_input_options(name, opts)
        folder = root/name
        folder.mkdir()
        run_native({'vllm_bench_serve': opts}, folder, 'all', True, {})
        if opts['dataset_name'] == 'prefix_repetition':
            prefixes = int(opts.get('prefix_repetition_num_prefixes', 10))
            if prefixes < 1 or any(count % prefixes for count in counts):
                raise ValueError(f'{name}: measured request counts must be multiples of prefix_repetition_num_prefixes ({prefixes})')
        summary['profiles'][name] = {'runs': []}
    snapshots = raw.get('immutable_snapshots', [])
    if not isinstance(snapshots, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in snapshots):
        raise ValueError('immutable_snapshots requires absolute snapshot paths')
    if validate_only:
        return 'validated'

    summary['operation'] = 'freeze_inputs'
    inputs = freeze_inputs(root, profiles, snapshots)
    plan = dict(schema='native_stages_v4', run_id=summary['run_id'], profiles=list(profiles), stage=stage,
                concurrency=scan, requests_per_concurrency=factor, overload_concurrency=overload,
                soak_requests=soak, slo=slo, near_peak_ratio=0.95, inputs=inputs, rounds=rounds)
    summary['operation'] = 'commit_plan'
    atomic_json(root/'setup.json', plan)
    for entry in rounds:
        name, phase = entry['profile'], entry['phase']
        summary.update(active_profile=name, phase=phase, operation='create_stage_directory')
        folder = root/name/phase
        folder.mkdir()
        summary['operation'] = 'prepare_stage'
        c = entry['concurrency']
        identity = dict(run_id=plan['run_id'], sequence=entry['sequence'])
        if phase == 'soak':
            selection_status, c = read_selection(root, plan, name)
            if selection_status != 'committed':
                raise ValueError(f'{name}: scan selection is {selection_status}; cannot start soak')
            if c is None:
                atomic_json(folder/'summary.json', dict(identity, overall_status='skipped_no_sweet_spot'))
                continue
        child = dict(identity, mode='vllm_bench_serve', overall_status='preparing', concurrency=c, errors=[])
        atomic_json(folder/'summary.json', child)
        params = dict(profiles[name], num_prompts=entry['expected'], max_concurrency=c,
                      request_rate='inf', ready_check_timeout_sec=0)
        if phase == 'connectivity' and params['dataset_name'] == 'prefix_repetition':
            params['prefix_repetition_num_prefixes'] = 1
        try:
            summary['operation'] = 'run_benchmark'
            status = run_native({'vllm_bench_serve': params}, folder, 'all', False, child,
                                measured_count=entry['expected'])
        except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
            # Base recovery on the last successful write, never on speculative memory.
            saved = json.loads((folder/'summary.json').read_text())
            interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            saved['overall_status'] = 'interrupted' if interrupted else 'failed'
            if not saved.get('measurement_status') and 'requests' in child:
                saved['requests'] = dict(child['requests'])
                saved['request_count_status'] = 'partial'
            if saved.get('traffic_status') == 'running':
                saved['traffic_status'] = saved['overall_status']
            if (saved.get('artifacts', {}).get('export') == 'pending' and not interrupted
                    and child.get('artifacts', {}).get('export') == 'failed'):
                saved['artifacts']['export'] = 'failed'
            detail = str(exc)
            secrets = [os.environ.get('OPENAI_API_KEY')] + [h.partition('=')[2] for h in params.get('header', [])]
            for value in filter(None, secrets):
                detail = detail.replace(value, '[REDACTED]')
            saved['errors'] = [{'type': type(exc).__name__, 'detail': detail}]
            try:
                atomic_json(folder/'summary.json', saved)
            except OSError:
                if not interrupted:
                    raise
            raise
        if phase == f'scan-c{scan[-1]}':
            summary['operation'] = 'commit_selection'
            commit_selection(root, plan, name)
        if phase == 'connectivity' and status != 'complete':
            raise ValueError(f'{name}: connectivity failed')
    return 'complete'


def stage_record_error(child, schema):
    if not isinstance(child, dict):
        return 'Unreadable stage summary'
    for field in ('artifacts', 'artifact_signatures', 'metrics', 'requests'):
        if field in child and not isinstance(child[field], dict):
            return f'Invalid stage field: {field}'
    for field in ('overall_status', 'measurement_status', 'request_count_status'):
        if field in child and not isinstance(child[field], str):
            return f'Invalid stage field: {field}'
    artifacts = child.get('artifacts', {})
    if ('export' in artifacts or (schema == 'native_stages_v4' and child.get('measurement_status'))) and (
            artifacts.get('export') not in ('complete', 'pending', 'failed', 'not_requested', 'not_generated')):
        return 'Invalid stage field: artifacts.export'
    errors = child.get('errors', [])
    if not isinstance(errors, list) or any(not isinstance(e, dict) for e in errors):
        return 'Invalid stage field: errors'
    return None


def export_integrity(folder, child):
    from perf_report_jobs import _inside
    from perf_resources import artifact_signature
    if 'export_files' not in child:
        # Older runs recorded an absolute destination without a content signature.
        path = child.get('export_path')
        if isinstance(path, str) and Path(path).is_relative_to(folder/'exports'):
            relative = Path(path).relative_to(folder)
            if not _inside(folder, relative).is_file():
                return 'missing'
        return 'unverified'
    files = child['export_files']
    if not isinstance(files, dict) or not files:
        return 'invalid'
    for name, expected in files.items():
        relative = Path(name)
        if not relative.is_relative_to('exports') or '..' in relative.parts:
            return 'invalid'
        path = _inside(folder, relative)
        if not path.is_file():
            return 'missing'
        if artifact_signature(path) != expected:
            return 'corrupt'
    return 'verified'


def read_round(root, entry, plan, selection=None):
    from perf_report_jobs import _inside
    schema = plan['schema']
    relative = Path(entry['profile'])/entry['phase']
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', entry[key]) for key in ('profile', 'phase')):
        raise ValueError('Invalid stage path in run plan')
    folder = _inside(root, relative)
    record = dict(entry)
    child_file = _inside(root, relative/'summary.json')
    try:
        child = json.loads(child_file.read_text()) if child_file.is_file() else {}
    except ValueError:
        child = None
    error = stage_record_error(child, schema)
    if error:
        return dict(record, status='invalid_stage_record', errors=[dict(detail=error)])
    status = child.get('overall_status')
    if schema in ('native_stages_v3', 'native_stages_v4') and child_file.is_file():
        mismatch = round_mismatch(plan, entry, child, selection)
        if mismatch:
            return dict(record, status='round_mismatch', errors=[dict(detail=mismatch)])
    if schema in ('native_stages_v3', 'native_stages_v4') and not child_file.is_file():
        return dict(record, status='unconfirmed' if folder.exists() else 'not_started_or_unrecorded',
                    request_count_status='unknown', requests=dict(selected=None, dispatched=None, successful=None))
    if status is None:
        status = 'unconfirmed' if folder.exists() else 'not_started_or_unrecorded'
    elif status in ('running', 'preparing', 'export_pending'):
        status = 'unconfirmed'
    record.update(status=status)
    for key in ('measurement_status', 'concurrency', 'requests', 'request_count_status', 'metrics', 'errors'):
        if key in child:
            record[key] = child[key]
    if 'measurement_status' not in record and status not in ('not_started_or_unrecorded', 'skipped_no_sweet_spot'):
        record['measurement_status'] = 'unconfirmed'
    record['export_status'] = child.get('artifacts', {}).get('export', 'unknown')
    if record['export_status'] == 'complete':
        try:
            record['export_integrity'] = export_integrity(folder, child)
        except ValueError:
            record['export_integrity'] = 'invalid'
    if record['export_status'] == 'pending':
        record['export_status'] = 'unconfirmed'
    if 'request_count_status' not in record:
        if child.get('measurement_status'):
            record['request_count_status'] = 'final'
        else:
            selected = child.get('requests', {}).get('selected')
            record.update(request_count_status='unknown',
                          requests=dict(selected=selected, dispatched=None, successful=None))
    from perf_resources import artifact_signature
    integrity = record['artifact_integrity'] = {}
    for key, filename in [('report', 'vllm-report.txt'), ('result', 'vllm-result.json')]:
        path = _inside(root, relative/filename)
        expected = child.get('artifact_signatures', {}).get(key)
        if path.is_file():
            record[key] = str(relative/filename)
            if expected is None:
                integrity[key] = ('missing_signature' if schema != 'native_stages_v1' and child.get('measurement_status')
                                  else 'unverified')
            else:
                integrity[key] = 'verified' if artifact_signature(path) == expected else 'corrupt'
        elif expected or child.get('measurement_status'):
            integrity[key] = 'missing'
    measurement_complete = False
    if {'corrupt', 'missing_signature'} & set(integrity.values()):
        record['status'] = 'corrupt_artifact'
    elif 'missing' in integrity.values():
        record['status'] = 'missing_artifact'
    elif integrity.get('result') == 'verified' and child.get('measurement_status'):
        result = json.loads((root/record['result']).read_text())
        record['metrics'] = result_metrics(result)
        counts = child.get('requests', {})
        coverage = counts.get('tpot_samples') if isinstance(counts, dict) else None
        successful = counts.get('successful') if isinstance(counts, dict) else None
        valid = (type(coverage) is int and type(successful) is int and
                 0 <= coverage <= successful == result.get('completed'))
        if schema == 'native_stages_v4':
            valid = valid and child.get('count_sha256') == count_checksum(child)
        measurement_complete = valid and child['measurement_status'] == 'complete'
        if not valid:
            record.update(status='invalid_stage_record', errors=[dict(detail='Invalid measurement counts')])
    elif schema in ('native_stages_v1', 'native_stages_v2') and record['status'] == 'complete':
        # Historical runs may only have the stage summary as measurement evidence.
        measurement_complete = True
    if measurement_complete and (entry['phase'].startswith('scan-') or entry['phase'] == 'soak'):
        record['acceptance'] = assess_point(record, plan['slo'])
    return record


def select_sweet(points, plan):
    if (len(points) != len(plan['concurrency']) or
            any(p.get('measurement_status') not in ('complete', 'incomplete', 'failed')
                or p['export_status'] not in ('complete', 'not_requested') for p in points)):
        return None
    eligible = [(p['concurrency'], p['metrics']['output_throughput'])
                for p in points if p.get('acceptance') == 'passed']
    peak = max((rate for _, rate in eligible), default=0)
    return min((c for c, rate in eligible if rate >= plan['near_peak_ratio'] * peak), default=None)


def round_mismatch(plan, entry, child, selection):
    if (child.get('run_id') != plan['run_id'] or type(child.get('sequence')) is not int
            or child['sequence'] != entry['sequence']):
        return 'Stage run_id or sequence does not match the plan'
    expected = entry['concurrency']
    if entry['phase'] == 'soak' and selection and selection[0] == 'committed':
        expected = selection[1]
        if (child.get('overall_status') == 'skipped_no_sweet_spot') != (expected is None):
            return 'Soak execution conflicts with the committed selection'
    if expected is not None and (type(child.get('concurrency')) is not int or child['concurrency'] != expected):
        return 'Stage concurrency does not match its planned or selected concurrency'
    if child.get('overall_status') == 'complete' and child.get('measurement_status') != 'complete':
        return 'Completed stage has no matching measurement commit'
    if child.get('measurement_status') == 'complete':
        counts = child.get('requests', {})
        for key in ('selected', 'dispatched', 'successful'):
            if not isinstance(counts, dict) or type(counts.get(key)) is not int or counts[key] != entry['expected']:
                return f'Completed stage {key} does not match the planned request count'
    return None


def scan_rounds(plan, name):
    return [r for r in plan['rounds'] if r['profile'] == name and r['phase'].startswith('scan-')]


def selection_checksum(decision):
    payload = json.dumps(decision, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def commit_selection(root, plan, name):
    rounds = scan_rounds(plan, name)
    points = [read_round(root, entry, plan) for entry in rounds]
    if any(p['status'] not in ('complete', 'failed', 'incomplete') or
           p.get('measurement_status') != p['status'] for p in points):
        raise ValueError(f'{name}: cannot commit selection from unconfirmed scan results')
    decision = dict(run_id=plan['run_id'], profile=name, scan_sequences=[r['sequence'] for r in rounds],
                    sweet_concurrency=select_sweet(points, plan))
    atomic_json(root/name/'selection.json', dict(decision, sha256=selection_checksum(decision)))


def read_selection(root, plan, name):
    from perf_report_jobs import _inside
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Invalid profile in run plan')
    try:
        file = _inside(root, Path(name)/'selection.json')
        decision = json.loads(file.read_text())
    except FileNotFoundError:
        return 'unconfirmed', None
    except (ValueError, IsADirectoryError):
        return 'invalid', None
    if not isinstance(decision, dict):
        return 'invalid', None
    checksum = decision.pop('sha256', None)
    c = decision.get('sweet_concurrency')
    if (checksum != selection_checksum(decision) or decision.get('run_id') != plan['run_id']
            or decision.get('profile') != name or 'sweet_concurrency' not in decision
            or decision.get('scan_sequences') != [r['sequence'] for r in scan_rounds(plan, name)]
            or (c is not None and (type(c) is not int or c not in plan['concurrency']))):
        return 'invalid', None
    return 'committed', c


def publish_reports(root):
    from perf_report_jobs import _inside
    plan = json.loads(_inside(root, 'setup.json').read_text())
    if plan['schema'] not in ('native_stages_v1', 'native_stages_v2', 'native_stages_v3', 'native_stages_v4'):
        raise ValueError('Unknown native stage plan')
    summary = {key: plan[key] for key in ('stage', 'concurrency', 'requests_per_concurrency',
                                          'overload_concurrency', 'soak_requests', 'slo')}
    summary.update(mode='native_stages', profiles={n: {'runs': []} for n in plan['profiles']}, errors=[])
    if 'run_id' in plan:
        summary['run_id'] = plan['run_id']
    index = []
    statuses = set()
    if plan['schema'] in ('native_stages_v3', 'native_stages_v4'):
        if plan['stage'] not in ('all', 'connectivity', 'scan', 'overload', 'soak'):
            raise ValueError('Unknown stage in saved plan')
        expected = planned_rounds(plan['profiles'], plan['stage'], plan['concurrency'],
                                  plan['requests_per_concurrency'], plan['overload_concurrency'], plan['soak_requests'])
        if not expected or plan.get('rounds') != expected:
            summary['errors'].append(dict(detail='Saved rounds differ from the configured stage plan'))
            statuses.add('invalid_stage_record')
            plan['rounds'] = expected
    selections = {}
    if plan['stage'] in ('all', 'scan', 'soak'):
        for name, state in summary['profiles'].items():
            selection = (read_selection(root, plan, name) if plan['schema'] in ('native_stages_v3', 'native_stages_v4')
                         else ('not_recorded', None))
            selections[name] = selection
            state['selection_status'], state['sweet_concurrency'] = selection
            if selection[0] in ('invalid', 'unconfirmed'):
                statuses.add('invalid_selection' if selection[0] == 'invalid' else 'unconfirmed')
    with NamedTemporaryFile(dir=root, delete=False) as output:
        tmp = Path(output.name)
    try:
        with tmp.open('wb') as output:
            for entry in plan['rounds']:
                name, phase = entry['profile'], entry['phase']
                record = read_round(root, entry, plan, selections.get(name))
                state = summary['profiles'][name]
                state['runs'].append(record)
                status = record['status']
                statuses.add(status)
                if phase == 'soak':
                    soak_status = status
                    if 'acceptance' in record:
                        soak_status = 'complete' if record['acceptance'] == 'passed' else 'failed_acceptance'
                    state['soak_status'] = soak_status
                    statuses.add(soak_status)
                if record.get('export_integrity') in ('missing', 'invalid', 'corrupt'):
                    statuses.add('export_artifact_error')
                    summary['errors'].append(dict(profile=name, phase=phase,
                                                  detail='Export archive: ' + record['export_integrity']))
                summary['errors'].extend(dict(error, profile=name, phase=phase) for error in record.get('errors', []))
                if 'report' in record or 'result' in record:
                    item = dict(profile=name, phase=phase, artifact_integrity=record['artifact_integrity'])
                    if 'report' in record:
                        item['text'] = record['report']
                        output.write(f'\n[{name}/{phase}]\n'.encode())
                        if record['artifact_integrity']['report'] in ('corrupt', 'missing_signature'):
                            output.write(f"[{record['artifact_integrity']['report']} vllm-report.txt]\n".encode())
                        else:
                            output.write((root/record['report']).read_bytes())
                    if 'result' in record:
                        item['json'] = record['result']
                    index.append(item)
        tmp.replace(_inside(root, 'vllm-report.txt'))
    finally:
        tmp.unlink(missing_ok=True)
    for state in summary['profiles'].values():
        if state.get('selection_status') == 'committed':
            scans = [r for r in state['runs'] if r['phase'].startswith('scan-')]
            state['selection_evidence'] = ('verified' if all(
                r.get('artifact_integrity') == {'report': 'verified', 'result': 'verified'}
                and r['status'] not in ('invalid_stage_record', 'round_mismatch')
                and r.get('measurement_status') in ('complete', 'failed', 'incomplete') for r in scans)
                else 'unconfirmed')
    overall = 'complete' if statuses == {'complete'} else 'incomplete'
    if 'interrupted' in statuses:
        overall = 'interrupted'
    elif statuses & {'failed', 'export_artifact_error', 'missing_artifact', 'corrupt_artifact', 'round_mismatch', 'invalid_selection', 'invalid_stage_record'}:
        overall = 'failed'
    error_file = _inside(root, 'run-error.json')
    if error_file.is_file():
        try:
            error = json.loads(error_file.read_text())
        except ValueError:
            error = None
        if not isinstance(error, dict) or error.get('status') not in ('interrupted', 'failed'):
            summary['errors'].append(dict(detail='Invalid run-error.json'))
            overall = 'failed'
        elif (error.get('run_id') and error['run_id'] != plan.get('run_id')):
            summary['warnings'] = ['Ignored run-error.json from another run']
        elif plan['schema'] == 'native_stages_v4' and not error.get('run_id'):
            summary['errors'].append(dict(detail='Missing run_id in run-error.json'))
            overall = 'failed'
        else:
            summary['run_error'] = error
            if not any(all(error.get(k) == e.get(k) for k in ('detail', 'type', 'profile', 'phase'))
                       for e in summary['errors']):
                summary['errors'].append(error)
            overall = error['status']
    elif statuses & {'unconfirmed', 'not_started_or_unrecorded'} and not summary['errors']:
        summary['termination_reason'] = 'unconfirmed'
    summary['overall_status'] = overall
    atomic_json(_inside(root, 'vllm-reports.json'), index)
    atomic_json(_inside(root, 'summary.json'), summary)
    return summary
