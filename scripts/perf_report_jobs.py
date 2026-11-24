"""Durable, offline and repeatable generation of completed-stage reports."""
from contextlib import contextmanager
import fcntl
import json
import os
import re
import shutil
import sys
from uuid import uuid4
from pathlib import Path

from perf_native_report import REPORT_CONTRACT_VERSION
from run_journal import apply_access, atomic_json


def error_detail(exc, config):
    detail = str(exc)
    for role in ('target','judge'):
        name = (config or {}).get(role,{}).get('api_key_env')
        key = os.environ.get(name,'') if name else ''
        if key: detail = detail.replace(key,'[REDACTED]')
    return detail


def sanitize_errors(summary, config):
    for error in summary.get('errors',[]):
        if 'detail' in error: error['detail'] = error_detail(error['detail'],config)


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from None


def _completed_points(state):
    for stage in ('scan', 'burst', 'overload'):
        yield from state.get(stage, [])
    if 'soak' in state:
        yield state['soak']


def _summary_points(summary):
    points = {}
    for profile, state in summary.get('profiles', {}).items():
        for point in _completed_points(state):
            log = point.get('requests_file')
            if not isinstance(log, str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.requests\.jsonl', log):
                raise ValueError(f'Invalid completed stage request log for {profile}: {log!r}')
            key = (profile, log.removesuffix('.requests.jsonl'))
            if key in points:
                raise ValueError(f'Duplicate completed stage: {profile}/{key[1]}')
            points[key] = point
    return points


def save_jobs(root, reports=(), discover=False, required_stages=()):
    path = root/'report-jobs.json'
    manifest = _read_json(path) if path.exists() else dict(schema_version=1,jobs=[])
    known = {job['result_file'] for job in manifest['jobs']}
    candidates = [(name,label,folder/(label+'.json')) for name,label,folder,*_ in reports]
    for entry in required_stages:
        profile,phase = entry['profile'],entry['phase']
        if not all(isinstance(part,str) and part and Path(part).name == part and part not in ('.','..')
                   for part in (profile,phase)):
            raise ValueError('Invalid required stage identity')
        candidates.append((profile,phase,root/profile/(phase+'.json')))
    # Recover the small gap between atomic stage-result and queue writes; also
    # support runs made before report-jobs.json was introduced.
    for result_file in sorted(root.glob('*/*.json')) if discover else ():
        if result_file.parent.name == 'inputs' or result_file.name.endswith('.vllm.json'):
            continue
        if str(result_file.relative_to(root)) in known:
            continue
        # Let the worker parse stage files, so failures are recorded against the job.
        is_stage = result_file.with_suffix('.requests.jsonl').is_file()
        if not is_stage:
            result = _read_json(result_file)
            is_stage = isinstance(result,dict) and 'requests_file' in result and 'duration' in result
        if is_stage:
            candidates.append((result_file.parent.name,result_file.stem,result_file))
    for name,label,result_file in candidates:
        relative = str(result_file.relative_to(root))
        if relative not in known:
            manifest['jobs'].append(dict(profile=name,phase=label,result_file=relative,status='pending'))
            known.add(relative)
    atomic_json(path,manifest)
    return manifest


def _inside(root, relative):
    root = root.resolve()
    path = root/relative
    if not path.is_relative_to(root) or path.resolve() != path:
        raise ValueError(f'Expected an unaliased run path inside {root}: {relative}')
    return path


def _link(path, target):
    """Switch a relative symlink with one rename on the output filesystem."""
    if path.is_symlink() and os.readlink(path) == str(target):
        return
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.symlink_to(target)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _limit_access(path, source, creation_mode=None):
    if creation_mode is None:
        creation_mode = path.stat().st_mode
    # chmod/chown on a multiply linked historical file would also change
    # the other owner's input. Detach the report before repairing permissions.
    if path.is_file() and path.stat().st_nlink > 1:
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(dir=path.parent,delete=False) as private:
            temporary = Path(private.name)
            try:
                with path.open('rb') as old:
                    shutil.copyfileobj(old,private)
                private.flush()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    original = source.stat()
    mode = creation_mode & original.st_mode & 0o777
    if path.is_dir():
        mode |= original.st_mode & 0o2000
    apply_access(path,source,mode)


def _new_version(root, prefix):
    versions = root/'.vllm-reports'
    directory = versions/(prefix+uuid4().hex)
    directory.mkdir(mode=root.stat().st_mode & versions.stat().st_mode & 0o777)
    # mkdir applies the live umask/default ACL. Keep the tree private until
    # the per-file restrictions and index have been installed.
    publish_mode = directory.stat().st_mode & 0o2777
    directory.chmod((publish_mode & 0o2000) | 0o700)
    return directory,publish_mode


def _publish_generation(root, generation, public_files):
    versions = generation.parent
    current = versions/'current'
    published = root/'vllm-reports.json'
    for path in public_files:
        if path.is_symlink() and (not path.resolve().is_relative_to(versions) or
                                  path.resolve() != (current/path.relative_to(root)).resolve()):
            raise ValueError(f'Unexpected public report link: {path}')
    old_files = [path for path in public_files if path.is_file()]
    # Bootstrap once, including recovery from an older implementation that
    # installed current but was interrupted before replacing its regular index.
    needs_snapshot = old_files and (not current.exists() or
        (published.is_file() and not published.is_symlink() and
         not current.resolve().name.startswith('legacy-')))
    if needs_snapshot:
        legacy,legacy_mode = _new_version(root,'legacy-')
        try:
            for path in old_files:
                dest = legacy/path.relative_to(root)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if path.parent != root:
                    _limit_access(dest.parent,path.parent)
                if path == published:
                    continue
                shutil.copyfile(path,dest)
                _limit_access(dest,path)
            if published in old_files:
                index = json.loads(published.read_text())
                # Readers acquiring this index during migration can keep it even
                # after current switches to the new generation.
                for entry in index:
                    for key in ('text','json'):
                        relative = entry[key]
                        if root/relative in public_files:
                            entry[key] = str(legacy.relative_to(root)/relative)
                atomic_json(legacy/published.name,index)
                _limit_access(legacy/published.name,published)
            _limit_access(legacy,root,legacy_mode)
            _link(current,legacy.name)
        finally:
            if current.resolve() != legacy:
                shutil.rmtree(legacy)

    # Each existing entry still resolves to its old bytes while being wired
    # up. Newly introduced entries become readable at the single commit below.
    for path in public_files:
        target = current/path.relative_to(root)
        _link(path,os.path.relpath(target,path.parent))
    _link(current,generation.name)


def check_report_destinations(root, profiles, config):
    from run_inputs import check_output, filesystem_locations, storage_location
    mounts = filesystem_locations()
    device, base = storage_location(root,mounts)
    writable = {'.vllm-reports','.report-rebuild.lock','report-jobs.json','summary.json',
                'vllm-reports.json','vllm-report.txt',*profiles}
    inputs = [config.get('tokenizer'),config.get('chat_template'),config.get('dataset',{}).get('path'),
              root/'inputs',root/'config.json',root/'setup.json']
    for name in writable:
        if not isinstance(name,str) or name in ('','.','..') or Path(name).name != name:
            raise ValueError(f'Invalid report destination: {name}')
        check_output(root/name, inputs, mounts)
    for mount, mount_device, mount_base in mounts:
        if mount == root or not mount.is_relative_to(root):
            continue
        relative = mount.relative_to(root)
        if relative.parts[0] in writable and (mount_device,mount_base) != (device,base/relative):
            raise ValueError(f'Report destination is a separate mount: {mount}')


@contextmanager
def _report_cleanup(path, interrupted, config):
    try:
        yield
    except Exception as exc:
        if not interrupted:
            raise
        print(f'Could not finalize {path}: {error_detail(exc, config)}', file=sys.stderr)


def _check_stage_record(result, phase, summary, point):
    log = phase + '.requests.jsonl'
    if result.get('requests_file') != log:
        raise ValueError(f"Expected {log}, stage specifies {result.get('requests_file')!r}")
    contract = result.get('report_contract')
    version = contract.get('version') if isinstance(contract, dict) else None
    if 'report_contract_version' in summary:
        required = summary['report_contract_version']
        if type(required) is not int or required not in (1, REPORT_CONTRACT_VERSION):
            raise ValueError(f'Unsupported run report_contract_version: {required!r}')
        if version != required:
            raise ValueError('Missing or conflicting stage report_contract for versioned run')
    pulse = re.fullmatch(r'overload-c([0-9]+)-r([0-9]+)', phase)
    if pulse:
        concurrency = int(pulse[1])
        if concurrency < 1 or pulse[2] not in ('1', '2', '3') or str(concurrency) != pulse[1]:
            raise ValueError(f'Invalid overload pulse name: {phase}')
        # write_report also verifies that the log contains metrics.requests rows.
        for field, value in (('concurrency', result.get('concurrency')),
                             ('request_limit', result.get('request_limit')),
                             ('metrics.requests', result['metrics'].get('requests'))):
            if type(value) is not int or value != concurrency:
                raise ValueError(f'{phase} requires {field}={concurrency}, found {value!r}')
    # A stage can commit just before the summary checkpoint. In that case its
    # own identity and sealed log are still sufficient for offline recovery.
    if point is None:
        return
    if version == REPORT_CONTRACT_VERSION or 'report_contract' in point:
        if point.get('report_contract') != contract:
            raise ValueError('Stage report contract conflicts with summary (identity or SHA-256)')
        for field in ('duration', 'concurrency', 'request_limit'):
            if point.get(field) != result.get(field):
                raise ValueError(f'Stage {field} conflicts with summary')


def generate_reports(root, config, summary, tokenizer, manifest):
    from perf_native_report import write_report
    root = Path(root).resolve()
    published = root/'vllm-reports.json'
    check_report_destinations(root, (job['profile'] for job in manifest['jobs']),config)
    versions = _inside(root,'.vllm-reports')
    versions.mkdir(exist_ok=True)
    pointer = versions/'current'
    if pointer.is_symlink():
        target = pointer.readlink()
        if target.parent != Path('.') or not target.name.startswith(('legacy-','run-')):
            raise ValueError(f'Unexpected report generation: {target}')
        _inside(versions,target)
    _limit_access(versions,root)
    previous_versions = [directory for directory in versions.glob('run-*')
                         if directory.is_dir() and not directory.is_symlink()
                         and (directory/published.name).is_file()]
    generation,publish_mode = _new_version(root,'run-')
    current = versions/'current'
    index = []
    stages = []
    public_files = {published,root/'vllm-report.txt'}
    summary['reporting_status'] = 'running'
    sanitize_errors(summary,config)
    interrupted = False
    try:
        atomic_json(root/'summary.json',summary)
        points = _summary_points(summary)
        seen = set()
        for job in manifest['jobs']:
            job['status'] = 'pending'
            job.pop('error',None)
        with (generation/'vllm-report.txt').open('w') as aggregate:
            for job in manifest['jobs']:
                job['status'] = 'running'
                atomic_json(root/'report-jobs.json',manifest)
                try:
                    expected = Path(job['profile'])/(job['phase']+'.json')
                    if (not all(re.fullmatch(r'[A-Za-z0-9_-]+',job[k]) for k in ('profile','phase'))
                            or job['profile'] == 'inputs' or str(expected) != job['result_file']):
                        raise ValueError('Report job must identify its own profile and stage')
                    path = _inside(root,expected)
                    relative = path.relative_to(root)
                    result = json.loads(path.read_text())
                    key = (job['profile'], job['phase'])
                    if key in seen:
                        raise ValueError(f'Duplicate report job: {relative}')
                    seen.add(key)
                    _check_stage_record(result, job['phase'], summary, points.get(key))
                    requests = _inside(root, path.with_suffix('.requests.jsonl'))
                    prefix = (generation/relative).with_suffix('')
                    prefix.parent.mkdir(parents=True,exist_ok=True)
                    names = write_report(requests,prefix,result['duration'],result['concurrency'],tokenizer,config,stage=result)
                    _limit_access(prefix.parent,path.parent)
                    for name in names.values():
                        public = path.parent/name
                        # The payload is derived from the request log. Older
                        # generated files may already carry the wrong group.
                        output = prefix.parent/name
                        mode = output.stat().st_mode
                        if public.is_file():
                            mode &= public.stat().st_mode
                        _limit_access(output,requests,mode)
                        public_files.add(public)
                    index.append(dict(profile=job['profile'],phase=job['phase'],
                        **{k:str((prefix.parent/v).relative_to(root)) for k,v in names.items()}))
                    stages.append((job,path,result,names,requests))
                    label = job['phase']
                    if (label.startswith(('scan-','burst-','overload-')) and 'recovery' not in label) or label=='soak':
                        text = (prefix.parent/names['text']).read_text()
                        heading = f"\n[{job['profile']}/{label}] vLLM v0.11.0 native metrics\n"
                        aggregate.write(heading+text)
                        print(heading+text,end='')
                    job['status'] = 'pending'
                except Exception as exc:
                    detail = f"{job['result_file']}: {error_detail(exc,config)}"
                    job.update(status='failed',error=detail)
                    raise RuntimeError(detail) from None

        missing = points.keys() - seen
        if missing:
            raise ValueError('Missing report jobs: ' + ', '.join(f'{p}/{s}' for p, s in sorted(missing)))
        atomic_json(generation/published.name,index)
        # All stages passed integrity checks. Old indices address historical
        # files directly, so permission repairs also cover those paths.
        for directory in previous_versions:
            for _,path,_,names,requests in stages:
                relative = path.parent.relative_to(root)
                folder = _inside(directory,relative)
                if folder.is_dir():
                    for name in names.values():
                        historical = _inside(directory,relative/name)
                        if historical.is_file():
                            _limit_access(historical,requests)
                    _limit_access(folder,path.parent)
        for path in (published,root/'vllm-report.txt'):
            output = generation/path.name
            mode = output.stat().st_mode
            if path.is_file():
                mode &= path.stat().st_mode
            _limit_access(output,root/'summary.json',mode)
            for directory in previous_versions:
                historical = _inside(directory,path.name)
                if historical.is_file():
                    _limit_access(historical,root/'summary.json')
        for directory in previous_versions:
            _limit_access(directory,root)
        _limit_access(generation,root,publish_mode)
        _publish_generation(root,generation,sorted(public_files))
        for job,path,result,names,_ in stages:
            if result.get('vllm_report') != names:
                result['vllm_report'] = names
                atomic_json(path,result)
            point = points.get((job['profile'], job['phase']))
            if point is not None:
                point['vllm_report'] = names
            job['status'] = 'complete'
        summary['reporting_status'] = 'complete'
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        summary['reporting_status'] = 'failed' if isinstance(exc,Exception) else 'pending'
        if isinstance(exc,Exception):
            summary.setdefault('errors',[]).append(dict(type=type(exc).__name__,phase='reporting',detail=error_detail(exc,config)))
        raise
    finally:
        # Inspect the actual pointer: interruption may follow rename immediately.
        with _report_cleanup(generation, interrupted, config):
            if not current.is_symlink() or current.readlink() != Path(generation.name):
                shutil.rmtree(generation)
        for job in manifest['jobs']:
            if job['status'] == 'running':
                job['status'] = 'pending'
        with _report_cleanup(root/'report-jobs.json', interrupted, config):
            atomic_json(root/'report-jobs.json',manifest)
        with _report_cleanup(root/'summary.json', interrupted, config):
            summary['vllm_reports'] = json.loads(published.read_text()) if published.exists() else []
            atomic_json(root/'summary.json',summary)


@contextmanager
def output_lock(root):
    """Keep load execution and offline rebuilds from sharing the journal."""
    path = _inside(root,'.report-rebuild.lock')
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o666)
    with os.fdopen(fd, 'a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Output directory already in use') from None
        yield


def rebuild_reports(root):
    root = Path(root).resolve(strict=True)
    setup = root/'setup.json'
    if setup.is_file() and _read_json(_inside(root, 'setup.json')).get('schema') in ('native_stages_v1', 'native_stages_v2', 'native_stages_v3', 'native_stages_v4'):
        from perf_native_stages import publish_reports
        with output_lock(root):
            publish_reports(root)
        return
    summary = _read_json(root/'summary.json')
    if isinstance(summary, dict) and summary.get('mode') in ('native_stages', 'vllm_bench_serve'):
        raise ValueError('Native reports are saved directly in each benchmark directory; see summary.json')
    if not isinstance(summary,dict) or not isinstance(summary.get('profiles'),dict) or not (root/'config.json').is_file():
        raise ValueError('Report rebuild requires an existing staged performance run')
    config = _read_json(root/'config.json')
    if not isinstance(config,dict) or not isinstance(config.get('target'),dict) or not config.get('tokenizer'):
        raise ValueError('Missing staged performance run configuration')
    check_report_destinations(root,summary['profiles'],config)
    with output_lock(root):
        _rebuild_reports(root)


class _LegacyTokenizer:
    """Only resolve NAS resources when a legacy native row actually needs encode."""
    def __init__(self, root, config):
        self.root = root
        self.path = config['tokenizer']
        self.tokenizer = None

    def encode(self, text, **kwargs):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            from perf_resources import configured_tokenizer_files, digest, resource_key, tokenizer_hashes
            setup = self.root/'setup.json'
            expected = json.loads(setup.read_text()).get('tokenizer_hashes') if setup.is_file() else None
            if not expected:
                raise ValueError('Legacy token fallback requires the original tokenizer fingerprint')
            directory = Path(self.path).resolve(strict=True)
            for resource in configured_tokenizer_files(directory):
                key = resource_key(directory,resource)
                if key not in expected:
                    raise ValueError(f'No original fingerprint for selected tokenizer resource: {key}')
            for name,checksum in expected.items():
                file = directory/name
                if not file.is_file() or digest(file) != checksum:
                    raise ValueError('Legacy token fallback: tokenizer fingerprint mismatch')
            tokenizer = AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
            if tokenizer_hashes(directory,tokenizer) != expected:
                raise ValueError('Legacy token fallback: tokenizer fingerprint mismatch')
            self.tokenizer = tokenizer
        return self.tokenizer.encode(text,**kwargs)


def _rebuild_reports(root):
    summary = json.loads((root/'summary.json').read_text())
    if summary.get('mode') == 'vllm_bench_serve':
        raise ValueError('Native runs save vllm-report.txt directly; stage report rebuild is unavailable')
    config = {}
    errors_before = len(summary.get('errors',[]))
    try:
        config = _read_json(root/'config.json')
        sanitize_errors(summary,config)
        published = root/'vllm-reports.json'
        required = list(summary.get('vllm_reports',[]))
        if published.exists():
            required.extend(_read_json(published))
        required.extend(dict(profile=p, phase=s) for p, s in _summary_points(summary))
        manifest = save_jobs(root, discover=True, required_stages=required)
        queued = {(job['profile'], job['phase']): job for job in manifest['jobs']}
        missing = []
        for profile, phase in sorted({(entry['profile'], entry['phase']) for entry in required}):
            relative = f'{profile}/{phase}.json'
            job = queued.get((profile, phase))
            if job is None or job['result_file'] != relative:
                raise ValueError(f'Required stage conflicts with report queue: {relative}')
            if not _inside(root, relative).is_file():
                job.update(status='failed', error=f'Missing completion record: {relative}')
                missing.append(relative)
        if missing:
            atomic_json(root/'report-jobs.json', manifest)
            raise ValueError('Missing completion records: ' + ', '.join(missing))
        if not manifest['jobs']:
            raise ValueError('No completed stages available for reporting')
        generate_reports(root,config,summary,_LegacyTokenizer(root,config),manifest)
    except Exception as exc:
        summary['reporting_status'] = 'failed'
        if len(summary.get('errors',[]))==errors_before:
            summary.setdefault('errors',[]).append(dict(type=type(exc).__name__,phase='reporting',detail=error_detail(exc,config)))
        atomic_json(root/'summary.json',summary)
        raise RuntimeError(error_detail(exc,config)) from None
