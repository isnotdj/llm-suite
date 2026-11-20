"""Run JSON-configured benchmarks with the pinned upstream serving CLI."""
import argparse
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from run_journal import atomic_json


def parse_options(parser, options):
    if not isinstance(options, dict):
        raise ValueError('vllm_bench_serve must be a JSON object')
    # Use registration order: upstream validates dataset_path against dataset_name.
    registered = {flag: action for action in parser._actions
                  for flag in action.option_strings if flag.startswith('--') and flag != '--help'}
    supplied = {}
    for key, value in options.items():
        if key == 'seed' and (type(value) is not int or not 0 <= value < 2**32):
            raise ValueError('vllm_bench_serve.seed must be an integer in [0, 4294967295]')
        flag = '--' + key.replace('_', '-')
        if flag not in registered:
            raise ValueError(f'Unknown vllm_bench_serve option: {key}')
        if flag in supplied:
            raise ValueError(f'Duplicate option: {flag}')
        supplied[flag] = value

    argv = []
    for flag, action in registered.items():
        if flag not in supplied:
            continue
        value = supplied[flag]
        if action.nargs == 0:
            if not isinstance(value, bool):
                raise ValueError(f'{flag} requires a JSON boolean')
            if value:
                argv.append(flag)
        elif action.nargs in ('+', '*'):
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ValueError(f'{flag} requires an array of strings')
            # An element starting with "--" must never become another option.
            if any(parser._parse_optional(v) is not None for v in value):
                raise ValueError(f'{flag} contains a value that would be parsed as a CLI option')
            argv.extend([flag, *value])
        else:
            if value is None or isinstance(value, (bool, list)):
                raise ValueError(f'Invalid value for {flag}')
            if isinstance(value, dict):
                if action.type in (str, float, int, None):
                    raise ValueError(f'{flag} requires a scalar')
                value = json.dumps(value)
            argv.append(f'{flag}={value}')
    return parser.parse_args(argv)


def run_native(raw, root, stage, validate_only, summary):
    if set(raw) != {'vllm_bench_serve'} or stage != 'all':
        raise ValueError('Use vllm_bench_serve alone, without suite fields or --stage')
    try:
        installed = version('vllm').split('+')[0]
    except PackageNotFoundError as exc:
        raise ValueError('Native benchmarks require vllm==0.11.0. See README for the native Docker command.') from exc
    if installed != '0.11.0':
        raise ValueError(f'Native benchmarks require vllm==0.11.0, found {installed}')
    from vllm.benchmarks import serve

    parser = argparse.ArgumentParser(prog='vllm bench serve', allow_abbrev=False)
    serve.add_cli_args(parser)
    try:
        args = parse_options(parser, raw['vllm_bench_serve'])
    except SystemExit as exc:
        raise ValueError('Invalid vllm_bench_serve options (see parser diagnostic)') from exc
    export_dir = Path(args.result_dir or '.')
    if export_dir.is_absolute() or '..' in export_dir.parts:
        raise ValueError('result_dir must be a relative directory inside this run’s exports/')
    args.result_dir = str(root/'exports'/export_dir)
    dataset_path = getattr(args, 'dataset_path', None)
    source = None
    if dataset_path and Path(dataset_path).exists():
        from run_inputs import restrict_run_readers
        source = Path(dataset_path).resolve(strict=True)
        restrict_run_readers(root, [dataset_path,source])
        args.dataset_path = str(source)
    options = portable(vars(args))
    if options.get('header'):
        options['header'] = [header.partition('=')[0] for header in options['header']]
        if any('=' not in header for header in args.header):
            raise ValueError('vllm_bench_serve.header requires NAME=VALUE')
    for key in ('base_url', 'endpoint'):
        if options.get(key):
            parts = urlsplit(options[key])
            query = urlencode([(name,'[REDACTED]') for name,_ in parse_qsl(parts.query,keep_blank_values=True)])
            options[key] = urlunsplit((parts.scheme,parts.netloc.rsplit('@',1)[-1],parts.path,query,''))
    setup = dict(mode='vllm_bench_serve', vllm_version=installed, api_key_env='OPENAI_API_KEY',
                 options=options)
    atomic_json(root/'setup.json', setup)
    export = bool(args.save_result or args.append_result)
    counts = dict(selected=None, dispatched=0, successful=0)
    artifacts = dict(report='not_generated', result='not_generated',
                     export='not_generated' if export else 'not_requested')
    summary.update(requests=counts, artifacts=artifacts, traffic_status='not_started')
    if export or args.result_filename is not None:
        export_name = export_filename(args)
    if validate_only:
        return 'validated'

    summary.update(overall_status='running', traffic_status='running')
    atomic_json(root/'summary.json', summary)
    report = NativeReport(root, artifacts)
    # The pinned CLI writes optional exports after printing the table. Run its
    # measurement once, persist the return value, then perform that export.
    args.save_result = args.append_result = False
    try:
        with observe_requests(serve, counts, root, summary), redirect_stdout(report):
            result = serve.main(args)
    finally:
        report.close()
        args.save_result, args.append_result = options['save_result'], options['append_result']
    traffic_status = 'failed'
    selected, sent, successful = counts['selected'], counts['dispatched'], counts['successful']
    consistent = (selected is not None and 0 <= successful <= sent <= selected
                  and successful == result.get('completed'))
    if consistent and successful:
        traffic_status = 'complete' if successful == selected else 'incomplete'
    summary['traffic_status'] = traffic_status
    if artifacts['report'] != 'complete':
        raise ValueError('vLLM returned without a complete serving benchmark report')
    if isinstance(result.get('errors'), list):
        credentials = [os.environ.get('OPENAI_API_KEY')]
        credentials.extend(h.partition('=')[2] for h in (args.header or []))
        for secret in filter(None, credentials):
            result['errors'] = [e.replace(secret,'[REDACTED]') if isinstance(e,str) else e
                                for e in result['errors']]
    artifacts['result'] = 'failed'
    atomic_json(root/'vllm-result.json', result)
    artifacts['result'] = 'complete'
    if export:
        artifacts['export'] = 'failed'
        # Claim a fresh, private tree. Never reuse a directory or follow an
        # existing export tree, even when append_result was requested.
        (root/'exports').mkdir(mode=0o700)
        path = Path(args.result_dir)/export_name
        path.parent.mkdir(parents=True, exist_ok=True)
        export_result(serve, path, result, options)
        artifacts['export'] = 'complete'
        summary['export_path'] = str(path)
    if not consistent:
        raise ValueError('vLLM request counts could not be reconciled')
    if not successful:
        raise ValueError('No successful measured requests')
    return traffic_status


def portable(value):
    # Native multimodal defaults contain tuple keys, and unlimited RPS is inf.
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@contextmanager
def observe_requests(serve, counts, root, summary):
    # Pinned v0.11.0 sends pbar (even when None) only on measured dispatches.
    # Readiness and start/stop-profiler calls bypass the measured semaphore.
    get_samples = serve.get_samples
    backends = serve.ASYNC_REQUEST_FUNCS
    def selected(*args, **kwargs):
        samples = get_samples(*args, **kwargs)
        counts['selected'] = len(samples)
        atomic_json(root/'summary.json', summary)
        if not samples:
            raise ValueError('No benchmark requests selected')
        return samples

    def observe(fn):
        async def request(*args, **kwargs):
            if 'pbar' not in kwargs:
                return await fn(*args, **kwargs)
            counts['dispatched'] += 1
            output = await fn(*args, **kwargs)
            if output.success:
                counts['successful'] += 1
            return output
        return request

    try:
        serve.get_samples = selected
        serve.ASYNC_REQUEST_FUNCS = {name: observe(fn) for name, fn in backends.items()}
        yield
    finally:
        serve.get_samples, serve.ASYNC_REQUEST_FUNCS = get_samples, backends


class NativeReport:
    def __init__(self, root, artifacts):
        self.destination = root/'vllm-report.txt'
        self.artifacts = artifacts
        self.pending = ''
        self.file = None

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if self.file is None:
                if self.artifacts['report'] == 'complete' or line.strip('= ') != 'Serving Benchmark Result':
                    continue
                self.artifacts['report'] = 'partial'
                self.file = NamedTemporaryFile(mode='w', encoding='utf-8',
                                              dir=self.destination.parent, delete=False)
            try:
                self.file.write(line+'\n')
                if line == '='*50:
                    self.file.close()
                    Path(self.file.name).replace(self.destination)
                    self.artifacts['report'] = 'complete'
                    self.file = None
            except OSError:
                self.artifacts['report'] = 'failed'
                raise
        return len(text)

    def flush(self):
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            Path(self.file.name).unlink(missing_ok=True)


def export_filename(args):
    name = args.result_filename
    if name is None:
        label = args.label or args.backend
        concurrency = '' if args.max_concurrency is None else f'-concurrency{args.max_concurrency}'
        traffic = (f'ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps'
                   if args.ramp_up_strategy else f'{args.request_rate}qps')
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        name = f'{label}-{traffic}{concurrency}-{args.model.split("/")[-1]}-{stamp}.json'
    if name in ('', '.', '..') or Path(name).name != name:
        raise ValueError('result_filename must be a single filename; use result_dir for subdirectories')
    return name


def export_result(serve, path, result, options):
    # A new run has no previous records. Both upstream save and append begin
    # with the same first JSON record. Publish only after serialization succeeds.
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp)/path.name
        with staged.open('x', encoding='utf-8') as stream:
            json.dump(result, stream)
        os.link(staged, path)
        if os.environ.get('SAVE_TO_PYTORCH_BENCHMARK_FORMAT'):
            serve.save_to_pytorch_benchmark_format(argparse.Namespace(**options), result, str(staged))
            sidecar = staged.with_suffix('.pytorch.json')
            if sidecar.is_file():
                os.link(sidecar, path.with_suffix('.pytorch.json'))
