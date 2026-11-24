"""Run JSON-configured benchmarks with the pinned upstream serving CLI."""
import argparse
import hashlib
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from perf_resources import artifact_signature
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


def connection_argv(api_url, argv):
    """Apply model defaults to one complete workload, keeping explicit CLI options last."""
    flags = {token.partition('=')[0] for token in argv}
    api_url = api_url.rstrip('/')
    base = api_url.removesuffix('/v1')
    if flags & {'--host', '--port'}:
        if '--base-url' in flags:
            raise ValueError('Choose either --base-url or --host/--port for a benchmark workload')
        return ['--endpoint', urlsplit(api_url).path + '/chat/completions', *argv]
    defaults = ['--endpoint', api_url[len(base):] + '/chat/completions']
    if '--base-url' not in flags:
        defaults.extend(['--base-url', base])
    return defaults + argv


def prepare_argv_config(raw):
    """Parse each complete argv with the installed official parser, before creating outputs."""
    from vllm.benchmarks import serve
    from perf_native_stages import options

    parser = argparse.ArgumentParser(prog='vllm bench serve', allow_abbrev=False)
    serve.add_cli_args(parser)
    base = raw.pop('bench')
    model_api_url = raw.pop('model_api_url', None)

    def parse(argv, staged):
        if model_api_url is not None:
            argv = connection_argv(model_api_url, argv)
        if staged:
            options({token.partition('=')[0][2:]: True for token in argv if token.startswith('--')})
        try:
            namespace = parser.parse_args(argv)
        except SystemExit as exc:
            raise ValueError('Invalid benchmark arguments') from exc
        # Only expose the fields used by stage planning, input binding and redaction.
        fields = ('model', 'tokenizer', 'dataset_path', 'dataset_name', 'header', 'lora_modules',
                  'prefix_repetition_num_prefixes', 'percentile_metrics', 'metric_percentiles',
                  'backend', 'endpoint', 'endpoint_type', 'random_batch_size', 'trust_remote_code', 'hf_name')
        return {**{k: getattr(namespace, k) for k in fields if getattr(namespace, k, None) is not None},
                '_namespace': namespace}

    if 'profiles' not in raw:
        return {'vllm_bench_serve': dict(parse(base, False), _single=True)}
    profiles = [dict(name=p['name'], vllm_bench_serve=parse(base + p['bench'], True))
                for p in raw.pop('profiles')]
    return dict(raw, profiles=profiles, vllm_bench_serve={})


def result_metrics(result):
    return {key: result[key] for key in ('output_throughput', 'p95_ttft_ms', 'p95_tpot_ms') if key in result}


def count_checksum(summary):
    # Compact official output omits output_lens. Bind observed counts to this result and round.
    payload = [summary['run_id'], summary['sequence'], summary['artifact_signatures']['result'], summary['requests']]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_native(raw, root, stage, validate_only, summary, *, measured_count=None):
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
        from copy import copy
        opts = raw['vllm_bench_serve']
        if '_namespace' in opts:
            args = copy(opts['_namespace'])
            for key, value in opts.items():
                if not key.startswith('_'):
                    setattr(args, key, float(value) if key == 'request_rate' else value)
        else:
            args = parse_options(parser, opts)
    except SystemExit as exc:
        raise ValueError('Invalid vllm_bench_serve options (see parser diagnostic)') from exc
    if not args.burstiness > 0:
        raise ValueError('vllm_bench_serve.burstiness must be positive')
    if getattr(args, 'dataset_name', None) == 'random-mm' and args.backend != 'openai-chat':
        raise ValueError('random-mm requires the openai-chat backend')
    for item in getattr(args, 'metadata', None) or []:
        if '=' not in item or not item.partition('=')[0].strip():
            raise ValueError('vllm_bench_serve.metadata requires KEY=VALUE')
    if getattr(args, 'goodput', None):
        serve.check_goodput_args(args)
    export_dir = Path(args.result_dir or '.')
    export = bool(args.save_result or args.append_result)
    export_name = export_filename(args) if export or args.result_filename is not None else None
    single = opts.get('_single', False)
    if single:
        from run_inputs import check_output, require_local_output
        directory = (root/export_dir).resolve()
        if export:
            inputs = json.loads(os.environ.get('VLLM_SUITE_INPUTS', '[]'))
            inputs.extend([args.dataset_path, args.tokenizer, args.model, *(args.lora_modules or [])])
            check_output(directory, inputs)
            require_local_output(directory)
            if not directory.is_relative_to(root):
                raise ValueError('result_dir must be inside this run’s perf directory')
            reserved = {root/name for name in ('setup.json', 'summary.json', 'vllm-result.json',
                                                'vllm-report.txt', '.report-rebuild.lock')}
            if any(directory.is_relative_to(path) for path in reserved):
                raise ValueError('Native export directory collides with a suite artifact')
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            for path in [directory/export_name, (directory/export_name).with_suffix('.pytorch.json')]:
                if path in reserved or path.is_symlink() or path.exists():
                    raise ValueError(f'Native export destination is reserved or already exists: {path}')
        args.result_dir = str(directory)
    else:
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
    counts = dict(selected=None, dispatched=0, successful=0)
    artifacts = dict(report='not_generated', result='not_generated',
                     export='not_generated' if export else 'not_requested')
    summary.update(requests=counts, artifacts=artifacts, traffic_status='not_started')
    if validate_only:
        return 'validated'

    summary.update(overall_status='running', traffic_status='running')
    checkpoint = summary
    if measured_count is not None:
        summary['request_count_status'] = 'unknown'
        checkpoint = dict(summary, requests=dict(selected=None, dispatched=None, successful=None))
    atomic_json(root/'summary.json', checkpoint)
    report = NativeReport(root, artifacts)
    # The pinned CLI writes optional exports after printing the table. Run its
    # measurement once, persist the return value, then perform that export.
    args.save_result = args.append_result = False
    try:
        with observe_requests(serve, counts, root, summary, measured_count), redirect_stdout(report):
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
    if not consistent:
        raise ValueError('vLLM request counts could not be reconciled')
    if measured_count is not None:
        summary['metrics'] = result_metrics(result)
        summary['artifact_signatures'] = {key: artifact_signature(root/filename) for key, filename in
                                          [('report', 'vllm-report.txt'), ('result', 'vllm-result.json')]}
        summary['count_sha256'] = count_checksum(summary)
        summary['request_count_status'] = 'final'
    if measured_count is not None or single:
        summary['measurement_status'] = traffic_status
        summary['overall_status'] = 'export_pending' if export else traffic_status
        artifacts['export'] = 'pending' if export else 'not_requested'
        atomic_json(root/'summary.json', summary)
    if export:
        artifacts['export'] = 'failed'
        if single:
            summary['export_dir'] = args.result_dir
        else:
            (root/'exports').mkdir(mode=0o700)
        path = Path(args.result_dir)/export_name
        path.parent.mkdir(parents=True, exist_ok=True)
        files = export_result(serve, path, result, options)
        if measured_count is not None:
            summary['export_files'] = {str(f.relative_to(root)): artifact_signature(f) for f in files}
        artifacts['export'] = 'complete'
        summary['export_path'] = str(path)
    if measured_count is not None:
        if export:
            summary['overall_status'] = traffic_status
            atomic_json(root/'summary.json', summary)
        return traffic_status
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
def observe_requests(serve, counts, root, summary, measured_count=None):
    # Pinned v0.11.0 sends pbar (even when None) only on measured dispatches.
    # Readiness and start/stop-profiler calls bypass the measured semaphore.
    get_samples = serve.get_samples
    backends = serve.ASYNC_REQUEST_FUNCS
    # Upstream removes output_lens from compact JSON after computing its report.
    # Count the lengths it actually used for TPOT before that optional removal.
    if measured_count is not None:
        benchmark = serve.benchmark

        async def measured_benchmark(*args, **kwargs):
            result = await benchmark(*args, **kwargs)
            counts['tpot_samples'] = sum(length > 1 for length in result['output_lens'])
            return result

    def selected(*args, **kwargs):
        samples = get_samples(*args, **kwargs)
        counts['selected'] = len(samples)
        snapshot = summary
        if measured_count is not None:
            snapshot = dict(summary, requests=dict(selected=len(samples), dispatched=None, successful=None))
        atomic_json(root/'summary.json', snapshot)
        if measured_count is not None and len(samples) != measured_count:
            raise ValueError(f'vLLM selected {len(samples)} requests; this stage requires {measured_count}')
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
        if measured_count is not None:
            serve.benchmark = measured_benchmark
        serve.get_samples = selected
        serve.ASYNC_REQUEST_FUNCS = {name: observe(fn) for name, fn in backends.items()}
        yield
    finally:
        serve.get_samples, serve.ASYNC_REQUEST_FUNCS = get_samples, backends
        if measured_count is not None:
            serve.benchmark = benchmark


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
        published = [path]
        if os.environ.get('SAVE_TO_PYTORCH_BENCHMARK_FORMAT'):
            serve.save_to_pytorch_benchmark_format(argparse.Namespace(**options), result, str(staged))
            sidecar = staged.with_suffix('.pytorch.json')
            if sidecar.is_file():
                destination = path.with_suffix('.pytorch.json')
                os.link(sidecar, destination)
                published.append(destination)
        return published
