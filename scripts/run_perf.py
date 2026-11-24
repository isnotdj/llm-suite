#!/opt/perf-venv/bin/python
"""Remote-only performance suite backed by pinned vLLM benchmark transport."""
import argparse
import asyncio
import json
import os
from uuid import UUID
from pathlib import Path
import signal
import sys
from contextlib import ExitStack, contextmanager
from run_journal import atomic_json
from perf_report_jobs import rebuild_reports, error_detail, output_lock


@contextmanager
def report_signals(config=None):
    """Synchronous report code must receive interrupts even inside asyncio.run."""
    received = None

    def interrupt(signum, frame):
        nonlocal received
        received = signal.Signals(signum).name
        raise KeyboardInterrupt(received)

    with ExitStack() as handlers:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, interrupt)
            handlers.callback(signal.signal, signum, previous)
        try:
            yield
        except Exception as exc:
            if received is None:
                raise
            # An I/O error during unwinding must not replace the signal.
            print(f'Report cleanup after {received} failed: {error_detail(exc, config)}', file=sys.stderr)
            raise KeyboardInterrupt(received) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--report-only',action='store_true',help='Rebuild reports from an existing output directory without API calls')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--existing-group', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--group-id', type=UUID, help=argparse.SUPPRESS)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--stage',choices=['all','connectivity','scan','overload','soak'],default='all')
    args = parser.parse_args(argv)
    def interrupted(signum,frame): raise KeyboardInterrupt('SIGTERM')
    if args.report_only:
        if args.config or args.validate_only or args.group_id or args.stage!='all':
            parser.error('--report-only uses the saved config and cannot combine with --config, --validate-only or --stage')
        try:
            with report_signals():
                rebuild_reports(args.output)
            return 0
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f'Report generation failed: {exc}',file=sys.stderr)
            return 1
    if args.config is None: parser.error('--config is required unless --report-only is used')
    root = args.output.resolve()
    from run_inputs import create_run
    try:
        raw = json.loads(args.config.read_text())
        if not isinstance(raw,dict):
            raise ValueError('Performance config must be a JSON object')
        if 'bench' in raw:
            from perf_bench_serve import prepare_argv_config
            raw = prepare_argv_config(raw)
        inputs = [args.config]
        if 'vllm_bench_serve' in raw:
            from perf_native_stages import native_inputs
            inputs.extend(native_inputs(raw))
        else:
            inputs.extend([raw.get('tokenizer'),raw.get('chat_template')])
            if isinstance(raw.get('dataset'),dict):
                inputs.append(raw['dataset'].get('path'))
        root = create_run(root, inputs, group_id=args.group_id, phase='perf', existing_group=args.existing_group)
    except (OSError, ValueError) as exc:
        print(f'Cannot start performance run: {exc}',file=sys.stderr)
        return 1
    with output_lock(root):
        return _run(args,root,interrupted,raw)


def _run(args,root,interrupted,raw):
    summary = dict(overall_status='preparing', errors=[])
    config = None
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        code = 1
        try:
            if 'vllm_bench_serve' not in raw:
                raise ValueError('Live benchmarks now require official vLLM 0.11.0. Migrate to perf.example.json and the native-bench image; use --report-only for historical runs.')
            summary = dict(mode='vllm_bench_serve', overall_status='preparing', errors=[])
            if 'profiles' in raw:
                from perf_native_stages import run_stages
                summary['mode'] = 'native_stages'
                summary['overall_status'] = run_stages(raw, root, args.stage, args.validate_only, summary)
            else:
                from perf_bench_serve import run_native
                summary['overall_status'] = run_native(raw,root,args.stage,args.validate_only,summary)
            code = 0 if summary['overall_status'] in ('complete','validated') else 1
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            if summary.get('traffic_status') == 'running': summary['traffic_status'] = 'interrupted'
            summary['overall_status'] = 'interrupted';code=130
            if summary.get('mode') == 'native_stages':
                save_run_error(root, summary, dict(type=type(exc).__name__, detail='Run interrupted'))
        except Exception as exc:
            detail = error_detail(exc,config)
            if summary.get('mode') in ('native_stages', 'vllm_bench_serve'):
                values = [os.environ.get('OPENAI_API_KEY')]
                native_blocks = [raw.get('vllm_bench_serve')]
                if isinstance(raw.get('profiles'), list):
                    native_blocks.extend(p.get('vllm_bench_serve') for p in raw['profiles'] if isinstance(p, dict))
                for block in native_blocks:
                    if isinstance(block,dict) and isinstance(block.get('header'),list):
                        values.extend(h.partition('=')[2] for h in block['header'] if isinstance(h,str))
                for value in values:
                    if value:
                        detail = detail.replace(value,'[REDACTED]')
            if summary.get('traffic_status') == 'running': summary['traffic_status'] = 'failed'
            summary['overall_status']='failed'
            summary['errors'].append({'type':type(exc).__name__,'detail':detail})
            if summary.get('mode') == 'native_stages':
                save_run_error(root, summary, summary['errors'][-1])
            print(f'Performance run failed: {detail}',file=sys.stderr)
        try:
            return _finish_summary(root, summary, code)
        except KeyboardInterrupt:
            summary['overall_status'] = 'interrupted'
            if summary.get('mode') == 'native_stages':
                summary.update(operation='publish_reports', active_profile=None, phase=None)
                save_run_error(root, summary, dict(type='KeyboardInterrupt', detail='Report publication interrupted'))
            return _finish_summary(root, summary, 130)
    finally:
        signal.signal(signal.SIGTERM,previous)


def save_run_error(root, summary, error):
    record = dict(error, run_id=summary.get('run_id'), operation=summary.get('operation', 'prepare_run'),
                  profile=summary.get('active_profile'), phase=summary.get('phase'),
                  status=summary['overall_status'])
    try:
        atomic_json(root/'run-error.json', record)
    except OSError as exc:
        print(f'Cannot record run error ({record["operation"]}): {exc}', file=sys.stderr)


def _finish_summary(root, summary, code):
    path = root/'summary.json'
    if summary.get('mode') == 'native_stages' and (root/'setup.json').is_file():
        from perf_native_stages import publish_reports
        try:
            derived = publish_reports(root)
            return code or int(derived['overall_status'] != 'complete')
        except (ValueError, OSError) as exc:
            print(f'Cannot publish stage reports: {exc}', file=sys.stderr)
            if not (root/'run-error.json').exists():
                save_run_error(root, dict(run_id=summary.get('run_id'), operation='publish_reports', overall_status='failed'),
                               dict(type=type(exc).__name__, detail=str(exc)))
            return code or 1
    try:
        atomic_json(path, summary)
    except OSError as exc:
        print(f'Cannot write {path}: {exc}', file=sys.stderr)
        return 130 if code == 130 else 1
    return code


if __name__=='__main__': sys.exit(main())
