#!/opt/perf-venv/bin/python
"""Remote-only performance suite backed by pinned vLLM benchmark transport."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
from importlib.metadata import distributions
from datetime import datetime, timezone
from perf_config import normalize
from perf_data import prepare, token_count
from perf_metrics import select, overload_verdict, stage_verdict, recovery_passes, HEALTH_POLICY
from perf_resources import digest, tokenizer_hashes
from run_journal import atomic_json
from perf_report_jobs import save_jobs, generate_reports, rebuild_reports, error_detail, output_lock


async def execute_stages(config, tokenizer, prepared, root, summary, stage, reports):
    import aiohttp
    from perf_transport import Transport
    from perf_engine import load
    timing = config['timing']
    def checkpoint():
        atomic_json(root/'summary.json',summary)
    timeout = aiohttp.ClientTimeout(total=timing['timeout'])
    async with aiohttp.ClientSession(timeout=timeout,connector=aiohttp.TCPConnector(limit=0)) as session:
        transport = Transport(config,tokenizer,session)
        # Connectivity is a prerequisite even for individual stage runs.
        summary['phase'] = 'connectivity';checkpoint()
        messages = [{'role':'user','content':'Reply with the word OK.'}]
        probe = {'id':'probe','messages':messages,'input_tokens':token_count(tokenizer,messages,config['target'].get('extra_body',{}).get('chat_template_kwargs',{}))}
        probes = [await transport(probe) for _ in range(3)]
        for data in prepared.values(): probes.append(await transport(data['samples'][0]))
        summary['connectivity'] = probes;checkpoint()
        if any(not r['success'] or r.get('input_token_delta') not in (None,0) for r in probes): raise ValueError('Connectivity or profile probe failed')
        if stage == 'connectivity': return
        for profile in config['profiles']:
            name, slo = profile['name'],profile['slo']
            if name not in prepared: continue
            samples = prepared[name]['samples']
            folder = root/name
            state = summary['profiles'].setdefault(name,{})
            summary.update(active_profile=name,phase='scan');checkpoint()
            async def run(label, concurrency, duration, maximum=None, minimum=0, guard=False):
                summary['phase'] = label;checkpoint()
                frozen = root/prepared[name]['path']
                if digest(frozen) != prepared[name]['sha256']:
                    raise ValueError('Frozen input was modified')
                result = await load(transport,samples,concurrency,duration,maximum or duration,
                                    minimum,config,folder,label,slo,guard)
                result['health'] = stage_verdict(result)
                result['acceptance_with_slo'] = stage_verdict(result,slo)
                atomic_json(folder/(label+'.json'),result)
                if result.get('requests_file'):
                    reports.append((name,label,folder,result,concurrency,tokenizer))
                    save_jobs(root,reports[-1:])
                return result
            # Individual overload/soak require this run's scan, avoiding stale
            # selections from different model/config/cache states.
            state['scan'] = []
            for concurrency in config['concurrency']:
                await run(f'warmup-c{concurrency}',concurrency,timing['warmup'])
                point = await run(f'scan-c{concurrency}',concurrency,timing['scan'],timing['scan_max'],timing['scan_requests'])
                state['scan'].append(point);checkpoint()
            state.update(select(state['scan'],slo));checkpoint()
            state['selection'] = [dict(concurrency=p['concurrency'],**stage_verdict(p,slo)) for p in state['scan']]
            baseline = next((p for p in state['scan'] if p['concurrency']==1 and stage_verdict(p,perfect=True)['passed']),None)
            async def recover(label):
                if baseline is None: return False
                consecutive = 0
                deadline = asyncio.get_running_loop().time() + timing['recovery']
                index = 0
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining < timing['recovery_window']: return False
                    try:
                        # The total recovery budget covers dispatch AND drain.
                        result = await asyncio.wait_for(run(f'{label}-recovery-{index}',1,
                            min(timing['recovery_window'],remaining)), timeout=remaining)
                    except asyncio.TimeoutError:
                        return False
                    ok = recovery_passes(result,baseline)
                    consecutive = consecutive+1 if ok else 0
                    if consecutive==3: return True
                    index += 1
            if stage in ('all','overload'):
                state['overload'] = []
                sat = state['saturation_concurrency']
                if sat is None:
                    state['overload_status'] = 'skipped_no_saturation'
                elif baseline is None:
                    state['overload_status'] = 'skipped_no_concurrency_one_baseline'
                else:
                    levels = sorted({math.ceil(sat*f) for f in (1.5,2,3) if math.ceil(sat*f)<=config['overload_cap']})
                    state['overload_status'] = 'complete' if levels else 'skipped_cap'
                    for concurrency in levels:
                        result = await run(f'overload-c{concurrency}',concurrency,timing['overload'],guard=True)
                        state['overload'].append(result);checkpoint()
                        result['recovered'] = await recover(f'overload-c{concurrency}')
                        result['acceptance'] = overload_verdict(result)
                        atomic_json(folder/f'overload-c{concurrency}.json',result)
                        if not result['acceptance']['passed']:
                            state['overload_status'] = 'failed_acceptance'
                        checkpoint()
                        if not result['recovered']: raise ValueError('Endpoint did not recover after overload')
                        if result['aborted']:
                            state['overload_status'] = 'failed_guard';break
            if stage in ('all','soak'):
                sweet = state['sweet_concurrency']
                if sweet is None:
                    state['soak_status'] = 'skipped_no_sweet_spot'
                else:
                    soak = await run('soak',sweet,timing['soak'],guard=True)
                    state['soak'] = soak
                    windows = soak['windows']
                    full = [w for w in windows if w['end']-w['start']>=timing['window']*.9]
                    soak['window_slo_pass_rate'] = sum(w['slo_pass'] for w in full)/len(full) if full else None
                    first = [w for w in windows if w['end']<=1800]
                    last = [w for w in windows if w['start']>=soak['dispatch_seconds']-1800]
                    def average(group,key):
                        vals = [w['metrics'][key] if key=='output_tokens_per_second' else w['metrics'][key]['p95'] for w in group]
                        vals = [v for v in vals if v is not None]
                        return sum(vals)/len(vals) if vals else None
                    soak['drift_last_vs_first_30min'] = {}
                    if soak['dispatch_seconds']>=3600:
                        for key in ('output_tokens_per_second','ttft','tpot'):
                            a,b = average(first,key),average(last,key)
                            soak['drift_last_vs_first_30min'][key] = b/a-1 if a and b is not None else None
                    state['soak_status'] = 'complete' if stage_verdict(soak,slo)['passed'] else 'failed'
                    atomic_json(folder/'soak.json',soak)
            checkpoint()



async def execute(config, tokenizer, prepared, root, summary, stage):
    # Durable jobs allow reporting after interruption without replaying traffic.
    reports = []
    error = None
    try:
        await execute_stages(config,tokenizer,prepared,root,summary,stage,reports)
    except BaseException as exc:
        save_jobs(root,reports,discover=True)
        summary['reporting_status'] = 'pending'
        atomic_json(root/'summary.json',summary)
        if not isinstance(exc,Exception): raise
        error = exc
    manifest = save_jobs(root,reports,discover=True)
    try:
        tokenizer = reports[0][-1] if reports else None
        generate_reports(root,config,summary,tokenizer,manifest)
    except Exception:
        if error is None: raise
    if error is not None:
        raise error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--report-only',action='store_true',help='Rebuild reports from an existing output directory without API calls')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--stage',choices=['all','connectivity','scan','overload','soak'],default='all')
    args = parser.parse_args(argv)
    def interrupted(signum,frame): raise KeyboardInterrupt('SIGTERM')
    if args.report_only:
        if args.config or args.validate_only or args.stage!='all':
            parser.error('--report-only uses the saved config and cannot combine with --config, --validate-only or --stage')
        previous = signal.signal(signal.SIGTERM,interrupted)
        try:
            rebuild_reports(args.output)
            return 0
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f'Report generation failed: {exc}',file=sys.stderr)
            return 1
        finally:
            signal.signal(signal.SIGTERM,previous)
    if args.config is None: parser.error('--config is required unless --report-only is used')
    root = args.output.resolve()
    from run_inputs import create_run
    try:
        raw = json.loads(args.config.read_text())
        if not isinstance(raw,dict):
            raise ValueError('Performance config must be a JSON object')
        inputs = [args.config]
        if 'vllm_bench_serve' in raw:
            native = raw['vllm_bench_serve']
            if isinstance(native,dict):
                inputs.extend(native.get(k) for k in ('model','tokenizer','dataset_path','dataset-path'))
                modules = native.get('lora_modules',native.get('lora-modules',[]))
                if isinstance(modules,list):
                    inputs.extend(modules)
        else:
            inputs.extend([raw.get('tokenizer'),raw.get('chat_template')])
            if isinstance(raw.get('dataset'),dict):
                inputs.append(raw['dataset'].get('path'))
        root = create_run(root, inputs)
    except (OSError, ValueError) as exc:
        print(f'Cannot start performance run: {exc}',file=sys.stderr)
        return 1
    with output_lock(root):
        return _run(args,root,interrupted,raw)


def _run(args,root,interrupted,raw):
    summary = dict(schema_version=1,report_contract_version=1,overall_status='preparing',profiles={},errors=[],
                   observation_scope='API only; internal queue and process restart unknown')
    config = None
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,interrupted)
    code = 1
    try:
        if 'vllm_bench_serve' in raw:
            from perf_bench_serve import run_native
            summary = dict(mode='vllm_bench_serve', overall_status='preparing', errors=[])
            summary['overall_status'] = run_native(raw,root,args.stage,args.validate_only,summary)
            return 0 if summary['overall_status'] in ('complete','validated') else 1
        config = normalize(raw)
        if 'api_key' in config['target']: raise ValueError('Use api_key_env, not api_key')
        dataset_hash = None
        dataset_snapshot = None
        if config['dataset']['name'] != 'random':
            from run_inputs import restrict_run_readers
            source = Path(config['dataset']['path'])
            pinned = source.resolve(strict=True)
            restrict_run_readers(root,[source,pinned])
            from perf_resources import freeze_dataset
            dataset_snapshot, dataset_hash = freeze_dataset(pinned,root)
            config['dataset']['path'] = str(pinned)
        tokenizer_dir = Path(config['tokenizer']).expanduser().resolve()
        if not tokenizer_dir.is_dir():
            raise ValueError(f'Local tokenizer directory not found: {tokenizer_dir}')
        config['tokenizer'] = str(tokenizer_dir)
        atomic_json(root/'config.json',config)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'],local_files_only=True,trust_remote_code=False)
        if config.get('chat_template'): tokenizer.chat_template = Path(config['chat_template']).read_text()
        resource_hashes = tokenizer_hashes(tokenizer_dir,tokenizer)
        preparation_config = config
        if dataset_snapshot:
            preparation_config = {**config, 'dataset': {**config['dataset'], 'path': str(dataset_snapshot)}}
        prepared, unavailable = prepare(preparation_config,tokenizer,root)
        summary['unavailable'] = unavailable
        scripts = Path(__file__).parent
        setup = dict(config=config,stage=args.stage,created_at=datetime.now(timezone.utc).isoformat(),
            inputs={k:{a:b for a,b in v.items() if a!='samples'} for k,v in prepared.items()},
            packages={d.metadata['Name']:d.version for d in distributions() if d.metadata['Name']},
            source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*scripts.glob('perf*.py'), scripts/'run_perf.py',scripts/'run_inputs.py',scripts/'run_journal.py']},
            vllm_version='v0.11.0',vllm_sha256=hashlib.sha256((scripts/'perf_vendor/endpoint_request_func.py').read_bytes()).hexdigest(),
            vllm_report_sha256=digest(scripts/'perf_vendor/serve.py'),
            health_policy=dict(HEALTH_POLICY),
            tokenizer_hashes=resource_hashes,
            dataset_sha256=dataset_hash,
            dataset_snapshot=str(dataset_snapshot.relative_to(root)) if dataset_snapshot else None)
        if config.get('chat_template'): setup['chat_template'] = tokenizer.chat_template
        atomic_json(root/'setup.json',setup)
        if not prepared: raise ValueError('No profiles have enough matching samples')
        if args.validate_only:
            summary['overall_status'] = 'validated' if not unavailable else 'incomplete'
        else:
            if not os.environ.get(config['target']['api_key_env']): raise ValueError('Missing API key environment variable')
            summary['overall_status'] = 'running'
            asyncio.run(execute(config,tokenizer,prepared,root,summary,args.stage))
            failed = bool(unavailable)
            for state in summary['profiles'].values():
                failed |= any(state.get(k,'complete')!='complete' for k in ('overload_status','soak_status'))
                failed |= any(not stage_verdict(p)['passed'] for p in state.get('scan',[]))
            summary['overall_status'] = 'incomplete' if failed else 'complete'
        code = 0 if summary['overall_status'] in ('complete','validated') else 1
    except KeyboardInterrupt:
        if summary.get('traffic_status') == 'running': summary['traffic_status'] = 'interrupted'
        summary['overall_status'] = 'interrupted';code=130
    except Exception as exc:
        detail = error_detail(exc,config)
        if summary.get('mode') == 'vllm_bench_serve':
            values = [os.environ.get('OPENAI_API_KEY')]
            native = raw.get('vllm_bench_serve')
            if isinstance(native,dict) and isinstance(native.get('header'),list):
                values.extend(h.partition('=')[2] for h in native['header'] if isinstance(h,str))
            for value in values:
                if value:
                    detail = detail.replace(value,'[REDACTED]')
        if summary.get('traffic_status') == 'running': summary['traffic_status'] = 'failed'
        summary['overall_status']='failed'
        summary['errors'].append({'type':type(exc).__name__,'detail':detail})
        print(f'Performance run failed: {detail}',file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM,previous)
        atomic_json(root/'summary.json',summary)
    return code


if __name__=='__main__': sys.exit(main())
