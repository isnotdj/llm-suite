"""Exercise archived staged run formats for report compatibility tests."""
import asyncio
from perf_data import token_count
from perf_metrics import select, overload_verdict, stage_verdict, soak_stability
from perf_resources import digest
from run_journal import atomic_json
from perf_report_jobs import save_jobs, generate_reports

async def execute_stages(config, tokenizer, prepared, root, summary, stage, reports, *, check_stop=None):
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
        probes = []
        for sample in [probe] * 3 + [data['samples'][0] for data in prepared.values()]:
            if check_stop: await check_stop()
            probes.append(await transport(sample))
        summary['connectivity'] = probes;checkpoint()
        if any(not r['success'] or r.get('input_token_delta') not in (None,0) for r in probes): raise ValueError('Connectivity or profile probe failed')
        if stage == 'connectivity': return
        for profile in config['profiles']:
            name, slo = profile['name'],profile['slo']
            if name not in prepared: continue
            samples = prepared[name]['samples']
            folder = root/name
            state = summary['profiles'].setdefault(name,{})
            if check_stop: await check_stop()
            summary.update(active_profile=name,phase='overload' if stage == 'overload' else 'scan');checkpoint()
            async def run(label, concurrency, duration, maximum=None, minimum=0, guard=False, *, request_limit=None):
                if check_stop: await check_stop()
                summary['phase'] = label;checkpoint()
                frozen = root/prepared[name]['path']
                if digest(frozen) != prepared[name]['sha256']:
                    raise ValueError('Frozen input was modified')
                result = await load(transport,samples,concurrency,duration,maximum or duration,
                                    minimum,config,folder,label,slo,guard,request_limit=request_limit,check_stop=check_stop)
                result['health'] = stage_verdict(result)
                result['acceptance_with_slo'] = stage_verdict(result,slo)
                if label.startswith('overload-'):
                    result['acceptance'] = overload_verdict(result)
                if label == 'soak':
                    result.update(soak_stability(result['windows'], result['dispatch_seconds'], timing['window']))
                atomic_json(folder/(label+'.json'),result)
                if result.get('requests_file'):
                    reports.append((name,label,folder,result,concurrency,tokenizer))
                    save_jobs(root,reports[-1:])
                return result
            if stage in ('all', 'scan', 'soak'):
                state['scan'] = []
                for concurrency in config['concurrency']:
                    count = concurrency * config['requests_per_concurrency']
                    point = await run(f'scan-c{concurrency}', concurrency, 0, minimum=count, request_limit=count)
                    state['scan'].append(point);checkpoint()
                state.update(select(state['scan'],slo));checkpoint()
                state['selection'] = [dict(concurrency=p['concurrency'],**stage_verdict(p,slo)) for p in state['scan']]
            if stage in ('all', 'overload'):
                if check_stop: await check_stop()
                summary['phase'] = 'overload'
                state['overload'] = []
                state['overload_status'] = 'running'
                for concurrency in config['overload_concurrency']:
                    for round_number in range(1, 4):
                        label = f'overload-c{concurrency}-r{round_number}'
                        point = await run(label, concurrency, 0, minimum=concurrency, request_limit=concurrency)
                        state['overload'].append(point)
                        checkpoint()
                passed = all(point['acceptance']['passed'] for point in state['overload'])
                state['overload_status'] = 'complete' if passed else 'failed_acceptance'
                checkpoint()
            if stage in ('all','soak'):
                if check_stop: await check_stop()
                summary['phase'] = 'soak'
                sweet = state['sweet_concurrency']
                if sweet is None:
                    state['soak_status'] = 'skipped_no_sweet_spot'
                else:
                    soak = await run('soak',sweet,timing['soak'],guard=True)
                    state['soak'] = soak
                    state['soak_status'] = 'complete' if soak['acceptance_with_slo']['passed'] else 'failed'
            checkpoint()



async def execute(config, tokenizer, prepared, root, summary, stage):
    # Durable jobs allow reporting after interruption without replaying traffic.
    reports = []
    error = None
    try:
        await execute_stages(config,tokenizer,prepared,root,summary,stage,reports)
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        interrupted = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
        status = 'interrupted' if interrupted else 'failed'
        phase = summary.get('phase')
        state = summary['profiles'].get(summary.get('active_profile'))
        if state is not None and phase:
            state[phase.split('-')[0] + '_status'] = status
            state[status + '_phase'] = phase
        summary['overall_status'] = status
        summary['reporting_status'] = 'pending'
        error = exc
    try:
        if error is not None:
            atomic_json(root/'summary.json', summary)
        manifest = save_jobs(root,reports,discover=True)
    except Exception:
        if error is not None:
            raise error
        raise
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        raise error
    try:
        tokenizer = reports[0][-1] if reports else None
        generate_reports(root,config,summary,tokenizer,manifest)
    except Exception:
        if error is None: raise
    if error is not None:
        raise error


