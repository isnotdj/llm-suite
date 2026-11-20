"""Closed-loop sustained load; finite worker count, no queued task explosion."""
import asyncio
from collections import deque
import hashlib
import json
import resource
import sys
import time
from perf_metrics import metrics, qualifies, MetricStore, HEALTH_POLICY


def guard_failure(row):
    return not row['success'] and (row.get('error_kind') in ('timeout','transport')
        or row.get('http_status') is None or row.get('http_status',0)>=500)


class FailureGuard:
    """Independent progress deadline and rolling error-rate window."""
    def __init__(self, started, no_success, error_window=HEALTH_POLICY['error_window_seconds']):
        self.started = self.last_success = started
        self.no_success, self.error_window = no_success, error_window
        self.recent = deque()
        self.failures = 0
        self.failed_since_success = False

    def observe(self, row):
        failure = guard_failure(row)
        self.recent.append((row['end'], failure))
        self.failures += failure
        if row['success']:
            self.last_success = row['end']
            self.failed_since_success = False
        else:
            self.failed_since_success = True

    def evaluate(self, now, progress):
        while self.recent and self.recent[0][0] < now-self.error_window:
            self.failures -= self.recent.popleft()[1]
        latest = max(self.started, self.last_success, progress)
        reason = None
        if self.failed_since_success and now-latest >= self.no_success:
            reason = 'no_success'
        elif now-self.started >= self.error_window and self.recent and self.failures/len(self.recent)>HEALTH_POLICY['error_rate']:
            reason = 'error_rate'
        return dict(reason=reason, seconds_without_progress=now-latest,
                    window_requests=len(self.recent), window_failures=self.failures,
                    error_window_seconds=self.error_window)


class LoopLagMonitor:
    """Track scheduled wakeups independently of the load/aggregation loop."""
    def __init__(self, period, record):
        self.period, self.record = period, record
        self.expected = time.perf_counter() + period

    def sample(self):
        now = time.perf_counter()
        self.record(max(0, now-self.expected))
        self.expected = now+self.period

    async def run(self):
        while True:
            await asyncio.sleep(max(0,self.expected-time.perf_counter()))
            self.sample()


async def load(transport, samples, concurrency, duration, max_duration, min_requests,
               config, folder, label, slo, guard=False):
    folder.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    rows = MetricStore(folder/(label+'.metrics.sqlite'))
    windows, interval = [], []
    health = FailureGuard(started, config['timing']['no_success'])
    guard_state = None
    counter = 0
    stop = asyncio.Event()
    aborted = None
    dispatch_end = None
    completed = False
    window_start = started
    cpu_start = time.process_time()
    lag = []
    window_seconds = config['timing']['window']
    log_hash = hashlib.sha256()
    log = (folder/(label+'.requests.jsonl')).open('wb')
    window_log = (folder/(label+'.windows.jsonl')).open('w')

    def flush_window(now):
        nonlocal interval, window_start, cpu_start, lag
        elapsed = now-window_start
        if elapsed <= 0: return
        value = dict(start=window_start-started,end=now-started,
            metrics=metrics(interval,elapsed),
            cpu_cores=(time.process_time()-cpu_start)/elapsed,
            rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            event_loop_lag_max=max(lag,default=0))
        value['slo_pass'] = qualifies(value['metrics'],slo)
        windows.append(value)
        window_log.write(json.dumps(value)+'\n');window_log.flush()
        interval, window_start, cpu_start, lag = [], now, time.process_time(), []

    async def worker():
        nonlocal counter
        while not stop.is_set():
            index = counter
            counter += 1
            row = await transport(samples[index % len(samples)])
            row.update(sequence=index, phase=label)
            rows.append(row)
            # Full native text/ITL stays on disk, not in minute-long windows.
            interval.append({k:v for k,v in row.items() if k not in ('native_output','error')})
            if guard: health.observe(row)
            encoded = (json.dumps(row)+'\n').encode('utf-8')
            log.write(encoded);log.flush()
            log_hash.update(encoded)
            # Even an immediately failing mock/client must yield to the watchdog.
            await asyncio.sleep(0)

    monitor = LoopLagMonitor(min(.01,window_seconds), lambda delay: lag.append(delay))
    monitor_task = asyncio.create_task(monitor.run())
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        while not stop.is_set():
            await asyncio.sleep(min(.25,window_seconds))
            now = time.perf_counter()
            for task in workers:
                if task.done() and task.exception(): raise task.exception()
            elapsed = now-started
            if guard:
                guard_state = health.evaluate(now,getattr(transport,'last_progress',started))
                aborted = guard_state['reason']
            if aborted or elapsed >= max_duration or (elapsed >= duration and len(rows)>=min_requests):
                stop.set(); dispatch_end = now
            if now-window_start >= window_seconds: flush_window(now)
        # Complete in-flight requests. Each transport has its own finite timeout.
        await asyncio.gather(*workers)
        completed = True
    finally:
        stop.set()
        for task in workers:
            if not task.done(): task.cancel()
        await asyncio.gather(*workers,return_exceptions=True)
        ended = time.perf_counter()
        monitor.sample()
        monitor_task.cancel()
        await asyncio.gather(monitor_task,return_exceptions=True)
        flush_window(ended)
        log.close();window_log.close()
        if not completed: rows.close()
    elapsed = ended-started
    aggregate = rows.summary(elapsed)
    count = len(rows)
    rows.close()
    if counter != count:
        raise ValueError(f'{label}: dispatched {counter} requests but recorded {count}')
    result = dict(concurrency=concurrency,metrics=aggregate,
        report_contract=dict(version=1,token_count_version=1,requests_sha256=log_hash.hexdigest()),
        sufficient=count>=min_requests and (dispatch_end-started)>=duration,
        aborted=aborted, guard=guard_state, duration=elapsed, dispatch_seconds=dispatch_end-started,
        drain_seconds=ended-dispatch_end, windows=windows,
        client_bottleneck_suspected=any(w['event_loop_lag_max']>.1 for w in windows),
        requests_file=label+'.requests.jsonl', windows_file=label+'.windows.jsonl')
    result['slo_pass'] = qualifies(result['metrics'],slo)
    return result
