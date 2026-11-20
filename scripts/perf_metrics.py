"""Explicit metrics and selection rules; seconds internally."""
import math

HEALTH_POLICY = dict(version=1,min_success_rate=.999,recovery_success_rate=1.,
                     saturation_tail_points=2,saturation_growth=.05,near_peak_ratio=.95,
                     recovery_latency_ratio=1.2,error_window_seconds=60,error_rate=.2)


def percentile(values, p):
    if not values: return None
    values = sorted(values)
    pos = (len(values)-1)*p/100
    a,b = math.floor(pos),math.ceil(pos)
    return values[a]+(values[b]-values[a])*(pos-a)


def distribution(values):
    return {f'p{p}':percentile(values,p) for p in (50,95,99)}


def metrics(rows, seconds):
    good = [r for r in rows if r['success']]
    seconds = max(seconds,1e-9)
    result = dict(requests=len(rows), completed=len(good), success_rate=len(good)/len(rows) if rows else 0,
        requests_per_second=len(good)/seconds,
        output_tokens_per_second=sum(r['output_tokens'] for r in good)/seconds,
        input_tokens_per_second=sum(r['input_tokens'] for r in good)/seconds,
        input_token_mismatches=sum(r.get('input_token_delta') not in (None,0) for r in rows),
        truncated=sum(r['truncated'] for r in rows),
        short_outputs=sum(r['output_tokens']<2 for r in good),
        estimated_tokens=sum(r.get('token_source')=='estimated' for r in rows),
        reuse_rate=1-len({r['sample_id'] for r in rows})/len(rows) if rows else 0,
        errors={str(status):sum(r.get('http_status')==status and not r['success'] for r in rows)
                for status in {r.get('http_status') for r in rows if not r['success']}})
    for key in ('ttft','tpot','latency','input_tokens','output_tokens'):
        result[key] = distribution([r[key] for r in good if r[key] is not None])
    return result


def qualifies(m, slo):
    return stage_verdict(dict(metrics=m,sufficient=True),slo)['passed']


def stage_verdict(point, slo=None, perfect=False):
    """One health contract for scan, selection, recovery and acceptance."""
    m = point['metrics']
    reasons = []
    if not point['sufficient']: reasons.append('insufficient_load')
    if point.get('aborted'): reasons.append('guard_triggered')
    if point.get('client_bottleneck_suspected'): reasons.append('client_bottleneck')
    if not m['requests']: reasons.append('no_requests')
    threshold = HEALTH_POLICY['recovery_success_rate' if perfect else 'min_success_rate']
    if m['success_rate'] < threshold: reasons.append('success_rate')
    if m['truncated']: reasons.append('truncated')
    if m.get('input_token_mismatches',0): reasons.append('input_token_mismatch')
    if slo is not None:
        for key in ('ttft','tpot'):
            value = m[key]['p95']
            if value is None: reasons.append(key+'_unavailable')
            elif value*1000 > slo[key+'_ms']: reasons.append(key+'_slo')
    return dict(passed=not reasons, reasons=reasons)


def recovery_passes(point, baseline):
    return (stage_verdict(point, perfect=True)['passed']
        and stage_verdict(baseline, perfect=True)['passed']
        and all(point['metrics'][k]['p95'] is not None
                and baseline['metrics'][k]['p95'] is not None
                and point['metrics'][k]['p95'] <= baseline['metrics'][k]['p95']*HEALTH_POLICY['recovery_latency_ratio']
                for k in ('ttft','tpot')))


def select(points, slo):
    healthy = [p for p in points if stage_verdict(p)['passed']]
    eligible = [p for p in points if stage_verdict(p,slo)['passed']]
    peak = max((p['metrics']['output_tokens_per_second'] for p in eligible),default=0)
    sweet = min((p['concurrency'] for p in eligible if p['metrics']['output_tokens_per_second'] >= HEALTH_POLICY['near_peak_ratio']*peak),default=None)
    saturation = None
    if len(points)>=2 and all(stage_verdict(p)['passed'] for p in points[-2:]):
        previous = points[-2]['metrics']['output_tokens_per_second']
        last = points[-1]['metrics']['output_tokens_per_second']
        if previous > 0 and last <= previous*(1+HEALTH_POLICY['saturation_growth']):
            maximum = max(p['metrics']['output_tokens_per_second'] for p in healthy)
            saturation = min(p['concurrency'] for p in healthy if p['metrics']['output_tokens_per_second'] >= maximum*HEALTH_POLICY['near_peak_ratio'])
    return dict(sweet_concurrency=sweet,saturation_concurrency=saturation)


class MetricStore:
    """Disk-backed exact percentiles for multi-hour runs; bounded Python memory."""
    def __init__(self,path):
        import sqlite3
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE samples (sample_id TEXT, success INTEGER, truncated INTEGER, estimated INTEGER, status TEXT, ttft REAL, tpot REAL, latency REAL, input_tokens INTEGER, output_tokens INTEGER, mismatch INTEGER)')
        self.count=0
    def __len__(self): return self.count
    def append(self,r):
        self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)',(
            r['sample_id'],int(r['success']),int(r['truncated']),int(r.get('token_source')=='estimated'),
            str(r.get('http_status')),r['ttft'],r['tpot'],r['latency'],r['input_tokens'],r['output_tokens'],int(r.get('input_token_delta') not in (None,0))))
        self.count+=1
        if self.count%1000==0: self.db.commit()
    def summary(self,seconds):
        self.db.commit()
        count,good,truncated,estimated,distinct=self.db.execute('SELECT COUNT(*), COALESCE(SUM(success),0), COALESCE(SUM(truncated),0), COALESCE(SUM(estimated),0), COUNT(DISTINCT sample_id) FROM samples').fetchone()
        inp,out,short=self.db.execute('SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(output_tokens<2),0) FROM samples WHERE success=1').fetchone()
        seconds=max(seconds,1e-9)
        result=dict(input_token_mismatches=self.db.execute('SELECT COALESCE(SUM(mismatch),0) FROM samples').fetchone()[0],requests=count,completed=good,success_rate=good/count if count else 0,
            requests_per_second=good/seconds,input_tokens_per_second=inp/seconds,output_tokens_per_second=out/seconds,
            truncated=truncated,estimated_tokens=estimated,short_outputs=short,reuse_rate=1-distinct/count if count else 0,
            errors=dict(self.db.execute('SELECT status,COUNT(*) FROM samples WHERE success=0 GROUP BY status')))
        for key in ('ttft','tpot','latency','input_tokens','output_tokens'):
            n=self.db.execute(f'SELECT COUNT({key}) FROM samples WHERE success=1').fetchone()[0]
            result[key]={}
            for p in (50,95,99):
                value=None
                if n:
                    pos=(n-1)*p/100
                    values=[r[0] for r in self.db.execute(f'SELECT {key} FROM samples WHERE success=1 AND {key} IS NOT NULL ORDER BY {key} LIMIT 2 OFFSET ?', (math.floor(pos),))]
                    value=values[0]+(values[-1]-values[0])*(pos-math.floor(pos))
                result[key][f'p{p}']=value
        return result
    def close(self): self.db.commit();self.db.close()


def overload_verdict(result):
    """External acceptance only; never claim to observe the internal queue."""
    m = result['metrics']
    rejected = sum(m['errors'].get(str(code), 0) for code in (429,503))
    reasons = ['insufficient_load_duration' if reason=='insufficient_load' else reason
               for reason in stage_verdict(result)['reasons']]
    if rejected: reasons.append('requests_rejected')
    if not result.get('recovered'): reasons.append('not_recovered')
    return dict(passed=not reasons, reasons=reasons,
                rejection_observed=bool(rejected), internal_queue='unknown')
