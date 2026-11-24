"""Boundary cases for shared health decisions and independent guard clocks."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from perf_engine import FailureGuard
from perf_metrics import metrics, stage_verdict, select, overload_verdict


def point():
    row=dict(sample_id='1',success=True,truncated=False,ttft=.1,tpot=.01,
             latency=.2,input_tokens=100,output_tokens=10,http_status=200)
    return dict(concurrency=1,metrics=metrics([row],1),sufficient=True,
                aborted=None,client_bottleneck_suspected=False)


class HealthPolicy(unittest.TestCase):
    def test_soak_stability_uses_complete_dispatch_windows(self):
        from perf_metrics import soak_stability
        def window(start, end, rate, passed=True):
            return dict(start=start, end=end, slo_pass=passed,
                        metrics=dict(output_tokens_per_second=rate, ttft={'p95': .1}, tpot={'p95': .01}))
        active = [window(t, t+60, 100) for t in range(0, 3600, 60)]
        drain = [window(t, t+60, 0, False) for t in range(3600, 4200, 60)]
        result = soak_stability(active + drain, 3600, 60)
        self.assertEqual(result['drift_last_vs_first_30min'], dict(output_tokens_per_second=0, ttft=0, tpot=0))
        self.assertEqual(result['window_slo_pass_rate'], 1)
        active[0] = window(0, 60, 0, False)
        active[0]['metrics']['ttft']['p95'] = None
        result = soak_stability(active + [window(3590, 3650, 0, False)], 3600, 60)
        self.assertAlmostEqual(result['window_slo_pass_rate'], 59/60)
        self.assertIsNone(result['drift_last_vs_first_30min']['ttft'])
        short = soak_stability([window(0, 58, 100)], 58, 60)
        self.assertIsNone(short['window_slo_pass_rate'])
        self.assertTrue(all(v is None for v in short['drift_last_vs_first_30min'].values()))
        zero = soak_stability([window(0, 60, 0), window(3540, 3600, 100)], 3600, 60)
        self.assertIsNone(zero['drift_last_vs_first_30min']['output_tokens_per_second'])

    def test_bad_points_excluded_consistently(self):
        for failure in ('sufficient','aborted','client','success','mismatch','empty'):
            p=point()
            if failure=='sufficient': p['sufficient']=False
            if failure=='aborted': p['aborted']='no_success'
            if failure=='client': p['client_bottleneck_suspected']=True
            if failure=='success': p['metrics']['success_rate']=.998
            if failure=='mismatch': p['metrics']['input_token_mismatches']=1
            if failure=='empty': p['metrics']['requests']=0
            with self.subTest(failure=failure):
                self.assertFalse(stage_verdict(p)['passed'])
                self.assertTrue(stage_verdict(p)['reasons'])
                self.assertIsNone(select([p],dict(ttft_ms=2000,tpot_ms=50))['sweet_concurrency'])
                self.assertFalse(overload_verdict(dict(p,recovered=True))['passed'])

    def test_length_termination_is_healthy(self):
        p = point()
        p['metrics']['truncated'] = 1
        self.assertTrue(stage_verdict(p)['passed'])
        self.assertTrue(overload_verdict(dict(p, recovered=True))['passed'])

    def test_health_and_latency_are_independent(self):
        p=point();p['metrics']['ttft']['p95']=3
        self.assertTrue(stage_verdict(p)['passed'])
        self.assertEqual(stage_verdict(p,dict(ttft_ms=2000,tpot_ms=50))['reasons'],['ttft_slo'])
        self.assertTrue(overload_verdict(dict(p,recovered=True))['passed'])

    def test_tail_must_be_adjacent_and_healthy(self):
        points=[point() for _ in range(3)]
        for n,p in enumerate(points): p['concurrency']=2**n
        self.assertEqual(select(points,dict(ttft_ms=2000,tpot_ms=50))['saturation_concurrency'],1)
        for index in (1,2):
            broken=copy.deepcopy(points);broken[index]['sufficient']=False
            self.assertIsNone(select(broken,dict(ttft_ms=2000,tpot_ms=50))['saturation_concurrency'])


class GuardPolicy(unittest.TestCase):
    def observe(self, guard, end, success=False, status=503):
        guard.observe(dict(end=end,success=success,http_status=status))

    def test_initial_success_does_not_mask_later_stall(self):
        guard=FailureGuard(0,30)
        self.observe(guard,1,True,200);self.observe(guard,2)
        self.assertIsNone(guard.evaluate(30.99,0)['reason'])
        self.assertEqual(guard.evaluate(31,0)['reason'],'no_success')

    def test_progress_and_success_reset_deadline(self):
        guard=FailureGuard(100,30)
        self.observe(guard,101)
        self.assertIsNone(guard.evaluate(140,139)['reason'])
        self.observe(guard,141,True,200)
        self.assertIsNone(guard.evaluate(180,139)['reason'])

    def test_waiting_without_failure_does_not_trip(self):
        guard=FailureGuard(100,30)
        self.assertIsNone(guard.evaluate(200,0)['reason'])

    def test_old_failure_does_not_disappear_before_progress_deadline(self):
        guard=FailureGuard(0,90)
        self.observe(guard,1)
        self.assertIsNone(guard.evaluate(62,0)['reason'])
        self.assertEqual(guard.evaluate(90,0)['reason'],'no_success')

    def test_error_rate_uses_separate_full_window(self):
        guard=FailureGuard(0,1000)
        self.observe(guard,10)
        for end in (20,30,40,50): self.observe(guard,end,True,200)
        self.assertIsNone(guard.evaluate(60,59)['reason'])  # exactly 20%
        self.observe(guard,60)
        self.assertEqual(guard.evaluate(60,59)['reason'],'error_rate')
        self.assertIsNone(guard.evaluate(121,120)['reason'])  # aged out

    def test_429_is_refusal_not_server_error(self):
        guard=FailureGuard(0,1000)
        self.observe(guard,59,status=429)
        self.assertIsNone(guard.evaluate(60,59)['reason'])
