"""Boundary cases for shared health decisions and independent guard clocks."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from perf_engine import FailureGuard
from perf_metrics import metrics, stage_verdict, recovery_passes, select, overload_verdict


def point():
    row=dict(sample_id='1',success=True,truncated=False,ttft=.1,tpot=.01,
             latency=.2,input_tokens=100,output_tokens=10,http_status=200)
    return dict(concurrency=1,metrics=metrics([row],1),sufficient=True,
                aborted=None,client_bottleneck_suspected=False)


class HealthPolicy(unittest.TestCase):
    def test_bad_points_excluded_consistently(self):
        for failure in ('sufficient','aborted','client','success','truncated','mismatch','empty'):
            p=point()
            if failure=='sufficient': p['sufficient']=False
            if failure=='aborted': p['aborted']='no_success'
            if failure=='client': p['client_bottleneck_suspected']=True
            if failure=='success': p['metrics']['success_rate']=.998
            if failure=='truncated': p['metrics']['truncated']=1
            if failure=='mismatch': p['metrics']['input_token_mismatches']=1
            if failure=='empty': p['metrics']['requests']=0
            with self.subTest(failure=failure):
                self.assertFalse(stage_verdict(p)['passed'])
                self.assertTrue(stage_verdict(p)['reasons'])
                self.assertIsNone(select([p],dict(ttft_ms=2000,tpot_ms=50))['sweet_concurrency'])
                self.assertFalse(recovery_passes(p,point()))
                self.assertFalse(overload_verdict(dict(p,recovered=True))['passed'])

    def test_health_and_latency_are_independent(self):
        p=point();p['metrics']['ttft']['p95']=3
        self.assertTrue(stage_verdict(p)['passed'])
        self.assertEqual(stage_verdict(p,dict(ttft_ms=2000,tpot_ms=50))['reasons'],['ttft_slo'])
        self.assertTrue(overload_verdict(dict(p,recovered=True))['passed'])

    def test_success_and_recovery_thresholds(self):
        p=point();p['metrics']['success_rate']=.999
        self.assertTrue(stage_verdict(p)['passed'])
        self.assertFalse(stage_verdict(p,perfect=True)['passed'])
        p=point();base=point()
        p['metrics']['ttft']['p95']=base['metrics']['ttft']['p95']*1.2
        self.assertTrue(recovery_passes(p,base))
        p['metrics']['ttft']['p95']+=1e-6
        self.assertFalse(recovery_passes(p,base))
        p=point();base['metrics']['tpot']['p95']=None
        self.assertFalse(recovery_passes(p,base))

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
