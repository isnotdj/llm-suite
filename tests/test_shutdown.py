import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_suite as suite

class Tests(unittest.TestCase):
    def test_sigterm_updates_both_reports_and_stops_child(self):
        self.check_shutdown(False)

    def test_full_disk_still_stops_child_and_reports_error(self):
        self.check_shutdown(True)

    def check_shutdown(self, fail_write):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config={'target':{'model':'t','base_url':'http://t/v1','api_key_env':'T'},
                    'judge':{'model':'j','base_url':'http://j/v1','api_key_env':'J'},
                    'datasets':['halluqa','ceval','followbench']}
            (root/'config.json').write_text(json.dumps(config))
            harness=root/'harness.py'
            harness.write_text('''import sys,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import run_suite as s
import compass_runner as c
c.prepare=lambda *args: []
root=Path(sys.argv[2])
(root/'inputs').mkdir()
s.read_mapping=lambda *args: {'ceval':str(root/'inputs'),'halluqa':str(root/'inputs'),'followbench_en':str(root/'inputs')}
s.preflight=lambda *args,**kwargs: {'ceval':str(root/'inputs'),'halluqa':str(root/'inputs'),'followbench_en':str(root/'inputs')}
def stage(name,tasks,config,folder,mappings):
    import json
    import run_journal
    score = folder/'score/run.json'
    score.parent.mkdir()
    score.write_text(json.dumps(dict(status='running')))
    if os.environ['FAIL_WRITE'] == '1':
        def fail(*args):
            raise OSError(28, 'No space left on device')
        s.atomic_json = fail
        run_journal.atomic_json = fail
    code="import os,time; from pathlib import Path; Path('child.pid').write_text(str(os.getpid())); time.sleep(60)"
    s.run([sys.executable,'-c',code],folder,score_report=score)
    raise AssertionError('child should be interrupted')
c.execute=stage
sys.argv=['eval-suite','--config',str(root/'config.json'),'--output',str(root/'run')]
sys.exit(s.cli())
''')
            proc=subprocess.Popen([sys.executable,str(harness),str(Path(suite.__file__).parent),str(root)],
                                  env=dict(os.environ,T='x',J='y',FAIL_WRITE=str(int(fail_write))),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
            child=None
            try:
                pidfile=None
                deadline=time.monotonic()+10
                while proc.poll() is None and time.monotonic()<deadline:
                    candidates=list((root/'run').glob('*/halluqa/child.pid'))
                    if candidates:
                        pidfile=candidates[0]
                        break
                    time.sleep(.05)
                self.assertIsNotNone(pidfile)
                child=int(pidfile.read_text())
                proc.send_signal(signal.SIGTERM)
                _, errors = proc.communicate(timeout=8)
                if fail_write:
                    self.assertEqual(proc.returncode,143)
                    self.assertIn('No space left on device',errors)
                else:
                    self.assertEqual(proc.returncode,143)
                    report=json.loads((pidfile.parent.parent/'summary.json').read_text())
                    self.assertEqual(report['status']['halluqa'],'interrupted')
                    self.assertEqual(report['overall_status'],'interrupted')
                    self.assertEqual(len(report['errors']),1)
                    self.assertEqual(report['errors'][0]['dataset'],'halluqa')
                    self.assertEqual(report['status']['ceval'],'not_run')
                    score=json.loads((pidfile.parent/'score/run.json').read_text())
                    self.assertEqual(score['status'],'interrupted')
                with self.assertRaises(ProcessLookupError):
                    os.kill(child,0)
            finally:
                if proc.poll() is None:
                    proc.kill();proc.wait()
                if child:
                    try:os.kill(child,signal.SIGKILL)
                    except ProcessLookupError:pass

if __name__=='__main__':unittest.main()
