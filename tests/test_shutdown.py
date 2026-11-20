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
    def test_sigterm_updates_report_and_stops_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config={'target':{'model':'t','base_url':'http://t/v1','api_key_env':'T'},
                    'judge':{'model':'j','base_url':'http://j/v1','api_key_env':'J'},
                    'datasets':['ceval','halluqa','followbench']}
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
    code="import os,time; from pathlib import Path; Path('child.pid').write_text(str(os.getpid())); time.sleep(60)"
    s.run([sys.executable,'-c',code],folder)
    raise AssertionError('child should be interrupted')
c.execute=stage
sys.argv=['eval-suite','--config',str(root/'config.json'),'--output',str(root/'run')]
sys.exit(s.main())
''')
            proc=subprocess.Popen([sys.executable,str(harness),str(Path(suite.__file__).parent),str(root)],
                                  env=dict(os.environ,T='x',J='y'),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            child=None
            try:
                pidfile=None
                deadline=time.monotonic()+10
                while proc.poll() is None and time.monotonic()<deadline:
                    candidates=list((root/'run').glob('*/ceval/child.pid'))
                    if candidates:
                        pidfile=candidates[0]
                        break
                    time.sleep(.05)
                self.assertIsNotNone(pidfile)
                child=int(pidfile.read_text())
                proc.send_signal(signal.SIGTERM)
                self.assertEqual(proc.wait(timeout=8),143)
                report=json.loads((pidfile.parent.parent/'summary.json').read_text())
                self.assertEqual(report['status']['ceval'],'interrupted')
                self.assertEqual(len(report['errors']),1)
                self.assertEqual(report['errors'][0]['dataset'],'ceval')
                self.assertEqual(report['status']['halluqa'],'not_run')
                with self.assertRaises(ProcessLookupError):
                    os.kill(child,0)
            finally:
                if proc.poll() is None:
                    proc.kill();proc.wait()
                if child:
                    try:os.kill(child,signal.SIGKILL)
                    except ProcessLookupError:pass

if __name__=='__main__':unittest.main()
