import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_suite as suite
import compass_runner as compass

class Tests(unittest.TestCase):
    def test_invalid_scope(self):
        for datasets in [[], ['unknown'], ['halluqa','halluqa'], 'halluqa', [1]]:
            with self.assertRaises(ValueError): suite.selected_stages({'datasets':datasets})

    def test_only_selected_paths_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={'datasets':['ceval'],'target':{'base_url':'http://t/v1'},'judge':{'base_url':'http://j/v1'}}
            self.assertEqual(suite.preflight(config,{'ceval':tmp,'financeiq':'/missing'}),{'ceval':str(Path(tmp).resolve())})

    def test_serial_execution_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'inputs').mkdir()
            config={'datasets':['halluqa','followbench'],
                    'target':{'model':'t','base_url':'http://t/v1','api_key_env':'T'},
                    'judge':{'model':'j','base_url':'http://j/v1','api_key_env':'J'}}
            cfg=root/'config.json';cfg.write_text(json.dumps(config))
            calls=[]
            def execute(name,*args):
                calls.append(name)
                if name=='halluqa':raise ValueError('test failure')
                return [['OpenCompass','followbench','HSR',50,'raw.json']]
            with patch.dict(os.environ,{'T':'secret-t','J':'secret-j'}), patch.object(sys,'argv',['suite','--config',str(cfg),'--output',str(root/'out')]), patch.object(suite,'read_mapping',return_value={'halluqa':str(root/'inputs'),'followbench_en':str(root/'inputs')}), patch.object(compass,'prepare',return_value=[]), patch.object(compass,'execute',side_effect=execute):
                self.assertEqual(suite.main(),1)
            self.assertEqual(calls,['halluqa','followbench'])
            text=(next((root/'out').iterdir())/'summary.json').read_text(); result=json.loads(text)
            self.assertEqual(result['overall_status'],'failed')
            self.assertEqual(result['status']['ceval'],'skipped')
            self.assertEqual(result['metrics'][0]['value'],50)
            self.assertNotIn('secret-t',text)
            self.assertIn('setup',result)

if __name__=='__main__':unittest.main()
