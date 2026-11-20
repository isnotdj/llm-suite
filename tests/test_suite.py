from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_suite as suite

class Tests(unittest.TestCase):
    def test_votes_reject_code_and_wrong_count(self):
        self.assertEqual(suite.votes("['YES','NO']",2),['YES','NO'])
        for text in ["[__import__('os').system('false')]", "['YES']", "['error','NO']"]:
            with self.assertRaises((ValueError,SyntaxError)):suite.votes(text,2)
    def test_failure_redacts_credentials(self):
        config={'target':{'api_key_env':'T'},'judge':{'api_key_env':'J'}}
        with patch.dict(suite.os.environ,{'T':'secret1','J':'secret2'}):
            detail=suite.failure_details(ValueError('secret1 secret2'),config)
        self.assertNotIn('secret1',detail);self.assertNotIn('secret2',detail)
    def test_reject_credentials_in_url(self):
        with self.assertRaises(ValueError):
            suite.preflight({'datasets':['ceval'],'target':{'base_url':'http://u:p@host'},'judge':{}},{})

if __name__=='__main__':unittest.main()
