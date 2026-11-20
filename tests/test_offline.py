import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from offline_runtime import configure

class Tests(unittest.TestCase):
    def test_resource_flags_without_api_restrictions(self):
        connect, resolver = socket.socket.connect, socket.getaddrinfo
        with patch.dict(os.environ, {'HTTPS_PROXY':'http://proxy:1234'}, clear=True):
            configure(['http://api.example:8000/v1'])
            for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):
                self.assertEqual(os.environ[key],'1')
            self.assertEqual(os.environ['HTTPS_PROXY'],'http://proxy:1234')
            self.assertNotIn('EVAL_API_ENDPOINTS',os.environ)
            self.assertNotIn('PYTHONPATH',os.environ)
            self.assertIs(socket.socket.connect,connect)
            self.assertIs(socket.getaddrinfo,resolver)

if __name__=='__main__':unittest.main()
