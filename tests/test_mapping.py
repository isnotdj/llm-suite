import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from dataset_mapping import read_mapping, resolve, working_copy

class Tests(unittest.TestCase):
    def test_absolute_mapping_does_not_require_unused_nas_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / 'bundled.json'
            data.write_text('[]')
            config = base / 'map.json'
            for root in ('unmounted-nas', str(base / 'unmounted-nas')):
                with self.subTest(root=root):
                    config.write_text(json.dumps({'version':1, 'root':root,
                        'datasets':{'halluqa':str(data), 'ceval':'missing'}}))
                    mappings = read_mapping(config)
                    self.assertEqual(resolve('halluqa', mappings), str(data.resolve()))
                    with self.assertRaises(FileNotFoundError):
                        resolve('ceval', mappings)
                    self.assertFalse((base / 'unmounted-nas').exists())

    def test_random_paths_are_mapped_without_renaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'NAS dir' / 'uuid-123'
            data.mkdir(parents=True)
            config = root / 'map.json'
            config.write_text(json.dumps({'version':1,'root':'NAS dir','datasets':{'gsm8k':'uuid-123', 'mmlu':'not-mounted'}}))
            args = {'gsm8k': {'dataset_id':resolve('gsm8k',read_mapping(config))}}
            self.assertEqual(args['gsm8k']['dataset_id'], str(data.resolve()))
            self.assertTrue(data.is_dir())
            with self.assertRaises(FileNotFoundError):
                resolve('mmlu', read_mapping(config))

    def test_halluqa_can_map_a_renamed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'random-name.json'
            data.write_text('[]')
            self.assertEqual(resolve('halluqa', {'halluqa':data}), str(data.resolve()))

    def test_working_copy_rejects_cache_under_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            source.mkdir()
            with patch.dict(os.environ, DATASET_WORK_CACHE=str(source/'cache')):
                with self.assertRaisesRegex(ValueError, 'overlaps input'):
                    working_copy(source, 'fixture')
            self.assertEqual(list(source.iterdir()), [])

    def test_working_copy_cannot_modify_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            original = source / 'dataset_infos.json'
            original.write_text('original')
            original.chmod(0o444)
            source.chmod(0o555)
            try:
                with patch.dict(os.environ, {'DATASET_WORK_CACHE': str(root / 'cache')}):
                    args = {'gsm8k':{'dataset_id':working_copy(source,'gsm8k')}}
                copied = Path(args['gsm8k']['dataset_id']) / original.name
                self.assertNotEqual(copied.stat().st_ino, original.stat().st_ino)
                copied.unlink()
                self.assertEqual(original.read_text(), 'original')
            finally:
                source.chmod(0o755)

    def test_unmapped_dataset_does_not_fall_back_silently(self):
        with self.assertRaises(ValueError):
            resolve('gsm8k', {})

if __name__ == '__main__':
    unittest.main()
