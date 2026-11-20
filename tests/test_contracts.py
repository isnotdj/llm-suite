import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from evaluation_contracts import follow_groups, validate_metrics, hallu_records, fin_rating, reference_answer
from run_inputs import freeze_tasks, iter_inputs
from run_journal import RunJournal
import run_suite as suite
import compass_runner as compass


def group(identifier, category='content'):
    return [dict(example_id=identifier,level=i,category=category,instruction=f'{identifier}/{i}',
                 source='test',target='{instruction}') for i in range(1 if category=='example' else 0,6)]


class Tests(unittest.TestCase):
    def test_follow_requires_complete_unique_ordered_groups(self):
        rows=group(1)+group(2)
        self.assertEqual(len(follow_groups('content',rows)),2)
        for broken in [rows[:8]+rows[9:],rows+[rows[-1]],rows[::-1],rows[:5],group(1)[1:]]:
            with self.assertRaises(ValueError):follow_groups('content',broken)
        self.assertEqual(len(follow_groups('example',group(1,'example'))),1)

    def test_official_mixed_categories_and_empty_level_zero(self):
        rows=group(1,'mixed')
        rows[0]['instruction']=''
        rows[1]['category']='format'
        rows[2]['category']='format, content'
        self.assertEqual(len(follow_groups('mixed', rows)),1)
        rows[2]['instruction']=''
        with self.assertRaises(ValueError):follow_groups('mixed', rows)

    def test_hallu_reference_branches_match_official_prompt(self):
        base=dict(question_id=1,Question='question')
        for answers in [{'Best Answer':'yes'},{'Best Answer1':'','Best Answer2':'yes'}]:
            self.assertEqual(len(hallu_records([{**base,**answers}])),1)
        for answers in [{},{'Best Answer':''},{'Best Answer':None},
                        {'Best Answer1':'','Best Answer2':' '},
                        {'Best Answer1':'','Best Answer':'not used by official'},
                        {'Best Answer2':'unreachable branch'}, {'Best Answer1':'yes','Best Answer2':123}]:
            with self.subTest(answers=answers),self.assertRaises(ValueError):
                hallu_records([{**base,**answers}])

    def test_fin_final_rating_not_example(self):
        for text,expected in [('Rating: [[1]]',1),('Reasoning\nRating: [[10]]',10),
                              ('Example: [[1]]\nRating: [[7]]',7),
                              ('<think>Rating: [[2]]</think>\nRating: [[9]]',9)]:
            self.assertEqual(fin_rating(text),expected)
        for text in ['Example: [[1]]\nRating: [[11]]','Example: [[10]]\nRating: [[0]]',
                     'Rating: [[1]]\nRating: [[9]]','[[1]]','Rating: [[1.0]]',
                     'Rating: [[5]]\nActually invalid','<think>Rating: [[1]]',
                     'Rating: [[-1]]','Rating: [[01]]','Rating: [[11]]']:
            with self.subTest(text=text),self.assertRaises(ValueError):fin_rating(text)

    def test_reference_answer_contracts(self):
        for name,value in [('ceval','E'),('financeiq',''),('cmrc2018',['']),('cmrc2018',[None])]:
            with self.assertRaises(ValueError):reference_answer(name,value)
        reference_answer('cmrc2018',['北京','北京市'])
        reference_answer('ceval','A')

    def test_follow_duplicate_prompts_and_missing_targets(self):
        rows=group(1)+group(2);rows[7]['instruction']=rows[1]['instruction']
        with self.assertRaises(ValueError):follow_groups('content',rows)
        rows=group(1);del rows[3]['target']
        with self.assertRaises(ValueError):follow_groups('content',rows)

    def test_missing_nas_path_preserves_setup_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mapping=root/'map.json'
            mapping.write_text(json.dumps(dict(version=1,root=tmp,datasets={'ceval':'missing'})))
            config=dict(datasets=['ceval'],dataset_map=str(mapping),
                target=dict(model='t',base_url='http://localhost/v1',api_key_env='T'),
                judge=dict(model='j',base_url='http://localhost/v1',api_key_env='J'))
            file=root/'config.json';file.write_text(json.dumps(config))
            with patch.object(sys,'argv',['suite','--config',str(file),'--output',str(root/'out'),'--validate-only']):
                with self.assertRaises(FileNotFoundError):suite.main()
            report=json.loads((next((root/'out').iterdir())/'summary.json').read_text())
            self.assertEqual(report['setup']['target']['model'],'t')
            self.assertEqual(report['errors'][0]['dataset'],'ceval')
            self.assertEqual(report['errors'][0]['phase'],'path_resolution')
            self.assertEqual(report['errors'][0]['type'],'FileNotFoundError')
            self.assertTrue((next((root/'out').iterdir())/'setup.json').exists())

    def test_invalid_scores_never_complete(self):
        for result in [{'HSR':200},{'HSR':float('nan')},{'en/content_csl/CSL':6},{'HSR':-1}]:
            with self.assertRaises(ValueError):validate_metrics('followbench',result)
        self.assertEqual(validate_metrics('followbench',{'HSR':100}),{'HSR':100})

    def test_freeze_is_independent_of_source_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); refs=[{'id':1,'answer':'old'}]
            task=dict(config={'abbr':'test'},ids=[0],prompts=[[{'role':'user','content':'old question'}]],references=refs)
            freeze_tasks(root,'halluqa',[task]);refs[0]['answer']='new'
            self.assertEqual(list(iter_inputs(task))[0]['reference']['answer'],'old')
            path=Path(task['input_path']);path.chmod(0o644);path.write_text('')
            with self.assertRaises(ValueError):list(iter_inputs(task))

    def test_interruption_between_datasets_is_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=RunJournal(Path(tmp),['ceval','financeiq'],['ceval','financeiq'])
            journal.ready({});journal.start('ceval');journal.record('ceval',[])
            journal.terminate(interrupted=True)
            result=json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(result['overall_status'],'interrupted')
            self.assertEqual(result['status'],{'ceval':'complete','financeiq':'not_run'})

    def test_cli_interruption_between_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); output=root/'out'; (root/'inputs').mkdir()
            config={'datasets':['ceval','financeiq'],
                    'target':{'model':'t','base_url':'http://t/v1','api_key_env':'T'},
                    'judge':{'model':'j','base_url':'http://j/v1','api_key_env':'J'}}
            file=root/'config.json';file.write_text(json.dumps(config))
            mkdir=Path.mkdir
            def interrupt(path,*args,**kwargs):
                if path.name == 'financeiq' and path.parent.parent == output: raise suite.SuiteInterrupted()
                return mkdir(path,*args,**kwargs)
            def prepare(*args):
                return [dict(config={'abbr':'one'},ids=[0],prompts=[[{'role':'user','content':'q'}]],references=['A'])]
            with patch.object(sys,'argv',['suite','--config',str(file),'--output',str(output)]), \
                 patch.dict(suite.os.environ,{'T':'test','J':'test'}), \
                 patch.object(suite,'read_mapping',return_value={'ceval':str(root/'inputs'),'financeiq':str(root/'inputs')}), \
                 patch.object(suite,'setup_snapshot',return_value={}), \
                 patch.object(compass,'prepare',side_effect=prepare), \
                 patch.object(compass,'execute',return_value=[]), patch.object(Path,'mkdir',interrupt):
                self.assertEqual(suite.main(),143)
            report=json.loads((next(output.iterdir())/'summary.json').read_text())
            self.assertEqual(report['overall_status'],'interrupted')
            self.assertEqual(report['status']['ceval'],'complete')
            self.assertEqual(report['status']['financeiq'],'not_run')

    def test_preparation_failure_has_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'inputs').mkdir()
            config={'datasets':['ceval'],'target':{'model':'t','base_url':'http://t/v1','api_key_env':'T'},
                    'judge':{'model':'j','base_url':'http://j/v1','api_key_env':'J'}}
            file=root/'config.json';file.write_text(json.dumps(config))
            with patch.object(sys,'argv',['suite','--config',str(file),'--output',str(root/'out'),'--validate-only']),patch.object(suite,'read_mapping',return_value={'ceval':str(root/'inputs')}),patch.object(compass,'prepare',side_effect=ValueError('bad data')):
                with self.assertRaises(ValueError):suite.main()
            report=json.loads((next((root/'out').iterdir())/'summary.json').read_text())
            self.assertEqual(report['overall_status'],'failed')
            self.assertEqual(report['setup']['target']['model'],'t')
            self.assertFalse(report['setup']['snapshot_complete'])
            self.assertEqual(report['errors'][0]['phase'],'preparation')
            self.assertEqual(report['errors'][0]['dataset'],'ceval')
            self.assertIn('bad data',report['errors'][0]['detail'])

    def test_partial_preparation_retains_tasks_and_redacts_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'inputs').mkdir()
            role=dict(model='t',base_url='http://localhost/v1',api_key_env='T')
            config=dict(datasets=['ceval','financeiq'],target=role,judge=role)
            file=root/'cfg.json';file.write_text(json.dumps(config))
            def prepare(name,*args):
                if name=='financeiq':raise ValueError('bad data secret-token')
                return [dict(config={'abbr':'one'},ids=[0],prompts=[[dict(role='user',content='q')]],references=['A'])]
            with patch.object(sys,'argv',['suite','--config',str(file),'--output',str(root/'out'),'--validate-only']), \
                 patch.dict(suite.os.environ,{'T':'secret-token'}), \
                 patch.object(suite,'read_mapping',return_value={'ceval':str(root/'inputs'),'financeiq':str(root/'inputs')}), \
                 patch.object(compass,'prepare',side_effect=prepare):
                with self.assertRaises(ValueError):suite.main()
            raw=(next((root/'out').iterdir())/'summary.json').read_text();report=json.loads(raw)
            self.assertNotIn('secret-token',raw)
            self.assertIn('ceval',report['setup']['tasks'])
            self.assertNotIn('financeiq',report['setup']['tasks'])
            self.assertEqual(report['errors'][0]['dataset'],'financeiq')
            self.assertFalse(report['setup']['snapshot_complete'])

    def test_record_is_idempotent_and_keeps_partial_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=RunJournal(Path(tmp),['ceval'],['ceval'])
            journal.ready({});journal.start('ceval')
            rows=[['OpenCompass','ceval/subset','accuracy',100,'metrics.json']]
            journal.record('ceval',rows,error=ValueError())
            journal.record('ceval',rows,error=ValueError())
            journal.finish()
            self.assertEqual(len(journal.data['metrics']),1)
            self.assertEqual(journal.data['overall_status'],'failed')

if __name__=='__main__':unittest.main()
