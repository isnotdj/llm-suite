"""Offline container integration: tiny fixtures + loopback OpenAI-compatible API.

Not benchmark results. Exercises the real OpenCompass and official scorers.
"""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from offline_runtime import configure
configure()
from compass_runner import load_configs, prepare, SUPPORTED


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def fixtures(root):
    paths={}
    for name in SUPPORTED:
        data=root/name;data.mkdir();paths[name]=str(data)
        if name=='ceval':
            for cfg in load_configs(name,str(data),{}):
                for split in ('dev','val','test'):
                    write_csv(data/split/(cfg['name']+'_'+split+'.csv'),
                        ['id','question','A','B','C','D','answer','explanation'],
                        [[i,'Choose A','A','B','C','D','A',''] for i in range(5)])
        elif name=='financeiq':
            for cfg in load_configs(name,str(data),{}):
                for split in ('dev','test'):
                    write_csv(data/split/(cfg['name']+'.csv'),
                        ['id','question','A','B','C','D','answer'],
                        [[i,'Choose A','A','B','C','D','A'] for i in range(5)])
        elif name=='mathbench':
            for cfg in load_configs(name,str(data),{}):
                file=Path(cfg['path'])/(cfg['name']+'.jsonl');file.parent.mkdir(parents=True,exist_ok=True)
                file.write_text(json.dumps(dict(question='Choose A',options=['A','B','C','D'],answer='A'))+'\n')
        elif name=='logiqa-zh':
            for split in ('train','eval','test'):
                (data/f'zh_{split}.txt').write_text('\na\nContext\nQuestion\nA. option one\nB. option two\nC. option three\nD. option four\n')
        elif name=='cmrc2018':
            file=data/'renamed.json';paths[name]=str(file)
            file.write_text(json.dumps({'data':[{'paragraphs':[{'context':'A','qas':[{'id':'1','question':'What?', 'answers':[{'text':'A','answer_start':0}]}]}]}]}))
        elif name=='longbench':
            configs=load_configs(name,str(data),{});assert len(configs)==21, len(configs)
            for cfg in configs:
                answer={'passage_retrieval_en':'Paragraph 1', 'passage_retrieval_zh':'段落1', 'passage_count':'1'}.get(cfg['name'],'A')
                (data/(cfg['name']+'.jsonl')).write_text(json.dumps(dict(context='A',input='Question',answers=[answer],all_classes=['A','B'],length=1,language='en'))+'\n')
        elif name=='finsafetybench':
            file=data/'renamed.csv';paths[name]=str(file)
            write_csv(file,['id','question_en','category_en','question_zh','category_zh','source'],
                [['1','Test refusal','test','测试拒答','测试','synthetic']])
        elif name=='halluqa':
            records=json.loads(Path('/opt/benchmarks/HalluQA/HalluQA.json').read_text())[:1]
            (data/'HalluQA.json').write_text(json.dumps(records,ensure_ascii=False))
        elif name=='followbench':
            for category in ('content','situation','style','format','mixed','example'):
                file=category+'_constraints.json'
                records=json.loads((Path('/opt/benchmarks/FollowBench/data')/file).read_text())
                first_id=records[0]['example_id']
                records=[r for r in records if r['example_id']==first_id]
                assert {r['level'] for r in records} == (set(range(1,6)) if category=='example' else set(range(6)))
                (data/file).write_text(json.dumps(records,ensure_ascii=False))
            paths['followbench_en']=paths.pop(name)
    zh=root/'followbench_zh';zh.mkdir()
    for file in (Path('/opt/benchmarks/FollowBench/data_zh')).glob('*_constraints.json'):
        records=json.loads(file.read_text());first_id=records[0]['example_id']
        (zh/file.name).write_text(json.dumps([r for r in records if r['example_id']==first_id],ensure_ascii=False))
    paths['followbench_zh']=str(zh)
    return paths


class RequestProbe:
    def __init__(self, expected):
        self.expected = expected
        self.arrivals = 0
        self.release = threading.Event()
        self.condition = threading.Condition()

    def before_response(self):
        with self.condition:
            if self.release.is_set():
                return
            self.arrivals += 1
            self.condition.notify_all()
        if not self.release.wait(90):
            raise AssertionError('Timed out waiting for the test to release the response')

    def verify_limit(self):
        # All counted requests are still waiting for their first response byte.
        with self.condition:
            assert self.condition.wait_for(lambda: self.arrivals >= self.expected, timeout=60), self.arrivals
            exceeded = self.condition.wait_for(lambda: self.arrivals > self.expected, timeout=0.5)
            self.release.set()
            assert not exceeded, (self.expected, self.arrivals)


class API(BaseHTTPRequestHandler):
    probes = {}
    calls=0
    truncate=False
    invalid_judge=False
    on_first=None
    def log_message(self,*args): pass
    def do_POST(self):
        assert self.path=='/v1/chat/completions'
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        API.calls+=1
        if API.on_first is not None:
            callback=API.on_first;API.on_first=None;callback()
        probe = API.probes.get(request['model'])
        if probe is not None:
            probe.before_response()
        payload = self.response_payload(request)
        body=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

    def response_payload(self, request):
        prompt='\n'.join(m['content'] for m in request['messages'])
        if request.get('n')==5: answer='否'
        elif 'Rating: [[rating]]' in prompt: answer='Example: [[10]]\nRating: [[0]]' if API.invalid_judge else 'Rating: [[1]]'
        elif 'Python LIST with' in prompt:
            level=int(re.findall(r'Python LIST with (\d+) element',prompt)[-1]);answer=repr(['YES']*level)
        elif 'Python列表' in prompt:
            level=int(re.findall(r'包含(\d+)个元素',prompt)[-1]);answer=repr(['YES']*level)
        else: answer='A'
        return dict(id='offline-test',object='chat.completion',created=0,model='test',
            choices=[dict(index=i,message=dict(role='assistant',content=answer),finish_reason='length' if API.truncate else 'stop')
                     for i in range(request.get('n',1))],usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2))


def main():
    root=Path(tempfile.mkdtemp(prefix='compass-integration-'))
    paths=fixtures(root)
    mapping=root/'map.json';mapping.write_text(json.dumps(dict(version=1,root='/',datasets=paths)))
    server=ThreadingHTTPServer(('127.0.0.1',0),API)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    role=dict(base_url=f'http://127.0.0.1:{server.server_port}/v1',model='test',api_key_env='TEST_API_KEY')
    config=dict(concurrency=3,judge_concurrency=2,datasets=list(SUPPORTED),dataset_map=str(mapping),target=role,judge=role,
        options={name:dict(limit=1) for name in SUPPORTED if name not in ('halluqa','followbench')},
        continue_on_error=True)
    file=root/'config.json';file.write_text(json.dumps(config))
    env=dict(os.environ,TEST_API_KEY='test-only',NO_PROXY='127.0.0.1',no_proxy='127.0.0.1')
    cmd=[sys.executable,str(Path(__file__).with_name('run_suite.py')),'--config',str(file)]
    try:
        subprocess.run(cmd+['--output',str(root/'validation'),'--validate-only'],check=True,env=env)
        assert API.calls==0
        def remove_nas_inputs():
            # These are disposable fixtures, not the bundled official data.
            (Path(paths['halluqa'])/'HalluQA.json').unlink()
            for source in Path(paths['followbench_en']).glob('*.json'):source.unlink()
        API.on_first=remove_nas_inputs
        subprocess.run(cmd+['--output',str(root/'run')],check=True,env=env)
        report=json.loads((next((root/'run').iterdir())/'summary.json').read_text())
        assert report['overall_status']=='complete'
        assert report['schema_version']==2
        assert report['setup']['snapshot_complete'] and not report['errors']
        fin={m['metric']:m['value'] for m in report['metrics'] if m['dataset'].startswith('finsafetybench/')}
        assert fin['all/mean_judge_score']==1 and fin['all/ASR_score_10 (%)']==0
        assert not (Path(paths['halluqa'])/'HalluQA.json').exists()
        assert report['setup']['tasks']['halluqa'][0]['input_sha256']
        assert all(report['status'][n]=='complete' for n in SUPPORTED)
        assert report['setup']['tasks']['mathbench'][0]['count']==4
        assert all(m['framework']=='OpenCompass' for m in report['metrics'])
        cmrc={m['metric']:m['value'] for m in report['metrics'] if m['dataset'].startswith('cmrc2018/')}
        assert cmrc=={'EM':100.0,'F1':100.0}, cmrc
        assert all(m['value']==100 for m in report['metrics'] if m['dataset'].startswith('ceval/') and m['metric']=='accuracy')
        comparisons = []
        for concurrency in (1, 4):
            subset_config = dict(config, datasets=['ceval'], concurrency=concurrency,
                                 options={'ceval': {'subsets': ['ceval-computer_network'], 'limit': 5}})
            file.write_text(json.dumps(subset_config))
            output = root / f'ceval-c{concurrency}'
            subprocess.run(cmd + ['--output', str(output)], check=True, env=env)
            run_dir = next(output.iterdir())
            samples, = (run_dir / 'ceval').rglob('samples.jsonl')
            metrics = json.loads((run_dir / 'summary.json').read_text())['metrics']
            comparisons.append((samples.read_bytes(), [(m['dataset'], m['metric'], m['value']) for m in metrics]))
        assert comparisons[0] == comparisons[1]
        config['datasets']=['followbench'];config['options']={}
        config['followbench']={'language':'zh'}
        measured = dict(config, concurrency=4, judge_concurrency=2,
                        target=dict(role, model='probe-target'), judge=dict(role, model='probe-judge'))
        API.probes = {'probe-target': RequestProbe(4), 'probe-judge': RequestProbe(2)}
        file.write_text(json.dumps(measured))
        with subprocess.Popen(cmd+['--output',str(root/'followbench-zh')], env=env) as process:
            try:
                API.probes['probe-target'].verify_limit()
                API.probes['probe-judge'].verify_limit()
            finally:
                for probe in API.probes.values():
                    probe.release.set()
            assert process.wait() == 0
        zh_report=json.loads((next((root/'followbench-zh').iterdir())/'summary.json').read_text())
        assert zh_report['overall_status']=='complete'
        peaks = {model: probe.arrivals for model, probe in API.probes.items()}
        assert peaks == {'probe-target': 4, 'probe-judge': 2}, peaks
        API.truncate=True
        config['datasets']=['ceval']
        config['options']={'ceval':{'subsets':['ceval-computer_network'],'limit':1}}
        file.write_text(json.dumps(config))
        subprocess.run(cmd+['--output',str(root/'length')],check=True,env=env)
        run_dir = next((root/'length').iterdir())
        report_length = json.loads((run_dir/'summary.json').read_text())
        assert report_length['overall_status'] == 'complete' and report_length['metrics']
        samples, = (run_dir/'ceval').rglob('samples.jsonl')
        assert json.loads(samples.read_text())['finish_reason'] == 'length'
        API.truncate=False; API.invalid_judge=True
        config['datasets']=['finsafetybench'];config['options']={}
        file.write_text(json.dumps(config))
        rejected=subprocess.run(cmd+['--output',str(root/'bad-judge')],env=env)
        assert rejected.returncode==1
        failed=json.loads((next((root/'bad-judge').iterdir())/'summary.json').read_text())
        assert failed['overall_status']=='failed' and not failed['metrics']
        before=API.calls
        invalid_hallu=root/'empty-hallu.json'
        record=json.loads(Path('/opt/benchmarks/HalluQA/HalluQA.json').read_text())[0]
        for key in list(record):
            if key.startswith('Best Answer'):record[key]=''
        invalid_hallu.write_text(json.dumps([record]))
        paths['halluqa']=str(invalid_hallu)
        mapping.write_text(json.dumps(dict(version=1,root='/',datasets=paths)))
        config['datasets']=['halluqa'];file.write_text(json.dumps(config))
        rejected=subprocess.run(cmd+['--output',str(root/'empty-hallu'),'--validate-only'],env=env)
        assert rejected.returncode!=0 and API.calls==before
        failed=json.loads((next((root/'empty-hallu').iterdir())/'summary.json').read_text())
        assert failed['overall_status']=='failed' and not failed['metrics']
        assert failed['setup']['target']['model']=='test' and not failed['setup']['snapshot_complete']
        assert failed['errors'][0]['phase']=='preparation' and failed['errors'][0]['dataset']=='halluqa'
        print(json.dumps(dict(status='passed',datasets=list(SUPPORTED),api_calls=API.calls,peak_requests=peaks,
            metrics=len(report['metrics']),frozen_nas_inputs=True,invalid_reference_rejected_before_api=True,final_rating_validated=True,artifacts=str(root)),ensure_ascii=False))
    finally:server.shutdown()

if __name__=='__main__':main()
