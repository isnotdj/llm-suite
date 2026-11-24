"""Pinned OpenCompass loaders/templates/scorers, with strict remote transport.

No model/tokenizer is loaded. Full prompts are sent without silent truncation.
"""
from eval_concurrency import records, write_records
import copy
import importlib.util
import json
import math
import os
from pathlib import Path

REVISION = 'f9d399363b4e00dbd5ada33096ca270969310274'
ROOT = Path(os.environ.get('OPENCOMPASS_ROOT', '/opt/benchmarks/OpenCompass'))
if (ROOT/'REVISION').exists():
    REVISION = (ROOT/'REVISION').read_text().strip()
CONFIGS = {
    'ceval': 'ceval/ceval_gen_5f30c7.py',
    'financeiq': 'FinanceIQ/FinanceIQ_gen_e0e6b5.py',
    'mathbench': 'MathBench/mathbench_2024_gen_19e486.py',
    'cmrc2018': 'CLUE_CMRC/CLUE_CMRC_gen_1bd3c8.py',
    'longbench': 'longbench/longbench_rawprompt.py',
}
SUPPORTED = tuple(CONFIGS) + ('logiqa-zh', 'finsafetybench', 'halluqa', 'followbench')


def selected(config):
    names = config.get('datasets')
    if (not isinstance(names, list) or not names or
            any(not isinstance(n, str) or n not in SUPPORTED for n in names) or
            len(set(names)) != len(names)):
        raise ValueError(f'datasets must be a unique nonempty list from {SUPPORTED}')
    return names


def load_configs(name, path, options):
    allowed = {'subsets', 'limit'} | ({'language'} if name in ('followbench','finsafetybench') else set())
    if not isinstance(options,dict) or set(options)-allowed:
        raise ValueError(f'{name}: unsupported options; allowed {sorted(allowed)}')
    limit = options.get('limit')
    if limit is not None and (isinstance(limit,bool) or not isinstance(limit,int) or limit <= 0):
        raise ValueError('limit must be a positive integer')
    if name in ('halluqa','followbench','finsafetybench') and 'subsets' in options:
        raise ValueError(f'{name}: subsets are not supported')
    if name in ('halluqa','followbench') and limit is not None:
        raise ValueError(f'{name}: limit is not supported')
    from mmengine.config import Config
    from opencompass.openicl.icl_prompt_template import PromptTemplate
    from opencompass.openicl.icl_retriever import ZeroRetriever
    from opencompass.openicl.icl_evaluator import AccEvaluator
    from opencompass.datasets import LogiQADataset
    from compass_plugins import LocalLongBench, CMRCOfficialEvaluator, OfficialDataset, OfficialEvaluator
    if name in ('halluqa', 'followbench', 'finsafetybench'):
        return [dict(type=OfficialDataset, abbr=name, path=path, name=name,
                     language=options.get('language', 'en'),
                     reader_cfg=dict(input_columns=['prompt'], output_column='reference'),
                     infer_cfg=dict(prompt_template=dict(type=PromptTemplate,
                         template='{prompt}'), retriever=dict(type=ZeroRetriever)),
                     eval_cfg=dict(evaluator=dict(type=OfficialEvaluator, name=name)))]
    if name == 'logiqa-zh':
        result = [dict(type=LogiQADataset, abbr='logiqa-zh-test', path=path, name='zh',
            reader_cfg=dict(input_columns=['context','query','A','B','C','D'],
                            output_column='correct_option', test_split='test'),
            infer_cfg=dict(prompt_template=dict(type=PromptTemplate,
                template='根据上下文回答逻辑推理题。只输出正确选项字母。\n{context}\n{query}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：'),
                retriever=dict(type=ZeroRetriever)),
            eval_cfg=dict(evaluator=dict(type=AccEvaluator)))]
    else:
        cfg = Config.fromfile(str(ROOT / 'opencompass/configs/datasets' / CONFIGS[name]))
        result = []
        for key, value in cfg.items():
            if key.endswith('_datasets') and isinstance(value, list):
                result.extend(copy.deepcopy(value))
        # Aggregate configs may expose both individual groups and combined lists.
        result = list({d['abbr']:d for d in result}.values())
        for d in result:
            d['path'] = path
            if name == 'mathbench':
                # Upstream uses mathbench_v1/<school level>/<question kind>.jsonl.
                original = next(x for x in cfg['mathbench_datasets'] if x['abbr'] == d['abbr'])
                d['path'] = str(Path(path) / Path(original['path']).name)
            elif name == 'longbench':
                d['type'] = LocalLongBench
            elif name == 'cmrc2018':
                d['eval_cfg']['evaluator'] = dict(type=CMRCOfficialEvaluator)
    subsets = options.get('subsets')
    if subsets is not None:
        available = {d['abbr'] for d in result}
        if (not isinstance(subsets, list) or not subsets or
            any(not isinstance(s,str) or s not in available for s in subsets) or len(set(subsets)) != len(subsets)):
            raise ValueError(f'{name}.subsets must select from {sorted(available)}')
        result = [d for d in result if d['abbr'] in subsets]
    if not result:
        raise ValueError(f'No configurations for {name}')
    return result


def construct(spec, **kwargs):
    spec = copy.deepcopy(spec)
    cls = spec['type']
    del spec['type']
    if not callable(cls):
        raise TypeError(f'Expected a pinned class, got {cls}')
    return cls(**spec, **kwargs)


def prepare(name, path, options):
    """Load and validate every selected dataset before any paid API call."""
    from opencompass.models.base_api import APITemplateParser
    from compass_plugins import limit_indices
    from evaluation_contracts import reference_answer
    parser = APITemplateParser(dict(round=[dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True)],
        reserved_roles=[dict(role='SYSTEM', api_role='system')]))
    tasks = []
    for config in load_configs(name, path, options):
        loader = {k:v for k,v in config.items() if k not in ('infer_cfg','eval_cfg')}
        dataset = construct(loader)
        infer = config['infer_cfg']
        retriever = construct(infer['retriever'], dataset=dataset)
        ice = construct(infer['ice_template']) if 'ice_template' in infer else None
        template = construct(infer['prompt_template']) if 'prompt_template' in infer else None
        indices = retriever.retrieve()
        test = dataset.test
        output = dataset.reader.output_column
        if not len(test) or len(indices) != len(test):
            raise ValueError(f'{config["abbr"]}: empty or inconsistent data')
        selected_ids = limit_indices(name, test, output, options.get('limit'))
        for example_id in {j for i in selected_ids for j in indices[i]}:
            reference_answer(name, dataset.train[example_id][output])
        prompts, references = [], []
        for i in selected_ids:
            reference = test[i][output]
            reference_answer(name, reference)
            prompt = retriever.generate_prompt_for_generate_task(i,
                retriever.generate_ice(indices[i], ice), ice_template=ice, prompt_template=template)
            parsed = parser.parse_template(prompt, mode='gen')
            if isinstance(parsed, str):
                messages = [dict(role='user', content=parsed)]
            else:
                messages = [dict(role=p['role'], content=p.get('content', p.get('prompt'))) for p in parsed]
            if not messages or any(not isinstance(p['content'],str) for p in messages):
                raise ValueError('Malformed prompt')
            prompts.append(messages)
            references.append(reference)
        tasks.append(dict(config=config, prompts=prompts, references=references,
                          ids=selected_ids, test=test.select(selected_ids)))
    return tasks


def execute(name, tasks, config, folder, mappings):
    from run_suite import client, generate_choice, save
    from opencompass.registry import TEXT_POSTPROCESSORS
    from compass_plugins import OfficialEvaluator, strict_option
    from run_inputs import iter_inputs, check_inputs
    from evaluation_contracts import validate_metrics
    api = client(config['target'])
    def predict(item):
        choice = generate_choice(api, config['target'], item['messages'])
        return dict(id=item['sample_id'], messages=item['messages'], reference=item['reference'],
                    prediction=choice.message.content, finish_reason=choice.finish_reason)

    rows = []
    for task in tasks:
        spec = task['config']; abbr = spec['abbr']
        work = folder / abbr; work.mkdir()
        raw = work / 'samples.jsonl'
        predictions, references = [], []
        write_records(raw, predict, iter_inputs(task), config.get('concurrency', 1), task['count'])
        for record in records(raw):
            predictions.append(record['prediction'])
            references.append(record['reference'])
        evaluator_cfg = spec['eval_cfg']
        pp = evaluator_cfg.get('pred_postprocessor')
        if pp:
            args = dict(pp); fn = args.pop('type')
            if isinstance(fn,str): fn = TEXT_POSTPROCESSORS.get(fn)
            predictions = [fn(p, **args) for p in predictions]
        if name == 'logiqa-zh': predictions = [strict_option(p) for p in predictions]
        evaluator = construct(evaluator_cfg['evaluator'])
        if isinstance(evaluator, OfficialEvaluator):
            evaluator.context = (config, work, mappings)
        result = evaluator.score(predictions=predictions, references=references)
        metrics = validate_metrics(name, result)
        check_inputs(task)
        if len(predictions) != len(references) or len(predictions) != task['count']:
            raise ValueError('Generation/scoring coverage is incomplete')
        report = work / 'metrics.json'
        save(report, dict(dataset=name, subset=abbr, count=len(predictions),
                          expected=task['count'], generated=len(predictions), scored=len(references),
                          coverage=len(predictions)/task['count'], metrics=metrics))
        for key, value in metrics.items():
            rows.append(['OpenCompass', name+'/'+abbr, key, value, str(report.relative_to(folder.parent))])
        # Save progress at subset boundaries even if a later subset fails.
        save(folder/'metrics.partial.json', rows)
    return rows
