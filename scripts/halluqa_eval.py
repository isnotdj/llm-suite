"""HalluQA transport adapter. Official source and prompts remain unchanged."""
from eval_concurrency import check_cancelled, records, write_records
from run_journal import atomic_json
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback


class JudgeFailure(ValueError):
    """The judge request failed or returned invalid votes."""


def official_prompt(path):
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_prompt')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['get_prompt']


def request(client, parameters, attempts=3, sleep=time.sleep):
    import openai
    for attempt in range(attempts):
        check_cancelled()
        try:
            return client.chat.completions.create(**parameters)
        except (openai.APIConnectionError, openai.APITimeoutError, openai.APIStatusError) as exc:
            status = getattr(exc, 'status_code', None)
            if status is not None and status != 429 and status < 500:
                raise
            if attempt + 1 == attempts:
                raise
            sleep(min(2 ** attempt + random.random(), 8))


def judge(client, parameters, sleep=time.sleep):
    import openai
    # Match official five votes and up to five attempts at a valid vote batch.
    for _ in range(5):
        try:
            response = request(client, {**parameters, 'n': 5}, sleep=sleep)
        except openai.APIError as exc:
            raise JudgeFailure('HalluQA judge request failed') from exc
        choices = getattr(response, 'choices', None)
        if not isinstance(choices, list) or len(choices) != 5:
            continue
        votes = [getattr(getattr(choice, 'message', None), 'content', None) for choice in choices]
        if (all(vote in ('是', '否') for vote in votes)
                and all(getattr(choice, 'finish_reason', None) in ('stop', 'length') for choice in choices)):
            return votes
    raise JudgeFailure('Judge failed to return exactly five valid 是/否 votes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--official-dir', type=Path, default=Path('/opt/benchmarks/HalluQA'))
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if not args.judge_model or not os.environ.get('OPENAI_API_KEY'):
        parser.error('JUDGE_MODEL/--judge-model and OPENAI_API_KEY are required')
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    from offline_runtime import configure
    configure([args.base_url or 'https://api.openai.com/v1'])
    from openai import OpenAI
    from importlib.metadata import version
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=args.base_url,
                    timeout=args.timeout, max_retries=0)
    source = args.official_dir / 'calculate_metrics.py'
    from dataset_mapping import read_mapping, resolve
    data_path = Path(resolve('halluqa', read_mapping(), args.official_dir / 'HalluQA.json'))
    if data_path.is_dir():
        data_path = data_path / 'HalluQA.json'
    from evaluation_contracts import hallu_records
    resource = {item['question_id']: item for item in hallu_records(json.loads(data_path.read_text()))}
    data = json.loads(args.responses.read_text())
    if not isinstance(data, list) or not data:
        parser.error('Responses must be a nonempty JSON array')
    ids = [item['question_id'] for item in data]
    if len(set(ids)) != len(ids) or any(i not in resource for i in ids):
        parser.error('Response question IDs must be unique and present in HalluQA')
    if any(not isinstance(item.get('response'), str) for item in data):
        parser.error('Every response must be a string')
    # Require the full official set to prevent partial-run scores masquerading as the benchmark.
    if set(ids) != set(resource):
        parser.error('Responses must cover the full official HalluQA question set')
    prompt = official_prompt(source)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'status': 'running', 'judge_model': args.judge_model,
        'base_url': str(client.base_url), 'openai_version': version('openai'),
        'temperature': 0, 'top_p': 0.5, 'votes': 5, 'max_tokens': 10,
        'timeout_seconds': args.timeout, 'transport_attempts': 3,
        'invalid_batch_attempts': 5, 'sample_count': len(data),
        'official_commit': '7e5c67398d7fee013c37ec1d6afafa76f9836180',
        'sha256': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (source, data_path, args.responses, Path(__file__))},
    }
    meta_path = args.output_dir / 'run.json'
    atomic_json(meta_path, metadata)
    def judge_sample(item):
        sample, messages = prompt(dict(item), resource)
        votes = judge(client, dict(model=args.judge_model, messages=messages,
                                  temperature=0, top_p=0.5, max_tokens=10))
        hallucination = votes.count('是') > 2 or sample['response'] == ''
        return {**sample, 'votes': votes, 'is_hallucination': hallucination}

    try:
        path = args.output_dir/'samples.jsonl'
        write_records(path, judge_sample, data, args.concurrency, len(data))
        correct = sum(not record['is_hallucination'] for record in records(path))
        metadata.update(status='complete', non_hallucination_rate=correct / len(data) * 100)
    except Exception as exc:
        # No aggregate score is emitted for incomplete/invalid evaluation.
        metadata.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        atomic_json(meta_path, metadata)


def cli():
    try:
        main()
    except JudgeFailure:
        traceback.print_exc()
        return 1
    except OSError:
        traceback.print_exc()
        return os.EX_IOERR
    except Exception:
        traceback.print_exc()
        return os.EX_SOFTWARE
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
