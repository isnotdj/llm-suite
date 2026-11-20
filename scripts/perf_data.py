"""Local, reproducible performance workloads. Never execute dataset code."""
from perf_resources import digest
import json
import math
import random
from pathlib import Path

SOURCES = {'random', 'sharegpt', 'livecodebench', 'longcodebench', 'long-code-arena'}


def token_count(tokenizer, messages, template_kwargs=None):
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **(template_kwargs or {})))


def records(path):
    path = Path(path)
    with path.open('rb') as source:
        if path.suffix == '.jsonl':
            for line in source:
                if line.strip(): yield json.loads(line)
        else:
            import ijson
            # Reject objects instead of silently accepting an empty iterator.
            char = source.read(1)
            while char and char.isspace(): char = source.read(1)
            if char != b'[': raise ValueError('Dataset must be a JSON array or JSONL')
            source.seek(0)
            try:
                yield from ijson.items(source, 'item', use_float=True)
            except ijson.common.JSONError as exc:
                raise ValueError(f'Invalid dataset JSON: {path}') from exc


def messages_for(row, source):
    if 'messages' in row:
        messages = row['messages']
    elif source == 'sharegpt':
        roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
        messages = [{'role': roles[r['from']], 'content': r['value']} for r in row['conversations']]
        # Never leak the recorded answer into the generation prompt.
        while messages and messages[-1]['role'] == 'assistant':
            messages.pop()
    elif source == 'livecodebench':
        messages = [{'role': 'user', 'content': row['question_content'] + '\n' + (row.get('starter_code') or '')}]
    else:
        # Long-context benchmarks have task-specific repository assembly. Consume
        # their prepared prompts, never concatenate unrelated repositories here.
        messages = [{'role': 'user', 'content': row['prompt']}]
    if not messages or messages[-1]['role'] != 'user' or any(
        m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str)
        for m in messages
    ):
        raise ValueError('Expected text chat messages ending in user')
    return messages


def prepare(config, tokenizer, root):
    source = config['dataset']['name']
    rng = random.Random(config['seed'])
    prepared, unavailable = {}, {}
    for profile in config['profiles']:
        name = profile['name']
        lo = math.ceil(profile['input_tokens'] * (1-profile['variation']))
        hi = math.floor(profile['input_tokens'] * (1+profile['variation']))
        if hi + config['max_generation_tokens'] > config['context_window']:
            raise ValueError(f'{name}: input upper bound {hi} plus output budget exceeds context_window')
        target = profile.get('output_target')
        hint = f'\nAim for approximately {target} output tokens; finish naturally when the answer is complete.' if target else ''
        samples = []
        if source == 'random':
            vocab = tokenizer.vocab_size
            special = set(tokenizer.all_special_ids)
            for i in range(config['sample_count']):
                desired = rng.randint(lo, hi)
                ids = [rng.randrange(vocab) for _ in range(desired)]
                ids = [n for n in ids if n not in special]
                # Round-trip may change token counts. Adjust and check the full
                # template instead of assuming random token IDs are preserved.
                for attempt in range(80):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    messages = [{'role':'user','content':text + hint}]
                    length = token_count(tokenizer, messages, config['target'].get('extra_body',{}).get('chat_template_kwargs',{}))
                    if lo <= length <= hi:
                        break
                    if length > hi:
                        ids = ids[:max(1, len(ids)-(length-desired))]
                    else:
                        ids.extend(rng.randrange(vocab) for _ in range(max(1, desired-length)))
                else:
                    raise ValueError(f'{name}: unable to generate in-range random input')
                samples.append({'id':str(i), 'messages':messages, 'input_tokens':length})
        else:
            matched = 0
            for i, row in enumerate(records(config['dataset']['path'])):
                messages = [dict(m) for m in messages_for(row, source)]
                messages[-1]['content'] += hint
                length = token_count(tokenizer, messages, config['target'].get('extra_body',{}).get('chat_template_kwargs',{}))
                if lo <= length <= hi:
                    item = {'id':str(row.get('id', row.get('question_id', i))),
                            'messages':messages, 'input_tokens':length}
                    matched += 1
                    if len(samples) < config['sample_count']:
                        samples.append(item)
                    else:
                        slot = rng.randrange(matched)
                        if slot < config['sample_count']: samples[slot] = item
            rng.shuffle(samples)
        if len(samples) < config['min_samples']:
            unavailable[name] = f'Only {len(samples)} matching samples; need {config["min_samples"]}'
            continue
        path = root/'inputs'/f'{name}.jsonl'
        path.parent.mkdir(exist_ok=True)
        with path.open('w') as out:
            for row in samples: out.write(json.dumps(row, ensure_ascii=False)+'\n')
        path.chmod(path.stat().st_mode & 0o444)
        prepared[name] = {'samples':samples, 'path':str(path.relative_to(root)),
                         'sha256':digest(path)}
    return prepared, unavailable
