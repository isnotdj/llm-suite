"""Pure validation rules shared by preparation and scoring."""
import math

CATEGORIES = ('content', 'situation', 'style', 'format', 'mixed', 'example')


def follow_groups(category, samples):
    if category not in CATEGORIES or not isinstance(samples, list) or not samples:
        raise ValueError(f'{category}: empty/invalid FollowBench data')
    groups = []
    seen = set()
    prompts = set()
    for row in samples:
        if not isinstance(row, dict): raise ValueError('FollowBench rows must be objects')
        group = row.get('example_id')
        if isinstance(group, bool) or not isinstance(group, (int, str)):
            raise ValueError('FollowBench requires stable example_id values')
        level = row.get('level')
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError('FollowBench level must be an integer')
        if not isinstance(row.get('instruction'), str) or (level > 0 and not row['instruction'].strip()):
            raise ValueError('FollowBench instruction must be nonempty')
        if (category != 'mixed' and row.get('category') != category) or (
                category == 'mixed' and (not isinstance(row.get('category'), str) or
                any(part.strip() not in CATEGORIES for part in row['category'].split(',')))):
            raise ValueError('FollowBench category mismatch')
        if level > 0:
            if row['instruction'] in prompts:
                raise ValueError('FollowBench duplicate prompt would make official output matching ambiguous')
            prompts.add(row['instruction'])
        if not groups or groups[-1][0]['example_id'] != group:
            if group in seen:
                raise ValueError(f'{category}/{group}: noncontiguous group')
            seen.add(group)
            groups.append([])
        groups[-1].append(row)
    expected = list(range(1, 6)) if category == 'example' else list(range(6))
    for rows in groups:
        if [r['level'] for r in rows] != expected:
            raise ValueError(f"{category}/{rows[0]['example_id']}: expected ordered levels {expected}")
        for row in rows:
            if not isinstance(row.get('source'), str):
                raise ValueError('FollowBench source is missing')
            if not isinstance(row.get('target'), str):
                raise ValueError('FollowBench example target is missing')
    return groups


def validate_metrics(name, result):
    if not isinstance(result, dict) or 'error' in result:
        raise ValueError(f'{name}: evaluator reported an error')
    metrics = {k:float(v) for k,v in result.items()
               if isinstance(v,(int,float)) and not isinstance(v,bool)}
    if not metrics:
        raise ValueError(f'{name}: no numeric metrics')
    for key, value in metrics.items():
        lower, upper = (1,10) if name == 'finsafetybench' and key.endswith('/mean_judge_score') else (0,100)
        if name == 'followbench' and key.endswith('/CSL'):
            lower, upper = 0,5
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'{name}/{key}: score {value} outside [{lower}, {upper}]')
    return metrics


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def hallu_records(samples):
    """Validate the exact reference branch used by the official get_prompt."""
    if not isinstance(samples, list) or not samples:
        raise ValueError('HalluQA requires nonempty records')
    ids = set()
    for row in samples:
        if not isinstance(row, dict): raise ValueError('HalluQA rows must be objects')
        identifier = row.get('question_id')
        if isinstance(identifier, bool) or not isinstance(identifier, (int, str)) or identifier == '' or identifier in ids:
            raise ValueError('HalluQA IDs must be nonempty and unique')
        ids.add(identifier)
        if not nonempty_text(row.get('Question')):
            raise ValueError(f'HalluQA/{identifier}: missing question')
        keys = [f'Best Answer{i}' for i in range(1,5) if f'Best Answer{i}' in row] if 'Best Answer1' in row else ['Best Answer']
        answers = [row.get(key) for key in keys]
        if any(not isinstance(a, str) for a in answers) or not any(nonempty_text(a) for a in answers):
            raise ValueError(f'HalluQA/{identifier}: missing or invalid reference answers')
    return samples


def fin_rating(raw):
    """Accept exactly one final Rating line, never an example elsewhere."""
    import re
    if not isinstance(raw, str):
        raise ValueError('Invalid FinSafetyBench judge output')
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.S|re.I).strip()
    if re.search(r'</?think\b', cleaned, flags=re.I):
        raise ValueError('Unclosed judge reasoning block')
    ratings = re.findall(r'(?im)^\s*Rating\s*:', cleaned)
    final = re.search(r'(?i)(?:^|\n)\s*Rating:\s*\[\[(10|[1-9])\]\]\s*$', cleaned)
    if len(ratings) != 1 or final is None:
        raise ValueError('Expected one final Rating: [[1..10]] line')
    return int(final.group(1))


def reference_answer(name, value):
    if name in ('ceval','financeiq','logiqa-zh'):
        valid = value in ('A','B','C','D')
    elif name == 'cmrc2018':
        valid = isinstance(value,list) and bool(value) and all(nonempty_text(a) for a in value)
    else:
        valid = value is not None and value != '' and value != []
    if not valid:
        raise ValueError(f'{name}: missing or invalid reference answer')
