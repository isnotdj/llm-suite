"""Small, guarded compatibility patches against the pinned upstream commits."""
from pathlib import Path
import sys


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f'Upstream changed: expected one occurrence in {path}: {old}')
    path.write_text(text.replace(old, new))


root = Path(sys.argv[1])
# Preserve upstream verbatim. Only the separate compatibility copy is changed.
import shutil
compat = root / 'FollowBench-api'
shutil.copytree(root / 'FollowBench', compat, ignore=shutil.ignore_patterns('.git'))
for directory, old_model in [('code', 'gpt-4o-mini'), ('code_zh', 'gpt-4')]:
    replace(compat / directory / 'llm_eval.py',
            f"model='{old_model}',", 'model=os.environ["JUDGE_MODEL"],')
