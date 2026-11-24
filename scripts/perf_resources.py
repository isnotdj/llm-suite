"""Bounded-memory resource fingerprints; never scan model weights."""
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def artifact_signature(path):
    return dict(filename=path.name, size=path.stat().st_size, sha256=digest(path))


def configured_tokenizer_files(root):
    """Resolve resource overrides passed to the local Transformers tokenizer."""
    config = root/'tokenizer_config.json'
    if not config.is_file():
        return set()
    options = json.loads(config.read_text())
    files = set()
    candidates = options.get('fast_tokenizer_files')
    if candidates:
        from transformers.tokenization_utils_base import get_fast_tokenizer_file
        selected = get_fast_tokenizer_file(candidates)
        if selected != 'tokenizer.json' or (root/selected).is_file():
            files.add(root/selected)
    # Transformers passes explicit tokenizer_file / vocab_file / merges_file
    # kwargs to constructors verbatim, including paths relative to the cwd.
    for key,value in options.items():
        if key.endswith('_file') and isinstance(value,str) and value:
            files.add(Path(value).absolute())
    return files


def resource_key(root, path):
    # Resolve directory aliases, keeping the leaf name used by older snapshots.
    path = path.parent.resolve()/path.name
    directory = root.resolve()
    return str(path.relative_to(directory) if path.is_relative_to(directory) else path)


def tokenizer_hashes(directory, tokenizer):
    root = Path(directory)
    names = {'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
             'added_tokens.json','vocab.json','vocab.txt','merges.txt','tokenizer.model',
             'spiece.model','sentencepiece.bpe.model','chat_template.jinja','config.json'}
    names.update(Path(v).name for v in getattr(tokenizer,'vocab_files_names',{}).values() if isinstance(v,str))
    files = {root/name for name in names if (root/name).is_file()}
    files.update(path for path in configured_tokenizer_files(root) if path.is_file())
    templates = root/'chat_templates'
    if templates.is_dir(): files.update(templates.glob('*.jinja'))
    return {resource_key(root,p):digest(p) for p in sorted(files)}


def freeze_dataset(source, root):
    """Copy one stable file and hash the same stream used for every profile."""
    from os import fstat
    from tempfile import NamedTemporaryFile

    folder = root/'inputs'/'source'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/('dataset'+source.suffix)
    hashed = hashlib.sha256()
    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    with source.open('rb') as incoming, NamedTemporaryFile(dir=folder, delete=False) as snapshot:
        temporary = Path(snapshot.name)
        try:
            before = identity(fstat(incoming.fileno()))
            for chunk in iter(lambda: incoming.read(1024*1024), b''):
                hashed.update(chunk)
                snapshot.write(chunk)
            if before != identity(fstat(incoming.fileno())) or before != identity(source.stat()):
                raise ValueError(f'Dataset changed during snapshot: {source}')
            snapshot.flush()
            temporary.chmod(0o400)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination, hashed.hexdigest()
