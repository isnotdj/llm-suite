"""Prepare stable local resources before the first staged benchmark."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from perf_resources import digest
from run_inputs import check_output, restrict_run_readers


def manifest(source, tokenizer):
    files = {}
    if source.is_file():
        return {'.': digest(source)}
    for folder, dirs, names in os.walk(source, onerror=raise_error):
        for name in dirs:
            if (Path(folder)/name).is_symlink():
                raise ValueError(f'Cannot snapshot linked directory: {folder}/{name}')
        for name in names:
            path = Path(folder)/name
            if tokenizer and (path.suffix in {'.safetensors', '.gguf', '.pth', '.pt'} or
                              name.startswith('pytorch_model') and name.endswith('.bin')):
                continue
            if not path.is_file():
                raise ValueError(f'Expected a regular input file: {path}')
            files[str(path.relative_to(source))] = digest(path)
    if not files:
        raise ValueError(f'Empty input directory: {source}')
    return files


def raise_error(error):
    raise error


def validate_stage_input_options(name, opts):
    dataset = opts['dataset_name']
    if opts.get('trust_remote_code'):
        raise ValueError(f'{name}: custom tokenizer code requires single-run mode')
    if dataset not in ('random', 'random-mm', 'prefix_repetition') and not opts.get('dataset_path'):
        raise ValueError(f'{name}: {dataset} requires local dataset_path')
    if dataset == 'hf':
        # These official adapters only consume embedded text/numeric fields.
        # Other HF adapters may open arbitrary media paths from individual rows.
        from vllm.benchmarks import datasets
        text_adapters = (datasets.InstructCoderDataset, datasets.MTBenchDataset,
                         datasets.AIMODataset, datasets.NextEditPredictionDataset,
                         datasets.MLPerfDataset)
        if not any(opts.get('hf_name') in adapter.SUPPORTED_DATASET_PATHS for adapter in text_adapters):
            raise ValueError(f'{name}: use a local text HF dataset with a supported hf_name; external media requires single-run mode')


def freeze_inputs(root, profiles, snapshots):
    dest = root/'inputs'
    dest.mkdir(mode=0o700)
    cached = {}
    resources = []
    bindings = {}
    roots = [Path(p).resolve(strict=True) for p in snapshots]

    # Resolve each configured name once, before the first copy. A NAS alias
    # can rotate between profiles even when both specify the same path.
    resolved = {}
    for opts in profiles.values():
        for value in (opts.get('tokenizer') or opts['model'], opts.get('dataset_path')):
            if value:
                name = Path(value).expanduser().absolute()
                if name not in resolved:
                    resolved[name] = name.resolve(strict=True)

    check_output(root, resolved.values())

    def pin(value, tokenizer=False):
        source = resolved[Path(value).expanduser().absolute()]
        key = (source, tokenizer)
        if key in cached:
            return cached[key]
        restrict_run_readers(root, [source])
        direct = any(source.is_relative_to(p) for p in roots)
        if direct:
            # The storage provider owns version stability. A link outside the
            # declared snapshot is not covered by that guarantee.
            if source.is_dir():
                for folder, dirs, names in os.walk(source, onerror=raise_error):
                    for name in dirs + names:
                        path = Path(folder)/name
                        if path.is_symlink() and (path.is_dir() or not any(
                                path.resolve(strict=True).is_relative_to(p) for p in roots)):
                            raise ValueError(f'Unpinned snapshot link: {path}')
            location = source
            info = dict(source=str(source), storage='immutable_snapshot', path=str(source))
        else:
            before = manifest(source, tokenizer)
            parent = dest/str(len(resources))
            with TemporaryDirectory(dir=dest) as temp:
                target = Path(temp)/'resource'/source.name
                for rel in before:
                    src = source if rel == '.' else source/rel
                    out = target if rel == '.' else target/rel
                    out.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copyfile(src, out)
                    out.chmod(0o400)
                if manifest(target, tokenizer) != before or manifest(source, tokenizer) != before:
                    raise ValueError(f'Input changed during preparation: {source}')
                (Path(temp)/'resource').rename(parent)
            location = parent/source.name
            fingerprint = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
            info = dict(source=str(source), storage='copy', path=str(location.relative_to(root)),
                        manifest_sha256=fingerprint)
        resources.append(info)
        cached[key] = location
        return location

    for name, opts in profiles.items():
        tokenizer = pin(opts.get('tokenizer') or opts['model'], True)
        if not tokenizer.is_dir():
            raise ValueError(f'{name}: expected a local tokenizer directory')
        config = tokenizer/'tokenizer_config.json'
        if config.is_file():
            settings = json.loads(config.read_text())
            if any(k.endswith('_file') and isinstance(v, str) and v for k, v in settings.items()):
                raise ValueError(f'{name}: tokenizer config has an external file override')
            for entry in settings.get('fast_tokenizer_files', []) or []:
                if Path(entry).name != entry:
                    raise ValueError(f'{name}: external fast tokenizer path')
        opts['tokenizer'] = str(tokenizer)
        bindings[name] = {'tokenizer': str(tokenizer.relative_to(root) if tokenizer.is_relative_to(root) else tokenizer)}
        dataset = opts['dataset_name']
        if opts.get('dataset_path'):
            path = pin(opts['dataset_path'])
            if dataset == 'sharegpt':
                rows = json.loads(path.read_text())
                for row in rows:
                    for key in ('image', 'video'):
                        ref = row.get(key)
                        if isinstance(ref, str) and ref:
                            media = Path(ref.removeprefix('file://'))
                            if (not media.is_absolute() or not any(media.is_relative_to(p) and
                                    media.resolve(strict=True).is_relative_to(p) for p in roots)):
                                raise ValueError(f'{name}: external {key} requires an absolute immutable snapshot path')
            opts['dataset_path'] = str(path)
            bindings[name]['dataset_path'] = str(path.relative_to(root) if path.is_relative_to(root) else path)
    return dict(resources=resources, profiles=bindings,
                resolved_paths={str(name): str(source) for name, source in resolved.items()})
