"""Resolve logical benchmark names without changing NAS files or directories."""
import json
import os
from pathlib import Path
import shutil


def read_mapping(filename=None):
    filename = filename or os.environ.get('DATASET_MAP')
    if not filename:
        return {}
    config = Path(filename).resolve(strict=True)
    value = json.loads(config.read_text())
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('Dataset mapping requires version: 1')
    root = Path(value.get('root', '.'))
    if not root.is_absolute():
        root = config.parent / root
    # Only compose paths here. An unused NAS root need not exist; resolve()
    # checks and pins the final path for each selected dataset at launch.
    entries = value.get('datasets')
    if not isinstance(entries, dict):
        raise ValueError('datasets must be an object of logical name -> path')
    paths = {}
    for name, path in entries.items():
        if not isinstance(name, str) or not isinstance(path, str) or not path:
            raise ValueError('Dataset names and paths must be nonempty strings')
        target = Path(path)
        paths[name] = target if target.is_absolute() else root / target
    return paths


def resolve(name, mappings, fallback=None):
    path = mappings.get(name, fallback)
    if path is None:
        raise ValueError(f'No path mapping for {name}')
    # Resolve at launch so a subsequent NAS current-pointer update cannot change this run.
    return str(Path(path).resolve(strict=True))


def working_copy(source, name):
    cache = Path(os.environ.get('DATASET_WORK_CACHE', '/cache/dataset-work'))
    from run_inputs import check_output
    source = Path(source).resolve(strict=True)
    if not name or Path(name).name != name:
        raise ValueError('Dataset work copy requires a single name')
    # Use the eventual child, so siblings under a common cache root are valid.
    from uuid import uuid4
    run = cache/(name+'-'+uuid4().hex)
    check_output(run,[source])
    run.mkdir(parents=True, mode=0o700)
    destination = run / 'data'
    # Dereference links and copy bytes: never use hard links to the immutable NAS source.
    shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copyfile)
    for directory in [destination, *[p for p in destination.rglob('*') if p.is_dir()]]:
        directory.chmod(directory.stat().st_mode | 0o700)
    return str(destination)
