"""Explicit run lifecycle, independent of dataset state aggregation."""
import errno
import json
import os
import stat
import sys
from uuid import uuid4
from datetime import datetime, timezone


def apply_access(path, source, mode):
    """Apply source identities before mode bits (chown may clear setgid)."""
    original = source.stat()
    current = path.stat()
    if (current.st_uid,current.st_gid) != (original.st_uid,original.st_gid):
        os.chown(path,original.st_uid,original.st_gid)
    if sys.platform.startswith('linux'):
        # chmod changes the ACL mask, but leaves inherited named entries intact.
        keys = ['system.posix_acl_access']
        if stat.S_ISDIR(original.st_mode):
            keys.append('system.posix_acl_default')
        for key in keys:
            try:
                value = os.getxattr(source,key)
            except OSError as exc:
                if exc.errno not in (errno.ENODATA,errno.ENOTSUP):
                    raise
                try:
                    os.removexattr(path,key)
                except OSError as missing:
                    if missing.errno not in (errno.ENODATA,errno.ENOTSUP):
                        raise
            else:
                os.setxattr(path,key,value)
    path.chmod(mode)


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        original = path.stat()
    except FileNotFoundError:
        original = None
    # Keep replacement bytes private until access metadata is restored.
    fd = os.open(temporary,os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                 0o600 if original is not None else 0o666)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as output:
            json.dump(value,output,indent=2,ensure_ascii=False,allow_nan=False)
            output.write('\n')
        if original is not None:
            apply_access(temporary,path,original.st_mode & 0o777)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class RunJournal:
    def __init__(self, root, selected, available):
        self.root = root
        self.last_error = None
        self.selected = list(selected)
        self.data = dict(schema_version=2, overall_status='preparing',
            setup={'stages':list(selected), 'tasks':{}, 'snapshot_complete':False},
            phase='configuration', active_dataset=None, errors=[],
            status={n:('pending' if n in selected else 'skipped') for n in available},metrics=[])
        self.flush()

    def flush(self):
        self.data['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.root/'summary.json',self.data)

    def checkpoint(self, phase, dataset=None, setup=None):
        self.data.update(phase=phase, active_dataset=dataset)
        if setup is not None:
            self.data['setup'] = setup
            atomic_json(self.root/'setup.json', setup)
        self.flush()

    def error(self, exc, detail, *, flush=True):
        if self.last_error is exc: return
        self.last_error = exc
        self.data['errors'].append(dict(phase=self.data['phase'],
            dataset=self.data['active_dataset'], type=type(exc).__name__, detail=detail))
        if flush:
            self.flush()

    def ready(self, setup, validate_only=False):
        self.data['setup'] = setup
        self.data.update(phase='validated' if validate_only else 'execution', active_dataset=None)
        self.data['overall_status'] = 'validated' if validate_only else 'running'
        if validate_only:
            self.data['status'].update({n:'validated' for n in self.selected})
        self.flush()

    def start(self, name):
        if self.data['overall_status'] != 'running' or self.data['status'][name] != 'pending':
            raise ValueError('Invalid dataset start transition')
        self.data.update(phase='execution', active_dataset=name)
        self.data['status'][name] = 'running'
        self.flush()

    def merge_metrics(self, name, rows):
        # Replace this dataset's committed results, so retries/interruption cannot duplicate rows.
        normalized = [r if isinstance(r,dict) else dict(zip(
            ('framework','dataset','metric','value','source'),r)) for r in rows]
        self.data['metrics'] = [r for r in self.data['metrics'] if r.get('run_dataset') != name]
        self.data['metrics'].extend(dict(r,run_dataset=name) for r in normalized)

    def record(self, name, rows, error=None):
        self.merge_metrics(name, rows)
        self.data['status'][name] = 'failed: '+type(error).__name__ if error else 'complete'
        self.data['active_dataset'] = None
        self.flush()

    def finish(self):
        self.data.update(phase='finished', active_dataset=None)
        states = [self.data['status'][n] for n in self.selected]
        self.data['overall_status'] = 'complete' if all(s=='complete' for s in states) else 'failed'
        self.data['status'].update({n:'not_run' for n,s in self.data['status'].items() if s=='pending'})
        self.flush()

    def terminate(self, interrupted=False):
        self.data['overall_status'] = 'interrupted' if interrupted else 'failed'
        self.data['status'] = {n:('interrupted' if interrupted else 'failed') if s=='running'
            else 'not_run' if s=='pending' else s for n,s in self.data['status'].items()}
        self.flush()
