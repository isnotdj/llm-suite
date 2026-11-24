"""Stream completed API records to disk, then publish them in input order."""
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Event, local

_worker = local()


def check_cancelled():
    stop = getattr(_worker, 'stop', None)
    if stop is not None and stop.is_set():
        raise InterruptedError('Evaluation interrupted')


def _call(function, item, stop):
    _worker.stop = stop
    check_cancelled()
    return function(item)


def _record_failure(path, phase, ordinal, error):
    error.add_note(f'{path}: {phase} ordinal={ordinal}')
    path.with_suffix('.failure.json').write_text(json.dumps(
        dict(phase=phase, ordinal=ordinal, error_type=type(error).__name__)))


def write_records(path, function, items, concurrency, expected):
    partial = path.with_suffix('.partial.jsonl')
    temporary = path.with_suffix('.ordered.tmp')
    offsets = {}
    stop = Event()
    pool = ThreadPoolExecutor(max_workers=concurrency)
    pending = {}
    count = 0
    source = iter(items)
    exhausted = False
    failure = None
    interrupted = False
    try:
        with partial.open('xb+') as output:
            while True:
                while not exhausted and failure is None and len(pending) < concurrency:
                    try:
                        item = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    except Exception as exc:
                        failure = exc
                        _record_failure(path, 'input', count, exc)
                        break
                    pending[pool.submit(_call, function, item, stop)] = count
                    count += 1
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    ordinal = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        if failure is None:
                            failure = exc
                            _record_failure(path, 'request', ordinal, exc)
                        continue
                    offsets[ordinal] = output.tell()
                    line = json.dumps(dict(ordinal=ordinal, record=record), ensure_ascii=False)
                    output.write((line + '\n').encode())
                    output.flush()
            if failure is not None:
                # An input/transport OSError is local to this dataset. OSErrors
                # from artifact writes must reach the run-level error handler.
                if isinstance(failure, OSError):
                    raise RuntimeError('Dataset input or request failed') from failure
                raise failure
            if len(offsets) != expected or count != expected:
                raise ValueError(f'Incomplete record coverage for {path}')
            with temporary.open('xb') as ordered:
                for ordinal in range(expected):
                    output.seek(offsets[ordinal])
                    entry = json.loads(output.readline())
                    if entry['ordinal'] != ordinal:
                        raise ValueError(f'Invalid record order for {path}')
                    ordered.write((json.dumps(entry['record'], ensure_ascii=False) + '\n').encode())
            temporary.replace(path)
        partial.unlink()
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        stop.set()
        pool.shutdown(wait=not interrupted, cancel_futures=True)
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            if not interrupted:
                raise
            print(f'Could not clean up {temporary}: {exc}', file=sys.stderr, flush=True)


def records(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)
