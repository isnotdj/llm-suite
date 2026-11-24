"""Persist the exact prepared API inputs and references for one run."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def filesystem_locations():
    """Map Linux mountpoints to their paths on the underlying filesystem."""
    if not sys.platform.startswith('linux'):
        return []
    mounts = []
    def unescape(raw):
        return Path(os.fsdecode(re.sub(rb'\\([0-7]{3})', lambda m: bytes([int(m[1], 8)]), raw)))
    for line in Path('/proc/self/mountinfo').read_bytes().splitlines():
        _, _, device, base, mountpoint, _ = line.split(b' ', 5)
        mounts.append((unescape(mountpoint), device, unescape(base)))
    return sorted(mounts, key=lambda mount: len(mount[0].parts), reverse=True)


def storage_location(path, mounts):
    for mountpoint, device, base in mounts:
        if path.is_relative_to(mountpoint):
            return device, base/path.relative_to(mountpoint)
    return None, path


def check_output(root, inputs, mounts=None):
    root = root.resolve()
    if mounts is None:
        mounts = filesystem_locations()
    device, output_path = storage_location(root,mounts)
    for item in inputs:
        if not isinstance(item, (str, Path)) or not item:
            continue
        source = Path(item).expanduser().resolve()
        source_device, input_path = storage_location(source,mounts)
        if (root.is_relative_to(source) or source.is_relative_to(root)
                or (device == source_device and
                    (output_path.is_relative_to(input_path) or input_path.is_relative_to(output_path)))):
            raise ValueError(f'Output directory overlaps input: {source}')


def create_output(root, inputs):
    """Claim a new output tree disjoint from the supplied input resources."""
    check_output(root,inputs)
    root.mkdir(parents=True, exist_ok=False)


def require_local_output(path):
    """Reject network filesystems for new runs."""
    path = path.resolve()
    if sys.platform.startswith('linux'):
        entries = []
        for line in Path('/proc/self/mountinfo').read_bytes().splitlines():
            left, right = line.split(b' - ', 1)
            mount = left.split()[4]
            mount = Path(os.fsdecode(re.sub(rb'\\([0-7]{3})', lambda m: bytes([int(m[1], 8)]), mount)))
            if path.is_relative_to(mount):
                entries.append((len(mount.parts), right.split()[0].decode()))
        fs_type = max(entries)[1] if entries else ''
    elif sys.platform == 'darwin':
        import subprocess
        existing = path
        while not existing.exists():
            existing = existing.parent
        try:
            info = subprocess.check_output(['/bin/df', '--libxo', 'json', '-Y', str(existing)])
            volume, = json.loads(info)['storage-system-information']['filesystem']
            fs_type = volume['type']
            if not isinstance(fs_type, str) or not fs_type.strip():
                raise ValueError('missing filesystem type')
        except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Cannot determine output filesystem type for {existing}') from exc
    else:
        return
    if fs_type.lower() in {'nfs','nfs4','cifs','smbfs','smb3','afpfs','sshfs','fuse.sshfs','9p','ceph','glusterfs','fuse.glusterfs'}:
        raise ValueError(f'Output directory must be on local storage: {path} ({fs_type})')


def create_run(output, inputs, *, group_id=None, phase=None, existing_group=False):
    from datetime import datetime
    from uuid import uuid4

    output = output.resolve()
    require_local_output(output)
    if group_id is None:
        check_output(output, inputs)
        output.mkdir(parents=True, exist_ok=True)
        name = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f') + '-' + uuid4().hex[:8]
        root = output/name
    else:
        group = output/str(group_id)
        if group.is_symlink():
            raise ValueError(f'Unexpected symbolic link for grouped run: {group}')
        # Accuracy claims only its own leaf in an already reserved group.
        check_output(group/phase if phase == 'accuracy' else group, inputs)
        if phase == 'perf' and not existing_group:
            group.mkdir(mode=0o700, parents=True, exist_ok=False)
            print(f'Run group: {group}', flush=True)
        elif phase not in ('accuracy', 'perf'):
            raise ValueError(f'Unknown grouped run phase: {phase}')
        root = group/phase
    root.mkdir()
    print(f'Run directory: {root}', flush=True)
    return root


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def freeze_tasks(root, name, tasks):
    for task in tasks:
        folder = root/'inputs'/name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/(task['config']['abbr']+'.jsonl')
        ids, prompts, references = task['ids'], task['prompts'], task['references']
        if not ids or not len(ids) == len(prompts) == len(references):
            raise ValueError('Prepared input count mismatch')
        with path.open('x') as out:
            for sample_id, messages, reference in zip(ids,prompts,references):
                out.write(json.dumps(dict(sample_id=sample_id,messages=messages,reference=reference),ensure_ascii=False)+'\n')
        path.chmod(path.stat().st_mode & 0o444)
        task['input_path'] = str(path)
        task['input_sha256'] = digest(path)
        task['count'] = len(ids)
        for key in ('prompts','references','test'):
            task.pop(key, None)
    return tasks


def check_inputs(task):
    if digest(task['input_path']) != task['input_sha256']:
        raise ValueError('Frozen run input was modified; refusing evaluation')


def iter_inputs(task):
    check_inputs(task)
    count = 0
    with Path(task['input_path']).open() as source:
        for line in source:
            row = json.loads(line)
            if count >= task['count'] or row['sample_id'] != task['ids'][count]:
                raise ValueError('Frozen sample identity/count mismatch')
            count += 1
            yield row
    if count != task['count']:
        raise ValueError('Frozen input is incomplete')
    check_inputs(task)


def restrict_run_readers(root, sources):
    """Gate the entire output tree before materializing prompts or responses.

    A directory dataset can contain privately permissioned shards. Without a
    complete file manifest, its safe common audience is the submitting account.
    """
    import os
    import stat
    import sys

    if not sources:
        return

    def owner_only():
        if sys.platform == 'darwin':
            import subprocess
            subprocess.run(['/bin/chmod','-N',root],check=True)
        root.chmod(0o700)

    restrictions = [(root,1)]
    for source in sources:
        path = Path(source).absolute()
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            owner_only()
            return
        restrictions.append((resolved,4))
        restrictions.extend((parent,1) for parent in set(path.parents) | set(resolved.parents))

    # Each constraint permits either everyone, one group, or only the owner.
    # An explicit ACL may contain named exceptions not visible in mode bits.
    if sys.platform == 'darwin':
        import subprocess
        # Darwin ACL grants can override mode bits. Use the conservative gate
        # if an ancestor or input carries an explicit access-control entry.
        paths = sorted({path for path,_ in restrictions})
        listing = subprocess.check_output(['/bin/ls','-lde',*paths])
        if any(line.lstrip()[:1].isdigit() and b': ' in line for line in listing.splitlines()):
            owner_only()
            return
    groups = set()
    for path,bit in set(restrictions):
        info = path.stat()
        if sys.platform.startswith('linux') and 'system.posix_acl_access' in os.listxattr(path):
            owner_only()
            return
        if not (info.st_mode & (bit << 6)) or not (info.st_mode & (bit << 3)):
            owner_only()
            return
        if not (info.st_mode & bit):
            groups.add(info.st_gid)
    if len(groups) > 1:
        owner_only()
    elif groups:
        group = groups.pop()
        try:
            if root.stat().st_gid != group:
                os.chown(root,-1,group)
        except PermissionError:
            owner_only()
            return
        root.chmod((stat.S_IMODE(root.stat().st_mode) & 0o750) | stat.S_ISGID)
