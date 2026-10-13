"""Create and verify a project-file snapshot without modifying its inputs.

Source Git and transient/runtime files are excluded explicitly. Large files are
stored in split Release assets, never in source Git. Requires the zstd command
for archive operations; manifest/destination verification use the standard library.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.pytest_cache', '.mypy_cache'}
MANIFEST_MEMBER = 'PROJECT_SNAPSHOT_MANIFEST.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe snapshot path')
    return root.joinpath(*p.parts)


def inventory(root):
    root = Path(root).resolve()
    entries, excluded, regular = [], [], []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs.sort(); files.sort()
        for name in list(dirs):
            p = Path(base)/name
            rel = p.relative_to(root).as_posix()
            if name in SKIP_DIRS:
                excluded.append({'path':rel, 'reason':'Git metadata or installed runtime/cache'})
                dirs.remove(name)
            elif p.is_symlink():
                entries.append({'path':rel, 'type':'symlink', 'target':os.readlink(p)})
                dirs.remove(name)
            else:
                entries.append({'path':rel, 'type':'directory', 'mode':p.stat().st_mode & 0o777})
        for name in files:
            p = Path(base)/name
            rel = p.relative_to(root).as_posix()
            if name.startswith('.~lock.') or name.endswith(('.pyc','.pyo')):
                excluded.append({'path':rel, 'reason':'Transient lock or Python bytecode'})
            elif p.is_symlink():
                entries.append({'path':rel, 'type':'symlink', 'target':os.readlink(p)})
            elif p.is_file():
                regular.append((p, rel))
            else:
                raise ValueError(f'Unsupported special file: {rel}')

    def record(item):
        p, rel = item
        before = p.stat()
        digest = sha(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns)!=(after.st_size, after.st_mtime_ns):
            raise ValueError(f'File changed during fingerprinting: {rel}')
        return dict(path=rel, type='file', size=before.st_size, sha256=digest,
                    mtime_ns=before.st_mtime_ns, mode=before.st_mode & 0o777)

    total = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(record, regular), 1):
            entries.append(row); total += row['size']
            if i%250==0:
                print(f'fingerprinted {i}/{len(regular)} files; {total/2**30:.2f} GiB', flush=True)
    revision = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'], text=True, capture_output=True)
    return dict(format_version=1, purpose='Current project-file snapshot; not proof of exact historical migration',
                git_head=revision.stdout.strip() if revision.returncode==0 else None,
                files=sorted(entries, key=lambda x:x['path']), excluded=excluded, bytes=total,
                historical_exception='The preserved pa_center_ch08_40_47/mlp_test_score_per_channel.npy '
                    'does not match its original migration checksum. This snapshot preserves current bytes only.')


class HashReader:
    def __init__(self, stream):
        self.stream, self.hash = stream, hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        return data


def pack(root, manifest_path, archive):
    root, manifest_path, archive = Path(root), Path(manifest_path), Path(archive)
    if archive.exists():
        raise FileExistsError(archive)
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    with archive.open('xb') as output:
        process = subprocess.Popen(['zstd','-T8','-3','-q','-c'], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT,
                              bufsize=1024*1024, copybufsize=1024*1024) as tar:
                info = tarfile.TarInfo(MANIFEST_MEMBER); info.size=len(payload)
                tar.addfile(info, io.BytesIO(payload))
                total = 0
                for i, row in enumerate(manifest['files'], 1):
                    p = checked_path(root, row['path'])
                    tar.inodes.clear()  # Record every regular path, even on hardlinked copies.
                    info = tar.gettarinfo(str(p), arcname=row['path'])
                    if row['type']=='file':
                        stat = p.stat()
                        if (not info.isfile() or stat.st_size!=row['size'] or
                                stat.st_mtime_ns!=row['mtime_ns']):
                            raise ValueError(f'File changed before packing: {p}')
                        with p.open('rb') as stream:
                            reader = HashReader(stream)
                            tar.addfile(info, reader)
                            if reader.hash.hexdigest()!=row['sha256']:
                                raise ValueError(f'Hash changed while packing: {p}')
                        if p.stat().st_mtime_ns != stat.st_mtime_ns:
                            raise ValueError(f'File changed while packing: {p}')
                        total += row['size']
                    elif row['type']=='symlink':
                        if not info.issym() or info.linkname!=row['target']:
                            raise ValueError(f'Link changed: {p}')
                        tar.addfile(info)
                    elif row['type']=='directory':
                        if not info.isdir():
                            raise ValueError(f'Directory changed: {p}')
                        tar.addfile(info)
                    if i%250==0:
                        print(f'packed {i}/{len(manifest["files"])} entries; {total/2**30:.2f} GiB', flush=True)
            process.stdin.close()
            if process.wait()!=0:
                raise RuntimeError('zstd compression failed')
        except BaseException:
            process.kill(); process.wait()
            raise
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass  # Preserve the original integrity error after terminating zstd.


def verify_archive(archive, manifest_path):
    expected = json.loads(Path(manifest_path).read_text())
    rows = {r['path']:r for r in expected['files']}
    seen = set(); total=0
    proc = subprocess.Popen(['zstd','-d','-q','-c',str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            first = next(iter(tar))
            if first.name!=MANIFEST_MEMBER or json.load(tar.extractfile(first))!=expected:
                raise ValueError('Embedded manifest mismatch')
            for member in tar:
                if member.name==MANIFEST_MEMBER:
                    continue
                row=rows.get(member.name)
                if row is None or member.name in seen:
                    raise ValueError('Unknown or repeated archive member')
                seen.add(member.name)
                if row['type']=='file':
                    if not member.isfile() or member.size!=row['size']:
                        raise ValueError('File type or size mismatch')
                    h=hashlib.sha256()
                    stream=tar.extractfile(member)
                    for block in iter(lambda:stream.read(8*1024*1024), b''):
                        h.update(block)
                    if h.hexdigest()!=row['sha256']:
                        raise ValueError(f'Archive hash mismatch: {member.name}')
                    total+=member.size
                elif row['type']=='symlink' and (not member.issym() or member.linkname!=row['target']):
                    raise ValueError('Archive symlink mismatch')
                elif row['type']=='directory' and not member.isdir():
                    raise ValueError('Archive directory mismatch')
                if len(seen)%250==0:
                    print(f'verified {len(seen)}/{len(rows)} entries; {total/2**30:.2f} GiB', flush=True)
        # Drain trailing compressed padding before waiting for the decompressor.
        for _ in iter(lambda:proc.stdout.read(8*1024*1024), b''):
            pass
        if proc.wait()!=0 or seen!=set(rows):
            raise ValueError('Incomplete archive')
    except BaseException:
        proc.kill();proc.wait();raise
    finally:
        proc.stdout.close()
    return dict(entries=len(seen), bytes=total, archive_sha256=sha(archive),
                manifest_sha256=sha(manifest_path), verified=True)


def verify_destination(root, manifest_path):
    root = Path(root)
    rows = json.loads(Path(manifest_path).read_text())['files']
    for row in rows:
        p=checked_path(root,row['path'])
        if row['type']=='file':
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:
                raise ValueError(f'Destination file mismatch: {p}')
        elif row['type']=='symlink':
            if not p.is_symlink() or os.readlink(p)!=row['target']:
                raise ValueError(f'Destination link mismatch: {p}')
        elif p.is_symlink() or not p.is_dir():
            raise ValueError(f'Destination directory mismatch: {p}')
    print(f'Verified {len(rows)} destination entries',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['create','pack','verify-archive','verify-destination'])
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args()
    if args.mode=='create':
        if args.manifest.exists():raise FileExistsError(args.manifest)
        payload=inventory(args.root)
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        args.manifest.write_text(json.dumps(payload,indent=2)+'\n')
        print(f'Created {len(payload["files"])} entries; {payload["bytes"]/2**30:.2f} GiB',flush=True)
    elif args.mode=='pack':pack(args.root,args.manifest,args.archive)
    elif args.mode=='verify-archive':print(json.dumps(verify_archive(args.archive,args.manifest)),flush=True)
    else:verify_destination(args.root,args.manifest)


if __name__=='__main__':main()
