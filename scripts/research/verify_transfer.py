"""Create or verify a content manifest for a complete project transfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['create', 'verify'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    root = args.root.resolve()
    manifest = root/'migration/manifest.json'
    if args.mode == 'create':
        rows = []
        total = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in sorted(files + [d for d in dirs if (Path(directory)/d).is_symlink()]):
                path = Path(directory)/name
                relative = path.relative_to(root).as_posix()
                if relative in ('migration/manifest.json', 'migration/manifest.json.tmp'):
                    continue
                if path.is_symlink():
                    rows.append({'path': relative, 'link': os.readlink(path)})
                    continue
                before = path.stat()
                checksum = digest(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f'File changed during hashing: {relative}')
                rows.append({'path': relative, 'size': before.st_size, 'sha256': checksum})
                total += before.st_size
                if len(rows) % 500 == 0:
                    print(f'{len(rows)} files, {total/2**30:.1f} GiB hashed', flush=True)
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'files': rows, 'bytes': total}, indent=2))
        temporary.replace(manifest)
        print(f'CREATED: {len(rows)} entries, {total/2**30:.2f} GiB', flush=True)
    else:
        data = json.loads(manifest.read_text())
        failures = []
        for row in data['files']:
            path = root/row['path']
            if 'link' in row:
                ok = path.is_symlink() and os.readlink(path) == row['link']
            else:
                ok = (path.is_file() and not path.is_symlink()
                      and path.stat().st_size == row['size'] and digest(path) == row['sha256'])
            if not ok:
                failures.append(row['path'])
        for name in failures:
            print('FAILED:', name)
        print(f'{len(data["files"])} entries checked; {len(failures)} failures')
        raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
