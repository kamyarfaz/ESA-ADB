"""Create the additive October research backup (datasets intentionally excluded)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve existing archive')
 paths=set()
 for parent,patterns in [('results_longrun/development',['inner_*','short_*','score_change_v1','telemetry_reference_v1','grid_cap_policy_v1']),('results_longrun/diagnostics',['inner_*','loss_weight*','channel_level*','short_no_roll*','grid_cap_comparison*'])]:
  for pattern in patterns:
   for root in Path(parent).glob(pattern):
    for f in ([root] if root.is_file() else root.rglob('*')):
     if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts:paths.add(f)
 # Preserve artifact build sources without external runtime links/caches.
 for root in [Path('outputs/federico-meeting-20260929'),Path('outputs/federico-meeting-20260930'),Path('outputs/short-no-roll-20261002')]:
  for f in root.rglob('*'):
   if f.is_file() and not f.is_symlink() and 'node_modules' not in f.parts and '__pycache__' not in f.parts:paths.add(f)
 entries=[dict(path=str(f),size=f.stat().st_size,sha256=sha(f)) for f in sorted(paths)]
 manifest=dict(purpose='Additive October research backup; restore alongside earlier releases; not a whole project mirror',files=entries)
 payload=json.dumps(manifest,indent=2).encode()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(a.output,'w:gz',compresslevel=1) as tar:
  info=tarfile.TarInfo('BACKUP_MANIFEST.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
  for row in entries:tar.add(row['path'],arcname=row['path'],recursive=False)
 with tarfile.open(a.output,'r:gz') as tar:
  for row in entries:
   h=hashlib.sha256();stream=tar.extractfile(row['path'])
   for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
   assert h.hexdigest()==row['sha256'],row['path']
   assert sha(Path(row['path']))==row['sha256'],row['path']
 checksum=sha(a.output)
 a.output.with_suffix(a.output.suffix+'.sha256').write_text(f'{checksum}  {a.output.name}\n')
 a.output.with_suffix(a.output.suffix+'.manifest.json').write_bytes(payload)
 print(json.dumps(dict(files=len(entries),unpacked_bytes=sum(x['size'] for x in entries),archive_bytes=a.output.stat().st_size,sha256=checksum)),flush=True)

if __name__=='__main__':main()
