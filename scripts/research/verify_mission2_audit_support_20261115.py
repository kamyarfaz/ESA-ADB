"""Verify fresh-download metadata/test/log supplement without extracting files."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/k_faz/project-backups/mission2-study-20261114')
local=BASE/'release-20261115';download=BASE/'fresh-download-20261115'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(local/'audit-support-manifest.json')==sha(download/'audit-support-manifest.json')
r=json.loads((download/'audit-support-manifest.json').read_text());p=download/r['asset']
assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
expected={x['name']:x for x in r['members']}
with tarfile.open(p) as tar:
 assert len(tar.getmembers())==len(expected)
 for member in tar.getmembers():
  assert member.isfile() and member.name in expected
  record=expected[member.name];assert member.size==record['bytes']
  assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==record['sha256']
receipt=dict(verified=True,asset=r['asset'],sha256=r['sha256'],members=len(expected),manifest_sha256=sha(download/'audit-support-manifest.json'),fresh_download_directory=str(download))
(ROOT/'docs/research/tables/mission2-three-technique-2026-11-14/audit-support-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
