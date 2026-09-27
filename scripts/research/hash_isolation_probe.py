import hashlib,io,json,pickle,subprocess,sys,zipfile
from pathlib import Path
rows=[]
p=Path('data/ESA-Mission1/channels/channel_42.zip')
blob=p.read_bytes()
for stage,data in [('synthetic',b'ESA-integrity-check-0123456789'*1000000),('compressed',blob)]:
 hs=[hashlib.sha256(data).hexdigest() for _ in range(8)]
 external=subprocess.run(['sha256sum'],input=data,stdout=subprocess.PIPE,check=True).stdout.decode().split()[0]
 rows.append(dict(stage=stage,hashes=hs,external=external));print(stage,len(set(hs)),external in hs,flush=True)
try:
 with zipfile.ZipFile(io.BytesIO(blob)) as z:decoded=z.read('channel_42')
 for trial in range(4):
  before=[hashlib.sha256(decoded).hexdigest() for _ in range(3)]
  external_before=subprocess.run(['sha256sum'],input=decoded,stdout=subprocess.PIPE,check=True).stdout.decode().split()[0]
  frame=pickle.loads(decoded)
  after=[hashlib.sha256(decoded).hexdigest() for _ in range(3)]
  external_after=subprocess.run(['sha256sum'],input=decoded,stdout=subprocess.PIPE,check=True).stdout.decode().split()[0]
  row=dict(stage='decoded',trial=trial,before=before,after=after,external_before=external_before,external_after=external_after)
  rows.append(row);print(json.dumps(row),flush=True)
except Exception as e:rows.append(dict(error=repr(e)));print(repr(e),flush=True)
Path('outputs/integrity-review-20260925/hash_isolation.json').write_text(json.dumps(rows,indent=2)+'\n')
