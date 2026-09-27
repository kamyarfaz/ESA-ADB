import io,zipfile,hashlib,json,sys,time
from pathlib import Path
rows=[]
for channel in (43,44):
 p=Path('data/ESA-Mission1/channels/channel_%s.zip'%channel)
 for trial in range(6):
  row={'channel':channel,'trial':trial,'python':sys.executable}
  try:
   data=p.read_bytes();row['compressed_sha']=hashlib.sha256(data).hexdigest()
   with zipfile.ZipFile(io.BytesIO(data)) as z:
    raw=z.read(z.namelist()[0]);row['decoded_sha']=hashlib.sha256(raw).hexdigest()
   row['compressed_sha_after']=hashlib.sha256(data).hexdigest()
  except Exception as e:row['error']=repr(e)
  rows.append(row);print(json.dumps(row),flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+'\n')
