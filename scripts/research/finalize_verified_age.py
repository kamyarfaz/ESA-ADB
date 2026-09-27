"""Recover only independently matching completed workers using two hash implementations."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path


def verify(path,expected):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    python_hash=h.hexdigest()
    external=subprocess.run(['sha256sum','--',str(path)],check=True,stdout=subprocess.PIPE,text=True).stdout.split()[0]
    if python_hash!=expected or external!=expected:
        raise ValueError(f'{path}: expected={expected}, streaming={python_hash}, sha256sum={external}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    selected=[];notes=[]
    for channel in range(40,48):
        groups={};c=f'channel_{channel}'
        for directory in sorted(a.source.glob(f'{c}_attempt_*')):
            if not (directory/'protocol.json').exists():continue
            protocol=json.loads((directory/'protocol.json').read_text())
            if protocol.get('status')!='complete' or protocol.get('features')!=[c]:raise ValueError('Invalid completed worker')
            if set(protocol['files_sha256'])!={c+'.npz','timestamps.npy'}:raise ValueError('Unexpected worker files')
            groups.setdefault(json.dumps(protocol,sort_keys=True),[]).append(directory)
        pairs=[(json.loads(key),paths) for key,paths in groups.items() if len(paths)>=2]
        if len(pairs)!=1:raise ValueError(f'{c}: require one unambiguous matching group')
        protocol,paths=pairs[0];paths=paths[:2]
        for path in paths:
            for name,expected in protocol['files_sha256'].items():verify(path/name,expected)
        selected.append((protocol,paths));notes.append({'channel':c,'pair':[str(path.resolve()) for path in paths]})
        print(c,'both independent copies verified by streaming SHA256 and sha256sum',flush=True)
    combined=selected[0][0].copy();files={};features=[];cadences={};pairs={}
    for protocol,paths in selected:
        for key in ['rows','train','calibration','assessment','profile','encoding','source_manifest_sha256','source_fingerprints']:
            if protocol[key]!=combined[key]:raise ValueError(f'Mismatching metadata: {key}')
        for name,expected in protocol['files_sha256'].items():
            if name in files:
                if files[name]!=expected:raise ValueError('Grid differs')
            else:
                shutil.copyfile(paths[0]/name,a.output/name);verify(a.output/name,expected);files[name]=expected
        features.extend(protocol['features']);cadences.update(protocol['cadence_seconds']);pairs[protocol['features'][0]]=[str(path.resolve()) for path in paths]
    for name,expected in files.items():verify(a.output/name,expected)
    combined.update(features=features,cadence_seconds=cadences,files_sha256=files,matching_worker_pairs=pairs,
        execution='Recovered matching completed worker pairs; both files independently checked with streaming hashlib and external sha256sum; combined files rechecked with both. No failed worker or mismatching file accepted.')
    recovery=a.source/'recovery.json'
    if recovery.exists():
        combined['grid_recovery']=json.loads(recovery.read_text())
        combined['execution'] += ' Timestamp copies were regenerated from the deterministic 30-second grid and matched the original recorded SHA256; feature arrays retain independent worker provenance. Original damaged copies are preserved.'
    (a.output/'verification.json').write_text(json.dumps(notes,indent=2)+'\n')
    (a.output/'protocol.json').write_text(json.dumps(combined,indent=2)+'\n')
    print('COMPLETE: verified full dataset assembled; original attempts preserved.',flush=True)

if __name__=='__main__':main()
