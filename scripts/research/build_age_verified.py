"""Bounded independent repetition gate for timing sidecars; never accept a failed worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--profile',choices=['development2004','full-period'],default='development2004')
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    script=Path(__file__).with_name('prepare_observation_age.py').resolve()
    attempts=[];chosen=[]
    for channel in range(40,48):
        groups={};accepted=None
        for trial in range(1,5):
            dest=out/f'channel_{channel}_attempt_{trial}'
            with (out/f'channel_{channel}_attempt_{trial}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(script),'--channel',str(channel),'--output',str(dest),'--profile',a.profile],stdout=log,stderr=subprocess.STDOUT)
            record=dict(channel=channel,attempt=trial,exit_code=result.returncode)
            if result.returncode==0:
                protocol=json.loads((dest/'protocol.json').read_text())
                if protocol['status']!='complete' or protocol['features']!=[f'channel_{channel}']:
                    raise ValueError('Worker protocol invalid')
                for name,expected in protocol['files_sha256'].items():
                    if digest(dest/name)!=expected:raise ValueError('Worker file changed after completion')
                key=json.dumps(protocol,sort_keys=True)
                groups.setdefault(key,[]).append(dest)
                record['matching_successes']=len(groups[key])
                if len(groups[key])==2:accepted=(protocol,groups[key])
            attempts.append(record)
            (out/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
            print(record,flush=True)
            if accepted:break
        if not accepted:raise RuntimeError(f'channel_{channel}: no matching pair within4 attempts; outputs remain incomplete')
        chosen.append(accepted)
    combined=chosen[0][0].copy();files={};cadences={};features=[];pairs={}
    for protocol,dirs in chosen:
        for key in ['rows','train','calibration','assessment','profile','encoding','source_manifest_sha256','source_fingerprints']:
            if protocol[key]!=combined[key]:raise ValueError('Channels use inconsistent metadata')
        for name,expected in protocol['files_sha256'].items():
            # Both independently computed copies must still match before accepting.
            if any(digest(d/name)!=expected for d in dirs):raise ValueError('Independent copies changed')
            if name in files:
                if files[name]!=expected:raise ValueError('Grid mismatch')
            else:
                shutil.copy2(dirs[0]/name,out/name);files[name]=expected
        cadences.update(protocol['cadence_seconds']);features.extend(protocol['features'])
        pairs[protocol['features'][0]]=[d.name for d in dirs]
    for name,expected in files.items():
        if digest(out/name)!=expected:raise ValueError('Combined file changed')
    combined.update(features=features,cadence_seconds=cadences,files_sha256=files,
        execution='two independent matching workers per channel; maximum4 attempts each; all failures logged',matching_worker_pairs=pairs)
    (out/'protocol.json').write_text(json.dumps(combined,indent=2)+'\n')
    print('COMPLETE: all8 channels independently reproduced twice; no training.',flush=True)

if __name__=='__main__':main()
