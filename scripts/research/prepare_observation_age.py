"""Build verified timing-only sidecars for the fixed2004 training/calibration fold."""
import argparse
import hashlib
import io
import json
import zipfile
import subprocess
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.observation_age import observation_age


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--profile',choices=['development2004','full-period'],default='development2004')
    parser.add_argument('--decoder',choices=['python','unzip'],default='python')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--channel',type=int,choices=range(40,48))
    mode.add_argument('--isolate-channels',action='store_true')
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    if args.isolate_channels:
        combined=None;files={};cadences={};features=[]
        for j in range(40,48):
            child=out/f'worker_{j}'
            command=[sys.executable,str(Path(__file__).resolve()),'--output',str(child),'--channel',str(j),'--decoder',args.decoder,'--profile',args.profile]
            with (out/f'worker_{j}.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            protocol=json.loads((child/'protocol.json').read_text())
            if protocol['status']!='complete' or protocol['features']!=[f'channel_{j}']:
                raise ValueError('Incomplete or wrong worker output')
            if combined is None:combined=protocol.copy()
            for key in ['rows','train','calibration','assessment','profile','encoding','source_manifest_sha256','source_fingerprints']:
                if protocol[key]!=combined[key]:raise ValueError(f'Worker mismatch: {key}')
            for name,digest in protocol['files_sha256'].items():
                if sha((child/name).read_bytes())!=digest:raise ValueError('Worker output digest mismatch')
                if name in files:
                    if files[name]!=digest:raise ValueError('Timestamp grids differ')
                else:
                    shutil.copy2(child/name,out/name);files[name]=digest
            cadences.update(protocol['cadence_seconds']);features.extend(protocol['features'])
            print(f'Fresh process for channel_{j} passed',flush=True)
        for name,digest in files.items():
            if sha((out/name).read_bytes())!=digest:raise ValueError(f'Combined output mismatch: {name}')
        combined.update(features=features,files_sha256=files,cadence_seconds=cadences,execution='one fresh process per channel; no automatic retries')
        (out/'protocol.json').write_text(json.dumps(combined,indent=2)+'\n')
        print('All eight isolated builds and combined checks passed.',flush=True)
        return
    root=Path(__file__).resolve().parents[2]
    manifest=root/'docs/research/tables/cadence-frequency-confirmed-2026-09-25/protocol.json'
    expected={x['channel']:x for x in json.loads(manifest.read_text())['sources']}
    training_end='2006-01-01' if args.profile=='full-period' else '2004-01-01'
    calibration_end='2007-01-01' if args.profile=='full-period' else '2004-04-01'
    end='2014-01-01' if args.profile=='full-period' else calibration_end
    grid=pd.date_range('2000-01-01',end,freq='30s',inclusive='left').asi8
    train_end=pd.Timestamp(training_end).value
    files={};cadences={}
    channels=[args.channel] if args.channel is not None else list(range(40,48))
    for j in channels:
        c=f'channel_{j}';record=expected[c]
        blob=(root/'data/ESA-Mission1/channels'/f'{c}.zip').read_bytes()
        if sha(blob)!=record['compressed_sha256']:raise ValueError(f'{c}: archive digest differs')
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if archive.namelist()!=[c]:raise ValueError(f'{c}: unexpected archive members')
            decoded=archive.read(c) if args.decoder=='python' else subprocess.run(['unzip','-p',str(root/'data/ESA-Mission1/channels'/f'{c}.zip'),c],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
        if sha(decoded)!=record['decoded_sha256']:raise ValueError(f'{c}: decoded digest differs')
        frame=pd.read_pickle(io.BytesIO(decoded))
        if sha(decoded)!=record['decoded_sha256'] or sha(blob)!=record['compressed_sha256']:
            raise ValueError(f'{c}: buffers changed during decoding')
        native=frame.index.asi8
        verified_prefix=native[native<pd.Timestamp('2004-04-01').value]
        if sha(verified_prefix.tobytes())!=record['native_timestamps_sha256']:raise ValueError(f'{c}: timestamp digest differs')
        native=native[native<pd.Timestamp(end).value]
        train=native[(native>=grid[0])&(native<train_end)]
        cadence=float(np.median(np.diff(train))/1e9);cadences[c]=cadence
        feature,available,age=observation_age(native,grid,cadence)
        # Independent as-of implementation on a deterministic timestamp subset.
        sample=np.unique(np.linspace(0,len(grid)-1,10000,dtype=int))
        index=pd.Index(native).get_indexer(grid[sample],method='pad')
        np.testing.assert_array_equal(available[sample],index>=0)
        reference=np.zeros(len(sample));valid=index>=0
        reference[valid]=(grid[sample[valid]]-native[index[valid]])/1e9
        np.testing.assert_array_equal(age[sample],reference)
        path=out/f'{c}.npz';np.savez(path,log_age=feature,available=available)
        with np.load(path) as check:
            np.testing.assert_array_equal(check['log_age'],feature)
            np.testing.assert_array_equal(check['available'],available)
        files[path.name]=sha(path.read_bytes())
        print(c,'verified; cadence=',cadence,'unavailable=',int((~available).sum()),flush=True)
        del blob,decoded,frame,native,train,feature,available,age
    path=out/'timestamps.npy';np.save(path,grid);files[path.name]=sha(path.read_bytes())
    # Recheck every output before accepting the directory.
    for name,digest in files.items():
        if sha((out/name).read_bytes())!=digest:raise ValueError(f'Output digest changed: {name}')
    (out/'protocol.json').write_text(json.dumps(dict(status='complete',decoder=args.decoder,features=[f'channel_{j}' for j in channels],
        profile=args.profile,train=['2000-01-01',training_end],calibration=[training_end,calibration_end],assessment=[calibration_end,end] if args.profile=='full-period' else None,rows=len(grid),
        encoding='log1p(seconds_since_latest_native_observation / training_median_native_interval); availability=false before first native observation; placeholder age0 then',
        cadence_seconds=cadences,source_manifest_sha256=sha(manifest.read_bytes()),source_fingerprints=expected,
        files_sha256=files,limitations='Timing metadata only, not additional telemetry channels. No model uses these yet; no improvement claim. Known intermittent compute/read faults unresolved; hashes validate this bounded build.'),indent=2)+'\n')
    print('Complete: sidecars only; no model training or prediction changes.',flush=True)

if __name__=='__main__':main()
