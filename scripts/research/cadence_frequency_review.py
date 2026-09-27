"""Fixed historical cadence audit; no training, score selection or assessment."""
from pathlib import Path
import json
import argparse
import io
import zipfile
import hashlib
import numpy as np
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[2]
    source=root/'results_longrun/diagnostics/baseline_2004_epoch35_v2'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output
    output.mkdir(parents=True,exist_ok=False)
    training=np.load(source/'training_window_errors.npz')
    calibration=np.load(source/'calibration_channel_scores.npz')
    grid=pd.date_range('2000-01-01','2004-04-01',freq='30s',inclusive='left')
    times=grid.asi8;boundary=pd.Timestamp('2004-01-01').value
    cal=times>=boundary
    np.testing.assert_array_equal(times[cal],calibration['timestamps'])
    sample_starts=np.searchsorted(times,training['timestamps'])
    np.testing.assert_array_equal(times[sample_starts],training['timestamps'])
    assert np.all(times[sample_starts+255]<boundary)
    rows=[];samples=[];sources=[]
    for j in range(8):
        channel=f'channel_{40+j}'
        path=root/'data/ESA-Mission1/channels'/f'{channel}.zip'
        before=path.stat()
        # Read once into an immutable snapshot; verify archive and decoded bytes.
        compressed=path.read_bytes()
        compressed_sha=hashlib.sha256(compressed).hexdigest()
        with zipfile.ZipFile(io.BytesIO(compressed)) as archive:
            assert len(archive.namelist())==1
            decoded=archive.read(archive.namelist()[0])  # Includes CRC validation.
        decoded_sha=hashlib.sha256(decoded).hexdigest()
        frame=pd.read_pickle(io.BytesIO(decoded))
        assert hashlib.sha256(decoded).hexdigest()==decoded_sha
        assert hashlib.sha256(compressed).hexdigest()==compressed_sha
        del compressed,decoded
        # Pickle storage is whole-mission; inspect only timestamps before calibration ends.
        native=frame.index.asi8
        native=native[native<pd.Timestamp('2004-04-01').value]
        assert np.all(np.diff(native)>0)
        train_native=native[(native>=times[0])&(native<boundary)]
        cadence=float(np.median(np.diff(train_native))/1e9)
        index=np.searchsorted(native,times,side='right')-1
        valid=index>=0
        age=np.full(len(times),np.nan)
        age[valid]=(times[valid]-native[index[valid]])/1e9
        periods=[(str(year),grid.year==year) for year in range(2000,2004)]
        periods += [('training_all',~cal),('calibration',cal)]
        periods += [(f'2004-{month:02}',cal&(grid.month==month)) for month in (1,2,3)]
        for period,keep in periods:
            values=age[keep&valid]
            rows.append(dict(channel=channel,period=period,training_median_native_interval_seconds=cadence,
                samples=int(keep.sum()),no_previous_observation=int((keep&~valid).sum()),
                age_median_seconds=float(np.median(values)),age_q99_seconds=float(np.quantile(values,.99)),
                max_age_seconds=float(values.max()),fraction_age_gt_2cadence=float(np.mean(values>2*cadence)),
                fraction_age_gt_5cadence=float(np.mean(values>5*cadence))))
        # Same wholly nominal reference windows used by completed diagnostic.
        stale=valid&(age>2*cadence);count=np.r_[0,np.cumsum(stale)]
        flagged=(count[sample_starts+256]-count[sample_starts])>0
        missing=np.r_[0,np.cumsum(~valid)]
        complete=(missing[sample_starts+256]-missing[sample_starts])==0
        for label,mask in [('contains_old_observation',flagged&complete),('no_old_observation',~flagged&complete),('incomplete_native_history',~complete)]:
            errors=training['errors'][mask,j]
            samples.append(dict(channel=channel,group=label,windows=int(mask.sum()),total_windows=len(flagged),
                median_window_error=float(np.median(errors)) if len(errors) else None,
                q99_window_error=float(np.quantile(errors,.99)) if len(errors) else None))
        assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
        sources.append(dict(channel=channel,bytes=before.st_size,mtime_ns=before.st_mtime_ns,compressed_sha256=compressed_sha,decoded_sha256=decoded_sha,native_timestamps_sha256=hashlib.sha256(native.tobytes()).hexdigest()))
        print(channel,'training cadence',cadence,'nominal reference windows with age>2cadence',int(flagged.sum()),flush=True)
    pd.DataFrame(rows).to_csv(output/'age_frequency.csv',index=False)
    pd.DataFrame(samples).to_csv(output/'nominal_training_windows.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(interval='[2000-01-01,2004-04-01)',
        training='2000–2003',calibration='Jan–Mar2004',reference='Per-channel median positive native interval in training',
        definitions='Observation age at30s grid; descriptive thresholds strictly>2x and>5x training cadence. Missing initial history separately counted.',
        scope='Full-period summaries include annotated data; sample window analysis uses saved8000 wholly nominal reference windows, not all optimization windows.',
        limitations='Staleness is not a packet-loss label. Whole-mission pickles read then restricted; no later-period analysis. File stats checked, not upstream cryptographic provenance. No alarm suppression/model/threshold changes.',sources=sources),indent=2)+'\n')
    print(pd.DataFrame(rows).query("period in ['training_all','calibration']").to_string(index=False),flush=True)

if __name__=='__main__':main()
