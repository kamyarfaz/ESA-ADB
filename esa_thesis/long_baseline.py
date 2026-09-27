"""Fixed long-period baseline: train 2000–2005, calibrate 2006, assess 2007–2013."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import config
from .development import development_spec, run_fold, QUANTILES
from .development_recovery import check_protocol, prepare_fold
from .protocol import evaluator_for

BOUNDARIES = ('2006-01-01', '2007-01-01', '2014-01-01')
EXPECTED_HASHES = {
    'train': 'fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad',
    'test': '9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c',
}


def verify_inputs():
    result = {}
    for kind, path in [('train', config.TRAIN_FILE), ('test', config.TEST_FILE)]:
        digest = hashlib.sha256()
        before = path.stat()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != EXPECTED_HASHES[kind]:
            raise ValueError(f'{kind} CSV checksum differs from verified dataset; preserve file and investigate before training')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Input changed during verification')
        result[str(path)] = dict(sha256=digest.hexdigest(), size=before.st_size, mtime_ns=before.st_mtime_ns)
    return result


def yearly_results(output):
    prediction = np.load(output/'assessment_prediction.npy', mmap_mode='r')
    times = pd.date_range(BOUNDARIES[1], BOUNDARIES[2], freq='30s', inclusive='left').asi8
    if len(prediction) != len(times):
        raise ValueError('Assessment predictions do not cover 2007–2013')
    results = []
    for year in range(2007, 2014):
        keep = (times >= pd.Timestamp(f'{year}-01-01').value) & (times < pd.Timestamp(f'{year+1}-01-01').value)
        results.append({'year': year, **evaluator_for(times[keep]).score(prediction[keep])})
    return pd.DataFrame(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-data', action='store_true', help='Verify checksums only; no output directory or training')
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1:
        p.error('epochs and batch size must be positive')
    spec = development_spec('baseline')
    plan = dict(experiment='long-baseline-v1', spec=spec,
        train=['2000-01-01', BOUNDARIES[0]], calibration=list(BOUNDARIES[:2]),
        assessment=list(BOUNDARIES[1:]), intervals='[start,end)',
        epochs=a.epochs, batch_size=a.batch_size, seed=a.seed, device=a.device,
        max_train_windows=0, window_policy='all globally nominal stride-16 windows, length 256; no random window cap',
        scoring='window_mean', precision='float32', quantiles=QUANTILES,
        merge_gaps=[0,16,64], min_durs=[1,8,32],
        checkpoint_selection='2006 corrected calibration F0.5 every 5 epochs and final epoch; earliest tie',
        assessment_policy='one frozen rule; continuous 2007–2013 scoring; pooled primary metric plus calendar-year slices, no yearly retuning',
        limitations='Retrospective AE; inherited prepared preprocessing; later-period covariates and historical test results previously inspected; not untouched test. More years and uncapped window budget both differ from historical development runs.')
    print(json.dumps(plan, indent=2), flush=True)
    if a.dry_run:
        return
    if a.verify_data:
        print(json.dumps(verify_inputs(), indent=2)); return
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; run in the server GPU environment')
    if a.output.exists() and not a.resume:
        p.error('Output exists; use --resume or a new directory')
    if a.resume and not (a.output/'protocol.json').is_file():
        p.error('--resume requires an existing protocol')
    print('Verifying both CSV checksums before training...', flush=True)
    plan['inputs'] = verify_inputs()
    for path in (config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE):
        plan['inputs'][str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    plan['settings'] = {k:v for k,v in vars(config).items() if k.isupper()}
    plan['source_sha256'] = {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.rglob('*.py'))}
    check_protocol(a.output, plan, a.resume)
    output = a.output/'full_period'
    result = prepare_fold(output, a.resume, False)
    if result is None:
        result = run_fold(spec, BOUNDARIES, output, epochs=a.epochs, device=a.device,
                          batch_size=a.batch_size, seed=a.seed,
                          assessment_path=config.TEST_FILE, max_windows=0)
    pd.DataFrame([result]).to_csv(a.output/'pooled_results.csv', index=False)
    annual = yearly_results(output)
    annual.to_csv(a.output/'yearly_results.csv', index=False)
    print(annual.to_string(index=False))


if __name__ == '__main__': main()
