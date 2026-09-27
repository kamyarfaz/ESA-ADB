"""One prespecified training-q99 normalization ablation using diagnostic caches only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config
from ..diagnose_channels import FEATURES, digest, event_diagnostics
from ..development import QUANTILES
from ..protocol import evaluator_for
from ..thresholds import sweep_thresholds, select_rows, postprocess


def fit_scale(training_errors):
    errors = np.asarray(training_errors)
    if errors.ndim != 2 or errors.shape[1] != 8 or len(errors) < 100:
        raise ValueError('Expected at least 100 nominal training windows by eight channels')
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError('Training errors must be finite and nonnegative')
    return np.maximum(np.percentile(errors, 99, axis=0), 1e-8)


def normalized_max(channel_scores, scale):
    scores, scale = np.asarray(channel_scores), np.asarray(scale)
    if scores.ndim != 2 or scores.shape[0] != 8 or scale.shape != (8,):
        raise ValueError('Expected eight channel scores and scales')
    if not np.isfinite(scores).all() or np.any(scores < 0) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid score/scale values')
    return (scores.astype(np.float64) / scale[:, None]).max(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; results are preserved')
    src = args.diagnostic
    prior = json.loads((src/'protocol.json').read_text())
    completed = json.loads((src/'results.json').read_text())
    if prior['features'] != FEATURES or completed['reproduction'] != 'passed within 1e-6':
        raise ValueError('Require completed fixed-channel diagnostic with baseline parity')
    package = Path(__file__).resolve().parents[1]
    for name in ['config.py', 'thresholds.py', 'evaluation/esa.py']:
        if digest(package/name) != prior['sources'][name]:
            raise ValueError(f'Diagnostic evaluation dependency changed: {name}')
    for name, expected in prior['annotations'].items():
        if digest(name) != expected:
            raise ValueError('Annotations differ from completed diagnostic')
    inputs = ['training_window_errors.npz','calibration_channel_scores.npz','results.json','protocol.json']
    hashes = {name: digest(src/name) for name in inputs}
    plan = dict(experiment='channel-normalization-q99-v1', diagnostic=str(src.resolve()),
                features=FEATURES, checkpoint_sha256=prior['checkpoint_sha256'],
                formula='max_c(aggregated_window_mean_error_c / max(training_window_error_q99_c, 1e-8))',
                training='2000–2005 nominal sampled windows', calibration='2006',
                fit='One prespecified q99 scale per channel. No percentile/floor search or centering.',
                quantiles=QUANTILES, merge_gaps=[0,16,64], min_durs=[1,8,32],
                selection='Existing calibration F0.5 selector with <=1% alarm fraction; original wins inter-method ties',
                input_sha256=hashes, source_sha256={str(p.relative_to(package)):digest(p) for p in package.rglob('*.py')},
                limitations='Calibration-only exploratory ablation; same year selected checkpoint. No independent validation, assessment access, training or OR ensemble. Window q99 scale is a channel weight, not a calibrated timestamp probability.')
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(plan, indent=2))
    with np.load(src/'training_window_errors.npz') as f:
        train = f['errors']; train_times = f['timestamps']
    if len(train_times) != len(train) or not np.all((train_times >= pd.Timestamp('2000-01-01').value) & (train_times < pd.Timestamp('2006-01-01').value)):
        raise ValueError('Training-cache interval mismatch')
    scale = fit_scale(train)
    pd.DataFrame({'channel':FEATURES, 'training_error_q99_scale':scale}).to_csv(args.output/'scales.csv', index=False)
    with np.load(src/'calibration_channel_scores.npz') as f:
        channels = f['scores']; times = f['timestamps']; frozen_pred = f['prediction']
    expected_times = pd.date_range('2006-01-01','2007-01-01',freq='30s',inclusive='left').asi8
    if not np.array_equal(times,expected_times) or channels.shape != (8,len(times)):
        raise ValueError('Calibration-cache coverage mismatch')
    evaluator = evaluator_for(times)
    reproduced = evaluator.score(frozen_pred)
    for key in ['EW_F_0.50','TPe','FPe','FNe']:
        if not np.isclose(reproduced[key],completed['calibration'][key],atol=1e-12,rtol=0):
            raise ValueError('Diagnostic-cache baseline parity failed')
    scores = {'original_max':channels.max(axis=0), 'training_q99_max':normalized_max(channels,scale)}
    results, rules, predictions = [], {}, {}
    for name, score in scores.items():
        print(f'Calibration sweep: {name}',flush=True)
        grid = sweep_thresholds(score,evaluator=evaluator,threshold_values=np.percentile(score,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
        grid.to_csv(args.output/f'{name}_candidates.csv',index=False)
        rule = select_rows(grid).iloc[0].to_dict()
        prediction = postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        if name == 'original_max' and not np.array_equal(prediction,frozen_pred):
            raise ValueError('Recalibrated original mask differs from diagnostic baseline')
        results.append(dict(mode=name,**evaluator.score(prediction)))
        rules[name] = rule; predictions[name] = prediction
        event_diagnostics(evaluator,prediction,score).to_csv(args.output/f'{name}_events.csv',index=False)
    table = pd.DataFrame(results)
    preferred = 'training_q99_max' if results[1]['EW_F_0.50'] > results[0]['EW_F_0.50'] else 'original_max'
    table['calibration_preferred'] = table['mode']==preferred
    # Verify cache bytes again, relevant after recurring data-integrity incidents.
    if any(digest(src/name)!=h for name,h in hashes.items()):
        raise ValueError('Diagnostic inputs changed during comparison')
    np.savez(args.output/'calibration_predictions.npz',timestamps=times,**predictions)
    (args.output/'rules.json').write_text(json.dumps(rules,indent=2))
    table.to_csv(args.output/'results.csv',index=False)
    print(table[['mode','EW_F_0.50','TPe','FPe','FNe','false_positive_seconds','calibration_preferred']].to_string(index=False),flush=True)
    print('Calibration only; no assessment evaluated and no model promoted.',flush=True)


if __name__=='__main__':main()
