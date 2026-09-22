"""Calibrate an optional specialist OR branch while retaining the frozen baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import config
from .data import RobustChannelScaler
from .development import FOLDS, QUANTILES, load_period
from .evaluation.compare_scoring import paired_scores
from .evaluation.esa import METRIC_VERSION
from .models import MultivariateAE
from .protocol import evaluator_for
from .thresholds import postprocess


def mask(scores, rule):
    return postprocess(scores > rule['threshold'], int(rule['merge_gap']), int(rule['min_dur'])).astype(bool)


def select_combination(baseline, specialist_scores, evaluator):
    """Calibration inputs only; ties retain the baseline-only candidate.

    Combined prediction rate must stay <=1%, with no more false alarm events
    than the baseline on calibration. This is not an assessment guarantee.
    """
    baseline = np.asarray(baseline, dtype=bool)
    scores = np.asarray(specialist_scores)
    if scores.shape != baseline.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid aligned calibration scores')
    base = evaluator.score(baseline)
    if baseline.mean() > .01:
        raise ValueError('Frozen baseline exceeds combined calibration rate cap')
    rows = []

    def add(enabled, threshold, gap, duration, prediction):
        metrics = evaluator.score(prediction)
        rows.append({'enabled': enabled, 'threshold': threshold, 'merge_gap': gap,
                     'min_dur': duration, 'prediction_rate': float(prediction.mean()),
                     'eligible': bool(prediction.mean() <= .01 and metrics['FPe'] <= base['FPe']),
                     **metrics})

    add(False, None, 0, 1, baseline)
    for threshold in np.unique(np.percentile(scores, QUANTILES)):
        for gap in [0, 16, 64]:
            for duration in [1, 8, 32]:
                rule = {'threshold': float(threshold), 'merge_gap': gap, 'min_dur': duration}
                add(True, float(threshold), gap, duration, baseline | mask(scores, rule))
    eligible = [r for r in rows if r['eligible']]
    # Prefer off on an equal F0.5; remaining ties reduce nominal alarm duration.
    best = min(eligible, key=lambda r: (-r['EW_F_0.50'], r['enabled'],
                r['false_positive_seconds'], -(r['threshold'] or 0), r['merge_gap'], r['min_dur']))
    return best, pd.DataFrame(rows)


def calibration_scores(run, features, fold, device, batch_size):
    saved = torch.load(run/'model_checkpoint.pt', map_location='cpu', weights_only=False)
    if saved['features'] != features:
        raise ValueError('Unexpected checkpoint channel order')
    model = MultivariateAE(len(features))
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device)
    scaler = RobustChannelScaler()
    scaler.median_ = np.asarray(saved['scaler_median'], dtype=np.float32)
    scaler.iqr_ = np.asarray(saved['scaler_iqr'], dtype=np.float32)
    if (scaler.median_.shape != (len(features),) or scaler.iqr_.shape != (len(features),)
            or not np.isfinite(scaler.median_).all() or not np.isfinite(scaler.iqr_).all()
            or np.any(scaler.iqr_ <= 0)):
        raise ValueError('Invalid checkpoint scaler')
    start, end, _ = FOLDS[fold]
    values, _, times = load_period(config.TRAIN_FILE, features, start, end)
    scores, uncovered = paired_scores(model, scaler.transform(values), device=device, batch_size=batch_size)
    if uncovered:
        raise ValueError('Uncovered calibration samples')
    return scores['window_mean'], times


def validate_run(run, fold, features):
    plan = json.loads((run.parent/'protocol.json').read_text())
    if (list(plan['folds'][fold]) != list(FOLDS[fold]) or plan['spec']['features'] != features
            or not plan['spec']['pseudo'] or plan['scoring'] != 'window_mean'):
        raise ValueError('Run does not match predeclared fold, channels, objective or scoring')
    for key, value in {'epochs':40, 'batch_size':64, 'seed':42, 'precision':'float32',
                       'quantiles':QUANTILES, 'merge_gaps':[0,16,64], 'min_durs':[1,8,32]}.items():
        if plan[key] != value:
            raise ValueError(f'Run settings differ: {key}')
    required = ['model_checkpoint.pt','frozen_rule.json','assessment_score.npy',
                'assessment_prediction.npy','results.json']
    if any(not (run/p).is_file() for p in required):
        raise ValueError(f'Incomplete source run: {run}')
    rule = json.loads((run/'frozen_rule.json').read_text())['rule']
    if rule.get('val_metric_version') != METRIC_VERSION:
        raise ValueError('Source rule uses an unsupported metric')
    for key in ['SEQ_LEN','PATCH_SIZE','D_MODEL','N_HEADS','N_LAYERS','D_FF_MULT',
                'DROPOUT','CLIP_Z','SCORE_STRIDE','STRIDE','LR','WEIGHT_DECAY',
                'GRAD_CLIP','MAX_TRAIN_WINDOWS','PA_LAMBDA','PA_MARGIN']:
        if plan['settings'][key] != getattr(config, key):
            raise ValueError(f'Source configuration differs: {key}')


def run_fold(fold, baseline_run, specialist_run, output, *, device, batch_size):
    baseline_features = [f'channel_{i}' for i in range(40,48)]
    specialist_features = ['channel_14','channel_21','channel_29']
    validate_run(baseline_run, fold, baseline_features)
    validate_run(specialist_run, fold, specialist_features)
    output.mkdir(parents=True, exist_ok=False)
    b_rule = json.loads((baseline_run/'frozen_rule.json').read_text())['rule']
    s_rule = json.loads((specialist_run/'frozen_rule.json').read_text())['rule']
    baseline, times = calibration_scores(baseline_run, baseline_features, fold, device, batch_size)
    specialist, other_times = calibration_scores(specialist_run, specialist_features, fold, device, batch_size)
    np.testing.assert_array_equal(times, other_times)
    base_mask = mask(baseline, b_rule)
    calibration_evaluator = evaluator_for(times)
    for scores, rule in [(baseline, b_rule), (specialist, s_rule)]:
        if 'val_esa_f05' in rule:
            np.testing.assert_allclose(calibration_evaluator.score(mask(scores, rule))['EW_F_0.50'],
                                       rule['val_esa_f05'], rtol=0, atol=1e-6)
    selected, grid = select_combination(base_mask, specialist, calibration_evaluator)
    grid.to_csv(output/'calibration_candidates.csv', index=False)
    np.save(output/'baseline_calibration_score.npy', baseline)
    np.save(output/'specialist_calibration_score.npy', specialist)
    sources = {}
    for name, run in [('baseline',baseline_run),('specialist',specialist_run)]:
        sources[name] = {p:hashlib.sha256((run/p).read_bytes()).hexdigest()
                         for p in ['model_checkpoint.pt','frozen_rule.json']}
    decision = {'fold':fold, 'selected':selected, 'sources':sources,
                'policy':'baseline frozen; optional postprocessed specialist OR; combined calibration rate <=1%; FPe <= baseline; ties prefer off',
                'limitations':'Retrospective development; sparse calibration; no assessment false-alarm guarantee'}
    (output/'frozen_rule.json').write_text(json.dumps(decision, indent=2))
    print(f'{fold}: combined rule frozen; specialist enabled={selected["enabled"]}; reading assessment', flush=True)
    # Assessment arrays and metrics are first read AFTER combined decision is frozen.
    start, end = FOLDS[fold][1:]
    assessment_times = pd.date_range(start, end, freq='30s', inclusive='left', tz='UTC').asi8
    evaluator = evaluator_for(assessment_times)
    predictions = {}
    for name, run, rule in [('baseline',baseline_run,b_rule),('specialist',specialist_run,s_rule)]:
        score = np.load(run/'assessment_score.npy')
        if score.shape != assessment_times.shape or not np.isfinite(score).all():
            raise ValueError('Invalid assessment score coverage')
        prediction = mask(score, rule)
        np.testing.assert_array_equal(prediction, np.load(run/'assessment_prediction.npy'))
        actual = evaluator.score(prediction)
        saved = json.loads((run/'results.json').read_text())
        for key in ['EW_F_0.50','TPe','FPe','FNe','false_positive_seconds']:
            np.testing.assert_allclose(actual[key], saved[key], atol=1e-12, rtol=0)
        predictions[name] = prediction
        if name == 'specialist':
            specialist_assessment = score
    predictions['combined'] = predictions['baseline'].copy()
    if selected['enabled']:
        predictions['combined'] |= mask(specialist_assessment, selected)
    rows = []
    for name, prediction in predictions.items():
        np.save(output/f'{name}_assessment_prediction.npy', prediction)
        rows.append({'fold':fold,'detector':name, **evaluator.score(prediction)})
    pd.DataFrame(rows).to_csv(output/'results.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specialist-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', nargs='+', choices=list(FOLDS), default=list(FOLDS))
    parser.add_argument('--device', choices=['cpu','cuda'], default=config.DEVICE)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1 or len(set(args.folds)) != len(args.folds):
        parser.error('New output, positive batch size and unique folds required')
    root = config.TRAIN_FILE.parents[4]/'results_longrun/development'
    plan = {'folds':args.folds, 'specialist_root':str(args.specialist_root.resolve()),
            'device':args.device, 'batch_size':args.batch_size, 'quantiles':QUANTILES,
            'policy':'Frozen baseline OR optional specialist; calibration-only selection; off wins F0.5 ties; <=1% combined rate and no extra calibration FPe'}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    args.output.mkdir(parents=True)
    plan['source_sha256'] = {str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path(__file__).parent.rglob('*.py'))}
    plan['inputs'] = {str(p): {'size':p.stat().st_size, 'mtime_ns':p.stat().st_mtime_ns}
                      for p in [config.TRAIN_FILE,config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]}
    plan['annotation_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]}
    (args.output/'protocol.json').write_text(json.dumps(plan, indent=2))
    rows = []
    for fold in args.folds:
        baseline = root/('ae_baseline_2003_seed42' if fold=='2003' else 'ae_baseline_remaining_seed42')/fold
        rows += run_fold(fold, baseline, args.specialist_root/fold, args.output/fold,
                         device=args.device, batch_size=args.batch_size)
        pd.DataFrame(rows).to_csv(args.output/'results.csv', index=False)
    print(pd.DataFrame(rows)[['fold','detector','EW_F_0.50','TPe','FPe','FNe']].to_string(index=False))


if __name__ == '__main__':
    main()
