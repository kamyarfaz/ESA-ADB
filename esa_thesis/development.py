"""Expanding-window AE development runs; never use the benchmark test CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import config
from .augmentation import make_pseudo_anomaly
from .data import NormalWindows, RobustChannelScaler, get_y_any
from .evaluation.compare_scoring import paired_scores
from .models import MultivariateAE
from .protocol import evaluator_for
from .runtime import set_seed
from .development_recovery import atomic_save, capture_rng, restore_rng, check_protocol, prepare_fold
from .thresholds import sweep_thresholds, select_rows, postprocess

FOLDS = {str(year): (f'{year}-01-01', f'{year}-04-01', f'{year+1}-01-01')
         for year in (2003, 2004, 2005)}
QUANTILES = [90, 95, 97.5, 98, 99, 99.5, 99.75, 99.9, 99.95, 99.99, 100]


def development_spec(channel_set):
    """Keep the original channel order, then append the fixed coverage ablation."""
    if channel_set not in ('baseline', 'expanded'):
        raise ValueError('Unknown channel set')
    original = next(s for s in config.RUN_SPECS if s['run'] == 'pa_center_ch08_40_47')
    spec = {**original, 'features': list(original['features'])}
    if channel_set == 'expanded':
        spec['run'] = 'pa_center_ch08_40_47_plus14_21_29'
        spec['features'] += ['channel_14', 'channel_21', 'channel_29']
    return spec


def load_period(path, features, start, end):
    """Strict finite inputs; no imputation across time boundaries."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    labels = [c for c in header if c.startswith('is_anomaly_')]
    if not set(features).issubset(header) or not labels:
        raise ValueError('Missing features or annotation columns')
    frames = []
    with pd.read_csv(path, usecols=['timestamp', *features, *labels], chunksize=100000) as reader:
        for frame in reader:
            times = pd.to_datetime(frame.timestamp, utc=True)
            keep = times < pd.Timestamp(end, tz='UTC')
            if start is not None:
                keep &= times >= pd.Timestamp(start, tz='UTC')
            frames.append(frame.loc[keep])
            if (times >= pd.Timestamp(end, tz='UTC')).any():
                break
    frame = pd.concat(frames, ignore_index=True)
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).as_unit('ns').asi8
    if len(times) < config.SEQ_LEN or not np.all(np.diff(times) == 30_000_000_000):
        raise ValueError('Period must contain complete 30-second windows')
    expected_start = pd.Timestamp(start or '2000-01-01', tz='UTC').value
    if times[0] != expected_start or times[-1] != pd.Timestamp(end, tz='UTC').value-30_000_000_000:
        raise ValueError('Period does not cover its declared boundaries')
    values = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(frame[labels].to_numpy(), [0, 1, 2, 3]).all():
        raise ValueError('Invalid features or labels; validate the source CSV')
    return values, get_y_any(frame), times


def run_fold(spec, boundaries, output, *, epochs, device, batch_size, seed):
    train_end, calibration_end, assessment_end = boundaries
    set_seed(seed)
    features = spec['features']
    print(f'{output.name}: loading training data before {train_end}', flush=True)
    x, y, _ = load_period(config.TRAIN_FILE, features, None, train_end)
    scaler = RobustChannelScaler().fit(x, y)
    x = scaler.transform(x)
    # Fail closed rather than invoking NormalWindows' historical all-window fallback.
    cumulative = np.r_[0, np.cumsum(y > 0)]
    starts = np.arange(0, len(y)-config.SEQ_LEN+1, config.STRIDE)
    if not np.any(cumulative[starts+config.SEQ_LEN] == cumulative[starts]):
        raise ValueError('No nominal training windows')
    if np.count_nonzero(y == 0) < 100:
        raise ValueError('Insufficient nominal samples for scaler')
    dataset = NormalWindows(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'{output.name}: {len(dataset)} nominal windows; loading calibration', flush=True)
    val, _, times = load_period(config.TRAIN_FILE, features, train_end, calibration_end)
    val = scaler.transform(val)
    evaluator = evaluator_for(times)
    model = MultivariateAE(len(features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best = None
    history = []
    best_checkpoint = best_grid = None
    start_epoch = 1
    latest = output/'checkpoint_last.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best, history = saved['best'], saved['history']
        best_checkpoint, best_grid = saved['best_checkpoint'], saved['best_grid']
        if not np.array_equal(scaler.median_, saved['scaler_median']) or not np.array_equal(scaler.iqr_, saved['scaler_iqr']):
            raise ValueError('Training scaler differs from checkpoint')
        start_epoch = saved['epoch']+1
        if best_checkpoint is not None:
            atomic_save(best_checkpoint, output/'model_checkpoint.pt')
            pd.DataFrame(best_grid).to_csv(output/'calibration_candidates.csv', index=False)
        pd.DataFrame(history).to_csv(output/'history.csv', index=False)
        restore_rng(saved['rng'])
        print(f'{output.name}: resuming at epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, epochs+1):
        model.train()
        losses = []
        for inputs in loader:
            inputs = inputs.to(device)
            optimizer.zero_grad(set_to_none=True)
            residual = (model(inputs)-inputs).square().mean(dim=(1, 2, 3))
            loss = residual.mean()
            if spec.get('pseudo', False):
                corrupted = make_pseudo_anomaly(inputs)
                abnormal = (model(corrupted)-corrupted).square().mean(dim=(1, 2, 3))
                loss = loss + config.PA_LAMBDA * torch.relu(config.PA_MARGIN-abnormal/(residual.mean().detach()+1e-8)).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        row = {'epoch': epoch, 'loss': float(np.mean(losses))}
        if epoch % 5 == 0 or epoch == epochs:
            scores, uncovered = paired_scores(model, val, device=device, batch_size=batch_size)
            score = scores['window_mean']
            grid = sweep_thresholds(score, evaluator=evaluator, threshold_values=np.percentile(score, QUANTILES),
                                    merge_gaps=[0, 16, 64], min_durs=[1, 8, 32])
            rule = select_rows(grid).iloc[0].to_dict()
            row['calibration_f05'] = rule['val_esa_f05']
            # Equal scores retain the earlier checkpoint, independently of assessment.
            if best is None or rule['val_esa_f05'] > best['rule']['val_esa_f05']:
                best = {'epoch': epoch, 'rule': rule, 'uncovered_calibration_samples': uncovered}
                best_checkpoint = {'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                   'features': features, 'scaler_median': scaler.median_, 'scaler_iqr': scaler.iqr_}
                best_grid = grid.to_dict('records')
                atomic_save(best_checkpoint, output/'model_checkpoint.pt')
                grid.to_csv(output/'calibration_candidates.csv', index=False)
        history.append(row)
        pd.DataFrame(history).to_csv(output/'history.csv', index=False)
        atomic_save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'best': best, 'history': history,
                     'best_checkpoint': best_checkpoint, 'best_grid': best_grid,
                     'scaler_median': scaler.median_, 'scaler_iqr': scaler.iqr_, 'rng': capture_rng()}, latest)
        print(output.name, row, flush=True)
    # Freeze decisions on disk before loading assessment observations.
    (output/'frozen_rule.json').write_text(json.dumps(best, indent=2))
    saved = torch.load(output/'model_checkpoint.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    print(f'{output.name}: decision frozen; scoring assessment', flush=True)
    assessment, _, times = load_period(config.TRAIN_FILE, features, calibration_end, assessment_end)
    scores, uncovered = paired_scores(model, scaler.transform(assessment), device=device, batch_size=batch_size)
    score = scores['window_mean']
    rule = best['rule']
    prediction = postprocess(score > rule['threshold'], int(rule['merge_gap']), int(rule['min_dur']))
    result = {'epoch': best['epoch'], 'calibration_f05': rule['val_esa_f05'],
              **evaluator_for(times).score(prediction), 'uncovered_assessment_samples': uncovered}
    np.save(output/'assessment_score.npy', score)
    np.save(output/'assessment_prediction.npy', prediction)
    result_temp = output/'results.json.tmp'
    result_temp.write_text(json.dumps(result, indent=2))
    result_temp.replace(output/'results.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', nargs='+', choices=list(FOLDS), default=list(FOLDS))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channel-set', choices=['baseline', 'expanded'], default='baseline')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=config.DEVICE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip complete folds and restore full epoch checkpoints')
    parser.add_argument('--restart-incomplete', action='store_true', help='With --resume, archive and restart old partial folds lacking full checkpoints')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.folds)) != len(args.folds):
        parser.error('Positive epochs/batch size and unique folds required')
    if args.restart_incomplete and not args.resume:
        parser.error('--restart-incomplete requires --resume')
    if args.resume and not (args.output/'protocol.json').is_file():
        parser.error('--resume requires an existing protocol.json')
    if args.output.exists() and not args.resume:
        parser.error('Output exists; use --resume or choose a new directory')
    spec = development_spec(args.channel_set)
    plan = {'channel_set': args.channel_set, 'spec': spec, 'folds': {f: FOLDS[f] for f in args.folds},
            'intervals': '[start,end); expanding train, 3-month calibration, 9-month assessment',
            'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
            'device': args.device, 'precision': 'float32', 'scoring': 'window_mean',
            'quantiles': QUANTILES, 'merge_gaps': [0, 16, 64], 'min_durs': [1, 8, 32],
            'checkpoint_selection': 'corrected calibration F0.5 every 5 epochs and final epoch; earliest tie',
            'limitations': 'Retrospective AE; fixed historical channel group; assessment folds are development data; exact GPU determinism not guaranteed'}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    plan['settings'] = {k: v for k, v in vars(config).items() if k.isupper()}
    plan['source_sha256'] = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.rglob('*.py'))}
    plan['inputs'] = {}
    for p in (config.TRAIN_FILE, config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE):
        plan['inputs'][str(p)] = {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
    check_protocol(args.output, plan, args.resume)
    results = []
    for fold in args.folds:
        output = args.output/fold
        result = prepare_fold(output, args.resume, args.restart_incomplete)
        if result is None:
            result = run_fold(spec, FOLDS[fold], output, epochs=args.epochs, device=args.device,
                              batch_size=args.batch_size, seed=args.seed)
        else:
            print(f'SKIP completed fold {fold}', flush=True)
        results.append({'fold': fold, **result})
        pd.DataFrame(results).to_csv(args.output/'results.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
