"""Chronological forecasting controls and Transformer comparison, without test access."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import config
from .data import RobustChannelScaler
from .development import FOLDS, QUANTILES, load_period
from .forecasting import CONTEXT, HORIZON, PATCH, WIDTH, LAYERS, make_model, NominalForecastWindows, forecast_scores
from .protocol import evaluator_for
from .runtime import set_seed
from .thresholds import sweep_thresholds, select_rows

DEFAULT_MODELS = ('persistence', 'mlp', 'transformer')
MODELS = (*DEFAULT_MODELS, 'transformer_residual')


def calibrate(model, values, evaluator, *, device, batch_size):
    scores, covered = forecast_scores(model, values, device=device, batch_size=batch_size)
    grid = sweep_thresholds(scores, evaluator=evaluator,
                            threshold_values=np.percentile(scores[covered], QUANTILES),
                            merge_gaps=[0], min_durs=[1])
    return select_rows(grid).iloc[0].to_dict(), grid, scores, covered


def run_model(name, features, boundaries, output, *, epochs, device, batch_size, seed):
    set_seed(seed)
    training_end, calibration_end, assessment_end = boundaries
    print(f'{output}: loading nominal training data', flush=True)
    train, labels, _ = load_period(config.TRAIN_FILE, features, None, training_end)
    if np.count_nonzero(labels == 0) < 100:
        raise ValueError('Insufficient nominal data for training-only normalization')
    scaler = RobustChannelScaler().fit(train, labels)
    train = scaler.transform(train)
    dataset = NominalForecastWindows(train, labels, seed=seed, limit=config.MAX_TRAIN_WINDOWS)
    del train, labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    calibration, _, times = load_period(config.TRAIN_FILE, features, training_end, calibration_end)
    calibration = scaler.transform(calibration)
    evaluator = evaluator_for(times)
    model = make_model(name).to(device)
    learned = name != 'persistence'
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY) if learned else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) if learned else None
    best = None
    history = []
    started = time.perf_counter()
    print(f'{name}: {len(dataset)} nominal windows, {sum(p.numel() for p in model.parameters())} parameters', flush=True)
    for epoch in (range(1, epochs+1) if learned else [0]):
        epoch_start = time.perf_counter()
        model.train()
        total, count = 0., 0
        if learned:
            for context, target in loader:
                context, target = context.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(context)-target).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                total += loss.item()*len(context)
                count += len(context)
            scheduler.step()
        row = {'epoch': epoch, 'loss': total/count if count else None}
        if epoch % 5 == 0 or epoch == epochs:
            rule, grid, scores, covered = calibrate(model, calibration, evaluator, device=device, batch_size=batch_size)
            row['calibration_f05'] = rule['val_esa_f05']
            if best is None or rule['val_esa_f05'] > best['rule']['val_esa_f05']:
                best = {'model': name, 'epoch': epoch, 'rule': rule,
                        'uncovered_calibration_samples': int((~covered).sum()),
                        'calibration_scope': evaluator.metadata()}
                torch.save({'model': name, 'model_state_dict': model.state_dict(), 'features': features,
                            'context': CONTEXT, 'horizon': HORIZON, 'scaler_median': scaler.median_,
                            'scaler_iqr': scaler.iqr_}, output/'model_checkpoint.pt')
                grid.to_csv(output/'calibration_candidates.csv', index=False)
                np.save(output/'calibration_score.npy', scores)
        row['seconds'] = time.perf_counter()-epoch_start
        history.append(row)
        pd.DataFrame(history).to_csv(output/'history.csv', index=False)
        print(name, row, flush=True)
    (output/'frozen_rule.json').write_text(json.dumps(best, indent=2))
    saved = torch.load(output/'model_checkpoint.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    # Release training buffers before assessment; never use assessment to select a model.
    del loader, dataset, calibration
    print(f'{name}: rule frozen; loading assessment', flush=True)
    values, _, times = load_period(config.TRAIN_FILE, features, calibration_end, assessment_end)
    scores, covered = forecast_scores(model, scaler.transform(values), device=device, batch_size=batch_size)
    prediction = covered & (scores > best['rule']['threshold'])
    assessment_evaluator = evaluator_for(times)
    result = {'model': name, 'epoch': best['epoch'], 'calibration_f05': best['rule']['val_esa_f05'],
              **assessment_evaluator.score(prediction), 'uncovered_assessment_samples': int((~covered).sum()),
              'parameters': sum(p.numel() for p in model.parameters()),
              'elapsed_seconds': time.perf_counter()-started}
    np.save(output/'assessment_score.npy', scores)
    np.save(output/'assessment_covered.npy', covered)
    np.save(output/'assessment_prediction.npy', prediction)
    (output/'results.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', nargs='+', choices=list(FOLDS), default=list(FOLDS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=config.DEVICE)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.folds)) != len(args.folds) or len(set(args.models)) != len(args.models):
        parser.error('Positive epochs/batch size and unique fold/model names required')
    if args.output.exists():
        parser.error('Use a new output directory; interrupted runs cannot resume')
    features = next(s['features'] for s in config.RUN_SPECS if s['run'] == 'pa_center_ch08_40_47')
    plan = {'version': 'causal_forecast_development_v1', 'features': features,
            'folds': {f: FOLDS[f] for f in args.folds}, 'models': args.models,
            'context': CONTEXT, 'horizon': HORIZON, 'forecast_stride': HORIZON,
            'patch': PATCH, 'width': WIDTH, 'layers': LAYERS, 'heads': 8, 'dropout': .1,
            'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
            'learning_rate': config.LR, 'weight_decay': config.WEIGHT_DECAY,
            'gradient_clip': config.GRAD_CLIP, 'max_train_windows': config.MAX_TRAIN_WINDOWS,
            'clip_z': config.CLIP_Z, 'device': args.device, 'precision': 'float32',
            'score': 'max-channel squared error per arriving target; no future residual averaging',
            'alarm': 'threshold only; no gap filling or retrospective duration filtering',
            'quantiles': QUANTILES, 'selection': 'calibration F0.5 every 5 epochs and last; earliest tie; 1% cap',
            'warmup': 'first 256 samples of each period unscored; zero alarms; full-period evaluation',
            'training': 'nominal contexts AND targets; training-only robust scaler; no pseudo anomalies',
            'residual_variant': 'transformer_residual centers context on its last observed value and adds that value to predicted deviations; targets and score units unchanged',
            'limitations': 'development folds; sparse calibration; AE has different information and alarm policies; GPU bitwise determinism not guaranteed',
            'software': {'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    plan['source_sha256'] = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.rglob('*.py'))}
    plan['inputs'] = {}
    for p in (config.TRAIN_FILE, config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE):
        plan['inputs'][str(p)] = {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
        if p != config.TRAIN_FILE:
            plan['inputs'][str(p)]['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(plan, indent=2))
    results = []
    for fold in args.folds:
        for name in args.models:
            output = args.output/fold/name
            output.mkdir(parents=True)
            result = run_model(name, features, FOLDS[fold], output, epochs=args.epochs,
                               device=args.device, batch_size=args.batch_size, seed=args.seed)
            results.append({'fold': fold, **result})
            pd.DataFrame(results).to_csv(args.output/'results.csv', index=False)
    print(pd.DataFrame(results)[['fold', 'model', 'epoch', 'calibration_f05', 'EW_F_0.50', 'TPe', 'FPe', 'FNe']].to_string(index=False))


if __name__ == '__main__': main()
