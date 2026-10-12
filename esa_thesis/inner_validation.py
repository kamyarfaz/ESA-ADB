"""Pre-2003 comparison of calibration-selected versus fixed-epoch checkpoints."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .augmentation_comparison import isolated_random, training_starts, score, verify_inputs
from .data import AllWindows, RobustChannelScaler
from .development import QUANTILES, load_period
from .development_recovery import atomic_save, check_protocol
from .diagnose_channels import FEATURES, TRAIN_SHA, digest
from .masked_pilot import assessment_scopes
from .models import MultivariateAE
from .protocol import evaluator_for
from .runtime import set_seed
from .short_augmentation import make_training_anomaly
from .thresholds import sweep_thresholds, select_rows, postprocess

FOLDS = {str(y): (f'{y}-01-01', f'{y}-07-01', f'{y+1}-01-01') for y in (2001, 2002)}
POLICIES = ('calibration_selected', 'fixed_epoch5')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def choose_decisions(history):
    if not history or [r['epoch'] for r in history] != list(range(1, len(history)+1)):
        raise ValueError('Incomplete epoch history')
    best = max(history, key=lambda r: r['rule']['val_esa_f05'])
    flag = all(r['rule']['val_esa_f05'] == 0 for r in history)
    return {policy: dict(epoch=row['epoch'], rule=row['rule'],
                         uninformative_calibration=flag,
                         selected_epoch_zero_f05=row['rule']['val_esa_f05'] == 0)
            for policy, row in zip(POLICIES, (best, history[-1]))}


def train(x, starts, val, times, scaler, out, args):
    """One trajectory, every epoch saved; epoch-addressed RNG supports replay."""
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = MultivariateAE(8).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    history = []
    if (out/'last.pt').exists():
        if not args.resume:
            raise ValueError('Resume required')
        saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); history = saved['history']
        for row in history:
            folder = out/f"epoch_{row['epoch']}"
            for name, sha in row['sha256'].items():
                if digest(folder/name) != sha:
                    raise ValueError('Saved epoch artifact changed')
    dataset = AllWindows(x); dataset.starts = starts
    ev = evaluator_for(times)
    for epoch in range(len(history)+1, args.epochs+1):
        began = time.monotonic(); set_seed(args.seed+epoch*100)
        generator = torch.Generator().manual_seed(args.seed+epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        model.train(); losses = []
        for batch, values in enumerate(loader):
            values = values.to(args.device); optimizer.zero_grad(set_to_none=True)
            normal = (model(values)-values).square().mean(dim=(1, 2, 3))
            with isolated_random(args.seed+1000000+epoch*100000+batch):
                corrupted = make_training_anomaly(values, mode='short_no_roll')
            abnormal = (model(corrupted)-corrupted).square().mean(dim=(1, 2, 3))
            penalty = torch.relu(config.PA_MARGIN-abnormal/(normal.mean().detach()+1e-8)).mean()
            loss = normal.mean()+config.PA_LAMBDA*penalty
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step(); losses.append([loss.item(), normal.mean().item(), penalty.item()])
            if batch % 100 == 0:
                print(out.name, epoch, batch+1, '/', len(loader), flush=True)
        scheduler.step()
        values = score(model, val, args)
        grid = sweep_thresholds(values, evaluator=ev, threshold_values=np.percentile(values, QUANTILES),
                                merge_gaps=[0, 16, 64], min_durs=[1, 8, 32])
        rule = select_rows(grid).iloc[0].to_dict()
        folder = out/f'epoch_{epoch}'; folder.mkdir(exist_ok=True)
        atomic_save(dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                         features=FEATURES, scaler_median=scaler.median_, scaler_iqr=scaler.iqr_,
                         augmentation='short_no_roll', epoch=epoch), folder/'model.pt')
        np.save(folder/'calibration_score.npy', values)
        grid.to_csv(folder/'calibration_candidates.csv', index=False)
        means = np.mean(losses, axis=0)
        row = dict(epoch=epoch, loss=means[0], normal_mse=means[1], pseudo_penalty=means[2],
                   calibration_f05=rule['val_esa_f05'], rule=rule, seconds=time.monotonic()-began,
                   sha256={p.name:digest(p) for p in folder.iterdir() if p.is_file()})
        history.append(row)
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), history=history), out/'last.pt')
        pd.DataFrame([{k:v for k,v in h.items() if k not in ('rule','sha256')} for h in history]).to_csv(out/'history.csv', index=False)
        print(out.name, {k:v for k,v in row.items() if k not in ('rule','sha256')}, flush=True)
    decisions = choose_decisions(history)
    write_json(out/'frozen_decisions.json', decisions)
    return decisions


def monthly_summary(x, y, times, scaler, period):
    months = pd.to_datetime(times, utc=True).strftime('%Y-%m')
    rows = []
    for month in np.unique(months):
        keep = (months == month) & (y == 0)
        if not keep.any():
            raise ValueError('No nominal samples in month')
        values = x[keep]
        scaled = (values-scaler.median_)/scaler.iqr_
        for j, channel in enumerate(FEATURES):
            rows.append(dict(period=period, month=month, channel=channel, nominal_samples=int(keep.sum()),
                             median=float(np.median(values[:,j])), q01=float(np.percentile(values[:,j],1)),
                             q99=float(np.percentile(values[:,j],99)),
                             clipping_fraction=float(np.mean(np.abs(scaled[:,j])>config.CLIP_Z))))
    return rows


def validate_frozen(out, boundaries, scaler, args):
    # No validation loading until both policies exist in a single atomic record.
    decisions = json.loads((out/'frozen_decisions.json').read_text())
    if set(decisions) != set(POLICIES):
        raise ValueError('Both decisions must be frozen')
    x, _, times = load_period(config.TRAIN_FILE, FEATURES, boundaries[1], boundaries[2])
    x = scaler.transform(x); rows = []; cache = {}
    np.save(out/'validation_timestamps.npy', times)
    for policy, decision in decisions.items():
        epoch = decision['epoch']
        if epoch not in cache:
            saved = torch.load(out/f'epoch_{epoch}'/'model.pt', map_location='cpu', weights_only=False)
            model = MultivariateAE(8).to(args.device); model.load_state_dict(saved['model_state_dict'])
            cache[epoch] = score(model, x, args)
        values = cache[epoch]; rule = decision['rule']
        pred = postprocess(values > rule['threshold'], int(rule['merge_gap']), int(rule['min_dur']))
        np.save(out/f'{policy}_validation_score.npy', values)
        np.save(out/f'{policy}_validation_prediction.npy', pred)
        for scope, evaluator in assessment_scopes(times).items():
            rows.append(dict(policy=policy, scope=scope, epoch=epoch,
                             calibration_f05=rule['val_esa_f05'],
                             uninformative_calibration=decision['uninformative_calibration'],
                             **evaluator.score(pred)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--resume', action='store_true'); p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.threads < 1: p.error('Threads must be positive')
    plan = dict(experiment='inner-selection-v1', folds=FOLDS, seeds=[42,43,44], features=FEATURES,
                epochs=5, max_train_windows=25000, batch_size=64, augmentation='short_no_roll',
                device=args.device, threads=args.threads, policies=list(POLICIES),
                quantiles=QUANTILES, merge_gaps=[0,16,64], min_durs=[1,8,32], alarm_cap=.01,
                settings={k:getattr(config,k) for k in ['SEQ_LEN','STRIDE','SCORE_STRIDE','D_MODEL','N_HEADS','N_LAYERS','DROPOUT','LR','WEIGHT_DECAY','GRAD_CLIP','PA_LAMBDA','PA_MARGIN','CLIP_Z']},
                advancement='Fixed policy must improve mean system F0.5 in both folds, with no mean FPe increase in either; ties retain existing policy',
                limits='Historical internal validation; no outer assessment evaluation; six-month calibration differs from outer Q1 protocol; CPU/GPU exact equivalence not guaranteed')
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run: return
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    print('Verifying training CSV integrity', flush=True)
    annotations = verify_inputs(); plan['annotations'] = annotations; plan['train_sha256'] = TRAIN_SHA
    package = Path(__file__).parent
    plan['source_sha256'] = {str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    plan['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__)
    check_protocol(args.output, plan, args.resume)
    all_rows = []
    for fold, boundaries in FOLDS.items():
        folder = args.output/fold; folder.mkdir(exist_ok=True)
        print(fold, 'loading training and calibration only', flush=True)
        raw, labels, times = load_period(config.TRAIN_FILE, FEATURES, None, boundaries[0])
        scaler = RobustChannelScaler().fit(raw, labels)
        ranges = monthly_summary(raw, labels, times, scaler, 'training'); x = scaler.transform(raw); del raw
        raw, cal_labels, cal_times = load_period(config.TRAIN_FILE, FEATURES, boundaries[0], boundaries[1])
        ranges += monthly_summary(raw, cal_labels, cal_times, scaler, 'calibration')
        val = scaler.transform(raw); del raw
        pd.DataFrame(ranges).to_csv(folder/'monthly_inputs.csv', index=False)
        np.save(folder/'calibration_timestamps.npy', cal_times)
        if verify_inputs() != annotations: raise ValueError('Input annotations changed')
        for seed in plan['seeds']:
            out = folder/f'seed{seed}'; out.mkdir(exist_ok=True)
            if (out/'complete.json').exists():
                manifest = json.loads((out/'complete.json').read_text())
                if any(digest(out/name) != sha for name,sha in manifest.items()):
                    raise ValueError('Completed artifacts changed')
                all_rows.extend(pd.read_csv(out/'results.csv').to_dict('records')); continue
            run_args = SimpleNamespace(seed=seed, epochs=5, batch_size=64, device=args.device, resume=args.resume)
            starts = training_starts(labels, 25000, seed)
            np.save(out/'training_starts.npy', starts)
            write_json(out/'preflight.json', dict(training_samples=len(x), calibration_samples=len(val),
                                                 nominal_windows=len(starts), scaler_median=scaler.median_.tolist(),
                                                 scaler_iqr=scaler.iqr_.tolist()))
            train(x, starts, val, cal_times, scaler, out, run_args)
            print(fold, seed, 'both policies frozen; validating', flush=True)
            rows = [dict(fold=fold, seed=seed, **r) for r in validate_frozen(out, boundaries, scaler, run_args)]
            if verify_inputs() != annotations: raise ValueError('Input annotations changed')
            pd.DataFrame(rows).to_csv(out/'results.csv', index=False)
            write_json(out/'complete.json', {str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f.name != 'complete.json'})
            all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    # CSV reloads infer year labels as integers; normalize resumed and new rows.
    frame['fold'] = frame['fold'].astype(str)
    frame.to_csv(args.output/'results.csv', index=False)
    summary = frame.groupby(['fold','policy','scope']).agg(f05_mean=('EW_F_0.50','mean'), f05_sd=('EW_F_0.50','std'), fp_mean=('FPe','mean'), recall_mean=('EW_recall','mean')).reset_index()
    summary.to_csv(args.output/'summary.csv', index=False)
    system = summary[summary.scope == 'system_all_channels'].set_index(['fold','policy'])
    advance = all(system.loc[(f,'fixed_epoch5'),'f05_mean'] > system.loc[(f,'calibration_selected'),'f05_mean'] and system.loc[(f,'fixed_epoch5'),'fp_mean'] <= system.loc[(f,'calibration_selected'),'fp_mean'] for f in FOLDS)
    write_json(args.output/'decision.json', dict(advancement_gate_passed=bool(advance), promoted=False))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
