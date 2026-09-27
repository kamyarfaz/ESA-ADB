"""Fixed-checkpoint channels 40–47 diagnostics; training and declared calibration only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import config
from .data import AllWindows, RobustChannelScaler
from .development import load_period
from .models import MultivariateAE
from .protocol import evaluator_for
from .thresholds import postprocess

FEATURES = [f'channel_{i}' for i in range(40, 48)]
TRAIN_SHA = 'fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def training_starts(labels, times, per_year, end_year=2006):
    """Equal quota per calendar year, evenly spaced over eligible window indices."""
    starts = np.arange(0, len(labels)-config.SEQ_LEN+1, config.STRIDE)
    c = np.r_[0, np.cumsum(labels != 0)]
    valid = starts[c[starts+config.SEQ_LEN] == c[starts]]
    years = pd.DatetimeIndex(times).year.to_numpy()
    chosen = []
    for year in range(2000, end_year):
        v = valid[(years[valid] == year) & (years[valid+config.SEQ_LEN-1] == year)]
        if len(v) == 0:
            raise ValueError(f'No nominal training windows for {year}')
        chosen.append(v[np.linspace(0, len(v)-1, min(per_year, len(v)), dtype=int)])
    return np.concatenate(chosen)


@torch.no_grad()
def window_errors(model, x, starts, device, batch_size, stage):
    dataset = AllWindows(x)
    dataset.starts = np.asarray(starts, dtype=np.int64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    result = np.empty((len(starts), x.shape[1]), dtype=np.float32)
    offset = 0
    model.eval()
    began = time.monotonic()
    for batch, inputs in enumerate(loader, 1):
        inputs = inputs.to(device)
        errors = (inputs-model(inputs)).square().squeeze(-1).mean(dim=-1).cpu().numpy()
        if not np.isfinite(errors).all():
            raise ValueError('Nonfinite reconstruction errors')
        result[offset:offset+len(inputs)] = errors
        offset += len(inputs)
        if batch == 1 or batch % 100 == 0 or offset == len(starts):
            print(f'{stage}: {offset}/{len(starts)} windows; {time.monotonic()-began:.0f}s', flush=True)
    return result


def aggregate(errors, starts, length):
    """Match the existing float32 overlapping window-mean calculation."""
    total = np.zeros((errors.shape[1], length), dtype=np.float32)
    count = np.zeros(length, dtype=np.float32)
    for offset in range(0, len(starts), 64):
        s = starts[offset:offset+64]
        ix = (s[:, None]+np.arange(config.SEQ_LEN)[None, :]).ravel()
        np.add.at(count, ix, 1)
        for channel in range(errors.shape[1]):
            np.add.at(total[channel], ix, np.repeat(errors[offset:offset+len(s), channel], config.SEQ_LEN))
    if np.any(count == 0):
        raise ValueError('Uncovered calibration samples; preserve baseline edge policy')
    return total / count[None, :]


def atomic_npz(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f, **arrays)
    temp.replace(path)


def event_diagnostics(evaluator, prediction, scores):
    starts, ends, closed = evaluator.prediction_intervals(prediction)
    rows = []
    for aid, intervals in evaluator.by_id.items():
        hits = np.zeros(len(starts), dtype=bool)
        mask = np.zeros(len(scores), dtype=bool)
        for lo, hi in intervals:
            hits |= evaluator._overlaps(starts, ends, closed, lo, hi)
            mask |= (evaluator.timestamps >= lo) & (evaluator.timestamps <= hi)
        rows.append(dict(event_id=aid, detected=bool(hits.any()), overlapping_alarm_intervals=int(hits.sum()),
                         interval_count=len(intervals), first_time=str(pd.Timestamp(intervals[0, 0], tz='UTC')),
                         last_time=str(pd.Timestamp(intervals[-1, 1], tz='UTC')),
                         samples_inside_raw_intervals=int(mask.sum()),
                         max_score_on_inside_samples=float(scores[mask].max()) if mask.any() else None))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True, help='Completed long-baseline root directory')
    p.add_argument('--fold', choices=['2003','2004','2005'], help='Historical development fold; omit for long baseline')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--windows-per-year', type=int, default=2000)
    p.add_argument('--resume', action='store_true', help='Reuse completed training/calibration inference stages')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if min(a.batch_size, a.windows_per_year) < 1:
        p.error('Batch size and windows per year must be positive')
    original = json.loads((a.source_run/'protocol.json').read_text())
    if original['spec']['features'] != FEATURES:
        p.error('Expected channels 40–47 in recorded order')
    if a.fold:
        train_end, cal_end, _ = original['folds'][a.fold]
        if [train_end,cal_end] != [f'{a.fold}-01-01',f'{a.fold}-04-01']:
            p.error('Unexpected historical fold boundaries')
        source = a.source_run/a.fold
        provenance = 'Historical protocol has no inference-source hashes; current dependencies recorded, strict frozen metric reproduction required'
    else:
        train_end, cal_end = '2006-01-01', '2007-01-01'
        if original['train'] != ['2000-01-01',train_end] or original['calibration'] != [train_end,cal_end]:
            p.error('Expected long-baseline protocol')
        source = a.source_run/'full_period'
        for name in ['models.py','config.py','data.py','evaluation/esa.py','thresholds.py']:
            if digest(Path(__file__).parent/name) != original['source_sha256'][name]:
                raise ValueError(f'Baseline inference dependency changed: {name}')
        provenance = 'Inference dependencies match historical source hashes'
    end_year = int(train_end[:4])
    checkpoint = source/'model_checkpoint.pt'
    frozen = json.loads((source/'frozen_rule.json').read_text())
    print(f"Diagnostic only: checkpoint epoch {frozen['epoch']}; training before {train_end}; calibration through {cal_end}; no assessment", flush=True)
    if a.dry_run:
        print(f'Up to {(end_year-2000)*a.windows_per_year} training windows; full calibration; progress every 100 batches')
        return
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use GPU server')
    print('Verifying training CSV checksum (no assessment file access)...', flush=True)
    if digest(config.TRAIN_FILE) != TRAIN_SHA:
        raise ValueError('Training CSV checksum mismatch; preserve file and investigate')
    stat = config.TRAIN_FILE.stat()
    plan = dict(version='ch40-47-diagnostic-v2', provenance=provenance, source_protocol_sha256=digest(a.source_run/'protocol.json'), features=FEATURES, windows_per_year=a.windows_per_year,
                sampling='Evenly spaced eligible stride-16 windows per year; wholly nominal and inside year',
                device=a.device, batch_size=a.batch_size, checkpoint_sha256=digest(checkpoint),
                frozen_rule_sha256=digest(source/'frozen_rule.json'), train_sha256=TRAIN_SHA,
                sources={str(f.relative_to(Path(__file__).parent)): digest(f) for f in Path(__file__).parent.rglob('*.py')},
                annotations={str(f): digest(f) for f in [config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE]},
                intervals={'train':['2000-01-01',train_end], 'calibration':[train_end,cal_end]},
                limitations='Training window sample, not independent validation. Retrospective. No threshold/model selection or new F0.5 improvement claim.')
    if a.output.exists():
        if not a.resume or json.loads((a.output/'protocol.json').read_text()) != plan:
            raise ValueError('Existing output requires --resume with unchanged protocol/source')
    else:
        a.output.mkdir(parents=True)
        (a.output/'protocol.json').write_text(json.dumps(plan, indent=2))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['features'] != FEATURES:
        raise ValueError('Checkpoint feature order differs')
    model = MultivariateAE(8).to(a.device)
    model.load_state_dict(saved['model_state_dict'])
    scaler = RobustChannelScaler()
    scaler.median_ = np.asarray(saved['scaler_median'], dtype=np.float32)
    scaler.iqr_ = np.asarray(saved['scaler_iqr'], dtype=np.float32)
    if (scaler.median_.shape != (8,) or scaler.iqr_.shape != (8,) or not np.isfinite(scaler.median_).all()
            or not np.isfinite(scaler.iqr_).all() or np.any(scaler.iqr_ <= 0)):
        raise ValueError('Invalid checkpoint scaler')
    train_cache = a.output/'training_window_errors.npz'
    if not train_cache.exists():
        print('Loading nominal training sample...', flush=True)
        x, labels, times = load_period(config.TRAIN_FILE, FEATURES, None, train_end)
        starts = training_starts(labels, times, a.windows_per_year, end_year)
        errors = window_errors(model, scaler.transform(x), starts, a.device, a.batch_size, 'training sample')
        atomic_npz(train_cache, errors=errors, starts=starts, timestamps=times[starts])
        del x, labels, times, errors
    else:
        print('Reusing completed training inference', flush=True)
    cal_cache = a.output/'calibration_window_errors.npz'
    if not cal_cache.exists():
        print(f'Loading calibration {train_end} to {cal_end}...', flush=True)
        x, labels, times = load_period(config.TRAIN_FILE, FEATURES, train_end, cal_end)
        atomic_npz(a.output/'calibration_inputs.npz', values=x, scaled=scaler.transform(x), timestamps=times)
        starts = np.arange(0, len(x)-config.SEQ_LEN+1, config.SCORE_STRIDE)
        errors = window_errors(model, scaler.transform(x), starts, a.device, a.batch_size, 'calibration')
        atomic_npz(cal_cache, errors=errors, starts=starts, timestamps=times, nominal=labels == 0)
        del x, labels, errors
    else:
        print('Reusing completed calibration inference', flush=True)
    with np.load(train_cache) as f:
        training = f['errors']
    with np.load(cal_cache) as f:
        times = f['timestamps']; nominal = f['nominal']; channels = aggregate(f['errors'], f['starts'], len(times))
    score = channels.max(axis=0)
    rule = frozen['rule']
    prediction = postprocess(score > rule['threshold'], int(rule['merge_gap']), int(rule['min_dur']))
    evaluator = evaluator_for(times)
    metrics = evaluator.score(prediction)
    for key in ['EW_F_0.50', 'EW_precision', 'EW_recall', 'TPe', 'FPe', 'FNe']:
        if not np.isclose(metrics[key], rule['val_'+key], rtol=0, atol=1e-6):
            raise ValueError(f'Frozen calibration reproduction failed for {key}: {metrics[key]} vs {rule["val_"+key]}')
    events = event_diagnostics(evaluator, prediction, score)
    labels = pd.read_csv(config.ANNOTATIONS_FILE)
    labels.StartTime = pd.to_datetime(labels.StartTime, utc=True)
    labels.EndTime = pd.to_datetime(labels.EndTime, utc=True)
    local = labels[labels.Channel.isin(FEATURES) & (labels.EndTime >= train_end) & (labels.StartTime < cal_end)]
    events['annotated_on_inputs'] = events.event_id.isin(local.ID)
    events.to_csv(a.output/'calibration_events.csv', index=False)
    winner = channels.argmax(axis=0)
    summary = []
    for j, c in enumerate(FEATURES):
        row = dict(channel=c, train_windows=len(training), nominal_calibration_samples=int(nominal.sum()),
                   dominant_alarm_samples=int(((winner == j) & (prediction > 0)).sum()),
                   dominant_nominal_alarm_samples=int(((winner == j) & (prediction > 0) & nominal).sum()))
        for q in [50, 90, 95, 99, 99.9, 100]:
            row[f'train_window_q{q}'] = float(np.percentile(training[:, j], q))
            row[f'cal_nominal_aggregated_q{q}'] = float(np.percentile(channels[j, nominal], q))
        summary.append(row)
    pd.DataFrame(summary).to_csv(a.output/'channel_error_summary.csv', index=False)
    atomic_npz(a.output/'calibration_channel_scores.npz', scores=channels, timestamps=times, prediction=prediction)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(events), 1, figsize=(12, 2.5*len(events)), squeeze=False, constrained_layout=True)
    for ax, (aid, intervals) in zip(axes.flat, evaluator.by_id.items()):
        lo, hi = intervals[0, 0]-86400*10**9, intervals[-1, 1]+86400*10**9
        k = (times >= lo) & (times <= hi)
        ax.plot(pd.to_datetime(times[k]), score[k], lw=.6)
        ax.axhline(rule['threshold'], color='red', ls='--')
        for start, end in intervals:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='orange', alpha=.3)
        detected = events.loc[events.event_id == aid, 'detected'].iloc[0]
        ax.set_title(f'{aid}: detected={detected}; orange = raw event intervals'); ax.set_ylabel('Score')
    fig.savefig(a.output/'calibration_events.png', dpi=120); plt.close(fig)
    # Inspect every frozen alarm, not just alarms that intersect positives.
    aa, bb, cc = evaluator.prediction_intervals(prediction)
    alarm_rows = []
    with np.load(a.output/'calibration_inputs.npz') as f:
        scaled = f['scaled']
    for index, (start, end) in enumerate(zip(aa, bb)):
        ids = [eid for eid, intervals in evaluator.by_id.items()
               if any(evaluator._overlaps(aa[index:index+1],bb[index:index+1],cc[index:index+1],lo,hi)[0] for lo,hi in intervals)]
        touches = any(evaluator._overlaps(aa[index:index+1],bb[index:index+1],cc[index:index+1],lo,hi)[0] for lo,hi in evaluator.all_intervals)
        inside = (times >= start) & ((times < end) | ((times == end) & cc[index]))
        peaks = channels[:,inside].max(axis=1)
        alarm_rows.append(dict(alarm=index, start=str(pd.Timestamp(start,tz='UTC')), end=str(pd.Timestamp(end,tz='UTC')),
                               false_positive=not touches, event_ids=';'.join(ids),
                               **{c:float(peaks[j]) for j,c in enumerate(FEATURES)}))
        context = (times >= start-4*3600*10**9) & (times <= end+4*3600*10**9)
        fig, axes = plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
        dates=pd.to_datetime(times[context])
        for j,c in enumerate(FEATURES):
            axes[0].plot(dates,channels[j,context],label=c,lw=.7)
            axes[1].plot(dates,scaled[context,j],label=c,lw=.7)
        axes[0].axhline(rule['threshold'],color='red',ls='--')
        for ax in axes:
            ax.axvspan(pd.Timestamp(start),pd.Timestamp(end),alpha=.15,color='red')
        axes[0].set_title(f"Frozen alarm {index}: FP={not touches}; events={','.join(ids)}")
        axes[0].set_ylabel('Channel reconstruction score');axes[0].legend(ncol=4,fontsize=8)
        axes[1].set_ylabel('Prepared input, training-scaled')
        fig.savefig(a.output/f'alarm_{index:02d}.png',dpi=120);plt.close(fig)
    pd.DataFrame(alarm_rows).to_csv(a.output/'calibration_alarms.csv',index=False)

    if (stat.st_size, stat.st_mtime_ns) != (config.TRAIN_FILE.stat().st_size, config.TRAIN_FILE.stat().st_mtime_ns):
        raise ValueError('Training CSV changed during diagnosis')
    print('Rechecking training CSV checksum before accepting diagnostics...', flush=True)
    if digest(config.TRAIN_FILE) != TRAIN_SHA:
        raise ValueError('Training CSV integrity changed during diagnosis; preserve outputs and rerun in a new directory after recovery')
    (a.output/'results.json').write_text(json.dumps(dict(calibration=metrics, checkpoint_epoch=frozen['epoch'],
        reproduction='passed within 1e-6', interpretation='Frozen baseline diagnostics only; no new score selection'), indent=2))
    print(events.to_string(index=False), flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
