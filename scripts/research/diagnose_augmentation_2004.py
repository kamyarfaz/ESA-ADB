"""Read-only matched checkpoint diagnosis on pre-2004 training and Q1 calibration.

No assessment arrays, threshold search, training, or checkpoint changes. Run from
the repository root with PYTHONPATH=.; use a new output directory each time.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esa_thesis import config
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import (
    FEATURES, TRAIN_SHA, aggregate, atomic_npz, digest, event_diagnostics,
    training_starts, window_errors,
)
from esa_thesis.models import MultivariateAE
from esa_thesis.protocol import evaluator_for
from esa_thesis.thresholds import postprocess, select_rows


def whole_nominal(nominal, starts, length=256):
    counts = np.r_[0, np.cumsum(~np.asarray(nominal, dtype=bool))]
    return counts[starts + length] == counts[starts]


def quantiles(values):
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite values')
    return {f'q{q}': float(np.percentile(values, q)) for q in [50, 90, 95, 99, 99.9, 100]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--comparison', choices=['short-coordinated', 'short-no-roll'],
                   default='short-coordinated')
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output directory; preserve earlier diagnostics')
    torch.set_num_threads(a.threads)
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    modes = (['legacy', 'short_coordinated'] if a.comparison == 'short-coordinated'
             else ['short_coordinated', 'short_no_roll'])
    campaign = a.comparison.replace('-', '_')
    arms = [(seed, mode) for seed in [42, 43, 44] for mode in modes]
    roots = {pair: Path(f'results_longrun/development/{campaign}_seed{pair[0]}_v1') for pair in arms}
    files = [config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE]
    protocols = {}
    for pair, root in roots.items():
        protocols[pair] = json.loads((root/'protocol.json').read_text())
        assert protocols[pair]['features'] == FEATURES
        assert protocols[pair]['folds']['2004'][:2] == ['2004-01-01', '2004-04-01']
        files += [root/'protocol.json', root/'2004'/'training_starts.npy']
        files += [root/'2004'/pair[1]/name for name in
                  ['model_checkpoint.pt', 'frozen_rule.json', 'history.csv', 'calibration_candidates.csv']]
        for name in ['models.py', 'data.py', 'config.py', 'diagnose_channels.py',
                     'development.py', 'thresholds.py', 'evaluation/esa.py', 'protocol.py']:
            assert digest(Path('esa_thesis')/name) == protocols[pair]['source_sha256'][name], name
        for filename, sha in protocols[pair]['annotations'].items():
            assert digest(filename) == sha
    hashes = {str(f): digest(f) for f in sorted(set(files))}
    plan = dict(status='running', comparison=a.comparison, device=a.device, batch_size=a.batch_size, threads=a.threads,
                checkpoints='Saved calibration-selected checkpoints only', inputs=hashes,
                training_sample='256 evenly spaced wholly nominal stride16 windows per year 2000-2003; identical probe for all models',
                calibration='All samples in [2004-01-01,2004-04-01); stride32 length256 window-mean scoring',
                policy='No assessment read, no training, no threshold/model selection or changes',
                train_sha256=TRAIN_SHA, script_sha256=digest(__file__),
                limitations='In-sample nominal probes; overlapping windows dependent; CPU/GPU rounding may affect exact threshold boundaries; descriptive diagnosis, not causality or independent validation')
    a.output.mkdir(parents=True)
    (a.output/'protocol.json').write_text(json.dumps(plan, indent=2)+'\n')
    print('Verifying training CSV before reading; no assessment access', flush=True)
    assert digest(config.TRAIN_FILE) == TRAIN_SHA
    x, labels, times = load_period(config.TRAIN_FILE, FEATURES, None, '2004-04-01')
    split = np.searchsorted(times, pd.Timestamp('2004-01-01', tz='UTC').value)
    train_x, train_y, train_times = x[:split], labels[:split], times[:split]
    cal_x, cal_y, cal_times = x[split:], labels[split:], times[split:]
    starts = training_starts(train_y, train_times, 256, 2004)
    train_windows = np.stack([train_x[s:s+config.SEQ_LEN] for s in starts])
    sample_times = train_times[starts]
    sample_years = pd.DatetimeIndex(sample_times).year.to_numpy()
    cal_starts = np.arange(0, len(cal_x)-config.SEQ_LEN+1, config.SCORE_STRIDE)
    clean = whole_nominal(cal_y == 0, cal_starts)
    months = pd.DatetimeIndex(cal_times).month.to_numpy()
    window_months = months[cal_starts]
    same_month = window_months == months[cal_starts+config.SEQ_LEN-1]
    ev = evaluator_for(cal_times)
    atomic_npz(a.output/'diagnostic_inputs.npz', training_windows=train_windows,
               training_timestamps=sample_times, training_starts=starts,
               calibration_values=cal_x, calibration_labels=cal_y, calibration_timestamps=cal_times)
    # Verify each seed's original sample remains strictly nominal and within training.
    for seed in [42, 43, 44]:
        used = np.load(roots[(seed, modes[0])]/'2004'/'training_starts.npy')
        assert len(used) == 25000 and np.all(used >= 0) and np.all(used+256 <= len(train_y))
        assert whole_nominal(train_y == 0, used).all()
    print('Loaded training probe and full calibration; checking CSV after read', flush=True)
    assert digest(config.TRAIN_FILE) == TRAIN_SHA
    del x, labels, times, train_x, train_y, train_times
    rows, metrics_rows, channel_rows, event_rows, alarm_rows, history_rows = [], [], [], [], [], []
    errors_all, channel_scores, checkpoints, rules = {}, {}, {}, {}
    reference_scaler = None
    for seed, mode in arms:
        key = (seed, mode); source = roots[key]/'2004'/mode
        saved = torch.load(source/'model_checkpoint.pt', map_location='cpu', weights_only=False)
        assert saved['features'] == FEATURES and saved['augmentation'] == mode
        scaler = RobustChannelScaler()
        scaler.median_ = np.asarray(saved['scaler_median'], dtype=np.float32)
        scaler.iqr_ = np.asarray(saved['scaler_iqr'], dtype=np.float32)
        if reference_scaler is None:
            reference_scaler = (scaler.median_.copy(), scaler.iqr_.copy())
        else:
            np.testing.assert_array_equal(scaler.median_, reference_scaler[0])
            np.testing.assert_array_equal(scaler.iqr_, reference_scaler[1])
        assert np.isfinite(scaler.median_).all() and np.isfinite(scaler.iqr_).all() and np.all(scaler.iqr_ > 0)
        frozen = json.loads((source/'frozen_rule.json').read_text()); rule = frozen['rule']; rules[key] = rule
        h = pd.read_csv(source/'history.csv')
        assert frozen['epoch'] == int(h.loc[h.calibration_f05.idxmax(), 'epoch'])
        assert np.allclose(h.loss, h.normal_mse+h.pseudo_penalty, rtol=0, atol=1e-6)
        selected = select_rows(pd.read_csv(source/'calibration_candidates.csv')).iloc[0]
        for col in ['threshold', 'merge_gap', 'min_dur', 'val_esa_f05']:
            assert np.isclose(selected[col], rule[col], rtol=0, atol=1e-12)
        history_rows.extend(h.assign(seed=seed, model=mode).to_dict('records'))
        model = MultivariateAE(8).to(a.device); model.load_state_dict(saved['model_state_dict']); model.eval()
        checkpoints[key] = saved
        train_flat = scaler.transform(train_windows.reshape(-1, 8))
        train_error = window_errors(model, train_flat, np.arange(len(starts))*256,
                                    a.device, a.batch_size, f'{seed}/{mode} training')
        scaled = scaler.transform(cal_x)
        errors = window_errors(model, scaled, cal_starts, a.device, a.batch_size, f'{seed}/{mode} calibration')
        errors_all[key] = errors
        channels = aggregate(errors, cal_starts, len(cal_x)); channel_scores[key] = channels
        score = channels.max(axis=0)
        prediction = postprocess(score > rule['threshold'], int(rule['merge_gap']), int(rule['min_dur']))
        metrics = ev.score(prediction)
        # Record any device boundary differences explicitly; never adjust frozen thresholds.
        differences = {k: float(metrics[k])-float(rule['val_'+k]) for k in
                       ['EW_F_0.50','EW_precision','EW_recall','TPe','FPe','FNe','false_positive_seconds']}
        reproduced = all(abs(differences[k]) <= 1e-6 for k in differences)
        metrics_rows.append(dict(seed=seed, model=mode, epoch=frozen['epoch'], threshold=rule['threshold'],
                                 frozen_calibration_f05=rule['val_esa_f05'], reproduced=reproduced, **metrics))
        dest = a.output/f'{seed}_{mode}'; dest.mkdir()
        atomic_npz(dest/'errors.npz', training=train_error, calibration=errors,
                   starts=cal_starts, scores=channels, prediction=prediction)
        (dest/'reproduction.json').write_text(json.dumps(dict(reproduced=reproduced, differences=differences), indent=2)+'\n')
        def summarize(period, kind, e):
            for j, name in enumerate(FEATURES+['max_across_channels']):
                values = e[:, j] if j < 8 else e.max(axis=1)
                rows.append(dict(seed=seed, model=mode, period=period, kind=kind, channel=name,
                                 windows=len(values), **quantiles(values)))
        summarize('all_training', 'sampled_nominal_window', train_error)
        for year in range(2000, 2004):
            summarize(str(year), 'sampled_nominal_window', train_error[sample_years == year])
        winner = channels.argmax(axis=0)
        for month in [1, 2, 3]:
            keep = clean & same_month & (window_months == month)
            summarize(f'2004-{month:02d}', 'full_nominal_window', errors[keep])
            nominal = (months == month) & (cal_y == 0)
            for j, name in enumerate(FEATURES):
                channel_rows.append(dict(seed=seed, model=mode, month=month, channel=name,
                    nominal_samples=int(nominal.sum()), dominant_nominal_samples=int(((winner == j)&nominal).sum()),
                    dominant_nominal_alarm_samples=int(((winner == j)&nominal&(prediction > 0)).sum()),
                    nominal_score_above_threshold_samples=int(((channels[j] > rule['threshold'])&nominal).sum()),
                    aggregated_nominal_q99=float(np.percentile(channels[j, nominal], 99)),
                    aggregated_nominal_max=float(channels[j, nominal].max())))
        events = event_diagnostics(ev, prediction, score)
        events['seed'] = seed; events['model'] = mode
        events['peak_threshold_ratio'] = events.max_score_on_inside_samples/rule['threshold']
        event_rows.extend(events.to_dict('records'))
        aa, bb, cc = ev.prediction_intervals(prediction)
        for idx, (lo, hi) in enumerate(zip(aa, bb)):
            inside = (cal_times >= lo)&(cal_times < hi)
            touches = any(ev._overlaps(aa[idx:idx+1], bb[idx:idx+1], cc[idx:idx+1], l, u)[0] for l, u in ev.all_intervals)
            dominant = np.bincount(winner[inside], minlength=8).argmax()
            alarm_rows.append(dict(seed=seed, model=mode, start=str(pd.Timestamp(lo, tz='UTC')),
                end=str(pd.Timestamp(hi, tz='UTC')), false_event=not touches,
                dominant_channel=FEATURES[dominant], peak=float(score[inside].max())))
        print(seed, mode, 'frozen calibration reproduction', reproduced, metrics['EW_F_0.50'], flush=True)
    for name, data in [('window_quantiles', rows), ('calibration_metrics', metrics_rows),
                       ('channel_summary', channel_rows), ('calibration_events', event_rows),
                       ('calibration_alarms', alarm_rows), ('history', history_rows)]:
        pd.DataFrame(data).to_csv(a.output/f'{name}.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = dict(zip(modes, ['#285780', '#C16B30']))
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(3, 1, figsize=(11,9), constrained_layout=True)
    for ax, seed in zip(axes, [42,43,44]):
        for mode in modes:
            q = frame[(frame.seed == seed)&(frame.model == mode)&(frame.channel == 'max_across_channels')&(frame.period != 'all_training')]
            for quantile, style in [('q50','--'),('q99','-')]:
                ax.plot(q.period, q[quantile], style, marker='o', color=colors[mode], label=f'{mode} {quantile}')
        ax.set_title(f'Seed {seed}: identical nominal probe, saved selected checkpoints')
        ax.set_ylabel('Window MSE'); ax.set_yscale('log'); ax.legend(ncol=2, fontsize=8)
    fig.savefig(a.output/'nominal_distributions.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for ax, seed in zip(axes,[42,43,44]):
        for mode in modes:
            # Preserve extrema in every 32-sample bin for plotting, not scoring.
            ratio = channel_scores[(seed,mode)].max(axis=0)/rules[(seed,mode)]['threshold']
            ax.plot(pd.to_datetime(cal_times[::32]), ratio.reshape(-1,32).max(axis=1),
                    color=colors[mode], lw=.7, label=mode)
        for intervals in ev.by_id.values():
            for lo,hi in intervals: ax.axvspan(pd.Timestamp(lo),pd.Timestamp(hi),color='gold',alpha=.2)
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set_title(f'Seed {seed}');ax.set_ylabel('Score / frozen threshold');ax.legend()
    fig.savefig(a.output/'calibration_scores.png',dpi=150);plt.close(fig)
    # Diagnostic examples: pick one high-error nominal window per month using
    # the mean within-model percentile rank, so the six arms share each example.
    # Error-based selection is descriptive and not a general-performance sample.
    rank = np.mean([pd.Series(e.max(axis=1)).rank(pct=True).to_numpy() for e in errors_all.values()],axis=0)
    probes = []
    for month in [1,2,3]:
        candidates = np.flatnonzero(clean&same_month&(window_months == month))
        probes.append((f'nominal_month_{month:02d}', int(candidates[np.argmax(rank[candidates])])) )
    event_mask = np.zeros(len(cal_times),bool)
    for intervals in ev.by_id.values():
        for lo,hi in intervals:event_mask |= (cal_times >= lo)&(cal_times <= hi)
    counts = np.r_[0,np.cumsum(event_mask)]
    event_candidates = np.flatnonzero(counts[cal_starts+256]-counts[cal_starts] > 0)
    probes.append(('annotated_event',int(event_candidates[np.argmax(rank[event_candidates])])) )
    probe_rows=[]
    for name, index in probes:
        start = cal_starts[index]
        # Pick one common channel using each model's share of this window's MSE.
        shares = np.mean([e[index]/max(e[index].sum(),1e-12) for e in errors_all.values()],axis=0)
        ch = int(np.argmax(shares))
        fig, axes = plt.subplots(3,2,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
        for ax, key in zip(axes.flat,arms):
            model=MultivariateAE(8).to(a.device);model.load_state_dict(checkpoints[key]['model_state_dict']);model.eval()
            scaler=RobustChannelScaler();scaler.median_=reference_scaler[0];scaler.iqr_=reference_scaler[1]
            values=scaler.transform(cal_x[start:start+256]);tensor=torch.from_numpy(values.T[None,:,:,None].copy()).to(a.device)
            with torch.no_grad(): reconstructed=model(tensor).cpu().numpy()[0,:,:,0].T
            probe_rows.append(dict(probe=name, seed=key[0], model=key[1], channel=FEATURES[ch],
                start=str(pd.Timestamp(cal_times[start],tz='UTC')), channel_mse=float(np.mean((values[:,ch]-reconstructed[:,ch])**2))))
            ax.plot(np.arange(256)/2,values[:,ch],color='black',lw=.9,label='Observed')
            ax.plot(np.arange(256)/2,reconstructed[:,ch],color=colors[key[1]],lw=.8,label='Reconstructed')
            ax.set_title(f'{key[0]} {key[1]}');ax.set_ylabel('Training-scaled value');ax.set_xlabel('Minutes');ax.legend(fontsize=8)
        fig.suptitle(f'{name}: {FEATURES[ch]}, {pd.Timestamp(cal_times[start])}; same window and channel for all models')
        fig.savefig(a.output/f'{name}.png',dpi=150);plt.close(fig)
    pd.DataFrame(probe_rows).to_csv(a.output/'reconstruction_probes.csv',index=False)
    assert hashes == {str(f):digest(f) for f in sorted(set(files))}
    plan.update(status='complete', source_hashes_stable=True,
                all_frozen_calibrations_reproduced=all(r['reproduced'] for r in metrics_rows),
                output_sha256={f.name:digest(f) for f in a.output.glob('*.csv')},
                calibration_event_ids=list(ev.by_id), training_probe_windows=len(starts), calibration_windows=len(cal_starts))
    (a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(pd.DataFrame(metrics_rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
