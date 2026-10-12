"""Read-only diagnosis of frozen internal validation scores; no tuning or training."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.diagnose_channels import digest
from esa_thesis.protocol import evaluator_for
from esa_thesis.thresholds import postprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output directory')
    monthly, events, rules = [], [], []
    for arm, name in [('lambda1', 'inner_selection_cuda_v2'), ('lambda0.1', 'inner_lambda01_cuda_v1')]:
        root = Path('results_longrun/development') / name
        for fold in ('2001', '2002'):
            for seed in (42, 43, 44):
                folder = root / fold / f'seed{seed}'
                manifest = json.loads((folder / 'complete.json').read_text())
                assert all(digest(folder / f) == h for f, h in manifest.items())
                decision = json.loads((folder / 'frozen_decisions.json').read_text())['calibration_selected']
                rule = decision['rule']; threshold = rule['threshold']
                rules.append(dict(arm=arm, fold=fold, seed=seed, epoch=decision['epoch'], **rule))
                times = np.load(folder / 'validation_timestamps.npy')
                score = np.load(folder / 'calibration_selected_validation_score.npy')
                pred = np.load(folder / 'calibration_selected_validation_prediction.npy')
                assert np.isfinite(score).all() and threshold > 0
                np.testing.assert_array_equal(pred, postprocess(score > threshold, int(rule['merge_gap']), int(rule['min_dur'])))
                ev = evaluator_for(times)
                saved = pd.read_csv(folder / 'results.csv')
                row = saved[(saved.policy == 'calibration_selected') & (saved.scope == 'system_all_channels')].iloc[0]
                assert abs(ev.score(pred)['EW_F_0.50'] - row['EW_F_0.50']) < 1e-9
                nominal = np.ones(len(times), bool)
                for lo, hi in ev.all_intervals:
                    nominal &= ~((times >= lo) & (times <= hi))
                months = pd.to_datetime(times, utc=True).strftime('%Y-%m')
                for month in np.unique(months):
                    keep = months == month
                    values = score[keep & nominal]
                    metrics = evaluator_for(times[keep]).score(pred[keep])
                    monthly.append(dict(arm=arm, fold=fold, seed=seed, month=month,
                        nominal_median_ratio=float(np.median(values) / threshold) if len(values) else np.nan,
                        nominal_q99_ratio=float(np.percentile(values, 99) / threshold) if len(values) else np.nan,
                        nominal_above_threshold=float(np.mean(values > threshold)) if len(values) else np.nan,
                        **metrics))
                starts, ends, closed = ev.prediction_intervals(pred)
                for aid, intervals in ev.by_id.items():
                    keep = np.zeros(len(times), bool); hit = False
                    for lo, hi in intervals:
                        keep |= (times >= lo) & (times <= hi)
                        hit |= bool(ev._overlaps(starts, ends, closed, lo, hi).any())
                    values = score[keep]
                    events.append(dict(arm=arm, fold=fold, seed=seed, event_id=aid, detected=hit,
                        maximum_ratio=float(values.max() / threshold) if len(values) else np.nan,
                        median_ratio=float(np.median(values) / threshold) if len(values) else np.nan))
    args.output.mkdir(parents=True)
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(args.output / 'monthly.csv', index=False)
    pd.DataFrame(events).to_csv(args.output / 'events.csv', index=False)
    pd.DataFrame(rules).to_csv(args.output / 'frozen_rules.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, fold in enumerate(('2001', '2002')):
        for arm, style in [('lambda1', '--'), ('lambda0.1', '-')]:
            for seed, color in zip((42, 43, 44), ('tab:blue', 'tab:orange', 'tab:green')):
                d = monthly[(monthly.fold == fold) & (monthly.arm == arm) & (monthly.seed == seed)]
                axes[0, col].plot(d.month.str[-2:], d.nominal_median_ratio, style, color=color, label=f'{arm}, seed {seed}')
                axes[1, col].plot(d.month.str[-2:], d.nominal_false_alarm_rate * 100, style, color=color)
        axes[0, col].axhline(1, color='black', linewidth=.8)
        axes[0, col].set_title(f'Real internal validation: {fold}')
        axes[0, col].set_ylabel('Nominal median score / frozen threshold')
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_ylabel('Nominal time under alarm (%)')
        axes[1, col].set_ylim(-2, 102); axes[1, col].set_xlabel('Month')
    fig.savefig(args.output / 'monthly.png', dpi=150); plt.close(fig)
    (args.output / 'verification.json').write_text(json.dumps(dict(verified_primary_runs=12,
        input='Saved aggregate scores, predictions, timestamps and raw annotations',
        limitations=['Nominal timestamp scores may include nearby annotated windows',
        'Monthly event counts are slices, not additive pooled counts',
        'No per-channel scores saved; channel attribution cannot be inferred',
        'Validation inspected for development; no thresholds or checkpoints changed']), indent=2)+'\n')
    print(monthly[monthly.fold == '2002'][['arm','seed','month','nominal_median_ratio','nominal_false_alarm_rate','FPe']].to_string(index=False))


if __name__ == '__main__':
    main()
