"""Fixed calibration-only label influence audit; never adopts diagnostic rules."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from esa_thesis.calibration_audit import (
    detection_ids, digest, load_annotations, period_annotations,
    temporal_components, verify_recorded, without_targets,
)
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.thresholds import postprocess, select_rows


def pick(frame):
    # Match original best-per-epoch, then earliest maximum calibration F0.5.
    choices = [select_rows(rows).iloc[0].to_dict() for _, rows in frame.groupby('epoch', sort=True)]
    return max(choices, key=lambda row: row['val_esa_f05'])


def rule_equal(a, b):
    return (int(a['epoch']) == int(b['epoch']) and
            int(a['merge_gap']) == int(b['merge_gap']) and
            int(a['min_dur']) == int(b['min_dur']) and
            np.isclose(a['threshold'], b['threshold'], rtol=1e-12, atol=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    root = Path(__file__).resolve().parents[2]
    base = root/'results_longrun/development/inner_selection_cuda_v2'
    label_paths = [root/'data/ESA-Mission1'/n for n in ('labels.csv', 'anomaly_types.csv')]
    dependencies = list(label_paths)
    for fold in ('2001', '2002'):
        dependencies.append(base/fold/'calibration_timestamps.npy')
        for seed in (42, 43, 44):
            folder = base/fold/f'seed{seed}'
            dependencies.append(folder/'frozen_decisions.json')
            for epoch in range(1, 6):
                dependencies.extend(folder/f'epoch_{epoch}'/name for name in
                                    ('calibration_candidates.csv', 'calibration_score.npy'))
    before = verify_recorded(root, dependencies)
    labels = load_annotations(*label_paths)
    audit_rows, baseline_rows, influence_rows = [], [], []
    verified_candidates = 0
    for fold in ('2001', '2002'):
        times = np.load(base/fold/'calibration_timestamps.npy', allow_pickle=False)
        if (times[0] != pd.Timestamp(f'{fold}-01-01', tz='UTC').value or
                times[-1] != (pd.Timestamp(f'{fold}-07-01', tz='UTC')-pd.Timedelta(seconds=30)).value):
            raise ValueError('Unexpected calibration period')
        frame = period_annotations(labels, pd.Timestamp(times[0], tz='UTC'),
                                   pd.Timestamp(times[-1]+1, tz='UTC'))
        ev = EventEvaluator(times, frame)
        groups = temporal_components(frame)
        for seed in (42, 43, 44):
            folder = base/fold/f'seed{seed}'
            original = json.loads((folder/'frozen_decisions.json').read_text())['calibration_selected']
            original_flat = dict(original['rule'], epoch=original['epoch'])
            candidates = []
            for epoch in range(1, 6):
                scores = np.load(folder/f'epoch_{epoch}'/'calibration_score.npy', allow_pickle=False)
                if scores.shape != times.shape or not np.isfinite(scores).all():
                    raise ValueError('Invalid calibration scores')
                grid = pd.read_csv(folder/f'epoch_{epoch}'/'calibration_candidates.csv')
                for index, candidate in grid.iterrows():
                    pred = postprocess(scores > candidate.threshold, int(candidate.merge_gap),
                                       int(candidate.min_dur))
                    metrics = ev.score(pred)
                    for key, value in metrics.items():
                        saved = candidate['val_'+key]
                        if isinstance(value, str):
                            if saved != value:
                                raise ValueError('Metric version changed')
                        elif not np.isclose(value, saved, rtol=1e-11, atol=1e-9):
                            raise ValueError(f'Saved grid mismatch {fold}/{seed}/{epoch}/{index}/{key}')
                    if not np.isclose(pred.mean(), candidate.val_pred_anomaly_rate, atol=1e-12, rtol=0):
                        raise ValueError('Alarm fraction mismatch')
                    hits = detection_ids(ev, pred)
                    row = candidate.to_dict()
                    row.update(epoch=epoch, candidate_index=index, hits=';'.join(hits),
                               metrics=metrics)
                    candidates.append(row)
                    verified_candidates += 1
                print(f'verified fold={fold} seed={seed} epoch={epoch}', flush=True)
            full = pd.DataFrame(candidates)
            chosen = pick(full)
            if not rule_equal(chosen, original_flat):
                raise ValueError('Original calibration decision did not reproduce')
            original_hits = set(chosen['hits'].split(';'))- {''}
            baseline_rows.append(dict(fold=fold, seed=seed, epoch=int(chosen['epoch']),
                                      threshold=chosen['threshold'], f05=chosen['val_esa_f05'],
                                      detected_ids=chosen['hits'],
                                      detected_components=sum(bool(original_hits & set(g)) for g in groups),
                                      components=len(groups), target_ids=len(ev.by_id)))
            for component, ids in enumerate(groups):
                modified = []
                for candidate in candidates:
                    hits = set(candidate['hits'].split(';')) - {''}
                    values = without_targets(candidate['metrics'], hits, ids, ev.by_id)
                    if values is None:
                        break
                    row = {k:v for k,v in candidate.items() if k!='metrics'}
                    row.update({f'val_{k}':v for k,v in values.items()})
                    row['val_esa_f05'] = values['EW_F_0.50']
                    modified.append(row)
                if not modified:
                    influence_rows.append(dict(fold=fold, seed=seed, component=component,
                                               withheld_ids=';'.join(ids), no_remaining_targets=True))
                    continue
                reduced = pd.DataFrame(modified)
                new = pick(reduced)
                hits = set(new['hits'].split(';')) - {''}
                influential = bool(original_hits & set(ids))
                remaining_orig_hits = original_hits-set(ids)
                influence_rows.append(dict(
                    fold=fold, seed=seed, component=component, withheld_ids=';'.join(ids),
                    no_remaining_targets=False, contained_original_detection=influential,
                    original_remaining_detected_ids=';'.join(sorted(remaining_orig_hits)),
                    original_remaining_f05=without_targets(chosen['metrics'], original_hits, ids,
                                                           ev.by_id)['EW_F_0.50'],
                    epoch=int(new['epoch']), threshold=new['threshold'],
                    threshold_ratio_to_original=new['threshold']/chosen['threshold'],
                    merge_gap=int(new['merge_gap']), min_dur=int(new['min_dur']),
                    rule_changed=not rule_equal(chosen, new), epoch_changed=new['epoch']!=chosen['epoch'],
                    remaining_calibration_f05=new['val_esa_f05'],
                    all_candidates_zero=bool((reduced.val_esa_f05==0).all()),
                    all_eligible_candidates_zero=bool((reduced.loc[
                        (reduced.val_pred_anomaly_rate<=.01) & (~reduced.val_saturated),
                        'val_esa_f05']==0).all()),
                    retained_detected_ids=';'.join(sorted(hits-set(ids))),
                    withheld_detected_ids=';'.join(sorted(hits&set(ids))),
                    remaining_false_events=int(new['val_FPe']),
                    remaining_false_positive_seconds=new['val_false_positive_seconds']))
            audit_rows.extend(dict(fold=fold, seed=seed, **{k:v for k,v in row.items() if k!='metrics'})
                              for row in candidates)
    if verify_recorded(root, dependencies) != before:
        raise ValueError('Sources changed during audit')
    args.output.mkdir(parents=True)
    for name, rows in [('baseline', baseline_rows), ('influence', influence_rows),
                       ('verified_candidates', audit_rows)]:
        pd.DataFrame(rows).to_csv(args.output/f'{name}.csv', index=False)
    (args.output/'verification.json').write_text(json.dumps(dict(
        input_sha256=before, verified_candidates=verified_candidates, reproduced_decisions=6,
        script_sha256=digest(Path(__file__)),
        module_sha256=digest(root/'esa_thesis/calibration_audit.py'),
        plan_sha256=digest(root/'docs/research/calibration-consolidation-plan-2026-10-13.md'),
        limits='Calibration-only label influence. Shared nominal background and grid: NOT cross-validation. '
               'No validation/test score arrays, weights or telemetry loaded; no detector adopted.'), indent=2)+'\n')
    print(pd.DataFrame(baseline_rows).to_string(index=False))
    print(pd.DataFrame(influence_rows).groupby('fold')[['rule_changed', 'epoch_changed',
                                                     'all_candidates_zero']].sum().to_string())


if __name__ == '__main__':
    main()
