"""Validation-only recalibration of legacy cached AE or MLP scores.

Does not train, alter source runs, or select a model on test scores. Existing
weights were selected with a legacy metric: outputs remain diagnostic baselines.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config
from ..metrics import metrics
from ..paths import default_run_dir
from ..protocol import read_timestamps, validation_split, evaluator_for
from ..thresholds import sweep_thresholds, select_rows, postprocess
from .esa import METRIC_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=default_run_dir())
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are rejected')
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--score-kind', choices=['ae', 'mlp'], default='ae')
    parser.add_argument('--validation-window', choices=['benchmark', 'cached'], default='benchmark')
    args = parser.parse_args()
    summaries = sorted(args.source.glob('*/summary.json'))
    available = {p.parent.name for p in summaries}
    if args.runs:
        unknown = set(args.runs) - available
        if unknown:
            parser.error('Unknown runs: ' + ', '.join(sorted(unknown)))
        summaries = [p for p in summaries if p.parent.name in args.runs]
    if not summaries:
        parser.error('No saved run summaries found')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    print('Reading and checking dataset timestamps...', flush=True)
    train_times = read_timestamps(config.TRAIN_FILE)
    test_times = read_timestamps(config.TEST_FILE)
    test_evaluator = evaluator_for(test_times)
    split = validation_split(train_times)
    args.output.mkdir(parents=True)
    prefix = '' if args.score_kind == 'ae' else 'mlp_'
    quantiles = [90, 95, 97.5, 98, 99, 99.5, 99.75, 99.9, 99.95, 99.99, 100]
    metadata = {'metric_version': METRIC_VERSION, 'source': str(args.source.resolve()),
                'score_kind': args.score_kind, 'validation_window': args.validation_window,
                'quantiles': quantiles, 'merge_gaps': [0, 16, 64], 'min_durs': [1, 8, 32],
                'validation_predicted_rate_cap': .01, 'test': test_evaluator.metadata(),
                'limitation': 'Legacy weights/checkpoints and historical test exposure; diagnostic recalibration, not fresh benchmark training',
                'label_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE)}}
    (args.output / 'protocol.json').write_text(json.dumps(metadata, indent=2))
    combined = []
    for summary_path in summaries:
        source = summary_path.parent
        summary = json.loads(summary_path.read_text())
        val = np.load(source / (prefix + 'val_score.npy'), mmap_mode='r')
        yval = np.load(source / 'val_y_true.npy', mmap_mode='r')
        if len(val) != summary['num_val_rows'] or summary['num_train_rows'] + len(val) != len(train_times):
            raise ValueError(f'Cached validation alignment mismatch: {source}')
        times = train_times[-len(val):]
        offset = int(np.searchsorted(times, train_times[split])) if args.validation_window == 'benchmark' else 0
        val, yval, times = val[offset:], yval[offset:], times[offset:]
        val_evaluator = evaluator_for(times)
        grid = sweep_thresholds(val, evaluator=val_evaluator,
                                threshold_values=np.percentile(val, quantiles),
                                merge_gaps=metadata['merge_gaps'], min_durs=metadata['min_durs'])
        row = select_rows(grid).iloc[0].to_dict()
        # Freeze the rule before opening test scores.
        target = args.output / source.name
        target.mkdir()
        grid.to_csv(target / 'validation_candidates.csv', index=False)
        (target / 'frozen_rule.json').write_text(json.dumps(row, indent=2))
        test = np.load(source / (prefix + 'test_score.npy'), mmap_mode='r')
        ytest = np.load(source / 'test_y_true.npy', mmap_mode='r')
        if len(test) != len(test_times) or len(ytest) != len(test_times):
            raise ValueError(f'Cached test alignment mismatch: {source}')
        if not np.isfinite(test).all():
            raise ValueError(f'Non-finite test scores: {source}')
        parameters = [int(row['merge_gap']), int(row['min_dur'])]
        vp = postprocess(val > row['threshold'], *parameters)
        tp = postprocess(test > row['threshold'], *parameters)
        row.update({'run': source.name, 'score_kind': args.score_kind})
        row.update({'val_' + k: v for k, v in metrics(yval, vp, evaluator=val_evaluator).items()})
        row.update({'test_' + k: v for k, v in metrics(ytest, tp, evaluator=test_evaluator).items()})
        row['original_legacy_test_f05'] = summary['plot_selection']['test_esa_f05']
        pd.DataFrame([row]).to_csv(target / 'result.csv', index=False)
        np.save(target / 'test_pred_mask.npy', tp)
        np.save(target / 'val_pred_mask.npy', vp)
        provenance = {'validation': val_evaluator.metadata(), 'source_summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                      'source_scores': {name: {'size': (source/name).stat().st_size, 'mtime_ns': (source/name).stat().st_mtime_ns}
                                        for name in (prefix+'val_score.npy', prefix+'test_score.npy')}}
        (target / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        combined.append(row)
        pd.DataFrame(combined).sort_values('run').to_csv(args.output / 'results.csv', index=False)
        print(f"{source.name}: validation={row['val_esa_f05']:.4f}, test={row['test_esa_f05']:.4f}", flush=True)
    print(f'Wrote {len(combined)} diagnostic results to {args.output}', flush=True)


if __name__ == '__main__':
    main()
