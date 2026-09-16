"""Paired, same-weights comparison of window-mean and per-timestep AE errors.

Both modes are retrospective. Thresholds are selected separately on validation;
test scores never choose a threshold or the preferred scoring method.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .. import config
from ..data import AllWindows, RobustChannelScaler
from ..models import MultivariateAE
from ..protocol import read_timestamps, validation_split, evaluator_for, manifest as source_manifest
from ..thresholds import sweep_thresholds, select_rows, postprocess
from .esa import METRIC_VERSION, timestamps_ns


@torch.no_grad()
def paired_scores(model, x, *, device='cpu', batch_size=64):
    """One forward pass feeds both aggregations; neither mode changes the input."""
    dataset = AllWindows(x)
    if not len(dataset):
        raise ValueError('Not enough observations for a complete window')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    channels, length = x.shape[1], len(x)
    window = np.zeros((channels, length), dtype=np.float32)
    point = np.zeros_like(window)
    count = np.zeros(length, dtype=np.float32)
    model.eval()
    offset = 0
    for inputs in loader:
        inputs = inputs.to(device)
        errors = (inputs - model(inputs)).square().squeeze(-1)
        means = errors.mean(dim=-1).cpu().numpy()
        errors = errors.cpu().numpy()
        starts = dataset.starts[offset:offset+len(inputs)]
        indices = (starts[:, None] + np.arange(config.SEQ_LEN)[None, :]).ravel()
        np.add.at(count, indices, 1)
        for channel in range(channels):
            np.add.at(window[channel], indices, np.repeat(means[:, channel], config.SEQ_LEN))
            np.add.at(point[channel], indices, errors[:, channel, :].ravel())
        offset += len(inputs)
    denominator = np.maximum(count, 1)[None, :]
    # Preserve the legacy edge policy for this controlled comparison. Report it.
    return {'window_mean': (window / denominator).max(axis=0),
            'per_timestep': (point / denominator).max(axis=0)}, int((count == 0).sum())


def load_features(path, features, expected_times, *, start=None):
    frames, observed = [], []
    for frame in pd.read_csv(path, usecols=['timestamp', *features], chunksize=100000):
        times = pd.to_datetime(frame.pop('timestamp'), utc=True).dt.as_unit('ns').astype('int64')
        keep = np.ones(len(times), dtype=bool) if start is None else times.to_numpy() >= start
        if keep.any():
            frames.append(frame.loc[keep, features])
            observed.append(times.to_numpy()[keep])
    if not frames or not np.array_equal(np.concatenate(observed), expected_times):
        raise ValueError('Feature rows and expected timestamps do not align')
    frame = pd.concat(frames, ignore_index=True)
    # Match existing inference preprocessing; do not silently change causality here.
    for name in features:
        frame[name] = pd.to_numeric(frame[name], errors='raise').ffill().bfill().fillna(0)
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite model input')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True, help='Directory containing model_checkpoint.pt')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=config.DEVICE)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    checkpoint_path = args.source_run / 'model_checkpoint.pt'
    if not checkpoint_path.is_file():
        parser.error(f'Missing {checkpoint_path}')
    if args.output.exists():
        parser.error('Output exists; choose a new directory')
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    if args.dry_run:
        print('Checkpoint:', checkpoint_path.resolve())
        print('Output:', args.output.resolve())
        print('Device:', args.device, '| batch:', args.batch_size)
        print('Compare window-mean and per-timestep scores; validation-only calibration; no training.')
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run on the GPU server or explicitly choose --device cpu')
    # Only load trusted local checkpoints produced by this project.
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    features = saved['features']
    model = MultivariateAE(len(features))
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(args.device)
    scaler = RobustChannelScaler()
    scaler.median_ = np.asarray(saved['scaler_median'], dtype=np.float32)
    scaler.iqr_ = np.asarray(saved['scaler_iqr'], dtype=np.float32)
    if scaler.median_.shape != (len(features),) or scaler.iqr_.shape != (len(features),):
        raise ValueError('Checkpoint scaler does not match feature list')
    train_times = read_timestamps(config.TRAIN_FILE)
    val_times = train_times[validation_split(train_times):]
    evaluator = evaluator_for(val_times)
    args.output.mkdir(parents=True)
    manifest = {'metric_version': METRIC_VERSION, 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                'provenance': source_manifest({'experiment': 'paired_scoring', 'features': features}),
                'features': features, 'validation': evaluator.metadata(), 'device': args.device,
                'batch_size': args.batch_size, 'context': config.SEQ_LEN, 'stride': config.SCORE_STRIDE,
                'quantiles': [90,95,97.5,98,99,99.5,99.75,99.9,99.95,99.99,100],
                'merge_gaps': [0,16,64], 'min_durs': [1,8,32],
                'limitations': 'Retrospective scores; old weights and historical test exposure; uncovered edge samples remain zero in both modes.'}
    (args.output/'protocol.json').write_text(json.dumps(manifest, indent=2))
    print('Scoring validation in one paired forward pass...', flush=True)
    x = scaler.transform(load_features(config.TRAIN_FILE, features, val_times, start=int(val_times[0])))
    scores, uncovered = paired_scores(model, x, device=args.device, batch_size=args.batch_size)
    del x
    rules = {}
    for mode, score in scores.items():
        grid = sweep_thresholds(score, evaluator=evaluator, threshold_values=np.percentile(score, manifest['quantiles']),
                                merge_gaps=manifest['merge_gaps'], min_durs=manifest['min_durs'])
        rules[mode] = select_rows(grid).iloc[0].to_dict()
        grid.to_csv(args.output/(mode+'_validation_candidates.csv'), index=False)
        np.save(args.output/(mode+'_val_score.npy'), score)
    # Record the preferred mode using validation only, before reading test data.
    preferred = min(rules, key=lambda m: (-rules[m]['val_esa_f05'], rules[m]['val_false_positive_seconds'], m))
    (args.output/'frozen_rules.json').write_text(json.dumps({'rules':rules, 'validation_preferred_mode':preferred,
                                                           'uncovered_validation_samples':uncovered}, indent=2))
    del scores
    print('Frozen rules. Scoring test...', flush=True)
    test_times = read_timestamps(config.TEST_FILE)
    evaluator = evaluator_for(test_times)
    x = scaler.transform(load_features(config.TEST_FILE, features, test_times))
    scores, uncovered = paired_scores(model, x, device=args.device, batch_size=args.batch_size)
    results = []
    for mode, score in scores.items():
        rule = rules[mode]
        pred = postprocess(score > rule['threshold'], int(rule['merge_gap']), int(rule['min_dur']))
        result = {'mode':mode, 'validation_preferred':mode==preferred, **rule,
                  **{'test_'+k:v for k,v in evaluator.score(pred).items()}, 'uncovered_test_samples':uncovered}
        results.append(result)
        np.save(args.output/(mode+'_test_score.npy'), score)
        np.save(args.output/(mode+'_test_pred.npy'), pred)
    pd.DataFrame(results).to_csv(args.output/'results.csv', index=False)
    print(pd.DataFrame(results)[['mode','validation_preferred','val_esa_f05','test_EW_F_0.50']].to_string(index=False))


if __name__ == '__main__':
    main()
