"""Frozen calibration-first target-output and boundary study. Never trains models."""
import fcntl
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.calibration_audit import digest
from esa_thesis.development_recovery import check_protocol
from esa_thesis.diagnose_channels import aggregate
from esa_thesis.full_combinations import load_model, inputs_at
from esa_thesis.full_short_no_roll import verify_frozen, report_predictions
from esa_thesis.full_similarity_mosad import write_json, save_npy
from esa_thesis.masked_pilot import assessment_scopes
from esa_thesis.runtime import set_seed
from scripts.research.target_boundary_common import (
    ROOT, SOURCE, CALIBRATION, OUTPUT, PROTOCOL, TARGETS, METHODS, ARMS, FAMILIES,
    QUANTILES, segments, detected_ids, nominal_mask, boundary_candidates, localize,
    family_candidates, coarse_transform, choose, candidate_metrics)

BATCH = 64
CHUNK_WINDOWS = 4096
SCRIPT = Path(__file__).resolve()
COMMON = SCRIPT.with_name('target_boundary_common.py')


def read(path):
    return json.loads(path.read_text())


def times_for(split):
    start, end = ('2006-01-01', '2007-01-01') if split == 'calibration' else ('2007-01-01', '2014-01-01')
    return pd.date_range(start, end, freq='30s', inclusive='left').asi8


def check_files(folder, receipt):
    for name, expected in receipt['files'].items():
        if digest(folder / name) != expected:
            raise ValueError('Artifact changed: ' + str(folder / name))


def verify_inputs():
    print('Checking destination frozen inputs and predecessor receipts', flush=True)
    verify_frozen(ROOT)
    prior = read(SOURCE / 'protocol.json')
    for name, expected in prior['source_sha256'].items():
        assert digest(ROOT / 'esa_thesis' / name) == expected, name
    for group in ('annotations', 'reference'):
        for name, expected in prior[group].items():
            assert digest(ROOT / name) == expected, name
    for folder in (SOURCE, CALIBRATION):
        check_files(folder, read(folder / 'complete.json'))
    assert prior['settings'] == {k: str(v) for k, v in vars(config).items() if k.isupper()}
    for k, v in dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda).items():
        assert prior['runtime'][k] == v, (k, prior['runtime'][k], v)
    return prior


def load_features(split, features):
    """Read numerical inputs/timestamps only; no inference-time annotation columns."""
    times = times_for(split)
    result = np.empty((len(times), len(features)), np.float32)
    offset = 0
    for frame in pd.read_csv(ROOT / 'data/frozen/mission1-reference-v1' / (split + '.csv'),
                             usecols=['timestamp', *features], chunksize=25000):
        stop = offset + len(frame)
        actual = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).as_unit('ns').asi8
        assert np.array_equal(actual, times[offset:stop]), 'Timestamp mismatch'
        result[offset:stop] = frame[features].to_numpy(dtype=np.float32)
        offset = stop
    assert offset == len(times) and np.isfinite(result).all()
    return result


def pooled_arrays(folder, pool):
    features = read(folder / 'receipt.json')['features']
    indexes = [features.index(f) for f in TARGETS] if pool == 'target' else list(range(len(features)))
    # Advanced indexing intentionally copies only the small requested channel block.
    coarse = np.load(folder / 'coarse_by_channel.npy', mmap_mode='r')[indexes].max(axis=0)
    fine = np.load(folder / 'fine_by_channel.npy', mmap_mode='r')[indexes].max(axis=0)
    return coarse, fine


@torch.no_grad()
def extract(method, split, selected):
    folder = OUTPUT / 'scores' / method / split
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        check_files(folder, receipt)
        print(method, split, 'verified score receipt, skipping completed inference', flush=True)
        return
    if split == 'testing' and not (OUTPUT / 'selection_complete.json').exists():
        raise ValueError('Assessment inference prohibited before joint selection freeze')
    path = SOURCE / selected['folder'] / 'model.pt'
    assert digest(path) == selected['model_sha256']
    model, scaler, features, masked = load_model(path, 'cuda')
    assert not masked, 'Both selected checkpoints must reconstruct unmasked inputs'
    assert features == (TARGETS if method == 'pretrain_finetune' else TARGETS + ['channel_14', 'channel_21', 'channel_29'])
    model.eval()
    raw = load_features(split, features)
    x = scaler.transform(raw)
    del raw
    starts = np.arange(0, len(x) - config.SEQ_LEN + 1, config.SCORE_STRIDE, dtype=np.int64)
    assert starts[-1] + config.SEQ_LEN == len(x), 'Uncovered edge'
    chunks = folder / 'chunks'
    chunks.mkdir(exist_ok=True)
    began = time.monotonic()
    for offset in range(0, len(starts), CHUNK_WINDOWS):
        subset = starts[offset:offset + CHUNK_WINDOWS]
        name = f'{offset:08d}.npz'
        receipt_name = chunks / f'{offset:08d}.json'
        lo, hi = int(subset[0]), int(subset[-1] + config.SEQ_LEN)
        if receipt_name.exists():
            receipt = read(receipt_name)
            check_files(chunks, receipt)
            assert receipt['offset'] == offset and receipt['lo'] == lo and receipt['hi'] == hi
            continue
        means = np.empty((len(subset), len(features)), np.float32)
        fine_sum = np.zeros((len(features), hi - lo), np.float32)
        count = np.zeros(hi - lo, np.float32)
        for j in range(0, len(subset), BATCH):
            batch_starts = subset[j:j + BATCH]
            inputs = inputs_at(x, batch_starts, 'cuda')
            error = (model(inputs) - inputs).square()
            means[j:j + len(batch_starts)] = error.mean(dim=(2, 3)).cpu().numpy()
            detailed = error.squeeze(-1).cpu().numpy()
            indexes = (batch_starts[:, None] - lo + np.arange(config.SEQ_LEN)[None, :]).ravel()
            np.add.at(count, indexes, 1)
            for ch in range(len(features)):
                np.add.at(fine_sum[ch], indexes, detailed[:, ch, :].ravel())
        assert np.isfinite(means).all() and np.isfinite(fine_sum).all() and (count > 0).all()
        temp = chunks / (name + '.tmp')
        with temp.open('wb') as stream:
            np.savez(stream, means=means, fine_sum=fine_sum, count=count, starts=subset)
        temp.replace(chunks / name)
        write_json(receipt_name, dict(offset=offset, lo=lo, hi=hi,
                                     files={name: digest(chunks / name)}))
        print(f'{method} {split}: {offset + len(subset)}/{len(starts)} windows; '
              f'{time.monotonic() - began:.0f}s', flush=True)
    del model, x
    gc.collect()
    torch.cuda.empty_cache()
    means = np.empty((len(starts), len(features)), np.float32)
    total = np.zeros((len(features), len(times_for(split))), np.float32)
    count = np.zeros(total.shape[1], np.float32)
    chunk_receipts = {}
    for offset in range(0, len(starts), CHUNK_WINDOWS):
        receipt = read(chunks / f'{offset:08d}.json')
        check_files(chunks, receipt)
        chunk_receipts[f'{offset:08d}.json'] = digest(chunks / f'{offset:08d}.json')
        with np.load(chunks / f'{offset:08d}.npz') as part:
            n = len(part['starts'])
            assert np.array_equal(part['starts'], starts[offset:offset + n])
            means[offset:offset + n] = part['means']
            lo, hi = receipt['lo'], receipt['hi']
            total[:, lo:hi] += part['fine_sum']
            count[lo:hi] += part['count']
    assert (count > 0).all()
    total /= count[None, :]
    save_npy(folder / 'fine_by_channel.npy', total)
    del total, count
    coarse = aggregate(means, starts, len(times_for(split)))
    save_npy(folder / 'window_means.npy', means)
    save_npy(folder / 'coarse_by_channel.npy', coarse)
    old = (CALIBRATION / method / 'calibration_score.npy' if split == 'calibration'
           else SOURCE / selected['folder'] / 'testing_score.npy')
    old_score = np.load(old)
    all_score = coarse.max(axis=0)
    assert np.allclose(all_score, old_score, rtol=1e-5, atol=3e-6), 'Original scorer mismatch'
    threshold = selected['rule']['threshold']
    assert np.array_equal(all_score > threshold, old_score > threshold), 'Original prediction mismatch'
    write_json(receipt_path, dict(method=method, split=split, features=features,
                                 model_sha256=digest(path), original_score_sha256=digest(old),
                                 original_max_abs_difference=float(np.max(np.abs(all_score - old_score))),
                                 exact_original_prediction_parity=True,
                                 files={n: digest(folder / n) for n in ('window_means.npy', 'coarse_by_channel.npy', 'fine_by_channel.npy')},
                                 chunk_receipts=chunk_receipts))
    print(method, split, 'complete: scorer and original prediction parity passed', flush=True)


def calibration_diagnostics(method, selected, evaluators):
    folder = OUTPUT / 'scores' / method / 'calibration'
    features = read(folder / 'receipt.json')['features']
    scores = np.load(folder / 'coarse_by_channel.npy')
    threshold = selected['rule']['threshold']
    original = scores.max(axis=0) > threshold
    winner = scores.argmax(axis=0)
    rows = []
    for ch, feature in enumerate(features):
        p = scores[ch] > threshold
        metrics = evaluators['input_channels_40_47'].score(p)
        rows.append(dict(method=method, feature=feature, direct_alarm_samples=int(p.sum()),
                         winning_alarm_samples=int(((winner == ch) & original).sum()),
                         detected_local_ids=detected_ids(evaluators['input_channels_40_47'], p),
                         **metrics))
    return rows


def calibrate(selections):
    ct = times_for('calibration')
    evaluators = assessment_scopes(ct)
    nominal = nominal_mask(evaluators['system_all_channels'])
    frozen = {}
    diagnostics = []
    for method in METHODS:
        diagnostics.extend(calibration_diagnostics(method, selections[method], evaluators))
    write_json(OUTPUT / 'calibration_channel_diagnostics.json', diagnostics)
    for arm, (method, requested_pool) in ARMS.items():
        folder = OUTPUT / arm
        folder.mkdir(exist_ok=True)
        selected = selections[method]
        old_score = np.load(CALIBRATION / method / 'calibration_score.npy')
        threshold = selected['rule']['threshold']
        original = old_score > threshold
        required_ids = detected_ids(evaluators['input_channels_40_47'], original)
        pool = requested_pool
        coarse, fine = pooled_arrays(OUTPUT / 'scores' / method / 'calibration', pool)
        target_status = 'not_applicable'
        if arm == 'expanded_target':
            cache = {}
            values = sorted(set(float(v) for v in np.r_[np.percentile(coarse, QUANTILES), threshold]))
            rows = [candidate_metrics(coarse > t, dict(index=i, threshold=t, original=False), evaluators, required_ids, cache)
                    for i, t in enumerate(values)]
            write_json(folder / 'target_threshold_candidates.json', rows)
            winner = choose(rows, allow_empty=True)
            if winner is None:
                target_status = 'infeasible_original_all_output_fallback'
                pool = 'all'
                coarse, fine = pooled_arrays(OUTPUT / 'scores' / method / 'calibration', pool)
            else:
                threshold = winner['rule']['threshold']
                target_status = 'eligible_target_only_challenger'
        base = coarse > threshold
        cache = {}
        base_row = candidate_metrics(base, dict(index=0, original=True), evaluators, required_ids, cache)
        assert base_row['feasible']
        rules = boundary_candidates(fine, nominal)
        boundary_rows = [candidate_metrics(localize(base, fine, r), r, evaluators, required_ids, cache) for r in rules]
        write_json(folder / 'boundary_candidates.json', boundary_rows)
        boundary = choose(boundary_rows)
        choices = {}
        unique_counts = {'boundary': len({r['prediction_sha256'] for r in boundary_rows})}
        for family in FAMILIES:
            rows = [candidate_metrics(localize(coarse_transform(coarse, threshold, r), fine, boundary['rule']),
                                      r, evaluators, required_ids, cache) for r in family_candidates(family)]
            write_json(folder / (family + '_candidates.json'), rows)
            choices[family] = choose(rows)
            unique_counts[family] = len({r['prediction_sha256'] for r in rows})
        original_metrics = evaluators['input_channels_40_47'].score(original)
        frozen[arm] = dict(method=method, source_folder=selected['folder'], requested_pool=requested_pool,
                           effective_pool=pool, target_status=target_status, threshold=threshold,
                           base=base_row, original_local=original_metrics, required_ids=required_ids,
                           base_improves_original=bool(base_row['local']['EW_F_0.50'] > original_metrics['EW_F_0.50']),
                           boundary=boundary, choices=choices, unique_masks=unique_counts,
                           distinct_masks_across_all_entries=len(cache),
                           calibration_receipt_sha256=digest(OUTPUT / 'scores' / method / 'calibration' / 'receipt.json'),
                           grid_sha256={p.name: digest(p) for p in folder.glob('*_candidates.json')})
        print(arm, 'calibration', json.dumps(dict(pool=pool, base_f05=base_row['local']['EW_F_0.50'],
                boundary=boundary['rule'], boundary_f05=boundary['local']['EW_F_0.50'],
                postprocessing={f: c['rule'] for f, c in choices.items()})), flush=True)
    path = OUTPUT / 'selection_complete.json'
    if path.exists():
        assert read(path) == frozen
    else:
        write_json(path, frozen)
    write_json(OUTPUT / 'selection_freeze_receipt.json', dict(selection_sha256=digest(path),
               frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
               note='All three arms frozen before any new assessment inference'))
    return frozen


def predictions(coarse, fine, frozen):
    result = {'base': coarse > frozen['threshold']}
    result['boundary'] = localize(result['base'], fine, frozen['boundary']['rule'])
    for family, choice in frozen['choices'].items():
        proposal = coarse_transform(coarse, frozen['threshold'], choice['rule'])
        result[family] = localize(proposal, fine, frozen['boundary']['rule'])
    return result


def calibration_stability(frozen):
    times = times_for('calibration')
    blocks = [('H1', '2006-01-01', '2006-07-01'), ('H2', '2006-07-01', '2007-01-01'),
              ('Q1', '2006-01-01', '2006-04-01'), ('Q2', '2006-04-01', '2006-07-01'),
              ('Q3', '2006-07-01', '2006-10-01'), ('Q4', '2006-10-01', '2007-01-01')]
    rows = []
    for arm, spec in frozen.items():
        coarse, fine = pooled_arrays(OUTPUT / 'scores' / spec['method'] / 'calibration', spec['effective_pool'])
        preds = predictions(coarse, fine, spec)
        selected = read(SOURCE / 'selection_complete.json')[spec['method']]
        preds['original'] = np.load(CALIBRATION / spec['method'] / 'calibration_score.npy') > selected['rule']['threshold']
        for block, start, end in blocks:
            keep = (times >= pd.Timestamp(start).value) & (times < pd.Timestamp(end).value)
            for scope, ev in assessment_scopes(times[keep]).items():
                for variant, p in preds.items():
                    rows.append(dict(arm=arm, variant=variant, block=block, scope=scope,
                                     detected_ids=','.join(detected_ids(ev, p[keep])), **ev.score(p[keep])))
    pd.DataFrame(rows).to_csv(OUTPUT / 'calibration_stability.csv', index=False)


def assess(frozen, selections):
    freeze = read(OUTPUT / 'selection_freeze_receipt.json')
    assert digest(OUTPUT / 'selection_complete.json') == freeze['selection_sha256']
    times = times_for('testing')
    records, events, diagnostics = [], [], []

    def report(arm, method, variant, pred, original):
        annual, pooled = report_predictions(times, pred)
        frame = pd.concat([annual, pooled])
        frame.insert(0, 'variant', variant)
        frame.insert(0, 'method', method)
        frame.insert(0, 'arm', arm)
        records.append(frame)
        diagnostics.append(dict(arm=arm, variant=variant, segments=len(segments(pred)), alarm_samples=int(pred.sum())))
        for period in [str(y) for y in range(2007, 2014)] + ['2007-2013']:
            keep = np.ones(len(times), bool) if '-' in period else ((times >= pd.Timestamp(period + '-01-01').value) &
                    (times < pd.Timestamp(str(int(period) + 1) + '-01-01').value))
            for scope, ev in assessment_scopes(times[keep]).items():
                before, after = set(detected_ids(ev, original[keep])), set(detected_ids(ev, pred[keep]))
                for aid in ev.by_id:
                    events.append(dict(arm=arm, method=method, variant=variant, period=period, scope=scope,
                                       event_id=aid, original_detected=aid in before, selected_detected=aid in after,
                                       gained=aid in after-before, lost=aid in before-after))

    for method, selected in selections.items():
        pred = np.load(SOURCE / selected['folder'] / 'testing_prediction.npy').astype(bool)
        report('reference_' + method, method, 'original', pred, pred)
    for arm, spec in frozen.items():
        coarse, fine = pooled_arrays(OUTPUT / 'scores' / spec['method'] / 'testing', spec['effective_pool'])
        original = np.load(SOURCE / spec['source_folder'] / 'testing_prediction.npy').astype(bool)
        for variant, pred in predictions(coarse, fine, spec).items():
            save_npy(OUTPUT / arm / (variant + '_prediction.npy'), pred)
            report(arm, spec['method'], variant, pred, original)
        print(arm, 'assessment complete', flush=True)
    result = pd.concat(records, ignore_index=True)
    assert len(result) == 336
    result.to_csv(OUTPUT / 'all_results.csv', index=False)
    pd.DataFrame(events).to_csv(OUTPUT / 'event_ids.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / 'segment_diagnostics.csv', index=False)


def main():
    lock = open(ROOT / '.target-boundary.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    set_seed(42)
    assert torch.cuda.is_available(), 'CUDA required for matching inference'
    prior = verify_inputs()
    plan = dict(experiment='target-boundary-20261101-v1', source_receipt=digest(SOURCE / 'complete.json'),
                calibration_receipt=digest(CALIBRATION / 'complete.json'), source_selection=digest(SOURCE / 'selection_complete.json'),
                runner=digest(SCRIPT), common=digest(COMMON), protocol=digest(PROTOCOL), source=prior['source_sha256'],
                settings=prior['settings'], runtime=prior['runtime'], batch=BATCH, chunk_windows=CHUNK_WINDOWS,
                device=torch.cuda.get_device_name(0), families={f: family_candidates(f) for f in FAMILIES})
    check_protocol(OUTPUT, plan, (OUTPUT / 'protocol.json').exists())
    if (OUTPUT / 'complete.json').exists():
        check_files(OUTPUT, read(OUTPUT / 'complete.json'))
        print('Already complete; do not repeat assessment', flush=True)
        return
    selections = read(SOURCE / 'selection_complete.json')
    for selected in selections.values():
        assert int(selected['rule']['merge_gap']) == 0 and int(selected['rule']['min_dur']) == 1
    for method in METHODS:
        extract(method, 'calibration', selections[method])
    if (OUTPUT / 'selection_complete.json').exists() and (OUTPUT / 'selection_freeze_receipt.json').exists():
        frozen = read(OUTPUT / 'selection_complete.json')
        assert digest(OUTPUT / 'selection_complete.json') == read(OUTPUT / 'selection_freeze_receipt.json')['selection_sha256']
        for arm, spec in frozen.items():
            for filename, sha in spec['grid_sha256'].items():
                assert digest(OUTPUT / arm / filename) == sha
    else:
        frozen = calibrate(selections)
    calibration_stability(frozen)
    print('Joint freeze verified; opening assessment features for fixed inference', flush=True)
    for method in METHODS:
        extract(method, 'testing', selections[method])
    assess(frozen, selections)
    verify_inputs()
    assert read(OUTPUT / 'protocol.json') == plan
    assert digest(SCRIPT) == plan['runner'] and digest(COMMON) == plan['common'] and digest(PROTOCOL) == plan['protocol']
    write_json(OUTPUT / 'complete.json', dict(needs_independent_verification=True,
               files={str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.rglob('*')
                      if p.is_file() and p.name != 'complete.json' and not p.name.endswith('.tmp')}))
    print('336 metric rows complete; independent verification required', flush=True)


if __name__ == '__main__':
    main()
