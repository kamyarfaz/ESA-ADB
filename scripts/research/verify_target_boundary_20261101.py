"""Independent transforms, selection and metrics; sampled CPU inference replay."""
import json
import hashlib
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis.evaluation.esa import EventEvaluator
from esa_thesis.full_combinations import load_model
from scripts.research.target_boundary_common import ROOT, SOURCE, CALIBRATION, OUTPUT, TABLE, TARGETS


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def groups(mask):
    positive = np.flatnonzero(mask)
    return [g for g in np.split(positive, np.flatnonzero(np.diff(positive) > 1) + 1) if len(g)]


def coarse_replay(scores, threshold, rule):
    factor = rule.get('low', 1) if not rule['original'] else 1
    out = scores > threshold * factor
    if rule['original']:
        return out
    if rule['family'] == 'temporal':
        pieces = groups(out)
        for previous, following in zip(pieces[:-1], pieces[1:]):
            if following[0] - previous[-1] - 1 <= rule['gap']:
                out[previous[-1] + 1:following[0]] = True
        for piece in groups(out):
            if len(piece) < rule['duration']:
                out[piece] = False
        return out
    for piece in groups(out):
        peak = float(scores[piece].max())
        family = rule['family']
        keep = (peak > threshold * rule['high'] if family == 'hysteresis' else
                len(piece) >= rule['duration'] if family == 'duration' else
                len(piece) >= rule['duration'] or peak / threshold >= rule['peak'])
        if not keep:
            out[piece] = False
    return out


def boundary_replay(proposals, fine, rule):
    out = np.zeros(len(proposals), bool)
    for piece in groups(proposals):
        if rule['original']:
            out[piece] = True
            continue
        evidence = piece[fine[piece] > rule['cutoff']]
        if not len(evidence):
            out[piece] = True
        else:
            start = max(int(piece[0]), int(evidence[0]) - rule['padding'])
            stop = min(int(piece[-1]) + 1, int(evidence[-1]) + rule['padding'] + 1)
            out[start:stop] = True
    return out


def ids(ev, pred):
    starts, ends, closed = ev.prediction_intervals(pred)
    return sorted(aid for aid, intervals in ev.by_id.items()
                  if any(ev._overlaps(starts, ends, closed, a, b).any() for a, b in intervals))


def same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (str, bool, list)) or a is None:
        assert a == b, (a, b)
    else:
        assert np.isclose(a, b, rtol=1e-11, atol=1e-9), (a, b)


def select(rows):
    feasible = [r for r in rows if r['feasible']]
    return sorted(feasible, key=lambda r: (-r['local']['EW_F_0.50'], r['local']['false_positive_seconds'],
                                          not r['rule'].get('original', False), r['rule']['index']))[0] if feasible else None


def pooled(folder, pool):
    receipt = read(folder / 'receipt.json')
    indexes = [receipt['features'].index(f) for f in TARGETS] if pool == 'target' else range(len(receipt['features']))
    coarse = np.load(folder / 'coarse_by_channel.npy', mmap_mode='r')
    fine = np.load(folder / 'fine_by_channel.npy', mmap_mode='r')
    return np.maximum.reduce([coarse[i] for i in indexes]), np.maximum.reduce([fine[i] for i in indexes])


def check_candidate(row, pred, evs, required):
    local_ids = ids(evs['input_channels_40_47'], pred)
    observed = dict(rule=row['rule'], local=evs['input_channels_40_47'].score(pred),
                    system=evs['system_all_channels'].score(pred), detected_ids=local_ids,
                    alarm_fraction=float(pred.mean()),
                    feasible=bool(set(required).issubset(local_ids) and pred.mean() <= .01),
                    prediction_sha256=hashlib.sha256(np.packbits(pred).tobytes()).hexdigest())
    same(observed, row)


@torch.no_grad()
def inference_spots(method, split, selected):
    """Independently average CPU residuals covering three predetermined grid points."""
    folder = OUTPUT / 'scores' / method / split
    receipt = read(folder / 'receipt.json')
    saved_coarse = np.load(folder / 'coarse_by_channel.npy', mmap_mode='r')
    saved_fine = np.load(folder / 'fine_by_channel.npy', mmap_mode='r')
    length = saved_coarse.shape[1]
    points = [0, length // 2, length - 1]
    starts = np.arange(0, length - 256 + 1, 32)
    selected_starts = np.unique(np.concatenate([starts[(starts <= p) & (starts + 256 > p)] for p in points]))
    windows = {int(s): np.empty((256, len(receipt['features'])), np.float32) for s in selected_starts}
    filled = {int(s): 0 for s in selected_starts}
    offset = 0
    for frame in pd.read_csv(ROOT / 'data/frozen/mission1-reference-v1' / (split + '.csv'),
                             usecols=receipt['features'], chunksize=50000):
        end = offset + len(frame)
        for s in selected_starts:
            a, b = max(offset, int(s)), min(end, int(s) + 256)
            if a < b:
                windows[int(s)][a-s:b-s] = frame.iloc[a-offset:b-offset][receipt['features']].to_numpy(np.float32)
                filled[int(s)] += b-a
        offset = end
    assert offset == length and all(n == 256 for n in filled.values())
    model, scaler, features, masked = load_model(SOURCE / selected['folder'] / 'model.pt', 'cpu')
    assert not masked and features == receipt['features']
    model.eval()
    inputs = torch.from_numpy(np.stack([scaler.transform(windows[int(s)]).T[:, :, None] for s in selected_starts]))
    error = (model(inputs) - inputs).square().squeeze(-1).numpy()
    differences = []
    for p in points:
        which = np.flatnonzero((selected_starts <= p) & (selected_starts + 256 > p))
        fine = np.stack([error[w, :, p-selected_starts[w]] for w in which]).mean(axis=0)
        coarse = error[which].mean(axis=2).mean(axis=0)
        assert np.allclose(fine, saved_fine[:, p], rtol=2e-4, atol=1e-5), (method, split, p, 'fine')
        assert np.allclose(coarse, saved_coarse[:, p], rtol=2e-4, atol=1e-5), (method, split, p, 'coarse')
        differences.append(dict(sample=p, fine_max_abs=float(np.max(np.abs(fine-saved_fine[:, p]))),
                                coarse_max_abs=float(np.max(np.abs(coarse-saved_coarse[:, p])))))
    del model, inputs, error
    gc.collect()
    return dict(method=method, split=split, windows=len(selected_starts), points=differences,
                rtol=2e-4, atol=1e-5, device='cpu', note='Sampled replication, not full independent inference')


def main():
    torch.set_num_threads(2)
    plan = read(OUTPUT / 'protocol.json')
    assert plan == read(OUTPUT / 'resume_protocol.json')
    for name, expected in read(OUTPUT / 'complete.json')['files'].items():
        assert sha(OUTPUT / name) == expected, name
    for key, path in [('runner', ROOT / 'scripts/research/run_target_boundary_20261101.py'),
                      ('common', ROOT / 'scripts/research/target_boundary_common.py'),
                      ('protocol', ROOT / 'docs/research/target-boundary-protocol-2026-11-01.md'),
                      ('source_receipt', SOURCE / 'complete.json'),
                      ('calibration_receipt', CALIBRATION / 'complete.json'),
                      ('source_selection', SOURCE / 'selection_complete.json')]:
        assert sha(path) == plan[key], key
    prior = read(SOURCE / 'protocol.json')
    for name, expected in plan['source'].items():
        assert sha(ROOT / 'esa_thesis' / name) == expected
    for group in ('annotations', 'reference'):
        for name, expected in prior[group].items():
            assert sha(ROOT / name) == expected
    base = ROOT / 'data/frozen/mission1-reference-v1'
    assert sha(base / 'manifest.json') == prior['manifest_sha']
    for name, record in read(base / 'manifest.json')['files'].items():
        assert sha(base / name) == record['sha256']
    for folder in (SOURCE, CALIBRATION):
        for name, expected in read(folder / 'complete.json')['files'].items():
            assert sha(folder / name) == expected
    freeze_receipt = read(OUTPUT / 'selection_freeze_receipt.json')
    assert sha(OUTPUT / 'selection_complete.json') == freeze_receipt['selection_sha256']
    frozen = read(OUTPUT / 'selection_complete.json')
    selections = read(SOURCE / 'selection_complete.json')
    annotations = pd.read_csv(ROOT / 'data/ESA-Mission1/labels.csv').merge(
        pd.read_csv(ROOT / 'data/ESA-Mission1/anomaly_types.csv'), on='ID', validate='many_to_one')
    local = annotations.copy()
    local.loc[~local.Channel.isin(TARGETS), 'Category'] = 'Out-of-scope (neutral)'
    anns = {'system_all_channels': annotations, 'input_channels_40_47': local}
    ct = pd.date_range('2006-01-01', '2007-01-01', freq='30s', inclusive='left').asi8
    tt = pd.date_range('2007-01-01', '2014-01-01', freq='30s', inclusive='left').asi8
    evs = {scope: EventEvaluator(ct, labels) for scope, labels in anns.items()}
    nominal = np.ones(len(ct), bool)
    for a, b in evs['system_all_channels'].all_intervals:
        nominal[(ct >= a) & (ct <= b)] = False
    candidate_count = 0
    for arm, spec in frozen.items():
        coarse, fine = pooled(OUTPUT / 'scores' / spec['method'] / 'calibration', spec['effective_pool'])
        original = np.load(CALIBRATION / spec['method'] / 'calibration_score.npy') > selections[spec['method']]['rule']['threshold']
        required = ids(evs['input_channels_40_47'], original)
        assert required == spec['required_ids']
        if arm == 'expanded_target':
            target, _ = pooled(OUTPUT / 'scores' / spec['method'] / 'calibration', 'target')
            grid = read(OUTPUT / arm / 'target_threshold_candidates.json')
            expected = sorted(set(float(v) for v in np.r_[np.percentile(target, [90,95,97.5,98,99,99.5,99.75,99.9,99.95,99.99,100]), selections[spec['method']]['rule']['threshold']]))
            assert [r['rule']['threshold'] for r in grid] == expected
            for row in grid:
                check_candidate(row, target > row['rule']['threshold'], evs, required)
            winner = select(grid)
            if winner is None:
                assert spec['effective_pool'] == 'all' and spec['target_status'].startswith('infeasible')
            else:
                assert spec['effective_pool'] == 'target' and spec['threshold'] == winner['rule']['threshold']
            candidate_count += len(grid)
        base_pred = coarse > spec['threshold']
        check_candidate(spec['base'], base_pred, evs, required)
        boundary_grid = read(OUTPUT / arm / 'boundary_candidates.json')
        assert len(boundary_grid) == 7 and boundary_grid[0]['rule'] == dict(index=0, original=True)
        for row in boundary_grid:
            rule = row['rule']
            if not rule['original']:
                same(rule['cutoff'], float(np.percentile(fine[nominal], rule['quantile'])))
            check_candidate(row, boundary_replay(base_pred, fine, rule), evs, required)
        assert select(boundary_grid) == spec['boundary']
        candidate_count += len(boundary_grid)
        for family, choice in spec['choices'].items():
            grid = read(OUTPUT / arm / (family + '_candidates.json'))
            assert [r['rule'] for r in grid] == plan['families'][family]
            assert len(grid) == {'temporal':48, 'strength_duration':49, 'hysteresis':16, 'duration':9}[family]
            for row in grid:
                pred = boundary_replay(coarse_replay(coarse, spec['threshold'], row['rule']), fine, spec['boundary']['rule'])
                check_candidate(row, pred, evs, required)
            assert select(grid) == choice
            candidate_count += len(grid)
        print(arm, 'calibration independently replayed', flush=True)
    stored = pd.read_csv(OUTPUT / 'all_results.csv', dtype={'period':str}).set_index(['arm', 'variant', 'period', 'scope'])
    assert len(stored) == 336 and stored.index.is_unique
    old = pd.read_csv(SOURCE / 'all_results.csv', dtype={'period':str})
    old = old[old.variant == 'selected'].set_index(['method', 'period', 'scope'])
    rows, events, diagnostics = [], [], []
    inventory = [('reference_' + m, m, {'original': np.load(SOURCE / s['folder'] / 'testing_prediction.npy').astype(bool)})
                 for m, s in selections.items()]
    for arm, spec in frozen.items():
        coarse, fine = pooled(OUTPUT / 'scores' / spec['method'] / 'testing', spec['effective_pool'])
        preds = {'base': coarse > spec['threshold']}
        preds['boundary'] = boundary_replay(preds['base'], fine, spec['boundary']['rule'])
        for family, choice in spec['choices'].items():
            preds[family] = boundary_replay(coarse_replay(coarse, spec['threshold'], choice['rule']), fine, spec['boundary']['rule'])
        for variant, pred in preds.items():
            path = OUTPUT / arm / (variant + '_prediction.npy')
            assert np.array_equal(pred, np.load(path))
            assert path.stat().st_mtime_ns > (OUTPUT / 'selection_freeze_receipt.json').stat().st_mtime_ns
        inventory.append((arm, spec['method'], preds))
    for arm, method, preds in inventory:
        original = np.load(SOURCE / selections[method]['folder'] / 'testing_prediction.npy').astype(bool)
        for variant, pred in preds.items():
            diagnostics.append(dict(arm=arm, variant=variant, segments=len(groups(pred)), alarm_samples=int(pred.sum())))
            for period in [str(y) for y in range(2007,2014)] + ['2007-2013']:
                keep = np.ones(len(tt), bool) if '-' in period else ((tt >= pd.Timestamp(period + '-01-01').value) & (tt < pd.Timestamp(str(int(period)+1) + '-01-01').value))
                for scope, labels in anns.items():
                    ev = EventEvaluator(tt[keep], labels)
                    actual = ev.score(pred[keep])
                    for key, value in actual.items():
                        same(value, stored.loc[(arm, variant, period, scope), key])
                        if variant == 'original':
                            same(value, old.loc[(method, period, scope), key])
                    rows.append(dict(arm=arm, method=method, variant=variant, period=period, scope=scope, **actual))
                    before, after = set(ids(ev, original[keep])), set(ids(ev, pred[keep]))
                    for aid in ev.by_id:
                        events.append(dict(arm=arm, method=method, variant=variant, period=period, scope=scope,
                                           event_id=aid, original_detected=aid in before, selected_detected=aid in after,
                                           gained=aid in after-before, lost=aid in before-after))
        print(arm, 'assessment independently replayed', flush=True)
    pd.testing.assert_frame_equal(pd.DataFrame(events), pd.read_csv(OUTPUT / 'event_ids.csv', dtype={'period':str}))
    pd.testing.assert_frame_equal(pd.DataFrame(diagnostics), pd.read_csv(OUTPUT / 'segment_diagnostics.csv'))
    spots = []
    for method in ('pretrain_finetune', 'expanded_short_no_roll'):
        for split in ('calibration', 'testing'):
            receipt = read(OUTPUT / 'scores' / method / split / 'receipt.json')
            assert receipt['model_sha256'] == selections[method]['model_sha256']
            coarse, _ = pooled(OUTPUT / 'scores' / method / split, 'all')
            path = CALIBRATION / method / 'calibration_score.npy' if split == 'calibration' else SOURCE / selections[method]['folder'] / 'testing_score.npy'
            score = np.load(path)
            assert np.allclose(coarse, score, rtol=1e-5, atol=3e-6)
            threshold = selections[method]['rule']['threshold']
            assert np.array_equal(coarse > threshold, score > threshold)
            if split == 'testing':
                for chunk in (OUTPUT / 'scores' / method / split / 'chunks').glob('*.json'):
                    assert chunk.stat().st_mtime_ns > (OUTPUT / 'selection_freeze_receipt.json').stat().st_mtime_ns
            spots.append(inference_spots(method, split, selections[method]))
            print(method, split, 'CPU spot replay passed', flush=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE / 'verified_results.csv', index=False)
    pd.DataFrame(events).to_csv(TABLE / 'event_ids.csv', index=False)
    (TABLE / 'inference_spots.json').write_text(json.dumps(spots, indent=2) + '\n')
    (TABLE / 'verification.json').write_text(json.dumps(dict(verified=True, candidate_rows=candidate_count,
        metric_rows=len(rows), complete_receipt_sha256=sha(OUTPUT / 'complete.json'),
        checks=['Frozen input/source/artifact hashes', 'Independent candidate transforms and calibration selection',
                'Exact prediction replay, 336 metric rows and event IDs', 'Unchanged original control parity',
                'Full all-output score/prediction parity', 'CPU inference spots in calibration and assessment',
                'Assessment chunks after joint selection freeze'],
        limitations=['Single seed', 'Reused benchmark', 'CPU neural replay sampled, not full',
                     'Same corrected metric implementation, supported by pinned-reference parity tests']), indent=2) + '\n')
    print('VERIFIED: 336 assessment rows and all declared calibration candidates', flush=True)


if __name__ == '__main__':
    main()
