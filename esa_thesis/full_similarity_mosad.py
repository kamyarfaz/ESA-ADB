"""Frozen full-period comparison requested 18 October 2026; sequential methods."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from . import config
from .calibration_audit import digest, verify_recorded, load_annotations
from .data import RobustChannelScaler
from .development import load_period, QUANTILES
from .development_recovery import atomic_save, capture_rng, restore_rng, check_protocol
from .diagnose_channels import aggregate
from .full_short_no_roll import verify_frozen, MANIFEST_SHA, report_predictions
from .protocol import evaluator_for
from .runtime import set_seed
from .thresholds import sweep_thresholds, select_rows, postprocess
from .similarity_mosad_methods import (FEATURES, make_mosad, MOSAD_ARGS, nominal_starts,
    year_reference, windows, nearest_distance, training_shots, mosad_batch_loss,
    mosad_components, combined_mosad)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/frozen/mission1-reference-v1'
SEED = 42
EPOCHS = 40


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def save_npy(path, values):
    with path.with_name(path.name+'.tmp').open('wb') as stream:
        np.save(stream, values)
    path.with_name(path.name+'.tmp').replace(path)


def manifest_files(folder, names):
    return {name: digest(folder/name) for name in names}


def verify_receipt(folder, name):
    path = folder/name
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    for filename, sha in receipt['files'].items():
        if digest(folder/filename) != sha:
            raise ValueError(f'Artifact checksum changed: {folder/filename}')
    return True


def select_rule(scores, times, folder, filename):
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite score')
    grid = sweep_thresholds(scores, evaluator=evaluator_for(times),
        threshold_values=np.percentile(scores, QUANTILES), merge_gaps=[0, 16, 64], min_durs=[1, 8, 32])
    row = select_rows(grid).iloc[0].to_dict()
    grid.to_csv(folder/filename, index=False)
    return row


def score_windows(x, method, folder, stage, model=None, reference=None, context=None):
    receipt = stage+'_receipt.json'
    if verify_receipt(folder, receipt):
        return np.load(folder/(stage+'_score.npy'), allow_pickle=False)
    starts = np.arange(0, len(x)-256+1, 32, dtype=np.int64)
    values = np.empty(len(starts), dtype=np.float32)
    nearest = np.empty(len(starts), dtype=np.int64) if method == 'similarity' else None
    components = np.empty((len(starts), 3), dtype=np.float32) if method == 'mosad' else None
    began = time.monotonic()
    for offset in range(0, len(starts), 128):
        batch = torch.from_numpy(windows(x, starts[offset:offset+128])).to('cuda')
        if method == 'similarity':
            score, indices = nearest_distance(batch, reference)
            values[offset:offset+len(batch)] = score.cpu().numpy()
            nearest[offset:offset+len(batch)] = indices.cpu().numpy()
        else:
            parts = mosad_components(model, batch, context['projection']).cpu().numpy()
            components[offset:offset+len(batch)] = parts
            values[offset:offset+len(batch)] = combined_mosad(parts, context['rec_min'], context['rec_max'])
        if offset == 0 or offset % 12800 == 0 or offset+len(batch) == len(starts):
            print(f'{method} {stage}: {offset+len(batch)}/{len(starts)} windows, {time.monotonic()-began:.0f}s', flush=True)
    score = aggregate(values[:, None], starts, len(x))[0]
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite aggregated score')
    names = [stage+'_score.npy']
    save_npy(folder/names[0], score)
    if nearest is not None:
        names.append(stage+'_nearest_reference.npy'); save_npy(folder/names[-1], nearest)
    if components is not None:
        names.append(stage+'_components.npy'); save_npy(folder/names[-1], components)
    write_json(folder/receipt, dict(scoring_seconds=time.monotonic()-began, files=manifest_files(folder, names)))
    return score


def fit_similarity(x, nominal, times, scaler, folder):
    if verify_receipt(folder, 'fit_complete.json'):
        return torch.load(folder/'reference.pt', map_location='cpu', weights_only=False)
    starts = year_reference(nominal, times)
    reference = windows(x, starts)
    state = dict(reference=torch.from_numpy(reference), starts=starts, features=FEATURES,
                 scaler_median=scaler.median_, scaler_iqr=scaler.iqr_)
    atomic_save(state, folder/'reference.pt')
    write_json(folder/'fit_complete.json', dict(reference_windows=len(starts),
        files=manifest_files(folder, ['reference.pt'])))
    print(f'Similarity fit complete: {len(starts)} normal reference windows, all eight channels', flush=True)
    return state


@torch.no_grad()
def scoring_context(model, x, reference_starts):
    model.eval()
    reference_ids = reference_starts[np.linspace(0, len(reference_starts)-1, 64, dtype=int)]
    normal = torch.from_numpy(windows(x, reference_ids)).to('cuda')
    projection = model.conhead(model.encoder(normal))
    fit_ids = reference_starts[np.linspace(0, len(reference_starts)-1, min(4096, len(reference_starts)), dtype=int)]
    errors = []
    for offset in range(0, len(fit_ids), 128):
        batch = torch.from_numpy(windows(x, fit_ids[offset:offset+128])).to('cuda')
        errors.append(mosad_components(model, batch, projection)[:, 0].cpu().numpy())
    errors = np.concatenate(errors)
    low, high = float(errors.min()), float(errors.max())
    if not high > low:
        raise ValueError('Degenerate MOSAD training error range')
    return dict(projection=projection, rec_min=low, rec_max=high,
                reference_starts=reference_ids, scale_starts=fit_ids)


def train_mosad(x, nominal, times, scaler, val, val_times, annotations, folder):
    if verify_receipt(folder, 'fit_complete.json'):
        return torch.load(folder/'selected_model.pt', map_location='cpu', weights_only=False)
    shots = training_shots(annotations, times)
    write_json(folder/'training_shots.json', shots)
    anomaly = torch.from_numpy(windows(x, [r['start'] for r in shots])).to('cuda')
    reference_starts = year_reference(nominal, times)
    model = make_mosad().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5, amsgrad=True)
    state = dict(epoch=1, batch=0, loss_sum=0., loss_count=0, history=[], best=None)
    state['training_step_seconds'] = 0.
    checkpoint = folder/'checkpoint_last.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
    def persist():
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng())
        atomic_save(state, checkpoint)
    for epoch in range(state['epoch'], EPOCHS+1):
        order = np.random.default_rng(SEED+epoch).permutation(nominal)
        # Merge a singleton final batch so contrastive normal anchors are always valid.
        batches = [order[i:i+64] for i in range(0, len(order), 64)]
        if len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = np.r_[batches[-1], last]
        model.train()
        for index in range(state['batch'], len(batches)):
            step_started = time.monotonic()
            normal = torch.from_numpy(windows(x, batches[index])).to('cuda')
            positive = anomaly[torch.randint(len(anomaly), (10,), device='cuda')]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = mosad_batch_loss(model, normal, positive)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite MOSAD loss epoch {epoch} batch {index}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite MOSAD gradient')
            optimizer.step()
            torch.cuda.synchronize()
            state['training_step_seconds'] += time.monotonic()-step_started
            state['loss_sum'] += float(loss.detach()); state['loss_count'] += 1; state['batch'] = index+1
            if index == 0 or (index+1) % 100 == 0 or index+1 == len(batches):
                persist()
                print(f'MOSAD epoch {epoch}/{EPOCHS} batch {index+1}/{len(batches)} loss={float(loss.detach()):.6g} components={parts}', flush=True)
        row = dict(epoch=epoch, loss=state['loss_sum']/state['loss_count'], cumulative_training_step_seconds=state['training_step_seconds'])
        if epoch % 5 == 0:
            context = scoring_context(model, x, reference_starts)
            scores = score_windows(val, 'mosad', folder, f'calibration_epoch{epoch}', model=model, context=context)
            rule = select_rule(scores, val_times, folder, f'calibration_epoch{epoch}_candidates.csv')
            row['calibration_f05'] = rule['val_esa_f05']
            if state['best'] is None or rule['val_esa_f05'] > state['best']['rule']['val_esa_f05']:
                state['best'] = dict(epoch=epoch, rule=rule)
                selected = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    context={k: v.cpu() if torch.is_tensor(v) else v for k, v in context.items()},
                    features=FEATURES, scaler_median=scaler.median_, scaler_iqr=scaler.iqr_, **state['best'])
                atomic_save(selected, folder/'selected_model.pt')
        state['history'].append(row)
        pd.DataFrame(state['history']).to_csv(folder/'history.csv', index=False)
        state.update(epoch=epoch+1, batch=0, loss_sum=0., loss_count=0)
        persist()
        print('MOSAD completed epoch', row, flush=True)
    selected = torch.load(folder/'selected_model.pt', map_location='cpu', weights_only=False)
    write_json(folder/'frozen_rule.json', dict(epoch=selected['epoch'], rule=selected['rule']))
    write_json(folder/'fit_complete.json', dict(files=manifest_files(folder,
        ['selected_model.pt', 'frozen_rule.json', 'training_shots.json', 'history.csv'])))
    return selected


def assess(method, folder, scaler, model=None, reference=None, context=None):
    if verify_receipt(folder, 'complete.json'):
        return
    if not (folder/'frozen_rule.json').is_file():
        raise ValueError('Rule must be persisted before assessment loading')
    frozen_hash = digest(folder/'frozen_rule.json')
    rule = json.loads((folder/'frozen_rule.json').read_text())['rule']
    raw, _, times = load_period(BASE/'testing.csv', FEATURES, '2007-01-01', '2014-01-01')
    scores = score_windows(scaler.transform(raw), method, folder, 'assessment', model=model, reference=reference, context=context)
    prediction = postprocess(scores > rule['threshold'], int(rule['merge_gap']), int(rule['min_dur']))
    save_npy(folder/'assessment_prediction.npy', prediction)
    save_npy(folder/'assessment_timestamps.npy', times)
    annual, pooled = report_predictions(times, prediction)
    annual.to_csv(folder/'yearly_results.csv', index=False); pooled.to_csv(folder/'pooled_results.csv', index=False)
    pd.concat([annual, pooled], ignore_index=True).to_csv(folder/'all_results.csv', index=False)
    if digest(folder/'frozen_rule.json') != frozen_hash:
        raise ValueError('Frozen rule changed during assessment')
    names = ['frozen_rule.json', 'assessment_score.npy', 'assessment_prediction.npy', 'assessment_timestamps.npy',
             'yearly_results.csv', 'pooled_results.csv', 'all_results.csv']
    write_json(folder/'complete.json', dict(independent_verification_pending=True, files=manifest_files(folder, names)))
    print(method, pooled.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    # Shared lock covers preflight and both jobs; no duplicate runner on this project.
    lock = (ROOT/'results_longrun/.similarity_mosad.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; do not silently change runtime')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    set_seed(SEED)
    print('Verifying immutable train/calibration/testing files and labels...', flush=True)
    manifest = verify_frozen(ROOT)
    annotation_hashes = verify_recorded(ROOT, [config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE])
    package = Path(__file__).parent
    plan = dict(experiment='similarity-mosad-full-20261018-v1', features=FEATURES, seed=SEED, runs=1,
        train=['2000-01-01', '2006-01-01'], calibration=['2006-01-01', '2007-01-01'], assessment=['2007-01-01', '2014-01-01'],
        window=256, train_stride=16, score_stride=32, inference_batch=128, threads=2,
        similarity=dict(reference_per_year=4096, distance='joint 8-channel nearest-reference MSE, global training median/IQR and clip10; no per-window z-normalization', gpu_reference_block=4096),
        mosad=dict(pooling="equivalent fixed-bin max for deterministic CUDA backward", architecture=MOSAD_ARGS, epochs=EPOCHS, nominal_batch=64, real_anomaly_shots_per_batch=10,
            real_library='one centered window per observable input-annotated training ID', optimizer='Adam amsgrad lr0.001 weight_decay0.00001',
            score_reference=64, reconstruction_scale_windows=4096, calibration_epochs=list(range(5, EPOCHS+1, 5))),
        quantiles=QUANTILES, merge_gaps=[0,16,64], min_durs=[1,8,32], alarm_cap=.01,
        primary_scope='system_all_channels', categories=['Anomaly', 'Rare Event'],
        limitations='One seed; reused benchmark. MOSAD uses real training positives; similarity/baseline do not. Adaptations, not exact paper reproductions. Same partitions and evaluation, different model/training budgets.',
        manifest_sha256=MANIFEST_SHA, annotations=annotation_hashes,
        runtime=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, gpu=torch.cuda.get_device_name(0),
                     cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), tf32=False, deterministic_algorithms=True, cublas_workspace=':4096:8'),
        source_sha256={str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')},
        vendor_manifest_sha256=digest(package/'vendor/mosad/upstream.json'))
    check_protocol(a.output, plan, a.resume)
    if not (a.output/'started.json').exists():
        write_json(a.output/'started.json', dict(unix_time=time.time(), utc=str(pd.Timestamp.now(tz='UTC'))))
    if verify_receipt(a.output, 'complete.json'):
        print('Comparison already complete and verified', flush=True); return
    annotations = load_annotations(config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE)
    raw, labels, times = load_period(BASE/'train.csv', FEATURES, '2000-01-01', '2006-01-01')
    scaler = RobustChannelScaler().fit(raw, labels); x = scaler.transform(raw); del raw
    nominal = nominal_starts(labels); del labels
    print(f'Training input ready: {len(x)} samples, {len(nominal)} globally nominal windows', flush=True)
    raw, _, val_times = load_period(BASE/'calibration.csv', FEATURES, '2006-01-01', '2007-01-01')
    val = scaler.transform(raw); del raw
    similarity = a.output/'similarity'; similarity.mkdir(exist_ok=True)
    if not verify_receipt(similarity, 'complete.json'):
        state = fit_similarity(x, nominal, times, scaler, similarity)
        reference = state['reference'].to('cuda')
        if not (similarity/'frozen_rule.json').exists():
            scores = score_windows(val, 'similarity', similarity, 'calibration', reference=reference)
            rule = select_rule(scores, val_times, similarity, 'calibration_candidates.csv')
            write_json(similarity/'frozen_rule.json', dict(rule=rule))
        assess('similarity', similarity, scaler, reference=reference)
        del reference, state
        torch.cuda.empty_cache()
    mosad = a.output/'mosad'; mosad.mkdir(exist_ok=True)
    if not verify_receipt(mosad, 'complete.json'):
        # Reset MOSAD initialization independently of the preceding similarity stage.
        set_seed(SEED)
        selected = train_mosad(x, nominal, times, scaler, val, val_times, annotations, mosad)
        model = make_mosad().to('cuda'); model.load_state_dict(selected['model']); model.eval()
        context = selected['context']; context['projection'] = context['projection'].to('cuda')
        assess('mosad', mosad, scaler, model=model, context=context)
    verify_frozen(ROOT)
    if verify_recorded(ROOT, [config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE]) != annotation_hashes:
        raise ValueError('Annotation dependencies changed')
    rows = []
    for name in ['similarity', 'mosad']:
        frame = pd.read_csv(a.output/name/'all_results.csv'); frame['method'] = name; frame['seed'] = SEED
        rows.append(frame[frame.scope == 'system_all_channels'])
    baseline = ROOT/'results_longrun/short_no_roll_full_20261014_seed42_v1/all_results.csv'
    prior = pd.read_csv(baseline); prior['method'] = 'short_no_roll_baseline'; prior['seed'] = SEED
    rows.append(prior[prior.scope == 'system_all_channels'])
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(a.output/'comparison_system.csv', index=False)
    write_json(a.output/'complete.json', dict(verified_frozen_data=True,
        elapsed_wall_seconds_including_interruptions=time.time()-json.loads((a.output/'started.json').read_text())['unix_time'],
        files=manifest_files(a.output, ['comparison_system.csv', 'similarity/complete.json', 'mosad/complete.json', 'protocol.json']),
        baseline_table_sha256=digest(baseline)))
    print('COMPLETE: both requested methods, all seven years and pooled system comparison', flush=True)


if __name__ == '__main__':
    main()
