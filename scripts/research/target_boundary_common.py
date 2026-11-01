"""Bounded score/localization study helpers; no model training or test selection."""
import hashlib
from pathlib import Path
import numpy as np
from esa_thesis.thresholds import postprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results_longrun/wandb_finetuning_20261024_seed42_v1'
CALIBRATION = ROOT / 'results_longrun/temporal_postprocessing_20261028_v1'
OUTPUT = ROOT / 'results_longrun/target_boundary_20261101_seed42_v1'
TABLE = ROOT / 'docs/research/tables/target-boundary-2026-11-01'
PROTOCOL = ROOT / 'docs/research/target-boundary-protocol-2026-11-01.md'
TARGETS = [f'channel_{i}' for i in range(40, 48)]
METHODS = ('pretrain_finetune', 'expanded_short_no_roll')
ARMS = {
    'expanded_target': ('expanded_short_no_roll', 'target'),
    'masked_all': ('pretrain_finetune', 'all'),
    'expanded_all': ('expanded_short_no_roll', 'all'),
}
FAMILIES = ('temporal', 'strength_duration', 'hysteresis', 'duration')
QUANTILES = (90, 95, 97.5, 98, 99, 99.5, 99.75, 99.9, 99.95, 99.99, 100)


def segments(mask):
    changes = np.diff(np.r_[0, mask, 0].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def mask_hash(prediction):
    return hashlib.sha256(np.packbits(prediction).tobytes()).hexdigest()


def detected_ids(evaluator, prediction):
    starts, ends, closed = evaluator.prediction_intervals(prediction)
    return sorted(aid for aid, intervals in evaluator.by_id.items()
                  if any(evaluator._overlaps(starts, ends, closed, a, b).any()
                         for a, b in intervals))


def nominal_mask(evaluator):
    """Calibration percentile fitting only; excludes every annotation category."""
    mask = np.ones(len(evaluator.timestamps), bool)
    for a, b in evaluator.all_intervals:
        start = np.searchsorted(evaluator.timestamps, a, side='left')
        stop = np.searchsorted(evaluator.timestamps, b, side='right')
        mask[start:stop] = False
    return mask


def boundary_candidates(fine, nominal):
    if not nominal.any() or not np.isfinite(fine).all():
        raise ValueError('Invalid calibration evidence')
    out = [dict(index=0, original=True)]
    for quantile in (95, 99, 99.9):
        cutoff = float(np.percentile(fine[nominal], quantile))
        for padding in (0, 16):
            out.append(dict(index=len(out), original=False, quantile=quantile,
                            cutoff=cutoff, padding=padding))
    return out


def localize(proposals, fine, rule):
    if len(proposals) != len(fine) or not np.isfinite(fine).all():
        raise ValueError('Invalid localization input')
    result = np.asarray(proposals, bool).copy()
    if rule['original']:
        return result
    if rule['padding'] < 0 or not np.isfinite(rule['cutoff']):
        raise ValueError('Invalid localization rule')
    for a, b in segments(result):
        crossing = np.flatnonzero(fine[a:b] > rule['cutoff'])
        if len(crossing):
            left = max(a, a + int(crossing[0]) - rule['padding'])
            right = min(b, a + int(crossing[-1]) + 1 + rule['padding'])
            result[a:left] = False
            result[right:b] = False
    return result


def family_candidates(family):
    durations = (1, 32, 48, 64, 96, 128, 256, 512)
    if family == 'temporal':
        out = [dict(family=family, original=g == 0 and d == 1, gap=g, duration=d)
               for g in (0, 1, 2, 4, 8, 16, 32, 64)
               for d in (1, 2, 4, 8, 16, 32)]
    else:
        out = [dict(family=family, original=True)]
        if family == 'strength_duration':
            out += [dict(family=family, original=False, duration=d, peak=p)
                    for d in durations for p in (1.1, 1.25, 1.5, 2, 3, 5)]
        elif family == 'hysteresis':
            out += [dict(family=family, original=False, low=l, high=h)
                    for l in (.5, .75, 1) for h in (1, 1.1, 1.25, 1.5, 2)]
        elif family == 'duration':
            out += [dict(family=family, original=False, duration=d) for d in durations]
        else:
            raise ValueError(family)
    return [dict(index=i, **r) for i, r in enumerate(out)]


def coarse_transform(score, threshold, rule):
    if threshold <= 0 or not np.isfinite(score).all():
        raise ValueError('Invalid coarse score/threshold')
    if rule['original']:
        return score > threshold
    family = rule['family']
    if family == 'temporal':
        return postprocess(score > threshold, rule['gap'], rule['duration']).astype(bool)
    pred = score > threshold * (rule['low'] if family == 'hysteresis' else 1)
    for a, b in segments(pred):
        if family == 'hysteresis':
            remove = not np.any(score[a:b] > threshold * rule['high'])
        elif family == 'duration':
            remove = b - a < rule['duration']
        elif family == 'strength_duration':
            remove = b - a < rule['duration'] and score[a:b].max() / threshold < rule['peak']
        else:
            raise ValueError(family)
        if remove:
            pred[a:b] = False
    return pred


def choose(rows, allow_empty=False):
    feasible = [r for r in rows if r['feasible']]
    if not feasible:
        if allow_empty:
            return None
        raise ValueError('No eligible candidate, including fallback')
    return min(feasible, key=lambda r: (-r['local']['EW_F_0.50'],
                                        r['local']['false_positive_seconds'],
                                        not r['rule'].get('original', False),
                                        r['rule']['index']))


def candidate_metrics(pred, rule, evaluators, required_ids, cache):
    key = mask_hash(pred)
    if key not in cache:
        local = evaluators['input_channels_40_47']
        ids = detected_ids(local, pred)
        cache[key] = dict(local=local.score(pred),
                          system=evaluators['system_all_channels'].score(pred),
                          detected_ids=ids, alarm_fraction=float(pred.mean()),
                          feasible=bool(set(required_ids).issubset(ids) and pred.mean() <= .01),
                          prediction_sha256=key)
    return dict(rule=rule, **cache[key])
