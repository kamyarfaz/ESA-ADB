"""Fixed nominal-tail diagnostic; no threshold search or model fitting."""
import numpy as np
from .evaluation.esa import timestamps_ns


def nominal_threshold(times, scores, annotations):
    times = timestamps_ns(times)
    scores = np.asarray(scores)
    if scores.shape != times.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid scores')
    nominal = np.ones(len(times), dtype=bool)
    for row in annotations.itertuples():
        lo = np.searchsorted(times, row.StartTime.value, side='left')
        hi = np.searchsorted(times, row.EndTime.value, side='right')
        nominal[lo:hi] = False
    if not nominal.any():
        raise ValueError('No nominal calibration samples')
    return float(np.quantile(scores[nominal], .999, method='linear')), int(nominal.sum())


def advancement(summary):
    checks = {}
    for fold in (2001, 2002):
        rows = summary[(summary.fold == fold) & (summary.scope == 'system_all_channels')].set_index('arm')
        a, b = rows.loc['nominal_q999'], rows.loc['control']
        checks[str(fold)] = bool(a.f05_mean > b.f05_mean and a.fp_mean <= b.fp_mean
                                and a.false_seconds_mean <= b.false_seconds_mean)
    return dict(advance=all(checks.values()), fold_pass=checks)
