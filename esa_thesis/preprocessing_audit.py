"""Read-only reconstruction of hold/restoration for non-counter channels."""
import numpy as np

STEP_NS = 30_000_000_000


def restore_candidates(raw_times, labels, step_ns=STEP_NS):
    """Return destination grid times and last annotated raw positions per bin.

    A bin is restored if it contains multiple samples, ends nominal, and contains
    an Anomaly/Rare Event. Communication gaps alone never trigger restoration.
    """
    times = np.asarray(raw_times, dtype=np.int64)
    labels = np.asarray(labels)
    if times.ndim != 1 or labels.shape != times.shape or not len(times):
        raise ValueError('Nonempty matching raw vectors required')
    if np.any(np.diff(times)<=0) or not np.isin(labels, [0, 1, 2, 3]).all() or step_ns<=0:
        raise ValueError('Invalid raw times, labels or resolution')
    bins = times // step_ns
    ends = np.r_[np.flatnonzero(np.diff(bins))+0, len(times)-1]
    starts = np.r_[0, ends[:-1]+1]
    positive = np.flatnonzero((labels==1)|(labels==2))
    if not len(positive):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    lookup = np.searchsorted(positive, ends, side='right')-1
    last_positive = positive[np.maximum(lookup, 0)]
    eligible = (ends>starts) & (labels[ends]==0) & (lookup>=0) & (last_positive>=starts)
    return (bins[ends[eligible]]+1)*step_ns, last_positive[eligible]


def held_values(raw_times, raw_values, grid):
    times, values, grid = np.asarray(raw_times), np.asarray(raw_values), np.asarray(grid)
    if times.ndim!=1 or values.shape!=times.shape or len(times)==0:
        raise ValueError('Invalid raw vectors')
    if np.any(np.diff(times)<=0) or np.any(np.diff(grid)<=0) or not np.isfinite(values).all():
        raise ValueError('Nonfinite values or unordered timestamps')
    positions = np.searchsorted(times, grid, side='right')-1
    # Matches historical first-value/backward initialization, explicitly noncausal.
    return values[np.maximum(positions, 0)]
