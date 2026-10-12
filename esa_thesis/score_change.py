"""Fixed, history-relative transform of retrospective reconstruction scores."""
import numpy as np
from scipy.signal import lfilter

HALF_LIFE_SAMPLES = 2880  # One day at the prepared 30-second cadence.


def transform(scores, state=None):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError('Need a nonempty finite nonnegative score vector')
    x=np.log1p(scores)
    alpha=-np.expm1(-np.log(2.)/HALF_LIFE_SAMPLES)
    previous=float(x[0]) if state is None else float(state)
    if not np.isfinite(previous):raise ValueError('Invalid state')
    mean,_=lfilter([alpha],[1.,-(1.-alpha)],x,zi=[(1.-alpha)*previous])
    lagged=np.r_[previous,mean[:-1]]
    return np.abs(x-lagged),float(mean[-1])
