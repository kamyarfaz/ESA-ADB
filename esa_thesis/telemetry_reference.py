"""Deterministic per-channel deviation from a lagged daily moving reference."""
import numpy as np
from scipy.signal import lfilter


def deviations(x, state=None):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty matrix required')
    initial=x[0].copy() if state is None else np.asarray(state,dtype=np.float64)
    if initial.shape!=(x.shape[1],) or not np.isfinite(initial).all():raise ValueError('Invalid state')
    alpha=-np.expm1(-np.log(2.)/2880)
    means,_=lfilter([alpha],[1.,-(1-alpha)],x,axis=0,zi=((1-alpha)*initial)[None,:])
    return np.abs(x-np.vstack([initial,means[:-1]])),means[-1].copy()


def fit_scale(errors, nominal):
    if not np.any(nominal):raise ValueError('No nominal training samples')
    return np.maximum(np.percentile(errors[nominal],99,axis=0),1e-6)
