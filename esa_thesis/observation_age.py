"""Causal observation-age metadata; independent of telemetry values and labels."""
import numpy as np


def observation_age(native_ns, grid_ns, cadence_seconds):
    """Return log1p(age/cadence), availability, and age seconds at each grid time.

    No future observation is used. Before the first observation, availability is
    zero and age is zero as a placeholder, not a claim that a measurement exists.
    Cadence must be estimated from training only by the caller.
    """
    native = np.asarray(native_ns)
    grid = np.asarray(grid_ns)
    for name, values in [('native', native), ('grid', grid)]:
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f'{name} timestamps must be a one-dimensional integer array')
        if len(values) > 1 and np.any(values[1:] <= values[:-1]):
            raise ValueError(f'{name} timestamps must be strictly increasing')
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
        raise ValueError('Cadence must be positive and finite')
    index = np.searchsorted(native, grid, side='right') - 1
    available = index >= 0
    age = np.zeros(len(grid), dtype=np.float64)
    age[available] = (grid[available] - native[index[available]]) / 1e9
    if not np.isfinite(age).all() or np.any(age < 0):
        raise ValueError('Invalid observation age')
    return np.log1p(age / cadence_seconds).astype(np.float32), available, age
