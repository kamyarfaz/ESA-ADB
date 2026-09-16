"""Split definitions and fail-closed experiment provenance checks."""
import hashlib
import json
import warnings
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from .evaluation.esa import METRIC_VERSION, EventEvaluator, timestamps_ns


def validation_split(timestamps, months=None):
    times = timestamps_ns(timestamps)
    months = config.VAL_MONTHS if months is None else months
    boundary = pd.Timestamp(times[-1], tz='UTC') - pd.DateOffset(months=months)
    split = int(np.searchsorted(times, boundary.value))
    if split < config.SEQ_LEN or len(times) - split < config.SEQ_LEN + config.MLP_PRED_LEN:
        raise ValueError('Dataset is too short for the calendar validation split and model windows')
    return split


@lru_cache(maxsize=4)
def _read_timestamps(path, size, mtime_ns):
    # These prepared files have unquoted ISO timestamps as their first column.
    # Read only that field: score recalibration does not consume numeric features.
    chunks, pending = [], []
    non_ascii = False
    with open(path, 'rb') as stream:
        if stream.readline().split(b',', 1)[0].strip() != b'timestamp':
            raise ValueError('Expected timestamp as first CSV column')
        for line in stream:
            non_ascii |= not line.isascii()
            pending.append(line.split(b',', 1)[0].decode('ascii').strip())
            if len(pending) == 250000:
                chunks.append(timestamps_ns(pending))
                pending = []
    if pending:
        # A final chunk may contain only one timestamp.
        dates = pd.DatetimeIndex(pd.to_datetime(pending, utc=True))
        if dates.hasnans:
            raise ValueError('Timestamps contain NaT')
        chunks.append(dates.as_unit('ns').asi8)
    if non_ascii:
        warnings.warn(f'{path}: non-ASCII bytes found outside the parsed timestamps. '
                      'Cached-score evaluation uses timestamps only; validate/regenerate '
                      'the feature CSV before training.', RuntimeWarning)
    return timestamps_ns(np.concatenate(chunks))


def read_timestamps(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _read_timestamps(str(path), stat.st_size, stat.st_mtime_ns)


def evaluator_for(timestamps):
    return EventEvaluator.from_files(timestamps, config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE)


def manifest(spec):
    settings = {k: v for k, v in vars(config).items() if k.isupper() and k not in ('DEVICE', 'OUT_ROOT', 'RUN_SPECS')}
    files = {}
    for path in (config.TRAIN_FILE, config.TEST_FILE, config.ANNOTATIONS_FILE, config.ANOMALY_TYPES_FILE):
        stat = path.stat()
        files[str(path.resolve())] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    sources = {}
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        sources[str(path.relative_to(Path(__file__).parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {'metric_version': METRIC_VERSION, 'settings': settings, 'spec': spec,
               'files': files, 'source_sha256': sources,
               'validation': 'last three calendar months; full-mission Anomaly + Rare Event IDs'}
    # Convert numpy scalars/arrays, Paths, tuples to a stable JSON representation.
    return json.loads(json.dumps(payload, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def guard_run(run_dir, expected):
    path = Path(run_dir) / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f'Incompatible cached run: {run_dir}. Use a new output root.')
    else:
        if any(Path(run_dir).iterdir()):
            raise ValueError(f'Unversioned cached run: {run_dir}. Legacy results cannot be resumed; use a new output root.')
        path.write_text(json.dumps(expected, indent=2) + '\n')
