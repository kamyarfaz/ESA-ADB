"""Check a checkout's research dependencies and dataset headers without training."""
import argparse
import importlib
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-data', action='store_true', help='Check software before downloading data')
    args = parser.parse_args()
    errors = []
    print('Python:', sys.version.split()[0])
    for name in ('numpy', 'pandas', 'torch', 'matplotlib', 'scipy', 'sklearn'):
        try:
            module = importlib.import_module(name)
            print(name + ':', module.__version__)
        except ImportError as exc:
            errors.append(f'{name}: {exc}')
    if errors:
        for error in errors:
            print('ERROR:', error)
        return 1
    import torch
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    else:
        print('CPU execution is available; full training may be slow.')
    if not args.skip_data:
        import pandas as pd
        from .config import TRAIN_FILE, TEST_FILE, RUN_SPECS, ANNOTATIONS_FILE, ANOMALY_TYPES_FILE
        for annotation in (ANNOTATIONS_FILE, ANOMALY_TYPES_FILE):
            if not annotation.is_file():
                errors.append(f'Missing raw annotation file: {annotation}')
        required = {f for spec in RUN_SPECS for f in spec['features']}
        for path in (TRAIN_FILE, TEST_FILE):
            path = Path(path)
            if not path.is_file():
                errors.append(f'Missing dataset: {path}; see README Data preparation.')
                continue
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
                missing = required - columns
                if missing:
                    errors.append(f'{path.name}: missing configured features {sorted(missing)}')
                if not any(c.startswith('is_anomaly_') for c in columns):
                    errors.append(f'{path.name}: expected per-channel is_anomaly_* labels')
                print('Dataset:', path, f'({len(columns)} columns)')
            except Exception as exc:
                errors.append(f'Cannot read {path}: {exc}')
    for error in errors:
        print('ERROR:', error)
    print('Checks failed.' if errors else 'Checks passed (full dataset contents and GPU training not tested).')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
