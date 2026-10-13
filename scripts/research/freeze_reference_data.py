"""Package the historical reference split without changing samples or preprocessing.

The files are read-only snapshots, not an untouched independent test set. Existing
research runners are not redirected by this utility. Verify before consuming them.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / 'data/preprocessed/multivariate/ESA-Mission1-semi-supervised'
EXPECTED = ('fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad',
            '9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c')
BOUNDS = ('2000-01-01 00:00:00', '2006-01-01 00:00:00',
          '2007-01-01 00:00:00', '2014-01-01 00:00:00')
NAMES = ('train.csv', 'calibration.csv', 'testing.csv')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def package(sources, expected_hashes, labels, labels_hash, output, bounds=BOUNDS):
    """Copy exact CSV rows into disjoint half-open chronological subsets."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a frozen dataset version')
    staging = output.with_name(output.name + '.incomplete')
    staging.mkdir(parents=True, exist_ok=False)
    dates = [datetime.fromisoformat(x) for x in bounds]
    if len(dates) != 4 or any(a >= b for a, b in zip(dates, dates[1:])):
        raise ValueError('Four strictly increasing split boundaries required')
    if len(sources) != 2 or len(expected_hashes) != 2:
        raise ValueError('Expected original train and test source files')
    expected_next = dates[0]
    counts = [0, 0, 0]
    header = None
    inputs = []
    handles = [ (staging / n).open('xb') for n in NAMES ]
    try:
        for source_index, (source, expected) in enumerate(zip(sources, expected_hashes)):
            source = Path(source)
            before = source.stat()
            h = hashlib.sha256()
            excluded = 0
            with source.open('rb') as stream:
                current_header = stream.readline()
                h.update(current_header)
                if header is None:
                    header = current_header
                    if not header.startswith(b'timestamp,'):
                        raise ValueError('Timestamp must be the first column')
                    for dst in handles:
                        dst.write(header)
                elif header != current_header:
                    raise ValueError('Source schemas differ')
                for raw in stream:
                    h.update(raw)
                    timestamp = datetime.fromisoformat(raw.split(b',', 1)[0].decode('ascii'))
                    # Original sources each include a duplicated/excluded endpoint.
                    source_lo = dates[0] if source_index == 0 else dates[2]
                    source_hi = dates[2] if source_index == 0 else dates[3]
                    if timestamp == source_hi:
                        excluded += 1
                        if excluded > 1:
                            raise ValueError('Repeated endpoint')
                        continue
                    if timestamp < source_lo or timestamp >= source_hi or excluded:
                        raise ValueError('Unexpected source boundary or ordering')
                    if timestamp != expected_next:
                        raise ValueError('Missing, duplicate or out-of-order timestamp')
                    expected_next += timedelta(seconds=30)
                    split = 0 if timestamp < dates[1] else 1 if timestamp < dates[2] else 2
                    handles[split].write(raw)
                    counts[split] += 1
            if h.hexdigest() != expected:
                raise ValueError('Source checksum mismatch; incomplete output preserved')
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed during packaging')
            inputs.append({'path': str(source), 'sha256': expected, 'excluded_endpoint_rows': excluded})
    finally:
        for handle in handles:
            handle.close()
    if expected_next != dates[-1]:
        raise ValueError('Incomplete time coverage')
    expected_counts = [int((b-a).total_seconds()/30) for a,b in zip(dates,dates[1:])]
    if counts != expected_counts:
        raise ValueError('Split row counts differ')
    labels = Path(labels)
    if digest(labels) != labels_hash:
        raise ValueError('Binary label file differs from independently validated export')
    shutil.copyfile(labels, staging / 'labels.csv.gz')
    if digest(staging / 'labels.csv.gz') != labels_hash:
        raise ValueError('Label copy verification failed')
    payload = {'version': 'mission1-historical-reference-v1', 'boundaries_utc': list(bounds),
               'intervals': '[start,end)', 'sample_interval_seconds': 30,
               'sources': inputs, 'rows': dict(zip(NAMES, counts)),
               'columns': header.decode('ascii').strip().split(','),
               'status': 'Previously inspected historical reference; not untouched confirmation',
               'data_policy': 'Exact source rows and columns retained; no feature or input-channel selection',
               'labels_policy': 'Binary any-channel Anomaly/Rare Event; is_ignored flags gap-only rows. Original CSV annotation columns retain 0/1/2/3 codes. Exact raw event IDs remain required for event scoring.',
               'files': {n: {'sha256': digest(staging/n), 'bytes': (staging/n).stat().st_size}
                         for n in (*NAMES, 'labels.csv.gz')}}
    manifest = staging / 'manifest.json'
    manifest.write_text(json.dumps(payload, indent=2)+'\n')
    for p in staging.iterdir():
        p.chmod(0o444)
    staging.rename(output)
    return payload


def verify(output):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    if set(manifest['files']) != {*NAMES, 'labels.csv.gz'}:
        raise ValueError('Unexpected frozen file inventory')
    for name, record in manifest['files'].items():
        p = output/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise ValueError('Frozen file changed: '+name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['create','verify'])
    parser.add_argument('--output', type=Path, default=ROOT/'data/frozen/mission1-reference-v1')
    args = parser.parse_args()
    if args.mode == 'create':
        audit = json.loads((ROOT/'outputs/thesis-summary-20261013/build/data-audit.json').read_text())
        package([SOURCE_DIR/'84_months.train.csv', SOURCE_DIR/'84_months.test.csv'], EXPECTED,
                ROOT/'outputs/thesis-summary-20261013/deliverables/ESA_Mission1_binary_labels.csv.gz',
                audit['sha256'], args.output)
    manifest = verify(args.output)
    print(json.dumps({'verified':str(args.output),'rows':manifest['rows'],
                      'manifest_sha256':digest(args.output/'manifest.json')},indent=2))


if __name__ == '__main__':
    main()
