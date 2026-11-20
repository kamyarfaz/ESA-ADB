"""Read-only full-array comparison against preserved v1 preparation."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from esa_thesis.calibration_audit import digest
from esa_thesis.full_similarity_mosad import verify_receipt


def main():
    old = ROOT / 'results_longrun/cross_mission_transfer_20261118_v1'
    new = ROOT / 'results_longrun/cross_mission_transfer_20261120_v2'
    before = json.loads((old / 'preparation.json').read_text())
    after = json.loads((new / 'preparation.json').read_text())
    assert before['raw'] == after['raw']
    for name in ('input-selection.json', 'target-normalization.json'):
        assert digest(old / name) == digest(new / name), name
    records = []
    for key in before['splits']:
        a, b = old / 'prepared' / key, new / 'prepared' / key
        assert digest(a / 'complete.json') == before['splits'][key]
        assert digest(b / 'complete.json') == after['splits'][key]
        assert verify_receipt(a, 'complete.json') and verify_receipt(b, 'complete.json')
        ma, mb = [json.loads((d / 'metadata.json').read_text()) for d in (a, b)]
        for field in ('key', 'mission', 'split', 'features', 'bounds', 'cadence_seconds', 'rows'):
            assert ma[field] == mb[field], (key, field)
        for name in ('times.npy', 'age_ns.npy'):
            assert digest(a / name) == digest(b / name), (key, name)
        changed = []
        if ma['mission'] != 1:
            for name in ('values.npy', 'available.npy', 'excluded.npy'):
                assert digest(a / name) == digest(b / name), (key, name)
        else:
            arrays = {name: [np.load(d / (name + '.npy'), mmap_mode='r') for d in (a, b)]
                      for name in ('values', 'available', 'excluded')}
            ages = np.load(b / 'age_ns.npy', mmap_mode='r')
            added = {c: 0 for c in ('channel_62', 'channel_63')}
            for lo in range(0, ma['rows'], 100000):
                sl = slice(lo, lo + 100000)
                for j, ch in enumerate(ma['features']):
                    x, y = [z[sl, j] for z in arrays['values']]
                    av, bv = [z[sl, j] for z in arrays['available']]
                    if ch not in added:
                        np.testing.assert_array_equal(x, y)
                        np.testing.assert_array_equal(av, bv)
                        continue
                    assert not np.any(av & ~bv)
                    np.testing.assert_array_equal(x[av], y[av])
                    extra = bv & ~av
                    assert np.all(np.isnan(x[extra])) and np.all(np.isfinite(y[extra]))
                    assert np.all((ages[sl, j][extra] > 600 * 10**9) &
                                  (ages[sl, j][extra] <= 900 * 10**9))
                    np.testing.assert_array_equal(bv, np.isfinite(y))
                    added[ch] += int(extra.sum())
                ea, eb = [z[sl] != 0 for z in arrays['excluded']]
                assert not np.any(eb & ~ea)
            changed = added
        records.append(dict(split=key,rows=ma['rows'],additional_valid_rows=changed))
        print('Verified amendment isolation:', key, flush=True)
    receipt = dict(passed=True,verified_utc=datetime.now(timezone.utc).isoformat(),
                   method='All five split hashes verified. Mission2 and transfer arrays byte-identical; all Mission1 values/availability compared in chunks; only62/63 gained finite readings aged>600s and<=900s. Times/ages, panels/map and transfer normalization unchanged.',
                   assessment_used=False,splits=records,verifier_sha256=digest(__file__),
                   v1_preparation_sha256=digest(old/'preparation.json'),
                   v2_preparation_sha256=digest(new/'preparation.json'))
    (new/'amendment-isolation-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    main()
