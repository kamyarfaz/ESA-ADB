"""Fixed epoch-five, training-only probe of synthetic separation and support.

This is an eval-mode diagnostic, not a replay of stochastic training losses.
All six saved models see identical nominal windows and synthetic corruptions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esa_thesis.data import RobustChannelScaler
from esa_thesis.diagnose_channels import FEATURES, digest
from esa_thesis.models import MultivariateAE
from esa_thesis.runtime import set_seed
from esa_thesis.short_augmentation import make_training_anomaly


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--comparison', choices=['short-coordinated', 'short-no-roll'],
                   default='short-coordinated')
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output directory')
    protocol = json.loads((a.diagnostic/'protocol.json').read_text())
    assert protocol['status'] == 'complete' and protocol['source_hashes_stable']
    assert protocol.get('comparison', 'short-coordinated') == a.comparison
    modes = (['legacy', 'short_coordinated'] if a.comparison == 'short-coordinated'
             else ['short_coordinated', 'short_no_roll'])
    campaign = a.comparison.replace('-', '_')
    torch.set_num_threads(2)
    inputs = a.diagnostic/'diagnostic_inputs.npz'
    input_sha = digest(inputs)
    with np.load(inputs) as data:
        # Every other window retains equal coverage: 128 windows per year.
        raw = data['training_windows'][::2].copy()
        times = data['training_timestamps'][::2].copy()
    assert len(raw) == 512 and np.all(times < pd.Timestamp('2004-01-01').value)
    reference = Path(f'results_longrun/development/{campaign}_seed42_v1/2004/{modes[0]}/model_checkpoint.pt')
    assert digest(reference) == protocol['inputs'][str(reference)]
    checkpoint = torch.load(reference, map_location='cpu', weights_only=False)
    scaler = RobustChannelScaler()
    scaler.median_ = checkpoint['scaler_median']; scaler.iqr_ = checkpoint['scaler_iqr']
    values = scaler.transform(raw.reshape(-1,8)).reshape(512,256,8)
    nominal = torch.from_numpy(values.transpose(0,2,1)[:,:,:,None].copy())
    original = nominal.clone()
    set_seed(29092026)
    synthetic, records = make_training_anomaly(nominal, mode='short_coordinated', return_metadata=True)
    assert torch.equal(nominal, original)
    changed = synthetic != nominal
    energy = (synthetic-nominal).square().mean(dim=(1,2,3)).numpy()
    a.output.mkdir(parents=True)
    sources = {str(inputs):input_sha, str(reference):digest(reference)}
    rows = []
    for seed in [42,43,44]:
        for mode in modes:
            path = Path(f'results_longrun/development/{campaign}_seed{seed}_v1/2004/{mode}/last.pt')
            sources[str(path)] = digest(path)
            last = torch.load(path, map_location='cpu', weights_only=False)
            assert last['epoch'] == 5
            model = MultivariateAE(8); model.load_state_dict(last['model']); model.eval()
            for start in range(0,512,64):
                stop = start+64
                with torch.no_grad():
                    normal_error = (model(nominal[start:stop])-nominal[start:stop]).square()
                    error = (model(synthetic[start:stop])-synthetic[start:stop]).square()
                denominator = normal_error.mean().item()+1e-8
                for i in range(64):
                    idx = start+i; mask = changed[idx]; metadata = records[idx]
                    mse = error[i].mean().item(); normal_mse = normal_error[i].mean().item()
                    rows.append(dict(seed=seed, model=mode, probe_index=idx,
                        time=str(pd.Timestamp(times[idx])), branch=metadata['branch'], kind=metadata.get('kind','legacy_mixed'),
                        requested_length=metadata.get('length'), changed_samples=int(mask.sum()),
                        changed_fraction=float(mask.float().mean()), synthetic_input_mse=float(energy[idx]),
                        normal_mse=normal_mse, synthetic_mse=mse, error_ratio=mse/denominator,
                        hinge_penalty=max(0,10-mse/denominator),
                        unchanged_support_mse=float(error[i][~mask].mean()) if (~mask).any() else None,
                        changed_support_mse=float(error[i][mask].mean()) if mask.any() else None))
            print('Finished fixed epoch5 probe',seed,mode,flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(a.output/'per_window.csv',index=False)
    grouped = frame.groupby(['seed','model','branch','kind']).agg(
        windows=('probe_index','size'), normal_mse=('normal_mse','mean'), synthetic_mse=('synthetic_mse','mean'),
        ratio_median=('error_ratio','median'), mean_hinge=('hinge_penalty','mean'),
        positive_hinge_fraction=('hinge_penalty',lambda x: float(np.mean(x>0))),
        changed_fraction_median=('changed_fraction','median'), input_energy_median=('synthetic_input_mse','median'),
        noop_count=('changed_samples',lambda x:int(np.sum(x==0))))
    grouped.to_csv(a.output/'summary.csv')
    assert sources == {filename:digest(filename) for filename in sources}
    result = dict(status='complete',comparison=a.comparison,probe_mixture='short_coordinated (including rolls), identical for both arms',epoch=5,probe_windows=512,years='2000-2003 nominal only',
        corruption_seed=29092026,model_device='cpu',sources=sources,script_sha256=digest(__file__),
        interpretation='Identical synthetic mixture applied to all final-epoch checkpoints. Eval mode; batch64 denominator mean normal MSE; not exact training-loss replay or real-event performance. No threshold/model changes. Synthetic input energy and support differences do not prove the causal source of calibration failures.')
    (a.output/'protocol.json').write_text(json.dumps(result,indent=2)+'\n')
    print(grouped.to_string())


if __name__ == '__main__':
    main()
