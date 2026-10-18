"""Independent saved-prediction replay; does not alter either frozen model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.full_short_no_roll import report_predictions
from esa_thesis.calibration_audit import digest
from esa_thesis.thresholds import postprocess
root=Path(__file__).resolve().parents[2]/'results_longrun/similarity_mosad_full_20261018_seed42_v1'
folder=root/'similarity'
for name in ['fit_complete.json','calibration_receipt.json','assessment_receipt.json','complete.json']:
    for file,sha in json.loads((folder/name).read_text())['files'].items():
        assert digest(folder/file)==sha,(name,file)
scores=np.load(folder/'assessment_score.npy');pred=np.load(folder/'assessment_prediction.npy');times=np.load(folder/'assessment_timestamps.npy')
expected=pd.date_range('2007-01-01','2014-01-01',inclusive='left',freq='30s').asi8
assert np.array_equal(times,expected) and len(scores)==7364160
assert np.isfinite(scores).all() and np.isin(pred,[0,1]).all()
rule=json.loads((folder/'frozen_rule.json').read_text())['rule']
assert np.array_equal(pred,postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur'])))
a,p=report_predictions(times,pred)
for actual,name in [(a,'yearly_results.csv'),(p,'pooled_results.csv')]:
    saved=pd.read_csv(folder/name)
    for c in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[c]):np.testing.assert_allclose(actual[c],saved[c],rtol=1e-12,atol=1e-12)
        else:assert actual[c].astype(str).tolist()==saved[c].astype(str).tolist(),c
out=root.parents[1]/'docs/research/tables/similarity-mosad-full-20261018'
out.mkdir(parents=True,exist_ok=True)
combined=pd.concat([a,p],ignore_index=True)
combined.to_csv(out/'similarity_verified_results.csv',index=False)
proof=dict(status='similarity independently replayed; MOSAD training pending',sample_count=len(scores),protocol_sha256=digest(root/'protocol.json'),files={str(f.relative_to(folder)):digest(f) for f in folder.iterdir() if f.is_file()},checks=['stage hashes','complete 2007-2013 timestamps','finite scores and binary predictions','exact frozen-rule prediction replay','all annual and pooled metrics recomputed for both scopes'])
(out/'similarity_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(combined[combined.scope=='system_all_channels'][['period','EW_F_0.50','EW_precision','EW_recall','nominal_false_alarm_rate']].to_string(index=False))
print('Independent similarity verification PASSED')
