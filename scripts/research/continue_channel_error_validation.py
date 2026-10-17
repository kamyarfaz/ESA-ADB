"""Finish validation-only evidence; never relabel failed calibration as passed."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('replay',ROOT/'scripts/research/channel_error_replay.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2)
source=m.SOURCE/'2002/seed44'
manifest=json.loads((source/'complete.json').read_text())
for name,sha in manifest.items(): assert m.digest(source/name)==sha
assert m.digest(m.config.TRAIN_FILE)==m.TRAIN_SHA
annotation_paths=[ROOT/'data/ESA-Mission1/labels.csv',ROOT/'data/ESA-Mission1/anomaly_types.csv']
annotation_hashes=m.verify_recorded(ROOT,annotation_paths)
annotations=m.load_annotations(*annotation_paths)
decision=json.loads((source/'frozen_decisions.json').read_text())['calibration_selected']
checkpoint=source/f"epoch_{decision['epoch']}"/'model.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
scaler=m.RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr']
model=m.MultivariateAE(8).to('cuda');model.load_state_dict(saved['model_state_dict']);model.eval()
raw,_,times_all=m.load_period(m.config.TRAIN_FILE,m.FEATURES,None,'2003-01-01')
for period,start,end in [('validation','2002-07-01','2003-01-01'),('training','2000-01-01','2002-01-01')]:
    stage=m.OUT/f'2002_44_{period}'
    if stage.exists(): raise ValueError('Preserve existing supplementary stage')
    take=(times_all>=pd.Timestamp(start).value)&(times_all<pd.Timestamp(end).value)
    times=times_all[take];x=scaler.transform(raw[take]);starts=np.arange(0,len(x)-m.config.SEQ_LEN+1,m.config.SCORE_STRIDE)
    errors=m.window_errors(model,x,starts,'cuda',64,f'supplementary seed44 {period}')
    values=m.aggregate(errors,starts,len(x));scalar=values.max(axis=0)
    replay=dict(fold=2002,seed=44,period=period,checkpoint_sha256=m.digest(checkpoint),device='cuda',threads=2,supplementary=True)
    if period=='validation':
        expected=np.load(source/'calibration_selected_validation_score.npy');rule=decision['rule']
        predict=lambda s:m.postprocess(s>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        np.testing.assert_allclose(scalar,expected,rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(times,np.load(source/'validation_timestamps.npy'))
        np.testing.assert_array_equal(predict(scalar),np.load(source/'calibration_selected_validation_prediction.npy'))
        replay.update(mask_exact=True,max_abs_error=float(np.max(np.abs(scalar-expected))))
    events,monthly=m.summarize(values,times,annotations,2002,44,period)
    stage.mkdir()
    np.save(stage/'channel_scores.npy',values)
    (stage/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    (stage/'monthly.json').write_text(json.dumps(monthly,indent=2)+'\n')
    (stage/'complete.json').write_text(json.dumps(dict(replay=replay,files={p.name:m.digest(p) for p in stage.iterdir() if p.is_file()}),indent=2)+'\n')
    print('Verified supplementary',period,flush=True)
assert m.digest(m.config.TRAIN_FILE)==m.TRAIN_SHA
assert m.verify_recorded(ROOT,annotation_paths)==annotation_hashes
(m.OUT/'supplementary_status.json').write_text(json.dumps(dict(validation_models_verified=6,calibration_models_verified=5,training_summaries=6,original_calibration_gate_still_failed=True,continuation_script_sha256=m.digest(Path(__file__))),indent=2)+'\n')
print('Supplementary validation cohort complete; calibration replay remains 5/6',flush=True)
