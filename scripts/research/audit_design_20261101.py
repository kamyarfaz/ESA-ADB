"""Read-only diagnostics of completed studies; never chooses new assessment rules."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd
from esa_thesis.thresholds import postprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/tables/research-audit-2026-11-01'
ALARM=ROOT/'results_longrun/alarm_evidence_20261028_v1'
TEMP=ROOT/'results_longrun/temporal_postprocessing_20261028_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def mask_sha(p):return hashlib.sha256(np.packbits(p).tobytes()).hexdigest()
def main():
 spec=importlib.util.spec_from_file_location('alarm_rules',ROOT/'scripts/research/run_alarm_evidence_20261028.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 all_results={'audit_date':'2026-11-01','purpose':'Inspect completed frozen studies; no new model/rule selection or training','models':{}}
 seg=pd.read_csv(ALARM/'calibration_segments.csv')
 for method,entry in read(ALARM/'selection_complete.json').items():
  score=np.load(TEMP/method/'calibration_score.npy');threshold=entry['threshold']
  assert len(score)%32==0
  blocks=score.reshape(-1,32);changes=np.flatnonzero(np.diff(score)!=0)+1
  quantized=bool(np.all(blocks==blocks[:,:1]));assert quantized
  unique=set();count=0;family=[]
  for name in module.FAMILIES:
   grid=read(ALARM/method/(name+'_candidates.json'));best=max(grid,key=lambda r:r['local_f05'])
   for row in grid:
    p=module.transform(score,threshold,row['rule'],entry['baseline_rule']);unique.add(mask_sha(p));count+=1
   family.append(dict(family=name,candidates=len(grid),feasible=sum(r['feasible'] for r in grid),baseline_local_f05=grid[0]['local_f05'],unconstrained_best=best))
  temporal={mask_sha(postprocess(score>threshold,g,d)) for g in (0,1,2,4,8,16,32,64) for d in (1,2,4,8,16,32)}
  s=seg[seg.method==method];total=float(s.nominal_false_seconds.sum())
  # Only one individually wholly nominal component exists in the masked model.
  # The other no-local-ID components are neutral/partly annotated; do not call them false events.
  all_prior=read(OUT/'score-mechanics-20261031.json')
  decomposition=all_prior['models'][method]['nominal_false_time'];assert np.isclose(total,decomposition['total_seconds'])
  all_results['models'][method]=dict(score_sha256=sha(TEMP/method/'calibration_score.npy'),samples=len(score),all_32_sample_blocks_constant=quantized,change_count=len(changes),change_index_mod32=sorted(set((changes%32).tolist())),nominal_candidates=count,unique_masks_across_families=len(unique),temporal_nominal_candidates=48,temporal_unique_masks=len(temporal),families=family,calibration_nominal_time=decomposition)
 df=pd.read_csv(ROOT/'docs/research/tables/wandb-finetuning-2026-10-25/verified_results.csv',dtype={'period':str})
 d=df[(df.method=='expanded_short_no_roll')&(df.variant=='selected')&(df.scope=='system_all_channels')].set_index('period')
 annual=d.drop(index='2007-2013');last=d.loc['2013']
 all_results['retrospective_assessment_diagnosis']=dict(method='expanded_short_no_roll',year=2013,year_false_seconds=float(last.false_positive_seconds),annual_false_seconds_sum=float(annual.false_positive_seconds.sum()),year_fraction_of_annual_false_seconds=float(last.false_positive_seconds/annual.false_positive_seconds.sum()),year_time_fpr=float(last.nominal_false_alarm_rate),pooled_time_fpr=float(d.loc['2007-2013'].nominal_false_alarm_rate),restriction='Descriptive existing assessment only; must not select new thresholds, channels or adaptive rules from this year')
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'design-diagnostics.json').write_text(json.dumps(all_results,indent=2)+'\n')
 print(json.dumps({m:dict(nominal_candidates=x['nominal_candidates'],unique_masks=x['unique_masks_across_families'],temporal_unique=x['temporal_unique_masks']) for m,x in all_results['models'].items()},indent=2))
if __name__=='__main__':main()
