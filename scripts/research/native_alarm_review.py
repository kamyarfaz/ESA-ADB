from pathlib import Path
import numpy as np,pandas as pd
root=Path('results_longrun/diagnostics/baseline_2004_epoch35_v2');z=np.load(root/'calibration_inputs.npz');alarms=pd.read_csv(root/'calibration_alarms.csv');rows=[];gaps=[]
out=Path('docs/research/tables/native-alarm-review-2026-09-25');out.mkdir(exist_ok=True)
for j in range(8):
 c=f'channel_{40+j}';f=pd.read_pickle(Path('data/ESA-Mission1/channels')/(c+'.zip'));ts=f.index.asi8;values=f[c].to_numpy();assert np.all(np.diff(ts)>0)
 for _,a in alarms.iterrows():
  start=pd.Timestamp(a.start).value;end=pd.Timestamp(a.end).value
  for period,lo,hi in [('before',start-4*3600*10**9,start),('alarm',start,end),('after',end,end+4*3600*10**9)]:
   mask=(z['timestamps']>=lo)&(z['timestamps']<hi);t=z['timestamps'][mask];actual=z['values'][mask,j]
   ix=np.searchsorted(ts,t,side='right')-1;age=(t-ts[ix])/1e9
   held=values[ix];match=np.isclose(actual,held,rtol=1e-6,atol=1e-7)
   native=(ts>=lo)&(ts<hi);v=values[native]
   rows.append(dict(alarm=int(a.alarm),channel=c,period=period,grid_samples=len(t),native_samples=int(native.sum()),native_unchanged_fraction=float(np.mean(np.diff(v)==0)) if len(v)>1 else np.nan,prepared_unchanged_fraction=float(np.mean(np.diff(actual)==0)),max_age_seconds=float(age.max()),fraction_age_over_300s=float(np.mean(age>300)),forward_hold_match_fraction=float(match.mean())))
  left=np.searchsorted(ts,start-4*3600*10**9)-1;right=np.searchsorted(ts,end+4*3600*10**9)
  for k in range(max(0,left),min(right,len(ts)-1)):
   duration=(ts[k+1]-ts[k])/1e9
   if duration>300:gaps.append(dict(alarm=int(a.alarm),channel=c,start=str(pd.Timestamp(ts[k],tz='UTC')),end=str(pd.Timestamp(ts[k+1],tz='UTC')),gap_seconds=duration))
 print(c,'done',flush=True)
r=pd.DataFrame(rows);r.to_csv(out/'native_vs_prepared.csv',index=False);pd.DataFrame(gaps,columns=['alarm','channel','start','end','gap_seconds']).to_csv(out/'native_gaps.csv',index=False)
print(r.query("period=='alarm'").to_string(index=False));print('GAPS',pd.DataFrame(gaps).to_string(index=False))
