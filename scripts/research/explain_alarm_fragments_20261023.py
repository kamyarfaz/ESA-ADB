"""Post-assessment explanation only; never feeds candidate/rule selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from esa_thesis.evaluation.esa import EventEvaluator

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results_longrun/false_alarm_reduction_20261023_seed42_v1'
OUT=ROOT/'docs/research/tables/false-alarm-reduction-2026-10-23'
assert (RUN/'complete.json').exists()
times=pd.date_range('2007-01-01','2014-01-01',freq='30s',inclusive='left').asi8
labels=pd.read_csv(ROOT/'data/ESA-Mission1/labels.csv').merge(pd.read_csv(ROOT/'data/ESA-Mission1/anomaly_types.csv'),on='ID',validate='many_to_one')
ev=EventEvaluator(times,labels)
folder=RUN/'expanded_short_no_roll'
before=np.load(folder/'original_prediction.npy');after=np.load(folder/'refined_prediction.npy')
assert np.all(after<=before)
a,b,c=ev.prediction_intervals(before);x,y,z=ev.prediction_intervals(after)
def touches(starts,ends,closed):
    matched=np.zeros(len(starts),bool)
    for lo,hi in ev.all_intervals:matched |= ev._overlaps(starts,ends,closed,lo,hi)
    return matched
old_touches=touches(a,b,c);new_touches=touches(x,y,z)
parent=np.searchsorted(a,x,side='right')-1
assert np.all(parent>=0) and np.all(y<=b[parent])
children=np.bincount(parent,minlength=len(a))
summary=dict(post_assessment_explanation_only=True,original_alarm_intervals=len(a),refined_alarm_intervals=len(x),original_intervals_split=int((children>1).sum()),original_false_alarm_events=int((~old_touches).sum()),refined_false_alarm_events=int((~new_touches).sum()),refined_false_events_from_previously_nominal_parents=int((~new_touches & ~old_touches[parent]).sum()),refined_false_events_from_annotation_touching_parents=int((~new_touches & old_touches[parent]).sum()))
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'fragmentation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
