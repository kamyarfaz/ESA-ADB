# Short coordinated augmentation: matched development pilot

Completed userGPU run: `results_longrun/development/short_coordinated_seed42_v1`.
Fiveepochs/25000globallynominal windows perarm/fold, seed42, channels40–47.
Legacy vs50/50legacy+shortcoordinated; same objective, architecture, budgets,
scaler and nominal samples. Independent RNG streams preserve same initialweights,
shuffle/dropout streams. Original scoring/threshold policy retained. Both rules
frozen before each fold's April–December assessment. Periods previously inspected:
these are exploratory development results, not untouched test estimates.

|Year|Legacy system F0.5|Candidate system F0.5|Legacy TP/FP/FN|Candidate TP/FP/FN|
|---|---:|---:|---|---|
|2003|0.500000|0.714284|1/0/5|2/0/4|
|2004|0.255770|0.377845|4/13/6|5/9/5|
|2005|0.869085|0.999366|4/0/3|7/0/0|

Input-channel F0.5 equals system in2003/2005.2004 local0.265974→0.395791
(4TP/13FP/3FN→5TP/9FP/2FN). Candidate improves all3matched folds in thisseed;
2005 finds all7events with zero event-level FP. Nominal alarm duration remains
15224.184seconds in2005, so event-level zeroFP does not mean zero nominal alarm
time. Tiny calendar datasets and previous research exposure limit generalization.

## Calibration caveat

2003 selectedepoch1 forboth, cal0.714286.2005 selectedepoch1 forboth,cal0.832880.
2004 legacyepoch2/cal0.237582; candidateepoch1/cal0 at ALLfiveepochs. Earliesttie
rule correctly selects1, but there is no positivecalibration signal for choosing
candidatecheckpoint in2004. Do not celebrate improvedassessment as proof that
calibration is fixed. All epochscompleted even though selectedepoch wasearly.

Review reproduced all6predictions exactly fromsaved scores/frozenrules and all12
metricrows within1e-9, verified fivehistoryrows and earliestmaxcheckpointselection.
No freshinference/fullCSVverification performed in thisreview. Runner reports
results only after itsintegritygates. No current source/model/rulechanges made.

## Decision

Promising candidate, not replacement forfullperiod40epochbaseline. This matched
control's values must not be substituted for the old40epochdevelopment baseline.
Systemtarget0.85 has not been robustly achieved across years orfull2007–2013.
Next: repeat unchanged matchedpilot with seeds43and44, preserving everyfold and
reporting calibrationzeros, local/system precision/recall/FPduration. No threshold
retuning orper-yearwinnerselection. Onlyconsider a larger run if benefits are
stable; donot rerun2007–2013 to optimize the design. No repeat launched yet.

[Tables, histories and frozenrules](tables/short-coordinated-results-2026-09-27/).
