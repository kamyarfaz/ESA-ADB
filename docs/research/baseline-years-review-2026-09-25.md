# Baseline consistency review: 2003–2005

Historical Transformer AE with pseudo-anomalies, inputs40–47, seed42, 40epochs.
Training precedes January of each year; January–March calibrates; April–December
is development assessment. Each year's own saved checkpoint, scores and frozen
rule were used. These are not the later 25k-window masked-pilot controls and not
the full-period epoch10 model. No inference, training or parameter selection.

| Year | Selected epoch | Calibration events | Calibration TP / FP | Calibration F0.5 | Assessment system F0.5 | Local F0.5 | Assessment TP / FP / FN |
|---|---:|---:|---|---:|---:|---:|---|
|2003|25|3|1 / 0|0.714241|0.833170|0.833170|5 / 1 / 1|
|2004|35|1|1 / 2|0.383907|0.428131|0.447280|6 / 9 / 4|
|2005|5|4|2 / 0|0.832880|0.869055|0.869055|4 / 0 / 3|

## What makes 2004 worse?

False alarms are the main metric weakness: corrected precision0.399520 despite
recall0.6. By comparison, 2005 recall is lower (0.571429), but precision0.999157
gives higher F0.5. F0.5 emphasizes precision. This is not evidence that misses
are unimportant or that simply raising a threshold will improve results.

2004 calibration already has two false-positive events for one detected event.
Only one positive event is available to choose among checkpoints and thresholds,
so its calibration is a weak guide to the rest of the year. Sparse calibration
also affects the other years (three and four events), and is not a unique proven
cause of 2004's failure.

The nine false-positive alarm intervals occur in May(2), July(1), August(2),
October(2), November(1), December(1). Durations range16–80minutes, totaling368minutes.
Total falsely alarmed nominal duration is428.919minutes because even alarms
touching a real event may extend into nominal time. These are different measures.

The monthly median nominal score/threshold remains approximately0.0185–0.0200
through April–December2004. The 99th percentile varies more,0.304–0.753. This is
consistent with occasional high-score episodes rather than a large sustained
change in the typical score. It does not establish unchanged raw telemetry:
this review used saved aggregate scores, not raw channel values or per-channel
reconstruction errors. Monthly distributions describe the development assessment
and must not be used to choose a threshold later called independently validated.

## Which events were missed?

- 2003: id_13, annotated on the selected inputs.
- 2004: id_91 on selected inputs; id_92/id_94/id_97 outside selected inputs.
- 2005: id_16/id_100/id_104, all annotated on selected inputs.

Thus the baseline's perfect local detection in 2006 does not hold across these
earlier development periods. Outside-input annotation does not prove absence
of indirect signal; local/system scores remain separately labeled.

## Decision

Keep the baseline; do not start another architecture or threshold sweep. The
next useful bounded diagnosis is raw-input/per-channel-error inspection around
the existing 2004 calibration false alarms and its positive event, with training
nominal examples for context. This can test whether ordinary transitions,
coverage gaps or a particular channel drive false alarms. Those are hypotheses,
not findings. Calibration error arrays were not saved by this historical runner,
so that deeper step requires fresh frozen-checkpoint inference, not retraining.
No such inference was launched here. Preserve the assessment as descriptive
evidence; do not tune rules to its nine known false alarms.

## Evidence and verification

Source directories: `results_longrun/development/ae_baseline_2003_seed42/2003`
and `results_longrun/development/ae_baseline_remaining_seed42/{2004,2005}`.
The review checks earliest-best selected epochs, reconstructs every prediction
from the saved score and frozen rule, and reproduces system metrics against saved
results. Local metrics use the same predictions and retain outside-input events
as neutral. Timestamp grids follow the declared complete30-second fold periods.
Nominal monthly statistics exclude bins touching any annotation; they are
sample summaries, not the evaluator's exact nominal-duration calculation.

Reproduce with the research environment from repository root:
`PYTHONPATH=. python scripts/research/review_baseline_years.py`.
Tables: `tables/baseline-years-2026-09-25/` (summary, histories, event IDs,
alarm intervals, monthly nominal score summaries). No benchmark2007–2013 access.
