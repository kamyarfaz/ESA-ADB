# Pre2006 local shape inventory and next experiment

Channels40–47,2000–2005, Anomaly+RareEvent annotations only. No2006–2013
annotations used in this inventory. Within each eventID, merge overlapping local
intervals into episodes; count peak unique channels overlapping for positive
duration. No inference, training or threshold changes. Source hashes stable.

46eventIDs contain79local episodes.32episodes across13IDs are shorter than the
current6minute synthetic minimum.56episodes across40IDs involve more than the
current maximum4simultaneous synthetic channels.16episodes across8IDs are both
short and wide: id_104,id_151,id_78,id_80,id_85,id_87,id_95,id_98.
This recurs beyondid_98. Counts are descriptive, not independent physical-event
counts; shared mechanisms and broad annotation support limit interpretation.
Pre2006 includes previously inspected development periods. Annotation support
does not establish equal physical signal change in all annotated channels.

## Proposed focused experiment (not launched)

Keepchannels40–47, Transformer architecture, scaler, nominal objective and scoring.
Compare original generator with a mixture that retains original corruptions and
adds short1–10sample (30sec–5min) corruptions on coordinated channel subsets,
including5–8channels. Use a fixed50/50 mixture as an experimental design choice,
not a claimed optimal frequency. Preserve original generator as default; expose
an explicit optional mode. Do not change magnitude/type/duration/channel-count
laws in the original branch. Validate alignment, changed support and immutable
input behavior before running. Some temporal-roll cases may be no-ops on constant
segments; quantify this rather than claiming every injected sample is anomalous.

First test on bounded development folds with equal budgets and documented seeds;
report nominal reconstruction stability and both local/system event metrics.
Checkpoints/rules chosen on calibration, not assessment. This is not yet an
approved operational model or evidence of F0.5 improvement. Historical failed
masked/residual/timing runs are not to be rerun unchanged. Avoid committing to
another full40epoch run until the generator and protocol are reviewed/tested.

[Inventory and provenance](tables/local-event-shapes-2026-09-27/).
