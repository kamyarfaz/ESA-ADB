# Short-no-roll checkpoint fine-tuning — 23 October 2026

The user authorized continuing fine-tuning of removing short rolls. Interpret this
as the original eight-input short-no-roll Transformer AE, not expanded inputs or
masked pretraining. Preserve the original completed experiment unchanged.

Start from its calibration-selected epoch10 model_checkpoint.pt and scaler in
results_longrun/short_no_roll_full_20261014_seed42_v1/full_period.
Continue training for exactly10 additional epochs using AdamW reset to LR1e-5,
weight decay unchanged, cosine schedule to1e-6. This is one prespecified continuation,
not a hyperparameter search. Original training LR was1e-4. Seed42, batch64,
all nominal training windows, unchanged short_no_roll augmentation/objective,
architecture, clipping, windowing and score aggregation. No real anomaly supervision.
Deterministic epoch order42+epoch; optimizer/scheduler/RNG checkpoints every100 batches.

Inputs40–47; train2000–2005; calibration2006; continuous assessment2007–2013.
Select from original checkpoint (continuation epoch0), epochs5 and10 using corrected
calibration system event F0.5, original/earliest on exact tie. Original decision
rule is retained as its candidate. New checkpoints use the same percentile grid,
merge gaps[0,16,64], minimum durations[1,8,32], and1% alarm cap as prior runs.
No assessment-driven candidate changes, checkpoints or year-specific settings.
Both event scopes and all annual/pooled metrics are reported. Selection objective
remains system F0.5; local40–47 results are a secondary report, not a selection switch.
Original checkpoint is a fallback, not a guarantee of improved assessment performance.
Previously reused benchmark, one seed and additional training compute must be disclosed.

Runner esa_thesis/short_no_roll_finetune.py; output
results_longrun/short_no_roll_finetune_20261023_seed42_v1.
Service esa-short-no-roll-finetune.service; log
/home/k_faz/project-backups/run-logs/short-no-roll-finetune-20261023.log.
MemoryHigh6GiB/Max7GiB/swap512MiB are limits, not resource reservations.
Verify destination frozen data hashes and original artifact receipts before work
and on completion. This does not certify unrelated historical migration content.
Strict source/config/runtime/protocol fingerprints block changed-settings resume.
Keep MOSAD and previous completed services stopped. Never duplicate workers.

After completion independently verify training history, frozen selection, prediction
replay and all32 original/selected annual/pooled metric rows. Report all candidates
including failures, update the workbook without erasing historical entries, preserve
artifacts in a checksum-verified Release and compact results in Git. Disable the
service and remove the follow-up after delivery. Do not tune further on these results.
