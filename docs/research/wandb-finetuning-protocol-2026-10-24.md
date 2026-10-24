# W&B three-method fine-tuning protocol — 24 October 2026

User authorized nine sequential runs: three learning rates per method and ten
additional epochs per run. This new authorization supersedes completed-run stop
notes only for this study. Methods: original short-no-roll; masked pretraining
then short-no-roll fine-tuning; short-no-roll with extra inputs.
Extra channels14,21,29 belong only to the last method. All target local events
remain40–47. MOSAD and unrelated experiments remain stopped.

W&B entity kamyarfaz-politecnico-di-torino, project esa-thesis-finetuning.
Historical originals are imported separately under historical-baselines with
job_type historical-import. Their assessment results are context, not tuning
objectives. New runs use group learning-rate-comparison-20261024-seed42-v1.
This is a fixed local grid recorded in W&B, not an adaptive W&B sweep.

Continue each original selected checkpoint, retaining its training scaler and
architecture. Training2000–2005, calibration2006, assessment2007–2013.
Three rates fixed before runs:3e-6,1e-5,3e-5. Ten additional epochs each, AdamW
reset, cosine schedule to1e-6, original weight decay and short_no_roll objective.
Seed42, batch64, all globally nominal windows; no real anomaly supervision.
Original short-no-roll at1e-5 repeats the earlier bounded continuation and is
identified as a repeat, not independent evidence. Starting models had different
training histories; compare methods with these compute differences disclosed.

Checkpoints at continuation epochs5 and10 are calibrated using the original
threshold quantiles, gap[0,16,64], duration[1,8,32] and1% total alarm cap.
Original selected model/rule is eligible as epoch0 for each run. Within runs,
strictly greater corrected system calibration F0.5 replaces the earlier choice.
Across learning rates select highest calibration system F0.5, original/earliest
epoch on exact tie, then lower LR. Local40–47 metrics are secondary, not another
selection objective. Freeze all three method selections after all nine runs
before opening assessment. Evaluate only selected models and unchanged controls.
No decisions, extra candidates or per-year settings from assessment results.
All attempts, calibration score arrays, candidate weights/grids and histories
are preserved. Do not promise improvement or robust target achievement.

Local checkpoints every100 batches store optimizer,scheduler,RNG and selection.
Stable W&B IDs with resume=allow preserve cloud run identity. Cloud logging
restores training RNG and local histories remain authoritative if logs are delayed
or replayed after interruption. W&B metadata/metrics only; no raw telemetry or
checkpoint uploads to W&B in this phase. Cloud init failures stop before training;
inspect and resume same worker, no duplicate processes or blind restart loops.
Do not change package source/data/settings/protocol while the service runs.

Runner esa_thesis/wandb_finetuning.py.
Output results_longrun/wandb_finetuning_20261024_seed42_v1.
Service esa-wandb-finetuning.service; log
/home/k_faz/project-backups/run-logs/wandb-finetuning-20261024.log.
MemoryHigh6GiB/Max7GiB/swap512MiB are limits, not reservations.
Frozen destination dataset hashes, raw annotations, starting-model receipts and
source/runtime/config/protocol hashes verified before work and data again after.
This does not certify unrelated historical migration contents.

After completion verify receipt hashes, all candidate histories and selection,
calibration score replay, frozen selection, assessment prediction replay, detected
and missed IDs, and all96 annual/pooled metric rows across3 methods x2 variants
x8 periods x2 scopes. Publish verified assessment summaries to separate W&B
evaluation runs, never sweep metrics. Update report and approved35-row workbook,
preserving historical rows and unsuccessful outcomes. Save compact results to Git
and essential artifacts to checksum-verified GitHub Release kamyarfaz/ESA-ADB.
Single seed, multiple calibration comparisons and reused benchmark require
explicit disclosure. Disable completed service and delete monitoring after delivery.

W&B API references:
https://docs.wandb.ai/models/runs/resuming
https://docs.wandb.ai/models/ref/python/experiments/run
