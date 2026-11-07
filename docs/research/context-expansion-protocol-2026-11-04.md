# Matched input expansion across three techniques — 4 November 2026

Authorized by the user for all three techniques, with a recurring progress and
recovery follow-up. This is a new Mission 1 experiment, not a restart of an old
completed job. Mission 2 training and MOSAD are outside this study.

## Question and fixed comparison

Does a small training-selected input expansion improve corrected event F0.5 for
events annotated on channels 40–47? Report system scope as secondary context.
The three families are short-no-roll; masked pretraining followed by short-no-roll;
and short-no-roll retaining the previous auxiliary inputs 14,21,29.
For each family compare a fresh control against its input-expanded counterpart,
with seeds 42,43,44: 18 sequential training runs, not 18 techniques.
Controls use 40–47, 40–47, and 40–47 plus 14,21,29 respectively. Expanded versions
append the same two training-selected channels. Never select a seed on assessment.

## Input selection, fixed before inspecting the new rankings

Use the verified raw five-minute Mission 1 training grid and quarterly matrices
from `results_longrun/all_channel_correlations_20261102_v1`, covering 2000–2005
only. Choose one representative from 16,24,32,33 for anchor40 and one from
48,49,50,51,52 for anchor47. An eligible pair has finite, consistently signed
Spearman correlations and at least 500 paired samples in every one of 24 quarters.
Rank by highest minimum absolute quarterly Spearman correlation, then highest
absolute pooled Spearman, then lowest channel number. Stop if a pool has no
eligible member; do not introduce alternative channels. Independently replay the
selected candidates and all competing quarterly coefficients from the raw grid.
Report Pearson and first-difference correlation as diagnostics, not additional
selection objectives. This ranks temporal consistency, not incremental predictive
value or causal usefulness; high level correlation can reflect slow common trends.

## Fixed training and scoring

Prepared dataset: `data/frozen/mission1-reference-v1`, original manifest
`a49a8c42c23a5d72d91a48596f485d762a805cc6101ad037c01a36c8037bf099`.
Train 2000–2005; calibration 2006; assessment 2007–2013. Check all four prepared
file hashes and the original annotation hashes on this destination before launch
and before assessment. This scoped verification does not certify the full historic
migration. Preserve all historical artifacts and outstanding migration exceptions.

All models start fresh because input dimensions change. Transformer architecture,
normal-only median/IQR scaling and clipping, all globally nominal training windows,
256-sample windows, stride16, batch64, AdamW LR1e-4, weight decay1e-4, gradient
clip1 and cosine schedule to1e-6 are fixed. No window subsampling. Each run receives
40 epochs. Masked pretraining uses the first20 epochs (normal reconstruction only),
then transfers shared weights into the unmasked AE and resets the optimizer for
20 short-no-roll epochs. Other families use40 short-no-roll epochs. Thus the two
arms within a family have matched budgets. Expanded inputs have greater compute.
Permutation seed is seed+epoch. Checkpoints retain all model/optimizer/scheduler,
RNG, best candidate, history, batch and partial loss state every100 batches and
at epoch boundaries. No architecture or threshold changes during execution.

Retain the historical score definition: unmasked reconstruction error averaged
within each256-sample window, stride32, float32 overlap averaging, maximum across
all input channels. Additional channels therefore also supply alarm evidence;
this is an input-and-score expansion, not an isolated context-only ablation.
No new target-output pooling or post-processing family is introduced here.
Memory-bounded aggregation must match the original aggregation exactly.

Checkpoint/rule selection uses **40–47 calibration corrected event F0.5**, not
system F0.5. Evaluate epochs5,10,...,40 for short-no-roll;25,30,35,40 after masked
pretraining. Retain every evaluated checkpoint, score and grid. Use the existing
11 quantiles, gaps0/16/64 samples, minimum durations1/8/32 and total alarm-rate
cap1%. Rule ties follow existing `select_rows`; checkpoint ties retain earliest.
This differs from historical system-scope selection, so fresh controls are essential.
All18 models and their rules must jointly freeze before any assessment inference.
No fallback across seeds or families and no test-based selection. Historical
selected models remain unchanged references, not selection candidates here.

## Execution, verification and delivery

Output: `results_longrun/context_expansion_20261104_v1`.
User service: `esa-context-expansion.service`.
Log: `/home/k_faz/project-backups/run-logs/context-expansion-20261104.log`.
W&B project: `kamyarfaz-politecnico-di-torino/esa-thesis-finetuning`;
group `context-expansion-20261104-v1`, stable per-run IDs, online training logs.
Preserve source/protocol/data/runtime fingerprints and receipts. Use a single
worker, user-service RAM limits and no automatic restart loop. Inspect a failure,
available resources and absence of another runner before resuming the same job.
The wall-clock duration is not fixed:18 full training runs may require many days.

Assess all frozen models continuously over2007–2013. Report all288 annual/pooled
rows (18 models ×8 periods ×2 scopes), corrected F0.5, precision, event TPR,
time FPR, detected/missed IDs, and paired expanded-minus-control changes per seed.
Summarize three-seed means/spread without presenting the best seed as the result.
Independently verify calibration grid/selection, saved checkpoint inference spots,
prediction replay, all metrics and IDs, and original source/input receipts.
Publish verified assessment to separate W&B evaluation runs. Update the academic
report, approved thesis workbook (preserving historical rows and scope columns)
and AI_HANDOFF. Commit compact code/evidence; back up essential weights/scores/
receipts to a GitHub Release with fresh-download checksum verification. Disable
the completed service and delete its follow-up only after verified delivery.

Limitations: three seeds provide limited variability evidence; channel pools and
technique choices are informed by prior research; 2006 and2007–2013 are reused.
Repeated benchmark evaluation is not independent confirmation. Annotation-informed
benchmark preprocessing, offline overlapping windows and neutral out-of-scope
annotations remain inherited. Do not claim the thesis target is robustly achieved.
