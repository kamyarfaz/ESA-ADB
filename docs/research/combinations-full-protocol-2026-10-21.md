# Frozen combination comparison — 21 October 2026

## Authorization and scope

The user authorized all three previously proposed combinations, identical date
splits, execution without further routine confirmation, and a scheduled monitor.
They explicitly permitted extra inputs **only for approach 3**. MOSAD stays stopped.
This new authorization supersedes the earlier stop for these combinations only.

Use the immutable Mission 1 reference: training 2000–2005, calibration 2006,
continuous assessment 2007–2013. Check manifest and all four data-file hashes
before execution and again on completion. The unresolved historical MLP-array
transfer mismatch is excluded; this does not certify a lossless whole-project
migration. Raw annotations are independently checksum-gated.

## Arms and hypotheses

1. **Score fusion:** existing completed short-no-roll Transformer AE plus a new
   masked reconstruction AE. The masked model uses the existing legacy synthetic
   anomaly objective on hidden targets, four interleaved mask phases and 40
   epochs. It is also reported separately as the standalone control. Normalize
   each selected detector's scores by its 99th percentile on globally nominal
   **2006 calibration** samples. Choose masked weight from [0,.25,.5,.75,1] on
   corrected calibration system F0.5; ties favor the smaller masked weight.
   Endpoints allow calibration to reject the combination. This is weighted score
   fusion, not an OR of binary alarms. Its hypothesis is complementary errors.
2. **Masked pretraining then short-no-roll fine-tuning:** 20 epochs of masked
   nominal reconstruction only; transfer every shared parameter to an unmasked
   AE, discard the mask token, reset optimizer/scheduler, then 20 epochs with
   the existing short-no-roll objective. Only fine-tuned checkpoints are eligible
   for selection. This tests contextual initialization at a fixed 40-epoch total
   budget; the 20/20 allocation is fixed before results, not optimized on testing.
3. **Expanded-input short-no-roll:** the same 40-epoch short-no-roll objective
   with inputs 40–47 plus 14,21,29 in that order. This tests additional information
   together with augmentation. Extra input information is a deliberate difference,
   not an equal-input comparison and not a pure architectural ablation.

All other new-model settings match the existing AE configuration. Batch64,
seed42, all globally nominal length256/stride16 training windows; no training
subsampling, no real anomaly supervision. Deterministic epoch permutations are
seeded by 42+epoch and independent of resume position. Masks cycle with epoch
and batch. Training median/IQR scaling and clipping are unchanged. The historical
short-no-roll reference uses its recorded original shuffle trajectory, so it is
an existing matched-protocol reference, not an identical-RNG rerun.

The reference checkpoint and prediction hashes are verified against its completed
receipt. Replaying the reference must reproduce its saved binary predictions.
The ensemble has two 40-epoch components (80 total model-epochs including the
reused reference) and four masked scoring passes plus one unmasked pass. It is
**not compute-matched** to the single-model 40-epoch methods. All results are
single-seed and the assessment benchmark has been inspected before.

## Selection and evaluation

Calibrate at epochs5,10,...40 for masked and expanded; at total epochs25,30,35,40
for the sequential arm. Earliest checkpoint wins exact F0.5 ties. Use the existing
percentiles [90,95,97.5,98,99,99.5,99.75,99.9,99.95,99.99,100], merge gaps[0,16,64],
minimum durations[1,8,32], and 1% total-alarm cap. Freeze all model selections,
fusion normalization/weight and decision rules before loading assessment data.
No tuning or model selection using 2007–2013, including annual results.

Score continuously with window-mean errors, overlap averaging and maximum across
input channels. Report corrected event F0.5, precision, event TPR, time FPR and
event counts for each year and pooled 2007–2013. Two evaluation scopes are fixed:
all-channel system events and events affecting channels40–47. The latter remains
40–47 even for expanded inputs, preserving a common target scope. Pooled metrics
are recomputed, not annual averages. No claim of independent validation or a
robustly achieved thesis target is permitted.

## Reliability and completion

Runner: `esa_thesis/full_combinations.py`.
Output: `results_longrun/combinations_full_20261021_seed42_v1`.
Service: `esa-combinations.service`; sequential GPU jobs, no parallel workers.
Log: `/home/k_faz/project-backups/run-logs/combinations-full-20261021.log`.
MemoryHigh6GiB, MemoryMax7GiB, swap512MiB; these are limits, not reserved RAM.
CSV loading is chunked into preallocated feature arrays. Checkpoints every100
batches plus epoch boundaries preserve model, optimizer, scheduler and RNG.
A process lock excludes duplicate comparison runners. Resume requires identical
source/data/config/runtime fingerprints; don't modify these during execution.
Service is enabled for reboot recovery; failures require monitor inspection, not
blind restart loops. The shared server can still interrupt work; no guarantee of
completion by tomorrow. Do not stop other users' processes or reserve their RAM.

Monitor every30 minutes. On interruption inspect cause, confirm no worker remains,
and resume the same service only when resources permit. Preserve artifacts and
protocol; no silent batch-size, precision, architecture or training-budget changes.
At completion independently replay predictions/metrics, verify frozen hashes and
receipts, generate the academic comparison and update the user's existing workbook
format without overwriting historical results. Treat `complete.json` as runner
completion only: it explicitly requires independent verification. Save compact
results in Git and essential artifacts in a verified Release. Disable the service
and delete the monitor after delivery. No new experiments or MOSAD restart.
