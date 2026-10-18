# Normal-reference similarity and MOSAD: frozen full-period comparison

## Authorization and scope

On 18 October 2026 the user explicitly authorized both methods, sequentially,
using all eight inputs channel_40 through channel_47, training 2000–2005,
calibration 2006 and assessment 2007–2013. This authorizes full-period work despite
the earlier suggestion to begin with a development pilot. It supersedes historical
migration/no-training notes for this run; it does not retroactively pass any gate.
The original damaged historical MLP array is not a dependency. Frozen files and
raw annotations were independently checksum-verified on this destination before
launch; do not describe the entire old migration as verified.

Primary output is corrected system-level event F0.5, event precision, event TPR
and duration-based nominal FPR for each year and the pooled seven-year interval.
Secondary input-channel scope is retained for continuity, not used to select a
winner. Target categories remain Anomaly and Rare Event, matching the existing
baseline; communication gaps stay neutral. All-channel events remain in the
system denominator even when input channels have no annotation for an event.

## Common controls

Use immutable data/frozen/mission1-reference-v1/train.csv, calibration.csv,
testing.csv and labels.csv.gz, verified against the original manifest SHA256.
Input order 40–47, cadence30seconds, window256, training stride16, score stride32.
Fit median/IQR scaling using globally nominal 2000–2005 observations only; clip
scaled values at10 as in the baseline. Never fit statistics on assessment data.
Run count one, seed42, matching the completed full-period baseline. No seed search.

Select decision rules from 2006 using the existing corrected event evaluator,
percentiles90,95,97.5,98,99,99.5,99.75,99.9,99.95,99.99,100; merge gaps0,16,64;
minimum durations1,8,32; maximum total predicted fraction1%. Existing tie-breaking
prefers less false-alarm duration and then the higher threshold. This cap is a
calibration constraint, not a guaranteed test FPR. Freeze each rule before its
assessment observations are loaded. Both methods/settings are specified before
either assessment. Never alter MOSAD after seeing similarity assessment results.

Overlap-average window scores to the original time grid. Apply postprocessing
once across2007–2013 before annual slicing. Score the pooled continuous interval
independently; do not average annual F0.5. No annual refitting or test tuning.
No ensemble or alarm union is included in this request's first comparison.

## Method 1: fixed normal-reference subsequence similarity

This is a new ESA adaptation of subsequence-distance detection, not a claimed
reproduction of DAMP or SMED. Reference library: up to4096 globally nominal,
complete256-sample windows per training year, evenly spaced over eligible starts
within that year, maximum24576. Keeping a bounded temporal reference represents
all six years without exhaustive quadratic comparisons against every training
window. All eight channels share each reference window, preserving their joint
pattern. A reference window must not cross a calendar-year boundary.

Score = minimum mean squared Euclidean distance to any reference across all
8×256 values, exhaustive FP32 search in blocks of4096, TF32 disabled. There is
no per-window z-normalization: sustained level/amplitude changes must remain
visible. Larger distance means less similarity. No additional channel weighting,
embedding learning, contamination updates or assessment-derived reference windows.
Fit once, calibrate once. This preserves temporal structure unlike the completed
pointwise telemetry-reference diagnostic. Joint averaging may dilute a single
channel deviation; that limitation is declared rather than tuned away afterward.

Reference: https://www.cs.ucr.edu/~eamonn/DAMP_long_version.pdf . DAMP's evolving
left-discord reference and normalization differ from this frozen implementation.

## Method 2: MOSAD ESA adaptation

Pinned author source https://github.com/Armanfard-Lab/MOSAD at
cf935f817a89bb773a0a999d9117e33322970920. Source hashes and original LICENSE are
preserved in esa_thesis/vendor/mosad. README and LICENSE disagree; actual supplied
CC BY-NC-SA text is retained, alongside embedded Apache attribution notices.
Do not call this an exact reproduction of the paper's scores.

Retain TCN encoder, channel-masked reconstruction head, absolute deviation head,
and anomaly-aware contrastive head. Author config: embedding120, projection32,
TCN channels20,8 layers plus output layer, kernel7, pooling8, normalized embedding,
dropout0.2 for deviation head, temperature0.07, deviation margin5. Adam AMSGrad,
lr0.001, weight decay0.00001. All globally nominal training windows are shuffled
per epoch (batch64). No nominal-window budget cap.

Real supervision: select one centered window per unique training event ID having
an Anomaly or Rare Event annotation on at least one of inputs40–47. Center on the
longest observed input interval; ties use channel/start order. Require an actual
sample within that interval; reject communication-gap overlap. Exclude other-channel
only events from positive training examples. Sample10 such windows with replacement
per normal batch, retaining the event-ID inventory and starts. No calibration or
assessment labels enter the gradient objective. This supervision is an advantage
unavailable to the normal-only reference detector and must be visible in reports.
It is an algorithm comparison, not a supervision-matched causal ablation.

Keep author COE/mixup rates0.05. Repair the upstream discarded endpoint sort and
zero-length COE swaps. Compute the same contrastive objective for normal anchors
only without undefined singleton-anomaly denominators. Package import changes do
not change the architecture. For the fixed256-sample input, replace adaptive
max pooling with eight nonoverlapping32-sample maxima: identical forward and
backward including ties, but supported by deterministic CUDA backward. Soft mixup labels and author's normal-target selection
are retained. Original source attribution and adaptations appear in NOTICE.md.

Use40 epochs and2006 checkpoint selection every5 epochs, earliest F0.5 tie,
matching the baseline's checkpoint-opportunity count. This intentionally differs
from the author30-epoch/early-stop setup. Freeze64 training-only nominal projection
references for deterministic contrastive inference instead of resampling at every
batch. Fit reconstruction min/max on4096 fixed nominal training reference windows
for each candidate checkpoint. Do not clip transformed reconstruction scores.
Sum the normalized reconstruction, deviation and contrastive scores as in MOSAD.

Paper: https://arxiv.org/html/2310.12294v3 . Training labels, sequence length,
preprocessing, epoch selection, deterministic inference and event evaluation are
explicit ESA adaptations. Assess all three heads together; no test-based ablation.

## Recovery, verification and reporting

Runner: python -m esa_thesis.full_similarity_mosad --output
results_longrun/similarity_mosad_full_20261018_seed42_v1 [--resume]. One advisory
process lock covers both methods. Immutable protocol includes source/data hashes,
GPU/software, settings, original vendor identity. No source changes during a run.
Disable TF32 and enable deterministic algorithms (CUBLAS_WORKSPACE_CONFIG=:4096:8).
Checkpoint MOSAD every100 batches and at epoch transitions, preserving optimizer,
RNG and exact next batch; atomic file replacement. Resume only identical protocol.
Completed inference stages have file-hash receipts; partial inference is recomputed
with the same checkpoint/settings, not tuned. A persistent user systemd service
runs similarity then MOSAD automatically. No duplicate worker or blind restart.

Before launch: brute-force distance parity and amplitude sensitivity; nominal
window exclusion; training/future/input-label boundaries; upstream contrastive and
inference parity on well-defined cases; finite gradients, RNG restoration and
fixed-reference repeatability; real-config GPU smoke; corrected evaluator parity
and full thesis suite. Synthetic checks never inspect assessment scores.

On completion independently verify original data, saved predictions/rules and all
annual/pooled metrics. Compare both with completed short-no-roll baseline in a
compact table; include supervision, reference/training budget, seed/run count and
wall time. Preserve arrays/checkpoints outside Git and back them up in verified
GitHub Releases. Summaries and code belong in source Git. Update the system-only
thesis workbook after verification. No name of a meeting participant in artifacts.

The benchmark has already been inspected. Results are a reused-benchmark,
single-seed comparison, not independent confirmation or proof of robustF0.5≥0.85.
The earlier frozen internal checkpoint's cross-platform calibration mismatch
remains historical evidence; this new experiment does not erase or repair it.

Prelaunch verification:132 thesis tests passed, real-config CUDA training/scoring
smoke passed,46 usable training event windows verified. GPU smoke took about8s
per training batch; full training may take multiple days. No reduced data budget
or early assessment-driven stopping is introduced to shorten the run.
