# Temporal post-processing of three fixed models — 28 October 2026

User authorized merging close detections then removing short segments for all
three latest selected techniques, with scheduled monitoring. This supersedes the
prior completed-study stop only for this post-processing experiment. No training.
Use W&B study selection_complete.json from20261024: original short-no-roll,
original masked pretrain/fine-tune and expanded short-no-roll LR3e-5 epoch10.
Inputs remain40–47 for first two and40–47 plus14,21,29 for third. Training2000–2005,
calibration2006, assessment2007–2013. Thresholds, weights and scalers stay fixed.

Earlier threshold grids already included coarse temporal processing; their
selected rules had gap0 and duration1. This isolates a finer temporal grid and
uses the latest expanded model. It is not a novel algorithm or independent test.

30-second samples. Gap values0,1,2,4,8,16,32,64 samples (0–32minutes).
Minimum durations1,2,4,8,16,32 samples (30seconds–16minutes). Fill internal zero
gaps of at most the gap parameter, then remove resulting segments shorter than
the minimum. Duration is the merged segment span including filled gaps. Never
extend leading/trailing alarms; process continuously before annual slicing.
This is offline processing with potential lookahead, not a real-time claim.

For each method evaluate all48 candidates on2006 only. Require retention of the
exact baseline-detected local40–47 IDs and total alarm fraction<=1%. Original
rule is fallback. Maximize corrected local40–47 event F0.5; ties favor lower
nominal false-alarm seconds, then unchanged rule, then smaller gap and duration.
System metrics are reported but not used to choose a second configuration.
No changed rule qualifies: retain baseline. All three selections and candidate
hashes must freeze before assessment score access. Report original and selected,
including failures, all96 annual/pooled rows and detected/missed/gained/lost IDs,
alarm segment counts, total alarm duration and nominal false-alarm duration.
No grid extensions or decisions from assessment. Same reused benchmark, seed42,
unequal prior compute and extra-input limitations; no robust target claim.

Use existing corrected evaluator and postprocess implementation. Verify frozen
destination datasets, original source/receipt hashes and selected input artifacts.
New output results_longrun/temporal_postprocessing_20261028_v1; runner
scripts/research/run_temporal_postprocessing_20261028.py; service
esa-temporal-postprocessing.service; log
/home/k_faz/project-backups/run-logs/temporal-postprocessing-20261028.log.
Lock and strict protocol checks prevent duplicate or changed resumes. Completed
calibration inference is reused with receipts; interrupted inference restarts
that inference only. No training changes. Pin source/runtime/protocol. Verify
all outputs independently, publish W&B and report/workbook preserving history,
Git and checksum-verified Release, then disable service/delete follow-up.
