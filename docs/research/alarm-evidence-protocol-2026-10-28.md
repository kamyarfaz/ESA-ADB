# Three post-processing families on three fixed models — 28 October 2026

Authorized by the user's request to test all three proposed methods on all three
techniques. No training or MOSAD. Reuse verified calibration scores from the
completed temporal study and the same W&B-selected models/testing scores.
Train 2000–2005, calibrate 2006, assess 2007–2013; targets 40–47, with system
scope also reported. Only expanded short-no-roll has extra inputs 14,21,29.

Freeze this protocol before calibration evaluation. Each family gets its own
unchanged fallback and selected rule for each model (nine comparisons).
30-second grid, strict score > threshold. No gap merging in these new families.
Duration grid in samples: 1,32,48,64,96,128,256,512 (0.5–256 minutes).
1. Strength-duration: remove a baseline-threshold segment only when its length
   is below duration AND its peak/threshold is below 1.1,1.25,1.5,2,3,5.
   Thus strong short events survive; long weak segments can survive too.
2. Hysteresis: low threshold factors 0.5,0.75,1; high factors 1,1.1,1.25,1.5,2.
   Keep connected low-threshold segments containing a sample strictly above
   the high threshold. This is offline connected-component hysteresis.
3. Duration: remove baseline-threshold segments shorter than each duration.
Each family includes an explicit original prediction fallback, even if duplicated.
49+16+9=74 candidates/model; 222 total. No grid extension after assessment.

Select highest corrected local calibration event F0.5 subject to retaining the
exact baseline-detected local IDs and <=1% total alarm fraction. Tie: least
nominal false-alarm seconds, original fallback, then candidate index. Record all
candidates and rejected choices. Baseline must satisfy constraints or fail closed.
Calibration segment diagnostics record length, peak, mean excess, overlapping
local IDs and nominal false-alarm seconds. Diagnostics do not expand this grid.
Freeze all nine selections together before opening assessment arrays. Report
all nine selected outcomes and three shared unchanged controls (192 annual/pooled
rows), with gained/lost IDs and duration diagnostics. No selecting a family based
on assessment. Calibration retention cannot guarantee assessment retention.

Verify destination frozen dataset hashes and all predecessor receipts first.
Fingerprint source/runtime/settings, runner/protocol and calibration receipt.
Use a new directory, exclusive lock and matching protocol on resume. Verify again
at end. Independent verifier uses a separately coded segment transform, replays
all calibration selections, assessment predictions and corrected metrics.
No training or inference needed; source weights remain unchanged.

Single seed, reused benchmark and small calibration event set limit inference.
Longer cutoffs can remove real anomalies; low hysteresis thresholds can increase
false-alarm time. Existing overlapping-window scores are not independent samples.
Publish unsuccessful results as well. Service esa-alarm-evidence.service; output
results_longrun/alarm_evidence_20261028_v1; log
/home/k_faz/project-backups/run-logs/alarm-evidence-20261028.log.
