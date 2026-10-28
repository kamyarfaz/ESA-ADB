# AI handoff — ESA satellite anomaly-detection thesis

Updated **23 October 2026**. This is the current entry point for a new assistant.
Read this before proposing work. The chat is not required to continue the project.

## 1. Current task and decision

**28 October 2026 — temporal post-processing completed and independently verified.**
All three latest W&B models retained unchanged processing (gap 0, minimum 1 sample).
The 144 calibration candidates produced no selected improvement; all 96 annual/pooled
assessment rows and prediction replays match unchanged controls. Calibration segments
were at least 16 minutes: the tested duration cutoffs (up to 16 minutes) removed none.
The first two methods had no gaps within the tested 32-minute merging range; expanded
inputs had some merges but no objective gain. Larger cutoffs remain untested.
All 147 thesis tests passed. W&B cloud verification downloaded three finished evaluation
runs and all 96 metric rows. Workbook numeric/history cells preserved; only A48 note updated.
Report: docs/research/temporal-postprocessing-results-2026-10-28.md.
Evidence: docs/research/tables/temporal-postprocessing-2026-10-28/.
Service esa-temporal-postprocessing.service is inactive and disabled. Release publication
and monitoring cleanup are being finalized; consult github-backup.json for verified backup.
No further experiment authorized. Single seed, reused benchmark, offline lookahead,
small calibration event set and extra inputs limit conclusions. No robust target claim.


**25 October GitHub preservation check:** completed study and workbook are on
origin/main; Release asset sizes and server SHA256 digests match the verified
backup receipt. Added the W&B analysis guide from the conversation and preserved
the historical MOSAD recovery note with an explicit stopped/superseded notice.
New investigation ideas are proposals only; no new training is running.

**25 October 2026 — W&B nine-run study completed and independently verified.**
All nine ten-epoch runs finished. Calibration kept original short-no-roll and
masked-pretraining models; expanded-input short-no-roll selects LR3e-5 epoch10.
Assessment 2007–2013: local40–47 F0.5 0.8173525522 -> 0.8308996735,
detections61/67 -> 62/67; system F0.5 0.8244050148 -> 0.8445732669,
detections76/91 -> 80/91. No lost IDs. Local gained id_49; system additionally
id_152,id_162,id_171. Original short-no-roll still has best local F0.5
0.9071395304 (47/67). No new experiments authorized by this completion.
Independent audit passed frozen data/source/runtime/settings/reference hashes,
18 full calibration grid replays, selected weights/scalers, exact prediction
replay, all96 annual/pooled metric rows and event IDs. All144 tests passed.
All nine W&B training runs finished; three separate evaluation runs published,
cloud summary files and all96 table rows downloaded and verified. Service is
inactive and disabled. Canonical workbook updated with35 technique rows and all
24 historical rows preserved; prior workbook retained in outputs backup.
Report: docs/research/wandb-finetuning-results-2026-10-25.md; evidence under
 docs/research/tables/wandb-finetuning-2026-10-25/.
Single seed, reused benchmark, extra inputs, unequal prior compute and repeated
short-no-roll LR1e-5 setting are disclosed. No robust target claim.
Release https://github.com/kamyarfaz/ESA-ADB/releases/tag/wandb-finetuning-2026-10-25
contains all run artifacts and reporting assets; all four assets downloaded and
SHA256/size verified (github-backup.json). Completion delivered in this chat;
monitoring follow-up removed. No further training scheduled.

**25 October09:46 CET milestone:** run8/9 (expanded_short_no_roll_lr2, LR1e-5)
finished; trained receipt hashes and ten history epochs verified. Unlike prior
runs, calibration selected continuation epoch10: system F0.5 0.97798520773566
versus original0.9778648035582577 (small calibration-only increase, not evidence
of assessment improvement). Same PID706712 started final run9/9
(expanded_short_no_roll_lr3, LR3e-5), epoch1 batch700/5461. W&B confirms run8
finished and run9 recording. Final cross-rate selection and assessment pending.

**25 October07:16 CET milestone:** run7/9 (expanded_short_no_roll_lr1, LR3e-6)
finished; trained receipt hashes and ten history epochs verified. Calibration
retained original epoch0 (system F0.5 0.9778648035582577). Same worker PID706712
started run8/9 (expanded_short_no_roll_lr2, LR1e-5), epoch1 batch4500/5461.
W&B confirms run7 finished and run8 recording. No assessment yet or settings changes.

**25 October04:46 CET milestone:** all three masked-pretraining/fine-tuning
runs finished (6/9 total). Run6 trained receipt hashes and ten history epochs
verified; all three method2 rates retain original epoch0 (system calibration
F0.5 0.7892758701516437). Same worker PID706712 started method3, run7/9
(expanded_short_no_roll_lr1, LR3e-6), epoch2 batch3000/5461. W&B confirms
run6 finished and run7 recording. No assessment yet; frozen protocol unchanged.

**25 October02:16 CET milestone:** run5/9 (pretrain_finetune_lr2, LR1e-5)
finished; trained receipt hashes and all ten history epochs verified. Calibration
retained original epoch0 (system F0.5 0.7892758701516437). Same worker PID706712
started run6/9 (pretrain_finetune_lr3, LR3e-5), epoch1 batch4300/5461. W&B
confirms run5 finished and run6 recording. No assessment yet or settings changes.

**25 October01:16 CEST milestone:** run4/9 (pretrain_finetune_lr1, LR3e-6)
finished; trained receipt hashes and all ten history epochs verified. Calibration
retained original epoch0 (system F0.5 0.7892758701516437). Same worker PID706712
started run5/9 (pretrain_finetune_lr2, LR1e-5), epoch2 batch5300/5461. W&B
confirms run4 finished and run5 recording. No assessment yet or settings changes.

**24 October22:46 CEST milestone:** all three original short-no-roll learning
rate runs finished. Run3 trained receipt hashes and ten history epochs verified;
all three retain original epoch0 (system calibration F0.5 0.8821382469600005).
Same worker PID706712 automatically started method2, run4/9
(pretrain_finetune_lr1, LR3e-6), epoch1 batch4700/5461. W&B confirms run3
finished and run4 recording. No assessment yet; frozen protocol unchanged.

**24 October20:46 CEST milestone:** run2/9 (short_no_roll_lr2, LR1e-5)
finished; all trained receipt hashes and ten history epochs verified. Calibration
again retained original epoch0 (system F0.5 0.8821382469600005). This setting
repeats the prior continuation; no independent confirmation claimed. Same worker
PID706712 advanced automatically to run3/9 (LR3e-5), epoch3 batch300/5461.
W&B API confirms run2 finished and run3 recording. Assessment remains unopened;
frozen settings unchanged.

**24 October18:16 CEST milestone:** W&B run1/9 (short_no_roll_lr1,
LR3e-6) completed all10epochs; trained receipt hashes and complete epoch history
verified. Calibration retained original epoch0 (system F0.5 0.8821382469600005).
No assessment performed yet. Same worker PID706712 automatically started run2/9
(short_no_roll_lr2, LR1e-5), epoch1 batch5200/5461. W&B API confirms first run
finished and second running with current metrics. Frozen protocol unchanged.

**24 October ACTIVE — authorized W&B nine-run fine-tuning:** user approved three
learning rates (3e-6,1e-5,3e-5), ten additional epochs each, sequentially for
original short-no-roll, masked pretraining plus fine-tuning, then expanded-input
short-no-roll. This authorization supersedes prior completed-run stop notes only
for this study. See docs/research/wandb-finetuning-protocol-2026-10-24.md.
All144 thesis tests passed. Service esa-wandb-finetuning.service is enabled and
started (initial PID706712); checks frozen destination data before training.
Output results_longrun/wandb_finetuning_20261024_seed42_v1; log
/home/k_faz/project-backups/run-logs/wandb-finetuning-20261024.log.
Heartbeat complete-w-b-nine-run-fine-tuning checks every30minutes, quiet unless
material progress/failure/completion. Never duplicate workers or change frozen
package source/data/protocol during the run. MOSAD remains stopped.
W&B project https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning
already contains three historical-baselines; receipt outputs/wandb-20261024/baseline-import.json.
New runs group learning-rate-comparison-20261024-seed42-v1, stable IDs and online
logging. No raw telemetry or weights uploaded to W&B. Training2000–2005,
calibration2006, assessment2007–2013. Inputs40–47; extra14,21,29 only expanded.
Original fallback eligible; select by system calibration F0.5, freeze all three
selections after nine runs before assessment. Preserve all attempts. Original
short-no-roll1e-5 repeats prior setting; disclose this and single-seed/reused
benchmark limitations. On completion independently verify all96 metric rows,
calibration selection/replay and assessment replay, publish separate W&B evaluation
runs, update report/workbook, Git and checksum-verified Release, then disable
service and delete follow-up. No improvement claimed before verified results.


**23 October COMPLETED — original short-no-roll fine-tuning:** ten additional
epochs at LR1e-5 completed successfully. Neither epoch5 (calibration F0.5
0.882118655982629) nor epoch10 (0.882099064642808) beat the original epoch10
checkpoint (0.8821382469600005), which remains selected as continuation epoch0.
Original weights, scaler, rule and assessment predictions match exactly.
No improvement: system F0.5 0.836070,48/91 events; local40–47 F0.5 0.907140,
47/67 events. All32 annual/pooled metric rows independently verified, frozen
hashes and receipts checked, no gained/lost event IDs.141 tests passed.
No further experiments or test-based tuning are authorized by this completed run.
Report docs/research/short-no-roll-finetune-results-2026-10-23.md; verified tables
docs/research/tables/short-no-roll-finetune-2026-10-23/.
Workbook outputs/short-no-roll-finetune-20261023/deliverables/ESA_thesis_results_short_no_roll_finetuning.xlsx
preserves all35 technique rows and original scores; status notes original retained.
Source/run artifacts: results_longrun/short_no_roll_finetune_20261023_seed42_v1.
Service esa-short-no-roll-finetune.service is inactive/successful and disabled.
Follow-up complete-short-no-roll-fine-tuning is removed after delivery.
Release https://github.com/kamyarfaz/ESA-ADB/releases/tag/short-no-roll-finetune-2026-10-23
with downloaded checksum evidence in tables/github-backup.json.
Single seed, reused benchmark and extra training compute are disclosed.
MOSAD remains stopped. Frozen destination dataset verification passed, not a
whole-project migration certification or replacement for external-drive backup.

**23 October COMPLETED — false-alarm reduction for both models:** calibration-only
research and frozen reassessment finished, with no neural retraining. All 64 annual
and pooled rows independently verified, including frozen fingerprints, original
score parity, prediction replay and event IDs. All 140 thesis tests passed.
Pretrain/fine-tune retained its original rule: system/local F0.5 0.795675/0.844315,
61/91 and 60/67 events. Expanded-input threshold +10% retained calibration IDs
and reduced calibration nominal alarm duration 18.24%, but failed on assessment:
system F0.5 0.824405 → 0.522741; local 0.817353 → 0.484623;
events 76→73 system and 61→60 local. False alarm events rose 11→70 because
22 original intervals fragmented. Time FPR only fell 5.930307%→5.8036%.
Neither refinement improved results. Do not tune further using these outcomes.
One seed and reused benchmark; no robust target or independent-validation claim.
Report: docs/research/false-alarm-reduction-results-2026-10-23.md.
Verified tables: docs/research/tables/false-alarm-reduction-2026-10-23/.
Workbook preserves all 35 technique rows and better original scores; failed
refinement remains fully reported. Output workbook:
outputs/false-alarm-reduction-20261023/deliverables/ESA_thesis_results_alarm_refinement.xlsx.
Service esa-false-alarm-reduction.service completed successfully and is disabled.
Run artifacts remain results_longrun/false_alarm_reduction_20261023_seed42_v1.
Release: https://github.com/kamyarfaz/ESA-ADB/releases/tag/false-alarm-reduction-2026-10-23
Release checksum evidence is in the verified tables github-backup.json.
The monitoring follow-up is removed on completion. MOSAD remains stopped.
Dataset destination verification passed; this does not certify unrelated historical
migration contents or replace the external-drive copy.

**22 October COMPLETED — all three combinations:** all training and continuous
2007–2013 assessment finished successfully. Independent verification passed all
frozen data/source/reference/artifact hashes, complete 40-epoch histories,
calibration checkpoint/rule/fusion selection, exact prediction replay from saved
scores, and all 80 corrected annual/pooled metric rows from raw annotations.
Runtime fingerprint matched; 136 thesis tests passed. No second neural inference
run was performed. No test-based tuning or additional experiments were run.
Pooled system / channels40–47 F0.5: short-no-roll reference 0.836070 / 0.907140;
masked control 0.806278 / 0.882146; fusion 0.811670 / 0.884610;
pretrain/fine-tune 0.795675 / 0.844315; expanded-input short-no-roll
0.824405 / 0.817353. None improves pooled system F0.5. Extra inputs detect76/91
events but time FPR is5.930307%, compared with48/91 and0.029194% for reference.
Single seed, reused benchmark, extra inputs and ensemble compute are disclosed.
Do not claim independent validation or robust target achievement. MOSAD stays stopped.
Full academic report: `docs/research/combinations-full-results-2026-10-22.md`;
all annual/pooled metrics and verification in `docs/research/tables/combinations-2026-10-22/`.
Updated workbook: `outputs/combinations-20261022/deliverables/ESA_thesis_results_combinations.xlsx`.
All32 original technique rows retained (MOSAD status corrected to stopped), three
combination rows added, all2007–2013 labels first. Original historical masked rows
are preserved; full-period masked control is in the report/CSV. Original workbook
backed up under `/home/k_faz/project-backups/combinations-20261022/`.
Release `combinations-full-2026-10-22` preserves176 artifacts. All three uploaded
assets were downloaded and matched local SHA256/size; evidence is in
`docs/research/tables/combinations-2026-10-22/github-backup.json`.
The completed combinations service is inactive and disabled (MainPID0), as is
MOSAD. The completed `complete-three-esa-combination-comparisons` follow-up was
deleted. No active run or scheduled research remains for this comparison.
The updated workbook also replaces the project-root working copy, with its
original preserved in the Release and `outputs/combinations-20261022/build/template.xlsx`.
Older active milestones below are historical and must not trigger a restart.

**22 October06:24 CEST milestone:** pretrain_finetune completed all40epochs
(20 masked pretraining plus20 short-no-roll fine-tuning); all four trained-receipt
artifact hashes and history epochs1–40 verified. Same service/PID1466507 has
automatically started expanded_short_no_roll, now epoch2/40 batch700/5461,
finite losses and current checkpoint. Masked standalone also complete. Fusion
calibration and all assessment remain pending until training finishes; no test
scores or protocol changes. MOSAD remains stopped.

**21 October23:54 CEST milestone:** masked standalone completed all40epochs;
its trained receipt and all four artifact hashes verified, history epochs1–40
complete. The same service/PID1466507 automatically advanced to
pretrain_finetune, now epoch3/40 batch4100/5461 with finite loss and current
checkpoint. Expanded inputs remain queued. No assessment scores yet; fusion
and assessment follow all training under the unchanged frozen protocol.

**21 October NEW USER AUTHORIZATION — three combination runs:**
The user authorized testing all three proposed combinations and scheduled
monitoring. This supersedes the earlier stop for these combinations only;
MOSAD remains stopped and disabled. After explicit clarification, approach3 may
add channels14,21,29; approaches1/2 retain40–47. Every arm uses frozen train
2000–2005, calibration2006, continuous assessment2007–2013. Read
`docs/research/combinations-full-protocol-2026-10-21.md` for the exact protocol.
New runner `esa_thesis/full_combinations.py`: masked40-epoch standalone feeding
score fusion with the verified existing short-no-roll reference; masked nominal
pretraining20 then unmasked short-no-roll fine-tuning20; expanded short-no-roll40.
Single seed42, batch64, all nominal windows. Ensemble compute and extra inputs
must be disclosed; no test tuning and no independent-validation claim.
Service `esa-combinations.service` enabled and started, startupPID1466507.
Output `results_longrun/combinations_full_20261021_seed42_v1`;
log `/home/k_faz/project-backups/run-logs/combinations-full-20261021.log`.
MemoryHigh6GiB/Max7GiB/swap512MiB; limits do not reserve shared-server RAM.
Atomic checkpoints every100batches and epoch boundaries; strict frozen protocol
and process lock. Resume same service after inspecting interruption, resources
and duplicate processes. Do not edit source/data/settings while running.
Tests:136 passed; exact CPU mid-epoch restart parity and CUDA masked/backward,
parameter transfer and short-no-roll/backward checks passed. Frozen file/reference checks passed and training launch confirmed: masked
epoch1/40 batch1/5461, finite loss9.49148, PID1466507 using1716MiB GPU.
The initial resumable checkpoint was saved; the other arms follow sequentially.
Active30-minute thread monitor `complete-three-esa-combination-comparisons`.
On completion independently verify results, preserve workbook historical rows,
report all annual/pooled metrics for both scopes, back up essential artifacts,
disable service and delete monitor. Earlier MOSAD follow-up stays deleted.

**21 October USER STOP instruction supersedes all prior run authorizations:**
Stop exploring new techniques and stop MOSAD now. Stopped and disabled
esa-similarity-mosad.service; verified inactive, MainPID0 and no runner.
Deleted complete-similarity-and-mosad-comparison automation. Preserve checkpoints
and artifacts; do not restart training or launch new experiments without new
user authorization. Current task is descriptive ranking of existing workbook
rows, irrespective of assessment period, using project-root
`ESA_thesis_results_full_and_historical.xlsx`. No workbook edits requested.

**21 October user status check and recovery:** Journal shows a server reboot
between09:49 and09:59 CEST, automatic service startup09:59, then explicit
OOM kill11:25:25 CEST. Saved checkpoint11:19:47, epoch5 batch3500/5461.
No runner remained; available RAM7.0GiB and swap9.3GiB. Restarted the same
service, unchanged source/protocol/memory limits, PID1237721 active.
Startup verification and new batch progress still need confirmation.
Preserved checkpoint at
`/home/k_faz/project-backups/run-logs/mosad-before-recovery-20261021.pt`.
No final MOSAD scores yet; first calibration remains due after epoch5.

**21 October03:30 CEST milestone:** MOSAD completed epoch4 and started
epoch5/40 under the unchanged frozen protocol. PID1296344 remains active,
RAM3.57GiB, swap0. Epoch-boundary checkpoint saved03:23:04 CEST.
First calibration evaluation is due after epoch5; no MOSAD assessment scores yet.

**20 October evening recovery:** systemd reports an explicit OOM kill at
21:00:28 CEST for PID3880307. Latest saved progress is epoch4 batch2700,
checkpoint timestamp18:25:36 CEST. At recovery inspection about20GiB RAM was
available (swap nearly full), no comparison worker remained. Started the same
service with unchanged limits, source and protocol; new PID1296344 is active,
Recovery confirmed at22:00 CEST: frozen input checks and loading passed,
training advanced through epoch4 batch2900/5461 with finite loss. A new
checkpoint was saved21:51:10 CEST. Service active, RAM3.56GiB, swap0.
Preserved checkpoint before restart:
`/home/k_faz/project-backups/run-logs/mosad-before-recovery-20261020-evening.pt`,
SHA256 `fc54978b01e04189121de494bc67db6e8cd9f0d6969baf8584720fde1ad840e4`,
verified equal to the live checkpoint before resume. Service peak on failed run
was4.0GiB and swap256MiB. OOM is confirmed; host versus cgroup cause is not.

**Latest workbook corrections:** all eight assessment labels containing
2007–2013 (including boundary-sample and planned MOSAD) are now sorted first.
Added a separate channels40–47 event F0.5 column with18 matching saved scores;
13 unavailable and MOSAD pending. Original system metrics/order otherwise kept.
Short-roll pooled local score0.9071395303690583. The Downloads workbook named
below is the latest user-facing version.

**20 October latest workbook instruction:** User supplied
`/home/k_faz/Downloads/ESA_thesis_results.xlsx` and requested a NEW workbook,
one row per technique, no false-event column, with FPR and event TPR.
Follow-up explicitly requests completed 2007–2013 rows first, then historical
assessments using their actual periods and filled metrics. Delivered
`/home/k_faz/Downloads/ESA_thesis_results_full_and_historical.xlsx`: 32 unique
techniques, five complete exact-full-period results first, historical results
next, MOSAD pending last. Historical representative rows use highest reported
period F0.5 (means across seeds), explicitly labelled descriptive historical
bests, not independent validation or new model selection. FPR recovered by
matching underlying grouped records in the 13 October all-methods workbook.
Unrecorded metadata and pending results are explicit text, never invented zeros.
All 32 exported rows checked against source-derived values; original files kept.
This supersedes the earlier missing-history and one-period-only presentation.

**20 October workbook simplification rejected and rolled back.** The user
rejected the simplified workbook and asked for the previous Excel results.
Restored the canonical workbook byte-for-byte from the pre-edit backup; matching
SHA256 is `79e6d115bfc0e713a37befb64de157c6abff0f1dfa082716710f2dc811f0ea64`.
The previous pooled and annual tables are restored. Do not repeat the
one-pooled-row-only interpretation without clarifying the intended workbook
and how to retain its results. All older workbooks remain intact.

**Superseded edit (rolled back):** The canonical
`outputs/thesis-system-20261016/deliverables/ESA_thesis_results.xlsx` now has one
pooled 2007–2013 system-evaluation row per technique: historical Transformer AE,
short-roll removal, verified normal-reference similarity, and MOSAD (pending,
blank metrics). Removed repeated annual rows from the presentation workbook;
underlying annual result files are preserved. Retain one calibration-selected
configuration per technique, never select variants using test scores. Training
2000–2005, calibration 2006, inputs 40–47 and all-channel event scope are explicit.
Prior workbook preserved outside the repository at
`/home/k_faz/project-backups/workbooks/ESA_thesis_results_before_dedup_20261020.xlsx`.
The one-row-per-technique redesign below was rejected; it is not the current format.
No training source, data, settings or service changes were made for this edit.

**20 October03:14 CEST recovery restarted.** Available RAM5.3GiB,
swap4.7GiB; no comparison worker remained. Same bounded service restarted as
PID3880307 with original --resume and unchanged source/data/settings. Frozen
checks and full training input loading passed; GPU activity confirmed for this
PID (2062MiB VRAM). Initial memory3.57GiB, swap0, peak4.00GiB during
startup. At20October03:44 CEST, progress confirmed through epoch3 batch1300,
checkpoint saved03:41:43; service active, memory3.57GiB, swap0. Original unexplained SIGKILL
cause remains unresolved. Keep existing4.5GiB cap and no blind restart policy.

**20 October00:00 CEST check: bounded recovery interrupted again.**
At19October22:52:45 CEST PID3497287 exited by SIGKILL, Result=signal,
not the earlier explicit oom-kill result. Cause is unconfirmed: this account
cannot read system/kernel/other-user journals. Service peak3.58GiB and swap8MiB
were below configured caps; no Python error appears. No runner or CUDA compute
process remained. Latest checkpoint saved19October22:37:16 at epoch3 batch1100.
At inspection only3.8GiB RAM was available (swap2.6GiB free), leaving almost no
startup headroom above the measured peak. Restart deferred; next heartbeat
reassess headroom (prefer at least5GiB available for the4.5GiB-capped service),
check no worker, and resume the same bounded service if safe. Do not assume the
SIGKILL was caused by the cap; do not raise limits or change frozen source.
If it recurs despite sufficient memory, administrator logs are needed to identify
the killer. Existing checkpoints and pre-recovery backup remain preserved.

**19 October memory-bounded recovery active (supersedes waiting note below).**
User explicitly requested resuming with less RAM and safeguards. Same service
restarted as PID3497287 with unchanged Python source, data, scientific settings,
and original --resume validation. Operational drop-in:
`~/.config/systemd/user/esa-similarity-mosad.service.d/memory.conf` sets
MALLOC_ARENA_MAX=2, MALLOC_TRIM_THRESHOLD_=0, MemoryHigh=4G,
MemoryMax=4608M, MemorySwapMax=256M; Restart=no remains in force.
Frozen checks and full training-data loading passed. GPU activity confirmed for
this PID (2070MiB VRAM); service memory current3.15GiB, peak3.58GiB, swap0
at the initial check. Recovery progress confirmed19October20:34 CEST: epoch3
batch200/5461, new checkpoint saved20:28:31 CEST; memory2.46GiB,
peak3.58GiB, negligible swap. Resume from epoch3 batch1 succeeded. These limits
reduce shared-server pressure but cannot guarantee survival under host OOM or
later calibration peaks. On failure inspect cause; do not blindly restart or
raise the cap. Monitor this bounded recovery instead of waiting for8GiB free.
A separate preserved checkpoint copy is in
`/home/k_faz/project-backups/run-logs/mosad-before-memory-recovery-20261019.pt`,
SHA256 e61952b0e728ac601bcfed61764643d6a42f21bc7ca465a175e2e6e64eb26192
(equal to the original before recovery). No unrelated processes were stopped.

**19 October interruption — awaiting memory headroom.** At15:40 CEST the
service was killed by the host OOM killer (systemd Result=oom-kill), shortly after
starting epoch3. At16:12 CEST no comparison runner remained; only2.1GiB RAM was
available and15GiB swap was almost full. Multiple Vivado workloads occupy most
RAM; do not terminate unrelated work. Do not blindly restart under this pressure.
checkpoint_last.pt saved15:26:36 CEST remains present (2,262,062 bytes); epochs1–2
completed. Next heartbeat reassess memory headroom (prior service peak5.5GiB plus
1.4GiB swap), then resume the same service only with sufficient headroom and no
runner. Wrapper uses --resume and strict source/data/runtime checks. No protocol,
source, data or settings changed. Final assessment remains pending.


**18 October similarity complete; MOSAD actively training.** Independent saved
prediction replay passed for all7,364,160 samples and both scopes, including all
annual/pooled metrics. System similarity F0.5=0.710676 versus baseline0.836070;
TPR30/91 versus48/91. Precision improves but nominal time FPR worsens
(0.000556 versus0.000292). Do not describe this as fewer total false alarms.
See `docs/research/similarity-mosad-full-interim-2026-10-18.md` and verified CSVs.
MOSAD first real-data batch completed; checkpoint_last.pt exists. Do not modify
its precommitted protocol in response to similarity's assessment result. Final
comparison/workbook and artifact Release backup remain pending.

**18 October similarity → MOSAD full-period run launched.** User explicitly
requested both methods and authorized training with all inputs40–47, training
2000–2005, calibration2006 and assessment2007–2013. This supersedes prior
no-training/pilot-only instructions for this specific frozen comparison.
Read `docs/research/similarity-mosad-full-protocol-2026-10-18.md` before touching it.
132 thesis tests and real-config deterministic CUDA smoke passed; frozen train,
calibration, testing, binary labels and annotation hashes verified on this host.
46 training-only input-annotated event shots validated. Historical damaged MLP
array is not a dependency; entire old migration is not claimed verified.

Service `esa-similarity-mosad.service` is enabled, with user lingering enabled.
It runs sequentially, uses an advisory lock, and checkpoints MOSAD every100 batches.
No blind restart: inspect any failure and process list before restarting the same
service. The wrapper adds --resume only when the protocol exists. Strict source,
data and runtime checks apply; do not edit Python source/settings while running.
Output: `results_longrun/similarity_mosad_full_20261018_seed42_v1`.
Log: `/home/k_faz/project-backups/run-logs/similarity-mosad-full-20261018.log`.
Heartbeat: `complete-similarity-and-mosad-comparison`,30min, quiet on unchanged
status. Similarity has a fixed normal reference; MOSAD uses real training positives
plus synthetic augmentation. Disclose this supervision difference. One seed42;
no test tuning, no claim of independent confirmation on this reused benchmark.
MOSAD smoke was about8s/batch; full training may take many days. No time-saving
change to the frozen training budget is authorized by mere elapsed time.

On completion independently verify saved predictions/rules and all corrected
annual/pooled metrics, publish seven years plus pooled system comparison with the
short-no-roll baseline, update system-only thesis workbook and concise report,
commit compact results and back up essential arrays/weights to a checksum-verified
GitHub Release. Then disable the service and pause the heartbeat. Existing
complete.json files are stage receipts, not independent result verification.

**17 October numerical audit completed; historical gate still failed.**
See `docs/research/calibration-numerics-2026-10-17.md`. Two fresh CUDA calls to the
original scoring function are bitwise identical to each other and prior CUDA
replay.128 tolerance failures have scores0.913–0.926, far below threshold72.1623;
CUDA mask exact. CPU32 mask changes are one-ULP threshold ties. All three original/
CUDA/CPU corrected calibration metric dictionaries match. Seven evaluator tests
passed. Inference dependencies and run artifacts/raw data hashes verified;
development.py differs but load_period AST matches checksum-verified historical
source exactly. No tolerances/thresholds changed, no training. Stop blind numerical
setting searches. Continue protocol design using documented stable local execution;
retain cross-platform reproduction limitation and original historical arrays.
Audit code/results pushed; release `calibration-numerics-2026-10-17` contains9
verified archived files. Both GitHub asset SHA256 digests and sizes verified.

**17 October supplementary validation diagnostic completed.**
See `docs/research/channel-error-validation-results-2026-10-17.md`.
All six internal models reproduce validation scores within the unchanged tolerance
and saved masks exactly. All17 completed stage file hashes passed;124 thesis tests passed. Full original
calibration gate remains failed:2002/seed44 calibration cannot meet both score and
mask checks on tested CUDA/CPU paths, including original two-thread setting.
V3 retains stopped.json and has supplementary_status.json, NOT complete.json.
No replay is running. No training, thresholds or historical predictions changed.
Compact results pushed; release `channel-validation-2026-10-17` preserves85
verified files. Both uploaded asset sizes/SHA256 digests match local files; see
the validation tables directory github-backup.json.
All five input-annotated validation IDs and three seeds are included: id86 shows
channel-specific upper-error evidence obscured by channel47 maximum; id25 has
ordinary/lower errors on both annotated inputs across all seeds. Aggregation alone
is not an established solution.2001 peaks can rank highly but frozen rules miss.
Retrospective tail ranks are not detection metrics. Do not repeat normalization
or launch a training sweep. Next design must address both channel comparability
and calibration transfer, distinguishing prior failed approaches. Numerical
calibration reproduction remains unresolved; do not relabel original gate passed.

**16 October development consolidation completed.** User approved investigating
training/development evidence. See `docs/research/development-evidence-review-2026-10-16.md`.
Recomputed 2000–2005 coverage: 95 unique IDs,46 input-annotated,49 outside inputs.
Verified prior joint internal score audit: 2002 nominal scores exceed event peaks
for49.07–67.50% of nominal timestamps across event/seed pairs. Reviewed failed
expansion, masking, residual normalization/forecasting, isolation and adaptive
scoring; checked primary literature with explicit metric/forecasting limitations.
No new model is justified yet. One next diagnostic is predeclared: replay six
frozen internal joint models with per-channel errors, first reproduce scalar
scores/masks, then assess per-channel error separation and max-score attribution.
It is **not executed yet** and changes no predictions or thresholds. No new
training, benchmark scoring, input choice or claims of independent validation.

**16 October missed-event audit completed after user approval.** See
`docs/research/full-period-missed-events-2026-10-16.md` and associated tables.
Pooled outcomes: both detect46, baseline-only8, candidate-only2, both miss35.
Candidate misses43: 20 input-annotated, 23 outside-input-only. All eight lost
detections have input annotations; the loss of recall cannot be explained solely
by restricted channel coverage. 2010 misses split5inside/4outside; 2012 split5/1.
20 of23 outside-input misses involve14/21/29, but prior expansion already increased
false alarms: this test-derived description must not select new inputs. Dependency
hashes and all annual/pooled event-count/metric reconciliations passed; seven
evaluator tests passed. No training/tuning occurred. Next is consolidation of
training/development coverage and score evidence with prior failed experiments,
then a literature-supported single controlled hypothesis, not another blind run.

**16 October completion:** full-period short-no-roll training and assessment are
finished. Independent artifact/source/frozen-data hash checks, prediction replay
and all 16 annual/pooled corrected metric rows passed. System pooled F0.5 is
**0.8360701968**, input-scope **0.9071395304**. System baseline was 0.8057217337;
precision improved but event recall fell (54/91 to 48/91). This is below the
system target and is a single-seed reused-benchmark comparison, not a pure roll-only
ablation or independent validation. No new experiment is authorized by completion.
See `docs/research/full-short-no-roll-results-2026-10-16.md` and its verified tables.
The latest system-only workbook includes all completed candidate annual and
pooled results with event TPR and duration-based FPR. All 16 workbook formulas
match replay; formula-error scan and visual checks passed. Seven evaluator tests
passed. Service completed successfully without restarts and has been disabled.
Older active/pending paragraphs below are historical. Compact results/workbook
were pushed in commit 0825e90. Release `short-no-roll-full-2026-10-16` backs up
22 verified files including predictions, scores and both checkpoints; both GitHub
asset sizes/SHA256 digests match local files. Evidence is in the report's tables
directory (`github-backup.json`). The completed heartbeat was deleted to stop
further checks. No training job or follow-up remains active for this run.

16 October workbook update: the user requires identical train/calibration/test
periods and run counts, system-only evaluation, and FPR/TPR instead of false-event
count. Revised compact workbook:
`outputs/thesis-system-20261016/deliverables/ESA_thesis_results.xlsx`.
See `docs/research/thesis-workbook-policy-2026-10-16.md`. Baseline full-period
metrics were replayed and verified; short-no-roll cells remain pending. When the
active run finishes, update this workbook with verified system-only annual and
pooled results. TPR is event recall; FPR is explicitly nominal false-alarm time
fraction. Do not mix old development protocols or select one favorable seed.

16 October recovery: the original process was absent; its log ended in epoch3,
batch3300 on14October15:31 Europe/Rome. Host boot time was14October15:55;
no traceback identifies the original stop cause. Epoch2 checkpoint is readable
and finite; every protocol-recorded source hash matches. No final scores exist.
Resuming under persistent user service `esa-short-no-roll.service`, enabled with
linger already active. Use `systemctl --user status esa-short-no-roll.service`.
Current log: `/home/k_faz/project-backups/run-logs/full-short-no-roll-resume-20261016.log`.
The service retries failures up to three starts/hour and resumes saved epochs;
do not start a second runner. Source/settings/data are unchanged. After verified
completion disable the service and pause the existing heartbeat. The original
PID/session notes below are historical and must not be treated as live state.

**Latest request: run short-no-roll on the full 2007–2013 period and report every
year plus pooled F0.5.** This explicit request supersedes the earlier no-extension
stopping decision for this candidate. Plan: `docs/research/full-short-no-roll-plan-2026-10-14.md`.
Use frozen training2000–2005/calibration2006/testing2007–2013; channels40–47,
40 epochs, all nominal windows, seed42. Runner `esa_thesis.full_short_no_roll`.
Output `results_longrun/short_no_roll_full_20261014_seed42_v1/`. The run is active in tool session8451 (initial Python PID1529365); log:
`/home/k_faz/project-backups/run-logs/full-short-no-roll-20261014.log`.
Heartbeat `complete-full-period-short-roll-assessment` checks every30minutes and
must deliver verified results then pause itself. Verify live process state rather
than trusting this PID. Do not claim results until complete.json and replay checks pass.
122 tests passed in12.840s before launch. Source/data/settings must stay unchanged
for strict resume; documentation updates are safe.
No yearly tuning or independent-test claim. The original baseline is unchanged.

**Current instruction, 14 October: autonomous research is authorized again,
including justified training and commands. This supersedes the older stop below.**
Follow `docs/research/research-resumption-2026-10-14.md`. The first authorized
comparison executes the predeclared nominal-quantile diagnostic using verified
saved scores only. The damaged historical MLP array is excluded. No final-test
selection or indiscriminate training sweep is authorized by the research protocol.
The dated stopped-state paragraphs below record history, not current instructions.

Completed the fixed q=0.999, epoch-five threshold comparison on all six original
joint runs. All original dependencies matched migration hashes; control masks and
metrics reproduced. All candidate rules were sealed before validation arrays were
loaded. Mean system F0.5: 2001 control 0 → candidate 0.384075; 2002 control 0 →
candidate 0.006370. Candidate 2002 occupancy is 87.20%; false-alarm duration rises.
**Advancement failed in both folds.** Do not tune neighboring quantiles or promote
this candidate. See `docs/research/nominal-quantile-results-2026-10-14.md` and
`docs/research/representation-decision-2026-10-14.md`. Frozen reference reverified;
120 tests passed in 11.846s. No new neural training or 2007–2013 scoring occurred.


GitHub synchronization completed **14 October 2026**. Source update `751fc1a`
is pushed; the full current project-file snapshot is published at
`project-snapshot-2026-10-13` (50 Release assets, 174.06 GiB before compression).
All 10,526 archived entries passed verification and all uploaded asset SHA256
digests matched local files. See `docs/research/github-sync-2026-10-13.md` and
`docs/research/github-snapshot-verification-2026-10-14.json` for evidence and
exclusions. The full thesis suite passed 117 tests before publication.
The original historical array mismatch remains unresolved. Research stays stopped.

**Current instruction: research is stopped at the user's explicit request on
13 October. Do not execute the nominal-quantile plan or launch training.**
The subsequent authorized task is repository synchronization and comprehensive
GitHub backup only. GitHub is authenticated as `kamyarfaz`; credentials are outside
this project. Before publication, local/remote main both pointed to `0fd67c8`.

Completed before the stop: calibration diversity and label-influence diagnostics,
a targeted literature review, and a training-only reconstruction of preprocessing
on channels 40–47. The influence audit reproduced 2,898 candidate rows and six
original decisions. Use `calibration-influence-2026-10-13-v2` (it adds the eligible
candidate flag); v1 remains preserved. Removing the detected 2001 temporal
component leaves all eligible candidates at zero; withholding id_84 in 2002
raises the selected threshold 2.60–3.04 times. These are diagnostics, not new
validation results or adopted rules. The preprocessing reconstruction matches
3,156,480 timestamps per input; annotation restoration changes five channel-values
at two timestamps. Neither analysis trains a model. The nominal-quantile plan is
**written but unexecuted**. The repository check passed **114 tests** in 11.525s.

The historical MLP checksum failure is still unresolved. A search of all 59
same-size local files found no matching original. See
`docs/research/migration-exception-2026-10-13.md` for the bounded diagnostic
continuation used before the stop. It never certified a lossless transfer.

GitHub synchronization records and restore instructions are in
`docs/research/github-sync-2026-10-13.md`. Keep raw datasets/weights out of source
Git; the full project-file snapshot belongs in split Release assets. Backing up
the current mismatching array preserves evidence, not a verified original.

The dated paragraphs below preserve earlier setup history; the latest resumption
instruction and protocol take precedence over earlier stop/continuation notes.

13 October continued setup: packaged and verified the existing reference as
`data/frozen/mission1-reference-v1/{train.csv,calibration.csv,testing.csv}` plus
`labels.csv.gz` and `manifest.json`. Files are read-only, source row bytes are
preserved, and disjoint 30-second coverage and checksums passed. Manifest SHA256:
`a49a8c42c23a5d72d91a48596f485d762a805cc6101ad037c01a36c8037bf099`.
Use `python scripts/research/freeze_reference_data.py verify` before consuming.
The historical runners still use the original sources; no silent redirect or new
experiment occurred. See `docs/research/frozen-reference-2026-10-13.md`.
Five new integrity tests and the existing suite pass: **103 tests**, 13.330 seconds.

Transfer recovery: restored `.gitattributes` and the October 12 backup report from
HEAD; restored eight missing symlinks and preserved external W&B log targets;
recovered the damaged archived `fine_ch08_35_42/test_score_per_channel.npy` from an
exact-hash local duplicate, retaining its damaged original. Corrected executable
bits on 2,216 tracked files only where contents exactly match the Git index; left
content-modified files untouched. Recovery evidence is under
`migration/recovery-20261013/`. The historical
`pa_center_ch08_40_47/mlp_test_score_per_channel.npy` still mismatches the transfer
hash. The September 17 release archive was downloaded, checksum-verified and
inspected; this array is listed under EXCLUDED files, not included. Preserve the
current array; do not regenerate it and claim exact recovery. The user supplied
the current checkout path and GitHub URL, not an independent original copy.
No research training may start while transfer recovery remains unresolved.
The post-recovery full destination scan completed: **12,517 entries; 16 failures**.
Only one failure is a research array; fifteen are newer documentation/Git state
and snapshot differences relative to the old manifest. All other recorded artifacts
match. See `docs/research/transfer-recovery-2026-10-13.md`; do not describe the full
transfer as zero-failure or erase the original manifest. The original Git-status
snapshot also cannot be reconstructed exactly from the available copy.

13 October destination environment verified after user installation:
`/home/k_faz/miniforge3/envs/esa-thesis` has Python 3.9.23 and
PyTorch 2.6.0+cu124. `pip check` and `python -m esa_thesis doctor` pass.
CUDA detects NVIDIA GeForce RTX 2080 Ti; a finite 256x256 matrix multiplication
and CUDA synchronization passed. All 98 thesis tests passed in 12.106 seconds
on this destination. These are environment/regression checks, not a research
training run. The subsequent recovery and updated test count are recorded above.
The full transfer remains unresolved; do not launch experiments yet.

13 October research brief: the user requires all-channel anomaly/event evaluation,
three permanently fixed train/calibration/testing files for final comparisons,
a separate binary ground-truth label file, and a concise thesis results workbook.
They clarified: evaluate all-channel events; decide model inputs after research.
Do not equate this with authorization to train a 76-channel model. Before proposing
another experiment, review existing diagnostics and primary literature, explain
the mechanism and hypothesis, and define one controlled comparison and stopping
criteria. Avoid speculative model sweeps. The user wants concise explanations.

Audit and deliverables: `outputs/thesis-summary-20261013/deliverables/` contains
`ESA_thesis_results.xlsx` (one sheet, 65 method-period rows),
`ESA_Mission1_binary_labels.csv.gz` (14,728,320 timestamps across the existing
reference splits, all 76 channel labels, plus a communication-gap ignore flag),
and `research_protocol_review.md`. Both original prepared CSV SHA256 checks pass.
The reference dates are train 2000–2005, calibration 2006, testing 2007–2013;
these originally came from two source files and are now packaged separately as
recorded above. No new date split or pipeline replacement was implemented. The
independent-test status remains to be settled; inspected
test periods cannot become unseen by freezing files. Raw event IDs/intervals remain
authoritative for corrected event scoring; do not substitute pointwise F0.5.

13 October destination update: the user confirmed this is a fresh server and
prioritized preparing research reporting before environment setup.
Created `outputs/research-update-20261013/deliverables/ESA_research_update_2026-10-13.pptx`:
12 slides in the September 30 deck's layout, covering completed work since that
meeting, with source references and speaking notes. No research decisions changed.
Also updated the comprehensive workbook as a new copy in the same deliverables
folder: `ESA_all_methods_results_2026-10-13.xlsx` (25 sheets, 14 charts, 256
assessment records, including 112 added October records). It preserves the
September workbook, adds all three no-roll seeds and internal primary/secondary
policies, and includes calibration/event/channel-intervention diagnostics.
Formula checks, source metric arithmetic, original-cell preservation and rendered
updated ranges were checked. No model inference or training was performed.
The neutral `research-update-20261013` deliverables supersede the earlier October
reporting copies, which remain preserved. User preference: use academic language
and omit supervisor names and personal meeting references from Excel and reports.
Both revised packages were checked for the prohibited name, including notes and
metadata; workbook numeric values and formulas are unchanged.

Destination transfer verification completed: **12,517 entries checked, 25 failures**.
The log is `migration/destination-verification-2026-10-13.log`. Failures include
Git metadata, handoff/readme files, links, and two older score arrays. Causes have
not been resolved; do not describe the destination as verified or delete source
copies. Tracked `.gitattributes` and the October 12 backup report are also missing
relative to HEAD, and many executable bits differ. System Python remains
3.12.3; the separate Python 3.9 research environment and destination thesis tests
now pass as recorded above. Presentation tooling used the bundled artifact runtime.
Retain the original transfer manifest as evidence; reconcile failures before
creating a new manifest for any subsequent transfer, including today's new files.

The user is moving the **whole project directory to another PC**. Migration is
active again. Complete destination verification and environment checks before
research. Do not launch training during transfer. No research jobs were found
running at the source check on 12 October. Old PIDs, pending runs and commands in
historical notes are obsolete unless independently verified.

All experiments through 8 October below are finished; none of the new internal
candidates passed its advancement rule. Keep the established full-period baseline.
Next research work is **consolidating findings and defining a more representative
calibration/generalization protocol**, not another blind model/threshold sweep.
No replacement protocol has yet been implemented or approved as a specific run.
The user prefers autonomous execution within the agreed task, without repeated
permission questions. Preserve results and disclose experimental limitations.

Start on the new PC:

1. Read [MIGRATION.md](MIGRATION.md); verify the copied manifest **before imports,
   tests or Git commands that could modify the transferred tree**.
2. Recreate the Python environment; run doctor and the thesis tests.
3. Read the linked reports for the next task. Do not rerun completed experiments
   merely because an old history entry says “next” or “active”.

## 2. Thesis and scope

Kamyar Faz's thesis is satellite anomaly detection using ESA Mission 1 telemetry.
Supervisor Federico Butchellato supplied initial main/models/utils Python files
and requested a Transformer approach aiming for F0.5 >=0.85. Tentative title:
“Artificial Intelligence for Anomaly Detection in Satellite Telemetry under Space
Operational Constraints”. No final title/deadline; a complete thesis manuscript
has not been written. Do not claim the target is robustly achieved.

Current inputs are **eight channels 40–47**, not all 76 telemetry channels.
“System scope” means evaluating those predictions against all-system event IDs;
it does not mean the model consumed all 76 channels. Local scope uses events
annotated on the eight inputs. Use identical frozen predictions for both scopes.

Supervisor requirements: investigate informative channels, correlations and
preprocessing; compare methods on the **same real assessment data**, never
synthetic test data; present organized results by period/channel set and plots.
Artificial anomalies are training augmentation only. The current research focus
remains 40–47; do not silently switch channel sets or remove channel47.

## 3. Established full-period reference

- Train nominal windows from **2000–2005**; calibrate on **2006**.
- One frozen model/rule evaluated continuously over **all months 2007–2013**.
- Local: **F0.5 0.862907**, TP53/FP7/FN14, 67 annotated event IDs.
- System: **F0.5 0.805722**, TP54/FP7/FN37, 91 annotated event IDs.
- Main run: `results_longrun/long_baseline/ch40_47_train2000_2005_seed42`.
- See [dual-scope evaluation](docs/research/dual-scope-evaluation.md) and the
  baseline/reproduction reports indexed in [docs/README.md](docs/README.md).

This is a retrospective Transformer autoencoder, not a causal forecaster.
Later-period covariates and test results have been inspected: not untouched data.
Local >0.85 is neither a system score >0.85 nor evidence of robust generalization.
Do not directly compare full-period scores with small internal-pilot scores.

## 4. Model, metric and split rules

`esa_thesis/models.py:MultivariateAE` already uses temporal AND channel Transformer
encoders/decoders (1,333,392 parameters for eight inputs). Do not suggest “adding
Transformers” as though the baseline were a dense AE. The residual forecaster is
a different model; its earlier negative result is documented separately.

Correct evaluator: `esa_thesis/evaluation/esa.py`, version `esa-ew-id-duration-v1`.
It scores Anomaly + Rare Event annotation IDs and penalizes nominal alarm duration:

P = TP/(TP+FP) * (1 - nominal_false_alarm_seconds/nominal_seconds)
R = TP/(TP+FN)
F0.5 = 1.25*P*R/(0.25*P+R)

Use evaluator zero-denominator/neutral-annotation conventions and parity tests;
do not substitute pointwise F0.5 or the legacy evaluator. The selection cap is
**1% of all alarm samples**, not 1% of nominal false-alarm time.

Historical development folds2003/04/05: expanding training before each year,
Jan–Mar calibration, Apr–Dec assessment. Later internal pilots use:

| Fold | Training | Calibration | Real validation |
|---|---|---|---|
| 2001 | 2000 | Jan–Jun2001 | Jul–Dec2001 |
| 2002 | 2000–2001 | Jan–Jun2002 | Jul–Dec2002 |

Internal neural runs: five epochs, 25k nominal windows, seeds42/43/44, batch64,
short_no_roll augmentation, training-only median/IQR scaler, clipping ±10,
window256/stride16, score stride32, window-mean overlap aggregation then channel
maximum. Calibrate checkpoints/thresholds only on calibration, then freeze.
The primary policy is calibration_selected; fixed_epoch5 is secondary and cannot
be substituted after seeing its validation results. These periods have repeatedly
informed development and must not be presented as independent confirmation.

## 5. Recent completed experiments (do not restart)

Paths below are under `results_longrun/development/`. Scores are mean system F0.5
across seeds unless deterministic; all use the internal periods above.

| Experiment / folder | 2001 | 2002 | Decision |
|---|---:|---:|---|
| Joint control `inner_selection_cuda_v2` | 0 | 0 | Retain as matched internal control |
| Loss weight0.1 `inner_lambda01_cuda_v1` | 0 | 0.002600 | Failed; 2002 meanFP53→159 |
| Isolated channels `inner_isolation_cuda_v1` | 0 | 0 | Failed; 2002 meanFP53→128.67 |
| Score-change filter `score_change_v1` | 0 | 0.238095 | Failed; only seed44 detects id25 |
| Direct telemetry `telemetry_reference_v1` | 0 | 0 | Deterministic; silent both years |

Score-change mean hides seeds42/43=0 and seed44=0.714286 system/0.833333 local.
False alarms disappeared but recall was not reliable. No promotion/outer run.
The direct reference is one run per year, not three artificial replicates.

Reports, full CSVs and plots:
- [Internal selection](docs/research/inner-selection-results-2026-10-02.md).
- [Loss weight](docs/research/loss-weight-results-2026-10-03.md).
- [Channel isolation](docs/research/channel-isolation-results-2026-10-08.md).
- [Score change](docs/research/score-change-results-2026-10-08.md).
- [Direct telemetry](docs/research/telemetry-reference-results-2026-10-08.md).

**Checkpoint warning:** isolated-model tensors have the same keys as the joint AE,
but replay requires `ChannelIsolatedAE` from `esa_thesis/channel_isolation.py`.
Loading into MultivariateAE silently changes attention semantics. Isolation keeps
nominal parameter count but changes effective capacity; shared training losses
still couple updates. Do not describe it as eight separately fitted models.

## 6. What the diagnostics established

- Channel47's normal level changes and reconstruction errors rise in2002.
  Frozen-model interventions show its input level can affect other outputs.
  Correcting it reduces other-channel errors in most, not all, models. This is
  model sensitivity, not spacecraft causality; centering may hide real anomalies.
- Calibration detects a narrow subset of events. Later event scores can be much
  smaller; in2002 many nominal scores exceed missed-event peaks.
- Missed local validation events are not clipped on their annotated channels.
  Use **v2** event-input statistics; v1 used ID-union intervals and is historical.
- Removing the alarm cap improves6/10 calibration cases in2001 and0/10 in2002.
  19/20 grids offer no higher-recall row under the cap. No universal cap fix.
- No evidence supports another full-period training run for these candidates.

Read [channel intervention](docs/research/channel-level-coupling-2026-10-03.md),
[missed-event audit](docs/research/inner-event-audit-2026-10-08.md), and
[calibration audit](docs/research/calibration-selection-audit-2026-10-08.md).
Earlier broader inputs, reconstruction-only, specialist/OR alarms, masking,
observation-age and augmentation experiments are documented in the docs index.
Their results remain preserved; do not repeat them without reviewing those reports.

## 7. Code and artifact map

- `esa_thesis/`: active research; root `setup.py` is legacy TimeEval, not this package.
- `esa_thesis/inner_validation.py`: source-hashed shared pilot runner.
- `isolation_comparison.py`, `loss_weight_comparison.py`: wrappers using that runner.
- `score_change.py`, `telemetry_reference.py`: fixed statistical transforms.
- `scripts/research/`: diagnostics, report verification and transfer tool.
- `tests/thesis/`: regression/evaluator tests; 98 tests passed again on12 October2026.
- `docs/research/`: protocols/results; `tables/`: compact CSVs/plots/provenance.
- `results_longrun/`: checkpoints, optimizer states, arrays, full run manifests.
- `data/`: raw/prepared datasets and annotations; copy it, do not regenerate casually.
- `outputs/`, `output/`: presentations/workbooks and other artifacts; preserve both.
- `migration/`: local transfer manifest and prior environment snapshots; copy it.

Older protocols contain absolute source paths and immutable source hashes. Some
scripts have hard-coded output names and deliberately refuse existing directories.
Read code/CLI before executing. Relocation can break strict resume/hash checks;
never rewrite original provenance to force a resume. Create a new run or explicit
relocation adapter only when needed and document it. Replaying saved results does
not require resuming training. Historical finish scripts/PIDs are not migration jobs.

## 8. Transfer and reproducibility cautions

See [MIGRATION.md](MIGRATION.md). Copy **everything**, including `.git`, untracked
files and ignored datasets/results. This checkout has substantial local work;
GitHub-only restore is not guaranteed to contain it. Release backups are partial snapshots; the12October additive archive covers
the recent campaigns explicitly listed in its manifest. The12October source and additive artifact backup are documented in
[GitHub backup](docs/research/github-backup-2026-10-12.md); datasets and older
artifacts outside the archive still require the full folder/earlier releases. Keep the original until destination verification succeeds.

The source is Linux/Python3.9, PyTorch2.6.0+cu124, environment named timeeval.
Create a fresh destination environment (e.g. esa-thesis); names need not match.
Follow README/requirements-test.txt; don't copy Conda or run `pip install -e .`.
Source doctor on12October passed and detected CUDA/RTX3060, with an NVML warning;
this was not a GPU training test. Logs are under migration/.
Two artifact-build node_modules links point to the old Codex runtime; reinstall
artifact dependencies for rebuilding (finished artifacts remain portable).
See MIGRATION.md for all five external symlinks.
CPU tests are supported; check CUDA on the new PC before GPU work. macOS/native
Windows setup has not been validated here; Linux/WSL is the documented route.

Prior CSV decoding/label failures led to strict integrity checks. The prepared
training golden SHA256 is
`fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad`.
Do not bypass corruption with permissive decoding or silently coerce invalid labels.

## 9. Full history and new-chat prompt

The previous1200-line handoff is preserved verbatim in
[historical handoff](docs/history/AI_HANDOFF-through-2026-10-08.md), including
supervisor references, early experiment results, restore order and detailed
chronology. It contains superseded plans/statuses. This document and the latest
verified result reports take precedence; history is evidence, not pending work.

Paste into the new chat:

> This is my ESA satellite anomaly-detection thesis project. Read AGENTS.md,
> AI_HANDOFF.md and MIGRATION.md first. We have moved computers: verify the copied
> manifest before changing files, then check the environment. Preserve all results.
> Read the latest linked reports and continue from the current research decision,
> not an old unfinished plan. Explain findings simply; do not ask me to repeat
> project history or automatically start another training sweep.
