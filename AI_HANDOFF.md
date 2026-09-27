# AI handoff — ESA satellite anomaly-detection thesis

Last updated: 27 September 2026. This is a portable summary of the working
conversation, not a transcript. Verify facts against saved artifacts if anything
has changed. Paths below are relative to this repository so the folder can move.

## Start here: current user intent

GitHub publication27Sep prepared: source/docs/tests plus separate213file essential
artifact release research-update-2026-09-27.73testsPASS. Backup coverage/exclusions
in docs/research/github-backup-2026-09-27.md; manifest checked in. This is not a
full dataset/diagnosticcache/optimizerbackup; preserve localfolder. User requested
push before furtherdevelopment; augmentation still primitive only.

SHORT COORDINATED AUGMENTATION PRIMITIVE IMPLEMENTED27Sep after useraskedresume
previouswork. esa_thesis/short_augmentation.py make_training_anomaly legacydefault
exactoutput/RNGparity; optional50/50legacy+short1–10samples,1–Cchannels,shared
segment/type/offsetsign+amp/noisewaveform/rollshift. Shortscale fullwindowstd
population+eps, length1noroll; no-opmetadata, immutableinputs.73testsPASS.
NOT wired toCLI/NOT trained/no newF0.5. Allproductioncalls unchanged. Report
 docs/research/short-coordinated-augmentation.md. Next explicitopt-in comparison
runner/protocol matchedbudget and RNGhandling before pilot. Do not enable silently
or resumeoldsourcehashedruns undernewsource. No pushrequested.


27SEP: USERBASELINE RERUN COMPLETE at ch40_47_repeat_seed42_v2. Exacthashmatches
oldrun pooled/yearly/history/frozenrule/assessmentscore/predictions. System.8057217,
54TP7FP37FN,epoch10cal.882148. No improvement, no newtraining launched byassistant.
Pre2006localannotationinventory:46IDs79episodes;32short(<6min)across13IDs;
56wide(>4simultaneouschannels)across40IDs;16short+wide across8IDs inclid98.
Supports recurring syntheticcoveragegap, not proven physicalmechanism/efficacy.
Report docs/research/local-event-shapes-2026-09-27.md proposes OPTIONAL50/50
legacy+shortcoordinatedaugmentation,1–10samples/including5–8channels; notyet
implemented/tested/launched. Keep defaultunchanged; next implement/testbounded
candidate then matcheddevelopmentpilot beforeanotherfullrun. User authorized
pursuing nextwork; no need to rerun baseline. All previous failedvariantsretained.


ID98 LOCAL REVIEW DONE. SystemMar25–28 extent is NOT localduration. Exact41–46
union9min: Mar25 03:59:55.134 5min(ch42/45),04:36:55.134 30sec(all6),04:56:55.134
3.5min(all6,46extends30sec).10/1/7gridpoints. Currentpseudo duration6–42min,
1–4channels: misses short/all6support in syntheticdesign. Not causalproof.
Contextpair correlations41/44,.938;42/45,.928;43/46,.915;prior-day .938/.920/.866;
no sustainedpairbreak established. Localpeakratios unchanged(.815673/.00005544).
Report docs/research/id98-local-review-2026-09-26.md; plots/tables. No training/
assessment/tuning. Next proposed inventory durations/simultaneous channelcounts
across training/developmentannotations before one predeclared augmentationtest;
do not fitid98alone or rerun rejectedpertimestepscore casually.


FULL2005CAL BOTHMODELS COMPARED: eachownfrozenrules reproducemetrics within1e-6.
Pseudo system.832880/local.908552;recononlysystem.832732/local.908375;both2TP0FP2FN
(system), local2TP0FP1FN. Bothdetectid10/id17,missid98local/id99outside.
id98 peak/ownthreshold pseudo.815673 vs recon.000055: reconcopies this anomaly
verywell, so smallnominalMSE != improveddetector. No modelpromotion/training/tuning
or assessmentaccess. Cachedverifiedinputs, hashesstable, no newfullCSVhash.
Report docs/research/calibration-model-comparison-2005-2026-09-26.md; fulloutputs
results_longrun/diagnostics/calibration_model_comparison_2005_v1. Nextquestion
inspectid98 annotated41–46 relationships vsnominal and augmentationcoverage before
newtraining; retainbaseline, do not thresholdchase (candidatealreadyfailed).


EXISTING RECONSTRUCTION-ONLY COMPARISON DONE on same6nominal2005calwindows.
Bothsavedepoch5,channels/scalersEXACTmatch,40epochbudget/seed42/batch64/splitmatch;
models.py/data.pyhashmatch, development.pyhashdiff, sample/orderidentityunverified.
Pseudo maxMSE highwindows12.70/10.44/11.47 vs recon-only~2–5e-6; normalrefs
pseudo~.08 vs recon~2–3e-6. Plots: recon followsinput; pseudolargenegativeoscillations.
Exploratory selected-on-pseudo-error examples, NOT causalproof or modelpromotion.
Decoderinputaccess means nearidentity may reconstruct anomalies too. Next bounded
step comparefixedannotatedcalibrationwindows plusnominalrefs for separability,
no newtraining/assessment. Report docs/research/nominal-reconstruction-comparison-2026-09-26.md;
plot/tables correspondingfolder. Checkpoints/protocols/sourcehashes recorded.


SIX NOMINAL2005CAL WINDOWS INSPECTED: highestmaxchannelMSE +medianwindow per
Jan/Feb/Mar, all256samplesgloballynominal. FreshCPU reconstructedMSE reproduces
cached rtol1e-5/atol1e-5; filesstable; no threshold/eventtolerance changes.
Higherror Jan3ch41 MSE12.70/Feb14ch43 10.44/Mar12ch44 11.47: observedroughly-2..1.3,
recon repeatedlynear-6 with offsets/oscillations. OrdinarymaxMSE~.08. Modeloutput
distortion on selectednominalexamples; no proven cause or95FPexplanation.
Report docs/research/nominal-window-inspection-2026-09-26.md;3plots+raw/scaled
stats under tables/nominal-windows-2005-2026-09-26. Next suggested boundedcheck:
compatible existingreconstruction-onlycheckpoint on samewindows, auditprovenance
and pseudoobjective; not causalinference unlessmatched. No newtraining/assessment.


TRAIN/CAL SCORE STABILITY AUDIT DONE: savedverified diagnostics, no assessment
access/newinference/training/rulechanges. Like-for-like whollynominal windowMSE;
training2000samplewindows/year vs allclean calibration stride32 windows.
2005 median train.07935/cal.0811–.0856, q99 train.7250/Jan4.3413/Feb5.9204/Mar1.4444;
q999train9.5954/cal10.19–12.13.2004mediansstable, q99calhigher. Suggests tail
frequency variation, not broad median shift; sampling/in-sample/dependence caveats.
DoesNOT explain95laterFP directly. Allfrozencalmetrics reproduced/hashesstable.
Report docs/research/training-calibration-stability-2026-09-26.md; plot/tables there.
Next proposed boundedstep inspect fixedsmallset higherror nominal windows plus
normalreferences in training/calibration (raw/scaled/perchannel), before deciding
training change. Keepbaseline. No claims physicalregime cause established.


GRID/CAP DEVELOPMENT ASSESSMENT DONE—REJECT CANDIDATE. Allrules frozen before
reading Apr–Dec2003/04/05 caches; original predictions+metrics reproduced;
sourcehashes before/after match. System original→candidate:
2003 .833170→.833170(5TP1FP1FN);2004 .428131→.428248(6TP9FP4FN);
2005 .869055→.082397(4TP0FP3FN→7TP95FP0FN). Local2004 .447280→.447408;
otherlocal==system. No promotion, no originalrulechange, no training/2007–2013
access. Development previouslyinspected, not untouchedtest. Keepbaseline.
Report docs/research/grid-cap-development-2026-09-26.md. Fulloutput
results_longrun/development/grid_cap_policy_v1; script assess_grid_cap_policy.py.
Next direction training/calibration score stability, not another threshold sweep
on assessment or per-yearwinner selection. Candidate calibrationgain failed.


GPU2003 REPRODUCTION PASS EXACT. Identicalcachedinputs/checkpointhashes;32GPU
scores equal threshold6.974297523498535 vs CPU nextfloat32. Explains16m alarm
extension. Device numericalboundary established for thisrun; no thresholdchange
or claims unrelatedintegrityfault fixed. Completed2x2 in grid_cap_comparison_v2,
all3folds verified.2003allpolicies.714241;2004original.383907/dense.384048;
2005original.832880/dense+total.832955/dense+nominal.936708(3TP0FP1FN).
Calibrationonly; no newtraining/assessment access. Report
 docs/research/grid-cap-comparison-complete-2026-09-26.md.
Next proposed boundedstep freezecandidatecalibrationrules then evaluate cached
Apr–Dec2003–2005 DEVELOPMENT scores, bothscopes/allfolds; historicallyinspected,
not untouchedtest. No2007–2013policyselection. FailedCPU and partialv1 preserved.


2003 BOUNDARY ISOLATED: originalpaired scorer and diagnosticCPU scores EXACTLY
match (maxdiff0) on savedcalinputs/checkpoint; inputhashes stable.32samples
Feb27 02:24–02:39:30 score6.974298000335693 = nextfloat32 above frozen6.9742975235.
Strict > extends alarm16min. Diagnosticthreshold+1e-6 restores historicalmetric
exactly but NOT adopted. CPU/GPU rounding plausible, unproven without GPUrepeat.
CUDA unavailablehere; next userGPUdiagnose command in
 docs/research/calibration-boundary-2003-2026-09-26.md, newoutput
baseline_2003_epoch25_gpu_v1. No rulechange/tolerance relaxation/2003acceptance.
Script reproduce_2003_boundary.py; evidence outputs/calibration-boundary-2003-20260926.


2x2 GRID/CAP STUDY PARTIAL: saved plan, fixedcheckpoints, original vs dense90–100
percentiles step0.1, total1% vs nominalfalsealarm1%.2004 sameeventcounts1TP2FP;
F0.5 original.383907/dense.384048 bothcaps.2005 original.832880 bothcaps;
dense+total.832955 (2TP0FP), dense+nominal.936708 (3TP0FP1FN), totalrate1.1975%,
nominalrate.1126%. Calibration-only, no rulechanges or assessment access.
2003 CPU diagnostic FAILED strictreproduction .7141835329 vs saved.7142410613;
same1TP0FP2FN but nominalalarm1705.224s vs745.224s (+960s). Causeunresolved;
DO NOT accept partial2003 or relax tolerance. Preserve baseline_2003_epoch25_v1.
Next resolve CPU/GPU/checkpoint discrepancy before claiming3foldcomparison.
Report docs/research/grid-cap-comparison-2026-09-26.md; script
scripts/research/calibration_grid_cap_comparison.py; outputs grid_cap_comparison_v1
statuspartial.68tests passed. No newtraining.


ALARM-CAP AUDIT COMPLETE: original selected-checkpoint grids2003/04/05;
original rules reproduced. Swapping total1% cap for nominalfalsealarm1% selects
IDENTICAL rules/F0.5 in all3folds. 2005 grid jumps6.353486 to15.792491, skipping
useful diagnostic12.633993, which ALSO violates totalcap. Cannot attribute gain
to capalone. No assessment access/rulechanges/training. Report
 docs/research/alarm-cap-audit-2026-09-26.md. Next proposed bounded2x2 calibration
study: original/dense grid x total/nominalcap at fixedcheckpoints, allfolds,
no assessment-based selection. Alternative1%nominalcap is exploratory, not an
agreed operational requirement.


2005 CALIBRATION DIAGNOSTIC COMPLETE: CPU epoch5 reproduces F0.5 exactly0.832880;
golden trainCSV SHA passes before/after. Detectedid10/id17; missedlocalid98
(Mar25–28, peak12.8815 on43 vs threshold15.7925); missedoutsideid99.
Diagnostic threshold*0.8 gets3TP0FP1FN F0.5=.936802 but totalpredictedrate1.1728%
exceeds existing1% cap (frozen rate1%). NOT adopted/NOT assessment improvement.
Next focused hypothesis review total-alarm1% cap versus false-alarm constraint
on development folds, no newtraining or assessment tuning. Report
 docs/research/baseline-2005-diagnostic-2026-09-26.md. All rawoutputs preserved in
 results_longrun/diagnostics/baseline_2005_epoch5_v1; compacttables in docs.


2006 CALIBRATION STABILITY AUDIT COMPLETE: frozen epoch10 scores/predictions
reproduce within1e-9; saved artifacts stable before/after hashes, no assessment
access. All6 locally annotated events detected;4 system misses only outside
inputs (ids105/108/111/113). Threshold -10% adds2FP/noTP; +25% loses4TP.
Do not retune from sensitivity audit. No newtraining. Report
 docs/research/calibration-stability-2006-2026-09-26.md and corresponding tables.
Next bounded work: inspect prior2003–2005 calibration local-event coverage and
per-channel patterns to define one development-only improvement; keep baseline.


FULL-PERIOD TIMING RUN COMPLETED (user server). Reviewed saved outputs26Sep:
control selectedepoch1 cal0.833264 system0.689598 local0.782049 TP28FP0FN63;
age selectedepoch5 cal0.833229 system0.595655 local0.652065 TP28FP8FN63.
Both trained5epochs;4pooled/28annualrows present after runner finalintegritygates.
Do NOT promote age candidate. Historical40epoch baseline remains separate
reference system0.805722/local0.862907; systemtarget0.85notmet. No newtraining
launched during review. Next focus baseline missed events/calibration stability,
not assessment-driven tuning. Report docs/research/timing-comparison-results-2026-09-26.md.
Earlier statement “no GPU training launched” below refers to preparation stage.


User explicitly selected FULL-PERIOD matched timing comparison: inputs40–47,
train2000–2005/cal2006/assessment2007–2013, one model per arm. Implemented
`timing-comparison` CLI + TimingAwareAE, default5epochs each, uncapped nominal
windows, same seed/objective/architecture, freeze both before assessment, pooled
and yearly local/system reports. 68 tests PASS. No GPU training launched; this
session CUDA=False. See docs/research/full-period-timing-comparison.md for command.

Full sidecar recovery SUCCESS: observation_age_full_ready_v1. Original worker
hash audit: all16 NPZ copies match, two timestamp copies differ; channel42 has
one bit difference. Rebuilt deterministic grid in NEW recovery copies and its
hash EXACTLY matches original protocol. Original files preserved. Independent
streaming hashlib+sha256sum checks passed all recovered workers and assembled
files; protocol embeds recovery provenance. This supersedes earlier no-accepted-
sidecars status below, but does NOT establish underlying fault fixed. Evidence
outputs/integrity-review-20260926; never silently ignore future hash failures.


Baseline inference repeatability PASS exactly: two fresh CPU processes, one torch
thread/deterministic mode, saved2004epoch35 on4calibrationblocks x57windows=228.
Raw/scaled inputs,timestamps,reconstruction,errors,scores,predictions all identical;
model/source/input hashes match. No training/newF05. CUDA unavailable so NOT GPU
verification; does not fix intermittent fault or accept sidecar build. See docs/
research/inference-repeatability-2026-09-25.md and script inference_repeatability.py.
Full results results_longrun/diagnostics/baseline_2004_repeatability_v1. Continue
prototype only with its own successful integrity gates, not via this CPU pass.

Single differing calculation ISOLATED: channel44 index4420357 grid2004-03-14
20:18:30 minus native20:18:25.179 =4.821s; log1p(age/30) float32 is .14902326464653015.
Earlier .14900924265384674 differs941float32steps, not normal rounding. Standalone
NumPy18,780,560 element evaluations +Python scalar100k ALL MATCH. No reproducible
formula/NumPy-specific bug; intermittent fault unresolved. No full build/retraining
or manualvaluepatch. See docs/research/single-age-calculation-2026-09-25.md and
scripts/research/single_age_calculation_probe.py. Failed sidecar remains unaccepted.

Further isolation DONE, still NO accepted sidecar dataset. Controlled hashes vs
sha256sum and4unpickles42 stable; bounded nativeC256MB8patterns3passes clean (not
proof hardware healthy). Fresh workers pass40–42 then43timestamp hash fails.
Bounded two-matching-worker builder made22attempts/5failures; successful44outputs
differed in ONE computed feature (index4420357: .14900924265 vs .14902326465).
Eventually pairs matched for8channels BUT finalcopy recheck failed, so no root
protocol/completion. Keep trainingblocked on these artifacts; no bypass/manual
assembly. All originaldata unchanged. See observation-age-prototype.md latest
section; scripts/research/build_age_verified.py logs max4attempts perchannel.
Raw/problemroot still unknown, direct investigation requested; no admin handoff.

Observation-age prototype IMPLEMENTED;62tests PASS, but complete sidecar build
FAILED integrity gates in3 distinct attempts (Python42decodedhash, CPU2Python41,
independentunzip42buffer hash after parsing). Do NOT use partial arrays/train.
Independent unzip+sha256sum41 matched reference; fault not confined to ZIPdecoder.
No rawdata modified, no new model. Functions esa_thesis/observation_age.py;
builder scripts/research/prepare_observation_age.py (--output newdir, optional
--decoder unzip). Encoding causal log1p(age/trainingmedian cadence)+availability;
same8physicalchannels,2000–2003train/Jan–Mar2004cal only. Initial unavailable
sample marked separately. See docs/research/observation-age-prototype.md and logs.
Continue direct investigation, not admin handoff; do not bypass integrity checks.

USER says investigate directly, not refer to administrator. Direct checks DONE.
24archive probes: same compressed hashes, systemPython2CRCfailures despite in-memory
input; conda0/12. Independent unzip passes. Guarded audit attempts also caught47CRC
and decoded-buffer hash inconsistency. Root cause still unknown; no system/data fixes.
TWO complete guarded cadence audits (audit_b and audit_cpu2) MATCH BYTE-FOR-BYTE
tables and protocol/source hashes. Confirmed bounded-audit evidence now in
docs/research/tables/cadence-frequency-confirmed-2026-09-25; original provisional
outputs preserved. Missing initial-history window counted separately. No claims
CPU2 reliable/hardware fixed. See integrity-investigation-2026-09-25.md under
docs/research. Can proceed with guarded research using reproduced evidence; no
new model or long training launched. Future hypothesis observation-age representation,
not alarm suppression, still inputs40–47. Avoid another administrator handoff.

PRIORITY: cadence-frequency verification encountered INTERMITTENT READ INTEGRITY
FAILURES. Initial8channel pass completed, then channel43 ZIP CRCfailed;2 isolated
SHA/CRC rechecks passed. Next full rerun failed timestampordering on44; isolated44
read then passed ordering; next44read failedCRC. No raw files modified. Cause
unknown; do NOT claim repaired/hardware diagnosis. New frequency tables marked
provisional STATUS.json; no new training/preprocessing conclusions until stable
reads. See docs/research/cadence-frequency-review-2026-09-25.md. Initial provisional
41–46age>60s frequency .1117%train/.0591%cal,2/8000 nominal sampled windows; incomplete
initial-history window correction not fully verified. Priority now data-read/server
integrity investigation with administrator, not additional model experiments.

Native alarm review COMPLETE: Jan24 FP prepared flats align with longer native
observation intervals. Channels41–46:209native observations vs384prepared bins;
latest-observation age up214.812s (41–47), intervals up240s, many210s on41/47.
All reviewed prepared values match latest-native hold within tolerance. No gaps
>300s in reviewed contexts. Jan1TP/Feb14FP ages<4.815s on41–47, so FebruaryFP
not explained by this pattern. Association, NOT causal proof/packetloss diagnosis.
No suppressing alarms/relabeling/addinginputs/training. See native-alarm-review-
2026-09-25.md under docs/research. Next hypothesis needs training/calibration
cadence-frequency audit before any observation-age/preprocessing experiment.

2004 diagnostic v2 COMPLETE, calibration reproduction exact (F05 .38390663 TP1FP2).
Reviewed3plots + recomputed frozen predictions/metrics. TP Jan1 channel47 peak3.55125;
FP Jan24 multichannel peak3.64783 (192min); FP Feb14 channel46 peak3.62879(48min).
Threshold3.13217: raising it to eliminate either FP also loses sole TP. No threshold
or voting sweep justified. Jan24 channels41–46 repeated-value fractions52.5–61.1%
vs9–23.2% adjacent4h controls. Missingness/forward-fill cause NOT proven. Next
bounded proposal: native timestamps/preparation around Jan24 plus control contexts,
inputs40–47 only; no new model/training. id12 spans calibration start (ongoingevent).
See docs/research/baseline-2004-diagnostic-results-2026-09-25.md. v1 failure notes
below historical; v2 succeeded, no claim hardware root cause resolved.

2004 diagnostic v1 FAILED before inference at NumPy isin label validation:
IndexError index549755813888 into size4 lookup. Current CSV golden SHA verified
before/after exact failed period reload. Root cause NOT established; do not claim
hardware/storage fixed or definitively blame NumPy. load_period now validates each
label column per selected chunk with explicit comparisons0/1/2/3 BEFORE int8
narrowing, avoiding giant matrix/table lookup while rejecting invalid values.
Exact2000–2003 load PASSED (4,207,680x8;3,779,574 nominal samples);60tests PASS,
including huge/overflow/fractional/missing/string invalid labels; dry-run PASS.
Use NEW output baseline_2004_epoch35_v2; v1 contains protocol only, preserve it.
Source change prevents v1 resume. No GPU inference launched by assistant.

2004 calibration diagnostic READY, not GPU-launched. diagnose-channels now accepts
--fold2004 with historical source ae_baseline_remaining_seed42 (use spaces).
Output results_longrun/diagnostics/baseline_2004_epoch35_v1. Frozen epoch35/scaler/
rule; up to8k training reference windows2000–2003 and Jan–Mar2004 calibration.
Prepared-input cache + per-alarm8channel score/input plots + FP/event table.
Expected calF05 .38390663095893557 TP1FP2FN0; strict reproduction+CSVhash gates.
Historical source hashes absent, explicit provenance limitation. Current hashes
recorded. v2 diagnostic protocol prevents old-cache resume. 59tests and dry-run
PASS. See docs/research/baseline-2004-diagnostic.md. No assessment or retraining.

2003–2005 ORIGINAL40epoch baseline review COMPLETE. SystemF05 .833170/.428131/
.869055; local .833170/.447280/.869055. Calibration event counts3/1/4, calFP0/2/0.
2004 assessment9FP across6months, durations16–80min; typical nominal aggregate
scores stable monthly but tails vary. Raw signal drift/cause NOT established.
Local misses:2003 id13;2004 id91;2005 id16/100/104. Three other2004 misses are
outside input annotations. Frozen predictions and metrics reproduced; no training
or neural inference. See docs/research/baseline-years-review-2026-09-25.md.
Next bounded proposal: frozen2004 checkpoint calibration/per-channel diagnostic
around its2 calibrationFP and1TP plus training nominal references. Requires
fresh calibration inference since arrays absent; not launched. Do not tune to
known assessment FP episodes or start another model sweep. Inputs40–47 only.

Baseline2006 event review COMPLETE: all6 input-annotated IDs detected, localF05
0.999737; system6/10 F05=0.882148. Correct prior suggestion: NO missed local IDs
in this calibration. Four missed IDs105/108/111/113 annotated14/21/29 only; peaks
0.9–1.8% threshold, not near misses. Three short detections<30s require interval
overlap, not proof of timing/localization. See baseline-event-review-2026-09-25.md
under docs/research. No assessment/training/new threshold or input expansion.
This nearly saturated local calibration cannot justify further local-recall tuning;
future hypotheses need historical chronological development checks. Baseline retained.

Hidden-overlap 2004 pilot COMPLETED and verified, superseding READY below.
Control/candidate systemF05=0.090409/0.092948; local=0.091652/0.094262;
bothTP4, FP48/47, FN6(system); nominal alarm hours90.781/34.133.
Candidate calibration ZERO all5epochs; earliest epoch1 chosen, zero calibration
alarms. Minimal consideration gate technically passes but tiny improvement and
poor absolute detection do NOT justify promotion/long training. STOP this variant
for now, retain full baseline. No new run launched. Frozen predictions/dual metrics
recomputed, histories and source/annotation hashes checked; control exactly
reproduces previous pilot. See docs/research/masked-overlap-results-2026-09-25.md.

Hidden-overlap pilot READY, not GPU-launched: masked-pilot --comparison hidden-overlap
--folds 2004 --output results_longrun/development/masked_overlap_2004_seed42_v1.
Original masked control vs masked_overlap; only placement conditioned to touch a
hidden target, same length/type/channel/amplitude laws, geometric not guaranteed
value change. Both5epochs/25k/seed42; separate loss-component logs added.
59tests PASS and CLI dry-run PASS. Protocol v3/source changes block old-run resume;
preserve old artifacts. See docs/research/masked-overlap-pilot.md for predeclared
decision and user command. No automatic long run/benchmark assessment/promotion.

25 September objective audit COMPLETE: fixed128 nominal training windows, same
synthetic inputs, saved2004 epoch1/5 checkpoints, CPU eval mode, no training or
calibration/assessment inference. Both normal MSE and pseudo penalty improve
while historical calibration deteriorates. This suggests synthetic-task mismatch,
not a proven cause. Exact augmentation geometry misses hidden targets15.1935%;
sample unchanged hidden targets17.3828% (includes no-op corruption). Visible
corruption still affects predictions; do NOT claim zero gradient for these cases.
Report docs/research/masked-objective-audit-2026-09-25.md; reproducible fixed probe
scripts/research/masked_objective_audit.py. Train CSV golden hash before/after PASS.
No next training launched. Potential bounded next ablation: hidden-target-aligned
augmentation versus original masked control, predeclared and assessed honestly.

2004 training/calibration diagnosis COMPLETED (saved artifacts only, no new run).
Calibration contains ONE positive event ID. Baseline epoch1 F0.5=1.0 is therefore
weak generalization evidence. Masked epoch1=0.171616: every saved grid candidate
detecting that event has at least6FP; frozen rule has9.6h nominal false alarms.
Both arms' total training loss falls while calibration deteriorates; total loss
cannot establish the cause. See docs/research/masked-calibration-diagnosis-2026-09-24.md.
Next useful diagnostic: separate reconstruction/pseudo loss and score distributions
on fixed training samples, including synthetic-corruption/hidden-mask overlap.
No new training authorized by this diagnosis; retain full baseline and input40–47.
The older READY/await-results entries below are historical and superseded by
the completed follow-up and diagnosis above.

2004/2005masked pilot COMPLETED and verified. System baseline/masked:
2004=0.594608/0.090409 (TP5 FP3 vs TP4 FP48);2005=0.869116/0.869085
(bothTP4 FP0). Local2004=0.640295/0.091652;2005same as system. Both select
epoch1; masked2004calibration already0.171616 vs baseline1.0. DO NOT promote
masked or run40epochs/benchmark assessment. Retain original long baseline.
See docs/research/masked-pilot-followup-2026-09-24.md. Next bounded diagnosis
of2004TRAINING/CALIBRATION before another model idea; no test-specific tuning
or hindsight model selection by assessment year. No new runs launched.

2004/2005follow-up READY: masked-pilot --folds2004 2005 --output
results_longrun/development/masked_pilot_2004_2005_seed42_v2 (use spaces in CLI).
V2 adds per-fold subdirectories, strict epoch resume/completed-fold skip, automatic
local+system assessment reporting. Same25k windows/5epochs/seed42/masking/grid.
Root results.csv=system; dual_scope_results.csv=both; calibration_f05 is system.
58tests pass and two-fold dry-run passes. GPU follow-up NOT launched; user command
provided. Preserve2003v1; v1 cannot resume after version/source update.

MASKED PILOT2003 COMPLETED: both5epochs; both selectepoch1, calibration0.714286.
Apr–Dec2003matched baselineF0.5 0.714286(TP2 FP0 FN4), masked0.908853(TP4 FP0 FN2).
Local/system equal because all6IDs annotated on40–47. Masked addsid_12/id_90,
retainsid_19/id_89; missesid_13/id_88. Not robust0.85claim or full-run comparison.
See docs/research/masked-pilot-results-2026-09-24.md. Next recommend SAME small
protocol on2004/2005; CLI currently hardcoded2003, needs fold support first.
No further runs launched; no new benchmark assessment. Keep inputs40–47.

NEW supervisor requirement: report input-channel AND full-system event detection
from identical frozen predictions. Existing input set40–47 differs from message41–47;
no silent input change. Dual-scope baseline reporting completed: assessment local
F0.5 0.862907 (TP53 FP7 FN14/67IDs), system0.805722 (TP54 FP7 FN37/91IDs).
Of24IDs without input-channel annotations,1detected. Local0.85 is NOT system target
achievement or a model improvement. Calibration local0.999737 vs system0.882148.
See docs/research/dual-scope-evaluation.md; add both scopes to pilot reports after
rules frozen. Out-of-scope annotations remain neutral; no localization claim.
No package source/model/threshold changes; ongoing pilot unaffected.

Masked pilot IMPLEMENTED and ready for user GPU run: python -m esa_thesis masked-pilot.
See docs/research/masked-pilot.md. Channels40–47 only; matched25k nominal windows
2000–2002, five epochs per baseline/masked arm, Jan–Mar2003calibration everyepoch,
Apr–Dec2003development comparison after BOTH rules frozen. Same pseudo objective;
masked hidden-patch loss and four-pass inference. No benchmark test/OR/newchannels.
Strict epoch resume, CSV hash gates and progress. Full suite54passed plus targeted
five masked tests passed (one new test). Dry-run passed; full GPU run NOT launched.
Await user pilot results; do not automatically run more folds or40epochs.

Fixed q99 channel-normalization comparison COMPLETED (cached errors; no GPU).
Original calibration F0.5 0.882148 vs normalized0.882099; both TP6 FP0 FN4,
same detected IDs. Normalization increases nominal alarm duration by2410.947s.
RETAIN original scoring; no assessment run, no further percentile tuning or
GPU chronological checks for this unsuccessful candidate. Reference assessment
remains0.805722. See docs/research/channel-normalization-results-2026-09-24.md.
50 tests pass. Keep scope40–47; next modelling idea requires a justified small pilot.

Diagnostic v2 COMPLETED; independently recomputed cached predictions match
calibration0.882148, TP6 FP0 FN4. All six IDs annotated on40–47 detected; missed
105/108/111/113 have no input-channel annotation (not proof of unobservability).
Channel47 wins46.875% of alarm samples; winner counts are not feature importance.
See docs/research/channel-diagnostics-results-2026-09-24.md. No new assessment
score/model training. Single normalization comparison remains prospective; 2006
alone is weak for selection. Other channels remain out of scope.

24 September diagnostic interrupted: training inference completed but calibration
CSV parse failed. Active CSV SHA was 741aee07...; five differing bytes found.
Reconstructed exact golden CSV from preserved backup; independently verified disk
copy before install, old active file archived under data/repairs/2026-09-24.
Calibration loader subsequently passed (1,051,200 x 8); installed SHA again golden.
Use NEW diagnostic output ch40_47_epoch10_v2; preserve v1 but do not trust/reuse
its cache. Diagnostic now repeats train SHA at end; 47 tests pass. Root cause
unknown; repeated integrity incidents warrant server hardware/filesystem checks.

Diagnostic command READY: python -m esa_thesis diagnose-channels. See
docs/research/channel-diagnostics.md. Fixed epoch-10 checkpoint; max 12,000
nominal training windows and full 2006 calibration; cached per-channel errors,
event attribution, plots, strict reproduction gate, stage resume. No assessment
access or model training. 47 tests passed; actual checkpoint dry-run passes.
Full GPU inference has NOT been run by the assistant. Await user run/results.

Kaggle metric checked: same corrected event-wise F0.5 formula; exact hidden-scorer
parity unverified (continuous segments/sample penalty in page vs raw IDs/duration
in local official-reference implementation). Seven evaluator tests passed. See
docs/research/kaggle-metric-check-2026-09-24.md. Do not call local 0.8057 a Kaggle score.

USER OVERRIDE: work only on channels 40–47 to improve F0.5 before other inputs.
Counter prototype and broad channel comparisons are PAUSED, superseding older next steps.
Focused data/method research completed; see docs/research/ch40-47-improvement-review-2026-09-24.md.
Native/monthly/correlation/annotation evidence saved for 2000–2006 only. Calibration
has 10 global event IDs, six annotated on these inputs; matching detected count does
not establish matching IDs. Keep global metric. Next: fixed-checkpoint per-channel
training/calibration error diagnosis, then one normalized-max scoring comparison.
No diagnostic GPU inference or new training started. Do not expand channel scope.

Counter aggregation audit completed (raw training 2000–2005, channels 4–11).
5,340 observed changes; old point-hold geometry misses 4,714 (88.3%). Causal
30s bin counts retain all observed changes, not unobserved activity or anomaly
events. Raw gaps up to 5h57m require missingness/observation-age features.
See docs/research/counter-aggregation-2026-09-24.md. No data/model changes.
Next prototype separately with coverage and annotation-overlap checks before training.

Scaling audit completed for all 87 prepared inputs, training 2000–2005 only.
No nominal values clipped at ±10; 29 inputs use IQR fallback (18 telemetry,
11 commands). Counter nominal variation is extremely sparse/tiny. See
docs/research/scaling-review-2026-09-24.md. No training or transform changes.
Next measure causal counter aggregation before deciding a small matched pilot.

Long baseline COMPLETED: all 40 epochs; calibration selected epoch 10 (0.882148).
Pooled assessment 2007–2013 F0.5 0.805722, TP54 FP7 FN37, zero uncovered samples.
See docs/research/long-baseline-results-2026-09-24.md and saved compact tables.
Target 0.85 remains unmet. No new run started; continue preprocessing/input review.

Corrected full-resolution nominal correlation review completed for all 76 telemetry
channels in the current training period 2000–2005 (6,312,960 rows, 5,615,041 nominal).
See docs/research/preprocessing-and-correlation-policy.md and its versioned tables.
Script: scripts/research/training_correlation_review.py. Pearson pooled/yearly,
adjacent-change matrices, exploratory hourly Spearman, complete-linkage diagnostic
groups retaining singletons. No new model inputs selected, no preprocessing applied,
and no package source changed by this audit. Reuse old experiments; next resolve
preprocessing effects and freeze a small justified comparison, not another sweep.

Latest steering: follow supervisor sequence (exploration → preprocessing →
correlation → input comparisons), include coherent correlated groups, and reuse
previous experiments. Nine historical correlation/representative variants with
corrected metrics were located; see docs/research/historical-channel-review-2026-09-23.md.
Do not repeat the old sweep or choose groups from historical test scores. Legacy
normal filtering and min_size=3 cluster exclusions need provenance/method review.
The longer baseline has finished; preserve its source/protocol provenance.

Latest authorization: implement/run the longer baseline with channels 40–47,
training 2000–2005, calibration 2006, frozen assessment 2007–2013.
New command `python -m esa_thesis long-baseline`; see docs/research/long-baseline.md.
It uses ALL eligible stride-16 nominal windows (no 250k cap), preserves the
Transformer/pseudo-anomaly objective, verifies CSV hashes, and supports strict
epoch recovery. Outputs pooled and annual frozen-rule metrics. 44 tests pass.
The user completed GPU training on the server; this assistant runtime cannot access CUDA.
DATA GATE CLEARED: full raw-source integrity comparison (all features/labels)
and canonical CSV serialization restored the exact historical test SHA256.
Candidate-on-disk and installed-file hash reads matched the golden checksum.
Damaged original and evidence preserved in data/repairs/2026-09-23.
See docs/research/dataset-recovery-2026-09-23.md. Root cause is unknown.
The long-baseline command still verifies both hashes on every start/resume.
The broader input study is descriptive, not completed feature selection; this
is an explicitly requested fixed eight-channel baseline. Older notes saying
all training is paused are superseded by this authorization.

The user explicitly corrected the scope: investigate ALL available years,
2000–2013 (14 full years), not only 2000–2006. Training is 2000–2006;
benchmark test is 2007–2013, with a single boundary row on January 1, 2014.
The full descriptive audit is recorded in
 docs/research/full-mission-input-review-2026-09-23.md.
It reads raw values and metadata, not anomaly labels or model scores. Test-period
covariates are descriptively inspected; record this exposure. Do not use these
summaries to select/tune models then claim independent test performance.
No input exclusion, source deletion or training is part of this overview.
Full scan completed: 10,836 input/year records, all 774 inputs across 2000–2013.
All telemetry channels vary over the full period; some 8–11 are constant in
individual later years. 681 command types occur, 17 are absent in this interval.
The 2000–2006 records agree with the earlier audit. Figures and tables are local
under full-mission-review-20260923 / full-mission-review-2026-09-23.

User clarification: investigate ALL available channels, then ignore uninformative
inputs only in a model configuration. Do not delete sources or assume fewer inputs
perform better. The source inventory contains 76 telemetry channels and 698
telecommand types; only 11 telecommands are in the prepared CSV. Training remains
paused. Broad descriptive review completed for first-fold training (2000–2002):
all 76 raw telemetry channels vary; 458/698 commands occur, including 447 excluded
from current preprocessing. No automatic input exclusions. See
 docs/research/all-input-review-2026-09-23.md and its 774-row inventory/atlas.
This first-fold review is not validated feature selection; a separate seven-year descriptive overview is now available; next investigate
complementarity and specify small matched validation before further full training.

Latest supervisor feedback: prioritize understanding and cleaning the data,
channel informativeness/redundancy, small controlled subsets before full training,
and a presentation Excel with one row per year/subset and separate validation /
assessment metrics. See docs/research/data-first-plan.md. A training-only
2000–2002 audit and presentation workbook are complete; a full raw-data audit
and justified channel removal are NOT complete. Specialist and combination runs
are now complete. Training is paused in favor of raw-data/preprocessing review.
The first pilot found 76 telemetry inputs plus 11 telecommands. Do not automatically
remove constant or correlated inputs. Physical meanings are anonymized.
The presentation workbook is a September 22 snapshot and is now stale; update it
only when the user requests presentation preparation.

Completed combined assessment F0.5: 2003 0.499681, 2004 0.428131, 2005 0.869055.
The specialist was enabled by calibration only in 2003, adding five false alarms
without an extra detected event. Retain baseline; no further training authorized
by this audit. Specialist standalone: 0.166589 / 0.769221 / 0.740641.
Sources: results_longrun/development/ae_specialist_seed42_verified and
results_longrun/development/combined_specialist_seed42/results.csv.
These newly completed binary artifacts are not in the September 22 release.
See docs/research/raw-training-audit-2026-09-23.md for the audit findings.
The follow-up native-resolution review found raw changes in all channels 4–11,
including 75–292 changes each in nominal-prepared-constant channels 8–11.
The trace is now complete: most difference pulses miss the 30s grid; every
surviving nonzero sample in 8–11 is excluded by the global nominal mask.
Numeric reconstruction agrees with prepared values (zero tolerance mismatches).
Do not call them empty raw channels. See
docs/research/counter-preprocessing-trace-2026-09-23.md.
See docs/research/counter-channel-review-2026-09-23.md.

The old transfer manifest is stale after these edits. Regenerate it only if
migration is requested again; the prior checks do not verify this changed tree.

The user values direct, practical help and explicit runnable commands. They have
asked for code, documentation and essential experiment artifacts to be backed up
to their GitHub repository. They explicitly agreed to exclude datasets from
GitHub because they can be downloaded again. Never confuse the essential GitHub
backup with the full working-folder copy.

Repository: https://github.com/kamyarfaz/ESA-ADB (default branch main).
Old structure/history is associated with previous_experiments. Do not reset or
replace branches. Local branch main was clean before this handoff was added.

## Thesis goal and provenance

- Satellite telemetry anomaly detection on ESA Mission 1.
- Supervisor Federico Butchellato supplied main.py, models.py and utils.py and
  asked for a Transformer-based model with F0.5 at least 0.85.
- Tentative title: Artificial Intelligence for Anomaly Detection in Satellite
  Telemetry under Space Operational Constraints. No final title or deadline.
- Code exists; a full LaTeX/PDF thesis manuscript has not been created here.
- Source review and attribution: `docs/thesis-research-context.md`.
- The supplied architecture is a starting point, not automatically an original
  thesis contribution. The user previously requested removal of a “supervised by
  Federico” line from the public README; do not reintroduce it there unprompted.

References supplied by the supervisor:
- https://arxiv.org/abs/2406.17826 (PDF: https://arxiv.org/pdf/2406.17826)
- https://ui.adsabs.harvard.edu/abs/2024arXiv240617826K/abstract
- https://github.com/kplabs-pl/ESA-ADB
- https://github.com/Google-Developers-Club-Guido-Carli/ESA-Spacecraft-Anomaly-Challenge
- https://openreview.net/pdf?id=FYEGPuUrpo
- https://www.researchgate.net/publication/375693831_Annotating_large_satellite_telemetry_dataset_for_ESA_international_AI_anomaly_detection_benchmark
- https://www.kaggle.com/competitions/esa-adb-challenge

## Important architectural correction

**The successful baseline already is a Transformer autoencoder.**
`esa_thesis/models.py:MultivariateAE` uses temporal and channel Transformer
encoders and decoders. Its eight-channel checkpoint strictly loads into the
current model; it has 1,333,392 parameters. An earlier assistant suggestion to
“add a Transformer reconstruction model” was mistaken and explicitly corrected.
The separate forecasting Transformer has 302,224 parameters and performs poorly.
Do not conflate forecasting with retrospective reconstruction.

## Evaluation and selection rules

Authoritative implementation: `esa_thesis/evaluation/esa.py`.
Metric version: `esa-ew-id-duration-v1`. It uses raw annotation-ID events,
Anomaly plus Rare Event categories, neutral treatment of other annotations,
and the official-style nominal-time penalty on event precision. Parity with the
bundled evaluator is tested. It is not sklearn pointwise F0.5 or point adjustment.

Chronological development folds in `esa_thesis/development.py`:

| Fold | Training ends before | Calibration | Assessment |
| --- | --- | --- | --- |
| 2003 | 2003-01-01 | Jan–Mar 2003 | Apr–Dec 2003 |
| 2004 | 2004-01-01 | Jan–Mar 2004 | Apr–Dec 2004 |
| 2005 | 2005-01-01 | Jan–Mar 2005 | Apr–Dec 2005 |

Intervals are [start,end). Expanding training starts in 2000. Calibration has
only 3, 1 and 4 selected events respectively; assessment has 6, 10 and 7.
Sparse calibration is a major limitation. Checkpoints and decision rules are
selected on calibration only, then frozen before assessment loading. All these
assessment folds have now informed development: do not call them untouched tests.
The benchmark test was examined in earlier historical comparisons; do not claim
it has never been accessed. Current development commands avoid it and the
reserved final 2006 validation interval.

AE baseline: channels 40–47, sequence 256, train stride 16, scoring stride 32,
training-only nominal median/IQR scaling with ±10 clipping, max 250,000 nominal
windows, pseudo-anomaly objective, float32, seed 42, batch 64, 40 epochs.
Score = per-channel window-mean squared residual, averaged across overlapping
windows, then maximum across channels. Fixed quantiles and gap/duration candidates
are in development.py. Select corrected calibration F0.5 every five epochs and
final epoch; earliest checkpoint wins ties, with 1% calibration prediction cap.
The reconstruction objective option disables only the pseudo-anomaly penalty.

## Verified experiment results

All table entries below are assessment event F0.5, not calibration scores.

| AE experiment | 2003 | 2004 | 2005 |
| --- | ---: | ---: | ---: |
| Baseline, pseudo-anomaly, channels 40–47 | 0.833170 | 0.428131 | 0.869055 |
| Expanded, append channels 14/21/29 | 0.909069 | 0.251365 | 0.148245 |
| Baseline channels, reconstruction-only | 0.525691 | 0.150184 | 0.868934 |

Baseline TPe/FPe/FNe: 2003 5/1/1; 2004 6/9/4; 2005 4/0/3.
Reconstruction-only: 2003 4/4/2; 2004 6/41/4; 2005 4/0/3.
Keep pseudo-anomaly training: removing it worsened this seed-42 comparison.
Tiny nominal reconstruction loss did not imply better anomaly detection.
Neither expanded channels nor reconstruction-only training gave consistent gains.
**The ≥0.85 target has not been robustly achieved across folds.**

Paths under `results_longrun/development/`:
- `ae_baseline_2003_seed42/2003`
- `ae_baseline_remaining_seed42/{2004,2005}`
- `ae_expanded_seed42/{2003,2004,2005}`
- `ae_reconstruction_seed42/{2003,2004,2005}`
- `forecast_2003_seed42/2003/{persistence,mlp,transformer}`
- `forecast_residual_2003_seed42/2003/transformer_residual`

2003 causal forecasts: persistence F0.5 0.042852; MLP 0.082774; Transformer
0.086696 (6 TP, 79 FP, 0 FN). Last-value-centered residual Transformer:
0.070614 (5 TP, 82 FP, 1 FN), selected epoch 15, calibration F0.5 0.057466.
It did not improve results; further residual forecasting runs were paused.
A separate causal alarm-confirmation experiment also failed to approach the AE.
Read `docs/research/forecast-development.md`, `causal-alarm-findings.md`, and
`residual-transformer.md` before proposing to repeat these experiments.

Earlier paired scoring with identical historical AE weights:
window_mean validation 0.882109, benchmark test 0.761655;
per_timestep validation 0.588188, benchmark test 0.671790.
Path: `results_longrun/scoring_comparison/paired_v1`.
Do not confuse the strong validation score with an achieved test target.

## Latest diagnoses and decisions

Read these reports in `docs/research/`:
- `expanded-channel-results.md`
- `channel-error-findings.md`
- `channel-regime-findings.md`
- `baseline-2004-findings.md`
- `reconstruction-objective.md`

Expanded 2004: 33 false alarms are dominated by original channels 41–46.
Expanded 2005: all 35 false alarms are dominated by channel 14 (8) or 21 (27),
all in October. Their sustained ranges are high relative to calibration.
Clipping is absent in channels 14/21/29 through 2005. Many false alarms lack
unusually large jumps. Score magnitudes overlap true-event alarms. This supports
investigating range sensitivity but does not establish physical causation.
Blindly dividing channel residuals by nominal error scale is not a demonstrated fix.

Baseline 2004: nine false alarms across six channels, lasting 16–80 minutes,
May–December. Nearest annotation is over 18 hours away. Peak scores overlap
true detections; simply raising threshold or duration is not a proven fix.
Missed IDs: id_91, id_92, id_94, id_97. Three are annotated only on 14/21/29,
outside baseline inputs; id_91 includes input channel 47. Missing channel coverage
is not proof of an absolute recall ceiling because correlated signals may exist.

Current approved research direction: a separate detector for channels 14/21/29,
retaining the eight-channel model. Implemented as `develop --channel-set specialist`.
`combine-specialist` retains the baseline rule and calibrates an optional specialist
OR branch: <=1% combined calibration prediction rate, no extra calibration false
alarm events, ties disable the branch. See `docs/research/specialist-detector.md`.
These constraints did not generalize: both runs completed and the combination
failed to improve the baseline. See the current-intent section. Training is paused.

## Code map and verification

- `esa_thesis/__main__.py`: CLI entry point (`python -m esa_thesis --help`).
- `models.py`: Transformer AE and legacy MLP.
- `development.py`: chronological AE experiments and objective/channel options.
- `development_recovery.py`: atomic full epoch checkpoints and strict resume checks.
- `forecasting.py`, `forecast_development.py`: causal forecasting experiments.
- `evaluation/esa.py`: corrected metric; `evaluation/compare_scoring.py`: scoring comparison.
- `scripts/research/`: read-only diagnoses and transfer verification.
- `docs/README.md`: research documentation index.

Full thesis test command: `python -m unittest discover -s tests/thesis -p 'test_*.py' -q`.
36 tests passed during migration preparation; 40 pass after the specialist workflow. The doctor passed dataset/header
checks; it does not certify full CSV contents or GPU training.
Python 3.9 environment, torch 2.6.0+cu124, numpy 1.26.4, pandas 2.3.3.
Dependencies and installation instructions are in README and requirements-test.txt.
Do not use the root pip editable install for the thesis; setup.py is old TimeEval.

AE `develop --resume` restores full epoch checkpoints and skips completed folds.
Legacy partial 2005 expanded run was archived under `interrupted/` and restarted
from epoch 1; the replacement finished. Forecast development has no resume.
Use tmux to survive logout. If already inside tmux, do not create a nested session.
Strict resume checks may reject relocation or code changes; never bypass them
silently. Completed runs do not need training resumed to read their results.

## Data integrity and migration

Full folder about 119 GiB; at least 150 GiB free recommended for a single copy.
Includes actual raw/prepared datasets and full local results, not just Git files.
Main datasets: `data/preprocessed/multivariate/ESA-Mission1-semi-supervised/`:
`84_months.train.csv` and `84_months.test.csv`, about 6.9 GiB each.
Official raw v1.0: https://zenodo.org/records/12528696 .
Mission1.zip MD5: 80750189d171f5f398fb3d96c49df12b.

Two corrupt numeric bytes in the train CSV were repaired using raw telemetry;
full validation followed. Preserve repair records and do not blindly rerun repairs.
Verified prepared-file SHA256:
- Train: fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad
- Test: 9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c

`MIGRATION.md` explains full external-disk copy/archive, destination checksum
verification and environment recreation. `migration/manifest.json` is local,
ignored by Git, and describes the source snapshot. It is now stale because research code and Git metadata changed after migration
preparation. Regenerate it before any future transfer.
Verify the destination BEFORE running imports or Git that may alter caches.
Keep the original until destination verification reports zero failures.

Three absolute legacy W&B debug-log links have a preserved local target copy in
`migration/external-logs/`. They are not needed for training. Conda, credentials,
Codex chat history and supervisor originals outside this project are not included
merely by copying the folder. The project source/provenance documents are included.

## GitHub backup restore order

Source and instructions are on main. Essential binary artifacts are release assets:
1. https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-backup-2026-09-17
2. https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-update-2026-09-21
3. https://github.com/kamyarfaz/ESA-ADB/releases/tag/migration-update-2026-09-22

Restore and verify in that order; later snapshots overlay older files. Read each
release's RESTORE.md. September 22 adds 742 new/changed essential files, about
606 MB compressed, including reconstruction-only runs and diagnostic supplements.
Archive SHA256: 3033a1c1ffd70ede2e4e314d2bc619bc60074bb11b30c5e55925367752acd804.
Uploaded digest matched the local archive. Datasets, some older large score arrays,
and environments are deliberately not in these releases. The external-drive copy
contains more than the GitHub essential backup.

## First message to a replacement assistant

“Read AGENTS.md, AI_HANDOFF.md and MIGRATION.md in this project. We moved servers
and may have lost the old chat. Check whether migration actually happened. Research resumed with an approved
three-channel specialist experiment; read its protocol and inspect saved outputs
to determine whether it has run. Do not delete artifacts or assume results exist.”
