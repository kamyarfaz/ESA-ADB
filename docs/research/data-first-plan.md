# Data-first research plan following the supervisor meeting

## Goal and sequencing

Understand what each input contains, justify a smaller channel set, and compare it
fairly with broader input sets before tuning model parameters. Let the existing
specialist training finish. Do not modify running source files, preprocessing,
inputs or protocols. Further training is gated on the channel study below.
This is anomaly detection with nominal training and annotated calibration/evaluation,
not ordinary supervised classification. Classifier feature-importance rules do not
automatically transfer to reconstruction-based anomaly detection.

## First completed pilot

Read all 3,156,480 prepared rows from 2000–2002 (first-fold training only).
2,819,808 are nominal across all 87 paired annotation columns. The 87 inputs are
76 telemetry channels and 11 telecommands. No assessment data entered this audit.
10 inputs are exactly constant across all nominal training rows: telemetry 8–11,
and telecommands 244, 350, 352, 353, 354, 376. Nine further inputs look constant in
the fixed hourly sample but are not constant at full resolution: telemetry 4–7
and telecommands 351, 36, 38, 39, 40. This demonstrates why sample-only filtering
would lose rare behavior. No channels were removed.

A 23,495-row hourly nominal sample supports exploratory distributions and Pearson/
Spearman correlations. There are 184 pairs with |Pearson| or |Spearman| >=0.95.
Channel 14–21 Pearson is about 0.99665. Correlations can change by year: for example
40–66 is approximately 0.9388, 0.7660, 0.9734 in 2000, 2001, 2002. One global
correlation threshold is not a reliable universal deletion rule.

The local metadata uses anonymized subsystem and physical-unit codes. Record
those codes, target flags and groups; do not invent temperature/voltage meanings.
The existing correlation command looks for a global label and can fall back to
all rows on this schema. Do not use that legacy output as nominal-only evidence.
The new pilot explicitly checks all paired labels and never invokes that fallback.

Prepared missingness is not raw missingness: the preprocessing resamples with
forward filling and later forward/backward filling, and transforms designated
monotonic channels. Finite prepared values do not prove uninterrupted raw sensing.
The pilot checks finite values, label codes and the 30-second grid, but is not a
complete source-to-preprocessed or physical-semantic audit. Hourly samples can
miss short events and alias periodic behavior. Constant nominal channels can
be valuable when they change abnormally; correlation breakdown can itself be an
anomaly. These flags are review candidates, not automatic deletion decisions.

## Personal working data area

`research_workspace/channel_study_2026-09-22/` holds:
- `pilot/`: channel profiles, hourly training sample, monthly summaries,
  correlation matrices/pairs and sampling protocol.
- `results_snapshot.csv` and JSON: completed experiment metrics captured from
  frozen result files. Pending specialist folds have blank metrics, never zeros.
- Later `subsets/`: immutable channel lists and row-index manifests, once justified.
- Later `processed/`: versioned derived subsets, never overwriting source data.

Original data stays under data/. Do not make another full 7 GB copy to create a
personal workspace. Store indexes, manifests and small samples first. Source
corruption incidents make checksum verification and a clean provenance chain
necessary before claiming that a cleaning operation improved model performance.

## Work plan and deliverables

1. **Inventory and raw-data quality.** Complete the channel dictionary and inspect
   raw timestamp gaps, duplicates, sampling cadence and missingness by channel.
   Separate telemetry, counters, categorical states and telecommands using actual
   metadata/source behavior. Check raw-to-prepared changes, filling and derivatives.
   Deliver a data dictionary, integrity report and explicit unresolved meanings.
2. **Distributions and temporal behavior.** Plot full-resolution training excerpts,
   monthly ranges, step/change rates, flat stretches and outliers. Audit normal
   and labelled training-event examples separately. Do not erase real anomalies
   as “outliers.” Document any winsorization, scaling, filling and its limitations.
3. **Redundancy and coverage.** Extend the current pilot to each fold's own training
   history, compare Pearson/Spearman and correlations of changes/lagged signals,
   and inspect stability by month/year and subsystem. Keep correlation-break
   pairs where needed. Use training annotations to describe covered event types,
   not assessment outcomes to choose channels. Provide reasons to keep/review/drop.
4. **Small matched pilot.** Freeze 3–4 candidate channel sets: original 8 as control,
   a quality-screened broad telemetry set, a representative-per-stable-group set,
   and a representative-plus-complement set. Exact members are not chosen yet.
   Treat telecommand inclusion as a separate design choice. Use identical nominal
   windows stratified across training months, a fixed seed and a small training
   budget (proposed 10,000 windows and 5 epochs). Keep original-resolution windows;
   do not train on the hourly EDA sample. Use training-internal chronological
   validation. These smoke comparisons are screening, not final performance claims.
5. **Full controlled comparison.** For candidates fixed after screening, use the
   same chronological protocol, full calibration and assessment periods, fixed
   preprocessing and evaluation scope. Compare channel counts AND window budgets
   separately (e.g. 10k/50k/current cap) so feature reduction and data reduction are
   not confused. Record wall time, peak memory, false-alarm duration and recall.
   No random time-series split. Existing assessment periods are development data.
6. **Tuning and robustness.** Only after selecting the data policy, freeze a bounded
   tuning grid, use calibration/internal validation, and repeat selected settings
   across seeds. Discuss calibration event scarcity and split rationale with the
   supervisor before changing evaluation. No tuning on benchmark test outcomes.

Do not execute the proposed pilot budgets or new subsets until the required data
checks and exact subset lists are recorded. No new model run was launched here.

## Presentation workbook

One row per experiment/year/channel subset, columns for calibration and assessment
F0.5, precision, recall, TP/FP/FN, false-positive duration, epoch, seed and completion.
The workbook separates the three full AE comparisons from forecasting designs,
which use different causal information/alarm policies. Specialist rows are marked
as an ongoing campaign. Summary plots exclude the incomplete campaign. Calibration
is Jan–Mar and assessment Apr–Dec; rows do not represent full-calendar-year scores.
Record model design, subset, source run identifier and current metric version.
Keep unsupported/missing values blank. Never present a single seed as uncertainty
quantification or a calibration score as a held-out test score.

## Sources and next questions

- Local channels.csv, telecommands.csv, prepared training CSV, preprocessing code,
  saved protocol/frozen-rule/results files and corrected evaluator.
- https://github.com/kplabs-pl/ESA-ADB — benchmark source, preprocessing and metrics.
- https://arxiv.org/abs/2406.17826 — benchmark paper.

Questions for the next meeting: which operational latency is acceptable; whether
full-year assessment or the current development split is preferred; whether physical
channel descriptions are available beyond anonymized codes; how to define channel
coverage versus global-event detection. These are open questions, not blockers to
training-only data exploration. “Fully understand the data” is the objective of
this staged study, not a claim already achieved by the first pilot.

## Reproduction and presentation files

Run `python scripts/research/training_channel_pilot.py --output NEW_DIRECTORY`
for the CPU-only training audit. A new output directory is required. The result
snapshot builder is `scripts/research/prepare_meeting_results.py`; rerunning it
refreshes the local snapshot from completed result files but does not touch models.
The figure builder is `scripts/research/plot_channel_pilot.py`. Both currently
use the dated study directory. The workbook builder takes a combined JSON payload
and output directory as arguments and requires @oai/artifact-tool in its Node
runtime. Its payload combines the result snapshot with profiles/pairs and protocol.
The workbook was visually checked on all seven sheets; the formula scan found no
errors. Native overview chart links to the Results cells. Three standalone PNGs
provide assessment comparisons, training correlations and channel examples.

Presentation: `outputs/esa-thesis-review-20260922/ESA_research_review.xlsx`.
Evidence tables: `docs/research/tables/channel-study-2026-09-22/`.
The workbook is a snapshot, not an automatic monitor of the ongoing specialist run.
