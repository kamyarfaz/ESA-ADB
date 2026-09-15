# Thesis research context and source-to-code review

Reviewed: 2026-09-15. This is a working reference, not the thesis manuscript.
Scope: supervisor-provided references, local research pipeline, metadata, saved summaries,
and targeted read-only diagnostics. No models were retrained and no experiment outputs changed.

## User context

The user has implemented the research code but has not yet created a LaTeX/PDF thesis.
The references were provided by supervisor Federico Butchellato. The user clarified that
Federico supplied three files and asked for a Transformer model to improve F0.5 to at least
0.85. All three were supplied and read in full: `main.py`, `models.py`, and `utils.py`
under `/home/k_faz/Downloads/Telegram Desktop/`. They establish the starting architecture
and custom scoring procedure, but contain no achieved baseline results or proof of which
evaluation definition the numerical target is intended to use. The tentative title is
*Artificial Intelligence for Anomaly Detection in Satellite Telemetry under Space
Operational Constraints*; it is not yet finalized.

There is no fixed deadline; the user wants completion as soon as reasonably possible.
User-reported training server: Intel Core i9-10980XE @ 3.00 GHz, NVIDIA GeForce RTX 3060
LHR (VRAM not reported), 62 GiB total RAM with 51 GiB available at the time of the command,
2 GiB swap, one Crucial MX500 2 TB SATA SSD and two Samsung 990 PRO 2 TB NVMe SSDs
(each displayed as 1.8T). Installed disk capacity does not establish available disk space.
The reported shell path was `~/projects/ESA-ADB-GitHub`; the inspected workspace here is
`/home/k_faz/projects/ESA-ADB`. Do not assume these are synchronized copies.

## Supplied baseline and attribution

- `models.py`: patch-based `MultivariateAE` with temporal/channel Transformer encoders
  and decoders, learned patch positions and channel embeddings. The local main script
  retains this core architecture, adapting channel count and configuration handling.
  Credit the supplied implementation as the starting point; the architecture is not
  established as the student's original contribution merely because it appears locally.
- `main.py`: six-channel experiment (41-46), normal-window MSE reconstruction training,
  AdamW, validation-selected checkpoint/threshold, W&B Bayesian sweep with a Hyperband
  configuration, test evaluation, plots, and checkpoints. The sweep metric is logged at
  the end of training; useful intermediate Hyperband behavior is not established here.
- `utils.py`: per-channel robust scaling on nominal training samples, chronological
  80/20 split, continuous-segment event counting, and the same nonstandard additive
  false-positive precision correction found in the local pipeline. The split and metric
  discrepancy were inherited from these files, not introduced solely by later thesis work.
- Main paths refer to `84_months.train_filtered.csv` and `84_months.test_filtered.csv`;
  the utilities contain other defaults which main overrides. No matching filtered files
  were found in the searched workspace paths. Their filtering procedure and retained label
  columns are unknown. The supplied loader ORs every `is_anomaly_*` column present,
  so six input channels do not by themselves prove six-channel ground-truth scope.
- Material scoring change: supplied `get_score_map` reduces only the singleton final
  tensor dimension, retaining per-timestep reconstruction errors before averaging overlapping
  windows. Current `score_series` also averages the time dimension and repeats each
  channel/window mean over its entire window. This changes anomaly localization and
  amplitude; it is not a numerically equivalent optimization. A NumPy shape diagnostic
  with one squared-error spike of 256 in a 256-point window gave one nonzero point of 256
  for the supplied calculation versus 256 points of 1 for the current calculation.
  This does not establish which scores perform better on real data.
- Later local additions include variable channel subsets, MLP forecasting, score ensembles,
  pseudo-anomaly separation training, calibration analyses, checkpoint resume logic,
  validation-rate constraints, and computation/memory changes. Personal authorship of
  individual additions has not been independently established.

## Sources and access

- [ESA-ADB arXiv abstract](https://arxiv.org/abs/2406.17826) and
  [PDF](https://arxiv.org/pdf/2406.17826): same work; current PDF is v2, revised
  2025-08-17, 87 pages. Reviewed main methodology and relevant appendices;
  visually checked the corrected-precision equation.
- [NASA ADS](https://ui.adsabs.harvard.edu/abs/2024arXiv240617826K/abstract):
  bibliographic link to the same arXiv identifier; direct retrieval failed.
- [Official implementation](https://github.com/kplabs-pl/ESA-ADB): benchmark source,
  distinct from the thesis-specific untracked scripts in this checkout.
- [Guido Carli reference repository](https://github.com/Google-Developers-Club-Guido-Carli/ESA-Spacecraft-Anomaly-Challenge):
  inspected README, repository tree, and hybrid notebook code without executing it.
  Its notebook combines Isolation Forest, moving-average prediction errors, engineered
  features, and XGBoost. Despite its headings, its `compute_telemanom_scores` function
  does not implement an LSTM; `use_lstm` is unused. Separate Telemanom implementation
  files exist under `BenchmarkAlgorithms/telemanom_esa`. Notebook cell 9 shuffles time
  samples and cells 11/12 independently fit validation/training scalers. Treat it as
  exploratory reference code, not a verified benchmark or an authoritative protocol.
- [Supplied OpenReview PDF](https://openreview.net/pdf?id=FYEGPuUrpo): indexed title is
  *European Space Agency Dataset and Benchmark for Anomaly Detection in Real-World
  Time Series*. Exact PDF, forum, and API retrieval were blocked by verification/403.
  Its full text and review status remain unverified.
- [2023 annotation paper](https://www.researchgate.net/publication/375693831_Annotating_large_satellite_telemetry_dataset_for_ESA_international_AI_anomaly_detection_benchmark):
  read accessible full text. Explains converting operator reports into channel-specific
  intervals through OXI, manual review, and algorithm-supported refinement. Its early
  dataset description should not replace the released dataset metadata.
- [Kaggle challenge](https://www.kaggle.com/competitions/esa-adb-challenge): read
  rendered overview/evaluation and data pages. It uses the public 14-year Mission 1
  series for training and a separate six-month test fragment. Its 87 inputs comprise
  58 target channels, 18 auxiliary channels, and 11 telecommands. Binary labels combine
  anomalies and rare nominal events; events for scoring are continuous positive segments.
  The false-alarm correction is multiplicative, not the custom local denominator.
- Additional authoritative source found through the official repository/site:
  [DMLR journal paper](https://data.mlr.press/assets/pdf/v03-23.pdf), volume 3,
  article 23, 2026, 76 pages. Its title is *European Space Agency Dataset and Benchmark
  for Real-World Anomaly Detection in Spacecraft Time Series*, associated with a
  different OpenReview ID, `bbpRMoatVO`. Reviewed relevant methodology and appendices.
  It discusses false-alarm amplification as channel count increases and the difficulties
  encountered with TranAD and Anomaly Transformer. It is not evidence of the exact
  contents or status of the supplied OpenReview submission.

## Research interpretation

Confirmed objective: use a Transformer-based model to reach F0.5 >= 0.85. This review does
not establish that the target has been reached under a correct, agreed evaluation protocol.
Working interpretation of the code: study the interaction between channel selection,
reconstruction/forecasting scores, and decision thresholds for Mission 1 anomaly detection.
Potential contributions include correlation-based channel reduction, temporal/channel
Transformer attention, synthetic-anomaly separation training, and calibration analysis.
These are implemented research directions, not yet established novelty or validated gains.

## Local data and pipeline

Verified from local `data/ESA-Mission1/channels.csv` and annotation files:

- Mission 1: 76 physical channels, 58 target and 18 auxiliary; 200 annotated event IDs
  comprising 118 anomalies, 78 rare events, and 4 communication gaps.
- Mission 2: 100 physical channels, 47 target and 53 auxiliary; 644 annotated event IDs,
  comprising 31 anomalies and 613 rare events.
- The active Mission 1 CSVs have 175 columns: timestamp, 87 inputs, and 87 corresponding
  label columns. There is no global `is_anomaly` column.
- Preprocessing uses a 30-second grid. `84_months.train.csv` starts at 2000-01-01;
  `84_months.test.csv` starts at 2007-01-01. These anonymized dates are dataset coordinates.
- Main saved runs have 5,891,328 training rows, 1,472,833 validation rows, and 7,364,161
  test rows. Validation is the last 20% of the training CSV, about 17 months.
- `get_y_any` collapses all positive label codes across every label column. This includes
  gap labels if present. Raw label categories and IDs are not retained by that operation.

Active entry point: `mission1_reconstruction_ae_sweep_optimized.py`.

1. Load selected input channels and all labels; fill missing values.
2. Make a chronological 80/20 training/validation split.
3. Fit median/IQR scaling on nominal training samples and clip to [-10, 10].
4. Train on windows without positive labels: length 256, stride 16, up to 250,000 windows.
5. Transformer AE uses temporal and channel encoders/decoders, patch size 16,
   dimension 128, eight heads, and up to 40 epochs. This is not DC-VAE or Telemanom.
6. AE scoring averages squared error over each window/channel, spreads that value over
   the window, averages overlapping windows, then takes the maximum across channels.
7. MLP forecasting uses 256 context points to predict 32 future points, with MAE scores.
8. Ensemble uses maximum of independently min-max-normalized AE and MLP score series.
9. Threshold, merge-gap, and minimum-duration candidates are evaluated. Current
   `pick_leakfree_row` selects using validation under a preferred 1% predicted-rate cap.
10. Save checkpoints, histories, score arrays, masks, plots, and summaries.

The final sweep log reports 31 successful runs and zero failures on 2026-07-17.
Older rescoring tables cover only 26 or 28 runs. Reusing an old run directory with
existence-based caching does not establish which code/configuration produced each artifact.

## Confirmed evaluation discrepancies

### Precision formula

Paper equation 1 and the bundled `timeeval/metrics/ESA_ADB_metrics.py` use:

`P = TPe / (TPe + FPe) * (1 - FPt/Nt)`.

The full benchmark uses durations for the last factor and preserves annotation IDs.
It reserves the last three months of the training half for validation. Its channel-aware
metric evaluates identification of affected channels, not just event coverage in a subset.

The main thesis script and several custom rescorers instead use:

`P_custom = TPe / (TPe + FPe + FPt/Nt)`.

A read-only synthetic diagnostic executed only the main script's extracted metric helpers:
89 separated true events, predictions positive at all 1,000 samples. It returned
`esa_f05 = 0.9910913`; the published correction gives zero precision. The script also flags
this case as saturated, so this diagnostic identifies a formula defect, not a claim that
the saturated candidate would be selected by the sweep.

Arithmetic using saved confusion counts, keeping predictions and event counts fixed:

| Saved mask/run | Custom F0.5 | Multiplicative formula only |
|---|---:|---:|
| `center_ch12_38_49`, 1% validation rule | 0.754755 | 0.727652 |
| `pa_center_ch12_38_49`, 1% validation rule | 0.745815 | 0.719817 |
| `pa_clust_med_t03_6ch`, 1% validation rule | 0.845215 | 0.737336 |

These are diagnostic recalculations, not official benchmark results: annotation-ID matching,
time-domain evaluation, threshold reselection, and protocol alignment remain outstanding.
In particular, the 0.857906 value displayed for `pa_center_ch12_38_49` in a sweep ranking
belongs to other validation-rate caps, not its saved primary 1% rule.

### Event identity and scope

The cached test binary mask contains 89 continuous positive segments. Merging gaps of up
to 300 samples (150 minutes) produces exactly 84 segments; 60 samples produces 87.
This demonstrates a way the saved 84-event table can arise, but does not prove its run
arguments because they are not recorded in that CSV. Raw annotations overlapping the
test period contain 91 event IDs (55 anomalies and 36 rare events). These counts answer
different questions and must not be interchanged.

`channel_aware_rescore_v2.py` evaluates an aggregated prediction against mission segments,
with either subset-relevant or all-event recall. It does not compute official channel
localization. Its header says precision is identical in both protocols, but the function
uses scoped TPe in precision, so changing scope can change precision as well.

### Temporal information and selection

- AE scores assigned to early positions use later values in the same 256-point window.
  They are retrospective scores unless an explicit window-end detection delay is used.
- Ensemble normalization independently uses minima/maxima over the entire validation and
  test series. It is not a fixed training/validation-fitted transformation and is not causal.
- The DSPOT-named script fits one POT threshold using an initial part of test scores and
  applies it to the whole series, including that calibration prefix. Its trailing median
  does not make this full-series evaluation causal; a deployment claim needs a warm-up
  exclusion and an explicit update protocol.
- Current main threshold selection is validation-based. Earlier outputs and auxiliary
  scripts are not automatically covered by that fix. `channel_importance_analysis.py`
  uses test labels and scores for channel ranking and thresholding; this is retrospective
  diagnostic analysis, not independent training-only feature selection.
- Comments such as “pre-registered,” “leak-free,” and “official” are claims to verify,
  not proof of experiment provenance.

## Channel selection details

`channel_correlation_analysis.py` claims normal-only input, but its loader looks for global
label names and not the actual `is_anomaly_*` columns. With the current CSV schema it
falls back to all loaded rows. Therefore normal-only clustering is not established by
the present implementation. The saved correlation matrix is 76 by 76; the comment claiming
65 Mission 1 channels is stale, while actual channel discovery includes all 76.

The six selected medoids are channels 3, 5, 13, 29, 48, 57. Channels 3 and 5 are auxiliary.
Auxiliary inputs can be useful predictors, but they are not separately annotated detection
targets. Input reduction and output monitoring scope must be described separately.

## Outstanding work, not performed by this review

- Establish how the supervisor's filtered CSVs were prepared and whether F0.5 >= 0.85
  refers to his supplied custom metric or the published corrected metric. Keep both
  explicitly labeled if reproducing his baseline alongside a benchmark-aligned evaluation.
- Obtain the exact supplied OpenReview PDF if comparison with that version is required.
- Establish one fixed evaluation protocol and verify custom metrics against the reference.
- Align baseline splits, channels, event categories, and online/offline assumptions.
- Reevaluate saved predictions consistently before interpreting model improvements.
- Audit experiment provenance, statistical variability, and test-informed design decisions.
- Preserve exploratory results as such; do not claim published-baseline superiority from
  the current heterogeneous score tables.

No thesis manuscript, code fixes, new training, or final performance certification was
requested or produced during this review.

## Subsequent cleanup: 2026-09-15

At the user's request, inspected and deleted the former `trash/` directory after retaining
its source scripts and small result/configuration records in
`archive/legacy_experiments_2026-09-15.zip`. See `archive/README.md` and the internal
manifest for scope and verification. Historical paths mentioned above may now refer to
archived records. Large historical model, score, prediction, and plot artifacts were
discarded; current experiment outputs and datasets were untouched.

## September 15 source organization

Active sources now live in `esa_thesis/`; consult `source-migration.json` for old-to-new filenames and the root README for commands. Earlier filename references above describe the reviewed pre-organization sources. Numerical definitions were preserved and regression-checked; the scientific limitations above still apply. The former `main/` snapshot is now `archive/snapshots/main_v1/`.
