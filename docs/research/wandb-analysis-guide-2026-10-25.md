# Reading the W&B fine-tuning study

W&B records and compares experiments. It does not itself establish model quality
or automatically improve F0.5. Training loss, calibration selection and assessment
answer different questions.

Project: https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning

## Find the relevant runs

Use the Runs page and filter by group:

- `historical-baselines`: original reference results.
- `learning-rate-comparison-20261024-seed42-v1`: nine training runs,
  three learning rates per technique.
- `verified-assessment-20261025`: three separate evaluation runs containing
  original and selected annual/pooled results.

Within one technique, compare its three rates. Plot `train/loss` against `epoch`
and inspect `calibration/system_all_channels/EW_F_0.50` and
`calibration/input_channels_40_47/EW_F_0.50`. Candidate calibration was recorded
at epochs5 and10. `selection/epoch = 0` means the original checkpoint won;
`selection/calibration_system_f05` is the retained calibration score.
Lower training loss alone does not establish better event detection.

The first two techniques retained their original models. Extra-input short-no-roll
selected LR3e-5, continuation epoch10. See the full calibration comparison rather
than comparing only the last training loss.

## Read assessment results

Open `assessment/annual_and_pooled` in an evaluation run. Compare `original`
against `selected`, using the same scope and period. For the target task use
`input_channels_40_47`; `system_all_channels` measures a different event population.
Start with `2007-2013`, then inspect each year. Do not select per-year settings
from these assessment results. Pooled metrics are recomputed, not annual averages.

- Event TPR: detected target IDs divided by all target IDs.
- Precision: corrected ESA event precision including nominal-time penalty.
- Time FPR: false-alarm duration divided by nominal duration.
- Event F0.5: precision-weighted balance with event recall.

Original short-no-roll has target F0.5 0.907140 and47/67 detections. Masked
pretraining plus fine-tuning has0.844315 and60/67. Tuned extra-input short-no-roll
has0.830900 and62/67. Extra-input tuning improved its own baseline, but did not
surpass original short-no-roll on target F0.5. Higher coverage is not automatically
higher F0.5. Full metrics and annual event IDs are linked in the results report.

## Proposed next investigation — not an authorized running experiment

Before another learning-rate search, diagnose2006 calibration alarms: identify
which channels contribute false alarms, then compare true and false alarm score
magnitudes, persistence, durations and cross-channel agreement. Previous alarm
refinements did not deliver assessment improvements; do not repeat them blindly.
A new candidate should follow a specific calibration finding and be checked on
chronological development periods before assessment.

The completed study selected on system calibration F0.5. If the primary objective
is channels40–47, a future protocol should explicitly predeclare target-channel
calibration F0.5 as the selection metric and a detection-retention constraint.
This would be a new protocol, not a retrospective change to the completed study.
No new search settings, training or follow-up schedule are authorized by this guide.

Use calibration/development evidence to choose candidates, freeze them before
assessment, and preserve unsuccessful outcomes. The repeatedly inspected2007–2013
benchmark is not fresh independent validation. A single seed, extra input
channels, differing prior compute and the repeated short-no-roll LR1e-5 setting
must remain disclosed. Improvement is not guaranteed.

[Verified results](wandb-finetuning-results-2026-10-25.md)

[W&B filtering and grouping documentation](https://docs.wandb.ai/guides/runs/filter-runs)
