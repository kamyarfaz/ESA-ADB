# Short-no-roll: 2004 training and calibration diagnosis

## Decision

Keep the existing full-period baseline. Removing short rolls reduces average
normal reconstruction error at a matched final epoch, but does not reliably
separate real calibration anomalies from normal telemetry. Do not promote the
candidate, choose a seed, relax the alarm cap, or change thresholds based on
assessment results. No new training was performed in this diagnosis.

## Data and checks

Six saved models: short_coordinated versus short_no_roll, seeds 42/43/44.
Inputs remain channels 40–47. Selected-checkpoint diagnostics use 1,024 identical
nominal windows from 2000–2003 and all January–March 2004 calibration samples
(8,183 scoring windows). The training CSV matched its golden SHA256 before and
after loading. Original training-window indices were checked for nominal labels
and dates before 2004. Model, protocol and annotation source hashes were checked.
No assessment arrays or 2007–2013 data were read.

Calibration contains just one selected event ID, id_12, clipped to January 1–8
at noon. This is a weak basis for choosing an epoch and a decision rule.

## Why the selected rules differ

These values describe the original GPU calibration candidates at the selected
epoch, not a newly tuned threshold sweep:

| Seed | No-roll epoch | Calibration F0.5 | Detected IDs | False alarm events | Minimum alarm fraction among detecting grid candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 1 | 0 | 0 | 0 | 1.9902% |
| 43 | 1 | 0 | 0 | 0 | 4.9939% |
| 44 | 3 | 0.093391 | 1 | 12 | 0.9890% |

For seeds 42 and 43, calibration F0.5 is zero at all five epochs. The existing
earliest-tie rule therefore selects epoch 1. At those selected epochs, all
detecting grid candidates exceed the fixed 1% total-alarm cap. Selection chooses
the maximum-score threshold, which produces no calibration alarms. It can still
produce alarms on later observations with larger scores.

Seed 44 finds a detecting rule within the cap, but it also makes 12 false alarm
events. Its event peak is only 1.114 times its threshold. The twelve false events
are dominated by channels 41 (4), 43 (3), 45 (1), 46 (1), and 47 (3). This does
not support removing a single supposedly bad channel.

## Normal-error tails and matched epochs

For seed 44's selected no-roll checkpoint, median maximum-channel window error
is 0.148 in nominal January windows and 0.153 in February. However, its 99th
percentile rises from 2.177 to 8.324. A low typical reconstruction error can
coexist with large errors on some normal observations. These are window-error
statistics; the decision threshold applies to overlap-aggregated scores.

Selected checkpoints differ in epoch, so their plots alone cannot attribute
changes to augmentation. A separate matched epoch-5 probe uses 512 identical
nominal training windows for all six models:

| Seed | Control mean normal MSE | No-roll mean normal MSE |
| --- | ---: | ---: |
| 42 | 0.168189 | 0.129057 |
| 43 | 0.215907 | 0.192715 |
| 44 | 0.228189 | 0.165146 |

Normal reconstruction improves on this training-only probe, but this is not
evidence of better real-anomaly separation. The epoch-5 synthetic probe deliberately
uses the same old short-coordinated mixture for both arms, including removed short
rolls. It is a diagnostic of response to identical inputs, not the candidate's
training loss or an assessment score. Short-roll hinge penalties remain high
(about 9.14–9.31 for no-roll models), as expected for weak changes. Removing them
from training has not solved the real calibration discrimination problem.

## Numerical limits

Four of six CPU calibration replays reproduce all checked saved metrics within
1e-6. Seed 43 control has 960 extra nominal alarm seconds and F0.5 changes from
0.11004108 to 0.11002661. Seed 43 no-roll has one extra 32-sample / 960-second
false alarm, with unchanged zero F0.5. All its alarm scores exceed the saved
threshold by only two float32 ULPs (0.000000953674). This is consistent with
CPU/GPU numerical boundary sensitivity; original GPU masks were not available
to prove the exact source. Original reported metrics and thresholds are retained.

## Next bounded step

Before another training comparison, audit how representative calibration is
using only training/calibration data: inventory earlier real events and normal
operating regimes, and specify a chronological inner-validation design within
the pre-assessment data. Compare any proposed selection policy with the existing
policy under a fixed budget; explicitly decide how zero-information calibration
ties should be handled. Do not silently expand the existing calibration period
or rewrite old results. Keep the same real April–December assessment observations
for comparisons and disclose that these periods have already informed development.

## Reproduction and artifacts

- `scripts/research/diagnose_augmentation_2004.py --comparison short-no-roll`
- `scripts/research/probe_short_augmentation_objective.py --comparison short-no-roll`
- `results_longrun/diagnostics/short_no_roll_2004_multiseed_v1/`: source manifest,
  calibration metrics, selected-grid audit, channel/event tables, matched-epoch
  histories, score plots and reconstruction plots.
- `results_longrun/diagnostics/short_no_roll_objective_2004_v1/`: fixed epoch-5
  probe, per-window results and hashes.

Both scripts preserve their previous default comparison. Seven corrected-metric
parity tests and seven augmentation tests passed. Both full diagnostic executions
completed with stable input artifact hashes. No source training/evaluation logic,
saved checkpoints, rules or prior results were changed. No GitHub push.
