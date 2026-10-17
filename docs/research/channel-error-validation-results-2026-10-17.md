# Channel-error diagnostic: complete validation cohort, incomplete calibration replay

All six frozen models now reproduce their original development-validation scores
within rtol=1e-5, atol=1e-6 and their saved binary predictions exactly. This covers
July–December 2001 and 2002, seeds 42/43/44. No model, threshold, channel set or
prediction changed. No 2006 or benchmark telemetry was evaluated.

The original full replay gate still fails: 17/18 stages are verified, comprising
six validation, five calibration and six training summaries. The missing stage
is 2002/seed44 calibration. Matching the original two CPU threads did not resolve
it: CUDA passes the mask check but fails score tolerance; CPU passes score tolerance
but differs at 32 binary samples. The failed outputs and `stopped.json` remain.
There is no whole-run `complete.json` or claim of full numerical reproducibility.

## Findings

The table includes every input-annotated validation event and all three seeds.
Values are the percentage of nominal timestamps with error at least the event's
peak; smaller means a more unusual peak. Medians/ranges span annotated channel ×
seed rows, not independent observations or confidence intervals. Each channel uses
its own annotated interval. All annotation intervals, including gaps, are excluded
from the nominal reference.

| Fold | Event | Channel × seed rows | Individual-channel percentage: median (range) | Scalar maximum percentage: median |
|---|---|---:|---:|---:|
| 2001 | id_37 | 18 | 3.356 (0.713–5.118) | 1.855 |
| 2001 | id_38 | 18 | 0.000 (0.000–0.039) | 0.020 |
| 2001 | id_80 | 18 | 0.043 (0.000–2.150) | 0.059 |
| 2002 | id_25 | 6 | 66.942 (56.533–83.180) | 67.023 |
| 2002 | id_86 | 21 | 0.000 (0.000–69.576) | 60.587 |

**There are at least two distinct score limitations.** For id_86, channels 42–46
have more unusual individual error peaks than the scalar maximum in every seed.
Channel 47 dominates the scalar score and masks this relative evidence. But for
id_25, both annotated input channels have ordinary upper error peaks in every
seed: 56.5–83.2% of nominal errors exceed them. Their event medians are lower than
nominal medians. Changing maximum aggregation alone therefore has no demonstrated
solution for all missed events. Lower errors do not establish a useful two-sided
rule; earlier adaptive-score results were inconsistent.

The 2001 examples often have unusual peaks but the saved frozen rules still miss
them. This is consistent with a calibration-transfer limitation; peak rarity alone
does not explain postprocessing or establish a usable alarm threshold.

These are retrospective descriptions, **not FPR, TPR, F0.5, a channel-selection
rule or evidence of improved detection**. Nominal references use the same internal
validation segments being described. The failed calibration replay limits any
claim about the full original protocol, even though each validation stage passed
its own strict checks. The six models are historical internal joint models, not
six repetitions of the final short-no-roll model.

## Research consequence

Do not start another training sweep or repeat fixed channel normalization, which
already failed development evaluation. A defensible next design must address both
channel-scale comparability and calibration transfer, and explain how it differs
from the previously failed normalization/adaptive-score experiments. These results
narrow the problem; they do not establish a new effective detector. Preserve the
fixed system-level evaluation and untouched historical results.

## Verification and provenance

`continue_channel_error_validation.py` performs only the missing validation and
training stages after the calibration failure. It checks the original input and
checkpoint hashes, score tolerance, timestamps and exact saved validation mask;
input/annotation hashes are checked again afterward. All 17 completed stages were
independently rehashed before summary export. Source rows, monthly summaries and
stage verification metadata are in `tables/channel-error-validation-2026-10-17/`.
The original partial report remains historical evidence. See the protocol addendum
for the explicit scope change; it does not override the original acceptance gate.

The full thesis suite passed: 124 tests in 13.065 seconds.
