# Full-period short-roll removal assessment

The completed candidate achieves pooled corrected system event F0.5 **0.836070**,
compared with **0.805722** for the historical baseline (absolute +0.030348).
This remains below 0.85. Improvements are not uniform across years.

| Test period | System F0.5, all-channel events | Input-channel event F0.5 |
|---|---:|---:|
| 2007 | 0.916357 | 0.964569 |
| 2008 | 0.853372 | 0.945594 |
| 2009 | 0.874739 | 0.920763 |
| 2010 | 0.624940 | 0.749914 |
| 2011 | 0.850982 | 0.930135 |
| 2012 | 0.666620 | 0.624956 |
| 2013 | 0.869422 | 0.975430 |
| **2007–2013 pooled** | **0.836070** | **0.907140** |

## Interpretation

| Pooled system metric | Historical baseline | Short-no-roll |
|---|---:|---:|
| Corrected event precision | 0.884871 | 0.979306 |
| TPR / event recall | 0.593407 | 0.527473 |
| FPR / nominal false-alarm time fraction | 0.042312% | 0.029194% |
| Corrected event F0.5 | 0.805722 | 0.836070 |

The candidate produces fewer false alarms but detects fewer annotated event IDs
(48/91 versus 54/91). F0.5 favors precision, so improved precision outweighs the
recall reduction in the pooled score. This is a trade-off, not a general improvement
in anomaly coverage. The lower local event denominator explains why input-channel
F0.5 is higher; it does not establish all-system performance above the thesis target.

The intended training mechanism is to avoid temporal displacement in the short
pseudo-anomaly branch, retaining short offset/noise perturbations. The legacy
augmentation branch still contains rolls. The historical baseline differs in other
augmentation details, so this comparison cannot isolate a causal benefit from
short-roll removal alone. No additional control experiment is claimed.

## Fixed protocol and verification

Both methods use training 2000–2005, calibration 2006, continuous testing
2007–2013, inputs 40–47, 40 epochs, all eligible nominal training windows,
batch 64, and one run with seed 42. System evaluation covers events from all
76 channels. Training uses nominal windows plus artificial anomaly augmentation.

The candidate selected epoch 10 using calibration alone (F0.5 0.88213824696),
threshold 4.275609493255615, merge gap 0, minimum duration 1. The rule was frozen
before test scoring. Annual results slice the same continuous predictions; pooled
results are rescored over all seven years, not averaged across years.

`scripts/research/verify_full_short_no_roll.py` independently checked saved artifact,
source and frozen-data hashes, 40-epoch history, earliest best calibration checkpoint,
all 7,364,160 predictions against the frozen threshold, and all 16 annual/pooled
metric rows directly from raw annotations using the corrected evaluator. Seven
evaluator parity/regression tests passed. Verification and full precision metrics:
`tables/full-short-no-roll-2026-10-16/{verification.json,verified_results.csv}`.

The original run was interrupted and resumed from epoch 2 under the persistent
service without changing the frozen protocol. This is one resumed run, not a new
seed or independent replicate. These previously inspected benchmark years are
reused assessment, not independent validation. No test-based tuning was performed.

The thesis workbook is
`outputs/thesis-system-20261016/deliverables/ESA_thesis_results.xlsx`.
It contains verified system-only precision, event TPR, duration-based FPR and F0.5
for both methods and all years. FPR is not a pointwise or event-count FPR.
