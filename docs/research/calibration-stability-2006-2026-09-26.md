# Frozen baseline calibration audit — 26 September 2026

Read-only review of saved epoch10 baseline scores for2006. No2007–2013
assessment files read; no model trained or frozen rule changed. Saved predictions
and corrected event metrics reproduced within1e-9. Input artifacts were hashed
before and after this audit and remained unchanged. This is cached-score
verification, not an independent inference or proof the intermittent fault is fixed.

## Findings

The baseline detects all6 calibration events annotated on inputs40–47.
Four missed system events (id_105,id_108,id_111,id_113) are annotated only on
other channels. This does not prove they are physically unobservable in40–47,
but they produce low scores and lowering the threshold in this audit does not
recover them. System calibration recall is6/10; the input-channel event count
is only6, limiting evidence for generalization.

| Threshold relative to frozen value | System F0.5 | TP | FP | FN |
|---|---:|---:|---:|---:|
| 50% | 0.282136 | 6 | 18 | 4 |
| 75% | 0.555037 | 6 | 5 | 4 |
| 90% | 0.713987 | 6 | 2 | 4 |
| 100% (frozen) | 0.882148 | 6 | 0 | 4 |
| 110% | 0.882217 | 6 | 0 | 4 |
| 125% | 0.555540 | 2 | 0 | 8 |
| 150% | 0.357142 | 1 | 0 | 9 |
| 200% | 0 | 0 | 0 | 10 |

The tiny improvement at110% comes from reduced nominal alarm duration, not
additional event detection. No candidate from this diagnostic is adopted.
A25% increase loses4 of6 detected events; a10% reduction adds2 false events.

Quarterly frozen system counts TP/FP/FN: Q1 0/0/1; Q2 2/0/1; Q3 1/0/0;
Q4 3/0/2. Q1 contains no input-channel positive event, so its zero system score
must not be interpreted as failure to detect an in-scope anomaly. Quarterly
scores are descriptive; sparse events and boundary clipping prevent treating
these slices as independent robust validation estimates.

Checkpoint calibration F0.5 peaks at epoch10 (0.882148) and declines to0.357143
at epoch40 while training loss continues decreasing. This supports selecting
checkpoints on event metrics rather than reconstruction loss; it does not
identify whether overfitting, score ranking or threshold selection causes the decline.

## Decision and next focused step

Keep the frozen baseline. Do not launch another architecture or lower thresholds
based on assessment outcomes. Before new training, use existing2003–2005
training/calibration artifacts to check how many local events each calibration
period contains, their durations and per-channel score patterns. The next
improvement must target a reproducible failure in those development periods,
then use a predeclared comparison. Retain both local and system reporting.
No claim that system F0.5 >=0.85 has been robustly achieved.

Reproduce: `PYTHONPATH=. python scripts/research/calibration_stability_2006.py`
(output directory must not already exist; previous artifacts are preserved).
[Audit tables](tables/calibration-stability-2006-2026-09-26/).
