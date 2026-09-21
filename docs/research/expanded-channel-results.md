# Expanded-channel AE results — 21 September 2026

All three folds completed. Frozen decision rules reproduce the saved masks; corrected
ESA F0.5, precision, recall, and TP/FP/FN counts were independently recomputed from
those masks and raw annotations, matching saved results within 1e-12. No rules or
weights were changed during this verification.

| Fold | Baseline F0.5 | Expanded F0.5 | Baseline detected / total | Expanded detected / total | Baseline FP | Expanded FP |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 2003 | 0.833170 | 0.909069 | 5 / 6 | 4 / 6 | 1 | 0 |
| 2004 | 0.428131 | 0.251365 | 6 / 10 | 9 / 10 | 9 | 33 |
| 2005 | 0.869055 | 0.148245 | 4 / 7 | 5 / 7 | 0 | 35 |

The expanded model adds channels 14, 21, and 29 to channels 40–47. It recovers
`id_92`, `id_94`, and `id_97`, the three targeted 2004 events. Its remaining missed
2004 ID is `id_91`. Other expanded misses are `id_12` and `id_13` in 2003, and
`id_100` and `id_16` in 2005.

This confirms that added channel coverage can recover missed events, but does not
establish a better detector. Expanded corrected precision is 0.999967, 0.212989,
and 0.123732 across the folds. The improved 2003 F0.5 comes with lower recall and
zero wholly false alarm events; it must not be presented as a robust 0.85 result.
The 2005 calibration score of 0.832880 did not transfer to assessment (0.148245).

Keep the eight-channel AE as the working baseline. Do not select a different model
for each fold retrospectively. A next development hypothesis is training-only
per-channel residual normalization, to test whether heterogeneous error scales
contribute to the maximum-across-channels false alarms. This cause has not yet
been established; adding inputs also changes learned weights and attention.
No further training is required merely to archive or interpret this experiment.

The original 2003/2004 runs completed before disconnection. The partial 2005 run
stopped after epoch 31 and had no optimizer state; it was preserved under
`interrupted/` and retrained from scratch with the same declared settings. The
completed 2005 run selected epoch 5; 2003 and 2004 selected epoch 40. Full epoch
recovery checkpoints are available for the restarted 2005 run.

Results: `results_longrun/development/ae_expanded_seed42/results.csv` (local release
artifact, excluded from Git). The chronological folds remain development data,
channel selection was informed by earlier development errors, and only seed 42
was tested. The benchmark test CSV was not used.
