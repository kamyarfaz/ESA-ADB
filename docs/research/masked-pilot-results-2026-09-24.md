# Masked pilot completed — 24 September 2026

Run: results_longrun/development/masked_pilot_2003_seed42_v1.
Both arms: channels40–47, same25,000normal training windows from2000–2002,
fiveepochs, seed42; Jan–Mar2003calibration, Apr–Dec2003development assessment.
Both selected epoch1 by the prespecified earliest calibration tie rule.
Both calibration scores are0.714286. Neither arm was selected on assessment.

| Arm | Input-channel F0.5 | System F0.5 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched baseline | 0.714286 | 0.714286 | 2 | 0 | 4 |
| Masked reconstruction | 0.908853 | 0.908853 | 4 | 0 | 2 |

All six event IDs have annotations on input channels, so this fold does not
separate local versus system generalization. Both raw-scope evaluations reproduce
the same values on these predictions; this cannot establish detection strength on
unobserved channels. Future folds must continue reporting both scopes.

Baseline detects id_19/id_89; masked also detects id_12/id_90, without losing either
baseline hit. Both miss id_13/id_88. Masked has0false-positive EVENTS but7915.212s
(about2.20h) nominal alarm duration; baseline has0s. Corrected precision accounts
for that extra duration. A raw OR would not add detected IDs beyond masked on this
fold. This is a descriptive observation, not authorization to tune an OR on these
assessment results.

The gain is0.194567absolute F0.5 under this small matched protocol. It is NOT a
comparison with the longer reference0.805722, nor proof of robust0.85achievement.
The earlier40epoch2003baseline scored0.833170 with5TP/1FP; budgets and checkpoint
selection differ. Do not claim the new model dominates every historical baseline.
The fiveepoch pilot may undertrain either approach; both best checkpoints being
epoch1 reinforces the limits of a tiny calibration set and single seed.

## Verification and next decision

Verified both histories contain epochs1–5; frozen rules choose epoch1; cached
predictions exactly equal postprocessed saved scores under frozen rules; recomputed
system F0.5 agrees within1e-12. Local scope uses input-channel intervals of selected
IDs; out-of-scope annotations remain neutral. Source hashes and compact tables
are saved under tables/masked-pilot-2003-2026-09-24/.

Next justified step is matching this SAME small design on2004and2005chronological
folds. Do not change masking, window budget, seed, epoch count or threshold grid
based on2003assessment. This requires fold support before issuing a new command;
no additional run was launched here. If gains persist, evaluate seed variability
and compute costs before a larger training run. Benchmark2007–2013remains unrerun.
Masked inference takes four forward passes; equal epochs are not equal FLOPs.
