# Masked pilot follow-up: improvement is not consistent

Completed2004/2005matched fiveepoch comparisons, same25,000window budget,
channels40–47 and seed42 as2003. Each fold trains only on preceding years,
calibrates January–March and assesses April–December. Both arms selected epoch1
in both new folds. All histories contain epochs1–5. Cached scores/predictions match
frozen rules exactly; independently recomputed local/system metrics agree within1e-12.

| Development year | Baseline local F0.5 | Masked local F0.5 | Baseline system F0.5 | Masked system F0.5 |
| --- | ---: | ---: | ---: | ---: |
| 2003 | 0.714286 | 0.908853 | 0.714286 | 0.908853 |
| 2004 | 0.640295 | 0.091652 | 0.594608 | 0.090409 |
| 2005 | 0.869116 | 0.869085 | 0.869116 | 0.869085 |

2004system: baseline TP5 FP3 FN5, masked TP4 FP48 FN6. Local scope has7events;
system scope10. Masked nominal false-alarm duration is326812.461s (~90.78h),
versus29790s (~8.28h) baseline. It detects fewer events and alarms more often.
The masked calibration score was already poor:0.171616 versus baseline1.0.
This does not isolate a cause; do not attribute it conclusively to masking,
regime drift or threshold instability without training/calibration diagnosis.

2005both scopes: both TP4 FP0 FN3. Masked nominal false-alarm duration is960s
longer (15224.184vs14264.184), explaining its tiny score reduction. These are
essentially tied, with no evidence of improvement.

## Decision

Do not promote this masked configuration, expand it to40epochs or evaluate it on
2007–2013 now. Retain the original long baseline (system0.805722/local0.862907).
The2003result alone was misleading about consistency. Single-seed, short-budget
pilots cannot rule out all masked models, but this specific approach is not robust.
No target achievement claim. No pooled F0.5 obtained by averaging annual values.

Next bounded investigation: inspect existing2004training/calibration histories,
threshold candidates and pseudo-anomaly/masking interaction before proposing another
run. New method choices must not fit the48assessment false alarms directly. Do not
mix baseline/masked by choosing the best model after seeing each assessment year.
Keep input set40–47, unchanged corrected evaluator and both reporting scopes.

Evidence: tables/masked-pilot-followup-2026-09-24/, including combined three-year
rows, copied histories/rules and original follow-up protocol. No new training,
threshold selection or assessment inference was performed during this review.
