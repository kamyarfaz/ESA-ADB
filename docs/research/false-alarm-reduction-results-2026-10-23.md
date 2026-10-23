# Calibration-constrained false-alarm reduction

This study reassessed two fixed seed42 models trained on2000–2005, using2006 only for alarm research and selection, then evaluating2007–2013. No neural weights were retrained. Both original and refined outcomes are published. Assessment data were previously inspected, so this is exploratory benchmark reuse, not independent validation.

Neither refinement improved assessment F0.5. Pretraining retained its original rule; the expanded-input refinement reduced detection and sharply increased false alarm events. Original results remain the better reported outcomes.

## Calibration research and frozen decisions

The pretrain/fine-tune model had one wholly false calibration alarm interval (960 seconds), but10379.778 seconds of nominal alarm duration overall. About90.75% of nominal alarm duration came from intervals that also touched annotations. The expanded model had no wholly false calibration alarm intervals, but15790.695 seconds of nominal alarm duration. Channel21 dominated349 of524 nominal alarm samples for the expanded model. Sample counts are a diagnostic, not a replacement for duration-based FPR. Long window-mean/overlap alarms can extend beyond short annotated events.

The prespecified study tested67 candidates per method: raw maximum scores and per-channel nominal2006 q99 normalization, a bounded threshold grid and minimum-duration filtering. Feasibility required retaining the exact original detected event IDs in both calibration scopes, no increase in nominal alarm duration or false alarm event count, and total alarm fraction<=1%. Selection used corrected system F0.5, then nominal alarm duration and fixed tie rules. Both rules were frozen before assessment. Normalized-score candidates did not satisfy the full feasibility constraints for either method.

Among normalized candidates retaining the original calibration IDs, the best system-F0.5 candidate for pretrain/fine-tune produced37870.695 nominal alarm seconds and2 false alarm events, compared with10379.778 seconds and1 for its original rule. For expanded inputs, the corresponding normalized candidate produced18059.778 nominal alarm seconds, exceeding the original15790.695. Normalization therefore failed the declared alarm constraints; it was not rejected based on assessment outcomes.

Pretrain/fine-tune retained its original threshold6.202751159667969: no tested change improved the objective while satisfying the constraints. Expanded inputs selected threshold6.900436115264893, a10% increase from6.273123741149902, with no gap filling and minimum duration1. This retained all9 detected system calibration IDs (and6 local IDs) while reducing nominal alarm duration from15790.695 to12910.695 seconds (18.24%). The small calibration event population does not guarantee later-year preservation.

## Pooled assessment results

| Method | Rule | Scope | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---|---:|---:|---:|---:|---:|
| Masked pretraining + fine-tuning | original | system_all_channels | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| Masked pretraining + fine-tuning | original | input_channels_40_47 | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| Masked pretraining + fine-tuning | refined | system_all_channels | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| Masked pretraining + fine-tuning | refined | input_channels_40_47 | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| Short-no-roll + extra inputs | original | system_all_channels | 76/91 | 0.821758 | 0.835165 | 5.930307 | 0.824405 |
| Short-no-roll + extra inputs | original | input_channels_40_47 | 61/67 | 0.796979 | 0.910448 | 5.930307 | 0.817353 |
| Short-no-roll + extra inputs | refined | system_all_channels | 73/91 | 0.480863 | 0.802198 | 5.803610 | 0.522741 |
| Short-no-roll + extra inputs | refined | input_channels_40_47 | 60/67 | 0.434753 | 0.895522 | 5.803610 | 0.484623 |

## Event retention and interpretation

- Masked pretraining + fine-tuning, system_all_channels: F0.5 0.795675 → 0.795675; events 61 → 61. Lost IDs: none. Gained IDs: none.
- Masked pretraining + fine-tuning, input_channels_40_47: F0.5 0.844315 → 0.844315; events 60 → 60. Lost IDs: none. Gained IDs: none.
- Short-no-roll + extra inputs, system_all_channels: F0.5 0.824405 → 0.522741; events 76 → 73. Lost IDs: id_115, id_135, id_51. Gained IDs: none.
- Short-no-roll + extra inputs, input_channels_40_47: F0.5 0.817353 → 0.484623; events 61 → 60. Lost IDs: id_51. Gained IDs: none.

The unchanged pretrain/fine-tune result is a valid negative calibration finding, not a new training result. Any assessment recall loss from the expanded refinement must remain visible despite its calibration retention. No additional thresholds or models were tried after inspecting these outcomes. The short-no-roll reference remains0.836070 system F0.5 and0.907140 for40–47, with48/91 and47/67 detected events respectively. Full event-level comparisons against it are in event_comparison.csv.

Precision is corrected ESA event precision, including its nominal-time penalty. Event TPR is detected target IDs divided by all target IDs. Time FPR is false-alarm duration divided by nominal duration, with other annotated intervals neutral. Smaller40–47 event scope is not an all-system result. Extra-input model still consumes14,21,29 as well as40–47. The study preserves existing offline window overlap and duration processing; no real-time latency claim is made.

The higher expanded-input threshold split 22 original alarm intervals. Total intervals increased from 115 to 211 and false alarm events from 11 to 70: 47 refined false intervals originated in annotation-touching parent intervals, and 23 in nominal parents. This explains why slightly less false-alarm duration nevertheless produced worse event precision. This post-assessment explanation did not feed back into selection; no gap-filling or further tuning was performed. See fragmentation.json.

## Annual and pooled metrics

### system_all_channels

| Method | Rule | Period | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---|---:|---:|---:|---:|---:|
| Masked pretraining + fine-tuning | original | 2007 | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| Masked pretraining + fine-tuning | original | 2008 | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| Masked pretraining + fine-tuning | original | 2009 | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| Masked pretraining + fine-tuning | original | 2010 | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| Masked pretraining + fine-tuning | original | 2011 | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| Masked pretraining + fine-tuning | original | 2012 | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| Masked pretraining + fine-tuning | original | 2013 | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| Masked pretraining + fine-tuning | original | 2007-2013 | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| Masked pretraining + fine-tuning | refined | 2007 | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| Masked pretraining + fine-tuning | refined | 2008 | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| Masked pretraining + fine-tuning | refined | 2009 | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| Masked pretraining + fine-tuning | refined | 2010 | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| Masked pretraining + fine-tuning | refined | 2011 | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| Masked pretraining + fine-tuning | refined | 2012 | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| Masked pretraining + fine-tuning | refined | 2013 | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| Masked pretraining + fine-tuning | refined | 2007-2013 | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| Short-no-roll + extra inputs | original | 2007 | 16/16 | 0.939796 | 1.000000 | 0.146683 | 0.951250 |
| Short-no-roll + extra inputs | original | 2008 | 10/13 | 0.999115 | 0.769231 | 0.088480 | 0.942766 |
| Short-no-roll + extra inputs | original | 2009 | 11/12 | 0.915634 | 0.916667 | 0.112677 | 0.915840 |
| Short-no-roll + extra inputs | original | 2010 | 9/12 | 0.817253 | 0.750000 | 0.113506 | 0.802855 |
| Short-no-roll + extra inputs | original | 2011 | 11/15 | 0.915871 | 0.733333 | 0.086796 | 0.872438 |
| Short-no-roll + extra inputs | original | 2012 | 5/10 | 0.555177 | 0.500000 | 0.068217 | 0.543188 |
| Short-no-roll + extra inputs | original | 2013 | 14/14 | 0.504152 | 1.000000 | 42.382655 | 0.559652 |
| Short-no-roll + extra inputs | original | 2007-2013 | 76/91 | 0.821758 | 0.835165 | 5.930307 | 0.824405 |
| Short-no-roll + extra inputs | refined | 2007 | 15/16 | 0.999053 | 0.937500 | 0.094685 | 0.986104 |
| Short-no-roll + extra inputs | refined | 2008 | 10/13 | 0.999362 | 0.769231 | 0.063780 | 0.942942 |
| Short-no-roll + extra inputs | refined | 2009 | 10/12 | 0.999139 | 0.833333 | 0.086101 | 0.960901 |
| Short-no-roll + extra inputs | refined | 2010 | 9/12 | 0.899189 | 0.750000 | 0.090121 | 0.864785 |
| Short-no-roll + extra inputs | refined | 2011 | 10/15 | 0.999306 | 0.666667 | 0.069426 | 0.908632 |
| Short-no-roll + extra inputs | refined | 2012 | 5/10 | 0.624745 | 0.500000 | 0.040784 | 0.595053 |
| Short-no-roll + extra inputs | refined | 2013 | 14/14 | 0.102127 | 1.000000 | 41.641758 | 0.124480 |
| Short-no-roll + extra inputs | refined | 2007-2013 | 73/91 | 0.480863 | 0.802198 | 5.803610 | 0.522741 |
### input_channels_40_47

| Method | Rule | Period | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---|---:|---:|---:|---:|---:|
| Masked pretraining + fine-tuning | original | 2007 | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| Masked pretraining + fine-tuning | original | 2008 | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| Masked pretraining + fine-tuning | original | 2009 | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| Masked pretraining + fine-tuning | original | 2010 | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| Masked pretraining + fine-tuning | original | 2011 | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| Masked pretraining + fine-tuning | original | 2012 | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| Masked pretraining + fine-tuning | original | 2013 | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| Masked pretraining + fine-tuning | original | 2007-2013 | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| Masked pretraining + fine-tuning | refined | 2007 | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| Masked pretraining + fine-tuning | refined | 2008 | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| Masked pretraining + fine-tuning | refined | 2009 | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| Masked pretraining + fine-tuning | refined | 2010 | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| Masked pretraining + fine-tuning | refined | 2011 | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| Masked pretraining + fine-tuning | refined | 2012 | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| Masked pretraining + fine-tuning | refined | 2013 | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| Masked pretraining + fine-tuning | refined | 2007-2013 | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| Short-no-roll + extra inputs | original | 2007 | 13/13 | 0.927209 | 1.000000 | 0.146683 | 0.940907 |
| Short-no-roll + extra inputs | original | 2008 | 7/9 | 0.999115 | 0.777778 | 0.088480 | 0.945312 |
| Short-no-roll + extra inputs | original | 2009 | 10/10 | 0.908067 | 1.000000 | 0.112677 | 0.925076 |
| Short-no-roll + extra inputs | original | 2010 | 8/8 | 0.799092 | 1.000000 | 0.113506 | 0.832545 |
| Short-no-roll + extra inputs | original | 2011 | 10/11 | 0.908302 | 0.909091 | 0.086796 | 0.908460 |
| Short-no-roll + extra inputs | original | 2012 | 4/8 | 0.499659 | 0.500000 | 0.068217 | 0.499727 |
| Short-no-roll + extra inputs | original | 2013 | 9/9 | 0.471415 | 1.000000 | 42.382655 | 0.527143 |
| Short-no-roll + extra inputs | original | 2007-2013 | 61/67 | 0.796979 | 0.910448 | 5.930307 | 0.817353 |
| Short-no-roll + extra inputs | refined | 2007 | 13/13 | 0.999053 | 1.000000 | 0.094685 | 0.999242 |
| Short-no-roll + extra inputs | refined | 2008 | 7/9 | 0.999362 | 0.777778 | 0.063780 | 0.945489 |
| Short-no-roll + extra inputs | refined | 2009 | 10/10 | 0.999139 | 1.000000 | 0.086101 | 0.999311 |
| Short-no-roll + extra inputs | refined | 2010 | 8/8 | 0.888088 | 1.000000 | 0.090121 | 0.908420 |
| Short-no-roll + extra inputs | refined | 2011 | 9/11 | 0.999306 | 0.818182 | 0.069426 | 0.956938 |
| Short-no-roll + extra inputs | refined | 2012 | 4/8 | 0.571196 | 0.500000 | 0.040784 | 0.555379 |
| Short-no-roll + extra inputs | refined | 2013 | 9/9 | 0.070030 | 1.000000 | 41.641758 | 0.086031 |
| Short-no-roll + extra inputs | refined | 2007-2013 | 60/67 | 0.434753 | 0.895522 | 5.803610 | 0.484623 |

## Verification and sources

All64 metric rows were independently recomputed from raw annotations. Exact original aggregate-score and binary-prediction parity passed after fresh per-channel inference. Artifact, source, frozen-data and decision hashes passed, as did calibration event-ID constraints and frozen candidate selection. The140-test thesis suite includes inference-interruption replay and corrected evaluator parity. Original results and unsuccessful candidates are preserved.

[Full precision metrics](tables/false-alarm-reduction-2026-10-23/verified_results.csv), [event IDs](tables/false-alarm-reduction-2026-10-23/event_comparison.csv), [verification](tables/false-alarm-reduction-2026-10-23/verification.json), [frozen protocol and primary methodological sources](false-alarm-reduction-protocol-2026-10-23.md).

[Essential artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/false-alarm-reduction-2026-10-23). Download checksum evidence is recorded in tables/false-alarm-reduction-2026-10-23/github-backup.json. This additive backup does not certify a whole-project migration or replace an external-drive snapshot.
