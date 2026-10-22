# Frozen three-combination comparison

All three combinations completed. None improves pooled corrected all-channel event F0.5 over the short-no-roll reference (0.836070). These are descriptive results on a reused benchmark with one seed, not independent validation or evidence that the thesis target has been robustly achieved.

## Matched periods and deliberate differences

Every method uses training 2000–2005, calibration 2006, and continuous assessment 2007–2013. Inputs are channels 40–47, except approach 3, which also uses channels 14, 21 and 29 with explicit authorization. The 40–47 event scope remains identical for every method, including approach 3. There is no real training anomaly supervision: training uses nominal windows, with artificial anomalies where specified. MOSAD remains stopped and has no completed assessment.

Approach 1 normalizes short-no-roll and masked scores using nominal 2006 quantiles, then combines them with calibration-selected masked weight 0.5. It uses two 40-epoch components (80 model-epochs including the reused reference) and five scoring passes. Approach 2 uses 20 epochs of nominal masked pretraining followed by 20 epochs of unmasked short-no-roll fine-tuning. Approach 3 uses 40 epochs of short-no-roll with extra inputs. The masked standalone control uses 40 epochs with its legacy synthetic objective. This control is reported separately below; the workbook retains the original historical masked rows unchanged.

All arms use seed 42, batch 64, length 256, training stride 16, and all eligible nominal training windows. Calibration selects checkpoints and alarm rules under a 1% total-alarm cap. Selected epochs: Masked standalone control 10, Approach 2: masked pretraining + short-no-roll 25, Approach 3: short-no-roll + extra inputs 15. The existing reference selected epoch 10 and retains its original shuffle trajectory. No checkpoint, threshold or combination was selected using assessment results.

## Pooled comparison

Precision is corrected ESA event precision; TPR is detected target event IDs divided by all target IDs. Time FPR is false-alarm duration divided by nominal duration. F0.5 is the corrected event score. Values below are fractions except explicitly marked percentages. Out-of-scope annotated intervals remain neutral for false alarms. Pooled metrics are recomputed over the full series, not averaged over annual scores.

| Method | System F0.5 | 40–47 F0.5 | System precision | System TPR | Time FPR (%) | System events |
|---|---:|---:|---:|---:|---:|---:|
| Short-no-roll reference | 0.836070 | 0.907140 | 0.979306 | 0.527473 | 0.029194 | 48/91 |
| Masked standalone control | 0.806278 | 0.882146 | 0.957142 | 0.494505 | 0.031883 | 45/91 |
| Approach 1: score fusion | 0.811670 | 0.884610 | 0.977528 | 0.483516 | 0.025544 | 44/91 |
| Approach 2: masked pretraining + short-no-roll | 0.795675 | 0.844315 | 0.834696 | 0.670330 | 0.110185 | 61/91 |
| Approach 3: short-no-roll + extra inputs | 0.824405 | 0.817353 | 0.821758 | 0.835165 | 5.930307 | 76/91 |

## Interpretation

Fusion reduces nominal false-alarm time from 0.029194% to 0.025544%, but detects 44 rather than 48 system events. Its F0.5 falls to 0.811670. Masked pretraining and fine-tuning detects 61 events, but precision falls to 0.834696 and F0.5 to 0.795675. Extra inputs detect 76 events, the highest coverage, but nominal false-alarm time reaches 5.930307% and F0.5 is 0.824405. The masked standalone control reaches 0.806278. Better coverage and lower false-alarm duration are different objectives; none of these changes improves the prespecified pooled system F0.5.

These observations do not isolate causal effects of pretraining, augmentation, or additional channels. Architectures/objectives, information and ensemble computation differ as declared. Historical high scores from other periods cannot establish superiority under this protocol. The 40–47 score has a smaller target-event population and must not be presented as all-system performance.

## Annual and pooled metrics for both scopes

Each annual row slices the same continuous predictions. An event crossing a year boundary can appear in more than one annual slice, so annual event counts need not sum to pooled unique IDs. These tables report every method and year, including unsuccessful results.

### All-channel system events

| Method | Period | Events detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| Short-no-roll reference | 2007 | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| Short-no-roll reference | 2008 | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| Short-no-roll reference | 2009 | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| Short-no-roll reference | 2010 | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| Short-no-roll reference | 2011 | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| Short-no-roll reference | 2012 | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| Short-no-roll reference | 2013 | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| Short-no-roll reference | 2007-2013 | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |
| Masked standalone control | 2007 | 13/16 | 0.999525 | 0.812500 | 0.047507 | 0.955535 |
| Masked standalone control | 2008 | 7/13 | 0.999420 | 0.538462 | 0.057953 | 0.853321 |
| Masked standalone control | 2009 | 6/12 | 0.999591 | 0.500000 | 0.040939 | 0.833106 |
| Masked standalone control | 2010 | 3/12 | 0.749855 | 0.250000 | 0.019384 | 0.535655 |
| Masked standalone control | 2011 | 5/15 | 0.999844 | 0.333333 | 0.015600 | 0.714222 |
| Masked standalone control | 2012 | 3/10 | 0.749837 | 0.300000 | 0.021685 | 0.576846 |
| Masked standalone control | 2013 | 8/14 | 0.999804 | 0.571429 | 0.019626 | 0.869446 |
| Masked standalone control | 2007-2013 | 45/91 | 0.957142 | 0.494505 | 0.031883 | 0.806278 |
| Approach 1: score fusion | 2007 | 12/16 | 0.999588 | 0.750000 | 0.041242 | 0.937210 |
| Approach 1: score fusion | 2008 | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| Approach 1: score fusion | 2009 | 6/12 | 0.999627 | 0.500000 | 0.037290 | 0.833126 |
| Approach 1: score fusion | 2010 | 2/12 | 0.999839 | 0.166667 | 0.016077 | 0.499968 |
| Approach 1: score fusion | 2011 | 4/15 | 0.999913 | 0.266667 | 0.008739 | 0.645132 |
| Approach 1: score fusion | 2012 | 5/10 | 0.833246 | 0.500000 | 0.010449 | 0.735240 |
| Approach 1: score fusion | 2013 | 8/14 | 0.999839 | 0.571429 | 0.016080 | 0.869468 |
| Approach 1: score fusion | 2007-2013 | 44/91 | 0.977528 | 0.483516 | 0.025544 | 0.811670 |
| Approach 2: masked pretraining + short-no-roll | 2007 | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| Approach 2: masked pretraining + short-no-roll | 2008 | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| Approach 2: masked pretraining + short-no-roll | 2009 | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| Approach 2: masked pretraining + short-no-roll | 2010 | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| Approach 2: masked pretraining + short-no-roll | 2011 | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| Approach 2: masked pretraining + short-no-roll | 2012 | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| Approach 2: masked pretraining + short-no-roll | 2013 | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| Approach 2: masked pretraining + short-no-roll | 2007-2013 | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| Approach 3: short-no-roll + extra inputs | 2007 | 16/16 | 0.939796 | 1.000000 | 0.146683 | 0.951250 |
| Approach 3: short-no-roll + extra inputs | 2008 | 10/13 | 0.999115 | 0.769231 | 0.088480 | 0.942766 |
| Approach 3: short-no-roll + extra inputs | 2009 | 11/12 | 0.915634 | 0.916667 | 0.112677 | 0.915840 |
| Approach 3: short-no-roll + extra inputs | 2010 | 9/12 | 0.817253 | 0.750000 | 0.113506 | 0.802855 |
| Approach 3: short-no-roll + extra inputs | 2011 | 11/15 | 0.915871 | 0.733333 | 0.086796 | 0.872438 |
| Approach 3: short-no-roll + extra inputs | 2012 | 5/10 | 0.555177 | 0.500000 | 0.068217 | 0.543188 |
| Approach 3: short-no-roll + extra inputs | 2013 | 14/14 | 0.504152 | 1.000000 | 42.382655 | 0.559652 |
| Approach 3: short-no-roll + extra inputs | 2007-2013 | 76/91 | 0.821758 | 0.835165 | 5.930307 | 0.824405 |

### Events on channels 40–47

| Method | Period | Events detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| Short-no-roll reference | 2007 | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| Short-no-roll reference | 2008 | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| Short-no-roll reference | 2009 | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| Short-no-roll reference | 2010 | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| Short-no-roll reference | 2011 | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| Short-no-roll reference | 2012 | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| Short-no-roll reference | 2013 | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| Short-no-roll reference | 2007-2013 | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |
| Masked standalone control | 2007 | 13/13 | 0.999525 | 1.000000 | 0.047507 | 0.999620 |
| Masked standalone control | 2008 | 7/9 | 0.999420 | 0.777778 | 0.057953 | 0.945531 |
| Masked standalone control | 2009 | 6/10 | 0.999591 | 0.600000 | 0.040939 | 0.882098 |
| Masked standalone control | 2010 | 3/8 | 0.749855 | 0.375000 | 0.019384 | 0.624919 |
| Masked standalone control | 2011 | 5/11 | 0.999844 | 0.454545 | 0.015600 | 0.806370 |
| Masked standalone control | 2012 | 3/8 | 0.749837 | 0.375000 | 0.021685 | 0.624910 |
| Masked standalone control | 2013 | 8/9 | 0.999804 | 0.888889 | 0.019626 | 0.975460 |
| Masked standalone control | 2007-2013 | 45/67 | 0.957142 | 0.671642 | 0.031883 | 0.882146 |
| Approach 1: score fusion | 2007 | 12/13 | 0.999588 | 0.923077 | 0.041242 | 0.983287 |
| Approach 1: score fusion | 2008 | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| Approach 1: score fusion | 2009 | 6/10 | 0.999627 | 0.600000 | 0.037290 | 0.882121 |
| Approach 1: score fusion | 2010 | 2/8 | 0.999839 | 0.250000 | 0.016077 | 0.624950 |
| Approach 1: score fusion | 2011 | 4/11 | 0.999913 | 0.363636 | 0.008739 | 0.740702 |
| Approach 1: score fusion | 2012 | 4/8 | 0.799916 | 0.500000 | 0.010449 | 0.714232 |
| Approach 1: score fusion | 2013 | 8/9 | 0.999839 | 0.888889 | 0.016080 | 0.975487 |
| Approach 1: score fusion | 2007-2013 | 43/67 | 0.977023 | 0.641791 | 0.025544 | 0.884610 |
| Approach 2: masked pretraining + short-no-roll | 2007 | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| Approach 2: masked pretraining + short-no-roll | 2008 | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| Approach 2: masked pretraining + short-no-roll | 2009 | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| Approach 2: masked pretraining + short-no-roll | 2010 | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| Approach 2: masked pretraining + short-no-roll | 2011 | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| Approach 2: masked pretraining + short-no-roll | 2012 | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| Approach 2: masked pretraining + short-no-roll | 2013 | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| Approach 2: masked pretraining + short-no-roll | 2007-2013 | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| Approach 3: short-no-roll + extra inputs | 2007 | 13/13 | 0.927209 | 1.000000 | 0.146683 | 0.940907 |
| Approach 3: short-no-roll + extra inputs | 2008 | 7/9 | 0.999115 | 0.777778 | 0.088480 | 0.945312 |
| Approach 3: short-no-roll + extra inputs | 2009 | 10/10 | 0.908067 | 1.000000 | 0.112677 | 0.925076 |
| Approach 3: short-no-roll + extra inputs | 2010 | 8/8 | 0.799092 | 1.000000 | 0.113506 | 0.832545 |
| Approach 3: short-no-roll + extra inputs | 2011 | 10/11 | 0.908302 | 0.909091 | 0.086796 | 0.908460 |
| Approach 3: short-no-roll + extra inputs | 2012 | 4/8 | 0.499659 | 0.500000 | 0.068217 | 0.499727 |
| Approach 3: short-no-roll + extra inputs | 2013 | 9/9 | 0.471415 | 1.000000 | 42.382655 | 0.527143 |
| Approach 3: short-no-roll + extra inputs | 2007-2013 | 61/67 | 0.796979 | 0.910448 | 5.930307 | 0.817353 |

## Verification and preservation

The independent verifier checked all recorded run artifacts, frozen data, package sources and reference hashes; complete finite 40-epoch histories; earliest best eligible calibration checkpoints; stored calibration-grid rule choices; fusion weight and nominal normalization; exact prediction replay from saved scores; control-reference identity; and fusion-score identity. All 80 metric rows were independently recomputed from raw annotations using the corrected evaluator. This is score-to-prediction replay, not a second neural inference run. The full thesis suite passed 136 tests, including evaluator parity and exact interrupted-training resume tests.

Full precision data and verification: [verified_results.csv](tables/combinations-2026-10-22/verified_results.csv), [verification.json](tables/combinations-2026-10-22/verification.json). Frozen design: [protocol](combinations-full-protocol-2026-10-21.md). The updated workbook preserves all 32 original technique rows and adds three combination rows. The original historical masked results remain visible, while its new full-period standalone control is reported in this report and CSV. MOSAD is marked stopped without inventing scores.

Essential artifacts and the original workbook are backed up in the [GitHub Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/combinations-full-2026-10-22). Download checksum evidence is recorded in [github-backup.json](tables/combinations-2026-10-22/github-backup.json). This additive backup does not replace the full external-drive snapshot or resolve the previously documented unrelated historical transfer exception. No whole-project destination-copy verification is claimed.
