# Three post-processing families on three techniques — 28 October 2026

All nine calibration-selected rules retained the unchanged fallback. No improvement is claimed. This study uses fixed W&B-selected weights and thresholds, training 2000–2005, calibration 2006, assessment 2007–2013. Inputs 40–47, plus 14,21,29 only for expanded short-no-roll. No new model training.

## What was tested

- Strength-duration: reject short segments only if peak score is also weak; 48 modifications and fallback per model.
- Hysteresis: connected low-threshold segments must contain a high-threshold sample; 15 modifications and fallback.
- Duration: cutoffs from 30 seconds to 256 minutes; eight settings and fallback.

The 222 candidates were evaluated on calibration only. Maximize corrected local40–47 F0.5 while retaining baseline-detected calibration IDs and <=1% total alarm fraction. Ties favor lower nominal false-alarm time, then original, then grid order. All nine choices froze together before assessment. Exact calibration retention is a conservative constraint; it does not guarantee later-year recall. Each family has its own fallback.

## Calibration evidence

| Model | Family | Candidates | Feasible | Rejected | Selected original | Best feasible F0.5 |
|---|---|---:|---:|---:|---|---:|
| short_no_roll | strength_duration | 49 | 19 | 30 | True | 0.999724246 |
| short_no_roll | hysteresis | 16 | 7 | 9 | True | 0.999724246 |
| short_no_roll | duration | 9 | 3 | 6 | True | 0.999724246 |
| pretrain_finetune | strength_duration | 49 | 13 | 36 | True | 0.882105852 |
| pretrain_finetune | hysteresis | 16 | 4 | 12 | True | 0.882105852 |
| pretrain_finetune | duration | 9 | 3 | 6 | True | 0.882105852 |
| expanded_short_no_roll | strength_duration | 49 | 19 | 30 | True | 0.999727433 |
| expanded_short_no_roll | hysteresis | 16 | 7 | 9 | True | 0.999727433 |
| expanded_short_no_roll | duration | 9 | 3 | 6 | True | 0.999727433 |

Unlike the earlier grid capped at 16 minutes, these rules could remove calibration alarms. However they did not improve the feasible calibration objective. Original short-no-roll already has local calibration F0.5 0.999724; expanded short-no-roll has 0.999727, leaving very little calibration headroom. Each detects six target calibration IDs.

For masked pretraining, a 16-minute false segment peaks at 1.051630 times its threshold; a 16-minute genuine segment detecting id_107 peaks at 1.051257 times threshold. A simple peak cutoff that removes the former also removes the latter when applied to those components; a duration-only cutoff above 16 minutes removes both. Other candidates, including hysteresis, were still evaluated; none improved the feasible objective. This illustrates overlapping score evidence, not proof that all possible post-processing is ineffective.

Expanded-model components without local IDs can overlap other annotated events, which remain neutral for local evaluation. They must not all be called false alarms. Calibration differs markedly from later years; these few calibration examples cannot establish generalization.

## Annual and pooled assessment

TPR is event-ID recall. Precision includes the corrected ESA nominal-time penalty. Time FPR is nominal false-alarm duration divided by nominal duration. Continuous predictions are sliced only for reporting; pooled values are recomputed. Each method has one shared original control and three selected family outputs (192 rows total).

### short_no_roll: system_all_channels

| Period | Rule | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| 2007 | strength_duration | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| 2007 | hysteresis | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| 2007 | duration | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| 2008 | original | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| 2008 | strength_duration | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| 2008 | hysteresis | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| 2008 | duration | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| 2009 | original | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| 2009 | strength_duration | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| 2009 | hysteresis | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| 2009 | duration | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| 2010 | original | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| 2010 | strength_duration | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| 2010 | hysteresis | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| 2010 | duration | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| 2011 | original | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| 2011 | strength_duration | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| 2011 | hysteresis | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| 2011 | duration | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| 2012 | original | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| 2012 | strength_duration | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| 2012 | hysteresis | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| 2012 | duration | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| 2013 | original | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| 2013 | strength_duration | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| 2013 | hysteresis | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| 2013 | duration | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| 2007-2013 | original | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |
| 2007-2013 | strength_duration | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |
| 2007-2013 | hysteresis | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |
| 2007-2013 | duration | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |

### short_no_roll: input_channels_40_47

| Period | Rule | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| 2007 | strength_duration | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| 2007 | hysteresis | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| 2007 | duration | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| 2008 | original | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| 2008 | strength_duration | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| 2008 | hysteresis | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| 2008 | duration | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| 2009 | original | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| 2009 | strength_duration | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| 2009 | hysteresis | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| 2009 | duration | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| 2010 | original | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| 2010 | strength_duration | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| 2010 | hysteresis | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| 2010 | duration | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| 2011 | original | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| 2011 | strength_duration | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| 2011 | hysteresis | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| 2011 | duration | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| 2012 | original | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| 2012 | strength_duration | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| 2012 | hysteresis | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| 2012 | duration | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| 2013 | original | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| 2013 | strength_duration | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| 2013 | hysteresis | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| 2013 | duration | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| 2007-2013 | original | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |
| 2007-2013 | strength_duration | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |
| 2007-2013 | hysteresis | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |
| 2007-2013 | duration | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |

### pretrain_finetune: system_all_channels

| Period | Rule | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| 2007 | strength_duration | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| 2007 | hysteresis | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| 2007 | duration | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| 2008 | original | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| 2008 | strength_duration | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| 2008 | hysteresis | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| 2008 | duration | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| 2009 | original | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| 2009 | strength_duration | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| 2009 | hysteresis | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| 2009 | duration | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| 2010 | original | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| 2010 | strength_duration | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| 2010 | hysteresis | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| 2010 | duration | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| 2011 | original | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| 2011 | strength_duration | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| 2011 | hysteresis | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| 2011 | duration | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| 2012 | original | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| 2012 | strength_duration | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| 2012 | hysteresis | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| 2012 | duration | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| 2013 | original | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| 2013 | strength_duration | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| 2013 | hysteresis | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| 2013 | duration | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| 2007-2013 | original | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| 2007-2013 | strength_duration | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| 2007-2013 | hysteresis | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| 2007-2013 | duration | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |

### pretrain_finetune: input_channels_40_47

| Period | Rule | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| 2007 | strength_duration | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| 2007 | hysteresis | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| 2007 | duration | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| 2008 | original | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| 2008 | strength_duration | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| 2008 | hysteresis | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| 2008 | duration | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| 2009 | original | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| 2009 | strength_duration | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| 2009 | hysteresis | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| 2009 | duration | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| 2010 | original | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| 2010 | strength_duration | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| 2010 | hysteresis | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| 2010 | duration | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| 2011 | original | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| 2011 | strength_duration | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| 2011 | hysteresis | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| 2011 | duration | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| 2012 | original | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| 2012 | strength_duration | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| 2012 | hysteresis | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| 2012 | duration | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| 2013 | original | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| 2013 | strength_duration | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| 2013 | hysteresis | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| 2013 | duration | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| 2007-2013 | original | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| 2007-2013 | strength_duration | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| 2007-2013 | hysteresis | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| 2007-2013 | duration | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |

### expanded_short_no_roll: system_all_channels

| Period | Rule | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519 |
| 2007 | strength_duration | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519 |
| 2007 | hysteresis | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519 |
| 2007 | duration | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519 |
| 2008 | original | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942782 |
| 2008 | strength_duration | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942782 |
| 2008 | hysteresis | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942782 |
| 2008 | duration | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942782 |
| 2009 | original | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099 |
| 2009 | strength_duration | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099 |
| 2009 | hysteresis | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099 |
| 2009 | duration | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099 |
| 2010 | original | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936993 |
| 2010 | strength_duration | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936993 |
| 2010 | hysteresis | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936993 |
| 2010 | duration | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936993 |
| 2011 | original | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932830 |
| 2011 | strength_duration | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932830 |
| 2011 | hysteresis | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932830 |
| 2011 | duration | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932830 |
| 2012 | original | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773 |
| 2012 | strength_duration | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773 |
| 2012 | hysteresis | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773 |
| 2012 | duration | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773 |
| 2013 | original | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | strength_duration | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | hysteresis | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | duration | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2007-2013 | original | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573 |
| 2007-2013 | strength_duration | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573 |
| 2007-2013 | hysteresis | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573 |
| 2007-2013 | duration | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573 |

### expanded_short_no_roll: input_channels_40_47

| Period | Rule | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848 |
| 2007 | strength_duration | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848 |
| 2007 | hysteresis | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848 |
| 2007 | duration | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848 |
| 2008 | original | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328 |
| 2008 | strength_duration | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328 |
| 2008 | hysteresis | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328 |
| 2008 | duration | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328 |
| 2009 | original | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716 |
| 2009 | strength_duration | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716 |
| 2009 | hysteresis | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716 |
| 2009 | duration | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716 |
| 2010 | original | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999423 |
| 2010 | strength_duration | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999423 |
| 2010 | hysteresis | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999423 |
| 2010 | duration | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999423 |
| 2011 | original | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692 |
| 2011 | strength_duration | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692 |
| 2011 | hysteresis | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692 |
| 2011 | duration | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692 |
| 2012 | original | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499811 |
| 2012 | strength_duration | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499811 |
| 2012 | hysteresis | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499811 |
| 2012 | duration | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499811 |
| 2013 | original | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | strength_duration | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | hysteresis | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | duration | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2007-2013 | original | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830900 |
| 2007-2013 | strength_duration | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830900 |
| 2007-2013 | hysteresis | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830900 |
| 2007-2013 | duration | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830900 |

## Event changes

- expanded_short_no_roll/duration/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_short_no_roll/duration/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_short_no_roll/hysteresis/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_short_no_roll/hysteresis/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_short_no_roll/strength_duration/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_short_no_roll/strength_duration/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- pretrain_finetune/duration/input_channels_40_47: gained none; lost none; missed id_11, id_157, id_174, id_20, id_24, id_42, id_55.
- pretrain_finetune/duration/system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_42, id_43, id_55.
- pretrain_finetune/hysteresis/input_channels_40_47: gained none; lost none; missed id_11, id_157, id_174, id_20, id_24, id_42, id_55.
- pretrain_finetune/hysteresis/system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_42, id_43, id_55.
- pretrain_finetune/strength_duration/input_channels_40_47: gained none; lost none; missed id_11, id_157, id_174, id_20, id_24, id_42, id_55.
- pretrain_finetune/strength_duration/system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_42, id_43, id_55.
- short_no_roll/duration/input_channels_40_47: gained none; lost none; missed id_11, id_121, id_138, id_139, id_14, id_140, id_147, id_149, id_15, id_150, id_169, id_174, id_20, id_24, id_40, id_42, id_47, id_49, id_51, id_55.
- short_no_roll/duration/system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_121, id_126, id_127, id_131, id_135, id_137, id_138, id_139, id_14, id_140, id_141, id_143, id_146, id_147, id_148, id_149, id_15, id_150, id_152, id_156, id_161, id_162, id_169, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_47, id_49, id_51, id_55.
- short_no_roll/hysteresis/input_channels_40_47: gained none; lost none; missed id_11, id_121, id_138, id_139, id_14, id_140, id_147, id_149, id_15, id_150, id_169, id_174, id_20, id_24, id_40, id_42, id_47, id_49, id_51, id_55.
- short_no_roll/hysteresis/system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_121, id_126, id_127, id_131, id_135, id_137, id_138, id_139, id_14, id_140, id_141, id_143, id_146, id_147, id_148, id_149, id_15, id_150, id_152, id_156, id_161, id_162, id_169, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_47, id_49, id_51, id_55.
- short_no_roll/strength_duration/input_channels_40_47: gained none; lost none; missed id_11, id_121, id_138, id_139, id_14, id_140, id_147, id_149, id_15, id_150, id_169, id_174, id_20, id_24, id_40, id_42, id_47, id_49, id_51, id_55.
- short_no_roll/strength_duration/system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_121, id_126, id_127, id_131, id_135, id_137, id_138, id_139, id_14, id_140, id_141, id_143, id_146, id_147, id_148, id_149, id_15, id_150, id_152, id_156, id_161, id_162, id_169, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_47, id_49, id_51, id_55.

## Verification and limits

Independent implementation replayed all 222 calibration candidates and nine selections, exact assessment predictions, all 192 corrected annual/pooled rows, unchanged-control parity and event IDs. Frozen dataset, source and predecessor receipts passed hash checks. All 153 thesis tests passed. Saved-score replay is not an independent inference replication. Single seed, reused benchmark, offline lookahead, extra input channels, unequal prior compute and few calibration events limit conclusions. No robust target claim or family selection from assessment.

[Protocol](alarm-evidence-protocol-2026-10-28.md), [metrics](tables/alarm-evidence-2026-10-28/verified_results.csv), [event IDs](tables/alarm-evidence-2026-10-28/event_ids.csv), [verification](tables/alarm-evidence-2026-10-28/verification.json).

[W&B project](https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning), [Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/alarm-evidence-2026-10-28). Unchanged weights and source scores are backed up in the prior W&B and temporal-postprocessing Releases.
