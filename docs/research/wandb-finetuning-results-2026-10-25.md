# W&B nine-run fine-tuning results — 25 October 2026

Extra-input short-no-roll improved on the reused 2007–2013 assessment: system event F0.5 rose from 0.824405 to 0.844573, and channels-40–47 event F0.5 from 0.817353 to 0.830900. Target detections increased from 61/67 to 62/67; system detections increased from 76/91 to 80/91. The original short-no-roll and masked-pretraining models were retained by calibration and reproduce their previous predictions exactly.

The extra-input result has the highest system F0.5 among these three selected models, but original short-no-roll still has the highest channels-40–47 F0.5 (0.907140) and substantially lower nominal-time FPR. Greater detection coverage does not imply better precision.

## Fixed protocol and selection

Training: 2000–2005 globally nominal windows. Calibration: 2006. Assessment: 2007–2013. Inputs are channels 40–47 for methods 1 and 2; method 3 additionally uses 14, 21, 29. All local event metrics target 40–47 only. Each existing checkpoint received three separate ten-epoch continuations at initial learning rates 3e-6, 1e-5 and 3e-5. AdamW was reset, cosine scheduling ended at 1e-6, batch size was 64 and seed was 42. Masked pretraining was not repeated. All methods retain their starting architecture, scaler and short-no-roll synthetic anomaly objective.

Candidate epochs 5 and 10 used the frozen threshold/postprocessing grid and a 1% calibration alarm cap. The original model/rule was eligible as epoch 0. Selection used corrected system calibration F0.5 only; ties favor the original/earlier epoch then lower learning rate. All nine runs completed and all three choices froze before assessment. Unselected candidates were not assessed. W&B recorded the fixed local grid; it did not optimize or execute training itself.

| Method | LR | Epoch 0 calibration F0.5 | Epoch 5 | Epoch 10 | Run-selected epoch |
|---|---:|---:|---:|---:|---:|
| Original short-no-roll | 3e-06 | 0.882138247 | 0.882138247 | 0.882138247 | 0 |
| Original short-no-roll | 1e-05 | 0.882138247 | 0.882118656 | 0.882099065 | 0 |
| Original short-no-roll | 3e-05 | 0.882138247 | 0.833229199 | 0.882089492 | 0 |
| Masked pretraining + short-no-roll | 3e-06 | 0.789275870 | 0.657775523 | 0.555555556 | 0 |
| Masked pretraining + short-no-roll | 1e-05 | 0.789275870 | 0.483341875 | 0.555552875 | 0 |
| Masked pretraining + short-no-roll | 3e-05 | 0.789275870 | 0.555552875 | 0.555545109 | 0 |
| Short-no-roll + extra inputs | 3e-06 | 0.977864804 | 0.977864804 | 0.977864804 | 0 |
| Short-no-roll + extra inputs | 1e-05 | 0.977864804 | 0.977937047 | 0.977985208 | 10 |
| Short-no-roll + extra inputs | 3e-05 | 0.977864804 | 0.977975943 | 0.978000023 | 10 |

Across rates, methods 1 and 2 retain epoch 0 (the lower-rate run represents the tie). Method 3 selects LR3e-5, epoch10, with calibration system F0.5 0.978000023 versus original 0.977864804. This small calibration gain was not a guarantee of assessment improvement.

## Annual and pooled metrics

Precision is the corrected ESA event precision including its nominal-time penalty. Event TPR counts detected target IDs; time FPR measures false-alarm duration divided by nominal duration. Non-target annotations remain neutral. Pooled metrics are recomputed over the continuous seven-year predictions, not averaged from annual values. IDs spanning years can appear in more than one annual count.

### Original short-no-roll — system_all_channels

| Period | Variant | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| 2007 | selected | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| 2008 | original | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| 2008 | selected | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| 2009 | original | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| 2009 | selected | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| 2010 | original | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| 2010 | selected | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| 2011 | original | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| 2011 | selected | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| 2012 | original | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| 2012 | selected | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| 2013 | original | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| 2013 | selected | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| 2007-2013 | original | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |
| 2007-2013 | selected | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |

### Original short-no-roll — input_channels_40_47

| Period | Variant | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| 2007 | selected | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| 2008 | original | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| 2008 | selected | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| 2009 | original | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| 2009 | selected | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| 2010 | original | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| 2010 | selected | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| 2011 | original | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| 2011 | selected | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| 2012 | original | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| 2012 | selected | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| 2013 | original | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| 2013 | selected | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| 2007-2013 | original | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |
| 2007-2013 | selected | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |

### Masked pretraining + short-no-roll — system_all_channels

| Period | Variant | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| 2007 | selected | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954989 |
| 2008 | original | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| 2008 | selected | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777118 |
| 2009 | original | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| 2009 | selected | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832392 |
| 2010 | original | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| 2010 | selected | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221 |
| 2011 | original | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| 2011 | selected | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633381 |
| 2012 | original | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| 2012 | selected | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906 |
| 2013 | original | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| 2013 | selected | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696 |
| 2007-2013 | original | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |
| 2007-2013 | selected | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675 |

### Masked pretraining + short-no-roll — input_channels_40_47

| Period | Variant | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| 2007 | selected | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999022 |
| 2008 | original | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| 2008 | selected | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863 |
| 2009 | original | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| 2009 | selected | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061 |
| 2010 | original | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| 2010 | selected | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948 |
| 2011 | original | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| 2011 | selected | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671168 |
| 2012 | original | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| 2012 | selected | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239 |
| 2013 | original | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| 2013 | selected | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917594 |
| 2007-2013 | original | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |
| 2007-2013 | selected | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844315 |

### Short-no-roll + extra inputs — system_all_channels

| Period | Variant | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 16/16 | 0.939796 | 1.000000 | 0.146683 | 0.951250 |
| 2007 | selected | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519 |
| 2008 | original | 10/13 | 0.999115 | 0.769231 | 0.088480 | 0.942766 |
| 2008 | selected | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942782 |
| 2009 | original | 11/12 | 0.915634 | 0.916667 | 0.112677 | 0.915840 |
| 2009 | selected | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099 |
| 2010 | original | 9/12 | 0.817253 | 0.750000 | 0.113506 | 0.802855 |
| 2010 | selected | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936993 |
| 2011 | original | 11/15 | 0.915871 | 0.733333 | 0.086796 | 0.872438 |
| 2011 | selected | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932830 |
| 2012 | original | 5/10 | 0.555177 | 0.500000 | 0.068217 | 0.543188 |
| 2012 | selected | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773 |
| 2013 | original | 14/14 | 0.504152 | 1.000000 | 42.382655 | 0.559652 |
| 2013 | selected | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2007-2013 | original | 76/91 | 0.821758 | 0.835165 | 5.930307 | 0.824405 |
| 2007-2013 | selected | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573 |

### Short-no-roll + extra inputs — input_channels_40_47

| Period | Variant | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/13 | 0.927209 | 1.000000 | 0.146683 | 0.940907 |
| 2007 | selected | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848 |
| 2008 | original | 7/9 | 0.999115 | 0.777778 | 0.088480 | 0.945312 |
| 2008 | selected | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328 |
| 2009 | original | 10/10 | 0.908067 | 1.000000 | 0.112677 | 0.925076 |
| 2009 | selected | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716 |
| 2010 | original | 8/8 | 0.799092 | 1.000000 | 0.113506 | 0.832545 |
| 2010 | selected | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999423 |
| 2011 | original | 10/11 | 0.908302 | 0.909091 | 0.086796 | 0.908460 |
| 2011 | selected | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692 |
| 2012 | original | 4/8 | 0.499659 | 0.500000 | 0.068217 | 0.499727 |
| 2012 | selected | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499811 |
| 2013 | original | 9/9 | 0.471415 | 1.000000 | 42.382655 | 0.527143 |
| 2013 | selected | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2007-2013 | original | 61/67 | 0.796979 | 0.910448 | 5.930307 | 0.817353 |
| 2007-2013 | selected | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830900 |

## Event-ID changes

- Original short-no-roll, system_all_channels: gained none; lost none; selected missed id_11, id_115, id_119, id_120, id_121, id_126, id_127, id_131, id_135, id_137, id_138, id_139, id_14, id_140, id_141, id_143, id_146, id_147, id_148, id_149, id_15, id_150, id_152, id_156, id_161, id_162, id_169, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_47, id_49, id_51, id_55.
- Original short-no-roll, input_channels_40_47: gained none; lost none; selected missed id_11, id_121, id_138, id_139, id_14, id_140, id_147, id_149, id_15, id_150, id_169, id_174, id_20, id_24, id_40, id_42, id_47, id_49, id_51, id_55.
- Masked pretraining + short-no-roll, system_all_channels: gained none; lost none; selected missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_42, id_43, id_55.
- Masked pretraining + short-no-roll, input_channels_40_47: gained none; lost none; selected missed id_11, id_157, id_174, id_20, id_24, id_42, id_55.
- Short-no-roll + extra inputs, system_all_channels: gained id_152, id_162, id_171, id_49; lost none; selected missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- Short-no-roll + extra inputs, input_channels_40_47: gained id_49; lost none; selected missed id_11, id_174, id_20, id_24, id_42.

## Interpretation and limits

For extra-input short-no-roll, false predicted events fell from 11 to 10 and nominal-time FPR decreased slightly (5.930307% to 5.909894%). The result combines more detected events and modestly fewer false alarms, but the long nominal alarm duration remains a weakness. It does not dominate original short-no-roll, which has lower coverage but much higher target precision and F0.5.

The first two methods did not improve under this bounded search. Their original models won calibration, so their unchanged assessment results are fallback outcomes rather than evidence that every continuation has identical behavior. Training-loss reductions alone do not establish improvements in event detection.

One seed was used. The original-short-no-roll LR1e-5 setting repeats the prior continuation and is not independent evidence. Nine runs and two candidate epochs per run increase calibration selection exposure. The benchmark has been reused across studies and guided research interest, so this is not independent confirmatory validation. Model starting histories and total compute differ, and method 3 receives extra information through three additional input channels. No robust achievement of the thesis F0.5 target is claimed.

## Verification and artifacts

Independent verification passed all receipt hashes, frozen source/runtime/settings/data/reference checks, nine ten-epoch histories, 18 full calibration grid replays from saved scores, within-run and cross-rate selection, selected weight/scaler equality, binary assessment replay, all96 corrected annual/pooled metric rows, and annual/pooled event IDs. Original controls match previously completed results. Verification replays stored scores; it does not independently rerun neural inference. All144 thesis tests passed. All nine training runs are finished in W&B.

[Full metrics](tables/wandb-finetuning-2026-10-25/verified_results.csv), [annual and pooled event IDs](tables/wandb-finetuning-2026-10-25/event_ids.csv), [calibration attempts](tables/wandb-finetuning-2026-10-25/calibration_comparison.csv), [verification](tables/wandb-finetuning-2026-10-25/verification.json), [protocol](wandb-finetuning-protocol-2026-10-24.md).

[W&B project](https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning). Verified assessment is published in three separate evaluation runs, with original and selected annual/pooled tables.

[GitHub Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/wandb-finetuning-2026-10-25). Release checksum evidence is recorded beside the metric tables. This additive backup does not replace the full external-drive copy or certify unrelated migration contents.
