# Matched input expansion across three techniques — 13 November 2026

Adding training-selected channels 33 and 49 increased mean pooled target 40–47 F0.5 in all three families. The short-no-roll comparison improved in all three seeds. Masked pretraining improved in two seeds and worsened in seed 42. The family retaining 14/21/29 improved over poor fresh controls, but remained unstable and had substantial nominal alarm duration. These are results on the reused 2007–2013 benchmark, not independent confirmation or robust achievement of the thesis target.

All 18 fresh models completed 40 epochs. All selections jointly froze at 11:43:46 UTC on 13 November; assessment finished at 12:19:39 UTC. There were no failed or omitted training runs. A 9 November host reboot interrupted run 6; the unchanged service resumed its checkpoint. The reboot cause is unknown. Performance failures are retained below.

## Design and rationale

Each family has a fresh matched control and an arm adding 33/49, paired by seeds 42,43,44. The first two families use 40–47 in the controls (8→10 inputs). The third already retains 14/21/29 (11→13 inputs). Both arms train 2000–2005 and select checkpoint and alarm rule using corrected 40–47 event F0.5 on 2006 only, with a 1% calibration alarm-fraction cap. No assessment scores choose channels, thresholds, checkpoints, seeds or models.

Channels 33 and 49 were selected solely from 2000–2005 training correlations. They had the strongest worst-quarter absolute Spearman correlation in the prespecified pools for anchors 40 and 47: 0.6673 and 0.5515 over 24 quarters; pooled values 0.99795 and 0.98766. Their first-difference Pearson correlations were near zero (0.00350 and −0.00260). The rationale was stable shared operating context, not demonstrated causal usefulness. High level correlations may reflect slow trends. All 225 candidate/quarter coefficients were replayed before launch.

The baseline is already a Transformer autoencoder. Architecture, nominal-only scaler fitting, training windows and optimizer were fixed. All runs receive 40 epochs; masked pretraining allocates 20 masked + 20 short-no-roll. Each epoch uses 349,461 windows and 5,461 batches. The 18 runs require 3,931,920 optimizer steps plus 120 calibration checkpoint evaluations and 18 assessment passes. The observed execution spans approximately 5 days 23 hours from 7 November 13:06 UTC through assessment completion, including input checks and the reboot. Adding inputs increases compute even with an equal epoch budget.

Reconstruction errors are averaged within 256-sample windows, overlapped at stride 32, and maximized over every input channel. New inputs therefore also become score sources. This study tests input-and-score expansion, not isolated contextual conditioning.

## Three-seed pooled results

F0.5 uses corrected event-ID/duration evaluation. Event TPR is detected IDs divided by annotated IDs; time FPR is false-alarm duration divided by nominal duration. Means and sample standard deviations below summarize three seeds, not confidence intervals. Pooled metrics are recomputed continuously over all seven years, not averaged from annual metrics.

| Technique | Scope | Control F0.5 mean ± SD | Added 33/49 mean ± SD | Paired change mean ± SD |
|---|---|---:|---:|---:|
| Short-no-roll | input_channels_40_47 | 0.848385 ± 0.074807 | 0.910879 ± 0.022037 | +0.062494 ± 0.059950 |
| Short-no-roll | system_all_channels | 0.779567 ± 0.066985 | 0.844667 ± 0.017542 | +0.065099 ± 0.049962 |
| Masked pretraining + short-no-roll | input_channels_40_47 | 0.827188 ± 0.080111 | 0.909984 ± 0.020285 | +0.082796 ± 0.094550 |
| Masked pretraining + short-no-roll | system_all_channels | 0.764581 ± 0.078250 | 0.853041 ± 0.019093 | +0.088461 ± 0.087922 |
| Short-no-roll retaining 14/21/29 | input_channels_40_47 | 0.244716 ± 0.083200 | 0.819257 ± 0.100583 | +0.574541 ± 0.148108 |
| Short-no-roll retaining 14/21/29 | system_all_channels | 0.263462 ± 0.084588 | 0.804662 ± 0.090050 | +0.541200 ± 0.144264 |

## Every paired seed, including adverse outcomes

| Technique | Seed | Scope | F0.5 control → added | Δ F0.5 | Detected control → added | Time FPR (%) control → added |
|---|---:|---|---:|---:|---:|---:|
| Short-no-roll | 42 | system_all_channels | 0.847929 → 0.864151 | +0.016222 | 48 → 51 | 0.02217 → 0.04286 |
| Short-no-roll | 42 | input_channels_40_47 | 0.921418 → 0.936029 | +0.014611 | 47 → 50 | 0.02217 → 0.04286 |
| Short-no-roll | 43 | system_all_channels | 0.776724 → 0.839722 | +0.062998 | 39 → 61 | 0.03404 → 0.07914 |
| Short-no-roll | 43 | input_channels_40_47 | 0.851815 → 0.894955 | +0.043140 | 38 → 60 | 0.03404 → 0.07914 |
| Short-no-roll | 44 | system_all_channels | 0.714049 → 0.830127 | +0.116079 | 41 → 47 | 0.04861 → 0.04642 |
| Short-no-roll | 44 | input_channels_40_47 | 0.771922 → 0.901652 | +0.129730 | 40 → 46 | 0.04861 → 0.04642 |
| Masked pretraining + short-no-roll | 42 | system_all_channels | 0.841235 → 0.834648 | -0.006587 | 51 → 62 | 0.05925 → 0.14761 |
| Masked pretraining + short-no-roll | 42 | input_channels_40_47 | 0.908683 → 0.888156 | -0.020527 | 50 → 61 | 0.05925 → 0.14761 |
| Masked pretraining + short-no-roll | 43 | system_all_channels | 0.684827 → 0.851710 | +0.166882 | 30 → 53 | 0.02599 → 0.06307 |
| Masked pretraining + short-no-roll | 43 | input_channels_40_47 | 0.748538 → 0.913540 | +0.165002 | 28 → 51 | 0.02599 → 0.06307 |
| Masked pretraining + short-no-roll | 44 | system_all_channels | 0.767679 → 0.872765 | +0.105086 | 49 → 62 | 0.06306 → 0.07301 |
| Masked pretraining + short-no-roll | 44 | input_channels_40_47 | 0.824342 → 0.928255 | +0.103913 | 48 → 60 | 0.06306 → 0.07301 |
| Short-no-roll retaining 14/21/29 | 42 | system_all_channels | 0.286549 → 0.701615 | +0.415066 | 71 → 66 | 5.21676 → 0.88137 |
| Short-no-roll retaining 14/21/29 | 42 | input_channels_40_47 | 0.253384 → 0.705582 | +0.452198 | 59 → 59 | 5.21676 → 0.88137 |
| Short-no-roll retaining 14/21/29 | 43 | system_all_channels | 0.169728 → 0.868227 | +0.698499 | 62 → 66 | 3.65650 → 5.91005 |
| Short-no-roll retaining 14/21/29 | 43 | input_channels_40_47 | 0.157521 → 0.896723 | +0.739203 | 56 → 57 | 3.65650 → 5.91005 |
| Short-no-roll retaining 14/21/29 | 44 | system_all_channels | 0.334110 → 0.844144 | +0.510033 | 64 → 62 | 5.39562 → 5.87759 |
| Short-no-roll retaining 14/21/29 | 44 | input_channels_40_47 | 0.323243 → 0.855465 | +0.532223 | 59 → 50 | 5.39562 → 5.87759 |

For short-no-roll, added inputs improve target F0.5 in all seeds, but seed 42/43 also increase nominal alarm duration. For masked pretraining, seed 42 gains 11 target detections (50→61) while false events rise 2→8, lowering F0.5; its timeFPR also rises. For the third family, fresh controls produce 141–356false events and target F0.5 of 0.158–0.323. Added inputs reduce false-event counts, but seeds 43/44 spend approximately 5.9% of nominal time in false alarms. Fewer long false alarms can yield high event F0.5 despite poor alarm duration. The calibration cap does not guarantee the same cap on assessment. The third family should not be presented as a reliable improvement based only on F0.5.

## Comparison with unchanged historical runs

Historical seed 42 models used system-scope calibration selection; this study uses target 40–47 selection and fresh training. Comparisons below are descriptive and cannot isolate the channel effect. The fresh paired controls above are the primary comparison. No historical row or selected checkpoint is overwritten.

| Technique | Historical target F0.5 | Fresh control target mean | Added 33/49 target mean | Historical system F0.5 |
|---|---:|---:|---:|---:|
| Short-no-roll | 0.907140 | 0.848385 | 0.910879 | 0.836070 |
| Masked pretraining + short-no-roll | 0.844315 | 0.827188 | 0.909984 | 0.795675 |
| Short-no-roll retaining 14/21/29 | 0.817353 | 0.244716 | 0.819257 | 0.824405 |

[Every new seed versus the historical reference](tables/context-expansion-2026-11-04/historical_comparison.csv).

## Frozen calibration selections

| Technique | Seed | Arm | Epoch | Target calibration F0.5 | Threshold | Gap | Minimum duration |
|---|---:|---|---:|---:|---:|---:|---:|
| Short-no-roll | 42 | control | 5 | 0.999916381 | 9.40145683 | 0 | 1 |
| Short-no-roll | 42 | context | 5 | 0.999774569 | 8.47592258 | 0 | 1 |
| Short-no-roll | 43 | control | 20 | 0.999875210 | 3.76938868 | 0 | 1 |
| Short-no-roll | 43 | context | 10 | 0.999790581 | 6.05175972 | 0 | 1 |
| Short-no-roll | 44 | control | 10 | 0.999749408 | 5.65638542 | 0 | 1 |
| Short-no-roll | 44 | context | 10 | 0.999824890 | 4.42221022 | 0 | 1 |
| Masked pretraining + short-no-roll | 42 | control | 25 | 0.909049624 | 7.74424791 | 0 | 1 |
| Masked pretraining + short-no-roll | 42 | context | 30 | 0.999107978 | 5.12772703 | 0 | 1 |
| Masked pretraining + short-no-roll | 43 | control | 25 | 0.833255230 | 7.92796516 | 0 | 1 |
| Masked pretraining + short-no-roll | 43 | context | 35 | 0.999916381 | 4.48375034 | 0 | 1 |
| Masked pretraining + short-no-roll | 44 | control | 25 | 0.999737114 | 6.86483145 | 0 | 1 |
| Masked pretraining + short-no-roll | 44 | context | 30 | 0.999866062 | 4.57279348 | 0 | 1 |
| Short-no-roll retaining 14/21/29 | 42 | control | 15 | 0.999866062 | 7.84069633 | 0 | 1 |
| Short-no-roll retaining 14/21/29 | 42 | context | 15 | 0.999866062 | 8.71276569 | 0 | 1 |
| Short-no-roll retaining 14/21/29 | 43 | control | 5 | 0.999891221 | 9.49903011 | 0 | 1 |
| Short-no-roll retaining 14/21/29 | 43 | context | 5 | 0.999611302 | 7.29566765 | 0 | 1 |
| Short-no-roll retaining 14/21/29 | 44 | control | 5 | 0.999815741 | 8.69624805 | 0 | 1 |
| Short-no-roll retaining 14/21/29 | 44 | context | 10 | 0.999824890 | 9.37422848 | 0 | 1 |

Only six target event IDs occur in 2006 calibration. Near-ceiling calibration results therefore provide weak evidence about future behavior. All 120 checkpoint grids and every rejected candidate are retained. No fallback was substituted after seeing assessment.

## Event changes and annual evidence

[All 288 annual/pooled metric rows](tables/context-expansion-2026-11-04/verified_results.csv), [all detected/missed IDs](tables/context-expansion-2026-11-04/verified_event_ids.csv), [all 144 paired annual/pooled comparisons](tables/context-expansion-2026-11-04/paired_changes.csv), and [three-seed summaries](tables/context-expansion-2026-11-04/seed_summary.csv).

- short_no_roll, seed 42, system_all_channels: gained id_157;id_29;id_31;id_40;id_48;id_51; lost id_149;id_177;id_44.
- short_no_roll, seed 42, input_channels_40_47: gained id_157;id_29;id_31;id_40;id_48;id_51; lost id_149;id_177;id_44.
- short_no_roll, seed 43, system_all_channels: gained id_117;id_118;id_121;id_122;id_123;id_124;id_125;id_14;id_140;id_142;id_147;id_149;id_15;id_150;id_153;id_26;id_27;id_44;id_47;id_48;id_50;id_55; lost none.
- short_no_roll, seed 43, input_channels_40_47: gained id_117;id_118;id_121;id_122;id_123;id_124;id_125;id_14;id_140;id_142;id_147;id_149;id_15;id_150;id_153;id_26;id_27;id_44;id_47;id_48;id_50;id_55; lost none.
- short_no_roll, seed 44, system_all_channels: gained id_123;id_125;id_128;id_139;id_149;id_150;id_176;id_182;id_26;id_28;id_31;id_40;id_51; lost id_121;id_124;id_140;id_142;id_177;id_50;id_55.
- short_no_roll, seed 44, input_channels_40_47: gained id_123;id_125;id_128;id_139;id_149;id_150;id_176;id_182;id_26;id_28;id_31;id_40;id_51; lost id_121;id_124;id_140;id_142;id_177;id_50;id_55.
- pretrain_finetune, seed 42, system_all_channels: gained id_122;id_128;id_130;id_14;id_147;id_153;id_157;id_165;id_40;id_49;id_51; lost none.
- pretrain_finetune, seed 42, input_channels_40_47: gained id_122;id_128;id_130;id_14;id_147;id_153;id_157;id_165;id_40;id_49;id_51; lost none.
- pretrain_finetune, seed 43, system_all_channels: gained id_118;id_124;id_128;id_129;id_130;id_132;id_138;id_140;id_142;id_147;id_149;id_150;id_153;id_155;id_159;id_160;id_165;id_175;id_176;id_177;id_183;id_184;id_186;id_187;id_44;id_51;id_55; lost id_14;id_182;id_47;id_48.
- pretrain_finetune, seed 43, input_channels_40_47: gained id_118;id_124;id_128;id_129;id_130;id_132;id_138;id_140;id_142;id_147;id_149;id_150;id_153;id_155;id_159;id_160;id_165;id_175;id_176;id_177;id_183;id_184;id_186;id_187;id_44;id_51;id_55; lost id_14;id_182;id_47;id_48.
- pretrain_finetune, seed 44, system_all_channels: gained id_121;id_129;id_14;id_153;id_155;id_157;id_165;id_176;id_187;id_40;id_49;id_51;id_55; lost none.
- pretrain_finetune, seed 44, input_channels_40_47: gained id_121;id_129;id_14;id_153;id_155;id_157;id_165;id_176;id_187;id_40;id_49;id_51;id_55; lost id_123.
- expanded_short_no_roll, seed 42, system_all_channels: gained id_47; lost id_119;id_120;id_138;id_178;id_188;id_189.
- expanded_short_no_roll, seed 42, input_channels_40_47: gained id_47; lost id_138.
- expanded_short_no_roll, seed 43, system_all_channels: gained id_131;id_135;id_161;id_51; lost none.
- expanded_short_no_roll, seed 43, input_channels_40_47: gained id_51; lost none.
- expanded_short_no_roll, seed 44, system_all_channels: gained id_115;id_120;id_126;id_127;id_131;id_148;id_161;id_162; lost id_149;id_150;id_153;id_157;id_172;id_177;id_26;id_40;id_44;id_46.
- expanded_short_no_roll, seed 44, input_channels_40_47: gained none; lost id_149;id_150;id_153;id_157;id_172;id_177;id_26;id_40;id_44.

## Independent verification and limits

Verified all 599 completion-file hashes, all four prepared input hashes, frozen source/runtime/settings, all 18 training/assessment receipts and histories. Independently refitted nominal-only scalers. Replayed all 120 grids (11880 candidate rows) with independently implemented postprocessing and rule selection. All 1416 fixed timestamp checks across 120 candidate and 18 selected-checkpoint assessments reproduced float32 scores exactly. All continuous predictions, 288 metric rows and detected/missed IDs matched. This is sampled neural replay, not a second full inference pass. The corrected evaluator is shared with production and checked against the bundled independent reference; interval postprocessing, spot-window enumeration and event-ID enumeration are independently implemented.

The original 178-test thesis suite and three additional verifier boundary tests passed. Three seeds are limited variability evidence. Both calibration and assessment benchmarks have been reused; channel pools and methods reflect prior research. Inherited annotation-informed benchmark preparation, offline overlapping windows with future context, neutral out-of-scope annotations, additional training/selection compute and expanded score sources limit interpretation. No robust F0.5 ≥ 0.85 or deployment/generalization claim is made. Scoped input integrity verification does not close unrelated historical migration exceptions.

[Protocol](context-expansion-protocol-2026-11-04.md), [verification receipt](tables/context-expansion-2026-11-04/verification.json), [checkpoint spots](tables/context-expansion-2026-11-04/inference_spots.csv), [W&B evaluation runs](tables/context-expansion-2026-11-04/wandb-evaluation.json), [artifact backup Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/context-expansion-2026-11-13).
