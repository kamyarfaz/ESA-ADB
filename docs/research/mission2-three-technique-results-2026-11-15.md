# Mission 2 three-technique study — 15 November 2026

The current Mission 2 setup has low detection recall and does not reproduce the strong Mission 1 results. Primary pooled mean selected-scope F0.5 is 0.413341 for short-no-roll, 0.367278 for masked pretraining and 0.481984 for expanded inputs. System-scope means are 0.137489, 0.218775 and 0.189472. Expansion improves selected-scope F0.5 in seeds42/43 but worsens seed44; masked pretraining improves seed42 but worsens seeds43/44. No consistent improvement across seeds is established.

All nine fresh runs completed 40 epochs and assessment, with no failed or omitted runs and no service restarts. These are Mission 2 results; they are not a continuation of the Mission 1 assessment period or a direct weight transfer.

## Design

Train on 2000, calibrate on January–September 2001, and assess continuously from 1 October 2001 through 30 June 2003. The last assessed grid timestamp is 2003-06-30 23:59:42. All nine checkpoint/rule choices jointly froze at 2026-11-15T17:39:20.283971+00:00 before assessment preparation or inference.

The baseline is already a Transformer autoencoder. Short-no-roll uses 40 epochs; masked pretraining uses 20 masked epochs followed by 20 short-no-roll epochs. The expanded technique uses 40 short-no-roll epochs with additional inputs. Each technique has fresh seeds 42, 43 and 44; seeds are random initialization/training repeats, not channel IDs. There are 98,228 eligible windows and 1,535 batches per epoch: 552,600 optimizer steps across nine models, plus 60 calibration-checkpoint evaluations and nine assessment passes.

Base inputs are 58,59,75,76,77,83,84,85. The expanded technique adds 90,86,89, selected only from Mission 2 training-year correlations within compatible anonymized subsystem/unit groups. Their worst-quarter absolute Spearman correlations with anchor59 are 0.98631, 0.98531 and 0.98516. This supports stable similarity, not proven incremental information; redundancy is explicitly allowed. Channel numbers and weights are not transferred from Mission 1.

Values use an 18-second causal previous-observation grid, a 600-second age limit, no backfill, and training-median missing-value imputation. Scalers use globally unannotated complete training rows. All methods share the same training exclusion mask. Values are not restored using anomaly annotations. A 256-sample window represents 76.8 minutes and reconstruction is retrospective with future context within a window. Scores maximize reconstruction error over all model inputs, so additional inputs also expand score sources.

Checkpoint and alarm-rule selection uses corrected selected-eight-channel Anomaly + Rare Event F0.5 on calibration only, subject to a 1% calibration sample-alarm cap. Only six selected-scope calibration event IDs are available; that is limited evidence for future behavior. The cap does not guarantee low assessment false-alarm duration.

## Pooled primary results: Anomaly + Rare Event

Selected scope always means the same eight base channels, including for the expanded model. System scope includes all Mission 2 channels. TPR is detected event IDs / annotated IDs; time FPR is false-alarm duration / nominal duration. Mean ± sample SD summarizes three seeds, not a confidence interval. Pooled metrics are recomputed continuously, not averaged from annual results.

| Technique | Scope | F0.5 mean ± SD | TPR mean | Time FPR mean |
|---|---|---:|---:|---:|
| Short-no-roll | system_all_channels | 0.137489 ± 0.084978 | 3.31% | 0.5278% |
| Short-no-roll | selected_eight_channels | 0.413341 ± 0.222897 | 16.67% | 0.5278% |
| Masked + short-no-roll | system_all_channels | 0.218775 ± 0.011879 | 5.52% | 1.4251% |
| Masked + short-no-roll | selected_eight_channels | 0.367278 ± 0.044344 | 15.38% | 1.4251% |
| Expanded-input short-no-roll | system_all_channels | 0.189472 ± 0.112803 | 4.79% | 0.9978% |
| Expanded-input short-no-roll | selected_eight_channels | 0.481984 ± 0.166671 | 23.08% | 0.9978% |

## Every seed and both event definitions

Anomaly-only is a secondary view of identical predictions. Rare-event and out-of-scope annotations remain neutral for false alarms under the corrected evaluator. This does not demonstrate recognition of rare events as normal.

| Technique | Seed | Definition | Scope | F0.5 | Detected / total | False events | Time FPR |
|---|---:|---|---|---:|---:|---:|---:|
| Short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.099979 | 9 / 362 | 13 | 0.1090% |
| Short-no-roll | 42 | anomaly_and_rare | selected_eight_channels | 0.166537 | 3 / 26 | 13 | 0.1090% |
| Short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 13 | 0.1090% |
| Short-no-roll | 42 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 13 | 0.1090% |
| Short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.234767 | 21 / 362 | 0 | 1.4686% |
| Short-no-roll | 43 | anomaly_and_rare | selected_eight_channels | 0.473502 | 4 / 26 | 0 | 1.4686% |
| Short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0 | 1.4686% |
| Short-no-roll | 43 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 0 | 1.4686% |
| Short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.077720 | 6 / 362 | 0 | 0.0059% |
| Short-no-roll | 44 | anomaly_and_rare | selected_eight_channels | 0.599983 | 6 / 26 | 0 | 0.0059% |
| Short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0 | 0.0059% |
| Short-no-roll | 44 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 0 | 0.0059% |
| Masked + short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.225626 | 20 / 362 | 0 | 1.4922% |
| Masked + short-no-roll | 42 | anomaly_and_rare | selected_eight_channels | 0.392858 | 3 / 26 | 0 | 1.4922% |
| Masked + short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0 | 1.4922% |
| Masked + short-no-roll | 42 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 0 | 1.4922% |
| Masked + short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.205058 | 20 / 362 | 11 | 1.3270% |
| Masked + short-no-roll | 43 | anomaly_and_rare | selected_eight_channels | 0.316074 | 6 / 26 | 11 | 1.3270% |
| Masked + short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 11 | 1.3270% |
| Masked + short-no-roll | 43 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 11 | 1.3270% |
| Masked + short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.225641 | 20 / 362 | 0 | 1.4560% |
| Masked + short-no-roll | 44 | anomaly_and_rare | selected_eight_channels | 0.392904 | 3 / 26 | 0 | 1.4560% |
| Masked + short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0 | 1.4560% |
| Masked + short-no-roll | 44 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 0 | 1.4560% |
| Expanded-input short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.256643 | 24 / 362 | 2 | 1.4923% |
| Expanded-input short-no-roll | 42 | anomaly_and_rare | selected_eight_channels | 0.559594 | 7 / 26 | 2 | 1.4923% |
| Expanded-input short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 2 | 1.4923% |
| Expanded-input short-no-roll | 42 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 2 | 1.4923% |
| Expanded-input short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.252534 | 23 / 362 | 0 | 1.4827% |
| Expanded-input short-no-roll | 43 | anomaly_and_rare | selected_eight_channels | 0.595697 | 6 / 26 | 0 | 1.4827% |
| Expanded-input short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0 | 1.4827% |
| Expanded-input short-no-roll | 43 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 0 | 1.4827% |
| Expanded-input short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.059240 | 5 / 362 | 10 | 0.0184% |
| Expanded-input short-no-roll | 44 | anomaly_and_rare | selected_eight_channels | 0.290660 | 5 / 26 | 10 | 0.0184% |
| Expanded-input short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 10 | 0.0184% |
| Expanded-input short-no-roll | 44 | anomaly_only | selected_eight_channels | 0.000000 | 0 / 1 | 10 | 0.0184% |

## Paired changes against short-no-roll

These compare the same seed, period and scope. The expanded-versus-base contrast tests added inputs and score sources together. The masked contrast changes the training procedure. No best seed is selected.

| Technique | Seed | Scope | Primary Δ F0.5 | Δ detected IDs | Δ time FPR (percentage points) |
|---|---:|---|---:|---:|---:|
| Masked + short-no-roll | 42 | selected_eight_channels | +0.226320 | +0 | +1.38320 |
| Expanded-input short-no-roll | 42 | selected_eight_channels | +0.393056 | +4 | +1.38329 |
| Masked + short-no-roll | 42 | system_all_channels | +0.125647 | +11 | +1.38320 |
| Expanded-input short-no-roll | 42 | system_all_channels | +0.156664 | +15 | +1.38329 |
| Masked + short-no-roll | 43 | selected_eight_channels | -0.157428 | +2 | -0.14156 |
| Expanded-input short-no-roll | 43 | selected_eight_channels | +0.122195 | +2 | +0.01408 |
| Masked + short-no-roll | 43 | system_all_channels | -0.029709 | -1 | -0.14156 |
| Expanded-input short-no-roll | 43 | system_all_channels | +0.017767 | +2 | +0.01408 |
| Masked + short-no-roll | 44 | selected_eight_channels | -0.207079 | -3 | +1.45004 |
| Expanded-input short-no-roll | 44 | selected_eight_channels | -0.309323 | -1 | +0.01245 |
| Masked + short-no-roll | 44 | system_all_channels | +0.147921 | +14 | +1.45004 |
| Expanded-input short-no-roll | 44 | system_all_channels | -0.018480 | -1 | +0.01245 |

## Frozen calibration choices

| Technique | Seed | Epoch | Calibration F0.5 | Threshold | Gap | Minimum duration |
|---|---:|---:|---:|---:|---:|---:|
| Short-no-roll | 42 | 15 | 0.500000 | 9.3674707 | 0 | 1 |
| Short-no-roll | 43 | 25 | 0.714254 | 7.234385 | 0 | 1 |
| Short-no-roll | 44 | 10 | 0.714254 | 18.343306 | 0 | 1 |
| Masked + short-no-roll | 42 | 40 | 0.714233 | 4.9583592 | 0 | 1 |
| Masked + short-no-roll | 43 | 25 | 0.714233 | 16.846483 | 0 | 1 |
| Masked + short-no-roll | 44 | 25 | 0.714286 | 20.729507 | 0 | 1 |
| Expanded-input short-no-roll | 42 | 30 | 0.833307 | 8.6981335 | 0 | 1 |
| Expanded-input short-no-roll | 43 | 40 | 0.714286 | 6.6534424 | 0 | 1 |
| Expanded-input short-no-roll | 44 | 5 | 0.714244 | 17.864586 | 0 | 1 |

## Verification and interpretation

Independent audit checked 322 completion-file hashes, raw inputs, source/runtime/settings, all nine final training states and histories, nominal-only scaler refits, all 60 calibration grids (5940 candidates), all frozen choices, and 564 exact float32 inference spots across 69 checkpoints/passes. Full prediction replay, all 144 annual/pooled metric rows and 7119 detected/missed ID records match. Independent pandas source lookup additionally checked 204,655 assessment value/availability/age samples. All 189 thesis tests passed, including evaluator-reference parity and verifier boundaries. Neural replay and raw-source replay are sampled, not full second passes. The corrected evaluator is reused; postprocessing, rule ranking, inference-window enumeration and event-ID enumeration are independent.

Three seeds, six selected calibration IDs, reused benchmark data, anonymized channels, different event composition and temporal resolution, neutral annotations and offline reconstruction limit generalization. Additional channels require additional compute and may be redundant. Differences from Mission 1 cannot be attributed solely to the mission or channel quality because preparation, chronology and event populations differ. No robust thesis F0.5 ≥ 0.85 achievement or deployment claim is made. Historical results are preserved.

[All 144 metrics](tables/mission2-three-technique-2026-11-14/verified_results.csv), [all detected/missed IDs](tables/mission2-three-technique-2026-11-14/verified_event_ids.csv), [all paired changes](tables/mission2-three-technique-2026-11-14/paired_changes.csv), [seed summaries](tables/mission2-three-technique-2026-11-14/seed_summary.csv), [verification](tables/mission2-three-technique-2026-11-14/verification.json).

[Protocol](mission2-three-technique-protocol-2026-11-14.md), [W&B runs](tables/mission2-three-technique-2026-11-14/wandb-evaluation.json), [artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/mission2-three-technique-2026-11-15).
