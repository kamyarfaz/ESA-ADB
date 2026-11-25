# Cross-mission whole-system and transfer results — 25 November 2026

All 18 fresh models and all 18 predefined transfer conditions completed. The experiment measures whole-system detection using broader mission-specific inputs and separately tests existing Mission 1 weights on Mission 2. All three seeds are retained. Results below use the corrected event evaluator; input-channel coverage is not counted as detection.

The primary whole-system mean F0.5 values are 0.526498/0.589682/0.690503 for Mission 1 and 0.719673/0.635863/0.789561 for Mission 2, in short-no-roll/masked/expanded order. Mission 2 primary results improve over its previous study, but this is mainly rare-event detection: eight of nine fresh models detect 0/6 anomalies; only masked seed 43 detects 1/6. Mission 1 remains seed-sensitive. Expanded inputs improve primary system F0.5 over the paired baseline only in seed 44 for Mission 2 and seeds 42/44 for Mission 1, so improvement is not consistent across seeds.

Direct transfer is unsuccessful: all nine frozen-source-scaling models alarm continuously, with 100% nominal-time FPR and F0.5=0. Target normalization leaves system primary F0.5 means 0.081592/0.029058/0.047814 and mean time FPR 76.606%/95.992%/81.051%, respectively. High detected counts under these conditions reflect excessive alarm coverage, not useful transfer.

## Design and chronology

Mission 1 trains on 2000–2005, calibrates on 2006 and assesses 2007–2013. Mission 2 trains on 2000, calibrates January–September 2001 and assesses October 2001–June 2003. These are separate time splits, not predictions over training years. All 18 fresh checkpoint/rule selections jointly froze at 2026-11-25T14:58:36.872206+00:00 before assessment arrays were prepared. Source-transfer models retain their historical Mission 1 2006 rules. No assessment result selected a channel map, seed, checkpoint or threshold.

The Transformer autoencoder uses 40 epochs and batch 32. Masked training uses 20 masked epochs followed by 20 short-no-roll epochs; other methods use 40 short-no-roll epochs. Seeds 42/43/44 are random training repeats, not channel numbers. Each mission uses 13 training-selected representatives of all target-channel groups; expanded models use 25 Mission 1 or 26 Mission 2 inputs. Nominal windows: 349,383 for Mission 1 and 98,224 for Mission 2, yielding 10,919 and 3,070 batches per epoch. Total fresh optimizer steps: 5,036,040. There are 120 calibration checkpoint evaluations, 18 fresh assessment passes and 18 transfer passes, plus independent auditing.

Mission 1 uses 30-second and Mission 2 uses 18-second grids; transfer uses 30 seconds to retain source numerical cadence. Raw values are causal and split-contained, with training-only median/IQR scaling. Only Mission 1 channels 62/63 allow 900-second-old readings, as explicitly approved after v1 preparation failed before any of its 18 runs started. All other inputs retain 600 seconds. V1 source, arrays and failure logs remain preserved. V2 completed with no failed training seeds or service restarts. Missing values are imputed from training statistics; existing small-IQR fallback 1 is retained for M1 channels 64/65 and M2 channels 27/26.

Fresh calibration selects corrected whole-system Anomaly+Rare Event F0.5 under a 1% calibration sample-alarm cap. The cap does not constrain assessment false-alarm duration. Historical source models instead retain the earlier selected 40–47 calibration rules. Scores take the maximum reconstruction error over model inputs, so expanded inputs also expand score sources. Overlapping reconstruction windows use context within the window; this is retrospective scoring, not demonstrated causal online anomaly detection.

## Channel meaning and transfer limitations

Public physical channel identities and units are anonymized. We cannot identify temperature or power sensors from correlation alone. The fixed mapping uses compatible anonymous groups and training behavior; none of its 13 assignments is stable across all four leave-season-out checks. It is an experimental correspondence, not established physical equivalence. Each mission has independently shifted/scaled time.

The nine historical Mission 1 context-arm models are assessed under unchanged source scaling and under target-training normalization fitted only on Mission 2 year 2000. Both preserve weights, mapping and source decision rules. Target normalization is target-data adaptation, not strict zero-shot transfer. Neither condition tunes a target threshold.

The old Mission 2 eight/eleven-input panels represented annotations for 14/198 training events, including 1/21 anomalies. New 13/26-input panels represent 192/198 and 198/198 events, respectively. Those are training annotation-membership counts, not actual detections or a limit on indirect detection. All system assessment IDs remain in the denominator.

## Pooled primary results

Primary events combine Anomaly and Rare Event. F0.5 means and sample standard deviations summarize all three seeds, not confidence intervals. TPR is detected event IDs divided by annotated IDs. Time FPR is false-alarm duration divided by nominal duration. Fixed-base scope means 13 representatives for fresh models, and the mapped original eight source inputs for transfer. Pooled metrics are recomputed continuously rather than averaged from years.

| Mission | Condition | Technique | Scope | F0.5 mean ± SD | Event TPR mean | Time FPR mean | Detected IDs, seeds 42 /43 /44 | Total IDs |
|---|---|---|---|---:|---:|---:|---|---:|
| M2 | fresh_training | Short-no-roll | system_all_channels | 0.719673 ± 0.216631 | 40.79% | 0.1999% | 203 / 183 / 57 | 362 |
| M2 | fresh_training | Short-no-roll | fixed_base_channels | 0.721180 ± 0.216586 | 41.02% | 0.1999% | 203 / 183 / 57 | 360 |
| M2 | fresh_training | Masked + short-no-roll | system_all_channels | 0.635863 ± 0.217761 | 34.53% | 0.0580% | 56 / 248 / 71 | 362 |
| M2 | fresh_training | Masked + short-no-roll | fixed_base_channels | 0.637390 ± 0.217542 | 34.72% | 0.0580% | 56 / 248 / 71 | 360 |
| M2 | fresh_training | Expanded-input short-no-roll | system_all_channels | 0.789561 ± 0.012406 | 44.11% | 0.1234% | 154 / 157 / 168 | 362 |
| M2 | fresh_training | Expanded-input short-no-roll | fixed_base_channels | 0.791127 ± 0.012387 | 44.35% | 0.1234% | 154 / 157 / 168 | 360 |
| M1 | fresh_training | Short-no-roll | system_all_channels | 0.526498 ± 0.362934 | 68.86% | 2.9921% | 64 / 62 / 62 | 91 |
| M1 | fresh_training | Short-no-roll | fixed_base_channels | 0.526498 ± 0.362934 | 68.86% | 2.9921% | 64 / 62 / 62 | 91 |
| M1 | fresh_training | Masked + short-no-roll | system_all_channels | 0.589682 ± 0.136476 | 56.78% | 2.0892% | 59 / 32 / 64 | 91 |
| M1 | fresh_training | Masked + short-no-roll | fixed_base_channels | 0.589682 ± 0.136476 | 56.78% | 2.0892% | 59 / 32 / 64 | 91 |
| M1 | fresh_training | Expanded-input short-no-roll | system_all_channels | 0.690503 ± 0.018601 | 66.30% | 0.7047% | 69 / 49 / 63 | 91 |
| M1 | fresh_training | Expanded-input short-no-roll | fixed_base_channels | 0.690503 ± 0.018601 | 66.30% | 0.7047% | 69 / 49 / 63 | 91 |
| M2 | frozen_source_scaling | Short-no-roll | system_all_channels | 0.000000 ± 0.000000 | 100.00% | 100.0000% | 362 / 362 / 362 | 362 |
| M2 | frozen_source_scaling | Short-no-roll | fixed_base_channels | 0.000000 ± 0.000000 | 100.00% | 100.0000% | 28 / 28 / 28 | 28 |
| M2 | target_training_normalization | Short-no-roll | system_all_channels | 0.081592 ± 0.043700 | 81.40% | 76.6060% | 326 / 212 / 346 | 362 |
| M2 | target_training_normalization | Short-no-roll | fixed_base_channels | 0.010781 ± 0.004620 | 88.10% | 76.6060% | 26 / 20 / 28 | 28 |
| M2 | frozen_source_scaling | Masked + short-no-roll | system_all_channels | 0.000000 ± 0.000000 | 100.00% | 100.0000% | 362 / 362 / 362 | 362 |
| M2 | frozen_source_scaling | Masked + short-no-roll | fixed_base_channels | 0.000000 ± 0.000000 | 100.00% | 100.0000% | 28 / 28 / 28 | 28 |
| M2 | target_training_normalization | Masked + short-no-roll | system_all_channels | 0.029058 ± 0.041007 | 95.95% | 95.9920% | 361 / 356 / 325 | 362 |
| M2 | target_training_normalization | Masked + short-no-roll | fixed_base_channels | 0.005906 ± 0.006719 | 98.81% | 95.9920% | 28 / 28 / 27 | 28 |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | system_all_channels | 0.000000 ± 0.000000 | 100.00% | 100.0000% | 362 / 362 / 362 | 362 |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | fixed_base_channels | 0.000000 ± 0.000000 | 100.00% | 100.0000% | 28 / 28 / 28 | 28 |
| M2 | target_training_normalization | Expanded-input short-no-roll | system_all_channels | 0.047814 ± 0.045672 | 83.61% | 81.0515% | 208 / 359 / 341 | 362 |
| M2 | target_training_normalization | Expanded-input short-no-roll | fixed_base_channels | 0.006716 ± 0.004098 | 88.10% | 81.0515% | 18 / 28 / 28 | 28 |

## Every seed, scope and event definition

Anomaly-only reuses identical predictions and treats rare-event intervals as neutral for false alarms, following the corrected evaluator. It does not establish that the model recognizes rare events as normal.

| Mission | Condition | Technique | Seed | Definition | Scope | F0.5 | Detected / total | Event TPR | Time FPR |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| M2 | fresh_training | Short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.859942 | 203 / 362 | 56.08% | 0.2825% |
| M2 | fresh_training | Short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.861402 | 203 / 360 | 56.39% | 0.2825% |
| M2 | fresh_training | Short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.2825% |
| M2 | fresh_training | Short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.2825% |
| M2 | fresh_training | Short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.828905 | 183 / 362 | 50.55% | 0.2514% |
| M2 | fresh_training | Short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.830410 | 183 / 360 | 50.83% | 0.2514% |
| M2 | fresh_training | Short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.2514% |
| M2 | fresh_training | Short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.2514% |
| M2 | fresh_training | Short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.470173 | 57 / 362 | 15.75% | 0.0657% |
| M2 | fresh_training | Short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.471729 | 57 / 360 | 15.83% | 0.0657% |
| M2 | fresh_training | Short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.0657% |
| M2 | fresh_training | Short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.0657% |
| M2 | fresh_training | Masked + short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.474553 | 56 / 362 | 15.47% | 0.0125% |
| M2 | fresh_training | Masked + short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.476167 | 56 / 360 | 15.56% | 0.0125% |
| M2 | fresh_training | Masked + short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.0125% |
| M2 | fresh_training | Masked + short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.0125% |
| M2 | fresh_training | Masked + short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.883563 | 248 / 362 | 68.51% | 0.1353% |
| M2 | fresh_training | Masked + short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.884824 | 248 / 360 | 68.89% | 0.1353% |
| M2 | fresh_training | Masked + short-no-roll | 43 | anomaly_only | system_all_channels | 0.086102 | 1 / 6 | 16.67% | 0.1353% |
| M2 | fresh_training | Masked + short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.089174 | 1 / 4 | 25.00% | 0.1353% |
| M2 | fresh_training | Masked + short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.549472 | 71 / 362 | 19.61% | 0.0262% |
| M2 | fresh_training | Masked + short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.551179 | 71 / 360 | 19.72% | 0.0262% |
| M2 | fresh_training | Masked + short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.0262% |
| M2 | fresh_training | Masked + short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.0262% |
| M2 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.777308 | 154 / 362 | 42.54% | 0.0951% |
| M2 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.778881 | 154 / 360 | 42.78% | 0.0951% |
| M2 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.0951% |
| M2 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.0951% |
| M2 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.789259 | 157 / 362 | 43.37% | 0.0954% |
| M2 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.790850 | 157 / 360 | 43.61% | 0.0954% |
| M2 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.0954% |
| M2 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.0954% |
| M2 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.802115 | 168 / 362 | 46.41% | 0.1797% |
| M2 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.803650 | 168 / 360 | 46.67% | 0.1797% |
| M2 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 0 / 6 | 0.00% | 0.1797% |
| M2 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000000 | 0 / 4 | 0.00% | 0.1797% |
| M1 | fresh_training | Short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.142052 | 64 / 91 | 70.33% | 7.8498% |
| M1 | fresh_training | Short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.142052 | 64 / 91 | 70.33% | 7.8498% |
| M1 | fresh_training | Short-no-roll | 42 | anomaly_only | system_all_channels | 0.085913 | 36 / 55 | 65.45% | 7.8498% |
| M1 | fresh_training | Short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.085913 | 36 / 55 | 65.45% | 7.8498% |
| M1 | fresh_training | Short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.863194 | 62 / 91 | 68.13% | 0.0490% |
| M1 | fresh_training | Short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.863194 | 62 / 91 | 68.13% | 0.0490% |
| M1 | fresh_training | Short-no-roll | 43 | anomaly_only | system_all_channels | 0.796815 | 33 / 55 | 60.00% | 0.0490% |
| M1 | fresh_training | Short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.796815 | 33 / 55 | 60.00% | 0.0490% |
| M1 | fresh_training | Short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.574249 | 62 / 91 | 68.13% | 1.0774% |
| M1 | fresh_training | Short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.574249 | 62 / 91 | 68.13% | 1.0774% |
| M1 | fresh_training | Short-no-roll | 44 | anomaly_only | system_all_channels | 0.426828 | 33 / 55 | 60.00% | 1.0774% |
| M1 | fresh_training | Short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.426828 | 33 / 55 | 60.00% | 1.0774% |
| M1 | fresh_training | Masked + short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.584954 | 59 / 91 | 64.84% | 6.1244% |
| M1 | fresh_training | Masked + short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.584954 | 59 / 91 | 64.84% | 6.1244% |
| M1 | fresh_training | Masked + short-no-roll | 42 | anomaly_only | system_all_channels | 0.461502 | 33 / 55 | 60.00% | 6.1244% |
| M1 | fresh_training | Masked + short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.461502 | 33 / 55 | 60.00% | 6.1244% |
| M1 | fresh_training | Masked + short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.455631 | 32 / 91 | 35.16% | 0.0621% |
| M1 | fresh_training | Masked + short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.455631 | 32 / 91 | 35.16% | 0.0621% |
| M1 | fresh_training | Masked + short-no-roll | 43 | anomaly_only | system_all_channels | 0.387262 | 21 / 55 | 38.18% | 0.0621% |
| M1 | fresh_training | Masked + short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.387262 | 21 / 55 | 38.18% | 0.0621% |
| M1 | fresh_training | Masked + short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.728460 | 64 / 91 | 70.33% | 0.0811% |
| M1 | fresh_training | Masked + short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.728460 | 64 / 91 | 70.33% | 0.0811% |
| M1 | fresh_training | Masked + short-no-roll | 44 | anomaly_only | system_all_channels | 0.618150 | 36 / 55 | 65.45% | 0.0811% |
| M1 | fresh_training | Masked + short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.618150 | 36 / 55 | 65.45% | 0.0811% |
| M1 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.697435 | 69 / 91 | 75.82% | 0.9092% |
| M1 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.697435 | 69 / 91 | 75.82% | 0.9092% |
| M1 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_only | system_all_channels | 0.585470 | 40 / 55 | 72.73% | 0.9092% |
| M1 | fresh_training | Expanded-input short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.585470 | 40 / 55 | 72.73% | 0.9092% |
| M1 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.704644 | 49 / 91 | 53.85% | 0.2702% |
| M1 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.704644 | 49 / 91 | 53.85% | 0.2702% |
| M1 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_only | system_all_channels | 0.567584 | 24 / 55 | 43.64% | 0.2702% |
| M1 | fresh_training | Expanded-input short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.567584 | 24 / 55 | 43.64% | 0.2702% |
| M1 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.669432 | 63 / 91 | 69.23% | 0.9348% |
| M1 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.669432 | 63 / 91 | 69.23% | 0.9348% |
| M1 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_only | system_all_channels | 0.535511 | 34 / 55 | 61.82% | 0.9348% |
| M1 | fresh_training | Expanded-input short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.535511 | 34 / 55 | 61.82% | 0.9348% |
| M2 | frozen_source_scaling | Short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.065767 | 326 / 362 | 90.06% | 86.4060% |
| M2 | target_training_normalization | Short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.008321 | 26 / 28 | 92.86% | 86.4060% |
| M2 | target_training_normalization | Short-no-roll | 42 | anomaly_only | system_all_channels | 0.001998 | 6 / 6 | 100.00% | 86.4060% |
| M2 | target_training_normalization | Short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000672 | 2 / 2 | 100.00% | 86.4060% |
| M2 | frozen_source_scaling | Short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.130999 | 212 / 362 | 58.56% | 50.4763% |
| M2 | target_training_normalization | Short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.016111 | 20 / 28 | 71.43% | 50.4763% |
| M2 | target_training_normalization | Short-no-roll | 43 | anomaly_only | system_all_channels | 0.002480 | 3 / 6 | 50.00% | 50.4763% |
| M2 | target_training_normalization | Short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.000830 | 1 / 2 | 50.00% | 50.4763% |
| M2 | frozen_source_scaling | Short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.048010 | 346 / 362 | 95.58% | 92.9357% |
| M2 | target_training_normalization | Short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.007912 | 28 / 28 | 100.00% | 92.9357% |
| M2 | target_training_normalization | Short-no-roll | 44 | anomaly_only | system_all_channels | 0.001826 | 6 / 6 | 100.00% | 92.9357% |
| M2 | target_training_normalization | Short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000617 | 2 / 2 | 100.00% | 92.9357% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Masked + short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.000000 | 361 / 362 | 99.72% | 100.0000% |
| M2 | target_training_normalization | Masked + short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Masked + short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Masked + short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Masked + short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.011209 | 356 / 362 | 98.34% | 98.9699% |
| M2 | target_training_normalization | Masked + short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.004503 | 28 / 28 | 100.00% | 98.9699% |
| M2 | target_training_normalization | Masked + short-no-roll | 43 | anomaly_only | system_all_channels | 0.001332 | 6 / 6 | 100.00% | 98.9699% |
| M2 | target_training_normalization | Masked + short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.000477 | 2 / 2 | 100.00% | 98.9699% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Masked + short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Masked + short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.075964 | 325 / 362 | 89.78% | 89.0060% |
| M2 | target_training_normalization | Masked + short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.013215 | 27 / 28 | 96.43% | 89.0060% |
| M2 | target_training_normalization | Masked + short-no-roll | 44 | anomaly_only | system_all_channels | 0.002662 | 5 / 6 | 83.33% | 89.0060% |
| M2 | target_training_normalization | Masked + short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.001078 | 2 / 2 | 100.00% | 89.0060% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 42 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 42 | anomaly_and_rare | system_all_channels | 0.093525 | 208 / 362 | 57.46% | 50.0278% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 42 | anomaly_and_rare | fixed_base_channels | 0.009714 | 18 / 28 | 64.29% | 50.0278% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 42 | anomaly_only | system_all_channels | 0.002190 | 4 / 6 | 66.67% | 50.0278% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 42 | anomaly_only | fixed_base_channels | 0.000549 | 1 / 2 | 50.00% | 50.0278% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 43 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 43 | anomaly_and_rare | system_all_channels | 0.002181 | 359 / 362 | 99.17% | 99.8245% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 43 | anomaly_and_rare | fixed_base_channels | 0.002047 | 28 / 28 | 100.00% | 99.8245% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 43 | anomaly_only | system_all_channels | 0.001645 | 6 / 6 | 100.00% | 99.8245% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 43 | anomaly_only | fixed_base_channels | 0.001097 | 2 / 2 | 100.00% | 99.8245% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.000000 | 362 / 362 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.000000 | 28 / 28 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 44 | anomaly_only | system_all_channels | 0.000000 | 6 / 6 | 100.00% | 100.0000% |
| M2 | frozen_source_scaling | Expanded-input short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000000 | 2 / 2 | 100.00% | 100.0000% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 44 | anomaly_and_rare | system_all_channels | 0.047737 | 341 / 362 | 94.20% | 93.3022% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 44 | anomaly_and_rare | fixed_base_channels | 0.008388 | 28 / 28 | 100.00% | 93.3022% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 44 | anomaly_only | system_all_channels | 0.001635 | 5 / 6 | 83.33% | 93.3022% |
| M2 | target_training_normalization | Expanded-input short-no-roll | 44 | anomaly_only | fixed_base_channels | 0.000662 | 2 / 2 | 100.00% | 93.3022% |

## Paired changes

The following fresh-model contrasts pair the same mission, seed, period and scope with short-no-roll. Expanded inputs alter both information and score sources; the masked contrast changes the training procedure. Full paired tables retain both scopes, both definitions and all periods. Transfer contrasts are descriptive comparisons between historical source methods.

| Mission | Technique | Seed | Scope | Primary Δ F0.5 | Δ detected IDs | Δ time FPR (percentage points) |
|---|---|---:|---|---:|---:|---:|
| M1 | Masked + short-no-roll | 42 | fixed_base_channels | +0.442902 | -5 | -1.72540 |
| M1 | Expanded-input short-no-roll | 42 | fixed_base_channels | +0.555382 | +5 | -6.94068 |
| M1 | Masked + short-no-roll | 42 | system_all_channels | +0.442902 | -5 | -1.72540 |
| M1 | Expanded-input short-no-roll | 42 | system_all_channels | +0.555382 | +5 | -6.94068 |
| M1 | Masked + short-no-roll | 43 | fixed_base_channels | -0.407563 | -30 | +0.01309 |
| M1 | Expanded-input short-no-roll | 43 | fixed_base_channels | -0.158550 | -13 | +0.22118 |
| M1 | Masked + short-no-roll | 43 | system_all_channels | -0.407563 | -30 | +0.01309 |
| M1 | Expanded-input short-no-roll | 43 | system_all_channels | -0.158550 | -13 | +0.22118 |
| M1 | Masked + short-no-roll | 44 | fixed_base_channels | +0.154212 | +2 | -0.99625 |
| M1 | Expanded-input short-no-roll | 44 | fixed_base_channels | +0.095183 | +1 | -0.14255 |
| M1 | Masked + short-no-roll | 44 | system_all_channels | +0.154212 | +2 | -0.99625 |
| M1 | Expanded-input short-no-roll | 44 | system_all_channels | +0.095183 | +1 | -0.14255 |
| M2 | Masked + short-no-roll | 42 | fixed_base_channels | -0.385234 | -147 | -0.27000 |
| M2 | Expanded-input short-no-roll | 42 | fixed_base_channels | -0.082521 | -49 | -0.18741 |
| M2 | Masked + short-no-roll | 42 | system_all_channels | -0.385389 | -147 | -0.27000 |
| M2 | Expanded-input short-no-roll | 42 | system_all_channels | -0.082634 | -49 | -0.18741 |
| M2 | Masked + short-no-roll | 43 | fixed_base_channels | +0.054414 | +65 | -0.11615 |
| M2 | Expanded-input short-no-roll | 43 | fixed_base_channels | -0.039560 | -26 | -0.15606 |
| M2 | Masked + short-no-roll | 43 | system_all_channels | +0.054657 | +65 | -0.11615 |
| M2 | Expanded-input short-no-roll | 43 | system_all_channels | -0.039646 | -26 | -0.15606 |
| M2 | Masked + short-no-roll | 44 | fixed_base_channels | +0.079450 | +14 | -0.03951 |
| M2 | Expanded-input short-no-roll | 44 | fixed_base_channels | +0.331921 | +111 | +0.11401 |
| M2 | Masked + short-no-roll | 44 | system_all_channels | +0.079300 | +14 | -0.03951 |
| M2 | Expanded-input short-no-roll | 44 | system_all_channels | +0.331942 | +111 | +0.11401 |

## Historical comparison

Previous Mission 2 pooled whole-system primary means were 0.137489 short-no-roll,0.218775 masked and 0.189472 expanded. Those models used 8/11 inputs, selected-eight calibration, batch 64 and slightly different training-window preparation. The new campaign uses 13/26 inputs, whole-system calibration and batch 32. Comparisons are descriptive; the change cannot be attributed solely to added channels. The previous Mission 1 context-arm whole-system means were 0.844667/0.853041/0.804662 in short-no-roll/masked/expanded order, all above the new means. Mission 1 also changes from historical restored prepared data to raw causal preparation and uses broader groups. The old selected 40–47 scope is not the new 13-input scope. Historical rows and original results are preserved.

## Independent verification and limits

The audit checked 778 completion-file hashes, all 183 raw fingerprints, frozen source/runtime/settings, all 18 terminal histories, independent scaler refits, all 120 calibration grids (11880 candidate rows), every chosen rule/checkpoint, and 1560 exact float32 inference spots across 156 checkpoint/condition passes. Full prediction replay matches all 720 annual/pooled metric rows and 32814 detected/missed ID records. Independent pandas source lookup checked 1361191 value/age/availability/mask samples across all eight prepared splits, including every assessment preparation. The 203-test suite passed, including bundled-reference event-metric parity. Neural/raw-source replay is sampled; full file hashes are checked. Corrected metric code is reused; postprocessing, rule ranking, windows and ID enumeration are independently implemented.

Only three neural seeds, reused benchmark years, anonymized physical meaning, unstable transfer mapping, changed preparation/selection/batch, increased score sources, and additional compute limit interpretation. No robust thesis F0.5 ≥0.85 achievement, independent generalization, or deployment claim follows from these results.

[All720 metric rows](tables/cross-mission-2026-11-20-v2/verified_results.csv), [detected/missed IDs](tables/cross-mission-2026-11-20-v2/verified_event_ids.csv), [paired changes](tables/cross-mission-2026-11-20-v2/paired_changes.csv), [seed summaries](tables/cross-mission-2026-11-20-v2/seed_summary.csv), [audit](tables/cross-mission-2026-11-20-v2/verification.json).

[Approved protocol](cross-mission-whole-system-protocol-v2-2026-11-20.md), [channel research](cross-mission-channel-research-2026-11-18.md), [W&B evaluation links](tables/cross-mission-2026-11-20-v2/wandb-evaluation.json), [essential artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/cross-mission-whole-system-2026-11-25).
