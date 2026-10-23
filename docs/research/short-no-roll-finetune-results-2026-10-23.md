# Short-no-roll fine-tuning results — 23 October 2026

Ten additional training epochs did not improve calibration. The original checkpoint and alarm rule were retained, and fresh assessment predictions exactly reproduce the original predictions. No improvement is claimed.

Training uses channels 40–47 and globally nominal windows from 2000–2005, with the original short-no-roll synthetic augmentation objective. The original calibration-selected epoch10 checkpoint was continued for ten epochs at initial learning rate 1e-5, one tenth of its original rate. AdamW and cosine scheduler were reset; seed42 and batch64 were fixed. All decisions used 2006 calibration. The original checkpoint was an eligible fallback, with original/earliest winning ties. No changes were selected from assessment results.

## Calibration selection

| Continuation epoch | Corrected system event F0.5 | Selected |
|---|---:|---|
| 0 | 0.882138246960001 | Yes |
| 5 | 0.882118655982629 | No |
| 10 | 0.882099064642808 | No |

Training loss fell from 0.839924 to 0.773839, but calibration event F0.5 did not improve. Training loss and the event-selection metric measure different objectives. The unsuccessful epoch5 and epoch10 candidate grids and training history are preserved. Unselected models were not assessed on 2007–2013.

## Annual and pooled assessment

Original and selected results are identical, so the following tables show both without duplicating columns. The CSV retains all 32 method/period/scope records.

### system_all_channels

| Period | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---:|---:|---:|---:|---:|
| 2007 | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916357 |
| 2008 | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853372 |
| 2009 | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739 |
| 2010 | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940 |
| 2011 | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850982 |
| 2012 | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666620 |
| 2013 | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422 |
| 2007-2013 | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070 |

### input_channels_40_47

| Period | Detected / total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---:|---:|---:|---:|---:|
| 2007 | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964569 |
| 2008 | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945594 |
| 2009 | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763 |
| 2010 | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749914 |
| 2011 | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930135 |
| 2012 | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624956 |
| 2013 | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975430 |
| 2007-2013 | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907140 |

## Verification and limitations

Independent verification passed frozen data/source/protocol/reference hashes, completion and training receipts, ten-epoch history, candidate-grid selection, exact selected-weight/scaler equality to the original checkpoint, binary prediction replay, and all32 corrected annual/pooled metric rows recomputed from raw annotations. No event IDs were gained or lost. The thesis suite passed141 tests, including interruption recovery and original fallback.

Precision uses the corrected ESA event definition with nominal-time penalty. TPR measures event-ID detection; time FPR measures alarm duration over nominal duration. Annual event counts can include IDs spanning year boundaries; pooled scores are recomputed, not averaged. One seed and a repeatedly inspected benchmark limit generalization claims. Additional training compute did not produce a selected model improvement. This is not independent validation or robust achievement of the thesis target.

[Full metrics](tables/short-no-roll-finetune-2026-10-23/verified_results.csv), [detected and missed event IDs](tables/short-no-roll-finetune-2026-10-23/event_ids.csv), [verification](tables/short-no-roll-finetune-2026-10-23/verification.json), [frozen protocol](short-no-roll-finetune-protocol-2026-10-23.md).

[Artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/short-no-roll-finetune-2026-10-23). Download checksum evidence is in github-backup.json beside the verified tables. This additive backup does not replace the full external-drive copy or certify unrelated migration contents.
