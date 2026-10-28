# Temporal post-processing results — 28 October 2026

Three fixed models were evaluated with gap merging followed by short-segment removal. Weights, scalers, channels and thresholds were unchanged. Training2000–2005, calibration2006, assessment2007–2013. Inputs40–47 for original short-no-roll and masked pretrain/fine-tune; extra inputs14,21,29 only for expanded short-no-roll. The latter is the latest W&B-selected LR3e-5 epoch10 model.

## Calibration selection

Each method used48 candidates: gaps0,1,2,4,8,16,32,64 samples and minimum durations1,2,4,8,16,32 samples, at30seconds per sample. Internal gaps are filled first; short merged segments are then removed. Duration includes filled gaps. This is offline processing with lookahead, not an online detection claim. Earlier studies included a coarser grid; this is a bounded temporal refinement, not a new detector.

Selection maximizes channels-40–47 calibration F0.5 subject to retention of the exact baseline-detected calibration target IDs and a1% total alarm fraction cap. Ties favor lower nominal alarm duration, then unchanged processing, then smaller gap and duration. System metrics are secondary. All choices froze before assessment. Retaining calibration detections does not guarantee retaining assessment detections.

| Technique | Merge gap (minutes) | Minimum duration (minutes) | Local calibration F0.5 | Unchanged |
|---|---:|---:|---:|---|
| short_no_roll | 0 | 0.5 | 0.999724246 | True |
| pretrain_finetune | 0 | 0.5 | 0.882105852 | True |
| expanded_short_no_roll | 0 | 0.5 | 0.999727433 | True |

The selected rules remain gap0 and duration1 for all three models: no assessment improvement is claimed. All47 modified candidates per method satisfied the retention/cap constraints, but none improved calibration F0.5. Each baseline calibration segment was at least16minutes long, so no tested minimum-duration cutoff removed any baseline segment. Minimum baseline gaps were288minutes (short-no-roll),304minutes (masked pretrain/fine-tune), and16minutes (expanded model). Thus gap filling could affect only the expanded model within the tested0–32minute range, without improving the selection objective. This is a narrow-range negative result, not evidence against all larger cutoffs or all temporal post-processing. Larger ranges were not assessed.

## Annual and pooled assessment

TPR is event-ID recall; precision includes the corrected ESA nominal-time penalty. Time FPR is nominal false-alarm duration divided by nominal duration. Non-target annotations remain neutral. Pooled metrics are recomputed over continuous predictions; annual counts can repeat IDs spanning years.

### short_no_roll: system_all_channels

| Period | Variant | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |
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

### short_no_roll: input_channels_40_47

| Period | Variant | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |
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

### pretrain_finetune: system_all_channels

| Period | Variant | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |
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

### pretrain_finetune: input_channels_40_47

| Period | Variant | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |
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

### expanded_short_no_roll: system_all_channels

| Period | Variant | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519 |
| 2007 | selected | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519 |
| 2008 | original | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942782 |
| 2008 | selected | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942782 |
| 2009 | original | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099 |
| 2009 | selected | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099 |
| 2010 | original | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936993 |
| 2010 | selected | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936993 |
| 2011 | original | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932830 |
| 2011 | selected | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932830 |
| 2012 | original | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773 |
| 2012 | selected | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773 |
| 2013 | original | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | selected | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2007-2013 | original | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573 |
| 2007-2013 | selected | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573 |

### expanded_short_no_roll: input_channels_40_47

| Period | Variant | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848 |
| 2007 | selected | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848 |
| 2008 | original | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328 |
| 2008 | selected | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328 |
| 2009 | original | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716 |
| 2009 | selected | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716 |
| 2010 | original | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999423 |
| 2010 | selected | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999423 |
| 2011 | original | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692 |
| 2011 | selected | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692 |
| 2012 | original | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499811 |
| 2012 | selected | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499811 |
| 2013 | original | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2013 | selected | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381 |
| 2007-2013 | original | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830900 |
| 2007-2013 | selected | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830900 |

## Operational segments and duration

Segment counts are alarm components, not distinct detected annotation IDs. Grid duration counts positive samples times30seconds; corrected evaluator duration uses timestamp intervals, so a final boundary can differ by one sample. Merging can reduce components while increasing false-alarm duration. Fewer segments alone do not establish higher F0.5.

| Technique | Variant | Alarm segments | Grid alarm hours | Nominal false-alarm hours |
|---|---|---:|---:|---:|
| short_no_roll | original | 62 | 957.8667 | 15.9278 |
| short_no_roll | selected | 62 | 957.8667 | 15.9278 |
| pretrain_finetune | original | 84 | 1095.4667 | 60.1161 |
| pretrain_finetune | selected | 84 | 1095.4667 | 60.1161 |
| expanded_short_no_roll | original | 95 | 5404.0000 | 3224.3839 |
| expanded_short_no_roll | selected | 95 | 5404.0000 | 3224.3839 |

## Event-ID changes

- short_no_roll, system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_121, id_126, id_127, id_131, id_135, id_137, id_138, id_139, id_14, id_140, id_141, id_143, id_146, id_147, id_148, id_149, id_15, id_150, id_152, id_156, id_161, id_162, id_169, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_47, id_49, id_51, id_55.
- short_no_roll, input_channels_40_47: gained none; lost none; missed id_11, id_121, id_138, id_139, id_14, id_140, id_147, id_149, id_15, id_150, id_169, id_174, id_20, id_24, id_40, id_42, id_47, id_49, id_51, id_55.
- pretrain_finetune, system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_42, id_43, id_55.
- pretrain_finetune, input_channels_40_47: gained none; lost none; missed id_11, id_157, id_174, id_20, id_24, id_42, id_55.
- expanded_short_no_roll, system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_short_no_roll, input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.

## Verification and limitations

All144 calibration candidates, selection, frozen hashes, prediction replay,96 metric rows, prior controls, event IDs and segment/duration diagnostics were independently verified.147 thesis tests passed. Original and selected outcomes are retained even when unsuccessful; assessment was not used to pick a new winner. One seed, repeatedly inspected benchmark, limited calibration events, unequal prior training compute and extra input channels limit generalization. No robust achievement of the thesis target is claimed.

[Metrics](tables/temporal-postprocessing-2026-10-28/verified_results.csv), [event IDs](tables/temporal-postprocessing-2026-10-28/event_ids.csv), [verification](tables/temporal-postprocessing-2026-10-28/verification.json), [protocol](temporal-postprocessing-protocol-2026-10-28.md).

## Published records

Three separate verified evaluation runs are available in [W&B](https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning?group=temporal-postprocessing-20261028).
The cloud summaries and all 96 annual/pooled table rows were downloaded and checked.
Artifacts are preserved in the [temporal-processing Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/temporal-postprocessing-2026-10-28).
The unchanged source models are dependencies from the [prior W&B study Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/wandb-finetuning-2026-10-25).
