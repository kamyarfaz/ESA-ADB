Satellite telemetry anomaly detection

Research progress for the meeting with Federico Butchellato

Kamyar Faz | 29 September 2026

Reporting period: 23-29 September 2026. The previous meeting is inferred to be Tuesday 22 September from the dated project records; it is not independently confirmed. Earlier work is included as background.

### Main conclusion

We now have a clearer picture of the data, a reproducible Transformer baseline, and controlled evidence about which modifications help or fail. The retained full-period model scores **0.8057 system F0.5** and **0.8629 on events annotated on its input channels**. The system target of 0.85 has not been robustly reached.

The latest short coordinated pseudo-anomaly method improves average development results in 2003 and 2005, but deteriorates in 2004. Repeating the pilot with three random seeds exposed this inconsistency. We should investigate the 2004 training/calibration behavior before spending time on another long run.

### What was completed since the previous meeting

- Audited all available Mission 1 years and input channels, with annual statistics and telemetry plots.

- Reviewed preprocessing, rare changes, scaling and correlations using permitted training data.

- Reported local and system detection from exactly the same frozen predictions.

- Completed full-period baseline training and verified an exact repeat of saved outputs.

- Tested normalization, masking, timing context, calibration rules and short coordinated augmentation; preserved negative results.

- Verified all three seeds of the latest pilot: 18 prediction arrays and 36 local/system metric rows.

### How to use this report

Pages 2-6 explain the data and method. Pages 7-13 present results and diagnostic plots. Pages 14-16 give limitations, next steps, a speaking summary and evidence references. The accompanying workbook contains calibration and assessment metrics for every latest seed/year/model/scope.


---


## 1  Data coverage and investigation

The available prepared Mission 1 data spans **14 complete years, 2000-2013**. The January 1, 2014 boundary row is not an additional year. The original benchmark separates 2000-2006 training data from 2007-2013 assessment data; our experiments further separate calibration from training.

| Data layer | Contents | Meaning |

| --- | --- | --- |

| Raw telemetry | 76 channels | Time-stamped measurements with anonymized metadata |

| Raw telecommands | 698 command types | Occurrence streams; constant payloads can still carry timing information |

| Prepared model input pool | 87 inputs | 76 telemetry plus 11 selected command inputs |

| Current model | 8 channels | Channels 40, 41, 42, 43, 44, 45, 46, 47 |

The full descriptive scan produced 10,836 input/year records and 12,768 telemetry/month records, summarizing 744,856,895 native telemetry observations. A 10-page atlas covers all 76 telemetry channels. Statistics include counts, ranges, variation, active months, timestamp order, duplicate timestamps and non-finite values.

### What we learned

All 76 telemetry channels have observations in every year and vary somewhere over the full period. Some are constant during particular years: channels 8-11 in 2008, 8-10 in 2010, and 8-9 in 2013. Therefore a flat channel in one interval is not automatically useless.

Of the 698 command types, 681 occur at least once in 2000-2013 and 17 do not occur in that interval. No source channels were deleted. The scans found no non-finite values, duplicate timestamps or backward steps within the examined annual source records; that does not imply gap-free sensing.

### What remains unknown

Anonymized subsystem and physical-unit codes do not justify calling a channel temperature, voltage or another physical quantity. More data also does not automatically reduce accuracy: additional inputs may add useful coverage or noise. Their value must be tested under matched conditions.

The full-period audit is descriptive. Later years were inspected, so they cannot be described as untouched data. Feature selection and new transformations must be justified using the allowed training/calibration periods.

Evidence: [1] Full mission inventory; [2] Data-first plan. Personal working area holds small samples and manifests while original source data remains preserved.


---


## 2  Preprocessing and channel relationships

We traced the inherited preparation instead of assuming a finite CSV meant complete information. It uses a 30-second grid, held values, filling, and special treatment of counters. Some preparation depends on annotations for restoring short events; this limits claims about operational deployment.

Figure 1. Pearson correlation on all globally nominal prepared training samples in 2000-2005, restricted here to channels 40-47. High correlation is a relationship to investigate, not a deletion rule.

![Figure 1. Pearson correlation on all globally nominal prepared training samples in 2000-2005, restricted here to channels 40-47. High correlation is a relationship to investigate, not a deletion rule.](/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260929/plots/correlation40_47.png)

The corrected correlation audit covers all 76 telemetry channels and 2,850 pairs using 5,615,041 globally nominal samples out of 6,312,960 training rows. The nominal mask checks all 87 paired label columns. We computed pooled and yearly Pearson correlations, adjacent-change correlations, and an exploratory hourly Spearman matrix.

Stable groups at absolute Pearson correlation at least 0.95 in every training year include **15/22/23/31/39** and **16/24/32/33/40**. At 0.90, 14/21/29 is coherent and 41/44 form a pair. However, channels 40/47 have pooled correlation 0.8833 but only 0.2518 in 2001. The current eight inputs are not one uniformly correlated group.

Historical correlation experiments were inventoried. The old loader could miss paired labels, so its claimed nominal-only provenance is not established. New corrected matrices are stored separately; historical results were not silently relabeled.

Evidence: [3] Preprocessing and correlation policy; [4] Historical channel inventory. Figure generated from the saved full-resolution correlation matrix.


---


## 3  Scaling and information loss

### Scaling audit

The production scaler uses a median and interquartile range fitted to globally nominal training samples, with clipping at +/-10. Across the 2000-2005 training audit, no nominal values exceeded that clipping limit. Raising it would therefore not change those nominal inputs.

Twenty-nine prepared inputs have zero or near-zero IQR and use the scale-of-one fallback: 18 telemetry channels and all 11 selected commands. Zero IQR means the central half of values is flat; it does not imply the entire signal contains no information.

### Counter representation

| Training-only measurement | Finding |

| --- | --- |

| Native numeric changes on counters 4-11 | 5,340 changes |

| Changes whose old hold interval contains no 30-second grid point | 4,714 changes, or 88.3% |

| Causal change-bin candidate | Retains all 5,340 observed changes as counts |

| Largest added bin delay | 29.67 seconds |

| Largest native observation gap | 5 hours 57 minutes |

These are counts of observed channel changes, not anomaly-event recall. The diagnostic did not simulate annotation-dependent restoration. A causal candidate places a difference at its later observation and aggregates it into the next 30-second boundary; counts or absolute sums preserve activity that signed sums may cancel.

A bin with no recorded change is not necessarily healthy or observed. Observation counts and time since the last measurement are needed to distinguish inactivity from missing data. Native timestamps are assumed to be availability times; true packet arrival times are not provided.

### Decision

Preserve the original raw and prepared data. No broad replacement cleaning pipeline or automatic channel deletion was applied. Counter transformations would require a separate versioned experiment. Following the current research scope, model improvements remain focused on channels 40-47.

Evidence: [5] Scaling review; [6] Counter aggregation review. These audits support preprocessing decisions but do not themselves demonstrate improved F0.5.


---


## 4  Training and calibration methodology

The baseline already is a **Transformer autoencoder**. It reconstructs the observed multi-channel window. Large reconstruction error becomes an anomaly score; it is not a conventional classifier trained with a label for each anomaly type.

| Component | Current configuration |

| --- | --- |

| Inputs and window | 8 channels; 256 samples at 30 seconds = 128 minutes |

| Tokenization | 16-sample patches; 16 patches per channel |

| Attention | Temporal and channel encoders and decoders; 2 layers in each stack |

| Representation | 128-dimensional embeddings; 8 attention heads; dropout 0.10 |

| Optimization | AdamW, learning rate 0.0001, weight decay 0.0001; cosine schedule |

| Normal training examples | Only windows whose globally paired labels are all nominal |

| Score | Mean squared error per window/channel, overlap aggregation, maximum across channels |

Figure 2. Chronological splits. Each development assessment covers April-December, not all 12 months. Full-period assessment uses one frozen model and rule for all 2007-2013.

![Figure 2. Chronological splits. Each development assessment covers April-December, not all 12 months. Full-period assessment uses one frozen model and rule for all 2007-2013.](/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260929/plots/splits.png)

**Training** learns model weights and the scaler from preceding data. **Calibration** selects a checkpoint, threshold and alarm postprocessing using labeled events. **Assessment** measures the frozen decision on later data. We do not train on the same months whose performance is being assessed.

Full run: train 2000-2005, calibrate 2006, assess 2007-2013; 349,461 eligible nominal windows, 40 epochs. Latest pilot: preceding years, January-March calibration, April-December assessment; 25,000-window cap, five epochs, batch 64, seeds 42/43/44. Results from these budgets must not be treated as a single matched comparison.

Evidence: [7] Long baseline protocol; [8] Augmentation comparison protocol; models.py and saved run protocols.


---


## 5  Evaluation and operational interpretation

We use the corrected **ESA event F0.5**, metric version **esa-ew-id-duration-v1**. It evaluates annotation event IDs and time intervals, not pointwise classification accuracy. Anomaly and Rare Event categories are positive; excluded categories and out-of-scope annotations are neutral under the stated scope convention.

### Metric formula

Let TP<sub>e</sub> be detected true event IDs, FN<sub>e</sub> missed true event IDs, and FP<sub>e</sub> wholly false alarm events under the evaluator. Let T<sub>FP</sub> be alarm duration during nominal time and T<sub>N</sub> total nominal duration.

**P = [TP<sub>e</sub> / (TP<sub>e</sub> + FP<sub>e</sub>)] x [1 - T<sub>FP</sub> / T<sub>N</sub>]**

**R = TP<sub>e</sub> / (TP<sub>e</sub> + FN<sub>e</sub>)**

**F<sub>0.5</sub> = 1.25 P R / (0.25 P + R)**

F0.5 emphasizes precision. False alarms can erase gains in event recall. Zero wholly false alarm events does not guarantee perfect precision: alarms touching true events may extend into nominal time and incur the duration penalty. The implementation documents zero-denominator and zero-nominal-duration conventions.

### One prediction sequence and two scopes

| Scope | Question answered |

| --- | --- |

| Input channels 40-47 | Does the detector find events annotated on its eight observed channels? |

| System all channels | Can those same eight inputs detect annotated events anywhere in the spacecraft system? |

Both scores use exactly the same model, threshold and prediction sequence. System evaluation does not mean the neural network received all channels. Local scoring restricts event annotation intervals to the input channels while other annotations remain neutral. It is not a per-channel localization metric.

Calibration searches thresholds plus merge gaps and minimum alarm durations. The retained predicted-alarm fraction cap of 1% is our experimental policy, not a universal ESA rule. Earliest checkpoint wins an exact calibration F0.5 tie. The evaluator was checked against the pinned upstream event-wise implementation; this is not a claim of reproducing every hidden competition scoring component.

Evidence: [9] Corrected evaluation; [10] Dual-scope evaluation; evaluation/esa.py and parity tests.


---


## 6  Full-period baseline results

Figure 3. Annual system scores from one frozen model. The dotted line is the continuous pooled score; it is not the arithmetic mean of annual scores.

![Figure 3. Annual system scores from one frozen model. The dotted line is the continuous pooled score; it is not the arithmetic mean of annual scores.](/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260929/plots/full_period.png)

| Period and scope | F0.5 | TP / FP / FN | True IDs |

| --- | --- | --- | --- |

| 2006 calibration, local | 0.999737 | 6 / 0 / 0 | 6 |

| 2006 calibration, system | 0.882148 | 6 / 0 / 4 | 10 |

| 2007-2013 assessment, local | 0.862907 | 53 / 7 / 14 | 67 |

| 2007-2013 assessment, system | 0.805722 | 54 / 7 / 37 | 91 |

System precision is 0.884871 and recall is 0.593407. Of the 24 system IDs without input-channel annotations, one was detected and 23 were missed. This describes a coverage limitation, not proof that other-channel events are fundamentally undetectable through correlated inputs.

Calibration selected **epoch 10** out of 40. Epoch 40 calibration F0.5 fell to 0.357143, showing why lower training loss or more epochs is not automatically better detection. Assessment has no uncovered samples.

The user repeated the 40-epoch configuration. Saved history, rules, pooled/yearly results and assessment scores/predictions matched the earlier run by SHA256. This supports repeatability for that configuration; it does not establish reliability across seeds.

Earlier corrected channels 40-47 assessment was 0.761655. The later 0.805722 is numerically higher by 0.044067, but training window budget, calibration interval and checkpoint selection differ. We cannot attribute the increase solely to using more years. The local score above 0.85 must not replace the system target.

Evidence: [11] Long baseline results; [10] Dual scope; [12] Baseline repeat. Benchmark periods had already been inspected, so this is not an untouched test claim.


---


## 7  Controlled alternatives and negative results

Several plausible changes did not generalize. Each comparison below uses its own control and budget; controls from different experiments are not interchangeable.

| Experiment | Evidence | Decision |

| --- | --- | --- |

| Channel error normalization | 2006 calibration F0.5: 0.882148 to 0.882099; same 6 detected IDs; more nominal alarm duration | Retain original score |

| Masked reconstruction, 5 epochs | System F0.5 control to masked: 2003 .714286 to .908853; 2004 .594608 to .090409; 2005 .869116 to .869085 | Do not promote |

| Corruption overlaps hidden targets | 2004 masked control .090409 to .092948; FP48 to47; candidate calibration zero | Pause variant |

| Observation age and availability, 5 epochs | 2007-2013 system .689598 to .595655; TP remains28; FP0 to8 | Do not promote |

| Denser threshold grid and nominal-time cap | 2005 calibration .832880 to .936708, but assessment .869055 to .082397; FP0 to95 | Reject selected policy |

| Short coordinated augmentation | Three-seed matched pilot improves 2003/2005 means, worsens2004; details on pages11-13 | Investigate before scaling |

### What these experiments taught us

The recurring difficulty is maintaining low false-alarm rates when normal score distributions change. A higher calibration score can fail to transfer to later months. Very sparse calibration events also make checkpoint selection unstable.

Masking was investigated to reduce direct copying of anomalous inputs. Observation-age features addressed irregular native sampling and held values. Their negative results apply to the tested configurations and budgets; they do not prove that either idea can never help.

No alternative was selected separately for each assessment year after looking at the results. Failed outputs and protocols are preserved. The current reference remains the original full-period Transformer autoencoder.

Evidence: [13] Normalization; [14] Masked follow-up; [15] Overlap results; [16] Timing results; [17] Grid/cap assessment.


---


## 8  Reconstruction quality is not detection quality

We inspected six nominal 2005 calibration windows: one high-error window and one median-error window in each of January, February and March. Both saved models use channels 40-47 and exactly matching scalers; no new training was needed.

Figure 4. Existing saved pseudo-anomaly and reconstruction-only models on the same nominal windows. Windows were chosen using pseudo-model error, so the comparison is biased toward its difficult examples.

![Figure 4. Existing saved pseudo-anomaly and reconstruction-only models on the same nominal windows. Windows were chosen using pseudo-model error, so the comparison is biased toward its difficult examples.](/home/k_faz/projects/ESA-ADB/docs/research/tables/nominal-reconstruction-comparison-2026-09-26/dominant_channels.png)

On the high-error examples, the pseudo-anomaly model has maximum channel MSE around 10.4-12.7, while reconstruction-only errors are around 2-5 x 10<super>-6</super>. The pseudo model shows large offsets or oscillations on these examples. This motivates investigation of the objective, but six selected windows do not establish general superiority.

On the full 2005 calibration period, both models detect the same two system events and miss two: system F0.5 is about 0.8329 for each. The reconstruction-only model also reconstructs a missed anomaly extremely well. Therefore excellent reconstruction can mean copying the anomaly rather than detecting it.

The architecture gives decoder queries access to input tokens. The pseudo-anomaly loss tries to create separation between normal and corrupted windows. Its behavior on nominal inputs must remain stable; decreasing training MSE alone is an insufficient checkpoint criterion.

Evidence: [18] Nominal reconstruction comparison; [19] Calibration model comparison. Training sample/order identity was not verified for these older objective runs, so causal attribution is limited.


---


## 9  Short coordinated events motivated the new method

A detailed review of calibration event **id_98** corrected an important interpretation. Its system annotation extends across March 25-28,2005, but the union on input channels 41-46 totals only **nine minutes** on March 25.

| Local interval start UTC | Duration | Annotated inputs |

| --- | --- | --- |

| 03:59:55.134 | 5 minutes | 42 and45 |

| 04:36:55.134 | 30 seconds | 41-46 |

| 04:56:55.134 | 3 minutes30 seconds | 41-46;46 continues30 seconds longer |

Figure 5. Existing diagnostic around the brief coordinated interval. Physical cause is unknown; the anonymized traces and annotation support do not establish subsystem semantics.

![Figure 5. Existing diagnostic around the brief coordinated interval. Physical cause is unknown; the anonymized traces and annotation support do not establish subsystem semantics.](/home/k_faz/projects/ESA-ADB/docs/research/tables/id98-local-review-2026-09-26/interval_2.png)

The old synthetic generator corrupts 12-84 samples, or 6-42 minutes, on 1-4 channels. It does not directly include the shortest or widest annotated patterns seen here. A128 minute reconstruction window may dilute a brief event, but the earlier per-timestep scoring comparison was worse; simply changing aggregation is not an established fix.

The broader 2000-2005 inventory found 46 event IDs with 79 local episodes:32 episodes across 13 IDs shorter than 6 minutes,56 episodes across 40 IDs involving more than 4 simultaneous channels, and 16 episodes across 8 IDs satisfying both. This made the hypothesis broader than one chosen event. These are descriptive annotation episodes, not independent physical events.

Evidence: [20] id_98 local review; [21] Local event-shape inventory. Correlation did not establish a sustained relationship breakdown in this example.


---


## 10  Short coordinated augmentation experiment

The candidate retains the original generator for half the synthetic examples and uses short coordinated corruptions for the other half. The 50/50 mixture is a fixed experimental choice, not an optimized estimate of real anomaly frequency.

| Property | Original branch | New short branch |

| --- | --- | --- |

| Duration | 12-84samples;6-42minutes | 1-10samples;30seconds-5minutes |

| Channels affected | 1-4 of8 | 1-8 of8 |

| Types | Offset, temporal roll, noise | Offset, temporal roll, noise |

| Coordination | Original behavior preserved | Shared interval and type; coordinated sign, noise or shift |

| Magnitude scale | Segment standard deviation | Full-window population standard deviation |

| Special handling | Original code path | Length1 excludes temporal roll; no-op cases recorded |

### What was held fixed

Channels 40-47, Transformer architecture, nominal reconstruction objective, scaler, optimizer, scoring and threshold/postprocessing policy remain the same. Within a seed/fold pair, both arms use the same nominal windows, initial weights, shuffle and dropout streams. Corruption randomness is isolated so different generators do not shift the other random streams.

Both arms train for five epochs on at most 25,000 nominal windows with batch 64. Calibration is evaluated every epoch and the earliest best checkpoint is retained. Both rules are written before assessment. We repeated the exact protocol with seeds 42,43 and 44 on all three chronological folds.

### Training objective

Let e<sub>N, i</sub> be normal-window MSE and e<sub>A, i</sub> synthetic-window MSE. The implemented loss is the batch mean normal MSE plus the mean of **max(0, 10 - e<sub>A, i</sub> / (stop_gradient(mean(e<sub>N</sub>)) + 10<super>-8</super>))**. The penalty weight is 1. This encourages high error on synthetic examples while retaining accurate nominal reconstruction.

Input immutability, generator behavior, random-state restoration, decision freezing and CPU resume equivalence have regression tests. The last recorded full thesis suite passed 77 tests. All latest runs completed before the server shutdown; their saved results survived.

Evidence: [8] Comparison protocol; [22] Short augmentation implementation; saved protocols and histories. No claim of exact GPU determinism across hardware.


---


## 11  Results across three random seeds

Figure 6. Each line connects a matched legacy/candidate pair. Different colors are seeds, not channel sets. Development assessments are April-December; all use inputs40-47.

![Figure 6. Each line connects a matched legacy/candidate pair. Different colors are seeds, not channel sets. Development assessments are April-December; all use inputs40-47.](/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260929/plots/three_seeds.png)

| Assessment year | Legacy mean +/- SD | Candidate mean +/- SD |

| --- | --- | --- |

| 2003 | 0.642418 +/-0.171621 | 0.836292 +/-0.123532 |

| 2004 | 0.415934 +/-0.364273 | 0.241707 +/-0.123166 |

| 2005 | 0.848011 +/-0.036528 | 0.931322 +/-0.065321 |

These are arithmetic means and sample standard deviations over three seeds. They are not pooled event scores, confidence intervals, or independent final-test estimates. Seeds alter initialization and sampled training windows; years share expanding training histories.

The candidate improves six of nine seed/year pairs, declines in two, and ties in one. Win count alone is misleading: the 2004 seed 44 result falls from 0.832848 to 0.209270. On 2003 seed 44 the F0.5 increase is tiny and accompanies fewer detected events, showing the precision-recall tradeoff.

The local-scope mean for 2004 is 0.451907 for legacy and 0.249824 for the candidate. For 2003 the local means equal the system means; for 2005 local means are 0.826132 and 0.931322. A scope change must never be presented as an improvement to the original system target.

**Decision:** keep the existing baseline. Do not choose the best seed or launch a full 40 epoch candidate run solely because 2005 looks strong.

Evidence: [23] Three-seed review and full numeric workbook. All18saved prediction arrays reproduced exactly;36metric rows reproduced within absolute tolerance1e-9.


---


## 12  Why 2004 needs investigation

| Seed | Legacy F0.5 | New F0.5 | Legacy TP/FP/FN | New TP/FP/FN |

| --- | --- | --- | --- | --- |

| 42 | .255770 | .377845 | 4 /13 /6 | 5 /9 /5 |

| 43 | .159185 | .138006 | 3 /18 /7 | 6 /45 /4 |

| 44 | .832848 | .209270 | 5 /0 /5 | 6 /27 /4 |

In 2004 the candidate detects more system events for every seed, but false alarms increase sharply for seeds 43 and 44. Because F0.5 emphasizes precision, the extra detections do not compensate for these false alarms.

Figure 7. January-March2004 calibration histories. There is only one positive system event in this calibration interval; checkpoint selection therefore has very limited positive-event evidence.

![Figure 7. January-March2004 calibration histories. There is only one positive system event in this calibration interval; checkpoint selection therefore has very limited positive-event evidence.](/home/k_faz/projects/ESA-ADB/outputs/federico-meeting-20260929/plots/calibration2004.png)

Candidate seed 42 has zero calibration F0.5 at all five epochs, so the earliest-tie rule selects epoch 1. Legacy seed 43 also selects a zero-score checkpoint. This is valid under the recorded selection policy, but it does not provide a useful positive-event signal for ranking those checkpoints.

A zero calibration score does not imply that assessment will produce no alarms. Future scores can exceed the frozen calibration threshold, including during nominal periods. Conversely, a high calibration score on one event cannot establish broad anomaly coverage.

### Next diagnosis before training

Compare the 2004 training and calibration distributions across all seeds: nominal score tails by month, per-channel residual contributions, reconstruction examples and selected thresholds. Track nominal MSE and synthetic penalty separately. Identify whether the candidate changes nominal reconstruction or anomaly separation. Use these permitted periods to justify any new method; do not tune a threshold against the 27 or 45 assessment false alarms.

Evidence: [23] Saved pilot results and histories. This plot is a descriptive review of completed selection, not a new checkpoint search.


---


## 13  Research limits and next plan

### What is established

The data inventory now covers the whole available mission. Training-only correlation and preprocessing audits identify concrete information-loss and stability issues. The original full-period model is reproducible for its repeated configuration. Local and system performance are explicitly separated. Multiple attractive changes failed controlled comparisons, and the latest repeat seeds prevented premature promotion.

### What is not established

We have not robustly achieved system F0.5>=0.85. We have not fully identified the physical meaning of anonymized channels. We have not shown that fewer inputs are always better, that high correlation makes a channel redundant, or that a low reconstruction loss implies useful anomaly detection.

The AE and window aggregation are retrospective. A128 minute window and inherited filling/restoration policies do not establish real-time detection latency or flight suitability. The current work evaluates event detection under a research protocol, not a deployed spacecraft monitoring system.

Earlier assessment periods and later benchmark covariates/results have already been inspected. Repeated development decisions can adapt to them even when each individual threshold is frozen correctly. A future final claim needs an agreed evaluation plan and honest disclosure of this exposure.

Intermittent data/processing integrity failures were investigated with strict checks and preserved evidence; their underlying cause is unresolved. Successful repeated outputs do not prove that issue is repaired. The latest review recomputed metrics from saved scores, without fresh inference or a full dataset rehash.

### Proposed next steps

| Order | Action | Deliverable |

| --- | --- | --- |

| 1 | Diagnose2004 training/calibration across seeds | Score distributions, channel residuals and reconstruction plots |

| 2 | Explain one failure mechanism before changing training | A documented hypothesis with evidence and a bounded comparison |

| 3 | Repeat any justified candidate under matched budgets | Local/system F0.5, precision, recall and false-alarm duration for all seeds |

| 4 | Agree final evaluation and operational constraints | Frozen scope, latency requirement, selection policy and uncertainty reporting |

No new long training is needed for today's meeting. Broader channel experiments remain deferred while we resolve behavior on 40-47. Preserve all runs, use new output directories, and back up new artifacts before relying on GitHub as the only copy.


---


## 14  Earlier results and meeting speaking notes

The following older 40 epoch development results provide context. They are not the five-epoch controls used in the latest experiment. Assessment is April-December in each listed year; seed 42 only.

| Model and inputs | 2003 F0.5 | 2004 F0.5 | 2005 F0.5 |

| --- | --- | --- | --- |

| Baseline Transformer AE,40-47 | .833170 | .428131 | .869055 |

| Expanded AE,40-47 plus14/21/29 | .909069 | .251365 | .148245 |

| Reconstruction-only AE,40-47 | .525691 | .150184 | .868934 |

| Specialist AE,14/21/29 | .166589 | .769221 | .740641 |

| Frozen baseline OR optional specialist | .499681 | .428131 | .869055 |

The broader model recovered some missed events but added many false alarms. The specialist did not make the OR combination consistently better. These results support examining coverage and alarm behavior rather than assuming more channels or an ensemble will improve F0.5.

### Suggested opening for the meeting

"Since our last meeting, I investigated all available channels and all 14 years of Mission 1, then examined the preprocessing and training-only correlations. I kept the current model focused on channels 40-47 so that changes could be compared fairly. I also separated detection on those inputs from detection across the whole system, using identical predictions."

"The retained full-period Transformer scores 0.8057 on the system and 0.8629 locally. Its repeated run reproduced the previous outputs. I tested several alternatives, including masking, timing context and calibration changes, but they did not improve reliably. Studying short coordinated events led to a new augmentation method. It improved 2003 and 2005 on average, but three-seed testing exposed a 2004 regression caused by additional false alarms."

"My next step is to inspect 2004 training and calibration score stability before another long run. I do not claim that the 0.85 system target is achieved. I would like to agree what operational detection delay is acceptable and how we should design the final evaluation given the already-inspected data."

### Questions to settle with Federico

1. Is system-level F0.5 the primary thesis target, with local F0.5 secondary?<br/>2. What alarm latency and false-alarm workload would be operationally acceptable?<br/>3. Is more physical channel metadata available?<br/>4. How should final evaluation and calibration-event scarcity be handled?

Evidence: Earlier saved development results and research reports. Latest conclusions use the matched three-seed experiment, not a retrospective choice among this table's winners.


---


## 15  Evidence and reproducibility index

All reported project findings come from local saved protocols, result tables and dated research reports. This report adds presentation plots and summaries; it does not retrain models or change thresholds. The workbook provides numeric rows with calibration F0.5, assessment F0.5, precision, recall, event counts, nominal false-alarm duration, selected epoch, seed and scope.

Reports below are under **docs/research/** in the project. Numbers correspond to evidence notes throughout the report.

[1] full-mission-input-review-2026-09-23.md

[2] data-first-plan.md

[3] preprocessing-and-correlation-policy.md

[4] historical-channel-review-2026-09-23.md

[5] scaling-review-2026-09-24.md

[6] counter-aggregation-2026-09-24.md

[7] long-baseline.md

[8] short-coordinated-comparison.md

[9] corrected-evaluation.md

[10] dual-scope-evaluation.md

[11] long-baseline-results-2026-09-24.md

[12] baseline-repeat-2026-09-27.md

[13] channel-normalization-results-2026-09-24.md

[14] masked-pilot-followup-2026-09-24.md

[15] masked-overlap-results-2026-09-25.md

[16] timing-comparison-results-2026-09-26.md

[17] grid-cap-development-2026-09-26.md

[18] nominal-reconstruction-comparison-2026-09-26.md

[19] calibration-model-comparison-2005-2026-09-26.md

[20] id98-local-review-2026-09-26.md

[21] local-event-shapes-2026-09-27.md

[22] short-coordinated-augmentation.md

[23] short-coordinated-multiseed-2026-09-29.md

Primary saved runs: short_coordinated_seed42_v1, short_coordinated_seed43_v1 and short_coordinated_seed44_v1 under results_longrun/development/; full-period reference ch40_47_train2000_2005_seed42 under results_longrun/long_baseline/.

Benchmark background: <link href="https://arxiv.org/abs/2406.17826" color="#285780">ESA Anomaly Detection Benchmark paper</link> and <link href="https://github.com/kplabs-pl/ESA-ADB" color="#285780">upstream ESA-ADB repository</link>. These are background links, not new literature findings from this report.

Code and selected older artifacts were published on27September. The new comparison runner and seed42/43/44artifacts have not yet been included in a subsequent GitHub backup. Local completion and remote backup are different states.
