# Target-output scoring, alarm boundaries and post-processing — 1 November 2026

This study changes score pooling and alarm processing using two fixed high-recall checkpoints. Original short-no-roll remains an unchanged reference. The three candidate arms are not three new trained models. All model training used2000–2005, calibration uses2006, and the reused assessment is2007–2013. No new training, new seeds or assessment-selected rules.

Expanded all-output boundary trimming produced only a tiny F0.5 gain, with all detections retained. Masked-model trimming lost one target event and reduced F0.5. Target-only scoring substantially reduced F0.5 because it fragmented alarms. None of the four additional processing families selected a gain beyond its boundary fallback. All outcomes and unchanged controls are reported below.

## Mechanisms and selection

Expanded target-only scoring keeps all11 inputs (40–47,14,21,29) but permits only reconstruction errors from40–47 to enter the alarm score. The other expanded arm keeps all11 output errors, isolating boundary effects. Masked pretraining/fine-tuning uses eight inputs40–47 and ordinary unmasked reconstruction at inference.

The unchanged detector proposes components using its128-minute window-mean score. Finer evidence averages squared residuals at their actual timestamps. Boundary localization crops each component to the first and last fine-score crossing, preserving the entire span between them. If there is no crossing, the component is retained. No true event boundaries enter this transformation. Fine evidence remains retrospective.

All thresholds and decisions were selected on2006 and jointly frozen before new assessment inference. The objective was corrected local40–47 eventF0.5, retaining all original calibration-detected local IDs and at most1% total alarm fraction. Each of four post-processing families was separately applied before the already selected boundary rule: temporal merging/minimum duration, strength-duration, hysteresis and duration. Every family has an unchanged-boundary fallback. No assessment winner is substituted for the frozen choices.

## Calibration choices

| Arm | Base F0.5 | Boundary F0.5 | Nominal alarm seconds before / after | Boundary cutoff |
|---|---:|---:|---:|---:|
| Expanded inputs, target40–47 score | 0.999727433 | 0.999746304 | 10400.067 / 9680.067 | 8.14379002189657 |
| Masked pretraining + fine-tuning | 0.882105852 | 0.882137990 | 10379.778 / 9029.778 | 10.59309514808659 |
| Expanded inputs, all11-output score | 0.999727433 | 0.999746304 | 10400.067 / 9680.067 | 8.150397123337555 |

Independent verification replayed 398 calibration candidate rows (387 boundary/family entries plus the target-threshold grid). These are settings, not independent experiments. Duplicate prediction masks and all rejected candidates are retained in the compact evidence.

| Arm | Family | Settings | Distinct masks | Feasible | Retained boundary fallback |
|---|---|---:|---:|---:|---|
| Expanded inputs, target40–47 score | temporal | 48 | 2 | 48 | True |
| Expanded inputs, target40–47 score | strength_duration | 49 | 8 | 19 | True |
| Expanded inputs, target40–47 score | hysteresis | 16 | 9 | 7 | True |
| Expanded inputs, target40–47 score | duration | 9 | 5 | 3 | True |
| Masked pretraining + fine-tuning | temporal | 48 | 1 | 48 | True |
| Masked pretraining + fine-tuning | strength_duration | 49 | 5 | 13 | True |
| Masked pretraining + fine-tuning | hysteresis | 16 | 8 | 4 | True |
| Masked pretraining + fine-tuning | duration | 9 | 5 | 3 | True |
| Expanded inputs, all11-output score | temporal | 48 | 2 | 48 | True |
| Expanded inputs, all11-output score | strength_duration | 49 | 8 | 19 | True |
| Expanded inputs, all11-output score | hysteresis | 16 | 9 | 7 | True |
| Expanded inputs, all11-output score | duration | 9 | 5 | 3 | True |

## Pooled comparison and measured changes

TPR is event-ID recall; timeFPR is nominal false-alarm duration divided by nominal duration. Corrected precision includes the ESA nominal-duration penalty. Scores below are measurements on the reused benchmark, not a robust generalization or deployment claim.

| Arm | Variant | Scope | Detected/total | Precision | Event TPR | Time FPR (%) | F0.5 |
|---|---|---|---:|---:|---:|---:|---:|
| Original short-no-roll reference | original | system_all_channels | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070197 |
| Original short-no-roll reference | original | input_channels_40_47 | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907139530 |
| Masked-pretrained unchanged reference | original | system_all_channels | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675499 |
| Masked-pretrained unchanged reference | original | input_channels_40_47 | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844314859 |
| Expanded unchanged reference | original | system_all_channels | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573267 |
| Expanded unchanged reference | original | input_channels_40_47 | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830899674 |
| Expanded inputs, target40–47 score | base | system_all_channels | 78/91 | 0.241436 | 0.857143 | 1.258901 | 0.281940949 |
| Expanded inputs, target40–47 score | base | input_channels_40_47 | 62/67 | 0.202044 | 0.925373 | 1.258901 | 0.239483505 |
| Expanded inputs, target40–47 score | boundary | system_all_channels | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| Expanded inputs, target40–47 score | boundary | input_channels_40_47 | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| Expanded inputs, target40–47 score | temporal | system_all_channels | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| Expanded inputs, target40–47 score | temporal | input_channels_40_47 | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| Expanded inputs, target40–47 score | strength_duration | system_all_channels | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| Expanded inputs, target40–47 score | strength_duration | input_channels_40_47 | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| Expanded inputs, target40–47 score | hysteresis | system_all_channels | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| Expanded inputs, target40–47 score | hysteresis | input_channels_40_47 | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| Expanded inputs, target40–47 score | duration | system_all_channels | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| Expanded inputs, target40–47 score | duration | input_channels_40_47 | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| Masked pretraining + fine-tuning | base | system_all_channels | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675499 |
| Masked pretraining + fine-tuning | base | input_channels_40_47 | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844314859 |
| Masked pretraining + fine-tuning | boundary | system_all_channels | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| Masked pretraining + fine-tuning | boundary | input_channels_40_47 | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| Masked pretraining + fine-tuning | temporal | system_all_channels | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| Masked pretraining + fine-tuning | temporal | input_channels_40_47 | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| Masked pretraining + fine-tuning | strength_duration | system_all_channels | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| Masked pretraining + fine-tuning | strength_duration | input_channels_40_47 | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| Masked pretraining + fine-tuning | hysteresis | system_all_channels | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| Masked pretraining + fine-tuning | hysteresis | input_channels_40_47 | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| Masked pretraining + fine-tuning | duration | system_all_channels | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| Masked pretraining + fine-tuning | duration | input_channels_40_47 | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| Expanded inputs, all11-output score | base | system_all_channels | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573267 |
| Expanded inputs, all11-output score | base | input_channels_40_47 | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830899674 |
| Expanded inputs, all11-output score | boundary | system_all_channels | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| Expanded inputs, all11-output score | boundary | input_channels_40_47 | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| Expanded inputs, all11-output score | temporal | system_all_channels | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| Expanded inputs, all11-output score | temporal | input_channels_40_47 | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| Expanded inputs, all11-output score | strength_duration | system_all_channels | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| Expanded inputs, all11-output score | strength_duration | input_channels_40_47 | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| Expanded inputs, all11-output score | hysteresis | system_all_channels | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| Expanded inputs, all11-output score | hysteresis | input_channels_40_47 | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| Expanded inputs, all11-output score | duration | system_all_channels | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| Expanded inputs, all11-output score | duration | input_channels_40_47 | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |

### Expanded inputs, target40–47 score

- system_all_channels, base: F0.5 0.844573267 → 0.281940949; detected IDs 80 → 78; timeFPR 5.909894% → 1.258901%.
- system_all_channels, boundary: F0.5 0.844573267 → 0.282011250; detected IDs 80 → 78; timeFPR 5.909894% → 1.232546%.
- input_channels_40_47, base: F0.5 0.830899674 → 0.239483505; detected IDs 62 → 62; timeFPR 5.909894% → 1.258901%.
- input_channels_40_47, boundary: F0.5 0.830899674 → 0.239544116; detected IDs 62 → 62; timeFPR 5.909894% → 1.232546%.
- temporal: retained the selected boundary prediction.
- strength_duration: retained the selected boundary prediction.
- hysteresis: retained the selected boundary prediction.
- duration: retained the selected boundary prediction.

### Masked pretraining + fine-tuning

- system_all_channels, base: F0.5 0.795675499 → 0.795675499; detected IDs 61 → 61; timeFPR 0.110185% → 0.110185%.
- system_all_channels, boundary: F0.5 0.795675499 → 0.774619394; detected IDs 61 → 60; timeFPR 0.110185% → 0.096856%.
- input_channels_40_47, base: F0.5 0.844314859 → 0.844314859; detected IDs 60 → 60; timeFPR 0.110185% → 0.110185%.
- input_channels_40_47, boundary: F0.5 0.844314859 → 0.821079545; detected IDs 60 → 59; timeFPR 0.110185% → 0.096856%.
- temporal: retained the selected boundary prediction.
- strength_duration: retained the selected boundary prediction.
- hysteresis: retained the selected boundary prediction.
- duration: retained the selected boundary prediction.

### Expanded inputs, all11-output score

- system_all_channels, base: F0.5 0.844573267 → 0.844573267; detected IDs 80 → 80; timeFPR 5.909894% → 5.909894%.
- system_all_channels, boundary: F0.5 0.844573267 → 0.844609096; detected IDs 80 → 80; timeFPR 5.909894% → 5.904953%.
- input_channels_40_47, base: F0.5 0.830899674 → 0.830899674; detected IDs 62 → 62; timeFPR 5.909894% → 5.909894%.
- input_channels_40_47, boundary: F0.5 0.830899674 → 0.830935471; detected IDs 62 → 62; timeFPR 5.909894% → 5.904953%.
- temporal: retained the selected boundary prediction.
- strength_duration: retained the selected boundary prediction.
- hysteresis: retained the selected boundary prediction.
- duration: retained the selected boundary prediction.

## Interpretation of the measured outcomes

Expanded all-output boundary refinement increased system F0.5 from0.844573267 to0.844609096 (+0.000035829) and target40–47 F0.5 from0.830899674 to0.830935471 (+0.000035797). It retained80/91 system and62/67 target detections with no gained or lost IDs. Nominal alarm duration fell by9704.730 seconds, about2.70 hours across seven years, or0.0836% relative to its original false-alarm duration. This is a tiny measured gain, not evidence of a practically substantial or robust improvement.

Masked-model boundary refinement reduced nominal alarm duration but lost id_40 in both scopes: 61→60 system detections and60→59 target detections. False event count increased12→14 even though the transform preserves the number of predicted components: trimming can remove the overlap that previously connected a component to a real annotation. Its F0.5 decreased in both scopes. The original result remains visible; this unsuccessful frozen outcome is not replaced using test scores.

Target-only scoring preserved62/67 target detections but lost system IDs id_188 and id_189. The selected numeric threshold was unchanged. Its mask was a strict subset of the original expanded mask, yet predicted components increased95→407 and false events10→241. A retrospective mask comparison shows one original component split into313 target components; no original component was entirely removed. Removing auxiliary-output evidence therefore reduced duration while fragmenting an extended alarm into many false events. This explains the large event-precision/F0.5 loss despite a lower timeFPR. Boundary trimming changes only outer edges and cannot repair this fragmentation. These diagnostics explain fixed results; they did not change selection or create new candidates.

All four post-processing families retained the selected boundary fallback in every arm. They provided no additional selected gain within the declared calibration grids. The repeated family rows are identical predictions, not independent replications. Original short-no-roll still has the highest target-scope pooled F0.5 here (0.907139530), while detecting47/67 target events. There is no single method that simultaneously maximizes recall and precision in this comparison.

[Fragmentation diagnostic](tables/target-boundary-2026-11-01/fragmentation_diagnostics.json).

## Descriptive calibration stability

The table reports the frozen base and boundary choices on2006 halves and quarters. These blocks were already used for model/rule selection and are not independent validation. Small denominators and uneven event coverage make near-ceiling full-year calibration scores weak evidence of generalization. All252 block/scope/variant rows are preserved in [the complete stability table](tables/target-boundary-2026-11-01/run-records/calibration_stability.csv).

| Arm | Block | Base detected/total | Boundary detected/total | Base F0.5 | Boundary F0.5 |
|---|---|---:|---:|---:|---:|
| Expanded inputs, target40–47 score | H1 | 2/2 | 2/2 | 0.999880843 | 0.999891858 |
| Expanded inputs, target40–47 score | H2 | 4/4 | 4/4 | 0.999574227 | 0.999600944 |
| Expanded inputs, target40–47 score | Q1 | 0/0 | 0/0 | 0.000000000 | 0.000000000 |
| Expanded inputs, target40–47 score | Q2 | 2/2 | 2/2 | 0.999796928 | 0.999816143 |
| Expanded inputs, target40–47 score | Q3 | 1/1 | 1/1 | 0.999614183 | 0.999632303 |
| Expanded inputs, target40–47 score | Q4 | 3/3 | 3/3 | 0.999530869 | 0.999566916 |
| Masked pretraining + fine-tuning | H1 | 2/2 | 2/2 | 0.999982615 | 0.999982615 |
| Masked pretraining + fine-tuning | H2 | 4/4 | 4/4 | 0.832876435 | 0.832937826 |
| Masked pretraining + fine-tuning | Q1 | 0/0 | 0/0 | 0.000000000 | 0.000000000 |
| Masked pretraining + fine-tuning | Q2 | 2/2 | 2/2 | 0.999964619 | 0.999964619 |
| Masked pretraining + fine-tuning | Q3 | 1/1 | 1/1 | 0.555436699 | 0.555436699 |
| Masked pretraining + fine-tuning | Q4 | 3/3 | 3/3 | 0.999111365 | 0.999258857 |
| Expanded inputs, all11-output score | H1 | 2/2 | 2/2 | 0.999880843 | 0.999891858 |
| Expanded inputs, all11-output score | H2 | 4/4 | 4/4 | 0.999574227 | 0.999600944 |
| Expanded inputs, all11-output score | Q1 | 0/0 | 0/0 | 0.000000000 | 0.000000000 |
| Expanded inputs, all11-output score | Q2 | 2/2 | 2/2 | 0.999796928 | 0.999816143 |
| Expanded inputs, all11-output score | Q3 | 1/1 | 1/1 | 0.999614183 | 0.999632303 |
| Expanded inputs, all11-output score | Q4 | 3/3 | 3/3 | 0.999530869 | 0.999566916 |

## Annual and pooled metrics

Predictions are continuous across calendar boundaries; annual slices are reporting only. Pooled metrics are recomputed over the complete period, not averaged from annual ratios.

### Original short-no-roll reference — system_all_channels

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 11/16 | 0.999539 | 0.687500 | 0.046099 | 0.916356744 |
| 2008 | original | 7/13 | 0.999508 | 0.538462 | 0.049188 | 0.853371732 |
| 2009 | original | 7/12 | 0.999574 | 0.583333 | 0.042598 | 0.874739052 |
| 2010 | original | 3/12 | 0.999808 | 0.250000 | 0.019182 | 0.624940049 |
| 2011 | original | 8/15 | 0.999858 | 0.533333 | 0.014158 | 0.850981791 |
| 2012 | original | 4/10 | 0.799915 | 0.400000 | 0.010585 | 0.666619619 |
| 2013 | original | 8/14 | 0.999764 | 0.571429 | 0.023608 | 0.869422399 |
| 2007-2013 | original | 48/91 | 0.979306 | 0.527473 | 0.029194 | 0.836070197 |

### Original short-no-roll reference — input_channels_40_47

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 11/13 | 0.999539 | 0.846154 | 0.046099 | 0.964568882 |
| 2008 | original | 7/9 | 0.999508 | 0.777778 | 0.049188 | 0.945593791 |
| 2009 | original | 7/10 | 0.999574 | 0.700000 | 0.042598 | 0.920763497 |
| 2010 | original | 3/8 | 0.999808 | 0.375000 | 0.019182 | 0.749913673 |
| 2011 | original | 8/11 | 0.999858 | 0.727273 | 0.014158 | 0.930134547 |
| 2012 | original | 3/8 | 0.749921 | 0.375000 | 0.010585 | 0.624955893 |
| 2013 | original | 8/9 | 0.999764 | 0.888889 | 0.023608 | 0.975429983 |
| 2007-2013 | original | 47/67 | 0.978881 | 0.701493 | 0.029194 | 0.907139530 |

### Masked-pretrained unchanged reference — system_all_channels

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954988652 |
| 2008 | original | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777117695 |
| 2009 | original | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832391803 |
| 2010 | original | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221430 |
| 2011 | original | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633380504 |
| 2012 | original | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906169 |
| 2013 | original | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696342 |
| 2007-2013 | original | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675499 |

### Masked-pretrained unchanged reference — input_channels_40_47

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999021942 |
| 2008 | original | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863440 |
| 2009 | original | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061422 |
| 2010 | original | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948031 |
| 2011 | original | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671167567 |
| 2012 | original | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239308 |
| 2013 | original | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917593785 |
| 2007-2013 | original | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844314859 |

### Expanded unchanged reference — system_all_channels

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519124 |
| 2008 | original | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942781722 |
| 2009 | original | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099270 |
| 2010 | original | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936992686 |
| 2011 | original | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932829892 |
| 2012 | original | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773118 |
| 2013 | original | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381004 |
| 2007-2013 | original | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573267 |

### Expanded unchanged reference — input_channels_40_47

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | original | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848367 |
| 2008 | original | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328117 |
| 2009 | original | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716178 |
| 2010 | original | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999422810 |
| 2011 | original | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692040 |
| 2012 | original | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499810932 |
| 2013 | original | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381004 |
| 2007-2013 | original | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830899674 |

### Expanded inputs, target40–47 score — system_all_channels

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | base | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519124 |
| 2007 | boundary | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | temporal | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | strength_duration | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | hysteresis | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | duration | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2008 | base | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942781722 |
| 2008 | boundary | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | temporal | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | strength_duration | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | hysteresis | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | duration | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2009 | base | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099270 |
| 2009 | boundary | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | temporal | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | strength_duration | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | hysteresis | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | duration | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2010 | base | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936992686 |
| 2010 | boundary | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | temporal | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | strength_duration | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | hysteresis | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | duration | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2011 | base | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932829892 |
| 2011 | boundary | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | temporal | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | strength_duration | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | hysteresis | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | duration | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2012 | base | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773118 |
| 2012 | boundary | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | temporal | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | strength_duration | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | hysteresis | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | duration | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2013 | base | 12/14 | 0.045107 | 0.857143 | 8.658748 | 0.055651332 |
| 2013 | boundary | 12/14 | 0.045184 | 0.857143 | 8.501740 | 0.055745748 |
| 2013 | temporal | 12/14 | 0.045184 | 0.857143 | 8.501740 | 0.055745748 |
| 2013 | strength_duration | 12/14 | 0.045184 | 0.857143 | 8.501740 | 0.055745748 |
| 2013 | hysteresis | 12/14 | 0.045184 | 0.857143 | 8.501740 | 0.055745748 |
| 2013 | duration | 12/14 | 0.045184 | 0.857143 | 8.501740 | 0.055745748 |
| 2007-2013 | base | 78/91 | 0.241436 | 0.857143 | 1.258901 | 0.281940949 |
| 2007-2013 | boundary | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| 2007-2013 | temporal | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| 2007-2013 | strength_duration | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| 2007-2013 | hysteresis | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |
| 2007-2013 | duration | 78/91 | 0.241500 | 0.857143 | 1.232546 | 0.282011250 |

### Expanded inputs, target40–47 score — input_channels_40_47

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | base | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848367 |
| 2007 | boundary | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | temporal | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | strength_duration | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | hysteresis | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | duration | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2008 | base | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328117 |
| 2008 | boundary | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | temporal | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | strength_duration | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | hysteresis | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | duration | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2009 | base | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716178 |
| 2009 | boundary | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | temporal | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | strength_duration | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | hysteresis | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | duration | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2010 | base | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999422810 |
| 2010 | boundary | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | temporal | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | strength_duration | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | hysteresis | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | duration | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2011 | base | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692040 |
| 2011 | boundary | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | temporal | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | strength_duration | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | hysteresis | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | duration | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2012 | base | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499810932 |
| 2012 | boundary | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | temporal | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | strength_duration | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | hysteresis | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | duration | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2013 | base | 9/9 | 0.034253 | 1.000000 | 8.658748 | 0.042452679 |
| 2013 | boundary | 9/9 | 0.034312 | 1.000000 | 8.501740 | 0.042525031 |
| 2013 | temporal | 9/9 | 0.034312 | 1.000000 | 8.501740 | 0.042525031 |
| 2013 | strength_duration | 9/9 | 0.034312 | 1.000000 | 8.501740 | 0.042525031 |
| 2013 | hysteresis | 9/9 | 0.034312 | 1.000000 | 8.501740 | 0.042525031 |
| 2013 | duration | 9/9 | 0.034312 | 1.000000 | 8.501740 | 0.042525031 |
| 2007-2013 | base | 62/67 | 0.202044 | 0.925373 | 1.258901 | 0.239483505 |
| 2007-2013 | boundary | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| 2007-2013 | temporal | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| 2007-2013 | strength_duration | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| 2007-2013 | hysteresis | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |
| 2007-2013 | duration | 62/67 | 0.202098 | 0.925373 | 1.232546 | 0.239544116 |

### Masked pretraining + fine-tuning — system_all_channels

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | base | 13/16 | 0.998778 | 0.812500 | 0.122227 | 0.954988652 |
| 2007 | boundary | 13/16 | 0.998992 | 0.812500 | 0.100778 | 0.955145521 |
| 2007 | temporal | 13/16 | 0.998992 | 0.812500 | 0.100778 | 0.955145521 |
| 2007 | strength_duration | 13/16 | 0.998992 | 0.812500 | 0.100778 | 0.955145521 |
| 2007 | hysteresis | 13/16 | 0.998992 | 0.812500 | 0.100778 | 0.955145521 |
| 2007 | duration | 13/16 | 0.998992 | 0.812500 | 0.100778 | 0.955145521 |
| 2008 | base | 7/13 | 0.873956 | 0.538462 | 0.119304 | 0.777117695 |
| 2008 | boundary | 7/13 | 0.874142 | 0.538462 | 0.098090 | 0.777235101 |
| 2008 | temporal | 7/13 | 0.874142 | 0.538462 | 0.098090 | 0.777235101 |
| 2008 | strength_duration | 7/13 | 0.874142 | 0.538462 | 0.098090 | 0.777235101 |
| 2008 | hysteresis | 7/13 | 0.874142 | 0.538462 | 0.098090 | 0.777235101 |
| 2008 | duration | 7/13 | 0.874142 | 0.538462 | 0.098090 | 0.777235101 |
| 2009 | base | 10/12 | 0.832157 | 0.833333 | 0.141190 | 0.832391803 |
| 2009 | boundary | 9/12 | 0.749048 | 0.750000 | 0.126913 | 0.749238327 |
| 2009 | temporal | 9/12 | 0.749048 | 0.750000 | 0.126913 | 0.749238327 |
| 2009 | strength_duration | 9/12 | 0.749048 | 0.750000 | 0.126913 | 0.749238327 |
| 2009 | hysteresis | 9/12 | 0.749048 | 0.750000 | 0.126913 | 0.749238327 |
| 2009 | duration | 9/12 | 0.749048 | 0.750000 | 0.126913 | 0.749238327 |
| 2010 | base | 8/12 | 0.998685 | 0.666667 | 0.131462 | 0.908221430 |
| 2010 | boundary | 8/12 | 0.998784 | 0.666667 | 0.121648 | 0.908286356 |
| 2010 | temporal | 8/12 | 0.998784 | 0.666667 | 0.121648 | 0.908286356 |
| 2010 | strength_duration | 8/12 | 0.998784 | 0.666667 | 0.121648 | 0.908286356 |
| 2010 | hysteresis | 8/12 | 0.998784 | 0.666667 | 0.121648 | 0.908286356 |
| 2010 | duration | 8/12 | 0.998784 | 0.666667 | 0.121648 | 0.908286356 |
| 2011 | base | 9/15 | 0.642314 | 0.600000 | 0.084464 | 0.633380504 |
| 2011 | boundary | 9/15 | 0.642345 | 0.600000 | 0.079727 | 0.633404195 |
| 2011 | temporal | 9/15 | 0.642345 | 0.600000 | 0.079727 | 0.633404195 |
| 2011 | strength_duration | 9/15 | 0.642345 | 0.600000 | 0.079727 | 0.633404195 |
| 2011 | hysteresis | 9/15 | 0.642345 | 0.600000 | 0.079727 | 0.633404195 |
| 2011 | duration | 9/15 | 0.642345 | 0.600000 | 0.079727 | 0.633404195 |
| 2012 | base | 5/10 | 0.624543 | 0.500000 | 0.073177 | 0.594906169 |
| 2012 | boundary | 5/10 | 0.624575 | 0.500000 | 0.068014 | 0.594929590 |
| 2012 | temporal | 5/10 | 0.624575 | 0.500000 | 0.068014 | 0.594929590 |
| 2012 | strength_duration | 5/10 | 0.624575 | 0.500000 | 0.068014 | 0.594929590 |
| 2012 | hysteresis | 5/10 | 0.624575 | 0.500000 | 0.068014 | 0.594929590 |
| 2012 | duration | 5/10 | 0.624575 | 0.500000 | 0.068014 | 0.594929590 |
| 2013 | base | 9/14 | 0.899072 | 0.642857 | 0.103165 | 0.832696342 |
| 2013 | boundary | 9/14 | 0.817476 | 0.642857 | 0.086287 | 0.775354090 |
| 2013 | temporal | 9/14 | 0.817476 | 0.642857 | 0.086287 | 0.775354090 |
| 2013 | strength_duration | 9/14 | 0.817476 | 0.642857 | 0.086287 | 0.775354090 |
| 2013 | hysteresis | 9/14 | 0.817476 | 0.642857 | 0.086287 | 0.775354090 |
| 2013 | duration | 9/14 | 0.817476 | 0.642857 | 0.086287 | 0.775354090 |
| 2007-2013 | base | 61/91 | 0.834696 | 0.670330 | 0.110185 | 0.795675499 |
| 2007-2013 | boundary | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| 2007-2013 | temporal | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| 2007-2013 | strength_duration | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| 2007-2013 | hysteresis | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |
| 2007-2013 | duration | 60/91 | 0.810025 | 0.659341 | 0.096856 | 0.774619394 |

### Masked pretraining + fine-tuning — input_channels_40_47

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | base | 13/13 | 0.998778 | 1.000000 | 0.122227 | 0.999021942 |
| 2007 | boundary | 13/13 | 0.998992 | 1.000000 | 0.100778 | 0.999193612 |
| 2007 | temporal | 13/13 | 0.998992 | 1.000000 | 0.100778 | 0.999193612 |
| 2007 | strength_duration | 13/13 | 0.998992 | 1.000000 | 0.100778 | 0.999193612 |
| 2007 | hysteresis | 13/13 | 0.998992 | 1.000000 | 0.100778 | 0.999193612 |
| 2007 | duration | 13/13 | 0.998992 | 1.000000 | 0.100778 | 0.999193612 |
| 2008 | base | 7/9 | 0.873956 | 0.777778 | 0.119304 | 0.852863440 |
| 2008 | boundary | 7/9 | 0.874142 | 0.777778 | 0.098090 | 0.853004851 |
| 2008 | temporal | 7/9 | 0.874142 | 0.777778 | 0.098090 | 0.853004851 |
| 2008 | strength_duration | 7/9 | 0.874142 | 0.777778 | 0.098090 | 0.853004851 |
| 2008 | hysteresis | 7/9 | 0.874142 | 0.777778 | 0.098090 | 0.853004851 |
| 2008 | duration | 7/9 | 0.874142 | 0.777778 | 0.098090 | 0.853004851 |
| 2009 | base | 10/10 | 0.832157 | 1.000000 | 0.141190 | 0.861061422 |
| 2009 | boundary | 9/10 | 0.749048 | 0.900000 | 0.126913 | 0.775046989 |
| 2009 | temporal | 9/10 | 0.749048 | 0.900000 | 0.126913 | 0.775046989 |
| 2009 | strength_duration | 9/10 | 0.749048 | 0.900000 | 0.126913 | 0.775046989 |
| 2009 | hysteresis | 9/10 | 0.749048 | 0.900000 | 0.126913 | 0.775046989 |
| 2009 | duration | 9/10 | 0.749048 | 0.900000 | 0.126913 | 0.775046989 |
| 2010 | base | 8/8 | 0.998685 | 1.000000 | 0.131462 | 0.998948031 |
| 2010 | boundary | 8/8 | 0.998784 | 1.000000 | 0.121648 | 0.999026577 |
| 2010 | temporal | 8/8 | 0.998784 | 1.000000 | 0.121648 | 0.999026577 |
| 2010 | strength_duration | 8/8 | 0.998784 | 1.000000 | 0.121648 | 0.999026577 |
| 2010 | hysteresis | 8/8 | 0.998784 | 1.000000 | 0.121648 | 0.999026577 |
| 2010 | duration | 8/8 | 0.998784 | 1.000000 | 0.121648 | 0.999026577 |
| 2011 | base | 9/11 | 0.642314 | 0.818182 | 0.084464 | 0.671167567 |
| 2011 | boundary | 9/11 | 0.642345 | 0.818182 | 0.079727 | 0.671194168 |
| 2011 | temporal | 9/11 | 0.642345 | 0.818182 | 0.079727 | 0.671194168 |
| 2011 | strength_duration | 9/11 | 0.642345 | 0.818182 | 0.079727 | 0.671194168 |
| 2011 | hysteresis | 9/11 | 0.642345 | 0.818182 | 0.079727 | 0.671194168 |
| 2011 | duration | 9/11 | 0.642345 | 0.818182 | 0.079727 | 0.671194168 |
| 2012 | base | 4/8 | 0.571010 | 0.500000 | 0.073177 | 0.555239308 |
| 2012 | boundary | 4/8 | 0.571040 | 0.500000 | 0.068014 | 0.555261622 |
| 2012 | temporal | 4/8 | 0.571040 | 0.500000 | 0.068014 | 0.555261622 |
| 2012 | strength_duration | 4/8 | 0.571040 | 0.500000 | 0.068014 | 0.555261622 |
| 2012 | hysteresis | 4/8 | 0.571040 | 0.500000 | 0.068014 | 0.555261622 |
| 2012 | duration | 4/8 | 0.571040 | 0.500000 | 0.068014 | 0.555261622 |
| 2013 | base | 9/9 | 0.899072 | 1.000000 | 0.103165 | 0.917593785 |
| 2013 | boundary | 9/9 | 0.817476 | 1.000000 | 0.086287 | 0.848448297 |
| 2013 | temporal | 9/9 | 0.817476 | 1.000000 | 0.086287 | 0.848448297 |
| 2013 | strength_duration | 9/9 | 0.817476 | 1.000000 | 0.086287 | 0.848448297 |
| 2013 | hysteresis | 9/9 | 0.817476 | 1.000000 | 0.086287 | 0.848448297 |
| 2013 | duration | 9/9 | 0.817476 | 1.000000 | 0.086287 | 0.848448297 |
| 2007-2013 | base | 60/67 | 0.832415 | 0.895522 | 0.110185 | 0.844314859 |
| 2007-2013 | boundary | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| 2007-2013 | temporal | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| 2007-2013 | strength_duration | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| 2007-2013 | hysteresis | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |
| 2007-2013 | duration | 59/67 | 0.807436 | 0.880597 | 0.096856 | 0.821079545 |

### Expanded inputs, all11-output score — system_all_channels

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | base | 16/16 | 0.888206 | 1.000000 | 0.076863 | 0.908519124 |
| 2007 | boundary | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | temporal | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | strength_duration | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | hysteresis | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2007 | duration | 16/16 | 0.888244 | 1.000000 | 0.072529 | 0.908551363 |
| 2008 | base | 10/13 | 0.999137 | 0.769231 | 0.086289 | 0.942781722 |
| 2008 | boundary | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | temporal | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | strength_duration | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | hysteresis | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2008 | duration | 10/13 | 0.999179 | 0.769231 | 0.082057 | 0.942811866 |
| 2009 | base | 11/12 | 0.784860 | 0.916667 | 0.108712 | 0.808099270 |
| 2009 | boundary | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | temporal | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | strength_duration | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | hysteresis | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2009 | duration | 11/12 | 0.784932 | 0.916667 | 0.099577 | 0.808160143 |
| 2010 | base | 9/12 | 0.999279 | 0.750000 | 0.072138 | 0.936992686 |
| 2010 | boundary | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | temporal | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | strength_duration | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | hysteresis | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2010 | duration | 9/12 | 0.999349 | 0.750000 | 0.065144 | 0.937041883 |
| 2011 | base | 14/15 | 0.932704 | 0.933333 | 0.067416 | 0.932829892 |
| 2011 | boundary | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | temporal | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | strength_duration | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | hysteresis | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2011 | duration | 14/15 | 0.932733 | 0.933333 | 0.064294 | 0.932853211 |
| 2012 | base | 6/10 | 0.599716 | 0.600000 | 0.047263 | 0.599773118 |
| 2012 | boundary | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | temporal | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | strength_duration | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | hysteresis | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2012 | duration | 6/10 | 0.599749 | 0.600000 | 0.041898 | 0.599798875 |
| 2013 | base | 14/14 | 0.576011 | 1.000000 | 42.398914 | 0.629381004 |
| 2013 | boundary | 14/14 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | temporal | 14/14 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | strength_duration | 14/14 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | hysteresis | 14/14 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | duration | 14/14 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2007-2013 | base | 80/91 | 0.836357 | 0.879121 | 5.909894 | 0.844573267 |
| 2007-2013 | boundary | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| 2007-2013 | temporal | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| 2007-2013 | strength_duration | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| 2007-2013 | hysteresis | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |
| 2007-2013 | duration | 80/91 | 0.836400 | 0.879121 | 5.904953 | 0.844609096 |

### Expanded inputs, all11-output score — input_channels_40_47

| Period | Variant | Detected/total | Precision | TPR | Time FPR (%) | F0.5 |
|---|---|---:|---:|---:|---:|---:|
| 2007 | base | 13/13 | 0.866001 | 1.000000 | 0.076863 | 0.889848367 |
| 2007 | boundary | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | temporal | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | strength_duration | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | hysteresis | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2007 | duration | 13/13 | 0.866038 | 1.000000 | 0.072529 | 0.889880087 |
| 2008 | base | 7/9 | 0.999137 | 0.777778 | 0.086289 | 0.945328117 |
| 2008 | boundary | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | temporal | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | strength_duration | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | hysteresis | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2008 | duration | 7/9 | 0.999179 | 0.777778 | 0.082057 | 0.945358424 |
| 2009 | base | 10/10 | 0.768395 | 1.000000 | 0.108712 | 0.805716178 |
| 2009 | boundary | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | temporal | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | strength_duration | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | hysteresis | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2009 | duration | 10/10 | 0.768465 | 1.000000 | 0.099577 | 0.805777989 |
| 2010 | base | 8/8 | 0.999279 | 1.000000 | 0.072138 | 0.999422810 |
| 2010 | boundary | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | temporal | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | strength_duration | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | hysteresis | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2010 | duration | 8/8 | 0.999349 | 1.000000 | 0.065144 | 0.999478781 |
| 2011 | base | 11/11 | 0.916049 | 1.000000 | 0.067416 | 0.931692040 |
| 2011 | boundary | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | temporal | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | strength_duration | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | hysteresis | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2011 | duration | 11/11 | 0.916077 | 1.000000 | 0.064294 | 0.931715725 |
| 2012 | base | 4/8 | 0.499764 | 0.500000 | 0.047263 | 0.499810932 |
| 2012 | boundary | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | temporal | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | strength_duration | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | hysteresis | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2012 | duration | 4/8 | 0.499791 | 0.500000 | 0.041898 | 0.499832396 |
| 2013 | base | 9/9 | 0.576011 | 1.000000 | 42.398914 | 0.629381004 |
| 2013 | boundary | 9/9 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | temporal | 9/9 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | strength_duration | 9/9 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | hysteresis | 9/9 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2013 | duration | 9/9 | 0.576027 | 1.000000 | 42.397252 | 0.629396878 |
| 2007-2013 | base | 62/67 | 0.810220 | 0.925373 | 5.909894 | 0.830899674 |
| 2007-2013 | boundary | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| 2007-2013 | temporal | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| 2007-2013 | strength_duration | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| 2007-2013 | hysteresis | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |
| 2007-2013 | duration | 62/67 | 0.810263 | 0.925373 | 5.904953 | 0.830935471 |

## Event changes

- expanded_all/base/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_all/base/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_all/boundary/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_all/boundary/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_all/duration/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_all/duration/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_all/hysteresis/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_all/hysteresis/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_all/strength_duration/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_all/strength_duration/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_all/temporal/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_all/temporal/system_all_channels: gained none; lost none; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_20, id_24, id_42, id_43.
- expanded_target/base/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_target/base/system_all_channels: gained none; lost id_188, id_189; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_188, id_189, id_20, id_24, id_42, id_43.
- expanded_target/boundary/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_target/boundary/system_all_channels: gained none; lost id_188, id_189; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_188, id_189, id_20, id_24, id_42, id_43.
- expanded_target/duration/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_target/duration/system_all_channels: gained none; lost id_188, id_189; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_188, id_189, id_20, id_24, id_42, id_43.
- expanded_target/hysteresis/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_target/hysteresis/system_all_channels: gained none; lost id_188, id_189; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_188, id_189, id_20, id_24, id_42, id_43.
- expanded_target/strength_duration/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_target/strength_duration/system_all_channels: gained none; lost id_188, id_189; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_188, id_189, id_20, id_24, id_42, id_43.
- expanded_target/temporal/input_channels_40_47: gained none; lost none; missed id_11, id_174, id_20, id_24, id_42.
- expanded_target/temporal/system_all_channels: gained none; lost id_188, id_189; missed id_11, id_137, id_141, id_143, id_146, id_156, id_174, id_188, id_189, id_20, id_24, id_42, id_43.
- masked_all/base/input_channels_40_47: gained none; lost none; missed id_11, id_157, id_174, id_20, id_24, id_42, id_55.
- masked_all/base/system_all_channels: gained none; lost none; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_42, id_43, id_55.
- masked_all/boundary/input_channels_40_47: gained none; lost id_40; missed id_11, id_157, id_174, id_20, id_24, id_40, id_42, id_55.
- masked_all/boundary/system_all_channels: gained none; lost id_40; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_55.
- masked_all/duration/input_channels_40_47: gained none; lost id_40; missed id_11, id_157, id_174, id_20, id_24, id_40, id_42, id_55.
- masked_all/duration/system_all_channels: gained none; lost id_40; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_55.
- masked_all/hysteresis/input_channels_40_47: gained none; lost id_40; missed id_11, id_157, id_174, id_20, id_24, id_40, id_42, id_55.
- masked_all/hysteresis/system_all_channels: gained none; lost id_40; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_55.
- masked_all/strength_duration/input_channels_40_47: gained none; lost id_40; missed id_11, id_157, id_174, id_20, id_24, id_40, id_42, id_55.
- masked_all/strength_duration/system_all_channels: gained none; lost id_40; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_55.
- masked_all/temporal/input_channels_40_47: gained none; lost id_40; missed id_11, id_157, id_174, id_20, id_24, id_40, id_42, id_55.
- masked_all/temporal/system_all_channels: gained none; lost id_40; missed id_11, id_115, id_119, id_120, id_126, id_127, id_131, id_135, id_137, id_141, id_143, id_146, id_148, id_152, id_156, id_157, id_161, id_162, id_171, id_174, id_178, id_179, id_181, id_188, id_189, id_20, id_24, id_40, id_42, id_43, id_55.

## Validation and limitations

All161 preflight tests passed, including independent-transform properties and pinned corrected-metric parity. Independent verification replays every calibration selection, prediction, all336 metric rows and event IDs; unchanged references match the prior study. Full re-extracted all-output scores agree with saved scores and produce exactly the original thresholded predictions. CPU spot checks independently reconstruct residuals at predetermined calibration and assessment locations. This is sampled neural replication, not a second full inference run.

All252 calibration half/quarter rows and detected IDs were independently replayed. This table is descriptive: the checkpoints and rules already used the entire2006 year. It cannot provide independent validation. A [chronological validation design](chronological-validation-design-2026-11-01.md) requires fold-specific training/scaling and separate calibration/validation before broader tuning; no such retraining was launched here. Only six target calibration IDs constrain selection.

Single seed, reused benchmark, offline future context, expanded contextual inputs, unequal prior training and additional inference/selection compute limit interpretation. Inherited ESA preparation restores some annotated numerical values. This study measures that fixed prepared benchmark and does not quantify the preparation effect or claim label-independent operational performance. Failed outcomes and original references are preserved.

[Protocol](target-boundary-protocol-2026-11-01.md), [verified metrics](tables/target-boundary-2026-11-01/verified_results.csv), [event IDs](tables/target-boundary-2026-11-01/event_ids.csv), [verification](tables/target-boundary-2026-11-01/verification.json), [W&B project](https://wandb.ai/kamyarfaz-politecnico-di-torino/esa-thesis-finetuning), [Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/target-boundary-2026-11-01).
