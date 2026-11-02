# Mission1/Mission2 channel correspondence and correlation investigation

The missions contain useful analogous signal families, but equal channel numbers
do not denote equivalent measurements. The data support a new Mission2 benchmark
with related channel types; they do not establish a one-to-one transfer of the
Mission1 model's eight inputs. Several Mission2 groups have strong, stable
within-mission correlation. No new F0.5, training, or assessment result is claimed.

## Preservation and scope

Before signal analysis, origin/main matched the completed work at4d7069a and the
five latest Release asset digests matched the verified backup receipt. An
[additional preservation Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/research-preservation-2026-11-02)
contains48 supporting outputs; a fresh download and all member hashes verified.
This is essential-results preservation, not a claim that GitHub replaces the full
raw-data/external-drive migration copy. The original workbook and all scores are unchanged.

Both raw version1 archives match the official checksums from
[Zenodo](https://zenodo.org/records/12528696). All183 extracted channel archives and
available metadata CSVs match those archive members. The scoped destination data
are verified; unrelated historical migration exceptions are not resolved by this.

The new analysis used Mission1 training2000–2005 and Mission2 training2000 only.
Mission2 January–September2001 is reserved for possible calibration and October2001
onward for assessment. Mission2's approximately3.5-year record cannot use the old
Mission1 assessment years2007–2013. Whole-file integrity/deserialization is distinct
from using later observations: raw series were sliced before computing statistics.
No future signal summaries, event counts, labels or prediction outcomes selected
candidates. Source hashes cover full files only to establish integrity.

## Metadata findings

The [author paper, Supplementary2.3–2.4](https://arxiv.org/pdf/2406.17826) states
that subsystem/unit names are consistent across missions, but dates are shifted
and scaled separately by undisclosed factors. Channel group identifiers do not
establish identity across missions. Therefore, matching anonymous dates is not a
valid way to measure cross-mission correlation; matching clock periods also does
not prove identical physical dynamics.

| Mission1 experimental channels | Metadata | Mission2 target candidates |
|---|---|---|
|40,47|subsystem6, physical_unit3|58,59,86,87,88,89,90,91|
|41–46|subsystem5, physical_unit4|70,71,72,73,75,76,77,78,79,80,81,83,84,85|

Mission2 channels40–46 are subsystem1/unit2, group20, non-target counters.
Mission2 channel47 is subsystem1/unit8, group4, non-target. Thus simply copying the
40–47 channel range into Mission2 would change the research task substantially.
The inventory includes all76 Mission1 and100 Mission2 channels; signal diagnostics
cover the eight anchors and all22 target channels passing the metadata rule.

## Measured within-Mission2 relationships

The following are actual Pearson coefficients between channels within Mission2,
not correlation coefficients between missions. They use nominal training2000,
five-minute causal sampling, at most ten minutes of observation age and no
annotation-driven replacement of signal values. Quarter results use all four
training quarters. Absolute threshold flags were fixed before computation.

| Mission2 group | Channels | Pooled level r across pairs | Lowest level r in any quarter | Pooled five-minute-change r across pairs | Interpretation |
|---|---|---:|---:|---:|---|
|31|75,76,77,83,84,85|0.997573–0.998379|0.996988|0.996292–0.997564|Strong common levels and changes; strongest coherent six-channel candidate|
|14|70,71,72,78,79,80|0.992293–0.995892|0.989576|0.992396–0.993721|Another strong coherent six-channel candidate|
|11|73,81|0.946012|0.939359|0.991840|Related pair with very strong shared changes|
|18|58,59|0.997282|0.997579|0.060571|Very strong common level trend, weak short-term coupling|
|13|86–91|0.995033–0.998874|0.992895|−0.254526–0.682992|Common level trend; change relationships differ by pair|

Across all231 candidate pairs,65 passed the fixed stable-level criterion
(same sign and absolute r≥0.90 in every quarter). Sixty-nine passed the separate
stable-change criterion (same sign and absolute r≥0.50 in every quarter).
These counts overlap; neither is a count of independent experiments.
Mission2 eligible grid coverage was99.60–99.90%; Mission1 was95.60–98.54%.
Five-minute sampling can miss fast transients and these diagnostics do not certify
the future model's higher-resolution input preparation.

## Cross-mission pattern similarity: informative, not proof of identity

Separately computed distribution, autocorrelation and change summaries were
compared within the metadata-compatible pools. A prespecified equal-family
distance ranks58/59 first for40 and47, and73/81 first for each of41–46. All100
eligible comparisons are saved, not just the best matches. The score is an
uncalibrated distance, not Pearson r, a percentage similarity, or a confidence.

Crucially, this is many-to-one, not a recovered eight-to-eight channel mapping.
The rankings also differ by feature family: distribution-only comparisons favor
85 for41–46; autocorrelation-only comparisons favor70 or73; change summaries
favor73. Unknown mission time scaling and unequal training durations further
limit the temporal comparison. We should not remap existing channel embeddings
and call that a physically validated cross-mission model.

Mission1's six-channel group41–46 also has paired and anticorrelated structure,
whereas Mission2 groups14/31 are nearly uniformly positively correlated. Same
subsystem/unit does not imply the same covariance pattern or equally difficult
anomaly-detection problem. Strong Mission2 within-group correlation supports
testing those groups, not claiming they reproduce the whole Mission1 task.

## Useful Mission1 alternatives already supported by training evidence

The existing full-resolution, globally nominal, prepared-data training audit
used2000–2005 only. Its strongest additional relationships to channel40 are:

| Pair | Pooled level r | Minimum absolute annual r |
|---|---:|---:|
|32 ↔40|0.998428|0.965890|
|16 ↔40|0.998343|0.963631|
|24 ↔40|0.998279|0.963445|
|33 ↔40|0.998495|0.961701|
|41 ↔44|0.908606|0.902713|

These are good training-derived context candidates, not proof of anomaly utility.
For the four channel40 pairs, some annual adjacent30-second change correlations
are near zero. Their high level r can reflect shared slow variation. Mission1
40↔47 has pooled r≈0.883 but annual r≈0.252 in2001, so the original eight-channel
range is not a uniformly correlated set. All target-related pairs and source
hashes are retained. The old full-resolution prepared-data audit and the new
raw five-minute, channel-wise masked analysis have different sampling/masking;
their coefficients should not be treated as identical estimators.

## Concrete next experiment

The recommended first question is whether the existing techniques work when
trained separately on Mission2, using a common Mission2 split and unchanged
evaluation definitions. Direct weight transfer is a separate question because
physical correspondence and input order are unresolved. The user has been asked
which objective to prioritize; no neural job was launched by this investigation.

A defensible eight-target starting candidate is **58,59,75,76,77,83,84,85**:
two subsystem6/unit3 channels and six subsystem5/unit4 channels, preserving the
two-plus-six metadata composition of Mission1 40–47. Group31 is selected as a
coherent candidate from training statistics, not from event counts or assessment
scores. Groups14 and11 remain documented alternatives; do not sweep candidate
sets against the final assessment. This is an analogous task, not an exact mapping.

For separate training, fix the following before any assessment:

1. Train2000, calibrate January–September2001, assess October2001 onward. Preserve
   split-contained windows and an explicitly documented overlap/embargo policy.
2. Train fresh models and scalers; do not reuse Mission1-trained weights while
   describing the result as separate Mission2 training. Keep the same targets
   for all three techniques. Extra-context channels must be chosen on training
   evidence before calibration; Mission1's14/21/29 identifiers do not transfer.
3. Use label-independent numerical preparation for the new Mission2 study and
   document its difference from the inherited Mission1 benchmark preparation.
   The descriptive five-minute grids are not model training arrays. Establish
   the model input cadence/window duration and budget in an execution protocol.
4. Preserve corrected eventF0.5, precision, eventTPR, timeFPR and detected/missed
   IDs. Keep selected-channel and system scopes separate. Preserve the existing
   Anomaly+Rare Event benchmark definition and additionally report anomaly-only
   performance; Mission2's rare-event-heavy composition matters. The anomaly-only
   analysis must explicitly document how rare events are treated for false alarms.
5. Compare methods within the same Mission2 task. A higher Mission2 F0.5 would not
   itself improve the original Mission1 result or establish robust cross-mission
   transfer. Start with the unchanged short-no-roll method as a matched reference;
   avoid new postprocessing searches in this generalization test.

## Reproducibility

Protocol/code freeze: d66ad58. Runner:
scripts/research/mission_cross_channel_review.py. Saved run:
results_longrun/mission_cross_channel_20261102_v1. All166 thesis tests passed.
The independent verifier checked20 output hashes, replayed5969 raw-source sample
checks, all1855 correlation rows and all100 similarity distances/ranks. It uses a
separate correlation implementation and pandas source lookup, including checks
around annotation boundaries. It does not certify complete source-to-grid replay
at every timestamp or any unperformed neural assessment.

Compact evidence: [tables/mission-cross-channel-2026-11-02](tables/mission-cross-channel-2026-11-02/).
Both successful and weak correlations, all candidates, the frozen protocol and
verification results are retained. Arrays and the figure belong in the Release,
not source Git. No existing checkpoint, prediction, dataset or workbook was changed.
