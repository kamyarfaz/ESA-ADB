# Cross-mission channel research and whole-system study

Status: research phase; training configuration has not yet been frozen.
Authorized 18 November 2026. The user explicitly confirmed **whole-system events
with separate training, calibration and assessment periods**, and requested
progress messages every three hours. This supersedes the completed-study hold
in AI_HANDOFF; it does not reopen or alter any historical result.

## Questions and evidence boundaries

1. What channel information is actually public? Consult the ESA authors' paper,
   official repository and version-1 dataset metadata. Do not infer physical
   temperature/power identities from anonymous unit numbers or correlations.
2. Which Mission2 signals resemble the Mission1 inputs, including33/49 and
   14/21/29? Measure marginal behavior and relationships between channels, not
   simultaneous cross-mission correlation. The missions' clocks are independently
   shifted/scaled and channel groups are independently normalized.
3. How much of each mission's event coverage is represented by the old inputs?
   Use training annotations to inform coverage decisions. Any discussion of the
   already-viewed assessment is explicitly post-hoc, never an input ranking.
4. Can the existing Mission1 context models work on Mission2 under a fixed,
   training-derived input mapping? Retain all three techniques and seeds42–44.
   Separate unchanged source scaling from any target-training normalization.
5. Train the three techniques afresh on each mission and assess whole-system
   events. Freeze the concrete channel panels, preparation and compute plan
   after research, before neural training or new assessment.

The previously viewed Mission1 and Mission2 assessment records are reused
benchmarks, not new independent validation. No performance guarantee is made.

## Research phase fixed before computation

Recheck the183 extracted raw-file SHA256 hashes against the official-version-1
archive-verified receipt, and the saved training grid/matrix hashes. Reuse only
Mission1 2000–2005 and Mission2 2000 for channel selection. Cached five-minute
grids use causal observations, maximum age600s, and per-channel nominal masks.
No later values or labels enter profiles, coverage ranking, or mapping.

Mapping source inputs are ordered40,41,42,43,44,45,46,47,33,49,14,21,29.
Candidates must be target channels with identical *anonymous* subsystem and unit.
One distinct Mission2 channel per source input; missing-compatible candidates
cause a recorded failure, not duplicated inputs or relaxed metadata constraints.
Features are the existing quarterly standardized quantiles, autocorrelations
at1/6/12/72/288 five-minute steps, and normalized change statistics. Feature
differences are divided by the pooled cross-channel IQR (minimum0.1), clipped
at3 and divided by3; equal averages within and across the three feature families.
Add equal-weight costs for discrepancies in within-mission level and first-change
Pearson matrices (absolute differences divided by2). The final mapping minimizes
the mean of unary, level-relationship, and change-relationship costs using an
initial Hungarian assignment plus deterministic local reassignment/swap search,
eight starts, at most50 improving moves per start. This is a heuristic assignment,
not a recovered physical identity, calibrated probability or global-optimum claim.
Random search starts use seed20261118, unrelated to neural seeds.

Repeat mapping after omitting each calendar quarter-of-year from both missions'
training summaries, solely as sensitivity analysis. The all-training mapping is
the sole primary map; sensitivity maps will not be evaluated to select a winner.
Keep signed relationships; no arbitrary sign inversion or time warping.

Report training event coverage of historical base/expanded inputs, the entire
compatible family, the mapped panel and all target channels, for Anomaly+Rare
Event and anomaly-only. These counts describe annotated-channel membership,
not an upper bound on possible event detection via correlated measurements.

## Execution and publication gates

Use new outputs under results_longrun/cross_mission_transfer_20261118_v1.
No historical source, model, rule, array or report is overwritten. Preparation
must be label-independent for input values; labels may exclude training windows
and provide calibration/evaluation targets. Test causal sampling, split boundaries,
mapping constraints and evaluator behavior. Independently check preparation and
freeze raw/prepared/source/runtime fingerprints before a sequential user service.
No duplicate worker or unrelated experiment. Same-checkpoint recovery only after
cause, available resources and frozen fingerprints are verified. Never change
frozen settings to bypass a failed run.

All fresh model/calibration choices must freeze jointly before new assessment.
Existing-source transfer choices must also freeze before Mission2 inference.
Use corrected event F0.5, event TPR, nominal-time FPR and detected/missed raw IDs;
report whole-system primary plus input-scope diagnostics, both event definitions,
annual and pooled results. Preserve every seed and failure. Independently replay
grids, selections, inference spots, predictions, metrics and IDs before final
W&B/report/workbook/GitHub publication. Preserve historical workbook rows and
verify fresh downloads of essential Release artifacts. Keep the full external
backup; scoped raw verification does not resolve unrelated migration exceptions.

Automation: cross-mission-channel-and-transfer-study, every3hours, including
healthy progress explicitly requested by the user. Disable the completed service
and delete this follow-up only after verified results have been delivered.
