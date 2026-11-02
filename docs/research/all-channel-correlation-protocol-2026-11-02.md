# All-channel correlation matrices for both missions

User request,2 November2026: provide matrices for both missions and identify
highly correlated channels. This extends the earlier eight/22-channel study to
all76 Mission1 and100 Mission2 telemetry channels. No training or test evaluation.
No existing experiment files are modified.

Use Mission1 [2000-01-01,2006-01-01) and Mission2 [2000-01-01,2001-01-01).
Retain the previous five-minute, causal, maximum-ten-minute-age sampling and
per-channel annotation exclusion, including intervals crossed by a held value.
No label-driven numerical restoration, interpolation or arbitrary encoding of
categorical values. Exclude all annotation categories for nominal diagnostics.
Verify every raw input against the scoped destination-integrity receipt from
mission-cross-channel-2026-11-02; use all channels, not only target channels.
Previously verified diagnostic grids may be reused after exact file hash checks.

Show all channel IDs in the matrices. Numeric telemetry and binary numeric flags
can have Pearson/Spearman correlations. Non-numeric channels, constant channels,
or pairs with fewer than100 eligible samples have undefined correlations (N/A),
never zero. Known counters (Mission1 4–11; Mission2 29–46) retain raw levels in
level matrices and are explicitly flagged. Their high shared trends do not
establish anomaly usefulness. Do not rank arbitrary categorical codes.

Calculate pooled and quarterly Pearson levels, Spearman levels and Pearson
adjacent five-minute changes. Differences never bridge invalid samples or quarter
boundaries. Report pair sample counts. Pooled high correlation means |r|>=0.90.
A stable high pair additionally has the same sign, |Pearson r|>=0.90 in EVERY
training quarter, and at least500 eligible paired samples in every quarter.
Strong changes are a separate flag: same sign and |change r|>=0.50 in every
quarter, with at least500 change pairs each quarter. Show signed coefficients.
All flags use unrounded values. Low Pearson r does not rule out nonlinear or
lagged relationships; correlation does not prove causality or improved F0.5.

List all unique pairs (2850 +4950), with filters for stable pairs, target/non-target
roles and counters. Provide high-correlation groups using complete linkage on
1 minus the minimum absolute quarterly correlation (unstable signs/undefined or
insufficient coverage give distance1). Cut at0.10. Thus every pair within a
reported group satisfies the stable high rule; groups are a convenient partition,
not an exhaustive list of all overlapping cliques. Target-only groups are listed
separately from all-channel groups to avoid confusing counters with targets.

Deliver one workbook with all six matrices, filterable pair tables and channel
metadata, plus PNG/PDF heatmaps and CSV matrices. Include focused matrices for
the original Mission1 eight channels and proposed Mission2 eight channels.
Use a fixed [-1,+1] colour scale and distinguish undefined entries from zero.
Dates are separately anonymized/scaled, so do not manufacture a cross-mission
date-aligned correlation matrix. Cross-mission similarities remain the preceding
report's descriptive findings. No fresh channel/model selection using assessment.

Validate matrix symmetry, bounds, undefined diagonals, counts, stable flags and
exact parity with previous raw-data diagnostic subsets. Independently replay
pooled correlations for all7800 unique pairs and quarterly coefficients for all
reported stable high pairs. Verify saved workbook values against computed arrays,
render all sheet types, and preserve source/evidence in Git and a verified Release.
