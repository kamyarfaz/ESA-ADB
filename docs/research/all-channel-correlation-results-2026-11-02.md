# All-channel correlation matrices: Mission 1 and Mission 2

All 76 Mission 1 channel IDs and all 100 Mission 2 channel IDs are included in
six matrices: Pearson signal levels, Pearson adjacent changes, and Spearman
signal levels for each mission. This is a training-only descriptive analysis.
It does not train a detector, evaluate anomalies, or establish an improved F0.5.

Download the workbook, eight-page matrix PDF and CSV collection from the
[artifact Release](https://github.com/kamyarfaz/ESA-ADB/releases/tag/all-channel-correlations-2026-11-02).
The workbook contains all six matrices, a filterable list of all 578 pooled-high
pairs, stable groups and the 176-channel inventory. The CSV collection additionally
contains all 7,800 unique pairs, including weak and undefined relationships,
sample-count matrices and verification receipts. Matrix entries remain numeric
at full saved precision; displayed rounding does not determine any flags.

## Scope and interpretation

Mission 1 uses 2000–2005; Mission 2 uses 2000. These are the training windows
fixed in the preceding investigation. No calibration or assessment observations
selected these correlations. The different durations mean that pair counts and
stability rates are not a matched-duration comparison between missions.

The [frozen protocol](all-channel-correlation-protocol-2026-11-02.md) uses causal
five-minute sampling, a maximum observation age of ten minutes, and channel-wise
exclusion of annotated intervals, including intervals crossed by a held value.
Differences never bridge invalid samples or quarter boundaries. Raw counters are
retained and flagged. Labels only exclude intervals for these nominal diagnostics;
they do not numerically replace measurements. These grids are not model inputs.

- **Pooled high:** absolute Pearson level correlation at least 0.90.
- **Stable high:** additionally, the same sign and absolute level correlation at
  least 0.90 in every training quarter, with at least 500 paired samples in each.
  Mission 1 has 24 quarters; Mission 2 has four.
- **Stable changes:** a separate, less strict descriptive flag requiring the same
  sign and absolute change correlation at least 0.50 in every quarter, with at
  least 500 change pairs per quarter. It does not use the level threshold of 0.90.
- **N/A / gray:** categorical, constant on the paired sample, or fewer than 100
  eligible observations. N/A is not zero correlation.

A positive coefficient means movement in the same direction; a negative one
means movement in opposite directions. Near-zero Pearson correlation does not
rule out nonlinear or delayed dependence. These are descriptive effect sizes,
not significance tests, evidence of causality, or a guarantee of anomaly utility.

| Mission | Channels | Unique pairs | Pooled high pairs | Stable high pairs | Stable high target–target pairs |
|---|---:|---:|---:|---:|---:|
| 1 | 76 | 2,850 | 440 | 1 | 1 |
| 2 | 100 | 4,950 | 138 | 84 | 82 |

## Mission 1: strong overall relationships and their limits

| Channels / pair | Pooled level r | Pooled change r | Quarter check |
|---|---:|---:|---|
| 62 ↔ 63 | 0.999980 | N/A | Stable high; lowest quarterly absolute r = 0.994045 |
| Every pair within 16, 24, 32, 33, 40 | 0.998259–0.998496 | −0.046168 to 0.003501 | None meets the every-quarter level criterion |
| 41 ↔ 44 | 0.908796 | 0.715440 | Lowest quarterly level r = 0.892649; stable changes |
| 42 ↔ 45 | 0.891341 | 0.682886 | Below the 0.90 pooled-high threshold; stable changes |
| 43 ↔ 46 | 0.887319 | 0.689359 | Below the 0.90 pooled-high threshold; stable changes |
| 40 ↔ 47 | 0.883047 | −0.000035 | Lowest quarterly absolute level r = 0.009490 |

The 440 pooled-high pairs are real measurements under the stated procedure; the
strict quarter test does not make them uncorrelated. It shows that most strong
overall associations are not uniformly strong locally. For example, 33 ↔ 40 has
pooled r = 0.998496 but its minimum quarterly absolute r is 0.445618, and its
change r is 0.003501. Shared slow variation can dominate a multi-year coefficient.
The 62 ↔ 63 change coefficient is undefined on the eligible adjacent samples,
so the near-perfect level coefficient alone does not establish fast coupling.

Within the original 40–47 range, 41 ↔ 44, 42 ↔ 45 and 43 ↔ 46 show the clearest
paired short-term relationships. The range is not a uniformly highly correlated
eight-channel group. Both the complete matrix and an annotated focused matrix
are provided. The raw five-minute results do not replace the older prepared,
full-resolution annual audit: sampling, masking and quarter versus annual
granularity differ. The prior raw diagnostic subsets match this extension exactly.

## Mission 2: stable groups

All pairs within each target group below pass the every-quarter level criterion.
The first two six-channel groups also have almost identical short-term changes.

| Channel group | Pooled level r across pairs | Pooled change r across pairs | Lowest quarterly absolute level r |
|---|---:|---:|---:|
| 75, 76, 77, 83, 84, 85 | 0.997573–0.998379 | 0.996292–0.997564 | 0.996988 |
| 70, 71, 72, 78, 79, 80 | 0.992293–0.995892 | 0.992396–0.993721 | 0.989576 |
| 73, 81 | 0.946012 | 0.991840 | 0.939359 |
| 58, 59, 86, 87, 88, 89, 90, 91 | 0.995033–0.999019 | −0.254526 to 0.682992 | 0.992895 |
| 74, 82 | 0.998756 | 0.629852 | 0.997631 |

Channels 74 and 82 are also strongly **inversely** related to every member of
58, 59, 86–91: the 16 cross-group coefficients range from −0.995901 to −0.992989.
Thus these ten channels form one absolute-correlation group, with two opposite
sign subgroups. Their change relationships are much weaker and more variable
than those of the first two six-channel groups.

Two additional stable pairs involve non-target channels and should not be
mistaken for interchangeable anomaly targets:

| Channels | Role | Pooled level r | Pooled change r |
|---|---|---:|---:|
| 61 ↔ 62 | Non-target channels | 0.999998 | 0.982508 |
| 40 ↔ 42 | Non-target counters | 0.998233 | 0.994874 |

The workbook's complete-linkage groups are a convenient disjoint partition, not
an exhaustive list of all overlapping groups. Its pair table lists all 84 stable
pairs, including relationships outside those partitioned groups. Filter Mission,
Stable high, Both targets and Any counter to inspect the intended subset.

The ten categorical channels are 4, 5, 47, 50–55 and 60. Channels 29–37 are
constant over the eligible Mission 2 training window. Their Pearson/Spearman
matrix rows are N/A. Channel 100 has only 1.389% eligible five-minute coverage;
the pair counts and coverage fields prevent that sparse series from being
treated as a complete observation record. Categorical-channel numeric coverage
of zero means ineligible for this numerical analysis, not absent raw telemetry.

## How this informs the mission comparison

These are correlations **within each mission**, not coefficients between missions.
The [ESA-AD authors, Supplementary Sections 2.3–2.4](https://arxiv.org/pdf/2406.17826)
describe independently shifted and scaled time axes. Matching the anonymous
dates would not align real measurements. Channel numbers are also mission-specific.
The [preceding correspondence report](mission-cross-channel-results-2026-11-02.md)
documents metadata-compatible groups and descriptive cross-mission similarities.

For a related Mission 2 experiment, 75, 76, 77, 83, 84, 85 and the alternative
70, 71, 72, 78, 79, 80 are coherent groups to investigate. The prior proposed
eight-target task remains 58, 59, 75, 76, 77, 83, 84, 85, preserving the original
two-plus-six subsystem/unit composition. This is a candidate analogous task,
not a proven eight-to-eight mapping or evidence that weight transfer will work.
No new training or model/channel search was started from these findings.

## Verification and preservation

Frozen protocol/code commit: 0fa3261. Saved run:
`results_longrun/all_channel_correlations_20261102_v1`. Every raw channel and
metadata input was checked against the scoped archive-integrity evidence from
the preceding investigation. Its verification covers all 183 extracted telemetry
and available metadata files; it does not certify unrelated migration exceptions.

Independent checks verified 19 output hashes; all 7,800 unique pooled level and
change coefficients and sample counts; 360 stable-pair quarter replays for both
levels and changes; 40 sampled Spearman coefficients; and all 1,644 earlier raw
subset matrix entries. The two previous raw grids were preserved exactly. A
direct paired-sample fallback in the verifier handles constant paired subsets
without accepting floating-point variance cancellation as a correlation.

All 169 thesis tests passed, including a regression check for that independent
verification edge case. Workbook validation compares every matrix cell and all
578 high-pair rows to the computed snapshot, checks the native filters, panes and
conditional formats, and inspects all nine rendered sheets. The source Git holds
compact matrices, tables, protocols and receipts. The Release holds the grids,
quarter arrays, workbook, figures, source snapshots, logs and a content manifest.
No existing model, historical result or thesis-results workbook was changed.
