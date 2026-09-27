# Counter aggregation review — 24 September 2026

Training-only native data, 2000–2005, channels 4–11. No prepared dataset changes.

## Findings

Across these channels there were **5,340 observed numeric changes**.
The old forward-difference/point-hold geometry misses **4,714 (88.3%)**
of their hold intervals because no 30-second grid point falls inside them.
These are channel-level changes, not unique anomaly events or detection recall.
Annotation-dependent restoration is not simulated here.

A causal candidate assigns each difference to its later observation (when it
becomes computable), then aggregates into the next 30-second boundary. Counts
retain all 5,340 observed changes in this interval; no changes land beyond
its end. Several changes can share one bin. Maximum added bin delay is 29.67s.
Count, signed sum and absolute sum are candidate features. Signed sums can cancel;
none of the occupied bins in this sample cancel exactly, but a synthetic check
confirms the risk. Counts/absolute sums retain evidence of such activity.

Native sampling has gaps up to 21,420s (5h57m). A bin with no recorded change must
not automatically mean healthy constant telemetry. A future representation needs
observation counts and time since last observation, plus an explicit missingness
policy. Observed changes here span at most 180s between observations; unobserved
activity during gaps cannot be recovered. The first sample has no invented delta.

## Decision

The counter representation warrants a separate versioned prototype, not deletion
of channels 4–11 or an immediate full neural run. Freeze bin semantics as
(t-30s,t], emitting at t; compute differences using only available observations.
Raw timestamps are assumed availability times; true packet arrival times are not
provided. Reset/wrap semantics require investigation before treating signed
changes as counts of physical activity. Do not forward-fill change pulses.

Before any model comparison, measure the prototype against training annotations,
verify missingness and normalization, and keep original prepared inputs intact.
This audit does not establish better F0.5 or benefit to baseline channels 40–47.

## Evidence and validation

Script: `scripts/research/counter_aggregation_review.py`.
Tables/protocol: `tables/counter-aggregation-2026-09-24/`.
Validated timestamp order, finite values, conservation of changes, causal delay
bounds, source size/mtime stability, and synthetic boundary/cancellation cases.
Only raw training observations with both difference endpoints inside the interval
were used. No calibration/test data or model training was used.
