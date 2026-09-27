# Training scaling review — 24 September 2026

Exact full-resolution audit of 6,312,960 prepared samples from 2000–2005,
including 5,615,041 globally nominal samples. All 87 inputs (76 telemetry,
11 commands) were examined. No calibration or assessment period was used.

## Findings

- **No nominal training values exceed the ±10 clipping limit in any input.**
  Relaxing the clipping limit would not change nominal training inputs for this
  interval. This does not measure clipping of synthetic pseudo-anomalies or later data.
- **29 inputs have zero/near-zero IQR and use the existing scale-of-one fallback:**
  18 telemetry channels and all 11 prepared command inputs. A zero IQR means the
  middle half of values has no spread; it does not mean the entire channel is constant.
- Counter channels 4–9 and 11 have only 1–3 nonmedian nominal samples each;
  channel 10 has none. Their post-scaling standard deviations are approximately
  5e-9 to 6e-8 (zero for channel 10). Their contribution to squared reconstruction
  error may therefore be very small compared with regularly varying inputs.
  This is an input-scale observation, not a measured model ablation.
- This supports investigating counter/event representation and sparse-input scaling
  before a broad-input neural experiment. It does not justify deleting raw channels.

## Decisions and next comparison

Keep the completed channels 40–47 baseline unchanged. Do not increase clipping
limits as a speculative improvement: this audit found no nominal clipping.
Do not blindly divide sparse channels by tiny standard deviations; a few isolated
samples could receive disproportionate weight.

Next preprocessing measurement should compare current forward differences/point
sampling with causal backward changes aggregated into 30-second bins, on training
counter data only. Measure retained changes, timestamp alignment, missingness and
boundary handling before training. Commands require a separate event representation.

Then freeze one matched pilot comparing the existing baseline with a justified
correlated group or complementary extension; use the same training budget and
calibration protocol. Existing 14/21/29 experiments already cover one stable group.
The stable 16/24/32/33/40 group is a candidate for review, not an automatically
selected winner. No training or new input exclusion was performed in this audit.

## Reproduction and verification

Run `python scripts/research/scaling_review.py --output NEW_DIRECTORY`.
The audit matches production float32 median/IQR, scale fallback and clipping.
Synthetic ordinary, sparse and tiny-IQR inputs matched the production scaler
exactly. Full row/nominal totals, finite inputs, label values and source size/mtime
were checked. Compact tables and protocol are in
`tables/scaling-review-2026-09-24/`. Temporary numeric storage was removed by the
script after successful completion; source data was not modified.
