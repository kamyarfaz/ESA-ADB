# All-input investigation — Mission 1

## Scope and purpose

Investigate all available inputs before choosing model inputs. Preserve every
source channel. Input omission is a model configuration decision, not deletion.
Neither more nor fewer channels are assumed to yield better event F0.5.

The inventory is 76 telemetry channels and 698 telecommand event types. Current
preprocessing retains all 76 telemetry channels but only the 11 telecommands with
metadata Priority >=3. That historical rule is not evidence that the other 687
commands carry no information.

This review uses [2000-01-01, 2003-01-01), the first fold's training period. It
covers all source inputs, not all mission years. Later calibration/assessment data
are not used to decide usefulness. Selection for later expanding folds must be
fitted within their permitted training interval; this is not a universal final
input list for every year.

## Evidence produced

- `all_inputs.csv`: a row for each of 774 source inputs, with observations,
  unique payload values, telemetry changes, time ordering/duplicates/non-finite
  checks, active months, metadata and inclusion in current prepared data.
- `monthly_raw_summary.csv`: 36 months per source input. Telemetry summaries use
  all native observations and include annotated periods. Telecommands record
  occurrences; they are not continuous measurements.
- `input_review.csv`: source inventory joined to existing full-resolution nominal
  prepared profiles where available. Empty profile fields for the other 687
  telecommands mean **not prepared**, not zero information.
- Ten telemetry atlas pages cover all 76 channels. Monthly minimum–maximum bands
  retain extremes but do not display exact short-event timing. The command heatmap
  covers all 698 types, with log-scaled monthly occurrence counts.
- Existing exploratory Pearson/Spearman evidence is retained separately. It uses
  hourly globally nominal prepared samples; it can miss rare or nonlinear signals.

Reproduce using `python scripts/research/all_input_review.py --output NEW_DIRECTORY`
with pandas, numpy and matplotlib. Inventory/file names and expected row counts
are asserted. Source size/mtime is checked around every pickle read. This is not
cryptographic verification or a complete raw-to-prepared reconstruction.

## How input decisions will be made

1. **Availability and quality:** distinguish no observed training occurrences,
   missingness, cadence, invalid values and preprocessing artifacts.
2. **Variation:** inspect rare changes at native resolution; constant nominal
   behavior can be informative when it changes abnormally. Channels 8–11 illustrate
   why nominal constancy alone must not trigger input exclusion.
3. **Complementarity:** review stable and changing correlations, subsystem groups,
   and event context. High correlation is a candidate for a controlled ablation,
   not proof of redundancy. No physical meaning is invented from anonymized codes.
4. **Small controlled validation:** freeze broad and reduced candidate lists, then
   compare the same windows, preprocessing, model budget and temporal validation
   protocol. Compare event F0.5, precision, recall, false alarms and resource use.
   Do not mix changing channel count with changing training duration.

No neural model or new preprocessing is run by this review. No validated final
subset or improved F0.5 is claimed. A command with constant payload 1 may still
encode valuable occurrence timing. A command absent from 2000–2002 can be omitted
from a first-fold learned input representation provisionally, but absence says
nothing about its usefulness in later periods.

## Remaining work

This is a broad descriptive review. Full-resolution cross-channel lag/relationship
analysis, telecommand-to-telemetry context, a label-independent preprocessing
policy and small matched validation remain necessary before selecting inputs.
Do not start another full training campaign merely because this inventory exists.

## Results of this pass

- All 774 metadata entries match raw source files; none is missing from inventory.
- 134,499,893 native telemetry observations and 368,116 telecommand occurrences
  were reviewed in the training interval.
- All 76 telemetry channels have more than one raw value. No telemetry input is
  justified for omission merely by claiming raw constancy.
- 458 telecommand types occur; 240 have no training occurrences. Of the active
  types, 447 are outside current preprocessing. All 11 currently included types
  occur in raw training, even those constant in the prepared nominal profile.
- No non-finite values, duplicate timestamps or backwards timestamps were found
  in the training observations. This does not establish gap-free data.
- Of 184 previously flagged exploratory correlation pairs, 26 maintain absolute
  Pearson correlation >=0.95 with the same sign in each training year. These are
  review pairs, not validated interchangeable inputs; within-year regime changes,
  sampling bias and rare events still require investigation.
- All 27,864 monthly records passed uniqueness and aggregate-count checks.
  Ten atlas pages cover every telemetry channel, and one command activity plot
  covers all 698 command types.

Tables: `tables/all-input-review-2026-09-23/`. Plots:
`../../outputs/all-input-review-20260923/`. No final input exclusions have been
applied. The 240 absent commands are the only straightforward first-fold
no-observation candidates; sources remain preserved for later training periods.
