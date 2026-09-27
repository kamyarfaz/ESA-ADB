# Native timestamps versus prepared calibration alarms

Read-only review of channels40–47 for the three frozen2004 calibration alarms,
including four-hour before/after contexts. Sources are trusted local mission
channel ZIP pickles and the completed diagnostic's prepared-input cache. No
training, threshold selection or assessment access.

## January24 false alarm

- Channels41–46 each have209 native observations during the192-minute alarm,
  versus384 prepared30-second samples. Channel47 has199 native observations.
- Native observation intervals in the surrounding context include210seconds
  repeatedly and up to240seconds for channels41 and47. Intervals above100seconds
  span approximately05:57:55 to09:10:25UTC, close to the06:08–09:20 alarm.
- During the alarm, the age of the latest native observation reaches214.812s
  on channels41–47. No observation interval exceeds300s in any reviewed context;
  a generic five-minute gap flag would therefore miss this cadence change.
- Channels41–46 repeat values in52.5–61.1% of prepared successive samples,
  versus7.7–23.1% of successive native observations during the alarm.
- Every compared prepared value across all eight channels, alarms and available
  contexts matches the most recent native value within rtol1e-6/atol1e-7.

Thus the flat prepared sections are consistent with holding values during longer
native observation intervals. This is evidence of observation-cadence change,
not proof of packet loss, a corrupted dataset or a particular spacecraft mode.
The counts differ because the time bases differ; they do not directly estimate
a lost-packet percentage. Channel40 has a different native cadence and must not
be treated as a30-second sensor.

## Controls

For the January1 detected event and February14 false alarm, latest-observation
age on channels41–47 stays below4.815seconds during their alarms. Their native
and prepared repetition fractions are much closer. The February14 false alarm
therefore cannot be explained by this same stale-observation pattern from the
evidence reviewed. Channel40 normally has older held values in these controls.

## Implication

Missing observation timing is a credible modelling/preprocessing hypothesis for
the January24 alarm. It remains an association, not a causal test. Do not suppress
alarms during sparse sampling or relabel these periods based on this finding:
true anomalies could also occur under sparse sampling.

Before introducing observation-age features or changing filling, quantify their
frequency in training and calibration separately, using each channel's native
cadence as reference. Any new preprocessing must be defined from training and
compared prospectively with the unchanged baseline. This finding does not justify
another architecture sweep. Inputs remain40–47 and the baseline remains retained.

Evidence: `tables/native-alarm-review-2026-09-25/native_vs_prepared.csv` and
`native_gaps.csv` (no gaps>300s). Reproduction script:
`PYTHONPATH=. python scripts/research/native_alarm_review.py` from repository root.
The script checks strictly increasing native timestamps; local before/after
contexts are clipped to available calibration cache boundaries. Raw files were
read, not modified; no new cryptographic verification of their upstream provenance.
