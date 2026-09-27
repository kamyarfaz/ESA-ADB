# Full Mission 1 descriptive input review: 2000–2013

## Scope correction

The available prepared Mission 1 dataset spans **14 full years**: training
2000–2006 and benchmark test 2007–2013. Its final January 1, 2014 row is a
boundary sample, not a fifteenth year. Earlier reports limited to 2000–2002 or
2000–2006 are partial-period studies, not the entire available dataset.

This report covers [2000-01-01, 2014-01-01), all 76 raw telemetry channels and
all 698 telecommand types. Every input receives a row in each year, including
explicit zero-observation rows. No channel is deleted or automatically excluded
from model inputs. No training is launched.

## What was checked

The read-only script loads trusted local source pickles and metadata, restricting
statistics to the 14-year interval. Annual records include observation counts,
active months, first/last observation, value cardinality, timestamp ordering,
duplicate timestamps and non-finite values. Telemetry records additionally report
minimum, maximum, mean, standard deviation, changes and positive sampling cadence.
The monthly telemetry envelopes use all native observations, not a sparse sample.
The 10-page atlas includes every telemetry channel and shades the benchmark test
period. Source size/mtime is checked around each read.

Counts are per-channel observations, not synchronized rows. Telecommands are
occurrence streams: an unchanged payload does not mean an uninformative signal.
No occurrence in a particular year must not be interpreted as no occurrence ever.
Changes and gaps are calculated within individual years, excluding cross-year
transitions. Raw statistics include all periods without a nominal-label filter.
Anonymized physical-unit and subsystem codes do not establish engineering meaning.

## Scope of the conclusion

This completes the descriptive annual inventory across all available full years.
It does not establish useful neural-network inputs, quantify detection accuracy,
or complete the full scientific data study. Monthly summaries do not preserve
short-event timing. Cross-channel relationships, command context, preprocessing
and controlled validation still need investigation before feature selection.

The scan opens no anomaly-label files or model scores. Nonetheless, 2006 and
2007–2013 values are now descriptively inspected. Record that exposure. Do not
use later-year patterns to select inputs, transformations or hyperparameters and
then claim independent performance on those same periods. Model selection remains
restricted to permitted training/calibration data. Historical test access and
results are preserved and must not be described as never examined.

## Reproduce

```bash
python scripts/research/full_mission_input_review.py --output NEW_DIRECTORY
```

Dependencies: pandas, numpy, matplotlib. Checks include metadata/file agreement,
10,836 unique input/year records, 12,768 telemetry/month records, and agreement
between monthly and yearly telemetry observation totals. Quality issues are
reported rather than silently repaired. Source size/mtime checks are not hashes
or a complete dataset-integrity certificate.

Tables: `tables/full-mission-review-2026-09-23/`. Figures:
`../../outputs/full-mission-review-20260923/`.

## Verified findings

| Year | Telemetry channels with variation | Active command types |
| --- | ---: | ---: |
| 2000 | 76/76 | 399/698 |
| 2001 | 76/76 | 286/698 |
| 2002 | 76/76 | 266/698 |
| 2003 | 76/76 | 228/698 |
| 2004 | 76/76 | 308/698 |
| 2005 | 76/76 | 322/698 |
| 2006 | 76/76 | 226/698 |
| 2007 | 76/76 | 505/698 |
| 2008 | 72/76 | 263/698 |
| 2009 | 76/76 | 540/698 |
| 2010 | 73/76 | 251/698 |
| 2011 | 76/76 | 246/698 |
| 2012 | 76/76 | 211/698 |
| 2013 | 74/76 | 279/698 |

- All 76 telemetry channels have observations in all 14 years and vary somewhere
  over the full period. This is not evidence that all are useful model inputs.
- Raw channels 8–11 are constant throughout 2008; 8–10 throughout 2010;
  8–9 throughout 2013. These are annual properties, not lifetime constancy.
- 681 telecommand types occur over 2000–2013; 17 have no occurrences in this
  interval. Their source files remain preserved and the list is descriptive.
- 744,856,895 native telemetry observations were summarized. Counts rise in
  2012–2013; that alone does not establish improved coverage or data quality.
- No non-finite values, duplicate timestamps or backwards steps were found in
  the annual source scans. This does not prove gap-free sensing.
- All 2000–2006 annual records agree with the previous audit across all shared
  fields. Full-period monthly/yearly count checks passed.

The updated atlas has 10 pages covering all 76 channels. No final feature subset,
training run or F0.5 improvement follows from these descriptive findings.
