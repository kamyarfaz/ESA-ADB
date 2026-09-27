# Counter-channel review — training only

## Finding

Every channel from 4 to 11 changes in the raw 2000–2002 training interval.
All eight contain 593,156 native observations. In particular, channels 8–11
are constant in the earlier **prepared nominal** profile but not in raw data.
Do not describe them as empty raw channels or remove them on that assumption.

| Channel | Raw distinct values | Nonzero adjacent differences |
| --- | ---: | ---: |
| 4 | 173 | 172 |
| 5 | 173 | 172 |
| 6 | 394 | 393 |
| 7 | 193 | 192 |
| 8 | 293 | 292 |
| 9 | 293 | 292 |
| 10 | 76 | 75 |
| 11 | 170 | 169 |

None has a channel-local Anomaly/Rare Event annotation overlapping this training
interval. This does not prove these changes are nominal: other channels may be
annotated simultaneously, and gap/other annotations are not excluded here.
It also does not establish predictive usefulness. These are data diagnostics.

## Decisions and limits

- Channels 4–7: retain in the review pool; coarse sampling can miss rare variation.
- Channels 8–11: change the provisional removal-candidate status to **review
  transformation and global nominal filtering**. Raw changes are demonstrated.
- Baseline channels 40–47 and all historical results remain unchanged.
- No model training, channel deletion or preprocessing replacement occurred.

Next trace the raw changes through differencing, resampling and the all-channel
nominal mask to establish why variation disappears. Only then decide whether
these inputs belong in a reduced feature set. The raw telecommand audit and
broader redundancy review remain unfinished.

## Reproducibility

Run `scripts/research/review_counter_channels.py --output NEW_DIRECTORY` with
pandas, numpy and matplotlib. Both endpoints of every difference must lie in
[2000-01-01, 2003-01-01); the terminal value is not replaced by an artificial zero.
Counts use every native sample, not an hourly sample. The overview plot displays
every nonzero difference, and detail plots use native samples around the first
change. Source size/mtime stability, finite values, sorted unique timestamps,
row counts and the partition of difference counts were checked during execution.

Tables: `tables/counter-review-2026-09-23/`. Figures:
`../../outputs/counter-review-20260923/`. The presentation workbook was not updated.

Follow-up completed: see [preprocessing trace](counter-preprocessing-trace-2026-09-23.md) for the measured resampling and global-filter explanation.
