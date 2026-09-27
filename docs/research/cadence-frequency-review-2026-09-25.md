# Cadence frequency audit — provisional, integrity checks failed

Scope: native timestamps for inputs40–47, training2000–2003 and calibration
January–March2004. No training or assessment analysis. Do not use these new
results to select preprocessing or justify a new experiment until reproduced
on a reliable data-reading environment.

## Initial pass findings (provisional)

Training median native cadence was90s for40 and30s for41–47. For41–46,
observation age>60s occurred in approximately0.1117% of training grid times
and0.0591% of calibration grid times. The initial8000-window nominal reference
sample had two windows per channel41–47 containing such old readings; this is
not the full training optimization dataset. Those very small groups had higher
reconstruction errors. Neither sample size nor this association proves causation.
Full-period frequency summaries include annotated periods, whereas the sampled
reference windows were wholly nominal under the existing loader.

One initial-history window begins before the first native observation. The
revised script separates it rather than classifying it as fresh, but that full
verification did not complete. Initial tables therefore remain provisional.

## Failed verification sequence

1. First full eight-channel pass completed and wrote frequency tables.
2. Verification rerun failed CRC32 while unpickling channel43.zip.
3. Two separate channel43 SHA256 reads agreed on
   `e664ea5b64ccea49796f32c9fd97cb6584c16d837442aebb2a6926c6f7413289`;
   both ZIP CRC checks passed. This is consistency, not an upstream golden hash.
4. Another full rerun failed strict timestamp ordering when processing channel44.
5. A separate channel44 read found zero nonincreasing timestamp pairs.
6. The immediately following channel44 read failed ZIP CRC32.

No raw files were modified. Two later successful reads do not resolve an
intermittent failure. The cause is unknown; these observations do not identify
a defective hardware component or prove permanent on-disk corruption. Earlier
CSV integrity incidents make this a priority before further modelling work.

## Decision

Pause new data-driven modelling conclusions and training. Preserve failed-run
evidence and seek server-administrator help checking memory/storage/filesystem
and system logs. Avoid silently sorting timestamps, skipping CRC errors, repairing
archives or overwriting source data. Reproduce the audit only after read integrity
is stable. Existing scores are not silently invalidated or rewritten by this
finding, but the new audit cannot be treated as verified evidence.

Evidence directory: `tables/cadence-frequency-2026-09-25/`, explicitly marked
`STATUS.json` as provisional. The script is
`scripts/research/cadence_frequency_review.py`; no model changes were made.
