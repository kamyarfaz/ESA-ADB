# Full-period observation-age comparison

Prepared 26 September 2026. No new training results yet.

## Agreed experiment

Inputs: telemetry channels 40–47. One model per comparison arm.
Train on 2000–2005; calibrate on 2006; freeze both decisions before assessing
2007–2013 continuously. Report pooled and calendar-year event F0.5, precision,
recall and event counts for input-channel and all-channel system scopes.
Use all eligible globally nominal stride-16 windows of length 256, without a cap.

Both arms use the same Transformer architecture, initialization, pseudo-anomaly
objective, seed and five-epoch budget. The control receives zero timing context;
the candidate receives causal log1p(observation age / training median cadence)
and observation availability. Only telemetry is reconstructed. Select each arm's
checkpoint and threshold using calibration only, with earliest checkpoint ties.
This five-epoch comparison is not equivalent to the historical 40-epoch baseline.
Previously inspected assessment years are not an untouched test.

## Data verification and recovery

The original full-period worker build remains preserved at
`results_longrun/diagnostics/observation_age_full_verified_v1`.
All 16 independent channel-feature NPZ copies matched recorded hashes in the
26 September audit. Two timestamp copies failed. Channel42 attempt1 contained
one differing int64 at index14611207: actual1385021005705032704 versus expected
1385021010000000000 (XOR 2**32). Root cause remains unresolved.

A new deterministic 30-second grid for [2000-01-01,2014-01-01), 14,728,320 rows,
was generated in a separate recovery directory. Its SHA256 exactly matched the
original recorded checksum, 351a18bbcaae6df7ca4d81d0b97ef35148f977793995a878fda355eeb4d42b7f.
Feature arrays were copied unchanged. Independent streaming hashlib and external
sha256sum verified both worker copies and assembled files. Recovery provenance
is embedded in the final protocol; original damaged files were not overwritten.
Accepted sidecars: `results_longrun/diagnostics/observation_age_full_ready_v1`.
Audit evidence: `outputs/integrity-review-20260926/`.
Successful checks do not establish that the intermittent fault is fixed.
Training performs additional initial and final integrity checks and fails on mismatch.

## Run

In a server terminal with CUDA available, use an existing tmux session or create
one with `tmux new -s esa-timing` if outside tmux. Then:

```bash
conda activate timeeval
cd ~/projects/ESA-ADB
python -m esa_thesis timing-comparison \
  --timing-dir results_longrun/diagnostics/observation_age_full_ready_v1 \
  --output results_longrun/timing_comparison/full_period_seed42_v1 \
  --device cuda --batch-size 64 --epochs 5 --seed 42
```

After interruption, repeat the same command with `--resume`; source and protocol
must match. Detach tmux with Ctrl-b then d. Do not modify experiment source while
running. Expected outputs: `pooled_results.csv` (4 rows), `yearly_results.csv`
(28 rows), plus per-arm checkpoints, histories, rules and predictions.

## Validation

68 thesis tests pass, including timing alignment, matching initial predictions,
candidate gradients, training/resume, and freezing both arms before assessment.
The assistant session reports CUDA unavailable; no full training was launched here.
