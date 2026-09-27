# Isolating the differing observation-age calculation

No dataset rebuild or training was performed. The arithmetic probe operates on
fixed integer timestamps and synthetic arrays, independent of archive reading.

## Exact case

- Channel44, grid index4420357:2004-03-14 20:18:30UTC.
- Grid nanoseconds:1079295510000000000.
- Latest native observation:2004-03-14 20:18:25.179UTC,
  nanoseconds1079295505179000000.
- Exact integer difference:4,821,000,000ns =4.821s.
- Encoding: `log1p(4.821 / 30)`.
- Python double result:0.1490232713924399.
- Rounded float32 result: **0.14902326464653015**.
- Earlier differing output:0.14900924265384674.

The second of the previously compared worker outputs agrees with the independent
scalar calculation. The first differs by941 float32 representable steps at this
magnitude, so this is not an ordinary final float32 rounding difference.

## Standalone test

`scripts/research/single_age_calculation_probe.py` compares NumPy integer
subtraction, seconds conversion, division, log1p and float32 conversion against
Python integers/math and standard-library float32 packing. It also checks that
input arrays retain their original integer values.

Array sizes1,16,1024,262144 and4469760 were tested with10000,1000,100,20 and3
rounds respectively: **18,780,560 element evaluations**, all matching. Float32
comparison is exact; the float64 log comparison permits2ULP for library rounding.
The independent scalar loop also matched in100,000 trials. Environment:
NumPy1.26.4 and the research Python3.9 interpreter.

## Conclusion and limits

The specific formula and tested NumPy path agree with the scalar reference.
This test does not reproduce the intermittent fault, prove a NumPy-specific bug,
or identify hardware as the cause. Possible upstream value changes and other
intermittent computation/read issues remain unresolved. Do not rewrite the
encoding to compensate for a single bad cached number, patch that value by hand,
or treat a clean arithmetic probe as approval of the failed full dataset build.

The sidecar dataset remains unaccepted because its final combined verification
failed. All originals and failed artifacts are preserved. Evidence tables:
`tables/single-age-calculation-2026-09-25/`.
