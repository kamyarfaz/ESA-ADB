# Optional short coordinated augmentation

Implemented27September2026 in `esa_thesis/short_augmentation.py`. Experimental
primitive only: NOT wired into a training CLI, NOT trained, NO new F0.5 result.

`make_training_anomaly(x, mode="legacy")` directly delegates to the original
generator without extra RNG draws. Test confirms exact output and both Python/
Torch random states match the old call. Existing production call sites unchanged.

`mode="short_coordinated"` selects per window:50%legacy,50%short. Short branch
uses1–10samples (30seconds–5minutes at30second cadence),1–Cchannels, shared start/
duration/type. Offset direction and standardized amplitude are shared; noise
uses a shared waveform scaled perchannel; temporal roll uses shared shift. Length1
chooses offset/noise, avoiding a meaningless one-sample roll. Scale is each
selected channel's full-window population standard deviation plus1e-6. This is
a deliberate experimental choice, distinct from legacy segment-local scaling.

Input is never mutated. Optional metadata reports actual changed support and
no-ops (constant signals can remain unchanged under rolls; tiny changes may round
away). No claim all generated perturbations are detectable physical anomalies.
Runtime rejects nonfinite candidate inputs/outputs. Random mixture selection and
extra corruption draws mean future training is not sample-order-matched merely
because seeds match; a controlled comparison needs explicit RNG handling.

Validation:73thesis tests pass. New tests cover legacy output/RNG parity,
one-sample all-eight-channel finite offset and exact support, constant roll no-op,
seed reproducibility, mixture coverage/support bounds and invalid input rejection.

Next engineering step: wire this as an explicit opt-in comparison arm with its
mode/settings saved in protocol, equal budgets, calibration-only model selection,
and frozen decisions before development assessment. Do not silently enable it
in the current baseline. Source additions affect strict source-hash resume checks;
never resume an old run under changed source without its original environment.
