# Proposed cadence amendment — not approved or launched

The frozen v1 study stopped in independent preparation verification before any
neural run. Mission1 channels62 and63 both have median and99th-percentile raw
training intervals of900 seconds. The600-second age limit makes each channel
incapable of providing a complete256-sample window on the30-second grid. Thus
the union training mask has zero eligible windows even though3,743,123 individual
rows are nominal and complete. These times refer to the anonymized dataset clock.

This is a preparation-design failure, not evidence of poor model performance,
insufficient hardware, corrupted source files, or a failed neural seed. Mission2's
training/calibration source preflight passed, but the all-mission preflight gate
correctly prevented all18 training runs. There are no new assessment results.

## Concrete proposed change

Subject to explicit user authorization, preserve v1 unchanged and introduce a
separately versioned, documented preparation protocol. For Mission1 channels62
and63 only, allow the most recent finite observation to remain valid for at most
900 seconds, matching their observed training sampling interval. Retain the
600-second limit for every other input, including all Mission2 and transfer
inputs. Apply each fixed limit identically in training, calibration and assessment;
do not estimate limits from calibration/assessment or use future observations.

Keep every selected channel, split, seed, method, epoch count, batch, model,
calibration objective, mapping, transfer condition, and joint18-selection gate.
Rebuild and independently verify preparation, training masks and scalers before
launch. Preserve the failed v1 preparation, source, protocol and logs. Establish a
new explicit source/runtime/data freeze; do not modify or restart the failed v1.
Use only one active training service, with the existing resource and duplicate
worker checks. Publication must describe this amendment and the failed preflight.

## Evidence and limits

A diagnostic performed full pandas causal source lookup for both slow channels,
including non-finite source values, on Mission1 training2000–2005 only. It also
independently reconstructed the existing annotation/held-observation exclusions
and reproduced the original training mask exactly. All other channels' saved
availability was unchanged. The proposed900-second rule yields349,383 nominal
training windows; original600-second rule yields0. Per-channel availability rises
from66.4869% to99.7291%. This establishes preparation feasibility only. It does not
establish that holding a measurement is physically accurate, improve detection,
or justify any performance claim. Held readings remain stepwise and may hide
changes between observations. Missing intervals beyond900 seconds remain missing.

An earlier age-only diagnostic considered2700 seconds. That was an unlaunched
diagnostic, not a frozen or recommended rule; it did not independently validate
finite source values. This proposal uses the observed900-second interval and the
full finite-value check. No neural outcomes selected either rule.

Dropping channels62/63 would instead remove the selected representation of a
target group and alter the whole-system input design. It is not proposed as a
silent fallback. Retaining the frozen600-second rule requires leaving v1 stopped.

The current heartbeat explicitly says, “Do not ... change frozen settings to
bypass failure.” The execution protocol also fixes maximum age600s and same-
settings recovery. Therefore this substantive preparation revision requires the
user to authorize a new protocol before implementation or launch. No approval
has been inferred from elapsed time or the general request to finish the study.

Evidence: [cadence feasibility](tables/cross-mission-2026-11-18/cadence-feasibility-diagnosis.json).
