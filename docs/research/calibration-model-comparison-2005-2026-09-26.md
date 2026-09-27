# Both saved models on full2005calibration

Full Jan–Mar2005 scored for reconstruction-only checkpoint; pseudo scores reused
from verified diagnostic. Both use their own frozen rules. Both reproduce original
system F0.5/precision/recall/event counts within1e-6. Samechannels/scalers; no
training, tuning, assessment access or checkpoint replacement. Source files
stable before/after. Cached verified inputs reused; no fresh fullCSV hash.

|Model|System F0.5|Input-channel F0.5|System TP/FP/FN|
|---|---:|---:|---|
|Pseudo-anomaly|0.832880|0.908552|2/0/2|
|Reconstruction-only|0.832732|0.908375|2/0/2|

Both detectid_10/id_17, missid_98(local) andid_99(outsideinputs).

|Event|Pseudo peak / own threshold|Reconstruction-only peak / own threshold|
|---|---:|---:|
|id_10|10.8756|102.7100|
|id_17|1.2802|1.5200|
|id_98|0.8157|0.000055|
|id_99|0.005919|0.000006|

Ratios are diagnostic peak scores on timestamp samples inside raw event
intervals; formal detection uses interval overlap and postprocessing, not this
peak ratio alone. id_98 is reconstructed with very little error by the
reconstruction-only model despite its annotation on inputs. Hence tiny nominal
reconstruction error does not establish improved anomaly separation. The pseudo
model givesid_98 a larger relative signal, but neither frozen rule detects it.
This does not establish stronger detection for all anomalies or causal attribution
to the pseudoobjective; historical checkpoint/source/order caveats remain.

Keep baseline. Do not switch to reconstruction-only merely to remove distorted
normal outputs, and do not lower threshold based on this one event: prior
calibration-driven cap relaxation failed development assessment. The useful next
question is what labelled relationship/behavior inid_98 differs from nominal
windows and whether the synthetic anomalies train that distinction. Investigate
annotated channels41–46 on calibration data before defining another training
experiment. No guaranteed path to system0.85 is established.

[Tables and verification](tables/calibration-model-comparison-2005-2026-09-26/).
