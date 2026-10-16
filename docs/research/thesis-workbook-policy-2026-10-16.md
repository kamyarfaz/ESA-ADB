# Thesis workbook comparison policy

The user requested identical training, calibration, testing and run counts,
system-level assessment, and FPR/TPR instead of false-event count.

Current workbook:
`outputs/thesis-system-20261016/deliverables/ESA_thesis_results.xlsx`.
It supersedes the October13 compact workbook for the final comparison, while
the older workbook remains an intact historical record of different protocols.

Shared reference: train2000–2005, calibration2006, continuous test2007–2013,
40epochs, all nominal windows, batch64, one planned run/seed42, inputs40–47.
Only the baseline is completed. The short-no-roll row and its annual slices
remain pending until the currently running experiment completes verification.
Do not turn missing results into zero, choose a favorable seed from historical
three-seed runs, or relabel old development periods as the common full-period test.
Other historical studies have different splits or budgets and cannot be put into
this comparison by spreadsheet edits alone. No additional training was launched.

All displayed metrics use system-level annotations across all76 telemetry
channels, distinct from the eight model inputs. TPR(event) is event recall:
detected target IDs / total target IDs. FPR(time) is the corrected evaluator's
nominal false-alarm duration / nominal duration. This is explicitly duration-based,
not a discrete event FP/(FP+TN) or pointwise binary FPR. An event-negative population
has not been defined, so no event FPR is invented. Neutral annotation intervals
are excluded from nominal duration under the existing evaluator convention.
Precision and event F0.5 retain their corrected duration-penalized definitions.

The full pooled and seven annual baseline rows were independently recomputed from
the original prediction mask and raw annotations. Inputs matched the preserved
migration SHA256 entries. All eight Excel F0.5 formulas match replayed values.
Formula-error inspection found no errors and both populated sections were rendered
and visually reviewed. Formula values are numeric; pending candidate cells are text.

The historical baseline is context for the short-no-roll augmentation method,
not a matched short-coordinated control that isolates only short-roll removal.
The workbook states this limitation. Identical splits/runs do not erase model or
augmentation differences. Benchmark years were previously inspected and remain
reused assessment, not independent confirmation.

At run completion update this workbook's pending system-level rows using verified
yearly/pooled outputs. Keep the common protocol and definitions. Training source,
frozen data and the running protocol were not modified by this reporting change.
