# Kaggle metric comparison — 24 September 2026

Checked the competition's indexed official overview:
https://www.kaggle.com/competitions/esa-adb-challenge/overview/description
Direct evaluation-tab fetch failed; search retrieval exposed the full official
Evaluation metric section. Hidden Kaggle scorer implementation was not obtained.

## Conclusion

Same corrected event-wise F0.5 formulation, but exact Kaggle implementation parity
is **not established**. Our current result is an ESA-ADB benchmark-style local score,
not a Kaggle leaderboard result.

Both combine corrected event precision P and event recall R as
F0.5 = 1.25 P R / (0.25 P + R), penalizing false alarms in nominal data.

Important distinctions:

- Kaggle describes continuous anomalous segments as events. Our evaluator follows
  the bundled official ESA-ADB code: unions of raw annotations grouped by event ID,
  including disjoint intervals of the same ID. These definitions can count events
  differently. The prose alone cannot resolve Kaggle's exact grouping.
- Kaggle describes its correction using false-positive sample count divided by
  nominal sample count. Our reference-aligned code uses falsely alarmed nominal
  duration divided by nominal duration. These coincide under suitable uniform-grid
  alignment, but raw off-grid annotation boundaries need not produce identical values.
- Our selected categories are Anomaly and Rare Event; communication gaps and other
  supplied non-selected annotation intervals are neutral for false positives.
  Competition-specific label preprocessing must be matched before asserting parity.
- Our split, selected inputs, raw annotation scope and assessment data differ from
  a Kaggle submission protocol. Scores are not directly interchangeable.

Verification: reran tests/thesis/test_esa_evaluation.py — 7 tests passed. The suite
checks parity to the SHA-pinned bundled ESAScores implementation (beta .5, matching
categories/range), including random regular/irregular predictions and boundaries.
This is reference-code parity on tested cases, not proof of hidden Kaggle parity.

No evaluation code or saved result was changed. Retain metric name/version
`esa-ew-id-duration-v1` and preserve current 0.805722 result with its documented scope.
