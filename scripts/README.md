# Original benchmark utilities

These tools belong to the ESA/TimeEval benchmark. Their paths are retained for
existing commands and imports. Run from the repository root with the benchmark
environment active; inspect each script's arguments and path settings first.

| Area | Files |
| --- | --- |
| Dataset preparation | `preprocess_dataset.py`, `concat_dataset.py`, `fill.py`, `train_val_dataset_split.py`, `convert2bin.py` |
| Dataset inspection and annotation | `analyze-datasets.py`, `infer_anomaly_types.py`, `extract_fragments_for_OXI_annotator.py` |
| Evaluation | `calculate_metric.py`, `recalculate_metrics.py`, `reevaluate.py` |
| Result aggregation and plots | `create_summary.py`, `merge_results.py`, `plot.py` |
| Mission timelines | `Mission1_timelines.ipynb`, `Mission2_timelines.ipynb`, `Mission3_timelines.ipynb` |
| Preprocessing checks | `test_preprocess_dataset.py` |

The thesis-specific alternatives live in `esa_thesis/analysis`,
`esa_thesis/evaluation`, and `esa_thesis/calibration`, accessible through
`python -m esa_thesis --help`. The original tools have not all been executed
as part of repository organization.
