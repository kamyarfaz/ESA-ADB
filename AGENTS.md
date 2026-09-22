# Instructions for AI assistants working in this repository

Read `AI_HANDOFF.md` first. It records the research state and the user's current
intent. Then read the documents and saved result files relevant to the task.
This file supplements, and does not override, the user's current instructions.

- Current priority is safe migration to another server. Research is paused.
  Do not start training, delete artifacts or initiate a new experiment unless
  the user requests it. Suggestions in research reports are not approved runs.
- The baseline `MultivariateAE` already uses Transformers. Do not propose adding
  a Transformer as though the baseline were a conventional dense autoencoder.
- Preserve existing results. Use new output directories for new experiments.
  Never choose thresholds, checkpoints, channel sets or models using benchmark
  test scores and report the same test as independent validation.
- Use the corrected event evaluator; do not substitute pointwise F0.5 or the
  legacy metric. Read `esa_thesis/evaluation/esa.py` and its parity tests.
- Do not claim the thesis F0.5 ≥0.85 target has been robustly achieved.
- Keep raw datasets, generated arrays and weights out of source Git. Essential
  artifacts are backed up using GitHub Releases. Keep the full external-drive
  copy too; release backups intentionally omit some older large arrays.
- Check Git status before editing. Preserve user changes and avoid destructive
  cleanup, force pushes, or silently rewriting historical protocols.
- The research environment is Python 3.9 with dependencies in
  `requirements-test.txt`; the root setup.py belongs to the legacy TimeEval
  framework. Read README before installing. Do not assume CUDA is available.
- For code changes run relevant tests; the complete thesis suite is:
  `python -m unittest discover -s tests/thesis -p 'test_*.py' -q`.
- Update `AI_HANDOFF.md` when the research state materially changes. If files
  change after a transfer manifest is generated, regenerate the manifest before
  copying, per `MIGRATION.md`. Never claim a destination copy is verified until
  verification has actually run there.
