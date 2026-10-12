# Parameter-matched channel-isolation comparison

## Question and single change

Does preventing cross-channel input mixing improve real-event detection while
retaining absolute levels? The candidate subclasses the existing Transformer AE
and applies diagonal-only masks to the channel encoder attention, channel decoder
self-attention and channel decoder memory attention. Temporal attention is unchanged.

All parameter tensors, shapes, initialization, channel identity embeddings and
normalization layers are retained. Both models have the same nominal parameter
count. Effective functional capacity is not identical: single-channel attention
has no choice among keys, so query/key projections can become functionally
redundant. Same parameter count does not mean equal expressiveness or GPU time.
The candidate shares learned weights across channels; it is not eight separately
trained networks. Joint losses, including their existing pseudo-anomaly
normalization, still couple training updates. Only input-to-output cross-channel
mixing is removed. No channel is removed and no level centering is applied.

## Fixed comparison

Control: `results_longrun/development/inner_selection_cuda_v2` (loss weight 1).
Candidate: `results_longrun/development/inner_isolation_cuda_v1` (loss weight 1).

- Inputs 40–47, seeds 42/43/44, five epochs, 25,000 nominal windows, batch 64.
- Fold 2001: train 2000, calibrate Jan–Jun 2001, real validation Jul–Dec 2001.
- Fold 2002: train 2000–2001, calibrate Jan–Jun 2002, real validation Jul–Dec 2002.
- Same scaler, initialization seed, window indices, corruption generator,
  optimizer, learning-rate schedule, score aggregation and threshold grid.
- Use calibration-selected checkpoints as primary; earliest ties retained.
- Both local and system scores use identical frozen predictions and real data.
- Fixed-epoch-five records are secondary, not a fallback chosen after results.

Advance only if mean system F0.5 improves in both folds without increased mean
false-event count in either. Report false-alarm duration and missed events too.
A pass permits considering extension, not automatic replacement of the main
baseline. These already inspected internal periods are development data; do not
claim independent validation or robust achievement of the thesis target.

## Provenance and recovery

The wrapper reuses the unchanged inner-validation runner, replacing its model
factory inside a context restored on exit. The sidecar comparison protocol locks
the architecture before training; the runner records source hashes. **Candidate
weights have the same tensor keys as the joint model but different attention
semantics. Always instantiate ChannelIsolatedAE when replaying them.** Loading
them into MultivariateAE would silently produce the wrong model.

```bash
python -m esa_thesis.isolation_comparison \
  --control results_longrun/development/inner_selection_cuda_v2 \
  --output results_longrun/development/inner_isolation_cuda_v1
```

Use `--resume` after interruption with identical sources and arguments. All
control artifacts are preserved and verified. No outer-period run or push is
part of this experiment. Record outcome in a separate results report.
