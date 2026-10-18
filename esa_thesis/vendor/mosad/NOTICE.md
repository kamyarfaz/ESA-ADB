# MOSAD source provenance

Author implementation: https://github.com/Armanfard-Lab/MOSAD
Commit cf935f817a89bb773a0a999d9117e33322970920, retrieved 18 October 2026.
Original author names and paper: Thomas Lai, Thi Kieu Khanh Ho, Narges Armanfard,
Open-Set Multivariate Time-series Anomaly Detection, ECAI 2024.

The repository README says MIT, but its LICENSE contains CC BY-NC-SA 3.0.
The actual LICENSE is retained verbatim; no MIT relicensing is asserted here.
network.py also retains its original Apache-2.0 attribution notices.
This vendored directory and modifications retain the supplied license terms.

Changes: package-relative imports; contextual-outlier segment endpoints actually
sorted (upstream discarded the sorted return), and zero-length swaps made nonempty.
No automatic upstream training/testing entry point is used. Our adapter provides
ESA splits, supervised training-event sampling, deterministic inference reference,
stable normal-anchor contrastive loss, checkpoint recovery and corrected evaluation.
It is explicitly an ESA adaptation, not an exact reproduction of published scores.

The external adapter replaces adaptive pooling with equivalent fixed-bin maxima
for divisible256-sample windows to support deterministic CUDA backward.
