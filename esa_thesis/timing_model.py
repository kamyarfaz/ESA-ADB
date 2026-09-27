"""Transformer reconstruction with optional observation-timing context."""
import torch
from torch import nn
from .models import MultivariateAE


class TimingAwareAE(MultivariateAE):
    """Reconstruct telemetry only; timing is auxiliary context, never a target.

    A zero context implements the matched control with identical parameter count.
    Zero initial context weights preserve the base model's initial predictions.
    """
    def __init__(self, channels):
        super().__init__(channels)
        self.timing_proj = nn.Linear(self.PS * 2, self.patch_proj.out_features, bias=False)
        nn.init.zeros_(self.timing_proj.weight)

    def forward(self, x, timing=None):
        tok = self._tokenize(x)
        if timing is None:
            timing = x.new_zeros((*x.shape[:3], 2))
        if timing.shape != (*x.shape[:3], 2):
            raise ValueError('Timing must have shape [batch, channels, time, 2]')
        context = timing.reshape(len(x), self.C, self.NP, self.PS * 2)
        tok = tok + self.timing_proj(context)
        return self.out_head(self._decode(tok, self._encode(tok))).reshape_as(x)
