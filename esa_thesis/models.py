"""Models components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from .config import (DROPOUT, D_FF_MULT, D_MODEL, MLP_HIDDEN, MLP_LAYERS, MLP_PRED_LEN, N_HEADS, N_LAYERS, PATCH_SIZE, SEQ_LEN)


class MultivariateAE(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.C = C; self.NP = SEQ_LEN // PATCH_SIZE; self.PS = PATCH_SIZE
        self.patch_proj  = nn.Linear(PATCH_SIZE, D_MODEL)
        self.patch_pos   = nn.Embedding(self.NP, D_MODEL)
        self.channel_emb = nn.Embedding(C, D_MODEL)
        enc_t = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        enc_c = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        dec_t = nn.TransformerDecoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        dec_c = nn.TransformerDecoderLayer(D_MODEL, N_HEADS, D_MODEL*D_FF_MULT, DROPOUT, batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_t, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL), enable_nested_tensor=False)
        self.channel_encoder  = nn.TransformerEncoder(enc_c, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL), enable_nested_tensor=False)
        self.temporal_decoder = nn.TransformerDecoder(dec_t, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL))
        self.channel_decoder  = nn.TransformerDecoder(dec_c, max(1, N_LAYERS//2), norm=nn.LayerNorm(D_MODEL))
        self.out_head = nn.Linear(D_MODEL, PATCH_SIZE)

    def _tokenize(self, x):
        B, Ch, T, _ = x.shape
        patches = x.squeeze(-1).reshape(B, Ch, self.NP, self.PS)
        tok = self.patch_proj(patches)
        tok = tok + self.patch_pos(torch.arange(self.NP, device=x.device)).view(1, 1, self.NP, -1)
        tok = tok + self.channel_emb(torch.arange(Ch, device=x.device)).view(1, Ch, 1, -1)
        return tok

    def _encode(self, tok):
        B, Ch, NP, D = tok.shape
        t = tok.reshape(B*Ch, NP, D); t = self.temporal_encoder(t).reshape(B, Ch, NP, D)
        t = t.permute(0, 2, 1, 3).reshape(B*NP, Ch, D); t = self.channel_encoder(t)
        return t.reshape(B, NP, Ch, D).permute(0, 2, 1, 3)

    def _decode(self, tok, z):
        B, Ch, NP, D = tok.shape
        t  = self.temporal_decoder(tok.reshape(B*Ch, NP, D), z.reshape(B*Ch, NP, D)).reshape(B, Ch, NP, D)
        tq = t.permute(0, 2, 1, 3).reshape(B*NP, Ch, D)
        zq = z.permute(0, 2, 1, 3).reshape(B*NP, Ch, D)
        t  = self.channel_decoder(tq, zq)
        return t.reshape(B, NP, Ch, D).permute(0, 2, 1, 3)

    def forward(self, x):
        tok = self._tokenize(x); z = self._encode(tok); out = self._decode(tok, z)
        B, Ch, _, _ = x.shape
        return self.out_head(out).reshape(B, Ch, SEQ_LEN, 1)


class MLPForecaster(nn.Module):
    """
    Feedforward MLP that predicts the next MLP_PRED_LEN timesteps from a
    SEQ_LEN-length context window.

    Anomaly score = per-channel MAE between predicted and actual future values.

    Scientific basis: unlike the AE which detects structural abnormality within
    a window, the MLP detects *temporal surprise* — the next values are not what
    the model expected given the history.  Both detectors are complementary:
    - AE fires when the channel pattern looks globally wrong
    - MLP fires when the channel trajectory changes unexpectedly
    An anomaly that triggers both is almost certainly real.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_ch     = n_channels
        self.seq_len  = SEQ_LEN
        self.pred_len = MLP_PRED_LEN
        in_dim        = SEQ_LEN  * n_channels
        out_dim       = MLP_PRED_LEN * n_channels

        layers: List[nn.Module] = [
            nn.Linear(in_dim, MLP_HIDDEN),
            nn.LayerNorm(MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        ]
        for _ in range(MLP_LAYERS - 1):
            layers += [
                nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                nn.LayerNorm(MLP_HIDDEN),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            ]
        layers.append(nn.Linear(MLP_HIDDEN, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, SEQ_LEN, 1) — same layout as AE input
        B, C, T, _ = x.shape
        flat = x.squeeze(-1).reshape(B, -1)           # (B, C*SEQ_LEN)
        out  = self.net(flat)                          # (B, C*MLP_PRED_LEN)
        return out.reshape(B, C, MLP_PRED_LEN, 1)
