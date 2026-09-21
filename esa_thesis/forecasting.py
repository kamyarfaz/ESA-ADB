"""Matched history-only forecasters and timestamp-aligned anomaly scores."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

CONTEXT = 256
HORIZON = 16
PATCH = 16
WIDTH = 128
LAYERS = 2


class Persistence(nn.Module):
    def forward(self, x):
        return x[:, :, -1:, :].expand(-1, -1, HORIZON, -1)


class ForecastMLP(nn.Module):
    """Shared channel-independent two-hidden-layer MLP control."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONTEXT, WIDTH), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(WIDTH, HORIZON))

    def forward(self, x):
        return self.net(x.squeeze(-1)).unsqueeze(-1)


class PatchForecaster(nn.Module):
    """Shared channel-independent patch encoder; all tokens precede the targets.

    Attention among historical patches is bidirectional. No target token is fed
    to the encoder, so a triangular attention mask is unnecessary for forecasting.
    """
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(PATCH, WIDTH)
        self.position = nn.Embedding(CONTEXT // PATCH, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, 8, 2*WIDTH, .1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS, norm=nn.LayerNorm(WIDTH),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(CONTEXT // PATCH * WIDTH, HORIZON)
        # Encoder clones otherwise start with identical parameters in each layer.
        for block in self.encoder.layers:
            for name, parameter in block.named_parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
                elif 'norm' in name and name.endswith('weight'):
                    nn.init.ones_(parameter)
                else:
                    nn.init.zeros_(parameter)

    def forward(self, x):
        batch, channels, length, _ = x.shape
        if length != CONTEXT:
            raise ValueError('Unexpected context length')
        tokens = self.projection(x.reshape(batch*channels, CONTEXT//PATCH, PATCH))
        tokens = tokens + self.position(torch.arange(CONTEXT//PATCH, device=x.device))
        return self.head(self.encoder(tokens).flatten(1)).reshape(batch, channels, HORIZON, 1)


class LevelResidualForecaster(PatchForecaster):
    """Predict deviations from the last observed level, then restore that level.

    Center each channel independently using history only. Targets and anomaly
    scores retain the existing scaled units; a new target jump is still scored.
    This adds no parameters and preserves the control's initialization.
    """

    def forward(self, x):
        level = x[:, :, -1:, :]
        return level + super().forward(x - level)


def make_model(name):
    return {'persistence': Persistence, 'mlp': ForecastMLP,
            'transformer': PatchForecaster,
            'transformer_residual': LevelResidualForecaster}[name]()


class NominalForecastWindows(Dataset):
    def __init__(self, values, labels, *, seed=42, limit=250000):
        self.values = np.asarray(values, dtype=np.float32)
        labels = np.asarray(labels)
        if len(values) != len(labels) or not np.isfinite(self.values).all():
            raise ValueError('Invalid training data')
        starts = np.arange(0, len(values)-CONTEXT-HORIZON+1, HORIZON)
        cumulative = np.r_[0, np.cumsum(labels != 0)]
        starts = starts[cumulative[starts+CONTEXT+HORIZON] == cumulative[starts]]
        if not len(starts):
            raise ValueError('No complete nominal context/target windows')
        if limit and len(starts) > limit:
            starts = np.sort(np.random.default_rng(seed).choice(starts, size=limit, replace=False))
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        context = self.values[start:start+CONTEXT].T[:, :, None].copy()
        target = self.values[start+CONTEXT:start+CONTEXT+HORIZON].T[:, :, None].copy()
        return torch.from_numpy(context), torch.from_numpy(target)


@torch.no_grad()
def forecast_scores(model, values, *, device='cpu', batch_size=64):
    """Predict disjoint future blocks, score each target when it arrives.

    All channels use the same block schedule. Keep a partial final block, so only
    the first CONTEXT samples lack forecasts. Never average over future residuals.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= CONTEXT or not np.isfinite(values).all() or batch_size < 1:
        raise ValueError('Need finite observations beyond the initial context')
    scores = np.zeros(len(values), dtype=np.float32)
    covered = np.zeros(len(values), dtype=bool)
    model.eval()
    starts = np.arange(0, len(values)-CONTEXT, HORIZON)
    for offset in range(0, len(starts), batch_size):
        batch = starts[offset:offset+batch_size]
        contexts = np.stack([values[s:s+CONTEXT].T[:, :, None] for s in batch])
        predictions = model(torch.from_numpy(contexts).to(device)).squeeze(-1).cpu().numpy()
        if predictions.shape != (len(batch), values.shape[1], HORIZON) or not np.isfinite(predictions).all():
            raise ValueError('Invalid forecast output')
        for start, prediction in zip(batch, predictions):
            first = start+CONTEXT
            last = min(first+HORIZON, len(values))
            error = (prediction[:, :last-first].T-values[first:last])**2
            scores[first:last] = error.max(axis=1)
            covered[first:last] = True
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite forecast scores')
    return scores, covered
