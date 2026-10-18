"""Fixed normal-reference distance and pinned MOSAD adaptation for ESA telemetry."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from . import config
from .vendor.mosad.model import MOSAD
from .vendor.mosad.loss import deviation_loss
from .vendor.mosad.utils import coe_batch, mixup_batch, getnormals

FEATURES = [f'channel_{i}' for i in range(40, 48)]
MOSAD_ARGS = dict(num_vars=8, num_node_features=256, embedding_dim=120,
    projection_dim=32, dropout_rate=.2, encoder_decoder_layers=8,
    projection_layers=1, detector_layers=2, tcn_kernel_size=7,
    tcn_out_channels=20, tcn_maxpool_out_channels=8, normalize_embedding=True,
    masked_recon=True, con_inf=True, dev_inf=True, node_level=False,
    coe_rate=.05, mixup_rate=.05, con_inf_batch_size=64)


class FixedBinMaxPool(torch.nn.Module):
    """Equivalent adaptive max pooling for divisible lengths, deterministic CUDA backward."""
    def __init__(self, bins):
        super().__init__()
        self.bins = bins

    def forward(self, x):
        if x.shape[-1] % self.bins:
            raise ValueError('Fixed pooling requires divisible sequence length')
        return x.reshape(*x.shape[:-1], self.bins, x.shape[-1]//self.bins).max(-1).values


def make_mosad(**overrides):
    args = {**MOSAD_ARGS, **overrides}
    model = MOSAD(**args)
    model.encoder.network[1] = FixedBinMaxPool(args['tcn_maxpool_out_channels'])
    return model


def nominal_starts(labels, length=256, stride=16):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary exclusion labels')
    starts = np.arange(0, len(labels)-length+1, stride, dtype=np.int64)
    counts = np.r_[0, np.cumsum(labels != 0)]
    return starts[counts[starts+length] == counts[starts]]


def year_reference(starts, times, quota=4096, length=256):
    years = pd.DatetimeIndex(times).year.to_numpy()
    selected = []
    for year in range(2000, 2006):
        choices = starts[(years[starts] == year) & (years[starts+length-1] == year)]
        if not len(choices):
            raise ValueError(f'No nominal reference windows for {year}')
        selected.append(choices[np.linspace(0, len(choices)-1, min(quota, len(choices)), dtype=int)])
    return np.concatenate(selected)


def windows(x, starts, length=256):
    return np.stack([x[int(s):int(s)+length].T for s in starts]).astype(np.float32)


@torch.no_grad()
def nearest_distance(query, reference, block=4096):
    """Exact exhaustive nearest-reference MSE, all channels/times jointly, FP32."""
    q = query.flatten(1)
    r = reference.flatten(1)
    best = torch.full((len(q),), float('inf'), device=q.device)
    ids = torch.zeros(len(q), dtype=torch.long, device=q.device)
    q2 = q.square().sum(1)[:, None]
    for offset in range(0, len(r), block):
        ref = r[offset:offset+block]
        d = (q2 + ref.square().sum(1)[None, :] - 2*(q @ ref.T)).clamp_min(0) / q.shape[1]
        v, j = d.min(dim=1)
        update = v < best
        best = torch.where(update, v, best)
        ids = torch.where(update, j+offset, ids)
    return best, ids


def training_shots(annotations, times, length=256):
    """One window per input-annotated training event; never use future annotations."""
    t0, t1 = pd.Timestamp('2000-01-01', tz='UTC'), pd.Timestamp('2006-01-01', tz='UTC')
    part = annotations[(annotations.StartTime < t1) & (annotations.EndTime >= t0)
                       & annotations.Channel.isin(FEATURES)
                       & annotations.Category.isin(['Anomaly', 'Rare Event'])].copy()
    part.StartTime = part.StartTime.clip(lower=t0)
    part.EndTime = part.EndTime.clip(upper=t1-pd.Timedelta(1, 'ns'))
    part['duration'] = (part.EndTime-part.StartTime).dt.total_seconds()
    rows = []
    gaps = annotations[(annotations.StartTime < t1) & (annotations.EndTime >= t0)
                       & (annotations.Category == 'Communication Gap')]
    for event, group in part.groupby('ID', sort=True):
        # Longest input interval, stable ties; event labels are used only during training.
        row = group.sort_values(['duration', 'Channel', 'StartTime'], ascending=[False, True, True]).iloc[0]
        mid = row.StartTime.value + (row.EndTime.value-row.StartTime.value)//2
        center = int(np.searchsorted(times, mid))
        start = int(np.clip(center-length//2, 0, len(times)-length))
        if not ((times[start:start+length] >= row.StartTime.value) &
                (times[start:start+length] <= row.EndTime.value)).any():
            raise ValueError(f'No sampled observation inside training event {event}')
        if ((gaps.StartTime.astype('int64') <= times[start+length-1]) &
            (gaps.EndTime.astype('int64') >= times[start])).any():
            raise ValueError(f'Training shot overlaps communication gap: {event}')
        rows.append(dict(event_id=str(event), category=row.Category, start=start,
                         timestamp=str(pd.Timestamp(times[start], tz='UTC'))))
    if not rows:
        raise ValueError('No observable real training anomalies')
    return rows


def normal_contrastive_loss(embedding, labels, temperature=.07):
    """MOSAD normal anchors only; avoid undefined loss for singleton anomaly classes."""
    anchors = labels == 0
    positives = torch.round(labels) == 0
    if anchors.sum() < 2:
        raise ValueError('Need at least two normal anchors')
    logits = embedding[anchors] @ embedding.T / temperature
    self_mask = torch.arange(len(labels), device=labels.device)[None, :] == torch.nonzero(anchors, as_tuple=True)[0][:, None]
    denominator = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
    mask = positives[None, :] & ~self_mask
    return (denominator - (logits*mask).sum(1)/mask.sum(1)).mean()


def mosad_batch_loss(model, normal, anomalies):
    x = torch.cat([normal, anomalies])
    y = torch.cat([torch.zeros(len(normal), device=x.device), torch.ones(len(anomalies), device=x.device)])
    coe, cy = coe_batch(x.transpose(1, 2), y, .05, x.shape[-1])
    x = torch.cat([x, coe.transpose(1, 2)]); y = torch.cat([y, cy])
    mix, my = mixup_batch(x.transpose(1, 2), y, .05)
    x = torch.cat([x, mix.transpose(1, 2)]); y = torch.cat([y, my])
    flat = x.reshape(-1, x.shape[-1])
    reconstructed, projection, deviation = model(flat, y, 8)
    normal_target = getnormals(flat, y, 8, node_labels=False)
    rec = F.mse_loss(reconstructed, normal_target)
    con = normal_contrastive_loss(projection, y)
    dev = deviation_loss(deviation, y)
    return rec+con+dev, (float(rec.detach()), float(con.detach()), float(dev.detach()))


@torch.no_grad()
def mosad_components(model, x, normal_projection):
    """Original three scores; cache a fixed training-only contrastive reference."""
    model.eval()
    reconstructed = torch.empty_like(x)
    for channel in range(x.shape[1]):
        masked = x.clone(); masked[:, channel] = 0
        predicted = model.rechead(model.encoder(masked))
        reconstructed[:, channel] = predicted[:, channel]
    mse = (reconstructed-x).square().mean((1, 2))
    embedding = model.encoder(x)
    dev = model.devhead(embedding).flatten()
    projection = model.conhead(embedding)
    con = 1 - (projection @ normal_projection.T).mean(1)
    return torch.stack([mse, dev, con], dim=1)


def combined_mosad(components, low, high):
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Degenerate nominal reconstruction range')
    return (components[:, 0]-low)/(high-low) + components[:, 1] + components[:, 2]
