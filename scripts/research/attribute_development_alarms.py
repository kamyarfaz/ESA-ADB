"""Reconstruct frozen 2004 false-alarm peak scores to identify driving channels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.data import RobustChannelScaler
from esa_thesis.models import MultivariateAE


def main():
    root = config.TRAIN_FILE.parents[4]
    base = root/'results_longrun/development'
    diagnostics = base/'diagnostics_verified_2026-09-17'
    run = base/'ae_baseline_remaining_seed42/2004'
    alarms = pd.read_csv(diagnostics/'false_alarms.csv')
    alarms = alarms[alarms.fold == 2004]
    saved = torch.load(run/'model_checkpoint.pt', map_location='cpu', weights_only=False)
    features = saved['features']
    model = MultivariateAE(len(features)); model.load_state_dict(saved['model_state_dict']); model.eval()
    torch.set_num_threads(2)
    scaler = RobustChannelScaler(); scaler.median_ = saved['scaler_median']; scaler.iqr_ = saved['scaler_iqr']
    scores = np.load(run/'assessment_score.npy')
    origin = pd.Timestamp('2004-04-01', tz='UTC')
    plans = []
    for alarm in alarms.itertuples():
        lo = int((pd.Timestamp(alarm.start)-origin).total_seconds()/30)
        hi = int((pd.Timestamp(alarm.end)-origin).total_seconds()/30)
        peak = lo+int(np.argmax(scores[lo:hi]))
        starts = np.arange(max(0, (peak-config.SEQ_LEN+config.SCORE_STRIDE)//config.SCORE_STRIDE), peak//config.SCORE_STRIDE+1)*config.SCORE_STRIDE
        plans.append((alarm.start, peak, starts))
    blocks = [[] for _ in plans]
    with pd.read_csv(config.TRAIN_FILE, usecols=['timestamp', *features], chunksize=100000) as reader:
        for frame in reader:
            times = pd.to_datetime(frame.timestamp, utc=True)
            for i, (_, _, starts) in enumerate(plans):
                keep = (times >= origin+pd.Timedelta(seconds=int(starts[0])*30)) & (times < origin+pd.Timedelta(seconds=int(starts[-1]+config.SEQ_LEN)*30))
                if keep.any(): blocks[i].append(frame.loc[keep, features])
            if times.iloc[-1] >= pd.Timestamp('2005-01-01', tz='UTC'): break
    rows = []
    with torch.no_grad():
        for (alarm_start, peak, starts), pieces in zip(plans, blocks):
            raw = pd.concat(pieces).to_numpy(dtype=np.float32)
            assert len(raw) == starts[-1]+config.SEQ_LEN-starts[0]
            values = scaler.transform(raw)
            windows = np.stack([values[s-starts[0]:s-starts[0]+config.SEQ_LEN].T[:, :, None] for s in starts])
            inputs = torch.from_numpy(windows)
            errors = (model(inputs)-inputs).square().mean(dim=(2, 3)).mean(dim=0).numpy()
            assert np.isclose(errors.max(), scores[peak], rtol=2e-4, atol=1e-5), (errors.max(), scores[peak])
            for channel, error in zip(features, errors):
                j = features.index(channel)
                rows.append({'alarm_start': alarm_start, 'peak_time': str(origin+pd.Timedelta(seconds=peak*30)),
                             'channel': channel, 'reconstruction_error': float(error),
                             'dominant': bool(error == errors.max()), 'raw_min': float(raw[:,j].min()),
                             'raw_max': float(raw[:,j].max()), 'raw_median': float(np.median(raw[:,j])),
                             'raw_max_step': float(np.abs(np.diff(raw[:,j])).max())})
    result = pd.DataFrame(rows)
    result.to_csv(diagnostics/'2004_false_alarm_channels.csv', index=False)
    print(result[result.dominant].to_string(index=False))


if __name__ == '__main__': main()
