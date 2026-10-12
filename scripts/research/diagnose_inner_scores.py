"""Read-only channel diagnosis of training/calibration for the inner campaign."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from esa_thesis import config
from esa_thesis.data import RobustChannelScaler
from esa_thesis.development import load_period
from esa_thesis.diagnose_channels import FEATURES, TRAIN_SHA, digest, window_errors, aggregate, training_starts
from esa_thesis.inner_validation import FOLDS
from esa_thesis.models import MultivariateAE
from esa_thesis.protocol import evaluator_for
from esa_thesis.thresholds import postprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    torch.set_num_threads(2)
    plan=json.loads((a.source/'protocol.json').read_text())
    assert all(digest(Path('esa_thesis')/f)==h for f,h in plan['source_sha256'].items())
    assert digest(config.TRAIN_FILE)==TRAIN_SHA
    sources={str(a.source/'protocol.json'):digest(a.source/'protocol.json')}
    for f in (config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE):sources[str(f)]=digest(f)
    labels=pd.read_csv(config.ANNOTATIONS_FILE).merge(pd.read_csv(config.ANOMALY_TYPES_FILE),on='ID')
    for col in ('StartTime','EndTime'):labels[col]=pd.to_datetime(labels[col],utc=True)
    a.output.mkdir(parents=True)
    rows=[];events=[];replays=[];probes=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fold,(train_end,cal_end,_) in FOLDS.items():
        print(fold,'loading training/calibration only',flush=True)
        raw,y,times=load_period(config.TRAIN_FILE,FEATURES,None,cal_end)
        split=np.searchsorted(times,pd.Timestamp(train_end,tz='UTC').value)
        sample=training_starts(y[:split],times[:split],256,int(fold))
        cal=raw[split:];cy=y[split:];ct=times[split:]
        starts=np.arange(0,len(cal)-256+1,config.SCORE_STRIDE)
        counts=np.r_[0,np.cumsum(cy!=0)]
        clean_windows=counts[starts+256]==counts[starts]
        ev=evaluator_for(ct);nominal=cy==0
        for seed in (42,43,44):
            folder=a.source/fold/f'seed{seed}'
            decision=json.loads((folder/'frozen_decisions.json').read_text())['calibration_selected']
            ep=folder/f"epoch_{decision['epoch']}";checkpoint=ep/'model.pt'
            for f in (checkpoint,folder/'frozen_decisions.json',ep/'calibration_score.npy'):sources[str(f)]=digest(f)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            scaler=RobustChannelScaler();scaler.median_=saved['scaler_median'];scaler.iqr_=saved['scaler_iqr']
            model=MultivariateAE(8).to(a.device);model.load_state_dict(saved['model_state_dict']);model.eval()
            te=window_errors(model,scaler.transform(raw[:split]),sample,a.device,64,f'{fold}/{seed} nominal training')
            x=scaler.transform(cal)
            errors=window_errors(model,x,starts,a.device,64,f'{fold}/{seed} calibration')
            channels=aggregate(errors,starts,len(cal));scores=channels.max(axis=0)
            original=np.load(ep/'calibration_score.npy');np.testing.assert_allclose(scores,original,rtol=1e-5,atol=1e-5)
            rule=decision['rule'];prediction=postprocess(scores>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
            metrics=ev.score(prediction)
            replays.append(dict(fold=fold,seed=seed,epoch=decision['epoch'],max_abs_score_difference=float(np.max(np.abs(scores-original))),saved_calibration_f05=rule['val_esa_f05'],**metrics))
            winner=channels.argmax(axis=0);z=(cal-scaler.median_)/scaler.iqr_
            high=nominal&(scores>=np.percentile(scores[nominal],99))
            clean_high=clean_windows&(errors.max(axis=1)>=np.percentile(errors[clean_windows].max(axis=1),99))
            clean_winner=errors.argmax(axis=1)
            for j,ch in enumerate(FEATURES):
                rows.append(dict(fold=fold,seed=seed,channel=ch,median=scaler.median_[j],iqr=scaler.iqr_[j],
                    train_window_error_q99=float(np.percentile(te[:,j],99)),cal_nominal_score_q99=float(np.percentile(channels[j,nominal],99)),
                    cal_nominal_score_max=float(channels[j,nominal].max()),nominal_clipped_fraction=float(np.mean(np.abs(z[nominal,j])>config.CLIP_Z)),
                    all_cal_clipped_fraction=float(np.mean(np.abs(z[:,j])>config.CLIP_Z)),high_nominal_winner_fraction=float(np.mean(winner[high]==j))))
                rows[-1].update(clean_cal_window_q99=float(np.percentile(errors[clean_windows,j],99)),
                                clean_cal_window_max=float(errors[clean_windows,j].max()),
                                high_clean_window_winner_fraction=float(np.mean(clean_winner[clean_high]==j)))
            clean_wi=int(np.argmax(np.where(clean_windows,errors.max(axis=1),-np.inf)))
            examples=[('wholly_nominal',int(starts[clean_wi]+128),clean_wi)]
            for event,intervals in ev.by_id.items():
                inside=np.zeros(len(ct),bool)
                for lo,hi in intervals:inside|=(ct>=lo)&(ct<=hi)
                if not inside.any():continue
                peak=int(np.argmax(np.where(inside,scores,-np.inf)));ch=int(winner[peak])
                annotations=labels[(labels.ID==event)&(labels.StartTime<pd.Timestamp(cal_end,tz='UTC'))&(labels.EndTime>=pd.Timestamp(train_end,tz='UTC'))]
                local=annotations.Channel.isin(FEATURES).any()
                events.append(dict(fold=fold,seed=seed,event=event,local=bool(local),peak_channel=FEATURES[ch],
                    peak_score=float(scores[peak]),threshold=rule['threshold'],peak_ratio=float(scores[peak]/rule['threshold']),
                    alarm_on_event_samples=bool(prediction[inside].any()),clipped_fraction_on_event=float(np.mean(np.abs(z[inside,ch])>config.CLIP_Z))))
                if local:examples.append((event,peak,None))
            if seed==42:
                for label,peak,forced_wi in examples:
                    eligible=np.flatnonzero((starts<=peak)&(starts+256>peak))
                    wi=forced_wi if forced_wi is not None else int(eligible[np.argmax(errors[eligible].max(axis=1))])
                    start=int(starts[wi])
                    values=x[start:start+256];tensor=torch.from_numpy(values.T[None,:,:,None].copy()).to(a.device)
                    with torch.no_grad():recon=model(tensor).cpu().numpy()[0,:,:,0].T
                    fig,axes=plt.subplots(4,2,figsize=(12,9),sharex=True,constrained_layout=True)
                    for j,ax in enumerate(axes.flat):
                        ax.plot(np.arange(256)/2,values[:,j],label='Scaled observation',color='black',lw=1)
                        ax.plot(np.arange(256)/2,recon[:,j],label='Reconstruction',color='#d27932',lw=1)
                        ax.set_title(FEATURES[j]);ax.set_xlabel('Minutes');ax.set_ylabel('Scaled value')
                        probes.append(dict(fold=fold,event_or_nominal=label,channel=FEATURES[j],start=str(pd.Timestamp(ct[start],tz='UTC')),
                            wholly_nominal=bool(clean_windows[wi]),
                            window_mse=float(np.mean((values[:,j]-recon[:,j])**2)),peak_point_error=float(np.max((values[:,j]-recon[:,j])**2)),
                            observed_min=float(values[:,j].min()),observed_max=float(values[:,j].max()),reconstructed_min=float(recon[:,j].min()),reconstructed_max=float(recon[:,j].max())))
                    axes[0,0].legend(fontsize=8)
                    fig.suptitle(f'{fold} seed42 epoch{decision["epoch"]}: {label}; selected calibration window {pd.Timestamp(ct[start])}')
                    fig.savefig(a.output/f'{fold}_{label}.png',dpi=120);plt.close(fig)
            np.savez_compressed(a.output/f'{fold}_{seed}_errors.npz',training=te,calibration=errors)
    assert digest(config.TRAIN_FILE)==TRAIN_SHA
    assert sources=={f:digest(f) for f in sources}
    for name,data in [('channel_summary',rows),('event_scores',events),('reproduction',replays),('window_probes',probes)]:pd.DataFrame(data).to_csv(a.output/f'{name}.csv',index=False)
    (a.output/'protocol.json').write_text(json.dumps(dict(status='complete',device=a.device,source_sha256=sources,train_sha256=TRAIN_SHA,
        policy='Saved calibration-selected checkpoints, seeds42/43/44; training/calibration only; no rule changes',
        interpretation='Channel training window errors differ from aggregated calibration scores; example windows deliberately selected by high error, not representative samples; unscaled inputs are prepared CSV values, not native telemetry'),indent=2)+'\n')
    print(pd.DataFrame(events).to_string(index=False),flush=True)


if __name__=='__main__':main()
