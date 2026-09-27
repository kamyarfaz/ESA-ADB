"""Experimental short coordinated corruptions; legacy generator remains default.

No labels or assessment statistics are used. Duration/support settings are an
experimental hypothesis, not a fitted model or operational recommendation.
"""
import random
import torch
from .augmentation import make_pseudo_anomaly


def make_training_anomaly(xb, *, mode='legacy', return_metadata=False):
    """Optional per-window 50/50 legacy and short-coordinated mixture.

    Short branch: 1–10 samples, 1–C channels, one shared segment/type. Offset
    direction/amplitude, noise waveform or temporal shift are shared across
    selected channels. Scale uses each channel's full-window standard deviation,
    avoiding undefined/small single-sample segment scales. A temporal roll can
    still be a no-op on constant input; metadata records actual changed support.
    The legacy mode delegates directly without consuming extra random numbers.
    """
    if mode not in ('legacy', 'short_coordinated'):
        raise ValueError('Unknown augmentation mode')
    if mode == 'legacy':
        if return_metadata:
            raise ValueError('Legacy metadata unavailable; use unchanged legacy output')
        return make_pseudo_anomaly(xb)
    if xb.ndim != 4 or xb.shape[-1] != 1 or min(xb.shape[:3]) < 1:
        raise ValueError('Expected nonempty [batch, channels, time, 1]')
    if not xb.is_floating_point() or not torch.isfinite(xb).all():
        raise ValueError('Expected finite floating point inputs')
    if xb.shape[2] < 16:
        raise ValueError('Mixture requires at least16 time samples for legacy branch')
    out=xb.detach().clone();records=[]
    for i in range(len(xb)):
        if random.random() < .5:
            out[i:i+1]=make_pseudo_anomaly(xb[i:i+1])
            record={'branch':'legacy'}
        else:
            channels,time=xb.shape[1:3]
            length=random.randint(1,min(10,time));start=random.randint(0,time-length)
            count=random.randint(1,channels);chosen=torch.randperm(channels,device=xb.device)[:count]
            kind=random.choice(['offset','noise'] if length==1 else ['offset','roll','noise'])
            segment=xb[i,chosen,start:start+length,0].detach()
            scale=xb[i,chosen,:,0].detach().std(dim=1,unbiased=False,keepdim=True)+1e-6
            if kind=='offset':
                direction=random.choice([-1.,1.]);amplitude=random.uniform(3.,8.)
                changed=segment+direction*amplitude*scale
            elif kind=='noise':
                noise=torch.randn((1,length),dtype=xb.dtype,device=xb.device)
                changed=segment+noise*scale*random.uniform(2.,5.)
            else:
                changed=torch.roll(segment,random.randint(1,length-1),dims=1)
            out[i,chosen,start:start+length,0]=changed
            record=dict(branch='short',kind=kind,start=start,length=length,selected_channels=chosen.cpu().tolist())
        changed=out[i]!=xb[i]
        record.update(changed_samples=int(changed.sum().item()),changed_channels=int(changed.any(dim=(1,2)).sum().item()))
        records.append(record)
    if not torch.isfinite(out).all():raise ValueError('Nonfinite augmented output')
    return (out,records) if return_metadata else out
