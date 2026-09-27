"""Small matched baseline/masked Transformer pilot; channels 40–47, chronological development folds."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from . import config
from .augmentation import make_pseudo_anomaly
from .data import AllWindows, RobustChannelScaler
from .development import load_period, QUANTILES
from .development_recovery import atomic_save, capture_rng, restore_rng
from .diagnose_channels import FEATURES, TRAIN_SHA, aggregate, digest
from .models import MultivariateAE
from .protocol import evaluator_for
from .evaluation.esa import EventEvaluator
from .runtime import set_seed
from .thresholds import postprocess, sweep_thresholds, select_rows


class MaskedAE(MultivariateAE):
    """Hide patch contents from BOTH encoder and decoder; keep position/channel identity."""
    def __init__(self, channels):
        super().__init__(channels)
        self.mask_token = nn.Parameter(torch.zeros(self.patch_proj.out_features))

    def forward(self, x, phase=0):
        hidden = patch_mask(self.NP, phase, x.device)
        b, c = x.shape[:2]
        patches = x.squeeze(-1).reshape(b, c, self.NP, self.PS)
        content = self.patch_proj(patches.masked_fill(hidden[None,None,:,None], 0))
        content = torch.where(hidden[None,None,:,None], self.mask_token.view(1,1,1,-1), content)
        tok = content + self.patch_pos(torch.arange(self.NP, device=x.device))[None,None,:,:]
        tok = tok + self.channel_emb(torch.arange(c, device=x.device))[None,:,None,:]
        z = self._encode(tok)
        return self.out_head(self._decode(tok,z)).reshape(b,c,config.SEQ_LEN,1)


def patch_mask(n_patches, phase, device):
    if n_patches % 4 or phase not in range(4):
        raise ValueError('Four complementary phases require patch count divisible by four')
    return torch.arange(n_patches, device=device) % 4 == phase


def reconstruction_error(model, x, masked, phase):
    out = model(x,phase) if masked else model(x)
    error = (out-x).square()
    if masked:
        error = error.reshape(len(x),x.shape[1],model.NP,model.PS)
        error = error[:,:,patch_mask(model.NP,phase,x.device),:]
    return error.mean(dim=(1,2,3))


@torch.no_grad()
def score_windows(model, x, masked, device, batch_size):
    model.eval()
    dataset = AllWindows(x)
    errors = np.empty((len(dataset),x.shape[1]),dtype=np.float32)
    offset = 0
    for batch, inputs in enumerate(DataLoader(dataset,batch_size=batch_size),1):
        inputs = inputs.to(device)
        if masked:
            # Every target patch is reconstructed exactly once while hidden.
            per_patch = torch.zeros((len(inputs),inputs.shape[1],model.NP,model.PS),device=device)
            for phase in range(4):
                mask = patch_mask(model.NP,phase,device)
                e = (model(inputs,phase)-inputs).square().reshape_as(per_patch)
                per_patch[:,:,mask,:] = e[:,:,mask,:]
            values = per_patch.mean(dim=(2,3))
        else:
            values = (model(inputs)-inputs).square().mean(dim=(2,3))
        errors[offset:offset+len(inputs)] = values.cpu().numpy();offset += len(inputs)
        if batch == 1 or batch % 100 == 0 or offset == len(dataset):
            print(f'  scoring {offset}/{len(dataset)} windows',flush=True)
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite inference errors')
    return aggregate(errors,dataset.starts,len(x)).max(axis=0)


def nominal_sample(labels, max_windows, seed):
    starts = np.arange(0,len(labels)-config.SEQ_LEN+1,config.STRIDE,dtype=np.int64)
    cumulative = np.r_[0,np.cumsum(labels != 0)]
    starts = starts[cumulative[starts+config.SEQ_LEN] == cumulative[starts]]
    if not len(starts):raise ValueError('No nominal training windows')
    if len(starts)>max_windows:
        starts = np.sort(np.random.default_rng(seed).choice(starts,max_windows,replace=False))
    return starts


def train_arm(name, x, starts, calibration, evaluator, scaler, out, epochs, batch_size, device, seed):
    masked = name in ('masked', 'masked_overlap')
    out.mkdir(exist_ok=True)
    set_seed(seed)
    model = (MaskedAE(8) if masked else MultivariateAE(8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
    dataset = AllWindows(x);dataset.starts=starts
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    history=[];best=None;best_state=None;best_grid=None;start=1
    latest=out/'last.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];best_state=saved['best_state'];best_grid=saved['best_grid'];start=saved['epoch']+1
        restore_rng(saved['rng']);print(f'{name}: resume epoch {start}',flush=True)
    for epoch in range(start,epochs+1):
        model.train();losses=[];components=[];began=time.monotonic()
        for batch,inputs in enumerate(loader):
            inputs=inputs.to(device);optimizer.zero_grad(set_to_none=True)
            phase=(epoch-1+batch)%4
            normal=reconstruction_error(model,inputs,masked,phase)
            target_mask = patch_mask(model.NP,phase,device).repeat_interleave(model.PS) if name=='masked_overlap' else None
            corrupted=make_pseudo_anomaly(inputs, target_mask=target_mask)
            abnormal=reconstruction_error(model,corrupted,masked,phase)
            penalty=torch.relu(config.PA_MARGIN-abnormal/(normal.mean().detach()+1e-8)).mean()
            loss=normal.mean()+config.PA_LAMBDA*penalty
            components.append((normal.mean().item(),abnormal.mean().item(),penalty.item()))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP);optimizer.step();losses.append(loss.item())
            if batch==0 or (batch+1)%100==0:
                print(f'{name}: epoch {epoch}/{epochs}, batch {batch+1}/{len(loader)}',flush=True)
        scheduler.step()
        print(f'{name}: calibrating epoch {epoch}',flush=True)
        score=score_windows(model,calibration,masked,device,batch_size)
        grid=sweep_thresholds(score,evaluator=evaluator,threshold_values=np.percentile(score,QUANTILES),merge_gaps=[0,16,64],min_durs=[1,8,32])
        rule=select_rows(grid).iloc[0].to_dict()
        if best is None or rule['val_esa_f05']>best['rule']['val_esa_f05']:
            best=dict(epoch=epoch,rule=rule)
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_grid=grid.to_dict('records')
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),normal_mse=float(np.mean(components,axis=0)[0]),pseudo_mse=float(np.mean(components,axis=0)[1]),pseudo_penalty=float(np.mean(components,axis=0)[2]),calibration_f05=rule['val_esa_f05'],seconds=time.monotonic()-began))
        atomic_save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,best_state=best_state,best_grid=best_grid,rng=capture_rng()),latest)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(name,history[-1],flush=True)
    atomic_save(dict(model_state_dict=best_state,features=FEATURES,scaler_median=scaler.median_,scaler_iqr=scaler.iqr_,masked=masked),out/'model_checkpoint.pt')
    pd.DataFrame(best_grid).to_csv(out/'calibration_candidates.csv',index=False)
    (out/'frozen_rule.json').write_text(json.dumps(best,indent=2))
    return best


def fold_boundaries(fold):
    year=int(fold)
    if year not in (2003,2004,2005):raise ValueError('Supported folds:2003,2004,2005')
    return f'{year}-01-01',f'{year}-04-01',f'{year+1}-01-01'


def assessment_scopes(times):
    system=evaluator_for(times)
    labels=system.annotations.copy()
    labels.loc[~labels.Channel.isin(FEATURES),'Category']='Out-of-scope (neutral)'
    return {'system_all_channels':system,'input_channels_40_47':EventEvaluator(times,labels)}


def run_pilot_fold(a, fold, output):
    output.mkdir(exist_ok=True)
    if a.resume and (output/'results.csv').exists() and (output/'dual_scope_results.csv').exists():
        print(f'SKIP completed fold {fold}',flush=True)
        return pd.read_csv(output/'results.csv').to_dict('records'),pd.read_csv(output/'dual_scope_results.csv').to_dict('records')
    train_end, calibration_end, assessment_end = fold_boundaries(fold)
    print(f'Fold {fold}: training before {train_end}; calibration through {calibration_end}',flush=True)
    x,y,_=load_period(config.TRAIN_FILE,FEATURES,None,train_end)
    scaler=RobustChannelScaler().fit(x,y);x=scaler.transform(x)
    starts=nominal_sample(y,a.max_windows,a.seed);np.save(output/'training_starts.npy',starts)
    val,_,vt=load_period(config.TRAIN_FILE,FEATURES,train_end,calibration_end);val=scaler.transform(val)
    if digest(config.TRAIN_FILE)!=TRAIN_SHA:raise ValueError('CSV changed while loading; stop before training')
    evaluator=evaluator_for(vt)
    decisions={}
    arms = ['masked','masked_overlap'] if getattr(a,'comparison','baseline')=='hidden-overlap' else ['baseline','masked']
    for name in arms:
        decisions[name]=train_arm(name,x,starts,val,evaluator,scaler,output/name,a.epochs,a.batch_size,a.device,a.seed)
    del x,y,val
    print(f'Fold {fold}: both rules frozen; loading {calibration_end} to {assessment_end}',flush=True)
    assessment,_,times=load_period(config.TRAIN_FILE,FEATURES,calibration_end,assessment_end);assessment=scaler.transform(assessment)
    evaluators=assessment_scopes(times);rows=[];dual_rows=[]
    for name in arms:
        masked=name in ('masked', 'masked_overlap');model=(MaskedAE(8) if masked else MultivariateAE(8)).to(a.device)
        checkpoint=torch.load(output/name/'model_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model_state_dict'])
        score=score_windows(model,assessment,masked,a.device,a.batch_size)
        best=decisions[name];rule=best['rule'];pred=postprocess(score>rule['threshold'],int(rule['merge_gap']),int(rule['min_dur']))
        np.save(output/name/'development_score.npy',score);np.save(output/name/'development_prediction.npy',pred)
        for scope, ev in evaluators.items():
            row=dict(fold=fold,model=name,epoch=best['epoch'],calibration_f05=rule['val_esa_f05'],**ev.score(pred))
            dual_rows.append(dict(scope=scope,**row))
            if scope=='system_all_channels':rows.append(row)
    print('Final data-integrity check...',flush=True)
    if digest(config.TRAIN_FILE)!=TRAIN_SHA:raise ValueError('CSV changed during pilot; preserve run but do not accept results')
    pd.DataFrame(dual_rows).to_csv(output/'dual_scope_results.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
    print(pd.DataFrame(dual_rows)[['fold','model','scope','epoch','calibration_f05','EW_F_0.50','TPe','FPe','FNe']].to_string(index=False),flush=True)
    return rows,dual_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',choices=['baseline','hidden-overlap'],default='baseline')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--folds',nargs='+',choices=['2003','2004','2005'],default=['2003'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--max-windows',type=int,default=25000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.max_windows,a.batch_size)<1:p.error('Budgets must be positive')
    if len(set(a.folds))!=len(a.folds):p.error('Fold list must not contain duplicates')
    plan=dict(experiment='masked-pilot-v3',comparison=a.comparison,features=FEATURES,folds={f:fold_boundaries(f) for f in a.folds},intervals='[start,end); expanding training from2000, Jan–Mar calibration, Apr–Dec development assessment',epochs=a.epochs,max_windows=a.max_windows,batch_size=a.batch_size,seed=a.seed,device=a.device,
        selection='Each arm: calibration every epoch, earliest F0.5 tie; freeze both before assessment',
        mask='25% temporal patches, all channels simultaneously; four interleaved phases; training phase=(epoch-1+batch)%4; inference four passes',
        objective='Both use existing pseudo-anomaly objective; masked arm loss only on hidden patches',
        scoring='Both window-mean, overlap-average, channel maximum; masked inference scores each target only while hidden',
        fairness='Same window indices, scaler, base initialization seed, epochs, optimizer and scheduler. Not equal FLOPs: masked inference uses four passes. Mask token adds 128 parameters.',
        limitations='Small single-seed development pilot; not comparable directly to full-run0.8057. Earlier folds already explored, not untouched validation. No2007–2013 access. No OR. Retrospective scoring; inherited preparation.')
    if a.comparison=='hidden-overlap':
        plan['arms']=['masked','masked_overlap']
        plan['augmentation']='Control: uniform valid start. Candidate: uniform start conditional on segment touching a hidden target; same length/type/channel/amplitude laws. No-op rolls remain possible.'
        plan['fairness']='Both masked models; same initialization, training windows, budgets, scaler, scoring and objective. Placement changes RNG trajectory; not identical synthetic examples.'
        plan['decision']='Exploratory 2004 mechanism pilot; report both arms/scopes; no per-year hindsight ensemble or automatic benchmark evaluation. Improvement here alone is insufficient for promotion.'
    print(json.dumps(plan,indent=2),flush=True)
    if a.dry_run:return
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable: run on GPU server')
    print('Verifying training CSV...',flush=True)
    if digest(config.TRAIN_FILE)!=TRAIN_SHA:raise ValueError('Training CSV checksum mismatch')
    plan['train_sha256']=TRAIN_SHA
    plan['annotations']={str(f):digest(f) for f in [config.ANNOTATIONS_FILE,config.ANOMALY_TYPES_FILE]}
    package=Path(__file__).parent
    plan['sources']={str(f.relative_to(package)):digest(f) for f in package.rglob('*.py')}
    if a.output.exists():
        if not a.resume or json.loads((a.output/'protocol.json').read_text())!=plan:raise ValueError('Existing output requires matching --resume; otherwise use new directory')
    else:
        a.output.mkdir(parents=True);(a.output/'protocol.json').write_text(json.dumps(plan,indent=2))
    rows=[];dual_rows=[]
    for fold in a.folds:
        r,d=run_pilot_fold(a,fold,a.output/fold)
        rows.extend(r);dual_rows.extend(d)
        pd.DataFrame(rows).to_csv(a.output/'results.csv',index=False)
        pd.DataFrame(dual_rows).to_csv(a.output/'dual_scope_results.csv',index=False)
    print('All requested folds complete. Local/system scores use identical frozen predictions.',flush=True)


if __name__=='__main__':main()
