"""Train through a 61-day recursive forecast, with optional direct-path consistency.

Training windows end before each forecast origin. Predictions fed back during
rollout remain attached to the autograd graph; no teacher forcing in rollout loss.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from weekly_baselines import read_matrix
from train_model import score

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/v11_weekly'
KEY=['route','date','hour']
EXOG=['daytype','workday','holiday','summer','sin_year','cos_year',
      'temp_mean','precipitation_sum','snowfall_sum','wind_speed_mean',
      'is_school_holiday','is_summer_school_break','edu_school_term',
      'edu_uni_teaching_proxy','edu_uni_exam_proxy','is_pre_new_year_week',
      'new_year_proximity_21d','service_cancelled','service_rerouted',
      'service_reduced_frequency','service_incident_fraction','season_2324']


def dataset():
    d=read_matrix().sort_values(['time','route','hour']).reset_index(drop=True)
    x=pd.read_csv(ROOT/'data/features/08_external_seasonality/moscow_monthly_passengers.csv',sep=';')
    x=x[(x.TransportType=='Трамвай')&x.Year.isin(['2023','2024'])].copy()
    names=['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь']
    x['month']=x.Month.map(dict(zip(names,range(1,13))))
    dates=pd.to_datetime(dict(year=x.Year.astype(int),month=x.month,day=1))
    x['daily']=x.PassengerTraffic.astype(float)/dates.dt.days_in_month
    x['index']=x.daily/x.groupby('Year').daily.transform('mean')
    index=x.groupby('month')['index'].mean()
    assert len(index)==12
    d['season_2324']=d.month.map(index)
    groups=d[['route','hour']].drop_duplicates().reset_index(drop=True)
    assert len(groups)==240 and len(d)==365*240
    y=d.boardings.to_numpy().reshape(365,240).astype(np.float32)
    ex=d[EXOG].to_numpy().reshape(365,240,len(EXOG)).astype(np.float32)
    assert np.isfinite(ex).all()
    return d,y,ex,groups


class WeeklyResidual(nn.Module):
    def __init__(self,features,scale,active):
        super().__init__()
        self.embedding=nn.Embedding(len(scale),8)
        self.net=nn.Sequential(nn.Linear(features*2+6+4+8,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('scale',torch.tensor(scale).reshape(1,-1,1))
        self.register_buffer('active',torch.tensor(active,dtype=torch.float32).reshape(1,-1,1))

    def block(self,history,hist_exog,future_exog,offset=0,direct=False):
        b,g,h,f=future_exog.shape
        slots=torch.arange(h,device=history.device)%7
        ix=21+slots
        lag=[history.index_select(2,ix-7*k) for k in range(4)]
        base=lag[0]
        lag.extend([history[:,:,-1:].expand(-1,-1,h),history.mean(2,keepdim=True).expand(-1,-1,h)])
        lag_features=torch.stack([torch.log1p(v/self.scale) for v in lag],dim=-1)
        anchor=hist_exog.index_select(2,ix)
        steps=torch.arange(1,h+1,device=history.device,dtype=history.dtype)
        age=torch.stack([(steps+offset)/61,steps/7,torch.full_like(steps,min(offset/28,1)),torch.full_like(steps,float(direct))],-1)
        age=age.reshape(1,1,h,4).expand(b,g,h,4)
        emb=self.embedding(torch.arange(g,device=history.device)).reshape(1,g,1,8).expand(b,g,h,8)
        inputs=torch.cat([future_exog,future_exog-anchor,lag_features,age,emb],dim=-1)
        residual=self.net(inputs).squeeze(-1)
        # Pseudocount permits recovery from zero or cancelled anchor intervals.
        log_value=(torch.log(base+30)+residual).clamp(math.log(30),math.log(1e6))
        return (torch.exp(log_value)-30)*self.active

    def rollout(self,history,hist_exog,future_exog,step=7,reset_age=False):
        predictions=[]
        for offset in range(0,future_exog.shape[2],step):
            x=future_exog[:,:,offset:offset+step]
            p=self.block(history,hist_exog,x,offset=0 if reset_age else offset,direct=False)
            predictions.append(p)
            # Keep the graph intact: later-horizon losses train earlier predictions.
            history=torch.cat([history,p],dim=2)[:,:,-28:]
            hist_exog=torch.cat([hist_exog,x],dim=2)[:,:,-28:]
        return torch.cat(predictions,dim=2)


def tensors(y,ex,origins,horizon,device):
    hist=np.stack([y[o-28:o].T for o in origins])
    hx=np.stack([ex[o-28:o].transpose(1,0,2) for o in origins])
    fx=np.stack([ex[o:o+horizon].transpose(1,0,2) for o in origins])
    target=np.stack([y[o:o+horizon].T for o in origins])
    assert np.isfinite(hist).all()
    return [torch.from_numpy(a).to(device) for a in [hist,hx,fx,target]]


def weighted_mae(pred,target,tail_weight=1,normalizer=None):
    w=torch.ones(pred.shape[-1],device=pred.device)
    w[-7:]=tail_weight
    denominator=target.sum() if normalizer is None else normalizer
    return ((pred-target).abs()*w).sum()/(denominator.detach().clamp_min(1)*w.mean())


def train(y,ex,origin,mode,epochs,device):
    torch.manual_seed(20260926);np.random.seed(20260926)
    mean=ex[:origin].mean(axis=(0,1),keepdims=True)
    std=ex[:origin].std(axis=(0,1),keepdims=True).clip(.1)
    z=(ex-mean)/std
    scale=np.maximum(30,np.quantile(y[:origin],.8,axis=0)).astype(np.float32)
    active=np.any(y[:origin]>0,axis=0)
    model=WeeklyResidual(ex.shape[-1],scale,active).to(device)
    horizon=7 if mode=='step7' else 61
    origins=np.arange(35,origin-horizon+1,5)
    assert len(origins)>0 and max(origins)+horizon<=origin
    all_data=tensors(y,z,origins,horizon,device)
    assert torch.isfinite(all_data[-1]).all()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
    rng=np.random.default_rng(20260926)
    start=time.monotonic();model.train()
    for epoch in range(epochs):
        order=rng.permutation(len(origins));loss_sum=0
        for start_batch in range(0,len(origins),4):
            ids=torch.tensor(order[start_batch:start_batch+4],device=device)
            history,hx,fx,target=[a.index_select(0,ids) for a in all_data]
            optimizer.zero_grad(set_to_none=True)
            if mode=='direct61':
                direct=model.block(history,hx,fx,direct=True)
                loss=weighted_mae(direct,target)
            elif mode=='step7':
                pred=model.block(history,hx,fx,direct=False)
                loss=weighted_mae(pred,target)
            else:
                pred=model.rollout(history,hx,fx,step=1 if 'daily' in mode else 7)
                tail=2 if mode.endswith('tail2') else 1
                loss=weighted_mae(pred,target,tail)
                if mode.startswith('consistent'):
                    direct=model.block(history,hx,fx,direct=True)
                    agreement=weighted_mae(pred,direct,tail,normalizer=target.sum())
                    consistency=5. if '_c5_' in mode else 1. if '_c1_' in mode else .1
                    loss=(loss+.5*weighted_mae(direct,target,tail)+consistency*agreement)/(1.5+consistency)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach().cpu())
        if epoch in [0,epochs-1] or (epoch+1)%20==0:
            print('TRAIN',origin,mode,epoch+1,'loss',round(loss_sum/math.ceil(len(origins)/4),5),'seconds',round(time.monotonic()-start,1),flush=True)
    return model,(mean,std),{'training_origins':len(origins),'max_training_label_day':int(max(origins)+horizon-1),'forecast_origin_day':origin,'seconds':time.monotonic()-start,'epochs':epochs}


@torch.no_grad()
def predict(model,y,ex,origin,horizon,normalization,device,mode,reset_age=False):
    model.eval();mean,std=normalization
    history,hx,fx,_=tensors(y,(ex-mean)/std,[origin],horizon,device)
    if mode=='direct':p=model.block(history,hx,fx,direct=True)
    else:p=model.rollout(history,hx,fx,step=int(mode),reset_age=reset_age)
    return p[0].T.cpu().numpy().reshape(-1)


def audit_gradient(model,y,ex,origin,normalization,device,step=7):
    model.eval();mean,std=normalization
    history,hx,fx,_=tensors(y,(ex-mean)/std,[origin],61,device)
    first=model.block(history,hx,fx[:,:,:step],direct=False);first.retain_grad()
    h=torch.cat([history,first],2)[:,:,-28:]
    h_ex=torch.cat([hx,fx[:,:,:step]],2)[:,:,-28:]
    for offset in range(step,61,step):
        xx=fx[:,:,offset:offset+step]
        final=model.block(h,h_ex,xx,offset=offset,direct=False)
        h=torch.cat([h,final],2)[:,:,-28:];h_ex=torch.cat([h_ex,xx],2)[:,:,-28:]
    final.sum().backward()
    norm=float(first.grad.abs().sum().detach().cpu())
    model.zero_grad(set_to_none=True)
    assert norm>0,'Last-block gradient cannot reach the first predicted week'
    return norm


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--origins',default='181,212,243')
    parser.add_argument('--modes',default='step7,direct61,rollout7,consistent7,consistent7_tail2')
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    args=parser.parse_args();DEST.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    device=torch.device(args.device)
    d,y,ex,groups=dataset();rows=[];audits=[]
    modes=args.modes.split(',')
    for origin in map(int,args.origins.split(',')):
        horizon=min(61,304-origin)
        frame=d[(d.time>=origin)&(d.time<origin+horizon)].reset_index(drop=True)
        cutoff=frame.date.min();truth=frame.boardings.to_numpy()
        for mode in modes:
            tag=f'{cutoff}_{mode}_e{args.epochs}'
            cache=DEST/(tag+'.npz');metadata=DEST/(tag+'.json')
            cache_ok=cache.exists() and metadata.exists()
            if cache_ok and mode=='step7':
                cache_ok=json.loads(metadata.read_text()).get('reset_recursive_age') is True
            if cache_ok:
                ps=dict(np.load(cache));audit=json.loads(metadata.read_text())
            else:
                model,norm,audit=train(y,ex,origin,mode,args.epochs,device)
                if mode=='direct61':paths=['direct']
                elif mode.startswith('consistent'):paths=['7','1','direct']
                else:paths=['7','1']
                ps={p:predict(model,y,ex,origin,horizon,norm,device,p,reset_age=mode=='step7') for p in paths}
                changed=y.copy();changed[origin:]=999999
                audit['target_perturbation_max_abs_change']=float(np.max(np.abs(predict(model,changed,ex,origin,horizon,norm,device,paths[0],reset_age=mode=='step7')-ps[paths[0]])))
                assert audit['target_perturbation_max_abs_change']<1e-4
                if horizon==61:audit['last_to_first_block_gradient_norm']=audit_gradient(model,y,ex,origin,norm,device,step=1 if 'daily' in mode else 7)
                if mode.startswith('consistent'):
                    ps['blend_direct_recursive']=.5*ps['direct']+.5*ps['7']
                    audit['recursive_direct_disagreement_wape']=float(np.abs(ps['7']-ps['direct']).sum()/truth.sum())
                audit.update(mode=mode,origin=cutoff,device=str(device),torch_version=torch.__version__,log_ratio_pseudocount=30,
                             reset_recursive_age=mode=='step7',rollout_step_training=1 if 'daily' in mode else 7)
                torch.save({'state_dict':model.cpu().state_dict(),'normalization':norm,'features':EXOG,'group_order':groups.to_dict('records'),'training_audit':audit},DEST/(tag+'.pt'))
                np.savez(cache,**ps);metadata.write_text(json.dumps(audit,indent=2))
                del model
                if device.type=='mps':torch.mps.empty_cache()
            audits.append(audit)
            out=frame[KEY+['boardings']].copy()
            for path,p in ps.items():
                out[path]=p
                days=frame.time.to_numpy()-origin+1
                for label,mask in [('all',days>0),('days_1_7',days<=7),('days_8_28',(days>=8)&(days<=28)),('days_29_61',days>=29),('last_7_days',days>days.max()-7)]:
                    if mask.any():rows.append(dict(origin=cutoff,mode=mode,path=path,window=label,score=score(truth[mask],p[mask]),epochs=args.epochs))
                print('SCORE',cutoff,mode,path,round(score(truth,p),6),flush=True)
            out.to_csv(DEST/('validation_'+tag+'.csv'),index=False)
            pd.DataFrame(rows).to_csv(DEST/'neural_metrics.csv',index=False)
    report={'method':'log-ratio residual, full 61-day unrolled supervision, optional direct supervision and consistency penalty',
            'recurrence_steps_training':{m:(None if m=='direct61' else 1 if 'daily' in m else 7) for m in modes},'recurrence_steps_evaluated':[1,7],
            'loss':'Count-space weighted MAE; no hidden labels; final 7-day weight compared at 1 and 2.',
            'consistency_weights':{m:(5. if '_c5_' in m else 1. if '_c1_' in m else .1) for m in modes if m.startswith('consistent')},'direct_supervision_weight':.5,
            'known_best_public_score':.89214,'public_score':None,'active_model_replaced':False,
            'evaluation':'Previously explored, overlapping development windows. The daily-named variants train through a one-day rollout; other recurrent variants train through seven-day rollout. Both inference steps are compared.',
            'exogenous_caveat':'Supplied retrospective weather and service flags; external seasonal index uses 2023–2024 only.',
            'audits':audits,'scores':pd.DataFrame(rows).query('window == "all"').to_dict('records')}
    (DEST/'neural_report.json').write_text(json.dumps(report,indent=2));print('DONE UNROLLED',flush=True)


if __name__=='__main__':main()
