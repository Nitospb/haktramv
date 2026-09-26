"""Shared-route Transformer: 56 historical days -> 61 future daily profiles.

MPS training, bounded wall time, fixed checkpoints, selection/control separation.
"""
import argparse
import json
import time
import os
import fcntl
import numpy as np
import pandas as pd
import torch
from torch import nn
from night_forecast_common import ROOT,OUT,Data,SELECTION,CONTROL,save_json,status,score

DAILY=['dow','daytype','workday','holiday','is_school_holiday','is_day_before_holiday',
       'is_day_after_holiday','consecutive_days_off','edu_school_term','edu_first_week_back',
       'edu_uni_teaching_proxy','edu_uni_exam_proxy','service_cancelled','service_rerouted',
       'service_reduced_frequency','service_recent_restoration','drone_report_count',
       'reported_drone_count_max','connectivity_report_count','mobile_internet_restrictions_possible',
       'scheduled_mean_moving_trams','scheduled_terminal_departures','route_length_km_max',
       'xfer_transfer_metro_500m_share']
HOURLY=['route_weather_temperature_2m','route_weather_precipitation','route_weather_snowfall',
        'route_weather_wind_speed_10m','road_crash_700m_past3h','scheduled_stop_events',
        'service_cancelled','drone_report_in_past_6h']


class Windows:
    def __init__(self,data,origin):
        self.data=data;self.origin=origin
        self.daily=[c for c in DAILY if c in data.d];self.hourly=[c for c in HOURLY if c in data.d]
        daily=data.d[self.daily].to_numpy(dtype=np.float32).reshape(365,10,24,-1).mean(2)
        hourly=data.d[self.hourly].to_numpy(dtype=np.float32).reshape(365,10,24,-1).reshape(365,10,-1)
        ex=np.concatenate([daily,hourly],axis=-1)
        train=ex[:origin].reshape(-1,ex.shape[-1]);self.mu=np.nanmedian(train,axis=0)
        self.sd=np.maximum(1,np.nanstd(train,axis=0))
        self.ex=np.clip((np.nan_to_num(ex,nan=-1)-self.mu)/self.sd,-8,8).astype(np.float32)
        self.scale=np.maximum(20,np.nanmean(data.y[:origin],axis=(0,2))).astype(np.float32)
        self.global_scale=max(20,float(np.nanmean(data.y[:origin].sum(1))))
        self.normalizer=max(20,float(np.nanmean(data.y[:origin])))
        self.active=[r for r,route in enumerate(data.routes) if route!=5]
        self.samples=[(c,r) for c in range(56,origin-6) for r in self.active]
        self.width=self.ex.shape[-1]

    def get(self,c,r,training=True):
        data=self.data;h=min(61,365-c)
        base=data.profile(c,56)[data.daytype[c:c+h],r]
        future=np.zeros((61,self.width+25),np.float32)
        future[:h,:self.width]=self.ex[c:c+h,r]
        future[:h,self.width:self.width+24]=np.log1p(base/self.scale[r])
        future[:, -1]=np.arange(1,62)/61
        ypast=data.y[c-56:c,r]/self.scale[r]
        globalpast=data.y[c-56:c].sum(1)/self.global_scale
        past=np.concatenate([np.log1p(ypast),np.log1p(globalpast),self.ex[c-56:c,r]],axis=-1).astype(np.float32)
        assert np.isfinite(past).all()
        baseline=np.zeros((61,24),np.float32);baseline[:h]=base
        target=np.zeros((61,24),np.float32);mask=np.zeros((61,24),np.float32)
        if training:
            n=min(h,self.origin-c);assert c+n<=self.origin
            target[:n]=data.y[c:c+n,r];mask[:n]=np.isfinite(target[:n])
            target=np.nan_to_num(target)
        return past,future,baseline,target,mask,r

    def batch(self,items,device,training=True):
        values=[self.get(c,r,training) for c,r in items]
        return [torch.tensor(np.stack([v[k] for v in values]),device=device,dtype=torch.long if k==5 else torch.float32) for k in range(6)]

class Network(nn.Module):
    def __init__(self,width,hidden=192):
        super().__init__();self.route=nn.Embedding(10,hidden)
        self.past=nn.Linear(width+48,hidden);self.future=nn.Linear(width+25,hidden)
        self.past_pos=nn.Parameter(torch.randn(1,56,hidden)*.02)
        self.future_pos=nn.Parameter(torch.randn(1,61,hidden)*.02)
        enc=nn.TransformerEncoderLayer(hidden,6,hidden*4,dropout=.15,activation='gelu',batch_first=True,norm_first=True)
        dec=nn.TransformerDecoderLayer(hidden,6,hidden*4,dropout=.15,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,4,enable_nested_tensor=False)
        self.decoder=nn.TransformerDecoder(dec,3)
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,24))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,past,future,baseline,route):
        emb=self.route(route)[:,None]
        memory=self.encoder(self.past(past)+self.past_pos+emb)
        decoded=self.decoder(self.future(future)+self.future_pos+emb,memory)
        return torch.clamp((baseline+5)*torch.exp(.8*torch.tanh(self.head(decoded)))-5,min=0)


def predict(model,windows,origin,device):
    model.eval()
    with torch.no_grad():
        past,future,base,_,_,routes=windows.batch([(origin,r) for r in range(10)],device,False)
        result=model(past,future,base,routes).cpu().numpy().transpose(1,0,2)
    return result.reshape(-1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=2)
    parser.add_argument('--steps',type=int,default=6000);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    folder=OUT/('sequence_smoke' if args.smoke else 'sequence');folder.mkdir(parents=True,exist_ok=True)
    lock=open(folder/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    torch.set_num_threads(4);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data=Data();data.causal_check();end=time.time()+3600*args.hours
    checkpoints=sorted(set([500,1500,3000,args.steps]));records=[];selected=None
    for origin in ([151] if args.smoke else SELECTION+[CONTROL,304]):
        if time.time()>end:status(folder,'budget_exhausted',records=records);return
        if origin==CONTROL:
            r=pd.DataFrame(records);r=r[r.split=='selection']
            eligible=r.groupby('step').origin.nunique();eligible=eligible[eligible==len(SELECTION)].index
            selected=int(r.groupby('step').score.mean().loc[eligible].idxmax())
            save_json(folder/'selection.json',dict(selected_step=selected,selection_origins=SELECTION,control_origin=CONTROL))
        steps=8 if args.smoke else (args.steps if origin in SELECTION else selected)
        windows=Windows(data,origin);torch.manual_seed(270927);rng=np.random.default_rng(270927)
        model=Network(windows.width).to(device);params=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=2e-5)
        # Saved checkpoints contain only tensors and numeric/string metadata.
        existing=sorted(folder.glob(str(origin)+'_step*.pt'),key=lambda p:int(p.stem.split('step')[1]))
        start_step=0
        if existing and not args.smoke:
            eligible=[p for p in existing if int(p.stem.split('step')[1])<=steps]
            if eligible:
                checkpoint=torch.load(eligible[-1],map_location=device,weights_only=True)
                model.load_state_dict(checkpoint['state_dict']);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler']);start_step=checkpoint['step']
                print('RESUME',origin,start_step,flush=True)
        status(folder,'training',origin=origin,parameters=params,device=str(device),target_steps=steps,step=start_step)
        start=time.time();last_loss=None
        # Recreate previous selection metrics when resuming rather than silently
        # losing checkpoints from earlier folds.
        prior_file=folder/('metrics_'+str(origin)+'.json')
        fold_records=json.loads(prior_file.read_text()) if prior_file.exists() and not args.smoke else []
        for step in range(start_step+1,steps+1):
            model.train();ids=rng.integers(0,len(windows.samples),size=24)
            batch=windows.batch([windows.samples[i] for i in ids],device)
            past,future,baseline,target,mask,routes=batch
            optimizer.zero_grad(set_to_none=True)
            pred=model(past,future,baseline,routes)
            hourly=((pred-target).abs()*mask).sum()/mask.sum().clamp_min(1)
            daily=((pred*mask).sum(-1)-(target*mask).sum(-1)).abs().sum()/mask.sum().clamp_min(1)
            weekly=((pred[:,:56]*mask[:,:56]).reshape(24,8,7,24).sum((2,3))-(target[:,:56]*mask[:,:56]).reshape(24,8,7,24).sum((2,3))).abs().sum()/mask.sum().clamp_min(1)
            loss=(hourly+.15*daily+.05*weekly)/windows.normalizer
            if not torch.isfinite(loss):raise RuntimeError('Non-finite sequence loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();last_loss=float(loss.detach().cpu())
            if step%100==0 or step==steps:
                status(folder,'training',origin=origin,step=step,target_steps=steps,loss=last_loss,parameters=params,seconds=time.time()-start,device=str(device))
                print('STEP',origin,step,'loss',last_loss,'seconds',time.time()-start,flush=True)
            if step in checkpoints or step==steps or time.time()>end:
                dest=folder/(str(origin)+'_step'+str(step)+'.pt')
                torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,origin=origin,width=windows.width,parameters=params),dest)
                save_json(dest.with_suffix('.json'),dict(origin=origin,step=step,parameters=params,daily_features=windows.daily,hourly_features=windows.hourly,mu=windows.mu.tolist(),sd=windows.sd.tolist(),scale=windows.scale.tolist(),global_scale=windows.global_scale,training_last_day=origin-1))
                p=predict(model,windows,origin,device)
                result=data.export(folder,'sequence_step'+str(step),origin,p)
                result.update(step=step,origin=origin,split='selection' if origin in SELECTION else ('control' if origin==CONTROL else 'submission'))
                fold_records=[r for r in fold_records if r['step']!=step]+[result]
                save_json(prior_file,fold_records)
                replay=Network(windows.width).to(device);replay.load_state_dict(torch.load(dest,map_location=device,weights_only=True)['state_dict'])
                np.testing.assert_allclose(p,predict(replay,windows,origin,device),rtol=1e-6,atol=1e-3)
                del replay
                print('EVAL',result,flush=True)
            if time.time()>end:status(folder,'budget_exhausted',origin=origin,step=step);return
        records.extend(fold_records);save_json(folder/'metrics.json',records)
        del model,optimizer,windows
        if device.type=='mps':torch.mps.empty_cache()
    save_json(folder/'report.json',dict(results=records,selected_step=selected,device=str(device),parameters=params,
        architecture='4-layer 192-wide Transformer encoder over 56 historical daily 24-hour profiles plus global cross-route history; 3-layer decoder over 61 future calendar/weather/schedule tokens, bounded residual to causal weekday profile.',
        loss='Absolute hourly error plus 0.15 daily-total and 0.05 weekly-total errors; unseen training-tail targets masked.',
        checkpoint_replay=True,public_score=None,active_model_changed=False,
        limitations=['Limited 2025 target history; large network may overfit.','Previously explored development cutoffs.','Retrospective external features allowed; no hidden Moscow target inputs.','Route 5 zero in standalone model.']))
    status(folder,'smoke_passed' if args.smoke else 'complete',results=records,selected_step=selected);print('DONE',flush=True)


if __name__=='__main__':main()
