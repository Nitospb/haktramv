"""v16: a separate whole-week network, recursively forecasting nine blocks.

Every route's entire four-week history is encoded as weekly patches. Each output
head predicts all 168 next-week hours jointly. No daily model predictions enter
training or inference; a later ensemble can therefore be tested independently.
"""
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import cascaded_forecast as engine
from cascaded_dense import masked_error


class WeeklyJoint(nn.Module):
    def __init__(self, scale, active, bound=.5):
        super().__init__(); self.bound = bound
        self.week_encoder = nn.Sequential(nn.Linear(336,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.history_encoder = nn.Sequential(nn.Linear(4*64,64),nn.SiLU())
        self.query = nn.Linear(64,32,bias=False); self.key = nn.Linear(64,32,bias=False)
        self.value = nn.Linear(64,32,bias=False)
        self.route_embedding = nn.Embedding(10,16)
        self.decoder = nn.Sequential(nn.Linear(7*len(engine.EXOG)*2+168*2+96+16+4+7,128),nn.SiLU(),
                                     nn.Linear(128,128),nn.SiLU(),nn.Linear(128,168))
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer('scale',torch.tensor(scale).reshape(1,240,1))
        self.register_buffer('active',torch.tensor(active,dtype=torch.float32).reshape(1,240,1))
        self.register_buffer('active_routes',torch.tensor(active).reshape(10,24).any(1))

    @staticmethod
    def weeks(x):
        return x.reshape(len(x),10,24,28).permute(0,1,3,2).reshape(len(x),10,4,168)

    def block(self, history, hx, future, hbase, fbase, offset=0):
        b,g,n,e = future.shape
        assert 1 <= n <= 7
        ratios = torch.log((history+30)/(hbase+30)).clamp(-3,3)
        levels = torch.log1p(history/self.scale)
        encoded = self.week_encoder(torch.cat([self.weeks(ratios),self.weeks(levels)],-1))
        state = self.history_encoder(encoded.reshape(b,10,-1))
        attn = (self.query(state) @ self.key(state).transpose(1,2))/math.sqrt(32)
        attn = attn.masked_fill(~self.active_routes[None,None,:],-1e4).softmax(-1)
        neighbours = attn @ self.value(state)
        fx = future.reshape(b,10,24,n,e).mean(2)
        previous_x = hx[:,:,-7:].reshape(b,10,24,7,e).mean(2)[:,:,:n]
        delta_x = fx-previous_x
        fx = F.pad(fx,(0,0,0,7-n)).reshape(b,10,-1)
        delta_x = F.pad(delta_x,(0,0,0,7-n)).reshape(b,10,-1)
        base_level = torch.log1p(fbase/self.scale).reshape(b,10,24,n).transpose(2,3)
        base_level = F.pad(base_level,(0,0,0,7-n)).reshape(b,10,168)
        last_week = ratios[:,:,-7:].reshape(b,10,24,7).transpose(2,3).reshape(b,10,168)
        emb = self.route_embedding(torch.arange(10,device=history.device))[None].expand(b,-1,-1)
        age = torch.tensor([offset/61,min(offset/28,1),n/7,offset/7/9],device=history.device,dtype=history.dtype)
        age = age[None,None].expand(b,10,-1)
        mask = (torch.arange(7,device=history.device) < n).float()[None,None].expand(b,10,-1)
        x = torch.cat([state,neighbours,fx,delta_x,base_level,last_week,emb,age,mask],-1)
        delta = self.bound*torch.tanh(self.decoder(x)).reshape(b,10,7,24)[:,:,:n]
        delta = delta.transpose(2,3).reshape(b,240,n)
        feedback = .3*ratios[:,:,21:21+n].clamp(-math.log(4),math.log(4))
        return ((fbase+30)*(delta+feedback).exp()-30).clamp_min(0)*self.active

    def rollout(self, history, hx, future, hbase, fbase, step=7, track_first=False):
        assert step == 7
        predictions = []; first = None
        for offset in range(0,future.shape[2],7):
            fx = future[:,:,offset:offset+7]; fb = fbase[:,:,offset:offset+7]
            p = self.block(history,hx,fx,hbase,fb,offset)
            if track_first and first is None:
                first = p; first.retain_grad()
            predictions.append(p)
            history = torch.cat([history,p],2)[:,:,-28:]
            hx = torch.cat([hx,fx],2)[:,:,-28:]
            hbase = torch.cat([hbase,fb],2)[:,:,-28:]
        result = torch.cat(predictions,2)
        return (result,first) if track_first else result


def fit(y, ex, origin, bank, config, seed, epochs, device):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mean = ex[:origin].mean((0,1),keepdims=True)
    std = ex[:origin].std((0,1),keepdims=True).clip(.1)
    z = (ex-mean)/std
    scale = np.maximum(30,np.quantile(y[:origin],.8,axis=0)).astype(np.float32)
    active = (y[:origin]>0).any(0)
    model = WeeklyJoint(scale,active,config['bound']).to(device)
    origins = np.arange(35,origin); available = np.minimum(61,origin-origins)
    y_training = y.copy(); y_training[origin:] = 0
    data = engine.batch(y_training,z,origins,bank,device,61)
    mask = torch.tensor(np.arange(61)[None]<available[:,None],dtype=torch.float32,device=device)
    assert torch.isfinite(data[-1]).all() and torch.count_nonzero(data[-1]*(1-mask[:,None])).item()==0
    weights = torch.tensor(.5**((origin-1-origins)/120)*np.minimum(1,available/14),dtype=torch.float32,device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    sparse = np.unique(np.r_[np.arange(35,origin-61+1,5),origin-61])
    batches_per_epoch = math.ceil(len(sparse)/4); seen = set(); started = time.monotonic()
    for epoch in range(epochs):
        model.train(); total = 0
        for _ in range(batches_per_epoch):
            sampled = rng.choice(len(origins),size=4,replace=False); seen.update(origins[sampled].tolist())
            ids = torch.tensor(sampled,device=device)
            h,hx,fx,hb,fb,target = [a.index_select(0,ids) for a in data]
            m = mask.index_select(0,ids); w = weights.index_select(0,ids)
            norm = (target*m[:,None]).sum((1,2)).detach().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            p = model.rollout(h,hx,fx,hb,fb)
            per,_ = masked_error(p,target,m,w,True)
            loss = (per/norm*w).sum()/w.sum()
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            total += float(loss.detach().cpu())
        if epoch==0 or (epoch+1)%20==0:
            print('WEEKLY',origin,config,epoch+1,round(total/batches_per_epoch,6),round(time.monotonic()-started,1),'seconds',flush=True)
    return model,(mean,std),{'forecast_origin_day':origin,'max_training_label_day':origin-1,
                            'long_training_origins':int((available==61).sum()),'short_training_origins':{'1':len(origins),'7':int((available>=7).sum())},
                            'short_latest_origin':{'1':origin-1,'7':origin-7},'available_origins':len(origins),'distinct_origins_used':len(seen),
                            'future_training_targets_zeroed_before_batching':True,'loss_mask_excludes_unavailable_targets':True,
                            'seed':seed,'epochs':epochs,'config':config,'seconds':time.monotonic()-started,
                            'weekly_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'architecture':'whole-week joint 168-hour head, independently trained from daily cascade'}


@torch.no_grad()
def predict(model,y,ex,origin,horizon,norm,bank,device):
    model.eval(); mean,std = norm
    h,hx,fx,hb,fb,_ = engine.batch(y,(ex-mean)/std,[origin],bank,device,horizon)
    return {'7':model.rollout(h,hx,fx,hb,fb)[0].T.cpu().numpy().reshape(-1)}


def audits(model,y,ex,origin,horizon,norm,bank,device):
    model.eval(); mean,std = norm
    h,hx,fx,hb,fb,_ = engine.batch(y,(ex-mean)/std,[origin],bank,device,horizon)
    p,first = model.rollout(h,hx,fx,hb,fb,track_first=True)
    p[:,:,-1].sum().backward(); flow = float(first.grad.abs().sum().detach().cpu()); model.zero_grad(set_to_none=True)
    h = h.clone().requires_grad_(True)
    p = model.block(h,hx,fx[:,:,:7],hb,fb[:,:,:7])
    grad = torch.autograd.grad(p[:,:24].sum(),h)[0]
    cross = float(grad[:,24:].abs().sum().detach().cpu())
    assert flow>0 and cross>0
    return {'last_day_gradient_to_first_step_7':flow,'cross_route_gradient_l1':cross}


def main():
    engine.DEST = engine.ROOT/'outputs/v16_weekly'
    engine.CONFIGS = {'weekly_joint':dict(bound=.5,learning_rate=.001,weight_decay=.03),
                      'weekly_regularized':dict(bound=.35,learning_rate=.0005,weight_decay=.1)}
    engine.fit = fit; engine.predict = predict; engine.audits = audits
    engine.main()


if __name__=='__main__':
    main()
