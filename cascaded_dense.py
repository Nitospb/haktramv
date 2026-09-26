"""Train from every historical origin, with an observed-prefix loss mask.

Early origins supervise a full 61-day daily/weekly cascade. Recent origins
supervise all remaining observed days, down to the latest one-day transition.
Future labels are zeroed BEFORE batches are built and excluded from every loss.
"""
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import cascaded_forecast as engine


def masked_error(prediction, target, mask, weights, multiscale):
    length = prediction.shape[-1]
    available = mask.sum(1)
    position = torch.arange(length, device=prediction.device)[None]
    temporal = mask * torch.where(position >= (available-7).clamp_min(0)[:, None], 2., 1.)
    temporal = temporal/(temporal.sum(1)/available.clamp_min(1))[:, None]
    per = ((prediction-target).abs()*temporal[:, None]).sum((1, 2))
    if multiscale:
        p = prediction.reshape(len(prediction), 10, 24, length).sum(2)
        t = target.reshape(len(target), 10, 24, length).sum(2)
        daily = ((p-t).abs()*temporal[:, None]).sum((1, 2))
        system = ((p.sum(1)-t.sum(1)).abs()*temporal).sum(1)
        per = (per+.15*daily+.1*system)/1.25
    return per, weights


def fit(y, ex, origin, bank, config, seed, epochs, device):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mean = ex[:origin].mean((0, 1), keepdims=True)
    std = ex[:origin].std((0, 1), keepdims=True).clip(.1)
    z = (ex-mean)/std
    scale = np.maximum(30, np.quantile(y[:origin], .8, axis=0)).astype(np.float32)
    active = (y[:origin] > 0).any(0)
    model = engine.Cascade(scale, active).to(device)
    origins = np.arange(35, origin)
    # Physically remove all unavailable target values before constructing data.
    y_training = y.copy(); y_training[origin:] = 0
    data = engine.batch(y_training, z, origins, bank, device, 61)
    available = np.minimum(61, origin-origins)
    mask = torch.tensor(np.arange(61)[None] < available[:, None], dtype=torch.float32, device=device)
    assert torch.isfinite(data[-1]).all() and origins.max()+available[-1] == origin
    assert torch.count_nonzero(data[-1]*(1-mask[:, None])).item() == 0
    weights = torch.tensor(.5**((origin-1-origins)/120)*np.minimum(1,available/14),dtype=torch.float32,device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
    # Match the approximate number of optimizer updates in the sparse control;
    # sample more DISTINCT origins rather than multiply epochs by ~five.
    sparse = np.unique(np.r_[np.arange(35,origin-61+1,5),origin-61])
    batches_per_epoch = math.ceil(len(sparse)/4)
    seen = set(); started = time.monotonic()
    for epoch in range(epochs):
        model.train(); total = 0
        for _ in range(batches_per_epoch):
            sampled = rng.choice(len(origins),size=4,replace=False)
            seen.update(origins[sampled].tolist())
            ids = torch.tensor(sampled,device=device)
            h,hx,fx,hb,fb,target = [a.index_select(0,ids) for a in data]
            m = mask.index_select(0,ids); w = weights.index_select(0,ids)
            norm = (target*m[:,None]).sum((1,2)).detach().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            daily = model.rollout(h,hx,fx,hb,fb,1)
            weekly = model.rollout(h,hx,fx,hb,fb,7)
            a,_ = masked_error(daily,target,m,w,True)
            b,_ = masked_error(weekly,target,m,w,True)
            c,_ = masked_error(daily,weekly,m,w,False)
            per = (.5*(a+b)+config['consistency']*c)/(1+config['consistency'])
            loss = (per/norm*w).sum()/w.sum()
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step(); total += float(loss.detach().cpu())
        if epoch == 0 or (epoch+1)%20 == 0:
            print('DENSE',origin,config,epoch+1,round(total/batches_per_epoch,6),
                  round(time.monotonic()-started,1),'seconds',flush=True)
    return model,(mean,std),{'forecast_origin_day':origin,'max_training_label_day':origin-1,
                            'long_training_origins':int((available==61).sum()),
                            'short_training_origins':{'1':int((available>=1).sum()),'7':int((available>=7).sum())},
                            'short_latest_origin':{'1':origin-1,'7':origin-7},
                            'available_origins':len(origins),'distinct_origins_used':len(seen),
                            'training_horizon_min':int(available.min()),'training_horizon_max':int(available.max()),
                            'future_training_targets_zeroed_before_batching':True,
                            'loss_mask_excludes_unavailable_targets':True,
                            'optimizer_updates':epochs*batches_per_epoch,'seed':seed,'epochs':epochs,'config':config,
                            'seconds':time.monotonic()-started,
                            'dense_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    engine.CONFIGS = {'dense_c1':dict(consistency=1.),'dense_c5':dict(consistency=5.)}
    engine.fit = fit
    engine.main()


if __name__ == '__main__':
    main()
