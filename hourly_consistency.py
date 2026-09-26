"""v14: coupled hourly residual states and full 1,464-hour supervision.

An affine recurrence is evaluated by a differentiable parallel prefix operation,
equivalent to a serial recurrence. No actual future boardings enter the state.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from anchored_consistency import ART, ProfileBank
from train_model import score
from unrolled_weekly import EXOG, KEY, dataset

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v14_hourly'
STEPS = [1, 3, 6, 12, 24]
CONFIGS = {f'h{step}': [step] for step in STEPS}
CONFIGS['mixed'] = STEPS


def to_hours(a):
    """day, route*hour, ... -> hour, route, ..."""
    tail = a.shape[2:]
    return a.reshape(len(a), 10, 24, *tail).swapaxes(1, 2).reshape(len(a)*24, 10, *tail)


def to_rows(a):
    return a.reshape(-1, 24, 10).swapaxes(1, 2).reshape(-1)


def affine_scan(a, b):
    """Inclusive composition of s[t] = a[t]*s[t-1] + b[t], over dim 2."""
    distance = 1
    while distance < a.shape[2]:
        aa = torch.cat([a[:, :, :distance], a[:, :, distance:]*a[:, :, :-distance]], dim=2)
        bb = torch.cat([b[:, :, :distance], b[:, :, distance:]+a[:, :, distance:]*b[:, :, :-distance]], dim=2)
        a, b = aa, bb
        distance *= 2
    return a, b


def hourly_states(innovation, tau, initial, step):
    """Refresh innovations every step hours; evolve the residual state hourly."""
    batch, routes, length, channels = innovation.shape
    pad = (-length) % step
    count = (length+pad)//step
    durations = torch.full((count,), float(step), device=innovation.device)
    durations[-1] = length-(count-1)*step
    u = F.pad(innovation, (0, 0, 0, pad)).reshape(batch, routes, count, step, channels).sum(3)/durations[None, None, :, None]
    t = F.pad(tau, (0, 0, 0, pad)).reshape(batch, routes, count, step, channels).sum(3)/durations[None, None, :, None]
    a = torch.exp(-durations[None, None, :, None]/t)
    pa, pb = affine_scan(a, (1-a)*u)
    after = pb+pa*initial[:, :, None, :]
    before = torch.cat([initial[:, :, None, :], after[:, :, :-1]], dim=2)
    age = torch.arange(1, step+1, dtype=innovation.dtype, device=innovation.device)
    decay = torch.exp(-age[None, None, None, :, None]/t[:, :, :, None, :])
    states = decay*before[:, :, :, None, :] + (1-decay)*u[:, :, :, None, :]
    return states.reshape(batch, routes, count*step, channels)[:, :, :length]


class HourlyResidual(nn.Module):
    def __init__(self, scale, active):
        super().__init__()
        self.embedding = nn.Embedding(240, 8)
        self.net = nn.Sequential(nn.Linear(len(EXOG)*2+8+4+8, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 5))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.log_tau = nn.Parameter(torch.tensor([math.log(24), math.log(168)]))
        self.mix = nn.Parameter(torch.tensor(0.))
        self.register_buffer('scale', torch.tensor(scale).reshape(1, 10, 24))
        self.register_buffer('active', torch.tensor(active, dtype=torch.float32).reshape(1, 10, 24))

    def propose(self, history, hx, future, hbase, fbase):
        b, routes, length, features = future.shape
        positions = torch.arange(length, device=history.device)
        hours = positions % 24
        ix = 504+positions % 168
        lags = [history.index_select(2, ix-168*k) for k in range(4)]
        bases = [hbase.index_select(2, ix-168*k) for k in range(4)]
        ratios = [torch.log((a+30)/(b+30)).clamp(-3, 3) for a, b in zip(lags, bases)]
        previous_day = torch.log((history.index_select(2, 648+hours)+30)/(hbase.index_select(2, 648+hours)+30)).clamp(-3, 3)
        recent = torch.log((history[:, :, -24:].sum(2, keepdim=True)+720)/(hbase[:, :, -24:].sum(2, keepdim=True)+720)).clamp(-.7, .7)
        scale = self.scale.index_select(2, hours)
        ratios += [previous_day, recent.expand(-1, -1, length), torch.log1p(fbase/scale),
                   torch.log1p(torch.stack(lags).mean(0)/scale)]
        age = torch.stack([(positions+1)/1464, (positions+1)/168,
                           torch.sin(2*math.pi*hours/24), torch.cos(2*math.pi*hours/24)], dim=-1)
        age = age.reshape(1, 1, length, 4).expand(b, routes, length, 4)
        identities = torch.arange(10, device=history.device)[:, None]*24+hours[None]
        emb = self.embedding(identities).unsqueeze(0).expand(b, -1, -1, -1)
        x = torch.cat([future, future-hx.index_select(2, ix), torch.stack(ratios, -1), age, emb], -1)
        raw = self.net(x)
        innovation = .5*torch.tanh(raw[..., :2])
        shape = .25*torch.tanh(raw[..., 2])
        tau = (self.log_tau.exp()*torch.exp(.25*torch.tanh(raw[..., 3:]))).clamp(3, 720)
        initial = torch.stack([torch.log((history[:, :, -window:].sum(2)+30*window)/
                                        (hbase[:, :, -window:].sum(2)+30*window)).clamp(-.7, .7)
                               for window in [24, 168]], -1)
        return innovation, tau, initial, shape, fbase

    def finish(self, state, shape, base):
        mixing = self.mix.sigmoid()
        correction = mixing*state[..., 0]+(1-mixing)*state[..., 1]+shape
        mask = self.active.index_select(2, torch.arange(base.shape[2], device=base.device) % 24)
        return ((base+30)*torch.exp(correction)-30).clamp_min(0)*mask

    def direct(self, proposed):
        innovation, tau, initial, shape, base = proposed
        return self.finish(innovation, shape, base)

    def rollout(self, proposed, step):
        innovation, tau, initial, shape, base = proposed
        return self.finish(hourly_states(innovation, tau, initial, step), shape, base)


def tensors(y, z, origins, bank, device, horizon=1464):
    # Origins are days; arrays y and z have an hourly time axis.
    arrays = [np.stack([y[o*24-672:o*24].T for o in origins]),
              np.stack([z[o*24-672:o*24].transpose(1, 0, 2) for o in origins]),
              np.stack([z[o*24:o*24+horizon].transpose(1, 0, 2) for o in origins]),
              np.stack([to_hours(bank.get(o))[:672].T for o in origins]),
              np.stack([to_hours(bank.get(o))[672:672+horizon].T for o in origins]),
              np.stack([y[o*24:o*24+horizon].T for o in origins])]
    return [torch.tensor(a, device=device) for a in arrays]


def objective(pred, target, normalizer, weights):
    w = torch.ones(pred.shape[2], device=pred.device); w[-168:] = 2
    losses = ((pred-target).abs()*w).sum((1, 2))/normalizer.clamp_min(1)/w.mean()
    return (losses*weights).sum()/weights.sum()


def fit(yday, y, ex, origin, bank, steps, seed, epochs, device):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mean = ex[:origin*24].mean((0, 1), keepdims=True)
    std = ex[:origin*24].std((0, 1), keepdims=True).clip(.1)
    z = (ex-mean)/std
    scale = np.maximum(30, np.quantile(yday[:origin], .8, axis=0)).astype(np.float32)
    active = np.any(yday[:origin] > 0, axis=0)
    model = HourlyResidual(scale, active).to(device)
    origins = np.unique(np.r_[np.arange(35, origin-61+1, 5), origin-61])
    data = tensors(y, z, origins, bank, device)
    assert torch.isfinite(data[-1]).all() and origins.max()+61 == origin
    weights = torch.tensor(.5**((origins.max()-origins)/120), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    start_time = time.monotonic()
    for epoch in range(epochs):
        order = rng.permutation(len(origins)); total = 0
        for start in range(0, len(origins), 4):
            ids = torch.tensor(order[start:start+4], device=device)
            h, hx, fx, hb, fb, target = [a.index_select(0, ids) for a in data]
            w = weights.index_select(0, ids); norm = target.sum((1, 2)).detach()
            optimizer.zero_grad(set_to_none=True)
            proposed = model.propose(h, hx, fx, hb, fb)
            direct = model.direct(proposed)
            step = steps[(epoch+start//4) % len(steps)]
            recursive = model.rollout(proposed, step)
            loss = (objective(recursive, target, norm, w)+.5*objective(direct, target, norm, w)+
                    5*objective(recursive, direct, norm, w))/6.5
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step(); total += float(loss.detach().cpu())
        if epoch in [0, epochs-1]:
            print('FIT', origin, steps, epoch+1, round(total/math.ceil(len(origins)/4), 6),
                  round(time.monotonic()-start_time, 1), 'seconds', flush=True)
    return model, (mean, std), {'seed': seed, 'epochs': epochs, 'steps_hours': steps,
                              'training_origins': len(origins), 'forecast_origin_day': origin,
                              'max_training_label_hour': int(origin*24-1),
                              'seconds': time.monotonic()-start_time}


def guide(frame, cutoff, name):
    if name == 'v8':
        file = ART / 'v8' / ('submission.csv' if cutoff == '2025-11-01' else 'validation_'+cutoff+'.csv')
    else:
        file = ART / 'v13_anchored' / ('submission.csv' if cutoff == '2025-11-01' else 'selected_validation_'+cutoff+'.csv')
    old = pd.read_csv(file, sep=';')
    p = frame[KEY].merge(old[KEY+['prediction']], on=KEY, how='left', validate='one_to_one').prediction
    assert p.notna().all()
    return p.to_numpy()


@torch.no_grad()
def forecasts(model, y, ex, origin, horizon, norm, bank, device, reference=None):
    model.eval(); mean, std = norm
    h, hx, fx, hb, fb, _ = tensors(y, (ex-mean)/std, [origin], bank, device, horizon)
    if reference is not None:
        ref = to_hours(reference.reshape(-1, 240)).T[None]
        fb = torch.tensor(ref, dtype=torch.float32, device=device)
    proposed = model.propose(h, hx, fx, hb, fb)
    ps = {'direct': model.direct(proposed)}
    ps.update({'h'+str(step): model.rollout(proposed, step) for step in STEPS})
    return {name: to_rows(value[0].T.cpu().numpy()) for name, value in ps.items()}


def verify_scan(device):
    torch.manual_seed(321)
    a = (.8+.19*torch.rand(2, 3, 43, 2, device=device)).requires_grad_()
    b = torch.randn(2, 3, 43, 2, device=device, requires_grad=True)
    initial = torch.randn(2, 3, 2, device=device)
    pa, pb = affine_scan(a, b); parallel = pb+pa*initial[:, :, None]
    value = initial; sequential = []
    for i in range(43):
        value = a[:, :, i]*value+b[:, :, i]; sequential.append(value)
    sequential = torch.stack(sequential, 2)
    torch.testing.assert_close(parallel, sequential, atol=3e-5, rtol=3e-5)
    ga, gb = torch.autograd.grad(parallel[:, :, -1].sum(), (a, b), retain_graph=True)
    sa, sb = torch.autograd.grad(sequential[:, :, -1].sum(), (a, b))
    torch.testing.assert_close(ga, sa, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(gb, sb, atol=3e-5, rtol=3e-5)
    return {'serial_vs_parallel_max_difference': float((parallel-sequential).abs().max().cpu()),
            'serial_vs_parallel_gradient_max_difference': float(max((ga-sa).abs().max().cpu(), (gb-sb).abs().max().cpu()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origins', default='181,212,243,273')
    parser.add_argument('--configs', default=','.join(CONFIGS))
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); device = torch.device('mps')
    assert torch.backends.mps.is_available()
    scan_audit = verify_scan(device)
    d, yday, exday, groups = dataset(); bank = ProfileBank(d)
    y = to_hours(yday); ex = to_hours(exday)
    np.testing.assert_array_equal(to_rows(y), yday.reshape(-1))
    rows = []
    for origin in map(int, args.origins.split(',')):
        days = 61 if origin == 304 else min(61, 304-origin)
        frame = d[d.time.between(origin, origin+days-1)].reset_index(drop=True)
        cutoff = str(frame.date.min())
        references = {name: guide(frame, cutoff, name) for name in ['v8', 'v13']}
        for config in args.configs.split(','):
            tag = f'{cutoff}_{config}_s{args.seed}_e{args.epochs}'
            saved = DEST / (tag+'.npz'); metadata = DEST / (tag+'.json')
            if saved.exists() and metadata.exists():
                ps = dict(np.load(saved)); audit = json.loads(metadata.read_text())
            else:
                model, norm, audit = fit(yday, y, ex, origin, bank, CONFIGS[config], args.seed, args.epochs, device)
                ps = {}
                for name, ref in [('profile', None), *references.items()]:
                    p = forecasts(model, y, ex, origin, days*24, norm, bank, device, ref)
                    ps.update({name+'_'+path: value for path, value in p.items()})
                changed = y.copy(); changed[origin*24:] = 999999
                again = forecasts(model, changed, ex, origin, days*24, norm, bank, device)
                audit['future_target_perturbation_max_difference'] = float(np.max(abs(again['h1']-ps['profile_h1'])))
                assert audit['future_target_perturbation_max_difference'] == 0
                mean, std = norm
                h, hx, fx, hb, fb, _ = tensors(y, (ex-mean)/std, [origin], bank, device, days*24)
                proposed = model.propose(h, hx, fx, hb, fb); proposed[0].retain_grad()
                final = model.rollout(proposed, 1)
                final[:, :, -1].sum().backward()
                audit['last_hour_gradient_to_first_innovation'] = float(proposed[0].grad[:, :, 0].abs().sum().detach().cpu())
                model.zero_grad(set_to_none=True)
                assert audit['last_hour_gradient_to_first_innovation'] > 0
                audit.update({'origin': cutoff, 'config': config, 'horizon_hours': days*24,
                              'scan_check': scan_audit, 'current_public_best': .89214,
                              'v13_user_reported_score': .89209, 'new_public_score': None,
                              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
                torch.save({'state_dict': model.cpu().state_dict(), 'normalization': norm,
                            'features': EXOG, 'audit': audit}, DEST / (tag+'.pt'))
                np.savez_compressed(saved, **ps); metadata.write_text(json.dumps(audit, indent=2))
                del model; torch.mps.empty_cache()
            own_paths = ['h'+str(step) for step in CONFIGS[config]]
            for name in ['profile', 'v8', 'v13']:
                ps[name+'_primary'] = np.mean([ps[name+'_direct'], *[ps[name+'_'+path] for path in own_paths]], axis=0)
            ps.update({'incumbent_v8': references['v8'], 'previous_v13': references['v13']})
            out = frame[KEY+['boardings']].copy()
            for path, value in ps.items():
                assert np.isfinite(value).all() and (value >= 0).all()
                out[path] = value
                if origin < 304:
                    for part, mask in [('all', np.ones(len(out), bool)),
                                       ('last_7_days', frame.time.to_numpy() >= origin+days-7)]:
                        rows.append(dict(origin=cutoff, model=config, path=path, part=part,
                                         score=score(frame.boardings.to_numpy()[mask], value[mask])))
            out.to_csv(DEST / (tag+'.csv.gz'), index=False)
            if origin < 304:
                pd.DataFrame(rows).to_csv(DEST / f'metrics_s{args.seed}.csv', index=False)
                print('SCORES', cutoff, config, {path: round(score(frame.boardings, ps[path]), 6)
                      for path in ['profile_primary', 'v8_primary', 'v13_primary', 'previous_v13']}, flush=True)
            else:
                print('FUTURE', config, 'complete', flush=True)
    print('DONE HOURLY', flush=True)


if __name__ == '__main__':
    main()
