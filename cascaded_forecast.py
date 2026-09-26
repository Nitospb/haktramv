"""v15: learn many short transitions AND backpropagate through 61-day rollouts.

The network can emit at most seven days per call. Daily and weekly chains share
weights and train against the full hidden horizon. One chain, one checkpoint,
no external model predictions and no prediction averaging at inference.
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

from anchored_consistency import ART, ProfileBank, batch
from unrolled_weekly import EXOG, KEY, dataset
from train_model import score

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v15_cascade'
CONFIGS = {
    'long_only': dict(short_weight=0., consistency=1.),
    'short_long_c1': dict(short_weight=1., consistency=1.),
    'short_long_c5': dict(short_weight=1., consistency=5.),
}


class Cascade(nn.Module):
    def __init__(self, scale, active):
        super().__init__()
        self.embedding = nn.Embedding(240, 8)
        # All 28 days and all routes enter the recurrent state representation.
        self.route_encoder = nn.Sequential(nn.Linear(56, 48), nn.SiLU(), nn.Linear(48, 32))
        self.query = nn.Linear(32, 16, bias=False)
        self.key = nn.Linear(32, 16, bias=False)
        self.value = nn.Linear(32, 16, bias=False)
        self.context = nn.Linear(48, 16)
        self.net = nn.Sequential(nn.Linear(len(EXOG)*2+8+4+8+16, 80), nn.SiLU(),
                                 nn.Linear(80, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('scale', torch.tensor(scale).reshape(1, 240, 1))
        self.register_buffer('active', torch.tensor(active, dtype=torch.float32).reshape(1, 240, 1))
        self.register_buffer('active_routes', torch.tensor(active).reshape(10, 24).any(1))

    def route_state(self, history, hbase):
        b = len(history)
        total = history.reshape(b, 10, 24, 28).sum(2)
        reference = hbase.reshape(b, 10, 24, 28).sum(2)
        ratio = torch.log((total+720)/(reference+720)).clamp(-3, 3)
        recent = total.mean(2, keepdim=True).clamp_min(720)
        x = torch.cat([ratio, torch.log1p(total/recent)], -1)
        encoded = self.route_encoder(x)
        logits = self.query(encoded) @ self.key(encoded).transpose(1, 2) / 4
        logits = logits.masked_fill(~self.active_routes[None, None, :], -1e4)
        neighbours = logits.softmax(-1) @ self.value(encoded)
        return torch.tanh(self.context(torch.cat([encoded, neighbours], -1)))

    def block(self, history, hx, future, hbase, fbase, offset=0):
        b, groups, length, features = future.shape
        assert 1 <= length <= 7, 'This network has no direct 61-day output'
        ix = 21+torch.arange(length, device=history.device)
        values = [history.index_select(2, ix-7*k) for k in range(4)]
        references = [hbase.index_select(2, ix-7*k) for k in range(4)]
        ratios = [torch.log((a+30)/(r+30)).clamp(-3, 3) for a, r in zip(values, references)]
        ratios += [torch.log((history[:, :, -1:]+30)/(hbase[:, :, -1:]+30)).expand(-1, -1, length).clamp(-3, 3),
                   torch.log((history.mean(2, keepdim=True)+30)/(hbase.mean(2, keepdim=True)+30)).expand(-1, -1, length).clamp(-3, 3),
                   torch.log1p(fbase/self.scale), torch.log1p(torch.stack(values).mean(0)/self.scale)]
        local = torch.arange(1, length+1, dtype=history.dtype, device=history.device)
        age = torch.stack([(local+offset)/61, local/7, torch.full_like(local, min(offset/28, 1)),
                           torch.full_like(local, length/7)], -1).reshape(1, 1, length, 4).expand(b, groups, length, 4)
        emb = self.embedding(torch.arange(groups, device=history.device))[None, :, None].expand(b, -1, length, -1)
        state = self.route_state(history, hbase)[:, :, None, None, :].expand(-1, -1, 24, length, -1).reshape(b, groups, length, 16)
        inputs = torch.cat([future, future-hx.index_select(2, ix), torch.stack(ratios, -1), age, emb, state], -1)
        correction = .7*torch.tanh(self.net(inputs).squeeze(-1)) + .35*ratios[0].clamp(-math.log(4), math.log(4))
        return ((fbase+30)*correction.exp()-30).clamp_min(0)*self.active

    def rollout(self, history, hx, future, hbase, fbase, step, track_first=False):
        assert step in [1, 7]
        predictions = []; first = None
        for offset in range(0, future.shape[2], step):
            fx = future[:, :, offset:offset+step]; fb = fbase[:, :, offset:offset+step]
            pred = self.block(history, hx, fx, hbase, fb, offset)
            if track_first and first is None:
                first = pred; first.retain_grad()
            predictions.append(pred)
            # Generated predictions stay in the graph for the entire rollout.
            history = torch.cat([history, pred], 2)[:, :, -28:]
            hx = torch.cat([hx, fx], 2)[:, :, -28:]
            hbase = torch.cat([hbase, fb], 2)[:, :, -28:]
        result = torch.cat(predictions, 2)
        return (result, first) if track_first else result


def error(prediction, target, normalizer, weights, multiscale=False):
    length = prediction.shape[-1]
    temporal = torch.ones(length, device=prediction.device)
    if length > 7:
        temporal[-7:] = 2
    per = ((prediction-target).abs()*temporal).sum((1, 2))/temporal.mean()
    if multiscale:
        daily_p = prediction.reshape(len(prediction), 10, 24, length).sum(2)
        daily_t = target.reshape(len(target), 10, 24, length).sum(2)
        daily = ((daily_p-daily_t).abs()*temporal).sum((1, 2))/temporal.mean()
        system = ((daily_p.sum(1)-daily_t.sum(1)).abs()*temporal).sum(1)/temporal.mean()
        per = (per+.15*daily+.1*system)/1.25
    return (per/normalizer.clamp_min(1)*weights).sum()/weights.sum()


def fit(y, ex, origin, bank, config, seed, epochs, device):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mean = ex[:origin].mean((0, 1), keepdims=True)
    std = ex[:origin].std((0, 1), keepdims=True).clip(.1)
    z = (ex-mean)/std
    scale = np.maximum(30, np.quantile(y[:origin], .8, axis=0)).astype(np.float32)
    active = (y[:origin] > 0).any(0)
    model = Cascade(scale, active).to(device)
    origins = np.unique(np.r_[np.arange(35, origin-61+1, 5), origin-61])
    long = batch(y, z, origins, bank, device, 61)
    assert origins.max()+61 == origin and torch.isfinite(long[-1]).all()
    weights = torch.tensor(.5**((origins.max()-origins)/120), dtype=torch.float32, device=device)
    short = {}; probabilities = {}; short_origins = {}
    if config['short_weight']:
        for horizon in [1, 7]:
            oo = np.arange(35, origin-horizon+1)
            short_origins[horizon] = oo
            short[horizon] = batch(y, z, oo, bank, device, horizon)
            assert oo.max()+horizon == origin and torch.isfinite(short[horizon][-1]).all()
            probabilities[horizon] = .5**((oo.max()-oo)/90)
            probabilities[horizon] /= probabilities[horizon].sum()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    started = time.monotonic()
    for epoch in range(epochs):
        model.train(); order = rng.permutation(len(origins)); total = 0
        for start in range(0, len(origins), 4):
            ids = torch.tensor(order[start:start+4], device=device)
            h, hx, fx, hb, fb, target = [a.index_select(0, ids) for a in long]
            w = weights.index_select(0, ids); norm = target.sum((1, 2)).detach()
            optimizer.zero_grad(set_to_none=True)
            daily = model.rollout(h, hx, fx, hb, fb, 1)
            weekly = model.rollout(h, hx, fx, hb, fb, 7)
            ll = .5*(error(daily, target, norm, w, True)+error(weekly, target, norm, w, True))
            consistency = error(daily, weekly, norm, w)
            loss = (ll+config['consistency']*consistency)/(1+config['consistency'])
            if config['short_weight']:
                short_loss = 0
                for horizon in [1, 7]:
                    jj = rng.choice(len(short_origins[horizon]), size=16, replace=True, p=probabilities[horizon])
                    j = torch.tensor(jj, device=device)
                    sh, sx, sf, sb, sfb, st = [a.index_select(0, j) for a in short[horizon]]
                    sp = model.block(sh, sx, sf, sb, sfb)
                    short_loss = short_loss + .5*error(sp, st, st.sum((1, 2)).detach(), torch.ones(len(j), device=device), True)
                loss = (loss+config['short_weight']*short_loss)/(1+config['short_weight'])
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step(); total += float(loss.detach().cpu())
        if epoch == 0 or (epoch+1) % 20 == 0:
            print('FIT', origin, config, epoch+1, round(total/math.ceil(len(origins)/4), 6),
                  round(time.monotonic()-started, 1), 'seconds', flush=True)
    return model, (mean, std), {'forecast_origin_day': origin, 'max_training_label_day': origin-1,
                               'long_training_origins': len(origins),
                               'short_training_origins': {str(h): len(o) for h, o in short_origins.items()},
                               'short_latest_origin': {str(h): int(o.max()) for h, o in short_origins.items()},
                               'seed': seed, 'epochs': epochs, 'config': config,
                               'seconds': time.monotonic()-started}


@torch.no_grad()
def predict(model, y, ex, origin, horizon, norm, bank, device):
    model.eval(); mean, std = norm
    h, hx, fx, hb, fb, _ = batch(y, (ex-mean)/std, [origin], bank, device, horizon)
    return {str(step): model.rollout(h, hx, fx, hb, fb, step)[0].T.cpu().numpy().reshape(-1)
            for step in [1, 7]}


def audits(model, y, ex, origin, horizon, norm, bank, device):
    model.eval(); mean, std = norm
    h, hx, fx, hb, fb, _ = batch(y, (ex-mean)/std, [origin], bank, device, horizon)
    result = {}
    for step in [1, 7]:
        pred, first = model.rollout(h, hx, fx, hb, fb, step, True)
        pred[:, :, -1].sum().backward()
        value = float(first.grad.abs().sum().detach().cpu())
        assert value > 0
        result['last_day_gradient_to_first_step_'+str(step)] = value
        model.zero_grad(set_to_none=True)
    h = h.clone().requires_grad_(True)
    pred = model.block(h, hx, fx[:, :, :1], hb, fb[:, :, :1])
    # Route 1 predictions must really depend on other routes' histories.
    grad = torch.autograd.grad(pred[:, :24].sum(), h)[0]
    other = float(grad[:, 24:].abs().sum().detach().cpu())
    assert other > 0
    result['cross_route_gradient_l1'] = other
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origins', default='181,212')
    parser.add_argument('--configs', default=','.join(CONFIGS))
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); device = torch.device('mps')
    assert torch.backends.mps.is_available()
    d, y, ex, groups = dataset(); bank = ProfileBank(d); rows = []
    for origin in map(int, args.origins.split(',')):
        horizon = 61 if origin == 304 else min(61, 304-origin)
        frame = d[d.time.between(origin, origin+horizon-1)].reset_index(drop=True)
        cutoff = str(frame.date.min())
        for name in args.configs.split(','):
            tag = f'{cutoff}_{name}_s{args.seed}_e{args.epochs}'
            saved = DEST/(tag+'.npz'); metadata = DEST/(tag+'.json')
            if saved.exists() and metadata.exists():
                ps = dict(np.load(saved)); audit = json.loads(metadata.read_text())
            else:
                model, norm, audit = fit(y, ex, origin, bank, CONFIGS[name], args.seed, args.epochs, device)
                ps = predict(model, y, ex, origin, horizon, norm, bank, device)
                changed = y.copy(); changed[origin:] = 999999
                again = predict(model, changed, ex, origin, horizon, norm, bank, device)
                audit['future_label_perturbation_max_difference'] = max(float(np.max(abs(ps[k]-again[k]))) for k in ps)
                assert audit['future_label_perturbation_max_difference'] == 0
                audit.update(audits(model, y, ex, origin, horizon, norm, bank, device))
                audit.update({'name': name, 'origin': cutoff, 'horizon_days': horizon,
                              'no_direct_61_day_output': True, 'no_external_forecast_inputs': True,
                              'maximum_output_days_per_call': 7,
                              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
                torch.save({'state_dict': model.cpu().state_dict(), 'normalization': norm, 'features': EXOG,
                            'group_order': groups.to_dict('records'), 'audit': audit}, DEST/(tag+'.pt'))
                np.savez_compressed(saved, **ps); metadata.write_text(json.dumps(audit, indent=2))
                del model; torch.mps.empty_cache()
            out = frame[KEY+['boardings']].copy()
            for path, p in ps.items():
                assert np.isfinite(p).all() and (p >= 0).all()
                out['step'+path] = p
                if origin < 304:
                    days = frame.time.to_numpy()-origin
                    for part, mask in [('all', days >= 0), ('last_7_days', days >= horizon-7)]:
                        rows.append(dict(origin=cutoff, model=name, path='step'+path, part=part,
                                         score=score(frame.boardings.to_numpy()[mask], p[mask]), seed=args.seed))
            out.to_csv(DEST/(tag+'.csv.gz'), index=False)
            if origin < 304:
                records = pd.DataFrame(rows); mp = DEST/f'metrics_s{args.seed}_e{args.epochs}.csv'
                if mp.exists():
                    records = pd.concat([pd.read_csv(mp), records]).drop_duplicates(['origin','model','path','part','seed'], keep='last')
                records.to_csv(mp, index=False)
                print('SCORES', cutoff, name, {path: round(score(frame.boardings, p), 6) for path, p in ps.items()}, flush=True)
            else:
                print('FUTURE', name, 'complete', flush=True)
    print('DONE CASCADE', flush=True)


if __name__ == '__main__':
    main()
