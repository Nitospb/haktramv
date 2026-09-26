"""v13: unrolled multi-step corrections to a fixed, origin-safe demand profile.

Profiles for each training example are constructed from its past only. A fixed
profile anchors the whole hidden horizon, while generated history still passes
gradients between blocks. Direct and short-step paths share the same network.
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

from research_v2 import adaptive
from train_model import score
from unrolled_weekly import EXOG, KEY, dataset

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v13_anchored'
ART = ROOT / 'outputs'
if not (ART / 'v8/submission.csv').exists():
    ART = ART / 'studio'
CONFIGS = {
    'direct': dict(gate=0, consistency=0, steps=[]),
    'weekly_c1': dict(gate=.35, consistency=1, steps=[7]),
    'weekly_c5': dict(gate=.35, consistency=5, steps=[7]),
    'multi_c1': dict(gate=.35, consistency=1, steps=[1, 3, 7]),
    'multi_c5': dict(gate=.35, consistency=5, steps=[1, 3, 7]),
}
PROFILE_COLS = ['route', 'hour', 'daytype', 'summer', 'holiday', 'time', 'boardings']


class ProfileBank:
    def __init__(self, d):
        self.d = d
        self.cache = {}

    def get(self, origin):
        if origin not in self.cache:
            history = self.d[self.d.time < origin]
            query = self.d[self.d.time.between(origin-28, origin+60)].drop(columns='boardings')
            p = adaptive(history[PROFILE_COLS], query, 28, .5)
            p *= 1 - .975 * query.service_cancelled.to_numpy()
            p = np.maximum(0, p).reshape(-1, 240).astype(np.float32)
            assert p.shape == (89, 240) and np.isfinite(p).all()
            self.cache[origin] = p
        return self.cache[origin]


class AnchoredResidual(nn.Module):
    def __init__(self, scale, active, gate):
        super().__init__()
        self.gate = gate
        self.embedding = nn.Embedding(240, 8)
        self.net = nn.Sequential(nn.Linear(len(EXOG)*2+8+4+8, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('scale', torch.tensor(scale).reshape(1, -1, 1))
        self.register_buffer('active', torch.tensor(active, dtype=torch.float32).reshape(1, -1, 1))

    def block(self, history, hx, future, hbase, fbase, offset=0, direct=False):
        b, groups, n, features = future.shape
        ix = 21 + torch.arange(n, device=history.device) % 7
        lags = [history.index_select(2, ix-7*k) for k in range(4)]
        bases = [hbase.index_select(2, ix-7*k) for k in range(4)]
        ratios = [torch.log((value+30)/(reference+30)).clamp(-3, 3)
                  for value, reference in zip(lags, bases)]
        ratios += [torch.log((history[:, :, -1:]+30)/(hbase[:, :, -1:]+30)).expand(-1, -1, n).clamp(-3, 3),
                   torch.log((history.mean(2, keepdim=True)+30)/(hbase.mean(2, keepdim=True)+30)).expand(-1, -1, n).clamp(-3, 3),
                   torch.log1p(fbase/self.scale), torch.log1p(torch.stack(lags).mean(0)/self.scale)]
        anchor_exog = hx.index_select(2, ix)
        local = torch.arange(1, n+1, device=history.device, dtype=history.dtype)
        age = torch.stack([(local+offset)/61, local/7,
                           torch.full_like(local, min(offset/28, 1)),
                           torch.full_like(local, float(direct))], -1)
        age = age.reshape(1, 1, n, 4).expand(b, groups, n, 4)
        emb = self.embedding(torch.arange(groups, device=history.device)).reshape(1, groups, 1, 8).expand(b, groups, n, 8)
        x = torch.cat([future, future-anchor_exog, torch.stack(ratios, -1), age, emb], -1)
        residual = .7 * torch.tanh(self.net(x).squeeze(-1))
        feedback = 0 if direct else self.gate * ratios[0].clamp(-math.log(4), math.log(4))
        p = ((fbase+30)*torch.exp(residual+feedback)-30).clamp_min(0)
        return p * self.active

    def rollout(self, history, hx, future, hbase, fbase, step=7):
        result = []
        for offset in range(0, future.shape[2], step):
            fx = future[:, :, offset:offset+step]
            fb = fbase[:, :, offset:offset+step]
            p = self.block(history, hx, fx, hbase, fb, offset=offset)
            result.append(p)
            # No detach: late errors train all earlier generated blocks.
            history = torch.cat([history, p], 2)[:, :, -28:]
            hx = torch.cat([hx, fx], 2)[:, :, -28:]
            hbase = torch.cat([hbase, fb], 2)[:, :, -28:]
        return torch.cat(result, 2)


def batch(y, z, origins, bank, device, horizon=61):
    data = [np.stack([y[o-28:o].T for o in origins]),
            np.stack([z[o-28:o].transpose(1, 0, 2) for o in origins]),
            np.stack([z[o:o+horizon].transpose(1, 0, 2) for o in origins]),
            np.stack([bank.get(o)[:28].T for o in origins]),
            np.stack([bank.get(o)[28:28+horizon].T for o in origins]),
            np.stack([y[o:o+horizon].T for o in origins])]
    return [torch.tensor(a, device=device) for a in data]


def loss(pred, target, normalizer, weights):
    # Each complete training origin receives equal normalized weight. Recent
    # windows are weighted more, but each label remains before the real cutoff.
    daily = torch.ones(pred.shape[-1], device=pred.device)
    daily[-7:] = 2
    per_origin = ((pred-target).abs()*daily).sum((1, 2)) / normalizer.clamp_min(1) / daily.mean()
    return (per_origin*weights).sum()/weights.sum()


def fit(y, ex, origin, bank, config, seed, epochs, device):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mean = ex[:origin].mean((0, 1), keepdims=True)
    std = ex[:origin].std((0, 1), keepdims=True).clip(.1)
    z = (ex-mean)/std
    scale = np.maximum(30, np.quantile(y[:origin], .8, axis=0)).astype(np.float32)
    active = np.any(y[:origin] > 0, axis=0)
    model = AnchoredResidual(scale, active, config['gate']).to(device)
    # Include the latest admissible training window, even if off the 5-day grid.
    origins = np.unique(np.r_[np.arange(35, origin-61+1, 5), origin-61])
    assert origins.min() >= 28 and origins.max()+61 == origin
    tensors = batch(y, z, origins, bank, device)
    assert torch.isfinite(tensors[-1]).all()
    weights = torch.tensor(.5**((origins.max()-origins)/120), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    started = time.monotonic()
    for epoch in range(epochs):
        order = rng.permutation(len(origins)); total = 0
        for start in range(0, len(origins), 4):
            ids = torch.tensor(order[start:start+4], device=device)
            h, hx, fx, hb, fb, target = [a.index_select(0, ids) for a in tensors]
            w = weights.index_select(0, ids); norm = target.sum((1, 2)).detach()
            optimizer.zero_grad(set_to_none=True)
            direct = model.block(h, hx, fx, hb, fb, direct=True)
            if not config['steps']:
                objective = loss(direct, target, norm, w)
            else:
                step = config['steps'][(epoch+start//4) % len(config['steps'])]
                recursive = model.rollout(h, hx, fx, hb, fb, step)
                objective = (loss(recursive, target, norm, w) + .5*loss(direct, target, norm, w) +
                             config['consistency']*loss(recursive, direct, norm, w))/(1.5+config['consistency'])
            assert torch.isfinite(objective)
            objective.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step(); total += float(objective.detach().cpu())
        if epoch in [0, epochs-1]:
            print('FIT', origin, config, epoch+1, round(total/math.ceil(len(origins)/4), 5), flush=True)
    return model, (mean, std), {'training_origins': len(origins), 'max_training_label_day': int(origins.max()+60),
                              'forecast_origin_day': origin, 'seconds': time.monotonic()-started,
                              'seed': seed, 'epochs': epochs, 'config': config}


def incumbent(frame, cutoff):
    name = 'submission.csv' if cutoff == '2025-11-01' else 'validation_'+cutoff+'.csv'
    source = pd.read_csv(ART / 'v8' / name, sep=';')
    p = frame[KEY].merge(source[KEY+['prediction']], on=KEY, validate='one_to_one', how='left').prediction
    assert p.notna().all()
    return p.to_numpy()


@torch.no_grad()
def forecasts(model, y, ex, origin, horizon, normalization, bank, device, guide=None):
    model.eval(); mean, std = normalization
    h, hx, fx, hb, fb, _ = batch(y, (ex-mean)/std, [origin], bank, device, horizon)
    if guide is not None:
        fb = torch.tensor(guide.reshape(horizon, 240).T[None], dtype=torch.float32, device=device)
    ps = {'direct': model.block(h, hx, fx, hb, fb, direct=True)}
    ps.update({'step'+str(step): model.rollout(h, hx, fx, hb, fb, step) for step in [1, 3, 7]})
    ps = {k: p[0].T.cpu().numpy().reshape(-1) for k, p in ps.items()}
    ps['all_paths_mean'] = np.mean(list(ps.values()), axis=0)
    ps['weekly_direct_mean'] = .5*ps['step7'] + .5*ps['direct']
    return ps


def gradient_check(model, y, ex, origin, norm, bank, device):
    model.eval(); mean, std = norm
    h, hx, fx, hb, fb, _ = batch(y, (ex-mean)/std, [origin], bank, device)
    first = model.block(h, hx, fx[:, :, :7], hb, fb[:, :, :7]); first.retain_grad()
    h = torch.cat([h, first], 2)[:, :, -28:]
    hx = torch.cat([hx, fx[:, :, :7]], 2)[:, :, -28:]
    hb = torch.cat([hb, fb[:, :, :7]], 2)[:, :, -28:]
    for offset in range(7, 61, 7):
        xx = fx[:, :, offset:offset+7]; bb = fb[:, :, offset:offset+7]
        p = model.block(h, hx, xx, hb, bb, offset=offset)
        h = torch.cat([h, p], 2)[:, :, -28:]
        hx = torch.cat([hx, xx], 2)[:, :, -28:]
        hb = torch.cat([hb, bb], 2)[:, :, -28:]
    p.sum().backward()
    result = float(first.grad.abs().sum().detach().cpu())
    model.zero_grad(set_to_none=True)
    return result


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
    d, y, ex, groups = dataset(); bank = ProfileBank(d)
    rows = []
    for origin in map(int, args.origins.split(',')):
        horizon = 61 if origin == 304 else min(61, 304-origin)
        frame = d[d.time.between(origin, origin+horizon-1)].reset_index(drop=True)
        cutoff = str(frame.date.min()); reference = incumbent(frame, cutoff)
        for name in args.configs.split(','):
            tag = f'{cutoff}_{name}_s{args.seed}_e{args.epochs}'
            saved = DEST / (tag+'.npz'); metadata = DEST / (tag+'.json')
            if saved.exists() and metadata.exists():
                ps = dict(np.load(saved)); audit = json.loads(metadata.read_text())
            else:
                model, norm, audit = fit(y, ex, origin, bank, CONFIGS[name], args.seed, args.epochs, device)
                ps = {}
                for guide_name, guide in [('profile', None), ('v8', reference)]:
                    p = forecasts(model, y, ex, origin, horizon, norm, bank, device, guide)
                    ps.update({guide_name+'_'+key: value for key, value in p.items()})
                changed = y.copy(); changed[origin:] = 999999
                again = forecasts(model, changed, ex, origin, horizon, norm, bank, device)
                audit['target_perturbation_max_difference'] = float(np.max(abs(again['step7']-ps['profile_step7'])))
                assert audit['target_perturbation_max_difference'] == 0
                if horizon == 61 and CONFIGS[name]['steps']:
                    audit['last_to_first_block_gradient_norm'] = gradient_check(model, y, ex, origin, norm, bank, device)
                    assert audit['last_to_first_block_gradient_norm'] > 0
                audit.update({'name': name, 'origin': cutoff, 'horizon': horizon,
                              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              'future_guides': ['profile fitted on pre-origin history only', 'saved incumbent v8 forecast'],
                              'current_public_incumbent_score': .89214, 'new_public_score': None})
                torch.save({'state_dict': model.cpu().state_dict(), 'normalization': norm,
                            'features': EXOG, 'group_order': groups.to_dict('records'), 'audit': audit}, DEST / (tag+'.pt'))
                np.savez_compressed(saved, **ps); metadata.write_text(json.dumps(audit, indent=2))
                del model; torch.mps.empty_cache()
            ps['incumbent_v8'] = reference
            ps['profile_base'] = bank.get(origin)[28:28+horizon].reshape(-1)
            out = frame[KEY+['boardings']].copy()
            for path, p in ps.items():
                assert np.isfinite(p).all() and (p >= 0).all()
                out[path] = p
                if origin < 304:
                    days = frame.time.to_numpy()-origin+1
                    for part, mask in [('all', days > 0), ('last_7_days', days > horizon-7)]:
                        rows.append(dict(origin=cutoff, model=name, path=path, part=part,
                                         score=score(frame.boardings.to_numpy()[mask], p[mask]), seed=args.seed))
            out.to_csv(DEST / (tag+'.csv.gz'), index=False)
            if origin < 304:
                print('SCORES', cutoff, name, {path: round(score(frame.boardings, ps[path]), 5) for path in
                      ['profile_base', 'profile_step7', 'profile_all_paths_mean', 'v8_all_paths_mean', 'incumbent_v8']}, flush=True)
                pd.DataFrame(rows).to_csv(DEST / ('metrics_s'+str(args.seed)+'.csv'), index=False)
            else:
                print('FUTURE', cutoff, name, 'complete', flush=True)
    print('DONE ANCHORED', flush=True)


if __name__ == '__main__':
    main()
