"""Short, overlapping rollouts of the incumbent v8 forecast engine.

Run on Studio. All 61 future days are hidden together. Each path appends only
its own predictions; the synchronized variant appends the streams' consensus.
CatBoost weights are fitted on real pre-origin labels and remain fixed. The
full engine recomputes historical profiles, CatBoost scales, service experts,
and monthly reconciliation after each block. No fitting on pseudo-labels.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from feature_models import base, features
from research_v2 import adaptive
from service_models import adjust, normal_profile
from train_model import score

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'outputs'
if not (ART / 'v4/training_matrix.csv').exists():
    ART = ART / 'studio'
DEST = ROOT / 'outputs/v12_multistream'
KEY = ['route', 'date', 'hour']
FOLDS = [('2025-07-01', '2025-08-30'), ('2025-08-01', '2025-09-30'),
         ('2025-09-01', '2025-10-31'), ('2025-10-01', '2025-10-31')]
CAT_NAMES = ['cat_full_6_1_MAE', 'cat_full_8_1_MAE', 'cat_full_8_1_RMSE']
PROFILE_COLS = ['route', 'hour', 'daytype', 'summer', 'holiday', 'time', 'boardings']


def reconcile(v, b, p):
    x = v[['route', 'month', 'service_cancelled']].copy()
    x['b'] = b; x['p'] = p
    total = x.groupby(['route', 'month'])[['b', 'p']].transform('sum')
    shape = np.asarray(p) * (total.b / (total.p + 1e-9)).to_numpy()
    empty = total.p.to_numpy() < 1e-6
    shape[empty] = np.asarray(b)[empty]
    return np.maximum(0, (.25 * np.asarray(b) + .75 * shape) *
                      (1 - .975 * v.service_cancelled.to_numpy()))


def keyed_prediction(path, v):
    x = pd.read_csv(path, sep=';')
    p = v[KEY].merge(x[KEY + ['prediction']], on=KEY, how='left', validate='one_to_one')
    assert p.prediction.notna().all(), path
    return p.prediction.to_numpy()


class Engine:
    def __init__(self, tr, future, origin, mode, threads=4):
        assert 'boardings' not in future
        assert tr.date.max() < future.date.min()
        self.mode = mode
        self.origin = origin
        self.future = future.copy()
        self.threads = threads
        self.audit = {'mode': mode, 'origin': origin, 'real_labels_through': tr.date.max(),
                      'future_label_column_removed': True, 'catboost_weights_fixed_during_rollout': True}
        self.weights = json.loads((ART / 'v4/ensemble_report.json').read_text())['weights']
        tag = 'submission.csv' if origin == '2025-11-01' else 'validation_' + origin + '.csv'
        self.expert = keyed_prediction(ART / 'v5_experts' / tag, future)
        self.incumbent = keyed_prediction(ART / 'v8' / tag, future)
        # v8 was originally reconciled over full July-August (62 days). Keep
        # that forecast-only normalization in the control, then evaluate 61.
        self.original_history_end = tr.date.max()
        self.models = {}
        if mode == 'full':
            for name in CAT_NAMES:
                self.models[name] = self.load_or_fit(name, tr)
        b = adaptive(tr[PROFILE_COLS], future, 28, .5)
        p = reconcile(future, b, self.expert)
        self.audit['control_crop_difference_max'] = float(np.max(abs(p - self.incumbent)))
        self.direct = self.predict(tr, future, 28)
        self.audit['direct_vs_saved_max'] = float(np.max(abs(self.direct - self.incumbent)))

    def load_or_fit(self, name, tr):
        path = DEST / 'models' / self.origin / (name + '.cbm')
        path.parent.mkdir(parents=True, exist_ok=True)
        model = CatBoostRegressor()
        if path.exists():
            model.load_model(str(path))
            return model
        saved = ART / 'v12_multistream/models' / self.origin / (name + '.cbm')
        if saved.exists():
            model.load_model(str(saved))
            return model
        if self.origin == '2025-11-01' and (ART / 'v4' / (name + '.cbm')).exists():
            model.load_model(str(ART / 'v4' / (name + '.cbm')))
            return model
        _, group, depth, power, loss = name.split('_')
        x = tr.copy()
        x['scale'] = np.maximum(30, base(tr, tr, float(power)))
        cols = features(x, group)
        if 'scale' not in cols:
            cols.append('scale')
        model = CatBoostRegressor(iterations=1600, depth=int(depth), learning_rate=.04,
                                 loss_function=loss, l2_leaf_reg=15, thread_count=self.threads,
                                 random_seed=2026, verbose=False, allow_writing_files=False)
        weights = x.scale.to_numpy() * .5 ** ((x.time.max() - x.time.to_numpy()) / 240)
        started = time.monotonic()
        model.fit(x[cols], x.boardings / x.scale, cat_features=['route', 'routehour'],
                  sample_weight=weights)
        model.save_model(str(path))
        path.with_suffix('.features.json').write_text(json.dumps(cols))
        print('FIT', self.origin, name, round(time.monotonic() - started, 1), 'seconds', flush=True)
        return model

    def predict(self, history, v, window=28):
        assert 'boardings' not in v, 'A forecast must never receive hidden targets'
        assert history.date.max() < v.date.min()
        b = adaptive(history[PROFILE_COLS], v, window, .5)
        if self.mode == 'cached_shape':
            p = pd.Series(self.expert, index=self.future.index).loc[v.index].to_numpy()
        else:
            x = v.copy()
            x['scale'] = np.maximum(30, base(history, v, 1))
            p = self.weights['v2'] * b
            for name, model in self.models.items():
                q = np.maximum(0, model.predict(x[model.feature_names_],
                                               thread_count=self.threads) * x.scale.to_numpy())
                q[v.route.to_numpy() == 5] = 0
                p += self.weights[name] * q
            p += self.weights['base_1_112_0.5'] * base(history, v, 1, 112, .5)
            p *= 1 - .975 * v.service_cancelled.to_numpy()
            changed = np.maximum(v.service_rerouted.to_numpy(), v.service_reduced_frequency.to_numpy())
            if np.any(changed):
                tnormal = normal_profile(history, history, .5, 365)
                vnormal = normal_profile(history, v.reset_index(drop=True), .5, 365)
                expert = adjust(history, v, tnormal, vnormal)
                p = p * (1 - changed) + expert * changed
        return reconcile(v, b, p)


def stream_configs():
    configs = []
    for step in [1, 3, 7, 14, 28]:
        for first in sorted({1, (step + 1) // 2, step}):
            configs.append(dict(step=step, first=first, window=28, scope='remaining'))
    for window in [14, 56]:
        for step in [1, 7]:
            configs.append(dict(step=step, first=step, window=window, scope='remaining'))
    # Reconciliation over a short block is a distinct hypothesis: test it
    # explicitly rather than silently replacing monthly normalization.
    for step in [1, 3, 7, 14, 28]:
        configs.append(dict(step=step, first=step, window=28, scope='block'))
    return configs


def config_name(c):
    return 's{step}_f{first}_w{window}_{scope}'.format(**c)


def rollout(engine, tr, v, step, first, window, scope):
    days = sorted(v.date.unique())
    h = tr.copy()
    result = np.full(len(v), np.nan)
    pos = 0
    while pos < len(days):
        end = min(len(days), pos + (first if pos == 0 else step))
        block = v.date.isin(days[pos:end])
        remaining = v.date >= days[pos]
        query = v.loc[remaining if scope == 'remaining' else block]
        forecast = engine.predict(h, query, window)
        aligned = pd.Series(forecast, index=query.index)
        pred = aligned.loc[v.index[block]].to_numpy()
        result[np.flatnonzero(block)] = pred
        pseudo = v.loc[block].copy(); pseudo['boardings'] = pred
        h = pd.concat([h, pseudo], ignore_index=True)
        pos = end
    assert np.isfinite(result).all() and np.min(result) >= 0
    return result


def synchronized(engine, tr, v, reducer='mean', anchor=0):
    """Different refresh cadences share a consensus history, never true future y."""
    days = sorted(v.date.unique())
    h = tr.copy()
    forecasts = {}
    steps = [1, 3, 7, 14, 28]
    result = np.full(len(v), np.nan)
    for pos, day in enumerate(days):
        remaining = v.date >= day
        current = v.date == day
        for step in steps:
            if pos % step == 0:
                q = v.loc[remaining]
                forecasts[step] = pd.Series(engine.predict(h, q, 28), index=q.index)
        proposals = np.stack([forecasts[s].loc[v.index[current]].to_numpy() for s in steps])
        pred = np.mean(proposals, axis=0) if reducer == 'mean' else np.median(proposals, axis=0)
        pred = (1 - anchor) * pred + anchor * engine.incumbent[np.flatnonzero(current)]
        result[np.flatnonzero(current)] = pred
        pseudo = v.loc[current].copy(); pseudo['boardings'] = pred
        h = pd.concat([h, pseudo], ignore_index=True)
    assert np.isfinite(result).all()
    return result


def add_ensembles(predictions, configs):
    canonical = [config_name(c) for c in configs if c['window'] == 28 and c['scope'] == 'remaining']
    all_remaining = [config_name(c) for c in configs if c['scope'] == 'remaining']
    canonical_pred = np.stack([predictions[n] for n in canonical])
    predictions['streams_mean'] = canonical_pred.mean(axis=0)
    predictions['streams_median'] = np.median(canonical_pred, axis=0)
    # Equal vote per step prevents steps with more phase offsets from receiving
    # more total weight just because they have more paths.
    per_step = [np.mean([predictions[config_name(c)] for c in configs
                        if c['step'] == step and c['window'] == 28 and c['scope'] == 'remaining'], axis=0)
                for step in [1, 3, 7, 14, 28]]
    predictions['streams_balanced'] = np.mean(per_step, axis=0)
    predictions['streams_windows_median'] = np.median([predictions[n] for n in all_remaining], axis=0)
    for name in ['streams_balanced', 'streams_median']:
        predictions[name + '_anchor50'] = .5 * predictions[name] + .5 * predictions['incumbent_v8']
    block = np.stack([predictions[config_name(c)] for c in configs if c['scope'] == 'block'])
    predictions['streams_blocks_mean'] = block.mean(axis=0)


def metrics(origin, v, predictions):
    target = v.boardings.to_numpy()
    day = (pd.to_datetime(v.date) - pd.Timestamp(origin)).dt.days.to_numpy() + 1
    records = []
    for name, p in predictions.items():
        for part, mask in [('all', day > 0), ('days_1_7', day <= 7),
                           ('days_8_28', (day >= 8) & (day <= 28)),
                           ('days_29_61', day >= 29), ('last_7_days', day > day.max() - 7)]:
            if mask.any():
                records.append(dict(origin=origin, model=name, part=part, days=int(day.max()),
                                    score=score(target[mask], p[mask]),
                                    volume_ratio=float(p[mask].sum() / target[mask].sum())))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['cached_shape', 'full'], default='cached_shape')
    parser.add_argument('--origin', choices=[s for s, _ in FOLDS] + ['2025-11-01'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--skip-sync', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    d = pd.read_csv(ART / 'v4/training_matrix.csv')
    configs = stream_configs()
    folds = FOLDS if not args.origin else [(args.origin, dict(FOLDS + [('2025-11-01', '2025-12-31')])[args.origin])]
    for origin, end in folds:
        started = time.monotonic()
        tr = d[d.date < origin].reset_index(drop=True)
        validation = d[d.date.between(origin, end)].reset_index(drop=True)
        v = validation.drop(columns='boardings')
        assert not tr.boardings.isna().any()
        engine = Engine(tr, v, origin, args.engine, args.threads)
        prefix = DEST / (args.engine + '_' + origin)
        cache = prefix.with_suffix('.npz')
        predictions = dict(np.load(cache)) if cache.exists() else {}
        predictions['incumbent_v8'] = engine.incumbent
        predictions['direct_engine'] = engine.direct
        for config in configs:
            name = config_name(config)
            if name not in predictions:
                predictions[name] = rollout(engine, tr, v, **config)
                np.savez_compressed(cache, **predictions)
            print(args.engine, origin, name,
                  round(score(validation.boardings, predictions[name]), 6) if origin != '2025-11-01' else 'forecast', flush=True)
        add_ensembles(predictions, configs)
        if not args.skip_sync:
            for reducer, anchor in [('mean', 0), ('median', 0), ('mean', .5)]:
                name = 'synchronized_' + reducer + '_anchor' + str(int(anchor * 100))
                if name not in predictions:
                    predictions[name] = synchronized(engine, tr, v, reducer, anchor)
                    np.savez_compressed(cache, **predictions)
                print(args.engine, origin, name,
                      round(score(validation.boardings, predictions[name]), 6) if origin != '2025-11-01' else 'forecast', flush=True)
        np.savez_compressed(cache, **predictions)
        out = validation[KEY + ['boardings']].copy()
        for name, pred in predictions.items():
            out[name] = pred
        out.to_csv(prefix.with_suffix('.csv.gz'), index=False)
        if origin != '2025-11-01':
            pd.DataFrame(metrics(origin, validation, predictions)).to_csv(str(prefix) + '_metrics.csv', index=False)
        engine.audit.update({'streams': configs, 'streams_count': len(configs),
                             'days': len(v.date.unique()), 'runtime_seconds': time.monotonic() - started,
                             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'evaluation': 'Overlapping development windows, not independent validation. '
                             'Retrospective supplied weather/service calendars; no future boarding labels.',
                             'public_score': None})
        prefix.with_suffix('.json').write_text(json.dumps(engine.audit, indent=2))
        print('DONE', args.engine, origin, round(time.monotonic() - started, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
