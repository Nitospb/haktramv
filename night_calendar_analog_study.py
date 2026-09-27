"""One bounded follow-up: avoid anchoring autumn forecasts to summer volume.

Calendar-matched historical profiles, with and without a soft weather match.
All target-derived features use history strictly before each pseudo-origin.
Two standalone MAE learners compare absolute and normalized-residual targets.
"""
import argparse
import fcntl
import json
import os
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from night_forecast_common import Data, OUT, SELECTION, CONTROL, KEY, save_json, status
from night_catboost_study import Deadline


class CalendarData(Data):
    def __init__(self):
        super().__init__()
        self.analog_cache = {}
        self.conditions = {}
        for name in ['is_summer_school_break', 'is_school_holiday', 'holiday', 'temp_mean']:
            self.conditions[name] = self.d[name].to_numpy().reshape(365, 10, 24).mean(2)

    def analog(self, origin, horizon):
        key = (origin, horizon)
        if key in self.analog_cache:
            return self.analog_cache[key]
        historical = self.daytype[:origin]
        future = self.daytype[origin:origin+horizon]
        exact = future[:, None] == historical[None, :]
        sameclass = (future[:, None] < 5) == (historical[None, :] < 5)
        weights = np.where(exact, 1., np.where(sameclass, .1, 0.))[:, :, None]
        weights = np.broadcast_to(weights, (horizon, origin, 10)).copy()
        weights *= np.exp2(-(origin-1-np.arange(origin))/112)[None, :, None]
        for name, mismatch in [('is_summer_school_break', .1), ('is_school_holiday', .4), ('holiday', .1)]:
            values = self.conditions[name]
            match = values[origin:origin+horizon, None, :] == values[None, :origin, :]
            weights *= np.where(match, 1., mismatch)
        valid = np.isfinite(self.y[:origin]) & (self.cancel[:origin] < .5)
        y = np.nan_to_num(self.y[:origin])*valid
        prior = self.profile(origin, 112)[future]
        result = {}
        for name in ['term', 'weather']:
            w = weights.copy()
            if name == 'weather':
                temp = self.conditions['temp_mean']
                distance = temp[origin:origin+horizon, None, :] - temp[None, :origin, :]
                w *= .3 + .7*np.exp(-.5*(distance/8)**2)
            denominator = np.einsum('fdr,drh->frh', w, valid.astype(float))
            numerator = np.einsum('fdr,drh->frh', w, y)
            # Two pseudo-observations prevent isolated weather/calendar analogs
            # from determining a route-hour profile without historical support.
            result[name] = ((numerator+2*prior)/(denominator+2)).reshape(-1)
            result[name+'_support'] = denominator.reshape(-1)
        self.analog_cache[key] = result
        return result

    def tabular(self, origin, horizon=61):
        x, frame = super().tabular(origin, horizon)
        for name, values in self.analog(origin, horizon).items():
            x['analog_'+name] = values
        return x, frame

    def causal_check(self):
        before, _ = self.tabular(151, 61)
        saved = self.y.copy()
        try:
            self.y[151:] = 999983
            self.cache.clear(); self.analog_cache.clear()
            after, _ = self.tabular(151, 61)
            pd.testing.assert_frame_equal(before, after)
        finally:
            self.y = saved
            self.cache.clear(); self.analog_cache.clear()


def fit(data, folder, origin, family, trees, end):
    x, meta, weights = data.training_pairs(origin)
    xx, _ = data.tabular(origin)
    residual = family == 'residual'
    y = meta.boardings.to_numpy()
    scale = np.maximum(30, x.analog_term.to_numpy())
    if residual:
        y = (y-x.analog_term.to_numpy())/scale
        weights *= scale
    good = (meta.route.to_numpy() != 5) & (meta.service_cancelled.to_numpy() < .5) & np.isfinite(y)
    path = folder/(family+'_'+str(origin)+'.cbm')
    m = CatBoostRegressor()
    if path.exists():
        m.load_model(str(path))
    if not path.exists() or m.tree_count_ < trees:
        status(folder, 'training', origin=origin, family=family, trees=trees, rows=int(good.sum()))
        m = CatBoostRegressor(iterations=trees, depth=7, learning_rate=.035, l2_leaf_reg=30,
            loss_function='MAE', thread_count=6, random_seed=270927, verbose=500,
            allow_writing_files=False, bootstrap_type='Bayesian', bagging_temperature=.5)
        m.fit(x.loc[good], y[good], sample_weight=weights[good], cat_features=['route', 'routehour', 'daytype'],
              callbacks=[Deadline(end, folder, family+'_'+str(origin))])
        m.save_model(str(path))
        save_json(path.with_suffix('.json'), dict(origin=origin, family=family, trees=m.tree_count_,
            features=list(x), training_max_day=int(meta.time.max()), rows=int(good.sum())))
    replay = CatBoostRegressor(); replay.load_model(str(path))
    np.testing.assert_array_equal(m.predict(xx), replay.predict(xx))
    def prediction(n):
        p = m.predict(xx, ntree_end=n)
        if residual:
            p = xx.analog_term.to_numpy()+p*np.maximum(30, xx.analog_term.to_numpy())
        return p
    return m, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=1.5)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    folder = OUT/('calendar_analog_smoke' if args.smoke else 'calendar_analog')
    folder.mkdir(parents=True, exist_ok=True)
    lock = open(folder/'run.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    # Respect the night-wide deadline, including later resumptions.
    end = min(time.time()+3600*args.hours, 1790483400.)  # 2026-09-27 04:30 UTC
    assert time.time() < end, 'Past training deadline'
    data = CalendarData(); data.causal_check()
    records = []; checkpoints = [12] if args.smoke else [500, 1500, 3000]
    for origin in ([151] if args.smoke else SELECTION):
        for name, p in data.analog(origin, 61).items():
            if name.endswith('support'):
                continue
            rec = data.export(folder, 'baseline_'+name, origin, p)
            rec.update(model='baseline_'+name, trees=0, split='selection'); records.append(rec)
        for family in ['absolute', 'residual']:
            if time.time() >= end:
                status(folder, 'budget_exhausted'); return
            model, predict = fit(data, folder, origin, family, max(checkpoints), end)
            for n in checkpoints:
                if n > model.tree_count_:
                    continue
                rec = data.export(folder, family+'_n'+str(n), origin, predict(n))
                rec.update(model=family, trees=n, split='selection'); records.append(rec)
                print('EVAL', rec, flush=True)
            pd.DataFrame(records).to_csv(folder/'metrics.csv', index=False)
            if model.tree_count_ < max(checkpoints):
                status(folder, 'budget_exhausted'); return
    if args.smoke:
        status(folder, 'smoke_passed', results=records); return
    selected = {}; summary = []
    r = pd.DataFrame(records)
    for family in ['absolute', 'residual']:
        subset = r[r.model == family]
        assert (subset.groupby('trees').origin.nunique() == len(SELECTION)).all()
        selected[family] = int(subset.groupby('trees').score.mean().idxmax())
    save_json(folder/'selection.json', selected)
    for origin in [CONTROL, 304]:
        for name, p in data.analog(origin, 61).items():
            if not name.endswith('support'):
                rec = data.export(folder, 'baseline_'+name, origin, p)
                rec.update(model='baseline_'+name, trees=0); summary.append(rec)
        for family, n in selected.items():
            if time.time() >= end:
                status(folder, 'budget_exhausted'); return
            model, predict = fit(data, folder, origin, family, n, end)
            if model.tree_count_ < n:
                status(folder, 'budget_exhausted'); return
            rec = data.export(folder, family+'_n'+str(n), origin, predict(n))
            rec.update(model=family, trees=n); summary.append(rec)
            print('EVAL', rec, flush=True)
    save_json(folder/'report.json', dict(selection=records, selected=selected, results=summary,
        rationale='Earlier 56-day recency profiles underestimated Sep-Oct volume by 10.8%; soft calendar matching reduces summer anchoring.',
        causal_check=True, saved_model_replay=True, public_score=None, active_best_changed=False,
        limitation='Already explored overlapping development windows; no promise of leaderboard improvement. Route 5 zero.'))
    status(folder, 'complete', results=summary)


if __name__ == '__main__':
    main()
