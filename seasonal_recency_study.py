"""Strict autumn/spring histories with weighted weekday profiles and CatBoost.

All label-derived state is fitted before the forecast origin. Calendar-age and
eligible-observation-age weighting are compared: removing summer need not make
all spring observations negligible merely because three calendar months passed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from weekly_baselines import read_matrix
from forecast_artifacts import artifact_directory

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/seasonal_recency_20260927'
KEY = ['route', 'date', 'hour']
SCOPES = {'all': list(range(1, 13)), 'spring_autumn': [3, 4, 5, 9, 10], 'autumn': [9, 10]}
FOLDS = [120, 243, 257, 273, 287]
HALVES = [7, 14, 28, 56, 112, 1000000000]
FEATURES = ['route', 'routehour', 'hour', 'daytype', 'dow', 'workday', 'holiday',
            'is_school_holiday', 'is_day_before_holiday', 'is_day_after_holiday',
            'is_shortened_workday', 'is_transferred_working_day',
            'days_to_next_holiday', 'days_since_previous_holiday', 'consecutive_days_off',
            'temp_mean', 'temp_min', 'temp_max', 'apparent_temp_mean',
            'precipitation_sum', 'rain_sum', 'snowfall_sum', 'wind_speed_mean',
            'humidity_mean', 'temp_change_24h', 'is_rain', 'is_snow', 'is_freezing',
            'edu_school_term', 'edu_first_week_back', 'edu_days_in_regime',
            'edu_uni_teaching_proxy', 'edu_uni_exam_proxy', 'service_rerouted',
            'service_reduced_frequency', 'service_stop_change', 'service_recent_restoration']


def metric(y, p):
    return float(max(0, 1 - np.abs(np.asarray(y) - p).sum() / np.sum(y)))


class Study:
    def __init__(self):
        self.frame = read_matrix().sort_values(['time', 'route', 'hour']).reset_index(drop=True)
        self.y = self.frame.boardings.to_numpy(dtype=float).reshape(365, 240)
        self.cancel = self.frame.service_cancelled.to_numpy(dtype=float).reshape(365, 240)
        daily = self.frame.groupby('time', sort=True).first()
        self.month = daily.month.to_numpy(dtype=int)
        self.daytype = daily.daytype.to_numpy(dtype=int)
        self.pooltype = np.where(self.daytype < 5, 0, self.daytype)
        self.groups = self.frame[['route', 'hour']].drop_duplicates().reset_index(drop=True)
        self.features = [c for c in FEATURES if c in self.frame]
        assert len(self.groups) == 240 and np.isfinite(self.y[:304]).all()
        assert self.frame.groupby('time').daytype.nunique().eq(1).all()
        assert np.isfinite(self.frame[self.features].to_numpy(dtype=float)).all()

    def days(self, origin, scope):
        days = np.flatnonzero((np.arange(365) < origin) & np.isin(self.month, SCOPES[scope]))
        assert len(days) == 0 or days.max() < origin
        return days

    def weights(self, days, origin, half, clock):
        age = origin - 1 - days if clock == 'calendar' else len(days) - 1 - np.arange(len(days))
        return np.exp2(-age / half)

    @staticmethod
    def stat(values, weights, kind):
        total = weights.sum(axis=0)
        if kind == 'mean':
            answer = np.divide((values * weights).sum(axis=0), total, out=np.zeros(240), where=total > 0)
        else:
            order = np.argsort(values, axis=0, kind='stable')
            sorted_y = np.take_along_axis(values, order, axis=0)
            sorted_w = np.take_along_axis(weights, order, axis=0)
            pos = (np.cumsum(sorted_w, axis=0) >= total[None, :] / 2).argmax(axis=0)
            answer = sorted_y[pos, np.arange(240)]
            answer[total == 0] = 0
        effective = np.divide(total ** 2, (weights ** 2).sum(axis=0), out=np.zeros(240), where=total > 0)
        return answer, effective

    def profile(self, origin, horizon, scope, half, clock, kind, shrink, values=None):
        days = self.days(origin, scope)
        assert len(days) >= 7, 'Insufficient eligible history; no out-of-scope fallback'
        y = (self.y if values is None else values)[days]
        weights = self.weights(days, origin, half, clock)[:, None] * (self.cancel[days] < .5)
        generic, _ = self.stat(y, weights, kind)
        bank = np.zeros((7, 240))
        for dow in range(7):
            exact = self.daytype[days] == dow
            pooled = self.pooltype[days] == (0 if dow < 5 else dow)
            coarse = self.stat(y[pooled], weights[pooled], kind)[0] if pooled.any() else generic
            if exact.any():
                detailed, effective = self.stat(y[exact], weights[exact], kind)
                detailed = np.where(effective > 0, detailed, coarse)
                mix = np.divide(effective, effective + shrink, out=np.zeros(240), where=effective + shrink > 0)
                bank[dow] = mix * detailed + (1 - mix) * coarse
            else:
                bank[dow] = coarse
        pred = bank[self.daytype[origin:origin + horizon]]
        return self.post(pred, origin, horizon), bank

    def post(self, prediction, origin, horizon):
        p = np.maximum(np.asarray(prediction).reshape(horizon, 240), 0)
        p = p * (1 - .975 * self.cancel[origin:origin + horizon])
        p[:, self.groups.route.to_numpy() == 5] = 0
        assert np.isfinite(p).all() and (p >= 0).all()
        return p

    def catboost(self, origin, horizon, scope, half, clock, save):
        days = self.days(origin, scope)
        assert len(days) >= 14
        rows = (days[:, None] * 240 + np.arange(240)[None, :]).reshape(-1)
        hist = self.frame.iloc[rows].copy()
        decay = np.repeat(self.weights(days, origin, half, clock), 240)
        good = (hist.route.to_numpy() != 5) & (hist.service_cancelled.to_numpy() < .5)
        hist = hist[good].copy()
        future = self.frame.iloc[origin * 240:(origin + horizon) * 240].copy()
        assert hist.time.max() < origin and set(hist.month).issubset(SCOPES[scope])
        model = CatBoostRegressor(iterations=700, depth=6, learning_rate=.045,
                                  loss_function='MAE', l2_leaf_reg=25, random_seed=250927,
                                  thread_count=6, verbose=False, allow_writing_files=False)
        model.fit(hist[self.features], hist.boardings, sample_weight=decay[good],
                  cat_features=['route', 'routehour', 'daytype'])
        p = self.post(model.predict(future[self.features]), origin, horizon)
        model.save_model(str(save))
        replay = CatBoostRegressor()
        replay.load_model(str(save))
        np.testing.assert_array_equal(p, self.post(replay.predict(future[self.features]), origin, horizon))
        future['boardings'] = 999983
        np.testing.assert_array_equal(p, self.post(model.predict(future[self.features]), origin, horizon))
        audit = dict(scope=scope, months=sorted(hist.month.unique().astype(int).tolist()),
                     first_label=str(hist.date.min()), last_label=str(hist.date.max()),
                     origin=origin, history_days=len(days), training_rows=len(hist), half_life=half,
                     weight_clock=clock, future_label_perturbation_max_diff=0., checkpoint_replay_max_diff=0.,
                     feature_names=self.features, model_sha256=hashlib.sha256(save.read_bytes()).hexdigest())
        save.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n')
        return p


def profile_name(scope, half, clock, kind, shrink):
    return f'profile_{scope}_h{half}_{clock}_{kind}_s{shrink}'


def validation(study):
    metrics, configs, missing = [], {}, []
    for origin in FOLDS:
        horizon = min(61, 304 - origin)
        date = str(study.frame.iloc[origin * 240].date)
        actual = study.y[origin:origin + horizon]
        for scope in SCOPES:
            days = study.days(origin, scope)
            if len(days) < 14:
                missing.append(dict(origin=date, scope=scope, history_days=len(days), reason='Fewer than 14 eligible days; no fallback to excluded months'))
                continue
            for half in HALVES:
                for clock in ['calendar', 'eligible']:
                    for kind in ['mean', 'median']:
                        for shrink in [0, 3]:
                            name = profile_name(scope, half, clock, kind, shrink)
                            p, _ = study.profile(origin, horizon, scope, half, clock, kind, shrink)
                            configs[name] = dict(family='profile', scope=scope, half=half, clock=clock, kind=kind, shrink=shrink)
                            metrics.append(dict(origin=date, horizon=horizon, history_days=len(days), scope=scope,
                                                model=name, family='profile', score=metric(actual, p),
                                                prediction_total=float(p.sum()), true_total=float(actual.sum())))
            for half, clock in [(28, 'calendar'), (56, 'eligible')]:
                name = f'cat_{scope}_h{half}_{clock}'
                p = study.catboost(origin, horizon, scope, half, clock, DEST / (date + '_' + name + '.cbm'))
                configs[name] = dict(family='cat', scope=scope, half=half, clock=clock)
                metrics.append(dict(origin=date, horizon=horizon, history_days=len(days), scope=scope,
                                    model=name, family='cat', score=metric(actual, p),
                                    prediction_total=float(p.sum()), true_total=float(actual.sum())))
                out = study.frame.iloc[origin * 240:(origin + horizon) * 240][KEY + ['boardings']].copy()
                out['prediction'] = p.reshape(-1)
                out.to_csv(DEST / (date + '_' + name + '.csv.gz'), index=False)
                print(date, name, 'score', round(metric(actual, p), 6), 'history_days', len(days), flush=True)
        baseline = artifact_directory('v8') / ('validation_' + date + '.csv')
        if baseline.exists():
            old = pd.read_csv(baseline, sep=';')
            keys = study.frame.iloc[origin * 240:(origin + horizon) * 240][KEY]
            aligned = keys.merge(old[KEY + ['prediction']], on=KEY, how='left', validate='one_to_one').prediction.to_numpy().reshape(horizon, 240)
            assert np.isfinite(aligned).all()
            for scale in [1., 1.03]:
                # +3% was chosen using public scores; it is a reference, not an untouched benchmark.
                p = np.rint(np.rint(aligned) * scale)
                metrics.append(dict(origin=date, horizon=horizon, history_days=origin, scope='reference',
                                    model='v8_scale_' + str(scale), family='reference', score=metric(actual, p)))
        pd.DataFrame(metrics).to_csv(DEST / 'metrics.csv', index=False)
        print('PROFILE RANK', date, pd.DataFrame(metrics).query('origin == @date').sort_values('score', ascending=False).head(6)[['model', 'score']].to_dict('records'), flush=True)
    m = pd.DataFrame(metrics)
    # Both selection windows and the later October diagnostic overlap. Never call this an independent holdout.
    selections = {}
    for scope in SCOPES:
        selections[scope] = {}
        for family in ['profile', 'cat']:
            subset = m[(m.scope == scope) & (m.family == family) & m.origin.isin(['2025-09-15', '2025-10-01'])]
            rank = subset.groupby('model').score.mean().sort_values(ascending=False)
            assert subset.groupby('model').origin.nunique().eq(2).all()
            name = rank.index[0]
            selections[scope][family] = dict(name=name, config=configs[name], selection_mean=float(rank.iloc[0]),
                                            scores=m[m.model == name].to_dict('records'))
    report = dict(selections=selections, missing_scopes=missing, configs=configs,
                  selection_origins=['2025-09-15', '2025-10-01'], public_best_score=.89402,
                  full_horizon_days=61, autumn_only_full_61_day_validation_available=False,
                  notes=['Previously explored overlapping development windows; no independent holdout.',
                         'May and September controls have 61 days; Sep15 47, Oct1 31, Oct15 17.',
                         'Autumn-only before Sep1 has no eligible observations and is skipped.',
                         'All label-derived profiles and CatBoost fitting exclude months outside their scope.',
                         'Supplied retrospective weather and service covariates; not an operational as-of weather forecast.',
                         'No global +3% scaling or v8 predictions enter the new standalone models.'])
    (DEST / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SELECTED', json.dumps(selections, indent=2), flush=True)


def final(study):
    report = json.loads((DEST / 'report.json').read_text())
    template = pd.read_csv(ROOT / 'data/test_submission.csv', sep=';')
    exports = []
    for scope in ['autumn', 'spring_autumn']:
        for family in ['profile', 'cat']:
            selection = report['selections'][scope][family]
            c = selection['config'].copy()
            c.pop('family')
            name = selection['name']
            if family == 'profile':
                p, bank = study.profile(304, 61, **c)
                changed = study.y.copy()
                changed[304:] = 999983
                np.testing.assert_array_equal(p, study.profile(304, 61, values=changed, **c)[0])
                np.savez_compressed(DEST / ('final_' + name + '.npz'), profile_bank=bank, prediction=p)
            else:
                p = study.catboost(304, 61, save=DEST / ('final_' + name + '.cbm'), **c)
            out = study.frame.iloc[304 * 240:][KEY].copy()
            out['prediction'] = np.rint(p.reshape(-1)).astype(np.int64)
            out = template[KEY].merge(out, on=KEY, how='left', validate='one_to_one')
            assert len(out) == 14640 and out[KEY].equals(template[KEY]) and out.prediction.notna().all()
            assert (out.prediction >= 0).all() and (out.loc[out.route == 5, 'prediction'] == 0).all()
            path = DEST / ('submission_' + scope + '_' + family + '.csv')
            out.to_csv(path, sep=';', index=False, lineterminator='\n')
            days = study.days(304, scope)
            exports.append(dict(submission=str(path.relative_to(ROOT)), selected_model=name,
                                config=c, training_months=sorted(np.unique(study.month[days]).astype(int).tolist()),
                                training_days=len(days), rows=len(out), prediction_total=int(out.prediction.sum()),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), public_score=None,
                                future_label_perturbation_max_diff=0.))
            print('EXPORTED', exports[-1], flush=True)
    (DEST / 'submissions.json').write_text(json.dumps(exports, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['validation', 'final'], default='validation')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    study = Study()
    (validation if args.phase == 'validation' else final)(study)


if __name__ == '__main__':
    main()
