"""Controlled weekly targets: absolute, additive, relative and log1p changes.

Fit only on observations before each origin. Each recursive call predicts seven
days from the preceding same-weekday values; all 61 future days stay hidden.
The direct control learns gaps of 1..9 weeks using only observed anchor history.
No forecast from v8 or any other fitted model enters these models.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from forecast_artifacts import artifact_directory
from weekly_baselines import read_matrix

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v17_weekly_targets'
KEY = ['route', 'date', 'hour']
SEED = 20260926
OBSERVED_DAYS = 304
CONFIGS = {
    'recursive_absolute': ('absolute', 'recursive'),
    'recursive_delta': ('delta', 'recursive'),
    'recursive_relative': ('relative', 'recursive'),
    'recursive_logratio': ('logratio', 'recursive'),
    'direct_logratio': ('logratio', 'direct'),
}
EXOG = [
    'daytype', 'workday', 'holiday', 'summer', 'sin_year', 'cos_year',
    'temp_mean', 'precipitation_sum', 'snowfall_sum', 'wind_speed_mean',
    'is_school_holiday', 'is_summer_school_break', 'edu_school_term',
    'edu_uni_teaching_proxy', 'edu_uni_exam_proxy', 'is_pre_new_year_week',
    'new_year_proximity_21d', 'service_cancelled', 'service_rerouted',
    'service_reduced_frequency', 'service_incident_fraction', 'season_2324',
]


def read_data():
    frame = read_matrix().sort_values(['time', 'route', 'hour']).reset_index(drop=True)
    external = pd.read_csv(ROOT / 'data/features/08_external_seasonality/moscow_monthly_passengers.csv', sep=';')
    external = external[(external.TransportType == 'Трамвай') & external.Year.astype(str).isin(['2023', '2024'])].copy()
    months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    external['month'] = external.Month.map(dict(zip(months, range(1, 13))))
    dates = pd.to_datetime(dict(year=external.Year.astype(int), month=external.month, day=1))
    external['daily'] = external.PassengerTraffic.astype(float) / dates.dt.days_in_month
    external['index'] = external.daily / external.groupby('Year').daily.transform('mean')
    seasonal = external.groupby('month')['index'].mean()
    assert len(seasonal) == 12
    frame['season_2324'] = frame.month.map(seasonal)
    groups = frame[['route', 'hour']].drop_duplicates().reset_index(drop=True)
    assert len(groups) == 240 and len(frame) == 365 * 240
    y = frame.boardings.to_numpy(dtype=float).reshape(365, 240)
    ex = frame[EXOG].to_numpy(dtype=float).reshape(365, 240, len(EXOG))
    assert np.isfinite(y[:OBSERVED_DAYS]).all() and np.isfinite(ex).all()
    return frame, y, ex, groups


def features(values, ex, groups, targets, anchors):
    """Label-derived inputs use anchor, anchor-7, anchor-14 and anchor-21 only."""
    targets = np.asarray(targets, dtype=int)
    anchors = np.asarray(anchors, dtype=int)
    assert (anchors >= 21).all() and (targets > anchors).all()
    assert ((targets - anchors) % 7 == 0).all()
    n = len(targets)
    lag = np.stack([values[anchors - 7 * k] for k in range(4)], axis=-1)
    assert np.isfinite(lag).all(), 'Attempt to read an unavailable history value'
    fx, ax = ex[targets], ex[anchors]
    data = {
        'route': np.tile(groups.route.to_numpy(dtype=int), n),
        'routehour': np.tile((groups.route * 24 + groups.hour).to_numpy(dtype=int), n),
        'hour': np.tile(groups.hour.to_numpy(dtype=int), n),
        'gap_weeks': np.repeat((targets - anchors) // 7, 240),
    }
    for j, name in enumerate(EXOG):
        data['future_' + name] = fx[:, :, j].reshape(-1)
        data['anchor_' + name] = ax[:, :, j].reshape(-1)
        data['change_' + name] = (fx[:, :, j] - ax[:, :, j]).reshape(-1)
    for k in range(4):
        data['lag_' + str(k + 1)] = lag[:, :, k].reshape(-1)
    data['lag_mean'] = lag.mean(axis=-1).reshape(-1)
    data['lag_median'] = np.median(lag, axis=-1).reshape(-1)
    data['lag_std'] = lag.std(axis=-1).reshape(-1)
    data['lag_log_change'] = (np.log1p(lag[:, :, 0]) - np.log1p(lag[:, :, 1])).reshape(-1)
    route_totals = lag.reshape(n, 10, 24, 4).sum(axis=2)
    route_base = np.repeat(route_totals[:, :, 0], 24, axis=1)
    data['anchor_route_total'] = route_base.reshape(-1)
    data['anchor_route_change'] = np.repeat(np.log1p(route_totals[:, :, 0]) - np.log1p(route_totals[:, :, 1]), 24, axis=1).reshape(-1)
    data['anchor_network_total'] = np.repeat(route_totals[:, :, 0].sum(axis=1), 240)
    data['anchor_hour_share'] = ((lag[:, :, 0] + 1) / (route_base + 24)).reshape(-1)
    result = pd.DataFrame(data)
    assert np.isfinite(result.to_numpy()).all()
    return result, lag[:, :, 0].reshape(-1)


def transformed_target(y, base, kind):
    if kind == 'absolute':
        return y
    if kind == 'delta':
        return y - base
    if kind == 'relative':
        return (y - base) / (base + 1)
    if kind == 'logratio':
        return np.log1p(y) - np.log1p(base)
    raise ValueError(kind)


def inverse_target(prediction, base, kind):
    if kind == 'absolute':
        p = prediction
    elif kind == 'delta':
        p = base + prediction
    elif kind == 'relative':
        p = base + (base + 1) * prediction
    elif kind == 'logratio':
        p = np.expm1(np.clip(np.log1p(base) + prediction, -20, np.log1p(1e9)))
    else:
        raise ValueError(kind)
    return np.maximum(p, 0)


def training_data(y, ex, groups, origin, kind, mode):
    observed = y.copy()
    observed[origin:] = np.nan
    gaps = [1] if mode == 'recursive' else range(1, 10)
    xs, ys, ws = [], [], []
    targets_used = []
    for gap in gaps:
        days = np.arange(21 + 7 * gap, origin)
        anchors = days - 7 * gap
        x, base = features(observed, ex, groups, days, anchors)
        actual = observed[days].reshape(-1)
        transformed = transformed_target(actual, base, kind)
        weights = np.repeat(0.5 ** ((origin - 1 - days) / 240), 240)
        # For the relative target this makes transformed MAE exactly original-
        # scale MAE. For log1p it is its local first-order approximation.
        if kind in ('relative', 'logratio'):
            weights *= base + 1
        xs.append(x)
        ys.append(transformed)
        ws.append(weights)
        targets_used.extend(days.tolist())
        assert (anchors < days).all() and days.max() < origin
    return (pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws),
            {'target_min_day': int(min(targets_used)), 'target_max_day': int(max(targets_used)),
             'training_gaps_weeks': list(gaps), 'origin_day': origin,
             'future_labels_excluded': True})


def predict(model, y, ex, groups, origin, horizon, kind, mode):
    values = np.full_like(y, np.nan)
    values[:origin] = y[:origin]
    if mode == 'direct':
        targets = np.arange(origin, origin + horizon)
        anchors = origin - 7 + np.arange(horizon) % 7
        x, base = features(values, ex, groups, targets, anchors)
        p = inverse_target(model.predict(x), base, kind).reshape(horizon, 240)
        p[:, groups.route.to_numpy() == 5] = 0
        return p
    for offset in range(0, horizon, 7):
        targets = np.arange(origin + offset, min(origin + offset + 7, origin + horizon))
        x, base = features(values, ex, groups, targets, targets - 7)
        p = inverse_target(model.predict(x), base, kind).reshape(len(targets), 240)
        p[:, groups.route.to_numpy() == 5] = 0
        values[targets] = p
    return values[origin:origin + horizon]


def baselines(y, origin, horizon):
    outputs = {}
    for period in (7, 14):
        values = np.full_like(y, np.nan)
        values[:origin] = y[:origin]
        for day in range(origin, origin + horizon):
            values[day] = values[day - period]
        outputs['repeat_' + str(period)] = values[origin:origin + horizon]
    anchors = origin - 7 + np.arange(horizon) % 7
    outputs['mean_4weeks'] = np.stack([y[anchors - k * 7] for k in range(4)]).mean(axis=0)
    return outputs


def evaluate(y, p):
    return float(max(0, 1 - np.abs(y - p).sum() / y.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origins', default='181,212,243,273')
    parser.add_argument('--models', default=','.join(CONFIGS))
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    frame, y, ex, groups = read_data()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metrics, audits = [], []
    names = args.models.split(',')
    for origin in map(int, args.origins.split(',')):
        horizon = min(61, (365 if origin == OBSERVED_DAYS else OBSERVED_DAYS) - origin)
        assert horizon > 0 and 84 <= origin <= OBSERVED_DAYS
        rows = frame[(frame.time >= origin) & (frame.time < origin + horizon)][KEY + ['boardings']].copy()
        rows.reset_index(drop=True, inplace=True)
        date = str(rows.date.min())
        outputs = baselines(y, origin, horizon)
        for name in names:
            start = time.time()
            kind, mode = CONFIGS[name]
            x, target, weights, audit = training_data(y, ex, groups, origin, kind, mode)
            model = CatBoostRegressor(
                iterations=args.iterations, depth=6, learning_rate=0.05,
                loss_function='MAE', l2_leaf_reg=15, random_seed=SEED,
                thread_count=args.threads, verbose=False, allow_writing_files=False)
            model.fit(x, target, sample_weight=weights, cat_features=['route', 'routehour'])
            p = predict(model, y, ex, groups, origin, horizon, kind, mode)
            assert np.isfinite(p).all() and (p >= 0).all()
            corrupted = y.copy()
            corrupted[origin:] = 999983
            np.testing.assert_array_equal(p, predict(model, corrupted, ex, groups, origin, horizon, kind, mode))
            model_path = DEST / (date + '_' + name + '.cbm')
            model.save_model(str(model_path))
            loaded = CatBoostRegressor()
            loaded.load_model(str(model_path))
            np.testing.assert_array_equal(p, predict(loaded, y, ex, groups, origin, horizon, kind, mode))
            audit.update(model=name, target=kind, forecast_mode=mode, origin=date,
                         forecast_days=horizon, rows=len(x), feature_names=list(x.columns),
                         iterations=args.iterations, code_sha256=code_hash,
                         model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                         future_label_perturbation_max_diff=0., checkpoint_replay_max_diff=0.,
                         seconds=time.time() - start)
            (DEST / (date + '_' + name + '.json')).write_text(json.dumps(audit, indent=2) + '\n')
            audits.append(audit)
            outputs[name] = p
            s = evaluate(y[origin:origin + horizon], p) if origin < OBSERVED_DAYS else None
            print(json.dumps({'origin': date, 'model': name, 'score': s,
                              'rows': len(x), 'seconds': round(audit['seconds'], 1)}), flush=True)
            del x, target, weights, model, loaded
        if origin < OBSERVED_DAYS:
            for family, prefix in [('v8', 'validation_'), ('v13_anchored', 'selected_validation_')]:
                path = artifact_directory(family) / (prefix + date + '.csv')
                if path.exists():
                    old = pd.read_csv(path, sep=';')
                    aligned = rows[KEY].merge(old[KEY + ['prediction']], on=KEY, how='left', validate='one_to_one')
                    assert aligned.prediction.notna().all()
                    outputs[family] = aligned.prediction.to_numpy().reshape(horizon, 240)
            for name, p in outputs.items():
                for label, sl in [('all', slice(None)), ('first_7', slice(0, 7)),
                                  ('days_8_28', slice(7, 28)), ('days_29_61', slice(28, None)),
                                  ('last_7', slice(-7, None))]:
                    actual = y[origin:origin + horizon][sl]
                    if len(actual):
                        metrics.append({'origin': date, 'days': horizon, 'model': name,
                                        'part': label, 'score': evaluate(actual, p[sl])})
        for name, p in outputs.items():
            rows[name] = p.reshape(-1)
        rows.to_csv(DEST / (date + '_predictions.csv.gz'), index=False)
        pd.DataFrame(metrics).to_csv(DEST / 'metrics.csv', index=False)
        (DEST / 'audit_summary.json').write_text(json.dumps(audits, indent=2) + '\n')
    if metrics:
        m = pd.DataFrame(metrics)
        table = m[m.part == 'all'].pivot(index='model', columns='origin', values='score')
        print(table.round(6).to_string(), flush=True)
        rank = m[(m.part == 'all') & m.origin.isin(['2025-07-01', '2025-08-01']) & m.model.isin(names)].groupby('model').score.mean().sort_values(ascending=False)
        report = {'ranking_july_august': rank.to_dict(), 'scores': metrics,
                  'code_sha256': code_hash, 'v8_public_best': 0.89214,
                  'v15_user_reported': 0.87084, 'v16_user_reported': 0.88501,
                  'future_labels_used': False, 'v8_is_model_input': False,
                  'caveat': 'Previously explored overlapping development windows; October is 31 days. Retrospective supplied exogenous features. No independent holdout or known new public score.'}
        (DEST / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
