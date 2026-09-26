"""Fixed-origin forecast of transaction-observed fleet, then boardings per vehicle.

Train only on Studio. No contemporaneous hidden fleet enters eligible forecasts.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from weekly_baselines import read_matrix
from seasonal_recency_study import FEATURES, metric
from forecast_artifacts import artifact_directory

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/fleet_forecast_20260927'
KEY = ['route', 'date', 'hour']


class FleetStudy:
    def __init__(self):
        self.d = read_matrix().sort_values(['date', 'route', 'hour']).reset_index(drop=True)
        op = pd.read_csv(ROOT / 'data/features/07_transaction_derived/operational_features_v2.csv')
        op = op[op.date < '2025-11-01']
        x = self.d[KEY + ['boardings']].merge(op[KEY + ['active_vehicles', 'raw_transactions', 'successful_transactions']], how='left', on=KEY, validate='one_to_one')
        known = x.date < '2025-11-01'
        np.testing.assert_array_equal(x.loc[known, 'boardings'], x.loc[known, 'successful_transactions'].fillna(0))
        # Absent transaction cells are zero observed vehicles. Transactions with
        # no usable garage identifier are unknown fleet, not zero vehicles.
        x.loc[known & x.raw_transactions.isna(), 'active_vehicles'] = 0
        self.f = x.active_vehicles.to_numpy(copy=True).reshape(365, 240)
        self.y = self.d.boardings.to_numpy(copy=True).reshape(365, 240)
        self.cancel = self.d.service_cancelled.to_numpy().reshape(365, 240)
        self.daytype = self.d.groupby('time').daytype.first().to_numpy()
        self.cats = ['route', 'routehour', 'daytype']
        supply = pd.read_csv(ROOT / 'data/schedule_research_20260927/archived_supply_hourly_2025.csv')
        self.d = self.d.merge(supply, on=KEY, how='left', validate='one_to_one')
        self.cols = [c for c in FEATURES if c in self.d] + [
            'scheduled_mean_moving_trams', 'scheduled_terminal_departures', 'schedule_day_covered']
        self.d[self.cols] = self.d[self.cols].fillna(-1)
        self.cache = {}
        self.audit = dict(known_rows=int(known.sum()), unknown_fleet_rows=int(np.isnan(self.f[:304]).sum()),
                          positive_boardings_unknown_fleet=int(((self.y[:304] > 0) & ~np.isfinite(self.f[:304])).sum()))

    def bank(self, origin, half, target):
        key = (origin, half, target)
        if key in self.cache:
            return self.cache[key]
        numerator = self.y[:origin] if target in ['passengers', 'occupancy'] else self.f[:origin]
        denominator = self.f[:origin] if target == 'occupancy' else np.ones_like(numerator)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (self.cancel[:origin] < .5)
        if target == 'occupancy':
            valid &= denominator > 0
        w = np.exp2(-(origin - 1 - np.arange(origin)) / half)[:, None] * valid
        n, den = np.nan_to_num(numerator), np.nan_to_num(denominator)
        def estimate(mask):
            ww = w * mask[:, None]
            return np.divide((ww * n).sum(axis=0), (ww * den).sum(axis=0),
                             out=np.zeros(240), where=(ww * den).sum(axis=0) > 0)
        pooled_type = np.where(self.daytype[:origin] < 5, 0, self.daytype[:origin])
        banks = []
        for daytype in range(7):
            exact = self.daytype[:origin] == daytype
            pool = pooled_type == (0 if daytype < 5 else daytype)
            exact_w = (w * exact[:, None]).sum(axis=0)
            coarse = estimate(pool) if pool.any() else estimate(np.ones(origin, bool))
            banks.append((estimate(exact) * exact_w + 3 * coarse) / (exact_w + 3))
        self.cache[key] = np.array(banks)
        return self.cache[key]

    def pack(self, origin, horizon):
        frame = self.d.iloc[origin * 240:(origin + horizon) * 240].copy()
        x = frame[self.cols].copy().reset_index(drop=True)
        for kind in ['passengers', 'fleet', 'occupancy']:
            for half in [28, 84]:
                x[kind + '_' + str(half)] = self.bank(origin, half, kind)[self.daytype[origin:origin + horizon]].reshape(-1)
        x['horizon'] = np.repeat(np.arange(1, horizon + 1), 240)
        assert np.isfinite(x.to_numpy(dtype=float)).all()
        return x, frame.reset_index(drop=True)

    def causal_check(self, origin):
        x, _ = self.pack(origin, 61)
        y, f = self.y.copy(), self.f.copy()
        try:
            self.y[origin:] = 999983; self.f[origin:] = 999979; self.cache.clear()
            changed, _ = self.pack(origin, 61)
            pd.testing.assert_frame_equal(x, changed)
        finally:
            self.y, self.f = y, f; self.cache.clear()

    def training(self, origin):
        chunks = [self.pack(c, min(61, origin - c)) for c in range(56, origin - 13, 28)]
        x = pd.concat([v[0] for v in chunks], ignore_index=True)
        meta = pd.concat([v[1] for v in chunks], ignore_index=True)
        idx = meta.time.to_numpy().astype(int) * 240 + meta.route.map({r:i * 24 for i,r in enumerate(sorted(self.d.route.unique()))}).to_numpy() + meta.hour.to_numpy()
        fleet = self.f.reshape(-1)[idx]
        age = origin - 1 - meta.time.to_numpy()
        weights = np.exp2(-age / 112) / meta.groupby(KEY).boardings.transform('size').to_numpy()
        assert meta.time.max() < origin
        return x, meta, fleet, weights

    def fit_predict(self, origin, horizon):
        x, meta, fleet, weights = self.training(origin)
        xx, future = self.pack(origin, horizon)
        y = meta.boardings.to_numpy()
        common = (meta.route.to_numpy() != 5) & (meta.service_cancelled.to_numpy() < .5)
        models = {}; audit = []
        for name in ['fleet_l1', 'fleet_poisson', 'occupancy_l1', 'direct_l1']:
            cols = list(x)
            if name == 'direct_l1':
                cols = [c for c in cols if not c.startswith(('fleet_', 'occupancy_'))]
            if name.startswith('fleet'):
                # Passenger profiles are excluded from the fleet model: fleet
                # remains a forecast from its own history + exogenous inputs.
                cols = [c for c in cols if not c.startswith(('passengers_', 'occupancy_'))]
                scale, scale_future = np.maximum(1, x.fleet_28.to_numpy()), np.maximum(1, xx.fleet_28.to_numpy())
                target = fleet / scale
                good = common & np.isfinite(fleet)
                weight = weights * scale
            elif name == 'occupancy_l1':
                cols = [c for c in cols if not c.startswith('passengers_')]
                scale, scale_future = np.maximum(5, x.occupancy_84.to_numpy()), np.maximum(5, xx.occupancy_84.to_numpy())
                target = np.divide(y, fleet, out=np.zeros(len(y)), where=fleet > 0) / scale
                good = common & np.isfinite(fleet) & (fleet > 0)
                # Weighted absolute occupancy error corresponds to absolute
                # passenger error when fleet is known during training.
                weight = weights * np.nan_to_num(fleet) * scale
            else:
                scale, scale_future = np.maximum(30, x.passengers_28.to_numpy()), np.maximum(30, xx.passengers_28.to_numpy())
                target = y / scale; good = common; weight = weights * scale
            model = lgb.LGBMRegressor(objective='poisson' if name.endswith('poisson') else 'regression_l1',
                n_estimators=500, learning_rate=.035, num_leaves=31, min_child_samples=120,
                reg_lambda=20, n_jobs=6, verbosity=-1, random_state=270927,
                deterministic=True, force_col_wise=True)
            model.fit(x.loc[good, cols], target[good], sample_weight=weight[good], categorical_feature=self.cats)
            pred = np.maximum(0, model.predict(xx[cols]) * scale_future)
            path = OUT / (str(origin) + '_' + name + '.txt')
            model.booster_.save_model(str(path))
            replay = np.maximum(0, lgb.Booster(model_file=str(path)).predict(xx[cols], num_threads=6) * scale_future)
            np.testing.assert_array_equal(pred, replay)
            models[name] = pred
            audit.append(dict(model=name, rows=int(good.sum()), max_training_day=int(meta.loc[good, 'time'].max()),
                              features=cols, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), replay_exact=True))
            print('FIT', origin, name, int(good.sum()), flush=True)
        cancel = np.clip(1 - .975 * future.service_cancelled.to_numpy(), 0, 1)
        active = (future.route.to_numpy() != 5).astype(float) * cancel
        p = {
            'direct_l1': models['direct_l1'] * active,
            'fleet_l1_x_occupancy': models['fleet_l1'] * models['occupancy_l1'] * active,
            'fleet_poisson_x_occupancy': models['fleet_poisson'] * models['occupancy_l1'] * active,
            'fleet_profile_x_occupancy': xx.fleet_28.to_numpy() * models['occupancy_l1'] * active,
            'factorized_profiles_28_84': xx.fleet_28.to_numpy() * xx.occupancy_84.to_numpy() * active,
        }
        (OUT / (str(origin) + '_audit.json')).write_text(json.dumps(audit, indent=2) + '\n')
        out = future[KEY + ['boardings']].copy()
        for name, values in p.items():out[name] = values
        out['predicted_observed_vehicles_l1'] = models['fleet_l1'] * active
        out['predicted_observed_vehicles_poisson'] = models['fleet_poisson'] * active
        out['observed_vehicles_profile'] = xx.fleet_28.to_numpy() * active
        out['predicted_boardings_per_vehicle'] = models['occupancy_l1']
        return out


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    e = FleetStudy(); results = []
    for origin in [181, 212, 243]:
        e.causal_check(origin)
        out = e.fit_predict(origin, 61)
        observed = e.f[origin:origin + 61].reshape(-1)
        good = np.isfinite(observed)
        for col in ['direct_l1', 'fleet_l1_x_occupancy', 'fleet_poisson_x_occupancy', 'fleet_profile_x_occupancy', 'factorized_profiles_28_84']:
            results.append(dict(origin=out.date.min(), model=col, score=metric(out.boardings, np.rint(out[col]))))
        for col in ['predicted_observed_vehicles_l1', 'predicted_observed_vehicles_poisson', 'observed_vehicles_profile']:
            results.append(dict(origin=out.date.min(), model=col, score=metric(observed[good], out[col].to_numpy()[good]), target='transaction-observed vehicles'))
        baseline = pd.read_csv(artifact_directory('v8') / ('validation_' + out.date.min() + '.csv'), sep=';')
        bp = out[KEY].merge(baseline[KEY + ['prediction']], on=KEY, validate='one_to_one').prediction.to_numpy()
        assert np.isfinite(bp).all()
        results.append(dict(origin=out.date.min(), model='confirmed_v8_up3', score=metric(out.boardings, np.rint(bp * 1.03))))
        out['observed_vehicles_DIAGNOSTIC_ONLY'] = observed
        out['oracle_fleet_DIAGNOSTIC_ONLY'] = observed * out.predicted_boardings_per_vehicle.to_numpy()
        results.append(dict(origin=out.date.min(), model='oracle_fleet_DIAGNOSTIC_ONLY', score=metric(out.boardings.to_numpy()[good], out.oracle_fleet_DIAGNOSTIC_ONLY.to_numpy()[good]), target='diagnostic on known-fleet subset; not eligible'))
        out.to_csv(OUT / ('validation_' + out.date.min() + '.csv'), index=False)
        pd.DataFrame(results).to_csv(OUT / 'metrics.csv', index=False)
        print('RESULT', results[-10:], flush=True)
    r = pd.DataFrame(results)
    eligible = ['fleet_l1_x_occupancy', 'fleet_poisson_x_occupancy', 'fleet_profile_x_occupancy', 'factorized_profiles_28_84']
    ranking = r[r.model.isin(eligible)].groupby('model').score.mean().sort_values(ascending=False)
    chosen = ranking.index[0]
    e.causal_check(304)
    out = e.fit_predict(304, 61)
    out.to_csv(OUT / 'fleet_and_passenger_forecast.csv', index=False)
    sub = out[KEY].copy();sub['prediction'] = np.rint(out[chosen]).astype(int)
    assert len(sub) == 14640 and not sub.duplicated(KEY).any() and sub.prediction.ge(0).all()
    sub.to_csv(OUT / 'submission_fleet.csv', sep=';', index=False)
    report = dict(selected=chosen, ranking=ranking.to_dict(), metrics=results, data_audit=e.audit,
        public_score=None, active_model_changed=False, confirmed_public_best=.89402,
        candidate_sha256=hashlib.sha256((OUT / 'submission_fleet.csv').read_bytes()).hexdigest(),
        definition='Number of distinct vehicles with at least one recorded transaction in hour, not true dispatched/concurrent fleet.',
        causal_checks='Perturbing all hidden passenger and observed-fleet labels leaves features exactly unchanged at all four origins.',
        selection='Mean of three previously explored overlapping 61-day development windows. No public selection.',
        caveats=['No hidden vehicle observations used in eligible forecasts.', 'Route 5 has no training fleet/passenger history; remains zero in this experiment, separate cold-start problem.',
                 'Current route-5 timetable not used as historical 2025 feature.', 'Weather/service covariates are existing retrospective inputs.',
                 'Archived May GTFS is incomplete before service start dates and has no holiday exception table.',
                 'Occupancy uses observed training fleet as a target denominator, not a future feature; errors compound at inference.'])
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('DONE', chosen, flush=True)


if __name__ == '__main__':
    main()
