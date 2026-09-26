"""Paired raw passenger models with predicted versus observed fleet input."""
import json
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from fleet_forecast_study import FleetStudy, ROOT, KEY, metric
from forecast_artifacts import artifact_directory

OUT = ROOT / 'outputs/fleet_feature_20260927'


def fit(x, y, w, path):
    m = lgb.LGBMRegressor(objective='regression_l1', n_estimators=800, learning_rate=.035,
        num_leaves=31, min_child_samples=100, reg_lambda=15, n_jobs=6,
        random_state=270927, verbosity=-1, deterministic=True, force_col_wise=True)
    m.fit(x, y, sample_weight=w, categorical_feature=['route', 'routehour', 'daytype'])
    m.booster_.save_model(str(path))
    pd.DataFrame({'feature':list(x), 'gain':m.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False).to_csv(path.with_suffix('.importance.csv'),index=False)
    return m


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    e = FleetStudy(); results = []
    for origin in [181, 212, 243, 304]:
        hist = e.d.iloc[:origin * 240].copy()
        v = e.d.iloc[origin * 240:(origin + 61) * 240].copy()
        x, xx = hist[e.cols].copy(), v[e.cols].copy()
        f = e.f[:origin].reshape(-1)
        fv = e.f[origin:origin + 61].reshape(-1)
        good = (hist.route != 5).to_numpy() & np.isfinite(f) & (hist.service_cancelled.to_numpy() < .5)
        # Both passenger alternatives train on exactly the same rows and weights.
        w = np.exp2(-(origin - 1 - hist.time.to_numpy()) / 365)
        w *= np.where(hist.month.isin([1, 2]), .35, np.where(hist.month.isin([6, 7, 8]), .5, 1.))
        cancel = np.clip(1 - .975 * v.service_cancelled.to_numpy(), 0, 1)
        active = cancel * (v.route.to_numpy() != 5)
        fm = fit(x.loc[good], f[good], w[good], OUT / (str(origin) + '_fleet.txt'))
        fp = np.maximum(0, fm.predict(xx))
        baseline = fit(x.loc[good], hist.boardings.to_numpy()[good], w[good], OUT / (str(origin) + '_baseline.txt'))
        xb, xxb = x.copy(), xx.copy()
        xb['vehicle_count'] = f
        xxb['vehicle_count'] = fp
        model = fit(xb.loc[good], hist.boardings.to_numpy()[good], w[good], OUT / (str(origin) + '_passengers_with_fleet.txt'))
        out = v[KEY + ['boardings']].copy()
        out['direct_baseline'] = np.maximum(0, baseline.predict(xx)) * active
        out['forecast_fleet_input'] = np.maximum(0, model.predict(xxb)) * active
        out['predicted_observed_vehicles'] = fp * active
        # Saved model replay verifies the submitted inference path.
        replay = lgb.Booster(model_file=str(OUT / (str(origin) + '_passengers_with_fleet.txt')))
        np.testing.assert_array_equal(model.predict(xxb), replay.predict(xxb, num_threads=6))
        if origin < 304:
            known = np.isfinite(fv)
            oracle = xxb.copy(); oracle['vehicle_count'] = fv
            out['known_fleet_DIAGNOSTIC'] = np.maximum(0, model.predict(oracle)) * active
            out['known_observed_vehicles_DIAGNOSTIC'] = fv
            for name in ['direct_baseline', 'forecast_fleet_input']:
                results.append(dict(origin=str(v.date.min()), model=name, score=metric(v.boardings, np.rint(out[name]))))
                results.append(dict(origin=str(v.date.min()), model=name + '_known_subset', score=metric(v.boardings.to_numpy()[known], np.rint(out[name].to_numpy()[known]))))
            results.append(dict(origin=str(v.date.min()), model='known_fleet_DIAGNOSTIC', score=metric(v.boardings.to_numpy()[known], np.rint(out.known_fleet_DIAGNOSTIC.to_numpy()[known]))))
            results.append(dict(origin=str(v.date.min()), model='fleet_forecast_score', score=metric(fv[known], out.predicted_observed_vehicles.to_numpy()[known])))
            ref = pd.read_csv(artifact_directory('v8') / ('validation_' + v.date.min() + '.csv'), sep=';')
            bp = v[KEY].merge(ref[KEY + ['prediction']], on=KEY, validate='one_to_one').prediction
            results.append(dict(origin=str(v.date.min()), model='confirmed_v8_up3', score=metric(v.boardings, np.rint(bp * 1.03))))
            print(results[-7:], flush=True)
        else:
            sub = out[KEY].copy(); sub['prediction'] = np.rint(out.forecast_fleet_input).astype(int)
            assert len(sub) == 14640 and sub.prediction.ge(0).all() and not sub.duplicated(KEY).any()
            sub.to_csv(OUT / 'submission_fleet_feature.csv', sep=';', index=False)
        out.to_csv(OUT / ('forecast_' + str(origin) + '.csv'), index=False)
        pd.DataFrame(results).to_csv(OUT / 'metrics.csv', index=False)
    report = dict(metrics=results, forecast_variant='Passenger model trained with observed historical fleet, inference uses separately predicted fleet.',
        observed_variant='User reports organizer permission for retrospective/leakage inputs. True future fleet not available; known fleet evaluated only on historical validation.',
        active_model_changed=False, public_score=None, confirmed_public_best=.89402,
        route5='No observed training data: kept zero; current timetable cannot determine passenger demand.',
        training_cutoffs=[181,212,243,304], paired_rows_and_weights=True, checkpoint_replay_exact=True,
        submission_sha256=hashlib.sha256((OUT / 'submission_fleet_feature.csv').read_bytes()).hexdigest(),
        limitations=['Observed distinct transaction vehicles is not actual dispatched fleet.', 'Teacher forcing: stage 2 sees true historical fleet while inference sees forecast fleet.',
                    'Retrospective weather and service flags retained; previously explored overlapping 61-day windows.',
                    'Known-fleet diagnostics use finite-fleet subset; matched baseline subset reported separately.'])
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('DONE',flush=True)


if __name__ == '__main__':
    main()
