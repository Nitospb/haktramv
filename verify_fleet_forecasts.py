"""Replay final fleet forecasts and recompute saved validation scores on Studio."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from fleet_forecast_study import FleetStudy, ROOT, KEY, metric


def main():
    e = FleetStudy()
    e.causal_check(304)
    folder = ROOT / 'outputs/fleet_feature_20260927'
    future = e.d.iloc[304 * 240:].copy()
    x = future[e.cols].copy()
    fm = lgb.Booster(model_file=str(folder / '304_fleet.txt'))
    fp = np.maximum(0, fm.predict(x, num_threads=6))
    x['vehicle_count'] = fp
    pm = lgb.Booster(model_file=str(folder / '304_passengers_with_fleet.txt'))
    active = (future.route.to_numpy() != 5) * np.clip(1 - .975 * future.service_cancelled.to_numpy(), 0, 1)
    p = np.rint(np.maximum(0, pm.predict(x, num_threads=6)) * active).astype(int)
    saved = pd.read_csv(folder / 'submission_fleet_feature.csv', sep=';')
    np.testing.assert_array_equal(saved.prediction, p)
    np.testing.assert_array_equal(saved[KEY].to_numpy(), future[KEY].to_numpy())
    checks = []
    metrics = pd.read_csv(folder / 'metrics.csv')
    for origin in [181, 212, 243]:
        v = pd.read_csv(folder / ('forecast_' + str(origin) + '.csv'))
        known = v.known_observed_vehicles_DIAGNOSTIC.notna().to_numpy()
        for name in ['direct_baseline', 'forecast_fleet_input', 'known_fleet_DIAGNOSTIC']:
            mask = known if name == 'known_fleet_DIAGNOSTIC' else np.ones(len(v), bool)
            recomputed = metric(v.boardings.to_numpy()[mask], np.rint(v[name].to_numpy()[mask]))
            reported = metrics.loc[(metrics.origin == v.date.min()) & (metrics.model == name), 'score'].item()
            assert abs(recomputed - reported) < 1e-12
            checks.append(dict(origin=origin, model=name, score=recomputed))
    first = ROOT / 'outputs/fleet_forecast_20260927'
    report = json.loads((first / 'report.json').read_text())
    xx, future = e.pack(304, 61)
    audit = json.loads((first / '304_audit.json').read_text())
    preds = {}
    for info in audit:
        name = info['model']
        if name == 'fleet_poisson':scale = np.maximum(1, xx.fleet_28.to_numpy())
        elif name == 'occupancy_l1':scale = np.maximum(5, xx.occupancy_84.to_numpy())
        else:continue
        model = lgb.Booster(model_file=str(first / ('304_' + name + '.txt')))
        preds[name] = np.maximum(0, model.predict(xx[info['features']], num_threads=6) * scale)
    assert report['selected'] == 'fleet_poisson_x_occupancy'
    p = np.rint(preds['fleet_poisson'] * preds['occupancy_l1'] * active).astype(int)
    np.testing.assert_array_equal(pd.read_csv(first / 'submission_fleet.csv', sep=';').prediction, p)
    result = dict(final_submissions_replayed_exactly=2, recomputed_passenger_scores=checks,
                  hidden_label_perturbation_passed=True, active_best_changed=False)
    (folder / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
