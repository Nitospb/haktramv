"""Replay fitted models and audit metric, feature coverage and submission keys."""
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from fleet_forecast_study import FleetStudy, ROOT, KEY, metric
from all_external_study import DATA, OUT


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--moscow-only',action='store_true');args=parser.parse_args()
    if args.moscow_only:
        OUT=ROOT/'outputs/all_moscow_external_20260927'
    d = pd.read_parquet(DATA / 'all_features_matrix.parquet')
    manifest = json.loads((DATA / 'feature_manifest.json').read_text())
    report = json.loads((OUT / 'report.json').read_text())
    cols = report.get('used_features',manifest['features'])
    xx = d[cols].astype(float).replace([np.inf,-np.inf], np.nan).fillna(-1)
    changed = d.copy(); changed['boardings'] = 999983
    pd.testing.assert_frame_equal(xx, changed[cols].astype(float).replace([np.inf,-np.inf], np.nan).fillna(-1))
    assert d.loc[d.date >= '2025-11-01','boardings'].isna().all()
    metrics = pd.read_csv(OUT / 'metrics.csv'); report = json.loads((OUT / 'report.json').read_text())
    e = FleetStudy(); replays=0; checked=0
    for origin in [181,212,243,304]:
        frame = d.iloc[origin*240:(origin+61)*240]
        x = xx.iloc[origin*240:(origin+61)*240].copy()
        active = np.clip(1-.975*frame.service_cancelled.to_numpy(),0,1)*(frame.route.to_numpy()!=5)
        saved = pd.read_csv(OUT / ('forecast_'+str(origin)+'.csv'))
        pd.testing.assert_frame_equal(frame[KEY].reset_index(drop=True),saved[KEY],check_dtype=False)
        for name in ['all_direct','without_road','without_uav_connectivity','fleet','all_fleet']:
            model = lgb.Booster(model_file=str(OUT / (str(origin)+'_'+name+'.txt')))
            features = model.feature_name()
            assert not set(features) & {'boardings','date'}
            if name=='all_fleet':
                fleet = lgb.Booster(model_file=str(OUT / (str(origin)+'_fleet.txt')))
                x['vehicle_count'] = np.maximum(0,fleet.predict(x[fleet.feature_name()],num_threads=6))
            pred = np.maximum(0,model.predict(x[features],num_threads=6))*active
            saved_name = 'predicted_observed_vehicles' if name=='fleet' else name
            np.testing.assert_allclose(pred,saved[saved_name],rtol=1e-12,atol=1e-9)
            replays += 1
        if origin < 304:
            finite = np.isfinite(e.f[origin:origin+61].reshape(-1))
            for _, row in metrics[metrics.origin==frame.date.min()].iterrows():
                subset = row.model.endswith('_known_subset') or row.model=='known_fleet_DIAGNOSTIC'
                name = row.model.removesuffix('_known_subset')
                mask = finite if subset else np.ones(len(saved),bool)
                s = metric(saved.boardings.to_numpy()[mask],np.rint(saved[name].to_numpy()[mask]))
                assert abs(s-row.score)<1e-12
                checked+=1
        else:
            for name in ['all_direct','all_fleet']:
                path=OUT/('submission_'+name+'.csv'); sub=pd.read_csv(path,sep=';')
                pd.testing.assert_frame_equal(sub[KEY],saved[KEY],check_dtype=False)
                np.testing.assert_array_equal(sub.prediction,np.rint(saved[name]).astype(int))
                assert list(sub)==KEY+['prediction'] and len(sub)==14640 and not sub.duplicated(KEY).any()
                assert sub.prediction.ge(0).all() and pd.api.types.is_integer_dtype(sub.prediction)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==report['submissions'][name]['sha256']
    weather = pd.read_parquet(DATA / 'route_hour_weather_2025.parquet')
    assert len(weather)==87600 and not weather.isna().any().any()
    for c in ['drone_report_count','route_weather_temperature_2m','road_crash_700m_hour']:
        assert c in cols and d.loc[d.date>='2025-11-01',c].gt(0).any()
    result=dict(model_replays=replays,recomputed_scores=checked,target_exclusion_check='passed',
                full_year_weather_complete=True,submission_format='passed',future_event_features_present=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    main()
