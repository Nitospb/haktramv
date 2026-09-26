"""Studio-only paired evaluation of complete retrospective external inputs."""
import hashlib
import json
import socket
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from fleet_forecast_study import FleetStudy, ROOT, KEY, metric
from fleet_feature_study import fit
from forecast_artifacts import artifact_directory

DATA = ROOT / 'data/complete_external_20260927'
OUT = ROOT / 'outputs/all_external_20260927'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--moscow-only', action='store_true')
    args=parser.parse_args()
    if args.moscow_only:
        OUT = ROOT / 'outputs/all_moscow_external_20260927'
    OUT.mkdir(parents=True, exist_ok=True)
    e = FleetStudy()
    d = pd.read_parquet(DATA / 'all_features_matrix.parquet')
    manifest = json.loads((DATA / 'feature_manifest.json').read_text())
    cols = manifest['features']
    excluded=[]
    if args.moscow_only:
        excluded=[c for c in cols if c.startswith('donor_') or c in ['time','day','month','sin_year','cos_year','external_season','days_to_year_boundary']]
        cols=[c for c in cols if c not in excluded]
    pd.testing.assert_frame_equal(d[KEY + ['boardings']], e.d[KEY + ['boardings']], check_dtype=False)
    d[cols] = d[cols].astype(float).replace([np.inf,-np.inf], np.nan).fillna(-1)
    assert 'boardings' not in cols and 'date' not in cols
    road_cols = [c for c in cols if c.startswith('road_crash_')]
    event_cols = [c for c in cols if c.startswith(('drone_', 'reported_drone_', 'airport_', 'connectivity_', 'mobile_internet_', 'whitelist_'))]
    results = []; audit = []; submissions = {}
    for origin in [181,212,243,304]:
        hist = d.iloc[:origin*240].copy(); future = d.iloc[origin*240:(origin+61)*240].copy()
        x, xx = hist[cols].copy(), future[cols].copy()
        f, fv = e.f[:origin].reshape(-1), e.f[origin:origin+61].reshape(-1)
        good = (hist.route != 5).to_numpy() & np.isfinite(f) & (hist.service_cancelled.to_numpy() < .5)
        w = np.exp2(-(origin - 1 - hist.time.to_numpy())/365)
        w *= np.where(hist.month.isin([1,2]),.35,np.where(hist.month.isin([6,7,8]),.5,1.))
        active = np.clip(1 - .975*future.service_cancelled.to_numpy(),0,1) * (future.route.to_numpy()!=5)
        out = future[KEY + ['boardings']].copy()
        # All three ablations have identical targets/rows/weights/hyperparameters.
        for name, selected in [('all_direct',cols),('without_road', [c for c in cols if c not in road_cols]),
                               ('without_uav_connectivity', [c for c in cols if c not in event_cols])]:
            m = fit(x.loc[good, selected],hist.boardings.to_numpy()[good],w[good],OUT / (str(origin)+'_'+name+'.txt'))
            out[name] = np.maximum(0,m.predict(xx[selected]))*active
            print('FIT',origin,name,flush=True)
        fm = fit(x.loc[good],f[good],w[good],OUT / (str(origin)+'_fleet.txt'))
        fp = np.maximum(0,fm.predict(xx)); out['predicted_observed_vehicles'] = fp*active
        xb, xxb = x.copy(),xx.copy(); xb['vehicle_count'] = f; xxb['vehicle_count'] = fp
        m = fit(xb.loc[good],hist.boardings.to_numpy()[good],w[good],OUT / (str(origin)+'_all_fleet.txt'))
        out['all_fleet'] = np.maximum(0,m.predict(xxb))*active
        known = np.isfinite(fv)
        if origin < 304:
            oracle = xxb.copy(); oracle['vehicle_count'] = fv
            out['known_fleet_DIAGNOSTIC'] = np.maximum(0,m.predict(oracle))*active
            # Previous study controls, same origin and fitting configuration.
            for name in ['direct_baseline','forecast_fleet_input']:
                old = pd.read_csv(ROOT / 'outputs/fleet_feature_20260927' / ('forecast_'+str(origin)+'.csv'))
                out['previous_'+name] = future[KEY].merge(old[KEY+[name]],on=KEY,validate='one_to_one')[name].to_numpy()
            ref = pd.read_csv(artifact_directory('v8') / ('validation_'+future.date.min()+'.csv'),sep=';')
            out['confirmed_v8_up3'] = future[KEY].merge(ref[KEY+['prediction']],on=KEY,validate='one_to_one').prediction.to_numpy()*1.03
            for name in [c for c in out if c not in KEY+['boardings','predicted_observed_vehicles']]:
                mask = known if name == 'known_fleet_DIAGNOSTIC' else np.ones(len(out),bool)
                results.append(dict(origin=future.date.min(),model=name,rows=int(mask.sum()),score=metric(out.boardings.to_numpy()[mask],np.rint(out[name].to_numpy()[mask]))))
            # Matched oracle subset, so missing vehicle identifiers cannot improve comparison.
            for name in ['all_direct','all_fleet']:
                results.append(dict(origin=future.date.min(),model=name+'_known_subset',rows=int(known.sum()),score=metric(out.boardings.to_numpy()[known],np.rint(out[name].to_numpy()[known]))))
            print('SCORES',results[-10:],flush=True)
        else:
            for name in ['all_direct','all_fleet']:
                sub = out[KEY].copy(); sub['route'] = sub.route.astype(int); sub['hour'] = sub.hour.astype(int)
                sub['prediction'] = np.rint(out[name]).astype(int)
                path = OUT / ('submission_'+name+'.csv'); sub.to_csv(path,sep=';',index=False)
                assert len(sub)==14640 and not sub.duplicated(KEY).any() and sub.prediction.ge(0).all()
                submissions[name] = dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),total=int(sub.prediction.sum()),public_score=None)
        out.to_csv(OUT / ('forecast_'+str(origin)+'.csv'),index=False)
        pd.DataFrame(results).to_csv(OUT / 'metrics.csv',index=False)
        audit.append(dict(origin=origin,train_max_date=hist.date.max(),forecast_min_date=future.date.min(),rows=int(good.sum()),feature_count=len(cols)))
    report = dict(host=socket.gethostname(),features=len(cols),used_features=cols,excluded_features=excluded,
                  feature_set='Local Moscow external features without numeric date/trend or transport donor indices/profiles (including Moscow monthly aggregates)' if args.moscow_only else 'All numeric external features',
                  metrics=results,submissions=submissions,fit_audit=audit,
                  weather_uav_connectivity_and_service='Actual retrospective external values supplied on both train and prediction dates.',
                  road='Police-reported crash proximity, not actual traffic flow. Independent ablation included.',
                  target='Hourly boardings, 61-day fixed-origin horizon.',
                  fleet='Observed train fleet; predicted future fleet. True future fleet unavailable. Known-fleet diagnostic is historical only.',
                  caveats=['Overlapping previously explored validation periods, not independent leaderboard.',
                           'Fleet model sees actual train vehicle counts while inference uses predictions.',
                           'Route 5 has no positive train labels; zero here. Separate public route-5 aggregate probe already available.',
                           'Source zero report counts do not prove no event; missing numeric fields use -1.'],
                  active_model_changed=False,confirmed_public_best=.89402)
    (OUT / 'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('DONE',flush=True)


if __name__ == '__main__':
    main()
