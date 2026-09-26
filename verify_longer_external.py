"""Verify continuation checkpoints, recompute WAPE, and replay final CSVs."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from fleet_forecast_study import ROOT, KEY, metric
from all_external_study import DATA


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--moscow-only',action='store_true');args=parser.parse_args()
    name='moscow' if args.moscow_only else 'all'
    out=ROOT/('outputs/longer_'+name+'_external_20260927')
    report=json.loads((out/'report.json').read_text());cols=report['features']
    base=ROOT/('outputs/all_moscow_external_20260927' if args.moscow_only else 'outputs/all_external_20260927')
    d=pd.read_parquet(DATA/'all_features_matrix.parquet')
    d[cols]=d[cols].astype(float).replace([np.inf,-np.inf],np.nan).fillna(-1)
    curves=pd.read_csv(out/'learning_curves.csv');replayed=0;checked=0
    for origin in [181,212,243,304]:
        v=d.iloc[origin*240:(origin+61)*240];x=v[cols].copy()
        active=np.clip(1-.975*v.service_cancelled.to_numpy(),0,1)*(v.route.to_numpy()!=5)
        models={n:lgb.Booster(model_file=str(out/(str(origin)+'_'+n+'.txt'))) for n in ['all_direct','fleet','all_fleet']}
        saved=pd.read_csv(out/('forecast_'+str(origin)+'.csv'))
        for n in ([800,1600,3200] if origin<304 else [None]):
            xx=x.copy();xx['vehicle_count']=np.maximum(0,models['fleet'].predict(x,num_iteration=n,num_threads=6))
            for name,model,features in [('direct',models['all_direct'],x),('fleet',models['all_fleet'],xx)]:
                pred=np.maximum(0,model.predict(features,num_iteration=n,num_threads=6))*active
                col=name+'_'+str(n) if n else name
                np.testing.assert_allclose(pred,saved[col],rtol=1e-12,atol=1e-9);replayed+=1
                if origin<304:
                    row=curves[(curves.origin==v.date.min())&(curves.model==name)&(curves.trees==n)].iloc[0]
                    assert abs(metric(saved.boardings,np.rint(pred))-row.validation_score)<1e-12;checked+=1
                    if n==800:
                        prior=pd.read_csv(base/('forecast_'+str(origin)+'.csv'))
                        np.testing.assert_allclose(pred,prior['all_'+name],rtol=1e-12,atol=1e-9)
                else:
                    path=out/('submission_longer_'+name+'.csv');sub=pd.read_csv(path,sep=';')
                    pd.testing.assert_frame_equal(sub[KEY],v[KEY].reset_index(drop=True),check_dtype=False)
                    np.testing.assert_array_equal(sub.prediction,np.rint(pred).astype(int))
                    assert len(sub)==14640 and not sub.duplicated(KEY).any() and sub.prediction.ge(0).all()
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['submission_hashes'][path.name]
    result=dict(replayed_predictions=replayed,recomputed_validation_scores=checked,first_800_match_previous=True,submission_format='passed')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':
    main()
