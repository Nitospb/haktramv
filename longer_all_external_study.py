"""Continue the identical expanded-feature models: 800 -> 1600 -> 3200 trees.

No held-out target enters fitting. Checkpoints are selected on the three
previously explored 61-day development windows, not the public leaderboard.
"""
import hashlib
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from fleet_forecast_study import FleetStudy, ROOT, KEY, metric
from all_external_study import DATA

BASE = ROOT / 'outputs/all_external_20260927'
OUT = ROOT / 'outputs/longer_all_external_20260927'
ROUNDS = [800,1600,3200]


def extend(x,y,w,source,dest,total):
    if total==800:
        shutil.copyfile(source,dest)
        return lgb.Booster(model_file=str(dest))
    model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=total-800,learning_rate=.035,
        num_leaves=31,min_child_samples=100,reg_lambda=15,n_jobs=6,random_state=270927,
        verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(x,y,sample_weight=w,categorical_feature=['route','routehour','daytype'],init_model=str(source))
    model.booster_.save_model(str(dest))
    assert model.booster_.num_trees()==total
    # Continuation must preserve the original 800-tree predictor exactly.
    first=lgb.Booster(model_file=str(source)).predict(x.iloc[:512],num_threads=6)
    np.testing.assert_array_equal(first,model.booster_.predict(x.iloc[:512],num_iteration=800,num_threads=6))
    return model.booster_


def main():
    global BASE, OUT
    parser=argparse.ArgumentParser();parser.add_argument('--moscow-only',action='store_true');args=parser.parse_args()
    if args.moscow_only:
        BASE=ROOT/'outputs/all_moscow_external_20260927'
        OUT=ROOT/'outputs/longer_moscow_external_20260927'
    OUT.mkdir(parents=True,exist_ok=True)
    e=FleetStudy(); d=pd.read_parquet(DATA/'all_features_matrix.parquet')
    cols=json.loads((BASE/'report.json').read_text()).get('used_features',json.loads((DATA/'feature_manifest.json').read_text())['features'])
    d[cols]=d[cols].astype(float).replace([np.inf,-np.inf],np.nan).fillna(-1)
    records=[];fit_audit=[]
    for origin in [181,212,243,304]:
        if origin==304:
            summary=pd.DataFrame(records).groupby(['model','trees']).validation_score.mean()
            choices={name:int(summary.loc[name].idxmax()) for name in ['direct','fleet']}
            print('SELECTED',choices,flush=True)
        hist=d.iloc[:origin*240].copy();future=d.iloc[origin*240:(origin+61)*240].copy()
        x,xx=hist[cols].copy(),future[cols].copy()
        y=hist.boardings.to_numpy(); f=e.f[:origin].reshape(-1)
        good=(hist.route.to_numpy()!=5)&np.isfinite(f)&(hist.service_cancelled.to_numpy()<.5)
        w=np.exp2(-(origin-1-hist.time.to_numpy())/365)
        w*=np.where(hist.month.isin([1,2]),.35,np.where(hist.month.isin([6,7,8]),.5,1.))
        active=np.clip(1-.975*future.service_cancelled.to_numpy(),0,1)*(future.route.to_numpy()!=5)
        xb=x.copy();xb['vehicle_count']=f
        models={}
        for name,train,target in [('all_direct',x,y),('fleet',x,f),('all_fleet',xb,y)]:
            total=3200 if origin<304 else choices['direct' if name=='all_direct' else 'fleet']
            source=BASE/(str(origin)+'_'+name+'.txt');dest=OUT/(str(origin)+'_'+name+'.txt')
            models[name]=extend(train.loc[good],target[good],w[good],source,dest,total)
            print('EXTENDED',origin,name,total,flush=True)
        forecast=future[KEY+['boardings']].copy()
        if origin<304:
            for n in ROUNDS:
                p=models['all_direct'].predict(xx,num_iteration=n,num_threads=6)
                predicted_f=np.maximum(0,models['fleet'].predict(xx,num_iteration=n,num_threads=6))
                xxb=xx.copy();xxb['vehicle_count']=predicted_f
                pf=models['all_fleet'].predict(xxb,num_iteration=n,num_threads=6)
                for name,pred,train_pred in [
                    ('direct',np.maximum(0,p)*active,models['all_direct'].predict(x.loc[good],num_iteration=n,num_threads=6)),
                    ('fleet',np.maximum(0,pf)*active,models['all_fleet'].predict(xb.loc[good],num_iteration=n,num_threads=6))]:
                    forecast[name+'_'+str(n)]=pred
                    records.append(dict(origin=future.date.min(),model=name,trees=n,
                        validation_score=metric(future.boardings,np.rint(pred)),
                        train_score_observed_inputs=metric(y[good],np.maximum(0,np.rint(train_pred)))))
            prior=pd.read_csv(BASE/('forecast_'+str(origin)+'.csv'))
            for name,prior_name in [('direct','all_direct'),('fleet','all_fleet')]:
                np.testing.assert_allclose(forecast[name+'_800'],prior[prior_name],rtol=1e-12,atol=1e-9)
            print('CURVE',records[-6:],flush=True)
        else:
            for name,n in choices.items():
                if name=='direct':
                    pred=models['all_direct'].predict(xx,num_threads=6)
                else:
                    xxb=xx.copy();xxb['vehicle_count']=np.maximum(0,models['fleet'].predict(xx,num_threads=6))
                    pred=models['all_fleet'].predict(xxb,num_threads=6)
                forecast[name]=np.maximum(0,pred)*active
                sub=forecast[KEY].copy();sub['route']=sub.route.astype(int);sub['hour']=sub.hour.astype(int)
                sub['prediction']=np.rint(forecast[name]).astype(int)
                sub.to_csv(OUT/('submission_longer_'+name+'.csv'),sep=';',index=False)
                assert len(sub)==14640 and sub.prediction.ge(0).all() and not sub.duplicated(KEY).any()
        forecast.to_csv(OUT/('forecast_'+str(origin)+'.csv'),index=False)
        pd.DataFrame(records).to_csv(OUT/'learning_curves.csv',index=False)
        fit_audit.append(dict(origin=origin,train_max_date=hist.date.max(),forecast_min_date=future.date.min(),training_rows=int(good.sum())))
    report=dict(features=cols,base_experiment=str(BASE.relative_to(ROOT)),rounds=ROUNDS,learning_rate=.035,validation_horizon_days=61,
        selection='Maximum mean of three previously explored overlapping development windows, separately for direct and fleet.',
        selected_trees=choices,mean_validation=pd.DataFrame(records).groupby(['model','trees']).validation_score.mean().reset_index().to_dict('records'),
        fit_audit=fit_audit,first_800_predictions_unchanged=True,public_score=None,active_model_changed=False,
        limitations=['Train fleet model uses observed fleet; forecast uses predicted fleet, so training and validation inputs differ.',
                    'Training WAPE shown on retained training rows, not the full validation grid.',
                    'Same previously explored cutoffs: checkpoint selection may overfit these development windows.',
                    'Route 5 remains zero; separate public-data cold-start scenario is available.'],
        submission_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('submission*.csv')})
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('DONE',flush=True)


if __name__=='__main__':
    main()
