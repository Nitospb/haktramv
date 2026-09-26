"""Explicit service regimes: normal demand, empirical diversion, novel cancellation prior.

Retrospective service calendars are supplied covariates, not forecast-origin-known events.
No future boarding values are used in the regime adjustment.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from train_model import score
from feature_models import DEST as PREVIOUS, features, base

DEST=PREVIOUS.parent/'v5';DEST.mkdir(exist_ok=True)
KEY=['route','date','hour']

def regime(d):
    return np.select([d.service_cancelled>0,d.service_rerouted>0,d.service_reduced_frequency>0],[3,2,1],default=0)

def normal_profile(tr,va,power=1,window=365):
    normal=tr[(regime(tr)==0)&(tr.service_incident_fraction==0)].copy()
    normal['value']=normal.boardings/normal.external_season**power
    keys=['route','daytype','hour']
    normal=normal[normal.time>tr.time.max()-window]
    stats=normal.groupby(keys).value.median().rename('p')
    out=va.merge(stats,on=keys,how='left').p
    fallback=base(tr,va,power)/va.external_season.to_numpy()**power
    return out.fillna(pd.Series(fallback)).to_numpy()*va.external_season.to_numpy()**power

def adjust(tr,va,train_normal_pred,future_normal_pred,prior=2500):
    # Pooled and time-band estimates: shrink sparse bands to route/day-regime factor.
    t=tr.copy();v=va.copy();t['mode']=regime(t);v['mode']=regime(v)
    t['normal']=np.maximum(0,train_normal_pred);t['band']=t.hour//4;v['band']=v.hour//4
    # Stop partial-event hours from contaminating the full-event multiplier.
    fraction=np.select([t['mode']==3,t['mode']==2,t['mode']==1],
        [t.service_cancelled,t.service_rerouted,t.service_reduced_frequency],default=0)
    t=t[(t['mode']>0)&(fraction>=.95)&(t.normal>=20)]
    keys=['route','workday','mode']
    pool=t.groupby(keys)[['boardings','normal']].sum()
    pool['prior']=pool.index.get_level_values('mode').map({1:.8,2:.7,3:.025})
    pool['factor']=(pool.boardings+prior*pool.prior)/(pool.normal+prior)
    bands=t.groupby(keys+['band'])[['boardings','normal']].sum().reset_index().merge(pool.factor,on=keys,how='left')
    bands['factor_band']=(bands.boardings+prior*bands.factor)/(bands.normal+prior)
    v=v.merge(pool.factor,on=keys,how='left').merge(bands[keys+['band','factor_band']],on=keys+['band'],how='left')
    factor=v.factor_band.fillna(v.factor).fillna(v['mode'].map({0:1,1:.8,2:.7,3:.025})).to_numpy()
    factor=np.clip(factor,.0,2)
    fraction=np.select([v['mode']==3,v['mode']==2,v['mode']==1],
        [v.service_cancelled,v.service_rerouted,v.service_reduced_frequency],default=0)
    return np.maximum(0,np.asarray(future_normal_pred)*(1-fraction+fraction*factor))

def fit(tr,va,depth=6,group='full',save=None):
    normal=tr[(regime(tr)==0)&(tr.service_incident_fraction==0)].copy()
    tr=tr.copy();va=va.copy()
    normal['scale']=np.maximum(30,normal_profile(tr,normal))
    tr['scale']=np.maximum(30,normal_profile(tr,tr));va['scale']=np.maximum(30,normal_profile(tr,va))
    cols=features(normal,group)
    cols=[c for c in cols if not c.startswith('service_')]
    if 'scale' not in cols:cols.append('scale')
    model=CatBoostRegressor(iterations=1500,depth=depth,learning_rate=.04,loss_function='MAE',
        l2_leaf_reg=15,thread_count=12,random_seed=2031,verbose=False,allow_writing_files=False)
    weights=normal.scale.to_numpy()*.5**((normal.time.max()-normal.time.to_numpy())/240)
    model.fit(normal[cols],normal.boardings/normal.scale,cat_features=['route','routehour'],sample_weight=weights)
    tp=np.maximum(0,model.predict(tr[cols])*tr.scale.to_numpy())
    vp=np.maximum(0,model.predict(va[cols])*va.scale.to_numpy())
    out=adjust(tr,va,tp,vp);out[va.route.to_numpy()==5]=0
    if save:
        model.save_model(str(save));Path(str(save)+'.features.json').write_text(json.dumps(cols))
    return out

def main():
    d=pd.read_csv(PREVIOUS/'training_matrix.csv');rows=[];frames={};preds={}
    folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    for start,end in folds:
        tr=d[d.date<start].reset_index(drop=True);v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);frames[start]=v
        old=pd.read_csv(PREVIOUS/('ensemble_validation_'+start+'.csv'),sep=';').prediction.to_numpy()
        ps={'v4':old,'v4_cancel_prior':old*(1-.975*v.service_cancelled.to_numpy())}
        for power in [.5,1]:
            for window in [112,365]:
                a=normal_profile(tr,tr,power,window);b=normal_profile(tr,v,power,window)
                ps['profile_%s_%s'%(power,window)]=adjust(tr,v,a,b)
        for depth,group in [(6,'full'),(8,'full'),(6,'weather')]:
            name='cat_%s_%s'%(depth,group);cache=DEST/(start+'_'+name+'.npy')
            if cache.exists():p=np.load(cache)
            else:p=fit(tr,v,depth,group);np.save(cache,p)
            ps[name]=p;print(start,name,score(v.boardings,p),flush=True)
        for n,p in ps.items():rows.append(dict(fold=start,model=n,score=score(v.boardings,p)))
        preds[start]=ps;np.savez(DEST/(start+'.npz'),**ps)
        pd.DataFrame(rows).to_csv(DEST/'metrics.csv',index=False)
        print(pd.DataFrame(rows).query('fold==@start').sort_values('score',ascending=False).to_string(index=False),flush=True)
    r=pd.DataFrame(rows);rank=r[r.fold.isin(['2025-07-01','2025-08-01'])].groupby('model').score.mean().sort_values(ascending=False)
    selected=rank.index[0]
    report={'selected':selected,'scores':r[r.model==selected].to_dict('records'),'selection':'Jul-Aug and Aug-Sep; retrospective provided service/weather. Other periods already examined; not a new independent test.','novel_cancellation_residual_prior':.025,'target_95_reached':bool((r[r.model==selected].score>=.95).all())}
    (DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    tr=d[d.date<'2025-11-01'].reset_index(drop=True);v=d[d.date>='2025-11-01'].reset_index(drop=True)
    if selected.startswith('cat'):
        _,depth,group=selected.split('_');p=fit(tr,v,int(depth),group,DEST/'model.cbm')
    elif selected.startswith('profile'):
        _,power,window=selected.split('_');p=adjust(tr,v,normal_profile(tr,tr,float(power),int(window)),normal_profile(tr,v,float(power),int(window)))
    else:
        p=pd.read_csv(PREVIOUS/'submission_ensemble.csv',sep=';').prediction.to_numpy()
        if selected=='v4_cancel_prior':p=p*(1-.975*v.service_cancelled.to_numpy())
    sub=v[KEY].copy();sub['prediction']=np.maximum(0,np.rint(p)).astype(int)
    assert len(sub)==14640 and not sub.duplicated(KEY).any()
    sub.to_csv(DEST/'submission.csv',sep=';',index=False)
    print('DONE V5',flush=True)

if __name__=='__main__':main()
