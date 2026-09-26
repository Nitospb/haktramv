"""Forecast-origin learning with route-hour weather, POI interactions, and lagged health."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from feature_models import DEST as V4, features
from service_models import normal_profile,adjust,KEY
from train_model import score

ROOT=Path(__file__).resolve().parent;DEST=V4.parent/'v6';DEST.mkdir(exist_ok=True)
HEALTH=pd.read_csv(ROOT/'data/enriched/health_daily.csv.gz')
HC=[c for c in HEALTH if c not in ['route','date']]

def build():
    d=pd.read_csv(V4/'training_matrix.csv')
    h=pd.read_csv(ROOT/'data/enriched/hourly_weather_by_route.csv.gz')
    d=d.merge(h,on=KEY,how='left',validate='one_to_one')
    src=ROOT/'data/features/06_schedules/gtfs_hourly_features_poi.csv'
    header=pd.read_csv(src,nrows=0).columns
    selected=[c for c in header if c.startswith('scheduled_poi_') and any(k in c for k in ['gravity_1500m_mean','within_500m_mean','nearest_m_mean'])]
    p=pd.read_csv(src,usecols=KEY+selected);p=p[p.date<='2025-12-31']
    assert not p.duplicated(KEY).any()
    d=d.merge(p,on=KEY,how='left',validate='one_to_one').copy()
    for cat in ['school','university','mall','metro','venue']:
        g=d['scheduled_poi_'+cat+'_gravity_1500m_mean'].fillna(0)
        d['interaction_'+cat+'_rain']=g*d.hourly_precipitation
        d['interaction_'+cat+'_commute']=g*d.hour.between(7,9)*d.workday
        d['interaction_'+cat+'_weekend']=g*(1-d.workday)
        if cat=='school':d['interaction_school_active']=g*d.edu_school_term*d.workday
        if cat=='university':d['interaction_university_active']=g*d.edu_uni_teaching_proxy*d.workday
    for c in d:
        if c not in ['date','boardings']:d[c]=pd.to_numeric(d[c],errors='raise').fillna(-1)
    assert len(d)==87600
    d.to_csv(DEST/'rich_matrix.csv.gz',index=False)
    return d

def origin(history,future):
    v=future.copy().reset_index(drop=True);cutoff=int(history.time.max())
    v['horizon']=v.time-cutoff;v['origin_time']=cutoff
    v['origin_month']=int(history.loc[history.time==cutoff,'month'].iloc[0])
    v['baseline_service']=adjust(history,v,normal_profile(history,history,.5),normal_profile(history,v,.5))
    h=history.copy();h['level']=h.boardings/np.sqrt(h.external_season)
    keys=['route','daytype','hour']
    for window in [14,28,56,112]:
        past=h[h.time>cutoff-window]
        z=past.groupby(keys)['level'].agg(['mean','median']).rename(columns={c:'history_%s_%s'%(window,c) for c in ['mean','median']})
        v=v.merge(z,on=keys,how='left')
    v['recent_change']=(v.history_14_mean+20)/(v.history_112_mean+20)
    # Seven-day embargo protects against delayed event ingestion and timestamp uncertainty.
    end=pd.Timestamp('2025-01-01')+pd.Timedelta(days=cutoff-6)
    for window in [14,28,84]:
        start=end-pd.Timedelta(days=window)
        z=HEALTH[(HEALTH.date>=start.strftime('%Y-%m-%d'))&(HEALTH.date<end.strftime('%Y-%m-%d'))].groupby('route')[HC].mean()
        z=z.rename(columns={c:'health_%s_%s'%(window,c) for c in HC})
        v=v.merge(z,on='route',how='left')
    v['interaction_drone_offline']=v.drone_report_in_past_6h*v.health_28_offline_delay_share_2h
    v['interaction_mobile_failure']=v.mobile_internet_restrictions_possible*v.health_28_peer_failure_rate
    v['scale']=np.maximum(30,v.baseline_service)
    return v.fillna(-1)

def training(d,cutoff):
    hist=d[d.date<cutoff];parts=[]
    for date in pd.date_range('2025-02-15',pd.Timestamp(cutoff)-pd.Timedelta(days=14),freq='21D'):
        before=hist[hist.date<date.strftime('%Y-%m-%d')]
        future=hist[(hist.date>=date.strftime('%Y-%m-%d'))&(hist.date<(date+pd.Timedelta(days=61)).strftime('%Y-%m-%d'))]
        parts.append(origin(before,future))
    return pd.concat(parts,ignore_index=True)

def cols_for(x,scope):
    cols=features(x,'full')
    if scope=='control':cols=[c for c in cols if not c.startswith(('hourly_','scheduled_poi_','interaction_','health_'))]
    if scope=='weather_poi':cols=[c for c in cols if not c.startswith('health_') and c not in ['interaction_drone_offline','interaction_mobile_failure']]
    return cols

def fit(x,v,scope,depth,save=None):
    cols=cols_for(x,scope)
    model=CatBoostRegressor(iterations=1300,depth=depth,learning_rate=.045,loss_function='MAE',
        l2_leaf_reg=25,thread_count=12,random_seed=2037,verbose=False,allow_writing_files=False)
    weight=x.scale.to_numpy()*.5**((x.origin_time.max()-x.origin_time.to_numpy())/180)
    model.fit(x[cols],x.boardings/x.scale,cat_features=['route','routehour'],sample_weight=weight)
    p=np.maximum(0,model.predict(v[cols])*v.scale.to_numpy());p[v.route.to_numpy()==5]=0
    if save:model.save_model(str(save));Path(str(save)+'.features.json').write_text(json.dumps(cols))
    return p

def main():
    d=build();rows=[];folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    configs=[('control',6),('weather_poi',6),('all',6),('all',8)]
    for start,end in folds:
        tr=d[d.date<start];v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True)
        x=training(d,start);z=origin(tr,v)
        print('Origin samples',start,x.shape,flush=True)
        # Perturbing future labels must not modify forecast features.
        changed=v.copy();changed['boardings']=999999
        check=origin(tr,changed)
        assert z.drop(columns='boardings').equals(check.drop(columns='boardings'))
        ps={}
        for scope,depth in configs:
            name='%s_%s'%(scope,depth);cache=DEST/(start+'_'+name+'.npy')
            if cache.exists():p=np.load(cache)
            else:p=fit(x,z,scope,depth);np.save(cache,p)
            ps[name]=p;rows.append(dict(fold=start,model=name,score=score(v.boardings,p)))
            print(rows[-1],flush=True);pd.DataFrame(rows).to_csv(DEST/'metrics.csv',index=False)
        np.savez(DEST/(start+'.npz'),**ps)
    r=pd.DataFrame(rows);rank=r[r.fold.isin(['2025-07-01','2025-08-01'])].groupby('model').score.mean().sort_values(ascending=False)
    selected=rank.index[0];scope,depth=selected.rsplit('_',1)
    report={'selected':selected,'scores':r[r.model==selected].to_dict('records'),'selection':'Jul-Aug and Aug-Sep; retrospective supplied events/weather; previous exploration means no new independent two-month holdout.','target_95_reached':bool((r[r.model==selected].score>=.95).all()),'health_embargo_days':7,'future_target_invariance_test':'passed'}
    (DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    tr=d[d.date<'2025-11-01'];v=d[d.date>='2025-11-01'].reset_index(drop=True);x=training(d,'2025-11-01');z=origin(tr,v)
    p=fit(x,z,scope,int(depth),DEST/'model.cbm');out=v[KEY].copy();out['prediction']=np.maximum(0,np.rint(p)).astype(int)
    assert len(out)==14640 and not out.duplicated(KEY).any()
    out.to_csv(DEST/'submission.csv',sep=';',index=False);print('DONE V6',flush=True)

if __name__=='__main__':main()
