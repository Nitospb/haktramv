"""Models using supplied exogenous features. Retrospective-weather evaluation is explicit."""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from train_model import load,calendar,DATA,OUT,score
from research_v2 import adaptive

ROOT=Path(__file__).resolve().parent
F=ROOT/'data/features';DEST=OUT/'v4';DEST.mkdir(exist_ok=True)
KEY=['route','date','hour']

def build():
    d=load();future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
    d=pd.concat([d,future],ignore_index=True)
    sources=[('03_weather_and_calendar/weather_calendar_extended.csv',['date']),
        ('04_education/education_calendar_2025.csv',['date']),
        ('02_repairs_and_service/route_hour_features_full_calendar.csv',KEY),
        ('01_connectivity_and_uav/route_hour_disruption_features_2025.csv',KEY),
        ('06_schedules/gtfs_hourly_features.csv',KEY),
        ('05_stops_and_nearby_objects/route_transfer_features.csv',['route'])]
    audit=[]
    for file,keys in sources:
        x=pd.read_csv(F/file)
        if 'date' in x:x=x[x.date.between('2025-01-01','2025-12-31')]
        assert not x.duplicated(keys).any(),file
        cols=[c for c in x if c not in d.columns or c in keys]
        d=d.merge(x[cols],on=keys,how='left',validate='many_to_one')
        audit.append({'source':file,'rows':len(x),'added_columns':len(cols)-len(keys)})
    s=json.loads((F/'08_external_seasonality/moscow_seasonality_index.json').read_text())['monthly_index']
    d['external_season']=d.month.map({int(k):v for k,v in s.items()})
    d['routehour']=d.route*24+d.hour
    for c in d:
        if c not in ['date','boardings']:
            d[c]=pd.to_numeric(d[c],errors='raise').fillna(-1)
    assert len(d)==87600 and not d.duplicated(KEY).any()
    (DEST/'source_audit.json').write_text(json.dumps({'sources':audit,'evaluation':'Historical weather and retrospective events supplied by user; not an as-of-Oct31 operational forecast. No contemporaneous payment-derived variables used.','rows':len(d)},indent=2))
    d.to_csv(DEST/'training_matrix.csv',index=False)
    return d

def base(tr,va,power,window=365,adjust=0):
    tr=tr.copy();tr['normalized']=tr.boardings/(tr.external_season**power)
    subset=tr[tr.time>tr.time.max()-window]
    keys=['route','daytype','hour']
    stats=subset.groupby(keys).normalized.median().rename('base')
    p=va.merge(stats,on=keys,how='left').base.fillna(0).to_numpy()*va.external_season.to_numpy()**power
    if adjust:
        recent=tr[tr.time>tr.time.max()-28].copy()
        rp=recent.merge(stats,on=keys,how='left').base.fillna(0).to_numpy()*recent.external_season.to_numpy()**power
        recent['expected']=rp
        z=recent.groupby('route')[['boardings','expected']].sum()
        ratio=((z.boardings+200)/(z.expected+200)).clip(.5,1.5)**adjust
        p*=va.route.map(ratio).to_numpy()
    return p

def features(d,group):
    drop=set(['date','boardings','time','month','day','sin_year','cos_year','external_season'])
    if group=='calendar':
        allowed=['route','hour','daytype','dow','holiday','workday','summer','routehour']
        allowed+=[c for c in d if c.startswith(('is_','edu_','days_','new_year','consecutive_','day_number')) and c not in ['is_rain','is_snow','is_precipitation','is_heavy_rain','is_heavy_snow','is_freezing','is_hot']]
        return list(dict.fromkeys(allowed))
    cols=[c for c in d if c not in drop]
    if group=='weather':cols=[c for c in cols if not c.startswith(('service_','scheduled_','route_variant','route_total','route_stop','route_length','xfer_','drone_','reported_','airport_','connectivity_','mobile_','whitelist_'))]
    return cols

CONFIGS=[('calendar',6,1,'MAE'),('weather',6,1,'MAE'),('full',6,1,'MAE'),('full',8,1,'MAE'),('full',8,0,'MAE'),('full',8,.5,'MAE'),('full',8,1,'RMSE'),('weather',8,1,'MAE')]

def fit(tr,va,config,save=None):
    group,depth,power,loss=config
    x=tr.copy();v=va.copy()
    # Training-only cell template; never computed using validation targets.
    x['scale']=np.maximum(30,base(tr,tr,power));v['scale']=np.maximum(30,base(tr,va,power))
    cols=features(x,group)
    if 'scale' not in cols:cols.append('scale')
    model=CatBoostRegressor(iterations=1600,depth=depth,learning_rate=.04,loss_function=loss,
        l2_leaf_reg=15,thread_count=12,random_seed=2026,verbose=False,allow_writing_files=False)
    weights=x.scale.to_numpy()*.5**((x.time.max()-x.time.to_numpy())/240)
    model.fit(x[cols],x.boardings/x.scale,cat_features=['route','routehour'],sample_weight=weights)
    p=np.maximum(0,model.predict(v[cols])*v.scale.to_numpy());p[v.route.to_numpy()==5]=0
    if save:
        model.save_model(str(save));Path(str(save)+'.features.json').write_text(json.dumps(cols))
    return p

def main():
    d=build();metrics=[];preds={};frames={};configs={}
    folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    for start,end in folds:
        tr=d[d.date<start];v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);frames[start]=v
        ps={'v2':adaptive(tr,v,28,.5)}
        for power in [.5,1,1.5]:
            for window in [56,112,365]:
                for adjust in [0,.5]:
                    name='base_%s_%s_%s'%(power,window,adjust);configs[name]=('base',power,window,adjust)
                    ps[name]=base(tr,v,power,window,adjust)
        for conf in CONFIGS:
            name='cat_'+'_'.join(map(str,conf));configs[name]=('cat',conf)
            cache=DEST/(start+'_'+name+'.npy')
            if cache.exists():p=np.load(cache)
            else:p=fit(tr,v,conf);np.save(cache,p)
            ps[name]=p;print(start,name,score(v.boardings,p),flush=True)
        preds[start]=ps
        for name,p in ps.items():metrics.append({'fold':start,'model':name,'score':score(v.boardings,p)})
        np.savez(DEST/(start+'.npz'),**ps)
        pd.DataFrame(metrics).to_csv(DEST/'metrics.csv',index=False)
        print(pd.DataFrame(metrics).query('fold==@start').sort_values('score',ascending=False).head(5).to_string(index=False),flush=True)
    # Earlier two-month folds choose candidate; Sep-Oct already explored in previous work.
    r=pd.DataFrame(metrics);ranking=r[r.fold.isin(['2025-07-01','2025-08-01'])].groupby('model').score.mean().sort_values(ascending=False)
    selected=ranking.index[0]
    report={'selected':selected,'scores':r[r.model==selected].to_dict('records'),'selection_folds':['Jul-Aug','Aug-Sep'],'caveat':'Supplied retrospective exogenous data; not operational as-of forecast. Sep-Oct previously explored, October only one month.','target_achieved':bool((r[r.model==selected].score>=.92).all()),'known_best_real_score':.88882,'new_real_score':None}
    (DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    tr=d[d.date<'2025-11-01'];v=d[d.date>='2025-11-01'].copy()
    if selected=='v2':p=adaptive(tr,v,28,.5)
    elif configs[selected][0]=='base':p=base(tr,v,*configs[selected][1:])
    else:p=fit(tr,v,configs[selected][1],DEST/'model.cbm')
    out=v[KEY].copy();out['prediction']=np.rint(np.maximum(0,p)).astype(int)
    assert len(out)==14640 and not out.duplicated(KEY).any()
    out.to_csv(DEST/'submission.csv',sep=';',index=False)
    print('DONE V4',flush=True)

if __name__=='__main__':main()
