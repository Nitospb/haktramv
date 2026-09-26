"""Reproducible two-month tram-demand forecasting, with chronological evaluation."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

DATA = Path(os.environ.get('TRAM_DATA_DIR', Path(__file__).resolve().parent / 'data'))
OUT = Path(__file__).resolve().parent / 'outputs'
OUT.mkdir(exist_ok=True)
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
HOLIDAYS = set(pd.date_range('2025-01-01', '2025-01-08').strftime('%Y-%m-%d')) | {
    '2025-02-23','2025-03-08','2025-05-01','2025-05-02','2025-05-08',
    '2025-05-09','2025-06-12','2025-06-13','2025-11-03','2025-11-04','2025-12-31'}

def calendar(d):
    d = d.copy()
    dt = pd.to_datetime(d.date)
    d['dow'] = dt.dt.dayofweek
    d['holiday'] = dt.dt.strftime('%Y-%m-%d').isin(HOLIDAYS).astype(int)
    d['daytype'] = d.dow.where(d.holiday == 0, 6)
    d.loc[dt == '2025-11-01', 'daytype'] = 4
    d['workday'] = (d.daytype < 5).astype(int)
    d['month'] = dt.dt.month
    d['day'] = dt.dt.day
    d['time'] = (dt - pd.Timestamp('2025-01-01')).dt.days
    d['sin_year'] = np.sin(2*np.pi*d.time/365)
    d['cos_year'] = np.cos(2*np.pi*d.time/365)
    d['summer'] = d.month.isin([6,7,8]).astype(int)
    return d

def load():
    raw = pd.concat([pd.read_csv(DATA/'labels'/('labels_day_'+s+'.csv'), sep=';') for s in ['train','test']])
    assert not raw.duplicated(['route','date','hour']).any()
    grid = pd.MultiIndex.from_product([ROUTES, pd.date_range('2025-01-01','2025-10-31').strftime('%Y-%m-%d'), range(24)], names=['route','date','hour']).to_frame(index=False)
    d = grid.merge(raw, how='left').fillna({'boardings':0})
    return calendar(d)

def score(y,p):
    return float(max(0,1-np.abs(np.asarray(y)-np.maximum(0,p)).sum()/np.sum(y)))

def profile(tr,va,window,kind='median'):
    recent = tr[tr.time > tr.time.max()-window]
    keys = ['route','daytype','hour']
    stats = recent.groupby(keys).boardings.agg(kind)
    return va[keys].merge(stats.rename('p'),on=keys,how='left').p.fillna(0).values

FEATURES = ['route','hour','dow','holiday','daytype','workday','month','day','time','sin_year','cos_year','summer']

def boosted(tr,va,depth,loss,half_life,save=None):
    model = CatBoostRegressor(iterations=1100, depth=depth, learning_rate=.055,
        loss_function=loss, random_seed=42, thread_count=6, verbose=False,
        allow_writing_files=False, l2_leaf_reg=5)
    weights = np.power(.5,(tr.time.max()-tr.time)/half_life)
    model.fit(tr[FEATURES], tr.boardings, sample_weight=weights, cat_features=['route'])
    p = np.maximum(0,model.predict(va[FEATURES]))
    p[va.route.values==5]=0
    if save: model.save_model(str(save))
    return p

def main():
    d=load()
    results=[]; predictions={}; targets={}
    for start,end in [('2025-05-01','2025-06-30'),('2025-07-01','2025-08-31'),('2025-09-01','2025-10-31')]:
        tr=d[d.date<start]; va=d[(d.date>=start)&(d.date<=end)]
        targets[start]=va
        for window in [14,28,56,84,120,365]:
            for kind in ['median','mean']:
                name='profile_%s_%s'%(window,kind)
                p=profile(tr,va,window,kind)
                predictions[start,name]=p
                results.append(dict(fold=start,model=name,score=score(va.boardings,p)))
        for depth,loss,hl in [(6,'MAE',90),(8,'MAE',90),(8,'RMSE',90),(8,'MAE',365)]:
            name='cat_%s_%s_%s'%(depth,loss,hl)
            p=boosted(tr,va,depth,loss,hl)
            predictions[start,name]=p
            results.append(dict(fold=start,model=name,score=score(va.boardings,p)))
            print(start,name,results[-1]['score'],flush=True)
        pd.DataFrame(results).to_csv(OUT/'validation_metrics.csv',index=False)
    # Select on earlier folds, reserving Sep-Oct for final evaluation.
    r=pd.DataFrame(results)
    ranking=r[r.fold!='2025-09-01'].groupby('model').score.mean().sort_values(ascending=False)
    selected=ranking.index[0]
    print('Selection on earlier folds:',ranking.to_string(),flush=True)
    print('Untouched Sep-Oct evaluation:',r[r.fold=='2025-09-01'].sort_values('score',ascending=False).to_string(index=False),flush=True)
    va=targets['2025-09-01'].copy()
    va['prediction']=predictions['2025-09-01',selected]
    va[['route','date','hour','boardings','prediction']].to_csv(OUT/'holdout_predictions.csv',sep=';',index=False)
    future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
    if selected.startswith('profile'):
        _,window,kind=selected.split('_')
        p=profile(d,future,int(window),kind)
        stats=d[d.time>d.time.max()-int(window)].groupby(['route','daytype','hour']).boardings.agg(kind)
        stats.to_csv(OUT/'model_profile.csv',sep=';')
    else:
        _,depth,loss,hl=selected.split('_')
        p=boosted(d,future,int(depth),loss,int(hl),OUT/'model.cbm')
    submission=future[['route','date','hour']].copy()
    submission['prediction']=np.maximum(0,np.rint(p)).astype(int)
    assert len(submission)==14640 and not submission.duplicated(['route','date','hour']).any()
    assert submission.prediction.notna().all() and (submission.prediction>=0).all()
    submission.to_csv(OUT/'submission.csv',sep=';',index=False)
    report=dict(selected_model=selected,selection_folds=['May-Jun','Jul-Aug'],holdout='Sep-Oct',holdout_score=score(va.boardings,va.prediction),target_score=.95,target_achieved=score(va.boardings,va.prediction)>=.95,rows=len(submission))
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
