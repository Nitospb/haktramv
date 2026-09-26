"""Forecast-origin training: every historical example uses only earlier history."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from train_model import load, calendar, DATA, OUT, score, profile

V2=OUT/'v2';V2.mkdir(exist_ok=True)
KEY=['route','daytype','hour']

def origin_features(history, future):
    v=future.copy().reset_index(drop=True)
    cutoff=history.time.max()
    v['horizon']=v.time-cutoff
    v['origin_month']=int(history.loc[history.time==cutoff,'month'].iloc[0])
    v['origin_time']=cutoff
    v['rh']=v.route*24+v.hour
    for window in [14,28,56,112,365]:
        h=history[history.time>cutoff-window]
        for agg in ['median','mean']:
            stats=h.groupby(KEY).boardings.agg(agg).rename('p_%d_%s'%(window,agg))
            v=v.merge(stats,on=KEY,how='left')
    for label,mask in [('school',history.summer==0),('summer',history.summer==1)]:
        h=history[mask & (history.holiday==0)]
        st=h.groupby(KEY).boardings.median().rename('p_'+label)
        v=v.merge(st,on=KEY,how='left')
    v['base']=v.p_28_median
    v['recent_ratio']=(v.p_14_mean+20)/(v.p_112_mean+20)
    return v.fillna(0)

def adaptive(tr,va,window,shrink):
    # Seasonal analog profile; route level adjusted using relative recent changes.
    x=tr.copy();v=va.copy()
    keys=['route','hour','daytype','summer']
    stats=x[x.holiday==0].groupby(keys).boardings.median().rename('p')
    out=v.merge(stats,on=keys,how='left')
    fallback=profile(tr,va,365)
    p=out.p.fillna(pd.Series(fallback)).to_numpy(copy=True)
    recent=x[x.time>x.time.max()-window]
    r=recent.merge(stats,on=keys,how='left')
    ratios=r.groupby('route')[['boardings','p']].sum()
    ratios['ratio']=(ratios.boardings/ratios.p.clip(lower=1)).clip(.5,1.5)
    p*=v.route.map(ratios.ratio).fillna(1).values**shrink
    return p

def make_training(d,cutoff):
    hist=d[d.date<cutoff]
    examples=[]
    # Labels after cutoff are never used in historical training examples.
    for origin in pd.date_range('2025-02-01',pd.Timestamp(cutoff)-pd.Timedelta(days=14),freq='14D'):
        before=hist[hist.date<origin.strftime('%Y-%m-%d')]
        future=hist[(hist.date>=origin.strftime('%Y-%m-%d')) & (hist.date<(origin+pd.Timedelta(days=61)).strftime('%Y-%m-%d'))]
        examples.append(origin_features(before,future))
    return pd.concat(examples,ignore_index=True)

def direct(train,test,mode,depth,save=None):
    features=[c for c in train.columns if c not in ['date','boardings']]
    model=CatBoostRegressor(iterations=1400,depth=depth,learning_rate=.045,
        loss_function='MAE',l2_leaf_reg=15,thread_count=12,random_seed=2026,
        verbose=False,allow_writing_files=False)
    scale=(train.base+100) if mode=='ratio' else np.ones(len(train))
    target=train.boardings/scale
    weights=np.asarray(scale)*(.5**((train.origin_time.max()-train.origin_time)/180))
    model.fit(train[features],target,cat_features=['route','rh'],sample_weight=weights)
    p=model.predict(test[features])
    if mode=='ratio':p*=test.base.to_numpy()+100
    p=np.array(np.maximum(0,p),copy=True);p[test.route.values==5]=0
    if save:model.save_model(str(save))
    return p

def main():
    d=load();metrics=[];preds={};vals={}
    folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    for start,end in folds:
        tr=d[d.date<start];va=d[(d.date>=start)&(d.date<=end)];vals[start]=va
        names=[];ps=[]
        for window in [14,28,56]:
            for shrink in [0,.5,1]:
                name='adaptive_%s_%s'%(window,shrink)
                p=adaptive(tr,va,window,shrink)
                names.append(name);ps.append(p)
        train=make_training(d,start);test=origin_features(tr,va)
        print('Training origin dataset',start,train.shape,flush=True)
        for mode,depth in [('raw',6),('ratio',6),('ratio',8)]:
            name='direct_%s_%s'%(mode,depth)
            cache=V2/(start+'_'+name+'.npy')
            if cache.exists():p=np.load(cache)
            else:
                p=direct(train,test,mode,depth);np.save(cache,p)
            names.append(name);ps.append(p)
            print(start,name,score(va.boardings,p),flush=True)
        for name,p in zip(names,ps):
            metrics.append({'fold':start,'end':end,'model':name,'score':score(va.boardings,p)})
            np.save(V2/(start+'_'+name+'.npy'),p)
        preds[start]=dict(zip(names,ps))
        pd.DataFrame(metrics).to_csv(V2/'metrics.csv',index=False)
        print(pd.DataFrame(metrics).query('fold == @start').sort_values('score',ascending=False).head(5).to_string(index=False),flush=True)
    r=pd.DataFrame(metrics)
    # Historical folds overlap: report explicitly as tuning, not independent CV.
    rank=r[r.fold!='2025-10-01'].groupby('model').score.mean().sort_values(ascending=False)
    name=rank.index[0]
    report={'selected':name,'scores':r[r.model==name].to_dict('records'),'evaluation':'Temporal tuning folds, partly overlapping. October is a one-month diagnostic, not a two-month score.','target_achieved':bool((r[r.model==name].score>=.95).all())}
    (V2/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
    if name.startswith('adaptive'):
        _,window,shrink=name.split('_');p=adaptive(d,future,int(window),float(shrink))
        d.to_csv(V2/'model_history.csv',sep=';',index=False)
    else:
        _,mode,depth=name.split('_');train=make_training(d,'2025-11-01');test=origin_features(d,future)
        p=direct(train,test,mode,int(depth),V2/'model.cbm')
    sub=future[['route','date','hour']].copy();sub['prediction']=np.maximum(0,np.rint(p)).astype(int)
    assert len(sub)==14640 and not sub.duplicated(['route','date','hour']).any()
    sub.to_csv(V2/'submission.csv',sep=';',index=False)
    print('DONE',flush=True)

if __name__=='__main__':main()
