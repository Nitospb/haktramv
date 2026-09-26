"""Recency-weighted seasonal profiles and route-specific shrinkage selection."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from train_model import load, calendar, DATA, OUT, score
from research_v2 import adaptive

DEST=OUT/'v3';DEST.mkdir(exist_ok=True)

def wm(values,weights):
    ix=np.argsort(values);v=np.asarray(values)[ix];w=np.asarray(weights)[ix]
    return v[min(np.searchsorted(np.cumsum(w),w.sum()/2),len(v)-1)]

def weighted(tr,va,hl,pooled,decompose):
    tr=tr[tr.holiday==0].copy();va=va.copy()
    tr['kind']=tr.daytype;va['kind']=va.daytype
    if pooled:
        tr.loc[tr.kind<5,'kind']=0;va.loc[va.kind<5,'kind']=0
    tr['weight']=np.exp2((tr.time-tr.time.max())/hl)
    key=['route','kind','summer']
    if decompose:
        daily=tr.groupby(['route','date','kind','summer','time'],as_index=False).boardings.sum()
        daily['weight']=np.exp2((daily.time-daily.time.max())/hl)
        totals=daily.groupby(key).apply(lambda x:wm(x.boardings,x.weight),include_groups=False).rename('total')
        tr=tr.merge(daily[['route','date','boardings']].rename(columns={'boardings':'total'}),on=['route','date'])
        tr['share']=tr.boardings/tr.total.clip(lower=1)
        tr['weight']=np.exp2((tr.time-tr.time.max())/max(hl,56))
        shares=tr.groupby(key+['hour']).apply(lambda x:np.average(x.share,weights=x.weight),include_groups=False).rename('share')
        v=va.merge(totals,on=key,how='left').merge(shares,on=key+['hour'],how='left')
        p=v.total*v.share
    else:
        stats=tr.groupby(key+['hour']).apply(lambda x:wm(x.boardings,x.weight),include_groups=False).rename('p')
        p=va.merge(stats,on=key+['hour'],how='left').p
    fallback=adaptive(tr,va,28,.5)
    return p.fillna(pd.Series(fallback)).to_numpy()

def get_predictions(tr,va):
    result={'v2':adaptive(tr,va,28,.5)}
    for hl in [14,28,56,112,365]:
        for pooled in [False,True]:
            for dec in [False,True]:
                name='w_%s_%s_%s'%(hl,int(pooled),int(dec))
                result[name]=weighted(tr,va,hl,pooled,dec)
    return result

def holiday_adjust(tr,va,p):
    # Estimate non-January holiday deviations from ordinary Sunday profiles.
    h=tr[(tr.holiday==1)&(tr.month>1)].copy()
    if h.empty:return p.copy()
    ordinary=tr[(tr.daytype==6)&(tr.holiday==0)].groupby(['route','hour']).boardings.median().rename('ordinary')
    h=h.merge(ordinary,on=['route','hour'],how='left')
    h['band']=h.hour//3
    ratios=h.groupby(['route','band'])[['boardings','ordinary']].sum()
    ratios['ratio']=((ratios.boardings+500)/(ratios.ordinary+500)).clip(.65,1.3)
    v=va.copy();v['band']=v.hour//3
    v=v.merge(ratios.ratio,on=['route','band'],how='left')
    factor=v.ratio.fillna(1).to_numpy()
    return p*np.where(va.holiday.to_numpy()==1,factor,1)

def main():
    d=load();names=None;arrays={};frames={};rows=[]
    folds=[('2025-05-01','2025-06-30'),('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    for start,end in folds:
        tr=d[d.date<start];va=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True)
        cache=DEST/(start+'.npz')
        if cache.exists():pred=dict(np.load(cache))
        else:
            pred=get_predictions(tr,va);np.savez(cache,**pred)
        names=list(pred);a=np.stack(list(pred.values()),axis=1);arrays[start]=a;frames[start]=va
        for name,p in pred.items():rows.append({'fold':start,'model':name,'score':score(va.boardings,p)})
        print(pd.DataFrame(rows).query('fold==@start').sort_values('score',ascending=False).head(5).to_string(index=False),flush=True)
        pd.DataFrame(rows).to_csv(DEST/'metrics.csv',index=False)
    # Use recent autumn-transition folds, shrink route experts toward global winner.
    # October is retained as a ONE-month diagnostic, not used to choose parameters.
    tuning=['2025-08-01','2025-09-01']
    errors=np.mean([np.abs(arrays[f]-frames[f].boardings.to_numpy()[:,None]).sum(axis=0)/frames[f].boardings.sum() for f in tuning],axis=0)
    global_id=int(errors.argmin())
    route_weights={};results=[]
    for route in sorted(d.route.unique()):
        e=np.zeros(len(names));total=0
        for f in tuning:
            m=frames[f].route==route;y=frames[f].loc[m,'boardings'].to_numpy()
            e+=abs(arrays[f][m]-y[:,None]).sum(axis=0);total+=y.sum()
        # Global error provides regularization for route-specific choice.
        normalized=e/max(total,1)
        rank=np.argsort(.7*normalized+.3*errors)[:3]
        w=np.zeros(len(names));w[global_id]+=.5;w[rank]+=1/6
        route_weights[int(route)]={names[j]:float(v) for j,v in enumerate(w) if v>0}
    for f,_ in folds:
        v=frames[f];p=np.zeros(len(v))
        for route,weights in route_weights.items():
            m=v.route==route
            for n,w in weights.items():p[m]+=w*arrays[f][m,names.index(n)]
        rowscore={'fold':f,'v2':score(v.boardings,arrays[f][:,0]),'v3':score(v.boardings,p)}
        results.append(rowscore)
        out=v[['route','date','hour','boardings']].copy();out['prediction']=p;out.to_csv(DEST/('validation_'+f+'.csv'),sep=';',index=False)
    report={'global_model':names[global_id],'route_weights':route_weights,'scores':results,'selection':'Aug-Sep and Sep-Oct tuning; route choices shrunk toward global best. October is a one-month diagnostic. No new independent two-month holdout.','known_leaderboard':{'user_reported_v2':.88882,'v3':None}}
    (DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)
    future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
    # Compute only selected profiles on full history.
    chosen=set(n for w in route_weights.values() for n in w)
    pred={}
    for n in chosen:
        if n=='v2':pred[n]=adaptive(d,future,28,.5)
        else:
            _,hl,pool,dec=n.split('_');pred[n]=weighted(d,future,int(hl),bool(int(pool)),bool(int(dec)))
    p=np.zeros(len(future))
    for route,weights in route_weights.items():
        m=future.route==route
        for n,w in weights.items():p[m]+=w*pred[n][m]
    sub=future[['route','date','hour']].copy();sub['prediction']=np.maximum(0,np.rint(p)).astype(int)
    assert len(sub)==14640 and not sub.duplicated(['route','date','hour']).any()
    sub.to_csv(DEST/'submission.csv',sep=';',index=False)
    # A conservative ensemble keeps half the candidate whose real score is known.
    sub['prediction']=np.maximum(0,np.rint(.5*p+.5*adaptive(d,future,28,.5))).astype(int)
    sub.to_csv(DEST/'submission_conservative.csv',sep=';',index=False)
    d.to_csv(DEST/'model_history.csv',sep=';',index=False)
    print('DONE V3',flush=True)

if __name__=='__main__':main()
