"""Extended CPU model search; final holdout never participates in model selection."""
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from train_model import DATA, OUT, FEATURES, calendar, load, profile, score

CONFIGS = [(6,'MAE',60),(8,'MAE',60),(10,'MAE',90),(8,'MAE',180),
           (8,'RMSE',90),(10,'RMSE',180),(8,'Poisson',90),(6,'MAE',365)]

def design(d, harmonics):
    a=[np.ones(len(d))]
    for j in range(1,harmonics+1):
        a.extend([np.sin(2*np.pi*j*d.time/365),np.cos(2*np.pi*j*d.time/365)])
    a.extend([(d.daytype==j).astype(float) for j in range(6)])
    a.append(d.summer.astype(float))
    return np.array(a).T

def seasonal(tr,va,harmonics,alpha,save=None):
    # Independent route-hour regressions, ridge-regularized annual seasonality.
    out=np.zeros(len(va)); models={}
    for (route,hour),g in tr.groupby(['route','hour']):
        mask=(va.route==route)&(va.hour==hour)
        scale=max(g.boardings.mean(),1)
        model=Ridge(alpha=alpha,fit_intercept=False)
        model.fit(design(g,harmonics),g.boardings/scale)
        out[mask]=np.maximum(0,model.predict(design(va[mask],harmonics))*scale)
        models[route,hour]=(model,scale)
    if save:
        with open(save,'wb') as f:pickle.dump(dict(models=models,harmonics=harmonics),f)
    return out

def fit_predict(name,tr,va,save=False):
    bits=name.split('_')
    if bits[0]=='profile':
        p=profile(tr,va,int(bits[1]),bits[2])
        if save:
            tr[tr.time>tr.time.max()-int(bits[1])].groupby(['route','daytype','hour']).boardings.agg(bits[2]).to_csv(OUT/(name+'.csv'))
        return p
    if bits[0]=='ridge':
        return seasonal(tr,va,int(bits[1]),float(bits[2]), OUT/(name+'.pkl') if save else None)
    _,depth,loss,hl=bits
    model=CatBoostRegressor(iterations=1800,depth=int(depth),learning_rate=.04,
        loss_function=loss,l2_leaf_reg=8,random_seed=42,thread_count=12,
        verbose=False,allow_writing_files=False)
    model.fit(tr[FEATURES],tr.boardings,cat_features=['route'],
        sample_weight=.5**((tr.time.max()-tr.time)/int(hl)))
    p=np.maximum(0,model.predict(va[FEATURES]))
    p[va.route.values==5]=0
    if save:model.save_model(str(OUT/(name+'.cbm')))
    return p

def main():
    d=load()
    names=['profile_%d_%s'%(w,k) for w in [14,28,56,84,120,365] for k in ['median','mean']]
    names+=['ridge_%d_%s'%(h,a) for h in [1,2] for a in [1,10,50]]
    names+=['cat_%d_%s_%d'%c for c in CONFIGS]
    folds=[('2025-05-01','2025-06-30'),('2025-07-01','2025-08-31'),('2025-09-01','2025-10-31')]
    preds={}; actual={}; frames={}; metrics=[]
    for start,end in folds:
        tr=d[d.date<start];va=d[(d.date>=start)&(d.date<=end)]
        actual[start]=va.boardings.values;frames[start]=va
        matrix=[]
        for name in names:
            cache=OUT/('cache_'+start+'_'+name+'.npy')
            if cache.exists():p=np.load(cache)
            else:
                p=fit_predict(name,tr,va);np.save(cache,p)
            matrix.append(p)
            s=score(va.boardings,p)
            metrics.append(dict(fold=start,model=name,score=s))
            print(start,name,round(s,6),flush=True)
            pd.DataFrame(metrics).to_csv(OUT/'advanced_metrics.csv',index=False)
        preds[start]=np.stack(matrix,axis=1)
    # Equal fold weighting; select ensemble using only May-Aug outcomes.
    y=np.concatenate([actual[f] / actual[f].sum() for f,_ in folds[:2]])
    x=np.concatenate([preds[f] / actual[f].sum() for f,_ in folds[:2]])
    initial=np.zeros(len(names));initial[np.argmin(np.abs(x-y[:,None]).sum(axis=0))]=1
    opt=minimize(lambda w:np.abs(x@w-y).sum(),initial,method='SLSQP',
        bounds=[(0,1)]*len(names),constraints={'type':'eq','fun':lambda w:w.sum()-1},
        options={'maxiter':300,'ftol':1e-10})
    w=opt.x if opt.success else initial
    w[w<.005]=0;w/=w.sum()
    hold='2025-09-01';p=preds[hold]@w
    report={'weights':{n:float(v) for n,v in zip(names,w) if v>0},
        'selection_periods':['2025-05-01/2025-06-30','2025-07-01/2025-08-31'],
        'untouched_holdout':'2025-09-01/2025-10-31',
        'holdout_score':score(actual[hold],p),'target_score':.95,
        'target_achieved_on_holdout':score(actual[hold],p)>=.95,
        'hidden_test_score':None,'optimizer_success':bool(opt.success)}
    for f,_ in folds: report['score_'+f]=score(actual[f],preds[f]@w)
    print(json.dumps(report,indent=2),flush=True)
    (OUT/'advanced_report.json').write_text(json.dumps(report,indent=2))
    v=frames[hold][['route','date','hour','boardings']].copy();v['prediction']=p
    v.to_csv(OUT/'holdout_predictions.csv',sep=';',index=False)
    by_route=[]
    for route,g in v.groupby('route'):
        by_route.append({'route':int(route),'score':score(g.boardings,g.prediction) if g.boardings.sum()>0 else None,'absolute_error':float(abs(g.boardings-g.prediction).sum())})
    (OUT/'route_metrics.json').write_text(json.dumps(by_route,indent=2))
    future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
    final=np.zeros(len(future))
    for name,weight in zip(names,w):
        if weight>0:
            print('Final training:',name,weight,flush=True)
            final+=weight*fit_predict(name,d,future,save=True)
    sub=future[['route','date','hour']].copy();sub['prediction']=np.rint(np.maximum(0,final)).astype(int)
    assert len(sub)==14640 and not sub.duplicated(['route','date','hour']).any()
    assert sub.prediction.notna().all() and (sub.prediction>=0).all()
    sub.to_csv(OUT/'submission.csv',sep=';',index=False)
    print('DONE: submission.csv, 14640 rows',flush=True)

if __name__=='__main__':main()
