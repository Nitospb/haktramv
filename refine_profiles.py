"""Refine shape changes by route, type of day, and time of day."""
import json
import numpy as np
import pandas as pd
from research_v2 import V2
from train_model import load,calendar,profile,DATA,score

def predict(tr,va,window,strength,grouping):
    keys=['route','hour','daytype','summer']
    stats=tr[tr.holiday==0].groupby(keys).boardings.median().rename('p')
    v=va.merge(stats,on=keys,how='left')
    v['p']=v.p.fillna(pd.Series(profile(tr,va,365)))
    r=tr[tr.time>tr.time.max()-window].merge(stats,on=keys,how='left')
    v['band']=v.hour//3;r['band']=r.hour//3
    groups={'route':['route'],'day':['route','workday'],'band':['route','workday','band'],'hour':['route','workday','hour']}[grouping]
    z=r.groupby(groups)[['boardings','p']].sum()
    z['ratio']=((z.boardings+200)/(z.p+200)).clip(.5,1.5)**strength
    v=v.merge(z.ratio,on=groups,how='left')
    return v.p.to_numpy()*v.ratio.fillna(1).to_numpy()

def main():
    d=load();folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    metrics=[]
    for window in [14,28,42,56,84]:
        for strength in [.25,.5,.75,1]:
            for grouping in ['route','day','band','hour']:
                for start,end in folds:
                    tr=d[d.date<start];va=d[(d.date>=start)&(d.date<=end)]
                    p=predict(tr,va,window,strength,grouping)
                    metrics.append(dict(window=window,strength=strength,grouping=grouping,fold=start,score=score(va.boardings,p)))
    m=pd.DataFrame(metrics);m.to_csv(V2/'refinement_metrics.csv',index=False)
    # Selection explicitly targets the two most recent TWO-month tuning folds.
    rank=m[m.fold.isin(['2025-08-01','2025-09-01'])].groupby(['window','strength','grouping']).score.mean().sort_values(ascending=False)
    window,strength,grouping=rank.index[0]
    chosen=m[(m.window==window)&(m.strength==strength)&(m.grouping==grouping)]
    report={'method':'seasonal analog with route/day/hour shape correction','window':int(window),'strength':float(strength),'grouping':grouping,'scores':chosen.to_dict('records'),'evaluation':'Selected on Aug-Sep and Sep-Oct. Tuning scores, no independent holdout. Oct is one-month diagnostic only.','target_achieved':bool((chosen.score>=.95).all())}
    (V2/'refinement_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
    p=predict(d,future,int(window),float(strength),grouping)
    sub=future[['route','date','hour']].copy();sub['prediction']=np.maximum(0,np.rint(p)).astype(int)
    assert len(sub)==14640 and not sub.duplicated(['route','date','hour']).any()
    sub.to_csv(V2/'submission_refined.csv',sep=';',index=False)
    d.to_csv(V2/'model_history.csv',sep=';',index=False)
    print('DONE REFINEMENT',flush=True)

if __name__=='__main__':main()
