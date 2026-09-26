"""Recent route levels learned on normal-service days, separate from disruption effects."""
import json
import numpy as np
import pandas as pd
from service_models import normal_profile,adjust,regime,KEY,PREVIOUS as V4
from train_model import score

DEST=V4.parent/'v5_levels';DEST.mkdir(exist_ok=True)

def predict(tr,v,power,window,strength,byday):
    tr=tr.copy();tr['normal']=normal_profile(tr,tr,power)
    future_normal=normal_profile(tr,v,power)
    recent=tr[(tr.time>tr.time.max()-window)&(regime(tr)==0)&(tr.service_incident_fraction==0)]
    keys=['route','workday'] if byday else ['route']
    z=recent.groupby(keys)[['boardings','normal']].sum()
    z['level']=((z.boardings+3000)/(z.normal+3000)).clip(.6,1.5)**strength
    level=v[keys].merge(z.level,on=keys,how='left').level.fillna(1).to_numpy()
    return adjust(tr,v,tr.normal.to_numpy(),future_normal*level)

def main():
    d=pd.read_csv(V4/'training_matrix.csv');rows=[];arrays={};frames={}
    folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    configs=[(p,w,s,g) for p in [.5,1] for w in [14,28,56] for s in [.5,1] for g in [False,True]]
    for start,end in folds:
        tr=d[d.date<start].reset_index(drop=True);v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);frames[start]=v;ps={}
        for c in configs:
            name='_'.join(map(str,c));p=predict(tr,v,*c);ps[name]=p
            rows.append({'fold':start,'model':name,'score':score(v.boardings,p)})
        arrays[start]=ps;np.savez(DEST/(start+'.npz'),**ps)
    r=pd.DataFrame(rows);r.to_csv(DEST/'metrics.csv',index=False)
    rank=r[r.fold.isin(['2025-07-01','2025-08-01'])].groupby('model').score.mean().sort_values(ascending=False)
    name=rank.index[0];c=configs[['_'.join(map(str,a)) for a in configs].index(name)]
    report={'config':list(c),'scores':r[r.model==name].to_dict('records'),'selection':'Jul-Aug and Aug-Sep tuning. All periods previously inspected; retrospective exogenous data.'}
    print(json.dumps(report,indent=2),flush=True);(DEST/'report.json').write_text(json.dumps(report,indent=2))
    tr=d[d.date<'2025-11-01'].reset_index(drop=True);v=d[d.date>='2025-11-01'].reset_index(drop=True);p=predict(tr,v,*c)
    out=v[KEY].copy();out['prediction']=np.maximum(0,np.rint(p)).astype(int);out.to_csv(DEST/'submission.csv',sep=';',index=False)
    print('DONE LEVELS',flush=True)

if __name__=='__main__':main()
