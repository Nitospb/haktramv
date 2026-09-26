"""Lagged fare composition × education calendar; only completed months before origin."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import enriched_origins as core
from train_model import score

DEST=core.DEST.parent/'v6_cohorts';DEST.mkdir(exist_ok=True)
ticket=pd.read_csv(core.ROOT/'data/features/07_transaction_derived/ticket_groups/ticket_month_route.csv')
ticket['sks']=ticket.good_type.str.contains('СКС',regex=False,na=False).astype(int)
ticket['sku']=ticket.good_type.str.contains('СКУ',regex=False,na=False).astype(int)
ticket['social']=ticket.good_type.str.contains('СКМ',regex=False,na=False).astype(int)
ticket['wallet']=ticket.good_type.str.contains('КОШЕЛЕК',regex=False,na=False).astype(int)
ticket['pass']=ticket.good_type.str.contains('дн',regex=False,na=False).astype(int)
GROUPS=['sks','sku','social','wallet','pass']
for c in GROUPS:ticket[c]=ticket[c]*ticket.n
original_origin=core.origin

def origin(history,future):
    v=original_origin(history,future)
    start=pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(history.time.max())+1)
    t=ticket[(ticket.month<start.month)&(ticket.month>=max(1,start.month-3))]
    z=t.groupby('route')[GROUPS+['n']].sum()
    for c in GROUPS:z['ticket_'+c+'_share']=z[c]/z.n.clip(lower=1)
    cols=['ticket_'+c+'_share' for c in GROUPS]
    v=v.merge(z[cols],on='route',how='left')
    for c in cols:v[c]=v[c].fillna(0)
    for typ,reg in [('sks','edu_uni_teaching_proxy'),('sku','edu_school_term')]:
        v['cohort_'+typ+'_active']=v['ticket_'+typ+'_share']*v[reg]
        v['cohort_'+typ+'_morning']=v['cohort_'+typ+'_active']*v.hour.between(7,10)*v.workday
        v['cohort_'+typ+'_afternoon']=v['cohort_'+typ+'_active']*v.hour.between(13,17)*v.workday
    v['cohort_social_weekend']=v.ticket_social_share*(1-v.workday)
    return v

core.origin=origin

def main():
    d=pd.read_csv(core.DEST/'rich_matrix.csv.gz');rows=[]
    folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    for start,end in folds:
        tr=d[d.date<start];v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True)
        x=core.training(d,start);z=origin(tr,v)
        p=core.fit(x,z,'all',6)
        np.save(DEST/(start+'.npy'),p)
        rows.append({'fold':start,'score':score(v.boardings,p)});print(rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(DEST/'metrics.csv',index=False)
    report={'scores':rows,'method':'Origin-aware model plus lagged fare shares and education interactions','selection':'Fixed ablation after examining previous folds; retrospective supplied weather/events; not an independent test.','future_ticket_months_used':False}
    (DEST/'report.json').write_text(json.dumps(report,indent=2))
    # Fit the candidate; subsequent comparison decides whether it is promoted.
    tr=d[d.date<'2025-11-01'];v=d[d.date>='2025-11-01'].reset_index(drop=True)
    p=core.fit(core.training(d,'2025-11-01'),origin(tr,v),'all',6,DEST/'model.cbm')
    out=v[core.KEY].copy();out['prediction']=np.maximum(0,np.rint(p)).astype(int)
    assert len(out)==14640 and not out.duplicated(core.KEY).any()
    out.to_csv(DEST/'submission.csv',sep=';',index=False);print('DONE COHORTS',flush=True)

if __name__=='__main__':main()
