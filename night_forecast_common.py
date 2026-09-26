"""Shared fixed-origin data and artifact utilities for the overnight study."""
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/night_20260927'
KEY=['route','date','hour']
SELECTION=[151,181,212]
CONTROL=243


def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def score(y,p):
    y=np.asarray(y);p=np.maximum(0,np.rint(p))
    return float(max(0,1-np.abs(y-p).sum()/max(1,y.sum())))


class Data:
    def __init__(self):
        self.d=pd.read_parquet(ROOT/'data/complete_external_20260927/all_features_matrix.parquet').sort_values(['date','route','hour']).reset_index(drop=True)
        manifest=json.loads((ROOT/'data/complete_external_20260927/feature_manifest.json').read_text())
        self.cols=[c for c in manifest['features'] if not c.startswith('donor_') and c not in ['time','day','month','sin_year','cos_year','external_season','days_to_year_boundary']]
        self.d[self.cols]=self.d[self.cols].astype(float).replace([np.inf,-np.inf],np.nan).fillna(-1)
        self.routes=sorted(self.d.route.unique());self.nr=len(self.routes)
        assert len(self.d)==365*10*24 and not self.d.duplicated(KEY).any()
        self.y=self.d.boardings.to_numpy(copy=True).reshape(365,10,24)
        self.daytype=self.d.daytype.to_numpy().reshape(365,10,24)[:,0,0].astype(int)
        self.cancel=self.d.service_cancelled.to_numpy().reshape(365,10,24)
        self.cache={}

    def frame(self,origin,horizon=61):
        return self.d.iloc[origin*240:(origin+horizon)*240].copy().reset_index(drop=True)

    def profile(self,origin,half=56):
        key=(origin,half)
        if key in self.cache:return self.cache[key]
        y=self.y[:origin];valid=np.isfinite(y)&(self.cancel[:origin]<.5)
        age=origin-1-np.arange(origin)
        w=np.exp2(-age/half)[:,None,None]*valid
        y=np.nan_to_num(y)
        profiles=[]
        for typ in range(7):
            exact=(self.daytype[:origin]==typ)[:,None,None]
            pool=((self.daytype[:origin]<5) if typ<5 else (self.daytype[:origin]>=5))[:,None,None]
            poolw=w*pool;exactw=w*exact
            coarse=np.divide((poolw*y).sum(0),poolw.sum(0),out=np.zeros((10,24)),where=poolw.sum(0)>0)
            p=((exactw*y).sum(0)+3*coarse)/(exactw.sum(0)+3)
            profiles.append(p)
        self.cache[key]=np.array(profiles,dtype=np.float32)
        return self.cache[key]

    def tabular(self,origin,horizon=61):
        frame=self.frame(origin,horizon);x=frame[self.cols].copy()
        for half in [14,28,56,112]:
            x['profile_'+str(half)]=self.profile(origin,half)[self.daytype[origin:origin+horizon]].reshape(-1)
        x['horizon']=np.repeat(np.arange(1,horizon+1),240)
        x['recent_growth']=np.log((x.profile_14+20)/(x.profile_112+20))
        for c in ['route','routehour','daytype']:x[c]=x[c].astype(int)
        return x,frame

    def training_pairs(self,origin):
        pairs=[self.tabular(c,min(61,origin-c)) for c in range(56,origin-6,14)]
        x=pd.concat([p[0] for p in pairs],ignore_index=True)
        meta=pd.concat([p[1] for p in pairs],ignore_index=True)
        assert meta.time.max()<origin
        weight=np.exp2(-(origin-1-meta.time.to_numpy())/112)/meta.groupby(KEY).boardings.transform('size').to_numpy()
        return x,meta,weight

    def causal_check(self):
        before,_=self.tabular(151,61);copy=self.y.copy()
        try:
            self.y[151:]=999983;self.cache.clear();after,_=self.tabular(151,61)
            pd.testing.assert_frame_equal(before,after)
        finally:self.y=copy;self.cache.clear()

    def prediction(self,origin,p):
        v=self.frame(origin);p=np.asarray(p).reshape(-1)
        p=np.maximum(0,p)*np.clip(1-.975*v.service_cancelled.to_numpy(),0,1)*(v.route.to_numpy()!=5)
        return v,p

    def export(self,folder,name,origin,p):
        v,p=self.prediction(origin,p);z=v[KEY+['boardings']].copy();z['prediction']=p
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        z.to_csv(folder/(name+'_'+str(origin)+'.csv'),index=False)
        if origin==304:
            sub=z[KEY].copy();sub['route']=sub.route.astype(int);sub['hour']=sub.hour.astype(int)
            sub['prediction']=np.rint(p).astype(int)
            assert len(sub)==14640 and not sub.duplicated(KEY).any() and sub.prediction.ge(0).all()
            dest=folder/('submission_'+name+'.csv');sub.to_csv(dest,sep=';',index=False)
            return dict(file=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),public_score=None,total=int(sub.prediction.sum()))
        return dict(origin=origin,date=v.date.min(),score=score(v.boardings,p),train_max_day=origin-1,horizon=61)


def status(folder,phase,**values):
    save_json(Path(folder)/'status.json',dict(phase=phase,updated_unix=time.time(),**values))
