"""Large CPU boosting and causal multi-horizon residuals; resumable two-hour run."""
import argparse
import json
import time
import os
import fcntl
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from night_forecast_common import ROOT,OUT,Data,SELECTION,CONTROL,save_json,status

CONFIGS={
    'deep_absolute':dict(depth=9,iterations=6000,learning_rate=.025,l2_leaf_reg=30),
    'deep_profile_residual':dict(depth=8,iterations=6000,learning_rate=.025,l2_leaf_reg=30),
}
CHECKPOINTS=[1000,3000,6000]


class Deadline:
    def __init__(self,end,folder,phase):self.end=end;self.folder=folder;self.phase=phase
    def after_iteration(self,info):
        if info.iteration%200==0:status(self.folder,self.phase,iteration=info.iteration)
        return time.time()<self.end


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=2);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    folder=OUT/('catboost_smoke' if args.smoke else 'catboost');folder.mkdir(parents=True,exist_ok=True)
    lock=open(folder/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    end=time.time()+args.hours*3600;data=Data();data.causal_check();records=[]
    config_items=list(CONFIGS.items())
    for name,config in config_items:
        for origin in ([151] if args.smoke else SELECTION):
            if time.time()>end:status(folder,'budget_exhausted',completed=records);return
            model,predict=fit_one(data,folder,name,config,origin,end,12 if args.smoke else None)
            for trees in ([model.tree_count_] if args.smoke else CHECKPOINTS):
                if trees>model.tree_count_:continue
                entry=data.export(folder,name+'_n'+str(trees),origin,predict(trees));entry.update(model=name,trees=trees,split='selection');records.append(entry)
            pd.DataFrame(records).to_csv(folder/'metrics.csv',index=False)
    if args.smoke:status(folder,'smoke_passed',records=records);print(records,flush=True);return
    r=pd.DataFrame(records)
    counts=r.groupby(['model','trees']).origin.nunique();eligible=counts[counts==len(SELECTION)].index
    assert len(eligible),'No complete three-window candidate; resume before selection.'
    means=r.groupby(['model','trees']).score.mean().loc[eligible].sort_values(ascending=False)
    choices=[]
    for name,trees in means.index:
        if name not in [n for n,t in choices]:choices.append((str(name),int(trees)))
    summary=[]
    for name,trees in choices:
        config=dict(CONFIGS[name],iterations=int(trees))
        for origin in [CONTROL,304]:
            if time.time()>end:status(folder,'budget_exhausted',selected=choices,completed=summary);return
            model,predict=fit_one(data,folder,name,config,origin,end)
            if model.tree_count_<trees:status(folder,'partial_model_requires_resume',origin=origin,model=name);return
            entry=data.export(folder,name+'_n'+str(trees),origin,predict(trees));entry.update(model=name,trees=int(trees),split='control' if origin==CONTROL else 'submission');summary.append(entry)
    save_json(folder/'report.json',dict(selection=records,selected=[dict(model=n,trees=int(t)) for n,t in choices],results=summary,
        retrospective_external_inputs=True,causal_profile_check=True,public_score=None,active_model_changed=False,
        limitation='Previously explored windows. Selection only on June/July/August origins; September control does not select hyperparameters in this run.'))
    status(folder,'complete',results=summary);print('DONE',summary,flush=True)


def fit_one(data,folder,name,config,origin,end,smoke_iterations=None):
    residual='residual' in name
    if residual:
        x,meta,w=data.training_pairs(origin);xx,future=data.tabular(origin)
        scale=np.maximum(30,x.profile_56.to_numpy());future_scale=np.maximum(30,xx.profile_56.to_numpy())
        target=(meta.boardings.to_numpy()-x.profile_56.to_numpy())/scale
        w=w*scale
    else:
        meta=data.d.iloc[:origin*240].copy();future=data.frame(origin)
        x=meta[data.cols].copy();xx=future[data.cols].copy()
        for c in ['route','routehour','daytype']:x[c]=x[c].astype(int);xx[c]=xx[c].astype(int)
        target=meta.boardings.to_numpy();w=np.exp2(-(origin-1-meta.time.to_numpy())/180)
        w*=np.where(meta.month.isin([1,2]),.35,np.where(meta.month.isin([6,7,8]),.5,1))
    good=(meta.route.to_numpy()!=5)&(meta.service_cancelled.to_numpy()<.5)&np.isfinite(target)
    path=folder/(name+'_'+str(origin)+'.cbm');meta_path=path.with_suffix('.json')
    model=CatBoostRegressor()
    desired=smoke_iterations or config['iterations']
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text())['trees']>=desired:
        model.load_model(str(path));print('REUSE',name,origin,flush=True)
    else:
        options=dict(config);options['iterations']=desired
        model=CatBoostRegressor(**options,loss_function='MAE',thread_count=6,random_seed=270927,
            verbose=500,allow_writing_files=False,bootstrap_type='Bayesian',bagging_temperature=.5)
        status(folder,'training_'+name,origin=origin,rows=int(good.sum()),target_trees=desired)
        start=time.time();model.fit(x.loc[good],target[good],sample_weight=w[good],cat_features=['route','routehour','daytype'],callbacks=[Deadline(end,folder,'training_'+name)])
        model.save_model(str(path))
        save_json(meta_path,dict(model=name,origin=origin,trees=model.tree_count_,rows=int(good.sum()),seconds=time.time()-start,features=list(x),train_max_day=int(meta.time.max()),config=options))
    def predict(trees):
        p=model.predict(xx,ntree_end=int(trees))
        return p*future_scale+xx.profile_56.to_numpy() if residual else p
    # The persisted model must reproduce eligible forecast values.
    replay=CatBoostRegressor();replay.load_model(str(path))
    np.testing.assert_array_equal(model.predict(xx.iloc[:256]),replay.predict(xx.iloc[:256]))
    return model,predict


if __name__=='__main__':main()
