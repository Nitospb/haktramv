"""Fixed-origin weekly baselines. Future observations never enter valid forecasts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2 import adaptive
from train_model import score

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v11_weekly'
KEY = ['route', 'date', 'hour']


def read_matrix():
    for p in [ROOT/'outputs/v4/training_matrix.csv', ROOT/'outputs/studio/v4/training_matrix.csv']:
        if p.exists():
            d = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError('Run feature_models.py or restore the saved feature matrix')
    assert not d.duplicated(KEY).any()
    return d


def weekly(history, future, period=7, reduce=None, recursive=False):
    dates = sorted(history.date.unique())
    cutoff = pd.Timestamp(max(dates))
    rows = history.pivot(index='date', columns=['route', 'hour'], values='boardings').sort_index()
    x = rows.to_numpy(copy=True)
    past_len = len(x)
    horizon = int((pd.Timestamp(future.date.max()) - cutoff).days)
    assert np.isfinite(x).all()
    values = np.full((past_len+horizon, x.shape[1]), np.nan)
    values[:past_len] = x
    for h in range(horizon):
        t = past_len+h
        if reduce:
            anchor = t-7 if recursive else past_len-1-((past_len-1-t) % 7)
            previous = values[[anchor-7*k for k in range(4)]]
            assert np.isfinite(previous).all()
            values[t] = np.mean(previous, axis=0) if reduce == 'mean' else np.median(previous, axis=0)
        else:
            values[t] = values[t-period]
    pred = pd.DataFrame(values[past_len:], index=pd.date_range(cutoff+pd.Timedelta(days=1),periods=horizon).strftime('%Y-%m-%d'),columns=rows.columns)
    pred = pred.rename_axis('date').stack(['route','hour'],future_stack=True).rename('prediction').reset_index()
    return future[KEY].merge(pred,on=KEY,how='left',validate='one_to_one').prediction.to_numpy()


def main():
    DEST.mkdir(exist_ok=True)
    d = read_matrix()
    records = []; correlations=[]
    origins = sorted(set(np.linspace(59,243,20).round().astype(int).tolist()+[181,212,243,273]))
    models = {
        'lag_1_recursive': dict(period=1),
        'lag_7_recursive': dict(period=7),
        'lag_14_recursive': dict(period=14),
        'mean_4weeks_frozen': dict(reduce='mean'),
        'median_4weeks_frozen': dict(reduce='median'),
        'mean_4weeks_recursive': dict(reduce='mean',recursive=True),
    }
    full = d[d.boardings.notna()].copy()
    for origin in origins:
        tr = full[full.time < origin].copy()
        v = full[(full.time >= origin)&(full.time < origin+61)].copy().reset_index(drop=True)
        assert len(v) and tr.time.max() < v.time.min()
        cutoff = str(v.date.min())
        ps = {name:weekly(tr,v,**config) for name,config in models.items()}
        ps['adaptive_v2'] = adaptive(tr,v,28,.5)
        # Diagnostic for a different task: each week's actual observations arrive.
        # Ineligible for the 61-day fixed-origin challenge, clearly kept separate.
        lag = full[KEY+['boardings']].copy()
        lag['date'] = (pd.to_datetime(lag.date)+pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d')
        ps['observed_weekly_updates_DIAGNOSTIC'] = v[KEY].merge(lag,on=KEY,how='left').boardings.to_numpy()
        incumbent = ROOT/'outputs/v8'/('validation_'+cutoff+'.csv')
        if not incumbent.exists():
            incumbent = ROOT/'outputs/studio/v8'/('validation_'+cutoff+'.csv')
        if incumbent.exists():
            b = pd.read_csv(incumbent,sep=';')
            b = v[KEY].merge(b[KEY+['prediction']],on=KEY,how='left',validate='one_to_one')
            assert b.prediction.notna().all()
            ps['incumbent_v8'] = b.prediction.to_numpy()
        changed = v.copy();changed['boardings']=np.arange(len(v))*7919
        np.testing.assert_array_equal(weekly(tr,changed),ps['lag_7_recursive'])
        days=(pd.to_datetime(v.date)-pd.Timestamp(cutoff)).dt.days.to_numpy()+1
        for name,p in ps.items():
            for label,mask in [('all',np.ones(len(v),bool)),('days_1_7',days<=7),('days_8_28',(days>=8)&(days<=28)),('days_29_61',days>=29),('last_7_days',days>days.max()-7)]:
                if mask.any():records.append(dict(origin=cutoff,model=name,window=label,days=int(days.max()),score=score(v.boardings.to_numpy()[mask],p[mask]),eligible=name!='observed_weekly_updates_DIAGNOSTIC'))
        correlations.append(dict(origin=cutoff,actual_lag7_correlation=float(np.corrcoef(v.boardings,ps['observed_weekly_updates_DIAGNOSTIC'])[0,1]),fixed_origin_weekly_correlation=float(np.corrcoef(v.boardings,ps['lag_7_recursive'])[0,1])))
        if origin in [181,212,243,273]:
            out=v[KEY+['boardings']].copy()
            for name,p in ps.items():out[name]=p
            out.to_csv(DEST/('baseline_validation_'+cutoff+'.csv'),index=False)
            print(cutoff,{name:round(score(v.boardings,p),6) for name,p in ps.items()},flush=True)
    r=pd.DataFrame(records);r.to_csv(DEST/'baseline_metrics.csv',index=False)
    report={'origins':len(origins),'origins_overlap':True,'target_perturbation_test':'passed','fixed_origin_horizon_days':61,
            'short_october_diagnostic_days':31,'known_public_incumbent':.89214,
            'mean_development_scores':r[(r.window=='all')&(r.days==61)].groupby('model').score.mean().to_dict(),
            'mean_development_score_origin_counts':r[(r.window=='all')&(r.days==61)].groupby('model').size().to_dict(),
            'correlations':correlations,
            'note':'All folds are development diagnostics and overlap. Incumbent v8 is available only at selected origins; do not compare its all-origin mean with other models. Observed-weekly-updates diagnostic uses target-period observations and is ineligible for the fixed-origin challenge.'}
    (DEST/'baseline_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='correlations'},indent=2),flush=True)


if __name__=='__main__':main()
