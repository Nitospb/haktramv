"""Standalone LightGBM and causal profile-residual boosting, fixed-origin forecasts."""
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from seasonal_recency_study import Study, FEATURES, KEY, metric
from forecast_artifacts import artifact_directory

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/gradient_recency_20260927'
CONFIGS = {
    'lean_l1_h112': dict(mode='raw', objective='regression_l1', soft=False, leaves=31, lean=True, half=112),
    'lean_l1_soft_h112': dict(mode='raw', objective='regression_l1', soft=True, leaves=31, lean=True, half=112),
    'lean_l1_soft_h365': dict(mode='raw', objective='regression_l1', soft=True, leaves=31, lean=True, half=365),
    'lean_poisson_soft_h365': dict(mode='raw', objective='poisson', soft=True, leaves=31, lean=True, half=365),
    'raw_l1_all': dict(mode='raw', objective='regression_l1', soft=False, leaves=31),
    'raw_l1_soft': dict(mode='raw', objective='regression_l1', soft=True, leaves=31),
    'raw_poisson_soft': dict(mode='raw', objective='poisson', soft=True, leaves=31),
    'residual_l1_all': dict(mode='residual', objective='regression_l1', soft=False, leaves=31),
    'residual_l1_soft': dict(mode='residual', objective='regression_l1', soft=True, leaves=31),
    'residual_l1_small': dict(mode='residual', objective='regression_l1', soft=True, leaves=15),
}


class Experiment:
    def __init__(self):
        self.s = Study()
        self.s.y = self.s.y.copy()
        self.d = self.s.frame
        self.routes = self.s.groups.route.to_numpy()
        self.bank_cache = {}
        self.half_cache = {}
        extra = ['month', 'sin_year', 'cos_year', 'day', 'summer',
                 'service_extended_service', 'service_days_since_start']
        self.features = list(dict.fromkeys([c for c in FEATURES + extra if c in self.d]))
        self.cats = ['route', 'routehour', 'daytype']

    def bank(self, origin, half):
        key = (origin, half)
        if key not in self.bank_cache:
            self.bank_cache[key] = self.s.profile(origin, 1, 'all', half, 'calendar', 'median', 3)[1]
        return self.bank_cache[key]

    def halves(self, origin):
        # Route-specific half-life selected using only the previous 28 observed days.
        # Each decision is nested inside its own origin, including pseudo-origins.
        if origin in self.half_cache:
            return self.half_cache[origin]
        halves = [28, 56, 112]
        start = origin - 28
        losses = []
        for half in halves:
            p = self.s.post(self.bank(start, half)[self.s.daytype[start:origin]], start, 28)
            losses.append(np.abs(self.s.y[start:origin] - p).reshape(28, 10, 24).sum(axis=(0, 2)))
        losses = np.array(losses)
        # Weak preference toward 56 days discourages changing on tiny differences.
        losses[[0, 2]] *= 1.01
        chosen = np.array(halves)[losses.argmin(axis=0)]
        self.half_cache[origin] = np.repeat(chosen, 24)
        return self.half_cache[origin]

    def input(self, origin, horizon):
        f = self.d.iloc[origin * 240:(origin + horizon) * 240].copy()
        halves = self.halves(origin)
        dynamic = np.zeros((7, 240))
        for half in [28, 56, 112]:
            mask = halves == half
            dynamic[:, mask] = self.bank(origin, half)[:, mask]
        base = dynamic[self.s.daytype[origin:origin + horizon]].reshape(-1)
        x = f[self.features].copy()
        x['profile'] = base
        for half in [28, 56, 112]:
            x['profile_' + str(half)] = self.bank(origin, half)[self.s.daytype[origin:origin + horizon]].reshape(-1)
        x['horizon'] = np.repeat(np.arange(1, horizon + 1), 240)
        x['origin_month'] = int(self.s.month[origin - 1])
        x['profile_half'] = np.tile(halves, horizon)
        assert np.isfinite(x.to_numpy(dtype=float)).all()
        return x, base, f

    def raw_input(self, f):
        return f[self.features].copy()

    def train_data(self, origin, mode):
        if mode == 'raw':
            frame = self.d.iloc[:origin * 240].copy()
            x = self.raw_input(frame)
            base = np.zeros(len(frame))
        else:
            # Every profile predates its example's target; no full-training target encoding.
            packs = [self.input(c, min(61, origin - c)) for c in range(56, origin - 13, 28)]
            x = pd.concat([p[0] for p in packs], ignore_index=True)
            base = np.concatenate([p[1] for p in packs])
            frame = pd.concat([p[2] for p in packs], ignore_index=True)
        good = (frame.route.to_numpy() != 5) & (frame.service_cancelled.to_numpy() < .5)
        x = x.loc[good].reset_index(drop=True)
        frame = frame.loc[good].reset_index(drop=True)
        base = base[good]
        assert frame.time.max() < origin
        multiplicity = frame.groupby(KEY).boardings.transform('size').to_numpy()
        return x, frame.boardings.to_numpy() - base, frame, multiplicity

    def fit(self, origin, horizon, name, config, training):
        x, y, hist, multiplicity = training[config['mode']]
        if config['mode'] == 'raw':
            future = self.d.iloc[origin * 240:(origin + horizon) * 240].copy()
            xx = self.raw_input(future)
            base = np.zeros(len(future))
        else:
            xx, base, future = self.input(origin, horizon)
        if config.get('lean'):
            keep = [c for c in x if c in FEATURES]
            x = x[keep].copy(); xx = xx[keep].copy()
        assert list(x) == list(xx) and 'boardings' not in x and 'time' not in x
        route_halves = dict(zip(self.s.groups.route.unique().astype(int), self.halves(origin)[::24].astype(int)))
        if 'half' in config:
            route_halves = {k:config['half'] for k in route_halves}
        age = origin - 1 - hist.time.to_numpy()
        weights = np.exp2(-age / hist.route.map(route_halves).to_numpy()) / multiplicity
        if config['soft']:
            weights *= np.where(hist.month.isin([1, 2]), .35, np.where(hist.month.isin([6, 7, 8]), .5, 1.))
        model = lgb.LGBMRegressor(objective=config['objective'], n_estimators=800,
                                 learning_rate=.035, num_leaves=config['leaves'],
                                 min_child_samples=100, reg_lambda=15, colsample_bytree=.9,
                                 n_jobs=6, random_state=270926, verbosity=-1,
                                 deterministic=True, force_col_wise=True)
        model.fit(x, y, sample_weight=weights, categorical_feature=self.cats)
        p = self.s.post(model.predict(xx) + base, origin, horizon)
        tag = ('final' if origin == 304 else str(future.date.min())) + '_' + name
        path = OUT / (tag + '.txt')
        model.booster_.save_model(str(path))
        replay = lgb.Booster(model_file=str(path))
        np.testing.assert_array_equal(p, self.s.post(replay.predict(xx, num_threads=6) + base, origin, horizon))
        audit = dict(origin=origin, horizon=horizon, max_training_time=int(hist.time.max()),
                     training_rows=len(hist), config=config, route_half_lives={str(k):int(v) for k,v in route_halves.items()},
                     features=list(x), label_derived_features='Causal pseudo-origin profiles only' if config['mode']=='residual' else 'None',
                     checkpoint_replay_exact=True, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(audit, indent=2)+'\n')
        pd.DataFrame({'feature':list(x),'gain':model.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False).to_csv(OUT/(tag+'_importance.csv'),index=False)
        return p

    def verify_causal_inputs(self, origin, horizon):
        x, base, _ = self.input(origin, horizon)
        halves = self.halves(origin).copy()
        saved = self.s.y.copy()
        try:
            self.s.y[origin:] = 999983
            self.bank_cache.clear(); self.half_cache.clear()
            changed, newbase, _ = self.input(origin, horizon)
            np.testing.assert_array_equal(x.to_numpy(), changed.to_numpy())
            np.testing.assert_array_equal(base, newbase)
            np.testing.assert_array_equal(halves, self.halves(origin))
        finally:
            self.s.y[:] = saved
            self.bank_cache.clear(); self.half_cache.clear()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['validation','final'],default='validation')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    e=Experiment()
    if args.phase=='validation':
        rows = pd.read_csv(OUT/'metrics.csv').to_dict('records') if (OUT/'metrics.csv').exists() else []
        for origin in [181,243,273]:
            horizon=min(61,304-origin)
            e.verify_causal_inputs(origin,horizon)
            training={mode:e.train_data(origin,mode) for mode in ['raw','residual']}
            f=e.d.iloc[origin*240:(origin+horizon)*240]
            date=str(f.date.min())
            for name,c in CONFIGS.items():
                if any(r['origin']==date and r['model']==name for r in rows):
                    continue
                p=e.fit(origin,horizon,name,c,training)
                row=dict(origin=date,horizon=horizon,model=name,score=metric(e.s.y[origin:origin+horizon],p),prediction_total=float(p.sum()))
                rows.append(row)
                out=f[KEY].copy();out['prediction']=p.reshape(-1)
                out.to_csv(OUT/(date+'_'+name+'.csv.gz'),index=False)
                print(json.dumps(row),flush=True)
            path=artifact_directory('v8')/('validation_'+date+'.csv')
            if path.exists() and not any(r['origin']==date and r['model']=='v8_scale_1.0' for r in rows):
                ref=pd.read_csv(path,sep=';')
                p=f[KEY].merge(ref[KEY+['prediction']],on=KEY,validate='one_to_one').prediction.to_numpy()
                assert len(p)==len(f)
                for scale in [1.,1.03]:
                    rows.append(dict(origin=date,horizon=horizon,model='v8_scale_'+str(scale),score=metric(f.boardings.to_numpy(),np.rint(np.rint(p)*scale))))
            pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        m=pd.DataFrame(rows)
        ranks=m[(m.horizon==61)&m.model.isin(CONFIGS)].groupby('model').score.mean().sort_values(ascending=False)
        report=dict(selected=ranks.index[0],ranking=ranks.to_dict(),metrics=rows,configs=CONFIGS,
                    selection='Mean score of July-August and September-October, full 61-day fixed-origin blocks; October is a shorter diagnostic only.',
                    caveats=['Previously explored development windows, not independent holdout.', 'Supplied retrospective weather and service covariates.', 'Public score unknown; current confirmed best .89402 unchanged.'],
                    causal_future_label_perturbation_passed=True,lightgbm_version=lgb.__version__)
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('SELECTED',report['selected'],json.dumps(report['ranking']),flush=True)
    else:
        report=json.loads((OUT/'report.json').read_text());name=report['selected'];c=CONFIGS[name]
        e.verify_causal_inputs(304,61)
        p=e.fit(304,61,name,c,{c['mode']:e.train_data(304,c['mode'])})
        f=e.d.iloc[304*240:][KEY].copy();f['prediction']=np.rint(p.reshape(-1)).astype(np.int64)
        template=pd.read_csv(ROOT/'data/test_submission.csv',sep=';')
        out=template[KEY].merge(f,on=KEY,how='left',validate='one_to_one')
        assert len(out)==14640 and out[KEY].equals(template[KEY]) and out.prediction.notna().all()
        assert (out.prediction>=0).all() and (out.loc[out.route==5,'prediction']==0).all()
        path=OUT/'submission_lightgbm.csv';out.to_csv(path,sep=';',index=False,lineterminator='\n')
        report['submission']=dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(out),prediction_total=int(out.prediction.sum()),public_score=None,standalone=True)
        report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('EXPORTED',json.dumps(report['submission']),flush=True)

if __name__=='__main__':
    main()
