"""Replay both independent networks and reproduce the three v16 submissions."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch

import cascaded_forecast as daily_engine
import weekly_joint as weekly_engine
from cascaded_stable import StableCascade
from weekly_stable import StableWeekly


def main():
    torch.set_num_threads(8);device=torch.device('mps')
    root=daily_engine.ROOT;dest=root/'outputs/v16_weekly'
    report=json.loads((dest/'dual_report.json').read_text());choice=report['selection']
    d,y,ex,groups=daily_engine.dataset();bank=daily_engine.ProfileBank(d)
    key=daily_engine.KEY;replays=[];model_checks=[];future_predictions={}
    for family,config in [('v15_cascade',choice['daily_model']),('v16_weekly',choice['weekly_model'])]:
        folder=root/'outputs'/family
        audits=[json.loads(f.read_text()) for f in folder.glob('*_e80.json')]
        assert all(a['max_training_label_day']==a['forecast_origin_day']-1 for a in audits)
        assert all(a['future_label_perturbation_max_difference']==0 for a in audits)
        assert all(a['last_day_gradient_to_first_step_7']>0 and a['cross_route_gradient_l1']>0 for a in audits)
        assert all(a['maximum_output_days_per_call']==7 and a['no_external_forecast_inputs'] for a in audits)
        model_checks.append({'family':family,'models_checked':len(audits),'training_boundaries_ok':True,
                             'future_target_perturbations_zero':True,'long_chain_and_cross_route_gradients_nonzero':True})
        for origin in [243,304]:
            frame=d[d.time.between(origin,origin+60)].reset_index(drop=True);cutoff=str(frame.date.min())
            tag=f'{cutoff}_{config}_s{choice["seed"]}_e{choice["epochs"]}'
            # These are our own locally trained torch checkpoints.
            ck=torch.load(folder/(tag+'.pt'),map_location='cpu',weights_only=False)
            assert ck['features']==daily_engine.EXOG and ck['group_order']==groups.to_dict('records')
            state=ck['state_dict'];scale=state['scale'].numpy().reshape(-1);active=state['active'].numpy().reshape(-1)
            if family=='v15_cascade':
                cls=StableCascade if 'volume_bound_log' in ck['audit'] else daily_engine.Cascade
                model=cls(scale,active);predict=daily_engine.predict;selected='1'
            else:
                cls=StableWeekly if 'volume_bound_log' in ck['audit'] else weekly_engine.WeeklyJoint
                model=cls(scale,active,ck['audit']['config']['bound']);predict=weekly_engine.predict;selected='7'
            model.load_state_dict(state);model.to(device)
            original=model.block;lengths=[]
            def traced(history,hx,future,hbase,fbase,offset=0):
                lengths.append(future.shape[2]);return original(history,hx,future,hbase,fbase,offset)
            model.block=traced
            replay=predict(model,y,ex,origin,61,ck['normalization'],bank,device)
            expected=([1]*61 if family=='v15_cascade' else [])+[7]*8+[5]
            assert lengths==expected
            cached=dict(np.load(folder/(tag+'.npz')))
            diffs={}
            for path,p in replay.items():
                np.testing.assert_allclose(p,cached[path],atol=1e-3,rtol=0)
                diffs[path]=float(np.max(abs(p-cached[path])))
                if 'volume_bound_log' in ck['audit']:
                    base=bank.get(origin)[28:].reshape(61,10,24).sum(2)
                    total=p.reshape(61,10,24).sum(2)
                    if family=='v16_weekly':
                        base=np.stack([base[t:t+7].sum(0) for t in range(0,61,7)])
                        total=np.stack([total[t:t+7].sum(0) for t in range(0,61,7)])
                    ratio=total[base>1e-5]/base[base>1e-5]
                    assert ratio.min()>=np.exp(-.15)-2e-5 and ratio.max()<=np.exp(.15)+2e-5
            altered=y.copy();altered[origin:]=777777
            changed=predict(model,altered,ex,origin,61,ck['normalization'],bank,device)
            for path in replay:np.testing.assert_array_equal(changed[path],replay[path])
            replays.append({'family':family,'origin':origin,'forward_steps_verified':True,
                            'checkpoint_max_abs_difference':diffs,'target_perturbation_max_difference':0})
            if origin==304:future_predictions[family]=replay[selected].astype(np.float64)
            del model;torch.mps.empty_cache()
    a=choice['daily_weight'];dp=future_predictions['v15_cascade'];wp=future_predictions['v16_weekly']
    predictions={'submission_dual_recursive_v16.csv':a*dp+(1-a)*wp,
                 'submission_daily_recursive_v15.csv':dp,'submission_weekly_recursive_v16.csv':wp}
    template=pd.read_csv(root/'data/test_submission.csv',sep=';')
    frame=d[d.time>=304].reset_index(drop=True);submissions={}
    for name,p in predictions.items():
        source=frame[key].copy();source['prediction']=np.rint(p).astype(np.int64)
        out=template[key].merge(source,on=key,how='left',validate='one_to_one')
        assert len(out)==14640 and out[key].equals(template[key]) and not out[key].duplicated().any()
        assert np.isfinite(out.prediction).all() and out.prediction.ge(0).all()
        assert out.loc[out.route==5,'prediction'].eq(0).all()
        encoded=out.to_csv(sep=';',index=False,lineterminator='\n').encode()
        exported=(root/'outputs'/name).read_bytes()
        assert encoded==exported
        digest=hashlib.sha256(encoded).hexdigest();assert digest==report['submissions'][name]['sha256']
        submissions[name]={'rows':14640,'reproduced_from_checkpoints':True,'sha256':digest}
    result={'model_audits':model_checks,'checkpoint_replays':replays,'submissions':submissions,
            'frozen_daily_weight':a,'different_model_classes':True,'independently_trained_components':True,
            'v8_or_v13_forecasts_used_as_components':False}
    (dest/'dual_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
