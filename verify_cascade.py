"""Replay the single cascade, check time boundaries and reproduce its CSV."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch

import cascaded_forecast as engine
from cascaded_dense import masked_error
from cascaded_stable import StableCascade


def main():
    torch.set_num_threads(8); device = torch.device('mps')
    report = json.loads((engine.DEST/'report.json').read_text())
    choice = report['selection']; key = engine.KEY
    d, y, ex, groups = engine.dataset(); bank = engine.ProfileBank(d)
    profiles = []
    torch.manual_seed(7)
    mock = torch.rand(3,240,61,requires_grad=True)
    target = torch.rand_like(mock)
    mask = (torch.arange(61)[None] < torch.tensor([1,17,61])[:,None]).float()
    weights = torch.ones(3)
    loss,_ = masked_error(mock,target,mask,weights,True)
    hidden_changed = target.clone()
    hidden_changed = torch.where(mask[:,None].bool(),hidden_changed,torch.full_like(hidden_changed,999999))
    changed_loss,_ = masked_error(mock,hidden_changed,mask,weights,True)
    torch.testing.assert_close(loss,changed_loss,rtol=0,atol=0)
    gradient = torch.autograd.grad(loss.sum(),mock)[0]
    assert (gradient*(1-mask[:,None])).abs().max().item() == 0
    for origin in [35,181,243,303,304]:
        altered = d.copy(); altered.loc[altered.time >= origin,'boardings'] = 999999
        np.testing.assert_array_equal(bank.get(origin),engine.ProfileBank(altered).get(origin))
        profiles.append({'origin':origin,'future_target_perturbation_max_difference':0})
    audit_files = list(engine.DEST.glob('*_e80.json'))
    audits = [json.loads(p.read_text()) for p in audit_files]
    assert audits and all(a['max_training_label_day'] == a['forecast_origin_day']-1 for a in audits)
    assert all(a['future_label_perturbation_max_difference'] == 0 for a in audits)
    for a in audits:
        assert a['no_direct_61_day_output'] and a['no_external_forecast_inputs']
        assert a['maximum_output_days_per_call'] == 7
        assert a['last_day_gradient_to_first_step_1'] > 0 and a['last_day_gradient_to_first_step_7'] > 0
        assert a['cross_route_gradient_l1'] > 0
        for horizon,origin in a['short_latest_origin'].items():
            assert origin+int(horizon) == a['forecast_origin_day']
    checks = []
    for origin in [243,304]:
        frame = d[d.time.between(origin,origin+60)].reset_index(drop=True)
        cutoff = str(frame.date.min())
        tag = f'{cutoff}_{choice["model"]}_s{choice["seed"]}_e{choice["epochs"]}'
        checkpoint = torch.load(engine.DEST/(tag+'.pt'),map_location='cpu',weights_only=False)
        assert checkpoint['features'] == engine.EXOG
        assert checkpoint['group_order'] == groups.to_dict('records')
        state = checkpoint['state_dict']
        cls = StableCascade if 'volume_bound_log' in checkpoint['audit'] else engine.Cascade
        model = cls(state['scale'].numpy().reshape(-1),state['active'].numpy().reshape(-1))
        model.load_state_dict(state); model.to(device)
        # Count real forward transitions; there must be 61 daily calls and nine
        # weekly calls (eight weeks plus a five-day tail), not one 61-day call.
        block = model.block; lengths = []
        def traced_block(history,hx,future,hbase,fbase,offset=0):
            lengths.append(future.shape[2])
            return block(history,hx,future,hbase,fbase,offset)
        model.block = traced_block
        replay = engine.predict(model,y,ex,origin,61,checkpoint['normalization'],bank,device)
        assert lengths == [1]*61+[7]*8+[5]
        saved = dict(np.load(engine.DEST/(tag+'.npz')))
        differences = {}
        for path,p in replay.items():
            np.testing.assert_allclose(p,saved[path],atol=1e-3,rtol=0)
            differences[path] = float(np.max(abs(p-saved[path])))
            if 'volume_bound_log' in checkpoint['audit']:
                base = bank.get(origin)[28:].reshape(61,10,24).sum(2)
                total = p.reshape(61,10,24).sum(2)
                ratio = total[base>1e-5]/base[base>1e-5]
                assert ratio.min() >= np.exp(-.15)-2e-5 and ratio.max() <= np.exp(.15)+2e-5
        altered = y.copy(); altered[origin:] = 123456
        changed = engine.predict(model,altered,ex,origin,61,checkpoint['normalization'],bank,device)
        for path in replay:
            np.testing.assert_array_equal(changed[path],replay[path])
        checks.append({'origin':origin,'forward_call_lengths_verified':True,
                       'checkpoint_replay_max_abs_difference':differences,'target_perturbation_max_difference':0})
        if origin == 304:
            source = frame[key].copy(); source['prediction'] = np.rint(replay[choice['path'][-1]]).astype(np.int64)
            template = pd.read_csv(engine.ROOT/'data/test_submission.csv',sep=';')
            out = template[key].merge(source,on=key,how='left',validate='one_to_one')
            assert len(out) == 14640 and out[key].equals(template[key]) and not out[key].duplicated().any()
            assert np.isfinite(out.prediction).all() and out.prediction.ge(0).all()
            assert out.loc[out.route == 5,'prediction'].eq(0).all()
            expected = out.to_csv(sep=';',index=False,lineterminator='\n').encode()
            exported = (engine.ROOT/report['submission']['path']).read_bytes()
            assert expected == exported
            assert hashlib.sha256(exported).hexdigest() == report['submission']['sha256']
        del model; torch.mps.empty_cache()
    result = {'models_checked':len(audits),'training_labels_end_before_origin':True,
              'short_examples_include_latest_available_labels':True,'profile_checks':profiles,
              'all_model_future_target_perturbations_zero':True,
              'all_final_to_initial_step_gradients_nonzero':True,
              'all_cross_route_gradients_nonzero':True,'checkpoint_replays':checks,
              'dense_loss_independent_of_masked_targets':True,
              'dense_loss_gradient_zero_at_unavailable_targets':True,
              'submission_reproduced_from_one_checkpoint_one_path':True,
              'submission_sha256':report['submission']['sha256'],'rows':14640}
    (engine.DEST/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
