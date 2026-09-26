"""Verify origin-safe profiles and a saved v13 checkpoint on Studio."""
import json

import numpy as np
import torch

from anchored_consistency import (DEST, EXOG, AnchoredResidual, ProfileBank,
                                 dataset, forecasts, incumbent)


def main():
    torch.set_num_threads(8)
    device = torch.device('mps')
    d, y, ex, groups = dataset()
    bank = ProfileBank(d)
    tests = []
    for origin in [35, 151, 243, 304]:
        changed = d.copy()
        changed.loc[changed.time >= origin, 'boardings'] = 999999
        altered = ProfileBank(changed).get(origin)
        original = bank.get(origin)
        np.testing.assert_array_equal(altered, original)
        tests.append({'origin_day': origin, 'profile_future_target_perturbation_max_diff':
                      float(np.max(abs(altered-original)))})
    tag = '2025-09-01_multi_c5_s20260926_e60'
    # This is our own locally generated checkpoint, not an untrusted attachment.
    checkpoint = torch.load(DEST / (tag+'.pt'), map_location='cpu', weights_only=False)
    assert checkpoint['features'] == EXOG
    assert checkpoint['group_order'] == groups.to_dict('records')
    state = checkpoint['state_dict']
    model = AnchoredResidual(state['scale'].numpy().reshape(-1),
                             state['active'].numpy().reshape(-1),
                             checkpoint['audit']['config']['gate'])
    model.load_state_dict(state); model.to(device)
    frame = d[d.time.between(243, 303)].reset_index(drop=True)
    guide = incumbent(frame, '2025-09-01')
    replay = forecasts(model, y, ex, 243, 61, checkpoint['normalization'], bank, device, guide)
    saved = dict(np.load(DEST / (tag+'.npz')))
    differences = {}
    for name, p in replay.items():
        error = float(np.max(abs(p-saved['v8_'+name])))
        np.testing.assert_allclose(p, saved['v8_'+name], rtol=0, atol=1e-3)
        differences[name] = error
    audits = [json.loads(path.read_text()) for path in DEST.glob('*_s*_e60.json')]
    assert audits and all(a['max_training_label_day'] == a['forecast_origin_day']-1 for a in audits)
    assert all(a['target_perturbation_max_difference'] == 0 for a in audits)
    report = {'profile_tests': tests, 'checkpoint_reproduction_max_abs_diff': differences,
              'trained_models_checked': len(audits),
              'training_labels_end_before_forecast_origin': True,
              'all_saved_target_perturbation_checks_zero': True}
    (DEST / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
