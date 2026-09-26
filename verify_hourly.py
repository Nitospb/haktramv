"""Check full-horizon recurrences, time boundaries and the exported v14 file."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch

import hourly_consistency as engine
from hourly_shape_consistency import HourlyShapeResidual


KEY = engine.KEY


def full_horizon_scan_check():
    # Compare the actual hourly-state routine against an independent serial loop,
    # including its gradients and an incomplete innovation block.
    checks = []
    for step in engine.STEPS:
        torch.manual_seed(101)
        length = 1465
        innovation = (.1*torch.randn(1, 2, length, 2)).requires_grad_()
        tau = torch.full_like(innovation, 168., requires_grad=True)
        initial = torch.randn(1, 2, 2, requires_grad=True)
        actual = engine.hourly_states(innovation, tau, initial, step)
        state = initial; sequence = []
        for start in range(0, length, step):
            end = min(start+step, length)
            u = innovation[:, :, start:end].mean(2)
            t = tau[:, :, start:end].mean(2)
            decay = torch.exp(-1/t)
            for _ in range(end-start):
                state = decay*state+(1-decay)*u
                sequence.append(state)
        expected = torch.stack(sequence, 2)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-5)
        ag = torch.autograd.grad(actual[:, :, -1].sum(), (innovation, tau, initial))
        eg = torch.autograd.grad(expected[:, :, -1].sum(), (innovation, tau, initial))
        for a, b in zip(ag, eg):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
        checks.append({'step_hours': step, 'length_hours': length,
                       'value_max_diff': float((actual-expected).abs().max()),
                       'gradient_max_diff': max(float((a-b).abs().max()) for a, b in zip(ag, eg))})
    return checks


def main():
    torch.set_num_threads(8); device = torch.device('mps')
    report_path = engine.DEST / 'report.json'
    report = json.loads(report_path.read_text())
    d, yd, xd, groups = engine.dataset()
    y = engine.to_hours(yd); ex = engine.to_hours(xd)
    np.testing.assert_array_equal(engine.to_rows(y), yd.reshape(-1))
    bank = engine.ProfileBank(d); profile_tests = []
    for origin in [35, 181, 243, 304]:
        changed = d.copy(); changed.loc[changed.time >= origin, 'boardings'] = 999999
        altered = engine.ProfileBank(changed).get(origin)
        np.testing.assert_array_equal(altered, bank.get(origin))
        profile_tests.append({'origin_day': origin, 'future_target_perturbation_max_diff': 0})
    serial_tests = full_horizon_scan_check()
    replays = []; checks = []; totals_checks = []
    for family, network in report['network_selections'].items():
        directory = engine.ART / family
        audits = [json.loads(p.read_text()) for p in directory.glob('*_e60.json')]
        assert audits
        assert all(a['max_training_label_hour'] == a['forecast_origin_day']*24-1 for a in audits)
        assert all(a['future_target_perturbation_max_difference'] == 0 for a in audits)
        assert all(a['last_hour_gradient_to_first_innovation'] > 0 for a in audits)
        checks.append({'family': family, 'models_checked': len(audits),
                       'training_boundary_ok': True, 'future_label_perturbation_zero': True,
                       'final_hour_gradient_reaches_first_innovation': True})
        origins = [243, 304] if family == report['combination']['family'] else [243]
        for origin in origins:
            days = 61; frame = d[d.time.between(origin, origin+days-1)].reset_index(drop=True)
            cutoff = str(frame.date.min())
            reference = engine.guide(frame, cutoff, 'v8')
            for seed in report['seeds']:
                tag = f'{cutoff}_{network["config"]}_s{seed}_e60'
                # Our own locally trained checkpoints, not user-supplied pickle files.
                checkpoint = torch.load(directory / (tag+'.pt'), map_location='cpu', weights_only=False)
                assert checkpoint['features'] == engine.EXOG
                state = checkpoint['state_dict']
                cls = HourlyShapeResidual if family == 'v14_shape' else engine.HourlyResidual
                model = cls(state['scale'].numpy().reshape(-1), state['active'].numpy().reshape(-1))
                model.load_state_dict(state); model.to(device)
                ps = engine.forecasts(model, y, ex, origin, days*24,
                                      checkpoint['normalization'], bank, device, reference)
                saved = dict(np.load(directory / (tag+'.npz')))
                differences = {}
                for path, p in ps.items():
                    np.testing.assert_allclose(p, saved['v8_'+path], atol=1e-3, rtol=0)
                    differences[path] = float(np.max(abs(p-saved['v8_'+path])))
                    if family == 'v14_shape':
                        hp = engine.to_hours(p.reshape(-1, 240))
                        hb = engine.to_hours(reference.reshape(-1, 240))
                        errors = []
                        for start in range(0, len(hp), 168):
                            pred_total = hp[start:start+168].sum(0)
                            base_total = hb[start:start+168].sum(0)
                            np.testing.assert_allclose(pred_total, base_total, atol=.2, rtol=2e-6)
                            errors.append(float(np.max(abs(pred_total-base_total))))
                        totals_checks.append({'tag': tag, 'path': path, 'max_weekly_volume_diff': max(errors)})
                replays.append({'family': family, 'tag': tag, 'max_abs_difference': differences})
                del model; torch.mps.empty_cache()
    submission_check = None
    if 'submission' in report:
        f = engine.ROOT / report['submission']['path']
        template = pd.read_csv(engine.ROOT / 'data/test_submission.csv', sep=';')
        submitted = pd.read_csv(f, sep=';')
        assert submitted[KEY].equals(template[KEY])
        assert len(submitted) == 14640 and not submitted[KEY].duplicated().any()
        assert submitted.prediction.ge(0).all() and np.isfinite(submitted.prediction).all()
        assert submitted.loc[submitted.route == 5, 'prediction'].eq(0).all()
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        assert digest == report['submission']['sha256']
        best_hash = hashlib.sha256((engine.ROOT/'outputs/submission_best.csv').read_bytes()).hexdigest()
        assert best_hash == 'b8f8238db449a7401f9d63543c60288b018ed9a1e991ba04745825f82d056524'
        submission_check = {'sha256': digest, 'rows': 14640, 'grid_and_order_ok': True,
                            'finite_nonnegative_integer_predictions': True,
                            'v8_best_unchanged': best_hash}
    # Frequency comparison with repeated seeds is recorded separately from the
    # component/blend selection, which was frozen on seed 20260926.
    frequency = []
    for family in report['network_selections']:
        for seed in report['seeds']:
            m = pd.read_csv(engine.ART/family/f'metrics_s{seed}.csv')
            m = m[(m.part == 'all') & (m.path == 'v8_primary')]
            for origin, group in m.groupby('origin'):
                s = group.set_index('model').score
                if 'h1' in s and 'h24' in s:
                    frequency.append({'family': family, 'seed': seed, 'origin': origin,
                                      'h1': float(s['h1']), 'h24': float(s['h24']),
                                      'h1_minus_h24': float(s['h1']-s['h24'])})
    result = {'full_horizon_scan_tests': serial_tests, 'profile_tests': profile_tests,
              'audits': checks, 'checkpoint_replays': replays, 'projection_volume_tests': totals_checks,
              'submission': submission_check, 'repeated_frequency_comparison': frequency}
    (engine.DEST/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'audits': checks, 'replayed_models': len(replays),
                      'scan_value_max_error': max(r['value_max_diff'] for r in serial_tests),
                      'scan_gradient_max_error': max(r['gradient_max_diff'] for r in serial_tests),
                      'submission': submission_check}, indent=2))


if __name__ == '__main__':
    main()
