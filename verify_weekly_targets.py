"""Check weekly recursion against an analytic path and perturb hidden labels."""
import json
from pathlib import Path

import numpy as np

import weekly_target_ablation as w
from forecast_artifacts import artifact_directory


class ConstantChange:
    def __init__(self, change):
        self.change = change
        self.calls = []

    def predict(self, frame):
        self.calls.append(frame.lag_1.to_numpy().copy())
        return np.full(len(frame), self.change)


def main():
    _, y, ex, groups = w.read_data()
    origin, horizon = 243, 61
    anchors = origin - 7 + np.arange(horizon) % 7
    actual = np.array([0., 1., 100., 1000.])
    base = np.array([10., 0., 110., 900.])
    for kind in ('absolute', 'delta', 'relative', 'logratio'):
        np.testing.assert_allclose(w.inverse_target(w.transformed_target(actual, base, kind), base, kind), actual, atol=1e-10)
    m = ConstantChange(np.log(1.02))
    p = w.predict(m, y, ex, groups, origin, horizon, 'logratio', 'recursive')
    expected = (y[anchors] + 1) * 1.02 ** (1 + np.arange(horizon)[:, None] // 7) - 1
    expected[:, groups.route.to_numpy() == 5] = 0
    np.testing.assert_allclose(p, expected, atol=1e-8, rtol=1e-12)
    np.testing.assert_array_equal(m.calls[1], p[:7].reshape(-1))
    assert [len(c) // 240 for c in m.calls] == [7] * 8 + [5]
    for mode in ('recursive', 'direct'):
        repeated = w.predict(ConstantChange(0.), y, ex, groups, origin, horizon, 'logratio', mode)
        np.testing.assert_allclose(repeated, y[anchors], atol=1e-8)
    changed = y.copy()
    changed[181:] = 1234567
    training_audit = {}
    for mode in ('recursive', 'direct'):
        x, target, weights, audit = w.training_data(y, ex, groups, 181, 'logratio', mode)
        changed_x, changed_target, changed_weights, _ = w.training_data(changed, ex, groups, 181, 'logratio', mode)
        np.testing.assert_array_equal(x.to_numpy(), changed_x.to_numpy())
        np.testing.assert_array_equal(target, changed_target)
        np.testing.assert_array_equal(weights, changed_weights)
        training_audit[mode] = {'rows': len(x), 'latest_training_day': audit['target_max_day'],
                                'future_perturbation_changes': 0}
        del x, target, weights, changed_x, changed_target, changed_weights
    dest = artifact_directory('v17_weekly_targets')
    audits = json.loads((dest / 'audit_summary.json').read_text())
    assert len(audits) == 20
    for item in audits:
        assert item['future_label_perturbation_max_diff'] == 0
        assert item['checkpoint_replay_max_diff'] == 0
        assert item['target_max_day'] < item['origin_day']
    report = {
        'target_inverse_with_zero_anchors': 'passed',
        'analytic_nine_block_rollout_max_diff': float(np.abs(p - expected).max()),
        'recursive_block_days': [len(c) // 240 for c in m.calls],
        'second_week_receives_own_first_week': True,
        'zero_change_equals_repeat_last_week': True,
        'training_future_label_perturbation': training_audit,
        'trained_models_with_hidden_label_and_reload_checks': len(audits),
    }
    (dest / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
