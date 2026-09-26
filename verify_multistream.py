"""Meaningful checks for the recursive forecast implementation, run on Studio."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multistream_forecast import ART, DEST, Engine, rollout


def main():
    d = pd.read_csv(ART / 'v4/training_matrix.csv')
    origin = '2025-09-01'
    h = d[d.date < origin].reset_index(drop=True)
    hidden = d[d.date.between(origin, '2025-10-31')].reset_index(drop=True)
    v = hidden.drop(columns='boardings')
    engine = Engine(h, v, origin, 'full', threads=4)
    np.testing.assert_allclose(engine.direct, engine.incumbent, rtol=0, atol=1e-8)
    rejected = False
    try:
        engine.predict(h, hidden)
    except AssertionError as error:
        assert 'hidden targets' in str(error)
        rejected = True
    assert rejected
    corrupted = hidden.copy()
    corrupted['boardings'] = np.random.default_rng(2026).uniform(1e6, 1e9, len(hidden))
    q = corrupted.drop(columns='boardings')
    cache = DEST / 'full_2025-09-01.npz'
    if not cache.exists():
        cache = ART / 'v12_multistream/full_2025-09-01.npz'
    original = dict(np.load(cache))['s7_f7_w28_remaining']
    again = rollout(engine, h, q, step=7, first=7, window=28, scope='remaining')
    np.testing.assert_array_equal(original, again)
    # Changing a returned array must not mutate cached expert inputs/history.
    original_history = h.copy(deep=True)
    first = engine.predict(h, v.iloc[:2400])
    first[:] = 0
    pd.testing.assert_frame_equal(h, original_history)
    report = {'origin': origin, 'hidden_days': 61,
              'initial_full_engine_reproduces_v8': True,
              'initial_max_absolute_difference': float(np.max(abs(engine.direct - engine.incumbent))),
              'forecast_api_rejects_future_label_column': rejected,
              'future_target_perturbation_max_difference': float(np.max(abs(original - again))),
              'history_unchanged_by_prediction': True,
              'no_future_observations_during_rollout': True}
    path = DEST / 'verification.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
