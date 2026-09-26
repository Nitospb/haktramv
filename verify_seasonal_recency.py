"""Verify strict seasonal exclusion, final exports and all chronological audits."""
import hashlib
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import seasonal_recency_study as engine
from forecast_artifacts import artifact_directory


def main():
    dest = artifact_directory('seasonal_recency_20260927')
    study = engine.Study()
    exports = json.loads((dest / 'submissions.json').read_text())
    template = pd.read_csv(engine.ROOT / 'data/test_submission.csv', sep=';')
    verified = []
    for item in exports:
        c = item['config'].copy()
        days = study.days(304, c['scope'])
        assert set(study.month[days]).issubset(engine.SCOPES[c['scope']])
        assert not np.isin(study.month[days], [1, 2, 6, 7, 8]).any()
        assert len(days) == (61 if c['scope'] == 'autumn' else 153)
        if item['selected_model'].startswith('profile_'):
            before, _ = study.profile(304, 61, **c)
            changed = study.y.copy()
            excluded = np.ones(365, dtype=bool)
            excluded[days] = False
            changed[excluded] = 999983
            after, _ = study.profile(304, 61, values=changed, **c)
            np.testing.assert_array_equal(before, after)
            exclusion_check = 'All excluded past months AND all future labels changed; prediction identical'
        else:
            model_path = dest / ('final_' + item['selected_model'] + '.cbm')
            audit = json.loads(model_path.with_suffix('.json').read_text())
            assert audit['model_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest()
            model = CatBoostRegressor()
            model.load_model(str(model_path))
            future = study.frame.iloc[304 * 240:].copy()
            before = study.post(model.predict(future[audit['feature_names']]), 304, 61)
            assert 'boardings' not in audit['feature_names']
            exclusion_check = 'Training-month and chronological audits; no target-derived CatBoost input features'
        out = study.frame.iloc[304 * 240:][engine.KEY].copy()
        out['prediction'] = np.rint(before.reshape(-1)).astype(np.int64)
        out = template[engine.KEY].merge(out, on=engine.KEY, validate='one_to_one')
        payload = out.to_csv(index=False, sep=';', lineterminator='\n').encode()
        file = dest / item['submission'].split('/')[-1]
        assert file.read_bytes() == payload
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
        verified.append(dict(model=item['selected_model'], rows=len(out), training_days=len(days),
                             submission_sha256=item['sha256'], check=exclusion_check, byte_exact_replay=True))
    count = 0
    for path in dest.glob('*cat_*.json'):
        audit = json.loads(path.read_text())
        assert set(audit['months']).issubset(engine.SCOPES[audit['scope']])
        assert pd.Timestamp(audit['last_label']) < pd.Timestamp('2025-01-01') + pd.Timedelta(days=audit['origin'])
        assert audit['future_label_perturbation_max_diff'] == 0 and audit['checkpoint_replay_max_diff'] == 0
        count += 1
    report = {'verified_exports': verified, 'chronological_catboost_audits': count,
              'validation_note': 'Autumn-only has no eligible history on Sep1, so its 61-day autumn backtest is unavailable.'}
    (dest / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
