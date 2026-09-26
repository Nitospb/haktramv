"""Export the user's requested pure recursive log1p-ratio submission."""
import hashlib
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import weekly_target_ablation as w
from forecast_artifacts import artifact_directory


def main():
    folder = artifact_directory('v17_weekly_targets')
    stem = '2025-11-01_recursive_logratio'
    checkpoint = folder / (stem + '.cbm')
    audit = json.loads((folder / (stem + '.json')).read_text())
    assert audit['origin_day'] == 304 and audit['target_max_day'] == 303
    assert audit['iterations'] == 500 and audit['forecast_mode'] == 'recursive'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == audit['model_sha256']
    model = CatBoostRegressor()
    model.load_model(str(checkpoint))
    frame, y, ex, groups = w.read_data()
    p = w.predict(model, y, ex, groups, 304, 61, 'logratio', 'recursive')
    changed = y.copy()
    changed[304:] = 999983
    np.testing.assert_array_equal(p, w.predict(model, changed, ex, groups, 304, 61, 'logratio', 'recursive'))
    assert np.isfinite(p).all() and (p >= 0).all()
    rows = frame[frame.time >= 304][w.KEY].copy()
    rows['prediction'] = np.rint(p.reshape(-1)).astype(np.int64)
    template = pd.read_csv(w.ROOT / 'data/test_submission.csv', sep=';')
    out = template[w.KEY].merge(rows, on=w.KEY, how='left', validate='one_to_one')
    assert len(out) == 14640 and out[w.KEY].equals(template[w.KEY])
    assert out.prediction.notna().all() and (out.prediction >= 0).all()
    assert out.date.min() == '2025-11-01' and out.date.max() == '2025-12-31'
    assert (out[out.route == 5].prediction == 0).all()
    name = 'submission_weekly_logratio_v17.csv'
    file = w.ROOT / 'outputs' / name
    out.to_csv(file, sep=';', index=False, lineterminator='\n')
    (folder / name).write_bytes(file.read_bytes())
    np.savez_compressed(folder / 'final_predictions.npz', recursive_logratio=p)
    report = {
        'submission': 'outputs/' + name,
        'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
        'rows': len(out), 'prediction_total': int(out.prediction.sum()),
        'model': 'recursive_logratio', 'model_sha256': audit['model_sha256'],
        'forecast_days': 61, 'step_days': 7, 'training_last_date': '2025-10-31',
        'training_rows': audit['rows'], 'iterations': audit['iterations'],
        'future_label_perturbation_max_diff': 0., 'checkpoint_replayed': True,
        'template_keys_and_order_verified': True, 'v8_is_component': False,
        'public_score': None, 'user_requested': True,
        'promotion_to_public_best': False,
        'development_september_october_score': 0.8630101892505414,
        'decision': 'Experimental submission requested by user; keep v8 public best 0.89214.'}
    # Re-exporting identical bytes must retain their known competition result.
    ledger_path = w.ROOT / 'outputs/leaderboard_results.json'
    if ledger_path.exists():
        for result in json.loads(ledger_path.read_text()):
            if (result.get('submission') == report['submission']
                    and result.get('sha256') == report['sha256']
                    and result.get('score') is not None):
                report.update(public_score=result['score'], public_score_source=result.get('source'),
                              submitted_at=result.get('submitted_at'), file_mapping=result.get('file_mapping'),
                              decision=result.get('decision', report['decision']))
    (folder / 'submission_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
