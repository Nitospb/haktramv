"""Replay and package the three final candidates on Studio; never train here."""
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from night_forecast_common import ROOT, OUT, KEY, score, save_json
from night_calendar_analog_study import CalendarData


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path):
    assert path.read_text().splitlines()[0] == 'route;date;hour;prediction'
    z = pd.read_csv(path, sep=';')
    assert len(z) == 14640 and list(z) == KEY+['prediction']
    assert not z.duplicated(KEY).any() and z.notna().all().all()
    assert np.isfinite(z.prediction).all() and z.prediction.ge(0).all()
    assert (z.prediction == np.rint(z.prediction)).all()
    expected = pd.MultiIndex.from_product([
        [1,5,7,11,12,17,25,26,28,50],
        pd.date_range('2025-11-01', '2025-12-31').strftime('%Y-%m-%d'),
        range(24)], names=KEY)
    assert pd.MultiIndex.from_frame(z[KEY]).sort_values().equals(expected.sort_values())
    return z


def compare_prediction(frame, p, path, submission=False, column='prediction'):
    z = pd.read_csv(path, sep=';' if submission else ',')
    aligned = frame[KEY].merge(z, on=KEY, validate='one_to_one', how='left')
    assert len(aligned) == 14640 and aligned[column].notna().all()
    if submission:
        np.testing.assert_array_equal(np.rint(p).astype(int), aligned[column])
    else:
        np.testing.assert_allclose(p, aligned[column], rtol=1e-10, atol=1e-8)


def main():
    public_best = ROOT/'outputs/submission_best.csv'
    best_hash = 'f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
    # Studio's top-level best file may be an older synchronized artifact.
    # Check the explicit copy of the authoritative local best and preserve both.
    reference = OUT/'public_best_reference.csv'
    assert digest(reference) == best_hash
    remote_best_before = digest(public_best)
    data = CalendarData(); data.causal_check()
    folder = OUT/'deliverables'; folder.mkdir(exist_ok=True)
    direct = ROOT/'outputs/longer_moscow_external_20260927'
    analog = OUT/'calendar_analog'
    metrics = []
    for origin in [181, 212, 243, 304]:
        model = lgb.Booster(model_file=str(direct/(str(origin)+'_all_direct.txt')))
        frame = data.frame(origin)
        p = model.predict(frame[model.feature_name()], num_iteration=1600, num_threads=6)
        frame, p = data.prediction(origin, p)
        if origin == 304:
            compare_prediction(frame, p, direct/'submission_direct_1600_autumn.csv', True)
            model.save_model(str(folder/'direct_1600.txt'), num_iteration=1600)
            compact = lgb.Booster(model_file=str(folder/'direct_1600.txt'))
            np.testing.assert_array_equal(model.predict(frame[model.feature_name()], num_iteration=1600, num_threads=6),
                                          compact.predict(frame[compact.feature_name()], num_threads=6))
        else:
            compare_prediction(frame, p, direct/('forecast_'+str(origin)+'.csv'), column='direct_1600')
            metrics.append(dict(model='direct_1600', origin=origin, score=score(frame.boardings, p)))
    for origin in [151, 181, 212, 243, 304]:
        frame, p = data.prediction(origin, data.analog(origin, 61)['term'])
        if origin == 304:
            compare_prediction(frame, p, analog/'submission_baseline_term.csv', True)
        else:
            compare_prediction(frame, p, analog/('baseline_term_'+str(origin)+'.csv'))
            metrics.append(dict(model='calendar_analog', origin=origin, score=score(frame.boardings, p)))
    # Recreate the separate route-5 scenario from the verified direct candidate.
    subprocess.run([sys.executable, str(ROOT/'attach_route5_to_external.py'), '--base',
                    'outputs/longer_moscow_external_20260927/submission_direct_1600_autumn.csv'], check=True)
    sources = [
        ('submission_01_direct.csv', direct/'submission_direct_1600_autumn.csv'),
        ('submission_02_calendar.csv', analog/'submission_baseline_term.csv'),
        ('submission_03_direct_route5.csv', direct/'submission_moscow_with_route5.csv'),
    ]
    manifests = []
    for name, source in sources:
        target = folder/name; shutil.copyfile(source, target)
        z = validate(target)
        manifests.append(dict(file=name, sha256=digest(target), source=str(source.relative_to(ROOT)),
            rows=len(z), total=int(z.prediction.sum()), public_score=None))
    base = pd.read_csv(folder/'submission_01_direct.csv', sep=';')
    five = pd.read_csv(folder/'submission_03_direct_route5.csv', sep=';')
    pd.testing.assert_frame_equal(base[base.route != 5], five[five.route != 5])
    assert five.loc[(five.route==5)&(five.date<'2025-12-16'), 'prediction'].eq(0).all()
    assert five.loc[(five.route==5)&five.date.between('2025-12-16','2025-12-22'), 'prediction'].sum() == 40000
    assert digest(reference) == best_hash
    assert digest(public_best) == remote_best_before
    save_json(folder/'verification.json', dict(submissions=manifests, replayed_forecasts=9,
        validation_scores=metrics, target_perturbation_check=True, grid_check=True,
        compact_model_replay_exact=True, compact_model_sha256=digest(folder/'direct_1600.txt'),
        route5_first_week_sum=40000, route5_other_nine_routes_identical=True,
        public_best_sha256=best_hash, public_best_score=.89402, public_best_changed=False,
        studio_existing_best_sha256=remote_best_before, studio_existing_best_changed=False,
        candidate_selection='Direct1600 is the previously explored autumn-selected checkpoint; calendar analog is an alternative standalone method; route5 is an unvalidated public-aggregate scenario.',
        limitation='These overlapping development windows were repeatedly inspected; local scores do not establish a leaderboard improvement. No hidden Moscow target inputs.'))
    shutil.copyfile(direct/'route5_public_scenario_report.json', folder/'route5_scenario.json')
    print(json.dumps(dict(submissions=manifests, scores=metrics), ensure_ascii=False))


if __name__ == '__main__':
    main()
