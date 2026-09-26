"""Spring analogs and a 2023–2024-only tram seasonal prior.

All route templates and calendar weights are fitted before each forecast origin.
Service/weather covariates remain retrospective, as in the incumbent experiment.
No Nov–Dec labels are available; development folds have already been inspected.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_models import DEST as V4, F
from service_models import adjust, regime
from train_model import score

ROOT = V4.parent
DEST = ROOT / 'v10_spring'
KEY = ['route', 'date', 'hour']
CELL = ['route', 'daytype', 'hour']
MONTHS = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
FOLDS = [('2025-07-01', '2025-08-31'), ('2025-08-01', '2025-09-30'),
         ('2025-09-01', '2025-10-31'), ('2025-10-01', '2025-10-31')]


def external_data():
    path = F / '08_external_seasonality/moscow_monthly_passengers.csv'
    x = pd.read_csv(path, sep=';')
    x = x[x.TransportType == 'Трамвай'].copy()
    x['year'] = x.Year.astype(int)
    x['month'] = x.Month.map(dict(zip(MONTHS, range(1, 13))))
    x['passengers'] = x.PassengerTraffic.astype(float)
    x['date'] = pd.to_datetime(dict(year=x.year, month=x.month, day=1))
    x['daily'] = x.passengers / x.date.dt.days_in_month
    assert not x.duplicated(['year', 'month']).any()
    return x, hashlib.sha256(path.read_bytes()).hexdigest()


def seasonal_prior(tr):
    """Remove the workday/off-day mixture before estimating a month effect."""
    clean = tr[(regime(tr) == 0) & (tr.service_incident_fraction == 0)]
    daily = clean.groupby(['route', 'date', 'workday', 'holiday']).boardings.agg(['sum', 'count']).reset_index()
    daily = daily[(daily['count'] == 24) & (daily.route != 5)]
    daily['kind'] = np.where(daily.holiday > 0, 'holiday', np.where(daily.workday > 0, 'work', 'off'))
    rates = daily.pivot_table(index='route', columns='kind', values='sum', aggfunc='median')
    off = float((rates['off'] / rates['work']).median())
    holiday = float((rates['holiday'] / rates['work']).median()) if 'holiday' in rates else off
    off = float(np.clip(off, .2, .95)); holiday = float(np.clip(holiday, .1, .95))
    c = pd.read_csv(F / '03_weather_and_calendar/weather_calendar_extended.csv',
                    usecols=['date', 'is_working_day', 'is_public_holiday'])
    c['date'] = pd.to_datetime(c.date)
    c = c[c.date.dt.year.isin([2023, 2024])].copy()
    c['year'] = c.date.dt.year; c['month'] = c.date.dt.month
    c['exposure'] = np.where(c.is_working_day, 1., np.where(c.is_public_holiday, holiday, off))
    exposure = c.groupby(['year', 'month']).exposure.sum()
    x, _ = external_data()
    x = x[x.year.isin([2023, 2024])].copy().merge(exposure, on=['year', 'month'])
    assert len(x) == 24
    x['rate'] = x.passengers / x.exposure
    x['index'] = x['rate'] / x.groupby('year')['rate'].transform('mean')
    index = x.groupby('month')['index'].mean().to_dict()
    return index, {'years': [2023, 2024], 'off_day_weight': off, 'holiday_weight': holiday,
                   'monthly_index': index}


def template(tr, va, source, power, index, recent=False):
    normal = tr[(regime(tr) == 0) & (tr.service_incident_fraction == 0) & (tr.holiday == 0)].copy()
    normal['value'] = normal.boardings / normal.month.map(index).to_numpy() ** power
    fallback = normal.groupby(CELL).value.median().rename('fallback')
    if source == 'spring':
        subset = normal[normal.month.isin([3, 4])]
    elif source == 'no_january':
        subset = normal[(normal.month != 1) & (normal.summer == 0)]
    else:
        subset = normal[normal.summer == 0]
    stats = subset.groupby(CELL).value.median().rename('estimate')

    def predict(v):
        z = v[CELL].merge(stats, on=CELL, how='left').merge(fallback, on=CELL, how='left')
        return z.estimate.fillna(z.fallback).fillna(0).to_numpy(copy=True) * v.month.map(index).to_numpy() ** power

    tp = predict(tr); vp = predict(va)
    if recent:
        # Estimate route changes on the latest school-season weeks, excluding disruptions.
        school = normal[normal.summer == 0]
        dates = school[school.time > school.time.max() - 56].copy()
        dates['expected'] = predict(dates)
        ratios = dates.groupby('route')[['boardings', 'expected']].sum()
        r = ((ratios.boardings + 200) / (ratios.expected + 200)).clip(.8, 1.25) ** .5
        vp *= va.route.map(r).fillna(1).to_numpy()
        tp *= tr.route.map(r).fillna(1).to_numpy()
    out = adjust(tr, va, tp, vp)
    out[va.route.to_numpy() == 5] = 0
    return out


def combine(v, incumbent, analog, strength, levels):
    z = v[['route', 'month']].copy()
    z['reference'] = np.asarray(incumbent); z['analog'] = np.asarray(analog)
    sums = z.groupby(['route', 'month'])[['reference', 'analog']].transform('sum')
    ratio = sums.reference.to_numpy() / np.maximum(1e-9, sums.analog.to_numpy())
    if levels == 'anchored':
        q = analog * ratio
    elif levels == 'bounded':
        relative = sums.analog.to_numpy() / np.maximum(1e-9, sums.reference.to_numpy())
        q = analog * ratio * np.clip(relative, .95, 1.05)
    else:
        q = np.asarray(analog).copy()
    q[sums.analog.to_numpy() < 1e-9] = incumbent[sums.analog.to_numpy() < 1e-9]
    # The hypothesis concerns the autumn return to the school/work timetable.
    active = v.month.isin([9, 10, 11, 12]).to_numpy()
    return np.maximum(0, incumbent + active * strength * (q - incumbent))


def write_submission(v, prediction, path, incumbent=None, anchored=False):
    out = v[KEY].copy()
    if anchored:
        out['prediction'] = np.floor(prediction).astype(int)
        for _, indices in v.groupby(['route', 'month']).groups.items():
            ix = np.asarray(list(indices))
            needed = int(np.sum(incumbent[ix]) - out.iloc[ix].prediction.sum())
            assert 0 <= needed <= len(ix), needed
            take = ix[np.argsort(-(prediction[ix] - np.floor(prediction[ix])), kind='stable')[:needed]]
            out.loc[take, 'prediction'] += 1
    else:
        out['prediction'] = np.rint(prediction).astype(int)
    assert len(out) == 14640 and not out.duplicated(KEY).any()
    assert np.isfinite(out.prediction).all() and (out.prediction >= 0).all()
    out.to_csv(path, sep=';', index=False)
    return out


def main():
    DEST.mkdir(exist_ok=True)
    d = pd.read_csv(V4 / 'training_matrix.csv')
    variants = {
        'spring_raw': ('spring', 0., False),
        'spring_recent': ('spring', 0., True),
        'spring_season_half': ('spring', .5, True),
        'spring_season_full': ('spring', 1., True),
        'no_january': ('no_january', 0., True),
        'prior_2023_2024': ('all', .5, True),
    }
    rows = []; frames = {}; incumbents = {}; forecasts = {}; priors = {}
    for start, end in FOLDS:
        tr = d[d.date < start].reset_index(drop=True)
        v = d[(d.date >= start) & (d.date <= end)].reset_index(drop=True)
        ref = pd.read_csv(ROOT / 'v8' / ('validation_' + start + '.csv'), sep=';')
        assert ref[KEY].equals(v[KEY])
        p0 = ref.prediction.to_numpy(); frames[start] = v; incumbents[start] = p0
        index, prior = seasonal_prior(tr); priors[start] = prior
        forecasts[start] = {}
        for name, (source, power, recent) in variants.items():
            p = template(tr, v, source, power, index, recent)
            forecasts[start][name] = p
            for strength in [.25, .5, 1.]:
                for levels in ['anchored', 'bounded', 'free']:
                    q = combine(v, p0, p, strength, levels)
                    rows.append(dict(fold=start, model=name, strength=strength, levels=levels,
                                     score=score(v.boardings, q), incumbent=score(v.boardings, p0)))
        np.savez(DEST / (start + '.npz'), **forecasts[start])
        best = max([r for r in rows if r['fold'] == start], key=lambda r: r['score'])
        print('FOLD', start, 'incumbent', score(v.boardings, p0), 'best diagnostic', best, flush=True)
    r = pd.DataFrame(rows); r.to_csv(DEST / 'metrics.csv', index=False)
    # Record the initial Aug–Sep choice, which fails the already-used autumn
    # diagnostics. The final candidate is an explicitly post-hoc, bounded blend.
    rank = r[r.fold == '2025-08-01'].sort_values(['score', 'strength'], ascending=[False, True])
    initial = rank.iloc[0]
    restricted = r[(r.fold >= '2025-08-01') & (r.levels == 'bounded') & (r.strength <= .5)].copy()
    restricted['gain'] = restricted.score - restricted.incumbent
    stability = restricted.groupby(['model', 'strength', 'levels']).gain.agg(['min', 'mean'])
    stability = stability[stability['min'] > 0].sort_values(['min', 'mean'], ascending=False)
    assert len(stability), 'No consistently improved bounded candidate; keep incumbent.'
    name, strength, levels = stability.index[0]
    strength = float(strength)
    selected_rows = r[(r.model == name) & (r.strength == strength) & (r.levels == levels)]
    report = {
        'selected': {'model': name, 'strength': strength, 'levels': levels},
        'scores': selected_rows.to_dict('records'),
        'selection': 'Post-hoc development selection: among blends <=50% with analog route-month volume bounded to +/-5%, maximize the minimum score gain on Aug–Sep, Sep–Oct, October. These windows overlap and were inspected. No independent validation claim.',
        'initial_aug_sep_selection_rejected': {'model': str(initial.model), 'strength': float(initial.strength), 'levels': str(initial.levels)},
        'final_route_month_volume_change_bound': .05 * strength,
        'prior': '2023–2024 tram totals only. 2020–2022 excluded. Calendar exposure fitted from past route labels only.',
        'actual_incumbent_public_score': .89214, 'candidate_public_score': None,
        'exogenous_caveat': 'Retrospective service calendar supplied by user.',
        'source_url': 'https://github.com/ArtNikVal/Analyseofmoscowtaransportproject',
        'source_sha256': external_data()[1],
        'nov_dec_target_available': False,
    }
    report['all_autumn_diagnostics_improve'] = bool((selected_rows[selected_rows.fold >= '2025-09-01'].score > selected_rows[selected_rows.fold >= '2025-09-01'].incumbent).all())
    for start, _ in FOLDS:
        v = frames[start]
        p = combine(v, incumbents[start], forecasts[start][name], strength, levels)
        out = v[KEY + ['boardings']].copy(); out['prediction'] = p
        out.to_csv(DEST / ('validation_' + start + '.csv'), sep=';', index=False)
    tr = d[d.date < '2025-11-01'].reset_index(drop=True)
    v = d[d.date >= '2025-11-01'].reset_index(drop=True)
    ref = pd.read_csv(ROOT / 'v8/submission.csv', sep=';')
    assert ref[KEY].equals(v[KEY])
    p0 = ref.prediction.to_numpy()
    index, prior = seasonal_prior(tr); priors['2025-11-01'] = prior
    source, power, recent = variants[name]
    analog = template(tr, v, source, power, index, recent)
    # Changing unknown future labels must not change any forecast.
    changed = v.copy(); changed['boardings'] = np.arange(len(v)) * 7919.
    np.testing.assert_allclose(template(tr, changed, source, power, index, recent), analog, rtol=0, atol=0)
    report['future_target_perturbation_test'] = 'passed: arbitrary Nov–Dec target values have no effect'
    prediction = combine(v, p0, analog, strength, levels)
    out = write_submission(v, prediction, DEST / 'submission.csv', p0, levels == 'anchored')
    comparison = v[KEY + ['month']].copy()
    comparison['incumbent'] = p0; comparison['candidate'] = out.prediction
    comparison.groupby(['route', 'month'])[['incumbent', 'candidate']].sum().to_csv(DEST / 'monthly_changes.csv')
    report['forecast_monthly_totals'] = comparison.groupby('month')[['incumbent', 'candidate']].sum().to_dict('index')
    report['mean_absolute_change'] = float(np.mean(np.abs(out.prediction.to_numpy() - p0)))
    report['target_95_achieved'] = False
    (DEST / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (DEST / 'seasonal_priors.json').write_text(json.dumps(priors, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
