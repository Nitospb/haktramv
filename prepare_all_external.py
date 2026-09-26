"""Assemble all available external feature families, with full-year coverage audit.

Raw passenger/validator records are not external features. They do not contain
the contest's hidden targets; future permission cannot create missing values.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/complete_external_20260927'
BASE = ROOT / 'data/features'
KEY = ['route', 'date', 'hour']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix_path = next(p for p in [ROOT / 'outputs/v4/training_matrix.csv', ROOT / 'outputs/studio/v4/training_matrix.csv'] if p.exists())
    d = pd.read_csv(matrix_path).sort_values(['date','route','hour']).reset_index(drop=True)
    target_reference = d[KEY + ['boardings']].copy()
    d = d.merge(pd.read_csv(ROOT / 'data/schedule_research_20260927/archived_supply_hourly_2025.csv'), on=KEY, how='left', validate='one_to_one')
    model_path = ROOT / 'outputs/fleet_feature_20260927/304_fleet.txt'
    previous_cols = next(line.split('=',1)[1].split() for line in model_path.read_text().splitlines() if line.startswith('feature_names='))
    d[previous_cols] = d[previous_cols].fillna(-1)
    original = list(d)
    joins = []

    def join(frame, key, source):
        nonlocal d
        cols = [c for c in frame if c not in d or c in key]
        assert not frame.duplicated(key).any(), source
        before = len(d)
        d = d.merge(frame[cols], on=key, how='left', validate='many_to_one', sort=False)
        assert len(d) == before
        joins.append(dict(source=source, features=[c for c in cols if c not in key], rows=len(frame)))

    poi_path = BASE / '06_schedules/gtfs_hourly_features_poi.csv'
    join(pd.read_csv(poi_path), KEY, str(poi_path.relative_to(ROOT)))
    for name in ['route_hour_weather_2025', 'route_hour_road_incidents_2025']:
        join(pd.read_parquet(OUT / (name + '.parquet')), KEY, name)

    # Historical climate is a reference, distinct from actual 2025 local weather.
    norms = pd.read_parquet(BASE / '03_weather_and_calendar/zones/climatology_2022_2023.parquet')
    weights = pd.read_csv(BASE / '03_weather_and_calendar/zones/route_zone_weights.csv')
    nc = list(norms.select_dtypes(include='number').columns.drop('hour'))
    z = weights.merge(norms, on='zone', validate='many_to_many')
    assert np.allclose(weights.groupby('route').weight.sum(), 1)
    z[nc] = z[nc].mul(z.weight, axis=0)
    norm = z.groupby(['route','month_day','hour'])[nc].sum(min_count=1).reset_index()
    norm = norm.rename(columns={c:'climate_' + c for c in nc})
    d['month_day'] = d.date.str[5:]
    join(norm, ['route','month_day','hour'], '2022-2023 route-weighted climate reference')
    d = d.drop(columns='month_day')
    for c in nc:
        if 'route_weather_' + c in d:
            d['weather_anomaly_' + c] = d['route_weather_' + c] - d['climate_' + c]

    # All previously downloaded Moscow/Kazakhstan historical seasonal priors.
    for country in ['moscow_modes', 'kazakhstan']:
        priors = json.loads((ROOT / 'data' / ('external_' + country) / 'priors.json').read_text())
        for name, item in priors.items():
            assert len(item['index']) == 12
            d['donor_' + country + '_' + name] = np.asarray(item['index'])[d.month.astype(int) - 1]

    other = BASE / '08_external_seasonality/other_cities'
    daily = pd.read_csv(other / 'external_daily_index.csv').rename(columns={'daily_index':'donor_malaysia_daily_index'})
    join(daily, ['date'], 'Malaysia daily retrospective external index')
    hourly = pd.read_parquet(other / 'external_hourly.parquet')
    hourly['date'] = pd.to_datetime(hourly.date)
    for source, h in hourly.groupby('source'):
        h = h.copy()
        h['month'] = h.date.dt.month; h['dow'] = h.date.dt.dayofweek
        if source == 'malaysia_komuter':
            h = h[h.date.dt.year == 2025].copy()
            h['donor_komuter_daily_volume'] = h.groupby('date').ridership.transform('sum')
            h['donor_komuter_hour_share'] = h.ridership / h.donor_komuter_daily_volume.clip(lower=1)
            h['date'] = h.date.dt.strftime('%Y-%m-%d')
            join(h[['date','hour','donor_komuter_daily_volume','donor_komuter_hour_share']], ['date','hour'], 'Malaysia observed 2025 hourly transport')
        else:
            h = h[h.date.dt.year == 2024].copy()
            h['share'] = h.ridership / h.groupby('date').ridership.transform('sum').clip(lower=1)
            template = h.groupby(['month','dow','hour']).share.mean().rename('donor_' + source + '_2024_hour_share').reset_index()
            join(template, ['month','dow','hour'], source + ' historical 2024 template')
    seoul = pd.read_parquet(other / 'seoul_bus_route_hour_2023_2024.parquet')
    seoul = seoul.groupby(['year_month','hour'])[['boardings','alightings']].sum().reset_index()
    for c in ['boardings','alightings']:
        seoul['donor_seoul_' + c + '_hour_share'] = seoul[c] / seoul.groupby('year_month')[c].transform('sum').clip(lower=1)
    seoul['month'] = seoul.year_month % 100
    join(seoul.groupby(['month','hour'])[[c for c in seoul if c.startswith('donor_')]].mean().reset_index(), ['month','hour'], 'Seoul 2023-2024 monthly hourly shares')

    d = d.sort_values(['date','route','hour']).reset_index(drop=True)
    assert len(d) == 87600 and not d.duplicated(KEY).any()
    pd.testing.assert_frame_equal(d[KEY + ['boardings']], target_reference)
    features = [c for c in d.select_dtypes(include=['number', 'bool']) if c != 'boardings']
    assert all(c in features for c in ['drone_report_count','connectivity_report_count','mobile_internet_restrictions_possible','route_weather_snow_depth','road_crash_700m_past3h'])
    coverage = []
    for c in features:
        hist, future = d.loc[d.date < '2025-11-01', c], d.loc[d.date >= '2025-11-01', c]
        coverage.append(dict(feature=c, historical_missing=int(hist.isna().sum()), future_missing=int(future.isna().sum()),
                             historical_unique=int(hist.nunique()), future_unique=int(future.nunique()),
                             future_nonzero=int((future.notna() & future.ne(0)).sum())))
    pd.DataFrame(coverage).to_csv(OUT / 'feature_coverage.csv', index=False)
    d.to_parquet(OUT / 'all_features_matrix.parquet', index=False)
    report = dict(rows=len(d), feature_count=len(features), features=features, previous_fleet_features=previous_cols,
                  additional_feature_count=len(set(features) - set(previous_cols)), joins=joins,
                  original_matrix_features=[c for c in original if c in features],
                  future_dates=['2025-11-01','2025-12-31'], retrospectively_observed_inputs_authorized=True,
                  excluded=dict(boardings='Target; unknown Nov-Dec.', transaction_derived='Observed Nov-Dec fleet, device and transaction statistics unavailable. Historical observed fleet used in separate model.',
                                zones_actual_weather='Original zones end October. Complete local IFS weather used instead; zone climatology retained.',
                                raw_ids_text_geometries='Represented by route, stop-weighted weather, transfer, schedule and POI features, not arbitrary strings.',
                                traffic_speed_intensity='No usable open 2025 road-hour archive located; nearby crash features separately identified.'),
                  missing_handling='NaN converted to -1 for model input. Original matrix already contains some zero-filled source fields; zero reports are not proof of no disruption.')
    (OUT / 'feature_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(rows=len(d), features=len(features), missing_future=sum(r['future_missing'] for r in coverage))), flush=True)


if __name__ == '__main__':
    main()
