"""Complete route-cell weather through Dec 2025; retain source/model provenance."""
import json
import hashlib
from pathlib import Path
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data/features/03_weather_and_calendar'
OUT = ROOT / 'data/complete_external_20260927'
FIELDS = ['temperature_2m', 'apparent_temperature', 'relative_humidity_2m',
          'precipitation', 'rain', 'snowfall', 'snow_depth', 'weather_code',
          'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'surface_pressure']


def fetch(name, params):
    path = OUT / (name + '.json')
    if not path.exists():
        r = requests.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=180)
        r.raise_for_status()
        path.write_text(r.text)
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    weights = pd.read_csv(BASE / 'districts/route_weather_cell_weights.csv')
    cells = sorted(weights.weather_cell.unique())
    coords = [c.split(',') for c in cells]
    common = dict(start_date='2025-11-01', end_date='2025-12-31', timezone='Europe/Moscow', elevation='nan')
    params = dict(common, latitude=','.join(c[0] for c in coords), longitude=','.join(c[1] for c in coords),
                  hourly=','.join(FIELDS), models='ecmwf_ifs')
    params['elevation'] = ','.join(['nan'] * len(cells))
    data = fetch('ifs_nov_dec_raw', params)
    assert isinstance(data, list) and len(data) == len(cells)
    parts = []; locations = []
    for cell, result, coord in zip(cells, data, coords):
        # API responses preserve requested location order. Verify snapped IFS cell.
        assert abs(result['latitude'] - float(coord[0])) < .002
        assert abs(result['longitude'] - float(coord[1])) < .002
        d = pd.DataFrame(result['hourly']).rename(columns={'time':'timestamp'})
        d['timestamp'] = pd.to_datetime(d.timestamp).dt.tz_localize('Europe/Moscow')
        d['weather_cell'] = cell
        parts.append(d)
        locations.append(dict(weather_cell=cell, latitude=result['latitude'], longitude=result['longitude'], units=result['hourly_units']))
    future = pd.concat(parts, ignore_index=True)
    # Historical supplied snow depth is city-centre ERA5-seamless; extend the same source.
    zones = pd.read_csv(BASE / 'zones/zones.csv')
    center = zones[zones.zone == 'center'].iloc[0]
    snow_params = dict(common, latitude=float(center.latitude), longitude=float(center.longitude), hourly='snow_depth', models='era5_seamless')
    snow = fetch('era5_snow_nov_dec_raw', snow_params)
    s = pd.Series(snow['hourly']['snow_depth'], index=pd.to_datetime(snow['hourly']['time']).tz_localize('Europe/Moscow'))
    future['snow_depth'] = future.timestamp.map(s)
    future['snow_depth_source'] = 'era5_seamless_center'
    hist = pd.read_parquet(BASE / 'districts/weather_hourly_target_route_cells.parquet')
    full = pd.concat([hist, future], ignore_index=True).sort_values(['weather_cell', 'timestamp'])
    assert len(full) == 9 * 365 * 24 and not full.duplicated(['weather_cell','timestamp']).any()
    assert np.isfinite(full[FIELDS].to_numpy(dtype=float)).all()
    # Weighted route-level weather; weather code remains categorical at cell level,
    # so represent it as shares of precipitation/fog/thunderstorm conditions.
    for name, values in [('fog',[45,48]),('rain',[51,53,55,56,57,61,63,65,66,67,80,81,82]),('snow',[71,73,75,77,85,86]),('thunder',[95,96,99])]:
        full['weather_' + name + '_share'] = full.weather_code.isin(values).astype(float)
    cols = [c for c in FIELDS if c != 'weather_code'] + [c for c in full if c.startswith('weather_') and c.endswith('_share')]
    joined = weights.merge(full, on='weather_cell', validate='many_to_many')
    assert np.allclose(weights.groupby('route').weight.sum(), 1)
    joined[cols] = joined[cols].mul(joined.weight, axis=0)
    route = joined.groupby(['route','timestamp'])[cols].sum(min_count=1).reset_index()
    route['date'] = route.timestamp.dt.strftime('%Y-%m-%d')
    route['hour'] = route.timestamp.dt.hour
    route = route.drop(columns='timestamp').rename(columns={c:'route_weather_' + c for c in cols})
    route.to_parquet(OUT / 'route_hour_weather_2025.parquet', index=False)
    full.to_parquet(OUT / 'weather_cells_full_2025.parquet', index=False)
    report = dict(source='https://open-meteo.com/en/docs/historical-weather-api', attribution='Open-Meteo / ECMWF IFS / ERA5',
                  retrospective_use_authorized_by_user=True, request=params, snow_request=snow_params,
                  locations=locations, rows=len(route), missing=int(route.isna().sum().sum()),
                  note='9 km model cells weighted by stop count, not measured district weather. Snow depth is common city-centre series.',
                  hashes={name:hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in [
                      'ifs_nov_dec_raw.json','era5_snow_nov_dec_raw.json',
                      'route_hour_weather_2025.parquet','weather_cells_full_2025.parquet']})
    (OUT / 'weather_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(rows=len(route), columns=list(route), missing=report['missing'])), flush=True)


if __name__ == '__main__':
    main()
