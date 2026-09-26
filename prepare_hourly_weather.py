"""Complete the supplied retrospective hourly weather using the same public IFS API."""
import json
from pathlib import Path
from urllib.request import urlopen,Request
from urllib.parse import urlencode
import pandas as pd

ROOT=Path(__file__).resolve().parent
F=ROOT/'data/features/03_weather_and_calendar/districts'
OUT=ROOT/'data/enriched';OUT.mkdir(exist_ok=True)
weights=pd.read_csv(F/'route_weather_cell_weights.csv')
cells=sorted(weights.weather_cell.unique())
cols=['temperature_2m','apparent_temperature','relative_humidity_2m','precipitation',
      'rain','snowfall','weather_code','cloud_cover','wind_speed_10m','wind_gusts_10m','surface_pressure']
rawfile=OUT/'weather_nov_dec_ifs.json'
if not rawfile.exists():
    args={'latitude':','.join(c.split(',')[0] for c in cells),'longitude':','.join(c.split(',')[1] for c in cells),
        'start_date':'2025-11-01','end_date':'2025-12-31','hourly':','.join(cols),
        'timezone':'Europe/Moscow','models':'ecmwf_ifs','elevation':','.join(['nan']*len(cells))}
    url='https://archive-api.open-meteo.com/v1/archive?'+urlencode(args)
    request=Request(url,headers={'User-Agent':'TramForecastResearch/1.0'})
    with urlopen(request,timeout=60) as response:payload=response.read()
    rawfile.write_bytes(payload)
    (OUT/'weather_source.json').write_text(json.dumps({'url':url,'documentation':'https://open-meteo.com/en/docs/historical-weather-api','type':'retrospective reanalysis, not an as-of forecast','cells':cells},indent=2))
payload=json.loads(rawfile.read_text());assert isinstance(payload,list) and len(payload)==len(cells)
parts=[]
for cell,item in zip(cells,payload):
    d=pd.DataFrame(item['hourly']);d['timestamp']=pd.to_datetime(d.pop('time')).dt.tz_localize('Europe/Moscow');d['weather_cell']=cell
    assert len(d)==61*24;parts.append(d)
future=pd.concat(parts,ignore_index=True)
past=pd.read_parquet(F/'weather_hourly_target_route_cells.parquet')[['timestamp','weather_cell']+cols]
d=pd.concat([past,future],ignore_index=True)
assert not d.duplicated(['timestamp','weather_cell']).any()
d=d.merge(weights[['route','weather_cell','weight']],on='weather_cell',validate='many_to_many')
for c in cols:d[c]=d[c]*d.weight
d=d.groupby(['route','timestamp'])[cols].sum(min_count=1).reset_index()
d['date']=d.timestamp.dt.strftime('%Y-%m-%d');d['hour']=d.timestamp.dt.hour
d=d.drop(columns='timestamp').rename(columns={c:'hourly_'+c for c in cols})
assert len(d)==87600 and not d.duplicated(['route','date','hour']).any()
d.to_csv(OUT/'hourly_weather_by_route.csv.gz',index=False)
print('Hourly route weather complete:',d.shape,flush=True)
