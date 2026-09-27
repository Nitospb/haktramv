import hashlib
import json
import shutil

import pytest
from fastapi.testclient import TestClient
from app.main import app, ARTIFACTS
from app.engine import ForecastEngine


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_preserved_winner(client):
    assert client.get('/readyz').json()['rows']==14640
    response=client.get('/v1/forecast.csv')
    assert response.status_code==200
    assert hashlib.sha256(response.content).hexdigest()=='f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
    assert client.get('/v1/forecast').json()['total']==12902502


@pytest.mark.parametrize('params',[
    {'start':'2026-01-01'},{'start':'2025-12-02','end':'2025-12-01'},
    {'routes':999},{'routes':'bad'},{'hour_from':15,'hour_to':12},
    {'weather_factor':-1},{'weather_factor':'nan'},{'season_factor':'inf'},
    {'event_factor':4},{'stop_id':'stop-1'},{'segment_id':'1'},
    {'aggregation':'year'},{'horizon':'year'},{'route':17},
])
def test_invalid_inputs(client,params):
    assert client.get('/v1/forecast',params=params).status_code==422
    assert client.get('/v1/forecast.csv',params=params).status_code==422


def test_hourly_rounding_and_aggregation(client):
    params={'start':'2025-11-01','end':'2025-11-30','routes':[1,17], 'weather_factor':1.1,'event_factor':.7}
    h=client.get('/v1/forecast',params=params).json()
    assert h['row_count']==1440 and h['scenario']
    for agg,count in [('day',60),('month',2),('total',2)]:
        d=client.get('/v1/forecast',params=dict(params,aggregation=agg)).json()
        assert d['row_count']==count and sum(r['prediction'] for r in d['rows'])==h['total']
    assert client.get('/v1/forecast').json()['total']==12902502


def test_zero_scenario_and_hour_bounds(client):
    d=client.get('/v1/forecast',params={'start':'2025-11-01','end':'2025-11-01','routes':17,'hour_from':8,'hour_to':10,'season_factor':0}).json()
    assert d['row_count']==2 and d['total']==0 and {r['hour'] for r in d['rows']}=={8,9}


def test_sources_and_geometry(client):
    sources={s['id']:s for s in client.get('/v1/sources').json()['sources']}
    assert sources['road_traffic']['status']=='missing'
    assert sources['road_crashes']['status']=='experiment_only'
    assert client.get('/v1/geo/stops?route=17').json()['stops']
    assert client.get('/v1/geo/routes?route=17').json()['features']
    assert client.get('/v1/geo/routes?route=999').status_code==422
    assert client.get('/').status_code==200
    assert client.get('/openapi.json').status_code==200


def test_corrupt_bundle_fails_closed(tmp_path):
    target=tmp_path/'artifacts';shutil.copytree(ARTIFACTS,target)
    with (target/'base.csv').open('a') as f:f.write('corrupt\n')
    with pytest.raises(RuntimeError,match='Invalid artifact'):ForecastEngine(target)


def test_incomplete_cancellation_grid_fails_even_with_updated_digest(tmp_path):
    target=tmp_path/'artifacts';shutil.copytree(ARTIFACTS,target)
    p=target/'cancellations.csv';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
    m=target/'manifest.json';data=json.loads(m.read_text());data['files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();m.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='service calendar'):ForecastEngine(target)
