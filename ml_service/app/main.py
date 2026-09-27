import json
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.engine import ForecastEngine

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('TRAM_ARTIFACTS',ROOT/'artifacts'))


@asynccontextmanager
async def lifespan(app):
    app.state.engine = ForecastEngine(ARTIFACTS)
    yield


app = FastAPI(title='Moscow tram forecast',version='1.0.0',lifespan=lifespan,
    description='Preserved public-best forecast (0.89402), Nov–Dec 2025. Day/month slicing; no live forecasting or stop-level target estimates.')


@app.get('/healthz')
async def health():
    return {'status':'ok'}


@app.get('/readyz')
async def ready():
    engine = getattr(app.state,'engine',None)
    if engine is None: raise HTTPException(503,'Model is not loaded')
    return {'status':'ready','model_id':engine.manifest['model_id'],'sha256':engine.digest,'rows':len(engine.keys)}


@app.get('/v1/model')
async def model():
    return app.state.engine.manifest


@app.get('/v1/sources')
async def sources():
    return json.loads((ARTIFACTS/'sources.json').read_text())


@app.get('/v1/routes')
async def routes():
    e=app.state.engine
    return {'routes':[{'route':r,'name':e.names[str(r)],'positive_training_labels':r!=5} for r in e.manifest['routes']]}


@app.get('/v1/geo/{kind}')
async def geo(kind:Literal['routes','stops'],route:int|None=None):
    if route is not None and route not in app.state.engine.manifest['routes']: raise HTTPException(422,'Unknown route')
    if kind=='routes':
        data=json.loads((ARTIFACTS/'routes.geojson').read_text())
        if route is not None:data['features']=[f for f in data['features'] if str(route) in f['properties'].get('routes',[])]
        return data
    data=json.loads((ARTIFACTS/'stops.json').read_text())
    return {'stops':[s for s in data if route is None or str(route) in s.get('routes',[])],
            'note':'Reference geometry only. No stop-level boarding forecast or occupancy estimate.'}


def parameters(request:Request,start:date=date(2025,11,1),end:date=date(2025,12,31),
    routes:Annotated[list[int]|None,Query(max_length=10)]=None,
    aggregation:Literal['hour','day','month','total']='hour',
    hour_from:Annotated[int,Query(ge=0,le=23)]=0,hour_to:Annotated[int,Query(ge=1,le=24)]=24,
    weather_factor:Annotated[float,Query(ge=0,le=3,allow_inf_nan=False)]=1,
    event_factor:Annotated[float,Query(ge=0,le=3,allow_inf_nan=False)]=1,
    season_factor:Annotated[float,Query(ge=0,le=3,allow_inf_nan=False)]=1,
    stop_id:str|None=None,segment_id:str|None=None):
    allowed={'start','end','routes','aggregation','hour_from','hour_to','weather_factor','event_factor','season_factor','stop_id','segment_id'}
    if set(request.query_params)-allowed:raise HTTPException(422,'Unknown query parameter; see /docs for the supported contract.')
    if stop_id is not None or segment_id is not None:
        raise HTTPException(422,'Stop/segment forecasts are not supported; use route forecasts and reference geometry.')
    if not date(2025,11,1)<=start<=end<=date(2025,12,31):
        raise HTTPException(422,'Supported dates: 2025-11-01 through 2025-12-31 inclusive; no extrapolation.')
    if hour_from>=hour_to: raise HTTPException(422,'hour_from must be less than hour_to (exclusive).')
    selected=tuple(sorted(set(routes or app.state.engine.manifest['routes'])))
    if any(r not in app.state.engine.manifest['routes'] for r in selected):raise HTTPException(422,'Unknown route')
    return start.isoformat(),end.isoformat(),selected,aggregation,hour_from,hour_to,weather_factor,event_factor,season_factor


@app.get('/v1/forecast')
async def forecast(params=Depends(parameters)):
    return Response(app.state.engine.forecast(*params)[0],media_type='application/json')


@app.get('/v1/forecast.csv')
async def export(params=Depends(parameters)):
    return Response(app.state.engine.forecast(*params)[1],media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition':'attachment; filename="forecast.csv"','X-Model-Id':'v8/level_scale_1.03'})


@app.get('/',include_in_schema=False)
async def index():
    return FileResponse(ROOT/'web/index.html')


app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
