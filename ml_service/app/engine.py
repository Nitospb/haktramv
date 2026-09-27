import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np


class ForecastEngine:
    def __init__(self, folder: Path):
        self.folder = folder
        self.manifest = json.loads((folder/'manifest.json').read_text())
        for name, digest in self.manifest['files'].items():
            if Path(name).name != name or hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Invalid artifact: '+name)
        def read(name, sep=';'):
            with (folder/name).open() as f:
                return list(csv.DictReader(f,delimiter=sep))
        def key(r):
            return int(r['route']),r['date'],int(r['hour'])
        base, expert = read('base.csv'), read('expert.csv')
        self.keys = tuple(key(r) for r in base)
        expected_keys = {(r,(date(2025,11,1)+timedelta(days=d)).isoformat(),h)
                         for r in self.manifest['routes'] for d in range(61) for h in range(24)}
        if len(self.keys)!=14640 or set(self.keys)!=expected_keys or tuple(key(r) for r in expert)!=self.keys:
            raise RuntimeError('Forecast grid mismatch')
        cancellations = {key(r):float(r['service_cancelled']) for r in read('cancellations.csv',',')}
        if set(cancellations)!=expected_keys or any(not 0 <= v <= 1 for v in cancellations.values()):
            raise RuntimeError('Incomplete or invalid service calendar')
        b = np.array([float(r['prediction']) for r in base]); e = np.array([float(r['prediction']) for r in expert])
        totals = defaultdict(lambda:[0.,0.])
        for k,v,w in zip(self.keys,b,e):
            totals[k[0],k[1][:7]][0] += v; totals[k[0],k[1][:7]][1] += w
        reconciled = np.array([v if totals[k[0],k[1][:7]][1] < 1e-6 else w*(totals[k[0],k[1][:7]][0]/(totals[k[0],k[1][:7]][1]+1e-9))
                               for k,v,w in zip(self.keys,b,e)])
        p = np.maximum(0,np.rint((.25*b+.75*reconciled)*np.array([1-.975*cancellations[k] for k in self.keys])))
        self.values = np.rint(p*1.03).astype(np.int64)
        self.values.setflags(write=False)
        rows = [dict(route=k[0],date=k[1],hour=k[2],prediction=int(v)) for k,v in zip(self.keys,self.values)]
        self.baseline_csv = self.to_csv(rows, 'hour')
        if self.baseline_csv != (folder/'expected.csv').read_bytes():
            raise RuntimeError('Reconstruction does not match the public-best CSV')
        self.digest = hashlib.sha256(self.baseline_csv).hexdigest()
        self.routes = np.array([k[0] for k in self.keys]); self.dates = np.array([k[1] for k in self.keys]); self.hours = np.array([k[2] for k in self.keys])
        self.names = json.loads((folder/'route_names.json').read_text())

    @staticmethod
    def to_csv(rows, aggregation):
        columns = {'hour':['route','date','hour','prediction'], 'day':['route','date','prediction'],
                   'month':['route','month','prediction'],'total':['route','prediction']}[aggregation]
        stream = io.StringIO(); writer = csv.DictWriter(stream,fieldnames=columns,delimiter=';',lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        return stream.getvalue().encode('utf-8')

    @lru_cache(maxsize=128)
    def forecast(self,start,end,routes,aggregation,hour_from,hour_to,weather,event,season):
        mask = (self.dates>=start)&(self.dates<=end)&np.isin(self.routes,routes)&(self.hours>=hour_from)&(self.hours<hour_to)
        indices = np.flatnonzero(mask); corrected = np.rint(self.values[mask].astype(float)*weather*event*season).astype(np.int64)
        rows = []; groups = defaultdict(int)
        for i,value in zip(indices,corrected):
            route,day,hour = self.keys[i]
            if aggregation=='hour': rows.append(dict(route=route,date=day,hour=hour,prediction=int(value)))
            else: groups[(route,day if aggregation=='day' else day[:7] if aggregation=='month' else '')] += int(value)
        for (route,period),value in sorted(groups.items()):
            row=dict(route=route,prediction=value)
            if aggregation!='total': row['date' if aggregation=='day' else 'month']=period
            rows.append(row)
        scenario = any(v!=1 for v in (weather,event,season))
        payload=dict(model_id=self.manifest['model_id'],mode=self.manifest['mode'],timezone='Europe/Moscow',
            start=start,end=end,aggregation=aggregation,unit='boardings',scenario=scenario,
            factors=dict(weather=weather,event=event,season=season),
            factors_note='Manual scenario multipliers, not learned causal effects. Applied per hour before aggregation.',
            baseline_public_score=.89402,scenario_public_score=None,rows=rows,
            row_count=len(rows),total=int(corrected.sum()),baseline_total=int(self.values[mask].sum()),
            warnings=['Route 5 has zero predictions in the preserved winner.'] if 5 in routes else [])
        return json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(), self.to_csv(rows,aggregation)
