"""HTTP load test against the actual limited Docker container; standard library only."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def snapshot(container):
    code = "import json,pathlib; p=pathlib.Path('/sys/fs/cgroup'); print(json.dumps({n:(p/n).read_text() for n in ['cpu.stat','memory.current','memory.peak','memory.swap.current']}))"
    data = json.loads(subprocess.check_output(['docker','exec',container,'python','-c',code]))
    data['cpu_usage_usec'] = int(dict(x.split() for x in data.pop('cpu.stat').splitlines())['usage_usec'])
    return {k:int(v) for k,v in data.items()}


def run_case(host, port, container, name, path, seconds, clients, unique=False, target_rps=0):
    barrier = threading.Barrier(clients+1)
    def worker(index):
        conn=http.client.HTTPConnection(host,port,timeout=10)
        # Warm each connection before starting, including model cache.
        conn.request('GET',path); r=conn.getresponse(); r.read()
        barrier.wait(timeout=30); started=time.perf_counter(); deadline=started+seconds
        latency=[]; failures=0; size=0; number=0
        while time.perf_counter()<deadline:
            if target_rps:
                due=started+(number*clients+index)/target_rps
                time.sleep(max(0,due-time.perf_counter()))
                if time.perf_counter()>=deadline:break
            query=path+('&weather_factor='+str(1+(index*100000+number)*1e-9) if unique else '')
            t=time.perf_counter()
            try:
                conn.request('GET',query); response=conn.getresponse(); body=response.read()
                failures+=response.status!=200; size+=len(body)
            except Exception:
                failures+=1; conn.close(); conn=http.client.HTTPConnection(host,port,timeout=10)
            latency.append((time.perf_counter()-t)*1000); number+=1
        conn.close(); return latency,failures,size
    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
        futures=[pool.submit(worker,i) for i in range(clients)]
        before=snapshot(container); start=time.perf_counter(); barrier.wait(timeout=30)
        results=[f.result() for f in futures]; elapsed=time.perf_counter()-start
        after=snapshot(container)
    lat=sorted(v for result in results for v in result[0]); n=len(lat)
    return dict(name=name,path=path,cache='unique scenario per request' if unique else 'warm',
        clients=clients,target_rps=target_rps or None,elapsed_seconds=round(elapsed,3),requests=n,errors=sum(r[1] for r in results),
        rps=round(n/elapsed,2),p50_ms=round(lat[int((n-1)*.5)],3),
        p95_ms=round(lat[int((n-1)*.95)],3),p99_ms=round(lat[int((n-1)*.99)],3),
        cpu_percent_of_two_vcpu=round((after['cpu_usage_usec']-before['cpu_usage_usec'])/1e6/elapsed/2*100,2),
        memory_before_mib=round(before['memory.current']/2**20,2),memory_after_mib=round(after['memory.current']/2**20,2),
        peak_memory_mib=round(after['memory.peak']/2**20,2),swap_bytes=after['memory.swap.current'],
        response_bytes=sum(r[2] for r in results))


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8088');p.add_argument('--container',required=True)
    p.add_argument('--seconds',type=int,default=10);p.add_argument('--clients',type=int,default=16)
    p.add_argument('--target-rps',type=int,default=0);p.add_argument('--only-uncached',action='store_true')
    p.add_argument('--output',type=Path,default=Path('ml_service/reports/benchmark.json'));args=p.parse_args()
    u=urlsplit(args.url);conn=http.client.HTTPConnection(u.hostname,u.port)
    conn.request('GET','/v1/forecast.csv');r=conn.getresponse();body=r.read();conn.close()
    assert r.status==200 and hashlib.sha256(body).hexdigest()=='f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
    inspect=json.loads(subprocess.check_output(['docker','inspect',args.container]))[0]
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),host=platform.platform(),
        transport='Host loopback → Docker Desktop published port; client and server on same machine',
        method='Closed-loop persistent HTTP/1.1, latency includes body read; not a production capacity guarantee',
        image=inspect['Image'],limits={k:inspect['HostConfig'][k] for k in ['NanoCpus','Memory','MemorySwap']},
        baseline_csv_sha256=hashlib.sha256(body).hexdigest(),cases=[])
    for name,path,unique in [
        ('day_route_hourly','/v1/forecast?start=2025-11-10&end=2025-11-10&routes=17',False),
        ('month_all_routes_daily','/v1/forecast?start=2025-11-01&end=2025-11-30&aggregation=day',False),
        ('full_61_day_csv','/v1/forecast.csv?aggregation=hour',False),
        ('month_uncached_scenarios','/v1/forecast?start=2025-11-01&end=2025-11-30&aggregation=day',True)]:
        if args.only_uncached and not unique:continue
        case=run_case(u.hostname,u.port,args.container,name,path,args.seconds,args.clients,unique,args.target_rps)
        report['cases'].append(case);print(json.dumps(case),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
