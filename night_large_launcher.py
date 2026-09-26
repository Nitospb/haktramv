"""Start the large MPS job after the initial sequence job releases its lock."""
import fcntl
import os
import time
from datetime import datetime,timezone
from night_forecast_common import ROOT,OUT,status

folder=OUT/'sequence_large';folder.mkdir(parents=True,exist_ok=True)
own=open(folder/'launcher.lock','w');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
own.write(str(os.getpid()));own.flush()
deadline=datetime(2026,9,27,4,30,tzinfo=timezone.utc).timestamp()  # 07:30 Moscow
status(folder,'waiting_for_initial_sequence',launcher_pid=os.getpid())
with open(OUT/'sequence/run.lock','a') as lock:
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.time()>=deadline:status(folder,'deadline_before_launch');raise SystemExit(0)
            time.sleep(20)
hours=min(2,(deadline-time.time())/3600)
if hours<.25:status(folder,'insufficient_time_before_deadline');raise SystemExit(0)
status(folder,'launching',hours=hours)
os.chdir(ROOT)
os.execvp('caffeinate',['caffeinate','-i',str(ROOT/'.venv/bin/python'),'-u','night_sequence_large.py','--hours',str(hours),'--steps','12000'])
