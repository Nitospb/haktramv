"""Whole-week model with a bounded route-level weekly volume correction."""
import hashlib
from pathlib import Path

import torch

import cascaded_forecast as engine
import weekly_joint as weekly


class StableWeekly(weekly.WeeklyJoint):
    def block(self, history, hx, future, hbase, fbase, offset=0):
        prediction=super().block(history,hx,future,hbase,fbase,offset)
        b,g,n=prediction.shape
        p=prediction.reshape(b,10,24,n);base=fbase.reshape(b,10,24,n)
        pt=p.sum((2,3),keepdim=True);bt=base.sum((2,3),keepdim=True)
        change=torch.log(pt.clamp_min(1e-6)/bt.clamp_min(1e-6))
        total=bt*(.15*torch.tanh(change/.15)).exp()
        out=torch.where(pt>1e-6,p*total/pt.clamp_min(1e-6),base)
        return out.reshape(b,240,n)


def fit(*args,**kwargs):
    model,norm,audit=weekly.fit(*args,**kwargs)
    audit['volume_bound_log']=.15
    audit['volume_bound_period']='predicted week or final partial week'
    audit['stable_weekly_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return model,norm,audit


def main():
    weekly.WeeklyJoint=StableWeekly
    engine.DEST=engine.ROOT/'outputs/v16_weekly'
    engine.CONFIGS={'weekly_stable':dict(bound=.35,learning_rate=.0005,weight_decay=.1)}
    engine.fit=fit;engine.predict=weekly.predict;engine.audits=weekly.audits
    engine.main()


if __name__=='__main__':main()
