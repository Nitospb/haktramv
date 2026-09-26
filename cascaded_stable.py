"""Bound route-level drift inside each recurrent daily/weekly transition.

The network still learns its hourly shape. The daily route volume can deviate
smoothly by at most exp(+/-0.15) from its own origin-safe historical profile.
This is an internal architectural constraint, not a blend with another model.
"""
import hashlib
from pathlib import Path

import torch

import cascaded_forecast as engine
import cascaded_dense as dense


class StableCascade(engine.Cascade):
    def block(self, history, hx, future, hbase, fbase, offset=0):
        prediction = super().block(history,hx,future,hbase,fbase,offset)
        b,g,n = prediction.shape
        p = prediction.reshape(b,10,24,n)
        base = fbase.reshape(b,10,24,n)
        ptotal = p.sum(2,keepdim=True); btotal = base.sum(2,keepdim=True)
        change = torch.log(ptotal.clamp_min(1e-6)/btotal.clamp_min(1e-6))
        bounded = .15*torch.tanh(change/.15)
        new_total = btotal*bounded.exp()
        result = p*new_total/ptotal.clamp_min(1e-6)
        result = torch.where(ptotal>1e-6,result,base)
        return result.reshape(b,240,n)


def fit(*args,**kwargs):
    model,norm,audit = dense.fit(*args,**kwargs)
    audit['volume_bound_log'] = .15
    audit['architecture'] = 'dense recursive cascade with bounded daily route volume'
    audit['stable_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return model,norm,audit


def main():
    engine.Cascade = StableCascade
    engine.CONFIGS = {'dense_stable_c5':dict(consistency=5.)}
    engine.fit = fit
    engine.main()


if __name__=='__main__':main()
