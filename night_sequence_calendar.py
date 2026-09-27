"""Test whether the Transformer helps after correcting its seasonal anchor.

Uses the previous medium architecture, calendar analog baseline, masked service
cancellations and early checkpoints including zero (no learned correction).
"""
import argparse
import fcntl
import os
import time
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from night_sequence_study import Network
from night_sequence_large import Windows as OriginalWindows
from night_calendar_analog_study import CalendarData
from night_forecast_common import OUT, SELECTION, CONTROL, save_json, status


class Windows(OriginalWindows):
    def get(self, c, r, training=True):
        past, future, baseline, target, mask, route = super().get(c, r, training)
        h = min(61, 365-c)
        baseline[:h] = self.data.analog(c, h)['term'].reshape(h, 10, 24)[:, r]
        future[:h, self.width:self.width+24] = np.log1p(baseline[:h]/self.scale[r])
        return past, future, baseline, target, mask, route


def predict(model, windows, origin, device):
    model.eval()
    with torch.no_grad():
        past, future, base, _, _, routes = windows.batch([(origin, r) for r in range(10)], device, False)
        return model(past, future, base, routes).cpu().numpy().transpose(1, 0, 2).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    folder = OUT/('sequence_calendar_smoke' if args.smoke else 'sequence_calendar')
    folder.mkdir(parents=True, exist_ok=True)
    lock = open(folder/'run.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    end = min(time.time()+3600*args.hours, 1790483400.)
    assert time.time() < end
    torch.set_num_threads(4)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data = CalendarData(); data.causal_check()
    checkpoints = [0, 8] if args.smoke else [0, 50, 150, 500, 1500]
    records = []; selected = None
    for origin in ([151] if args.smoke else SELECTION+[CONTROL, 304]):
        if time.time() >= end:
            status(folder, 'budget_exhausted'); return
        if origin == CONTROL:
            r = pd.DataFrame(records)
            assert (r.groupby('step').origin.nunique() == 3).all()
            selected = int(r.groupby('step').score.mean().idxmax())
            save_json(folder/'selection.json', dict(step=selected, origins=SELECTION))
        steps = max(checkpoints) if origin in SELECTION or args.smoke else selected
        windows = Windows(data, origin)
        torch.manual_seed(270927); rng = np.random.default_rng(270927)
        model = Network(windows.width).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
        params = sum(p.numel() for p in model.parameters())
        fold_path = folder/('metrics_'+str(origin)+'.json')
        fold_records = json.loads(fold_path.read_text()) if fold_path.exists() and not args.smoke else []
        existing = sorted(folder.glob(str(origin)+'_step*.pt'), key=lambda p:int(p.stem.split('step')[1]))
        existing = [p for p in existing if int(p.stem.split('step')[1]) <= steps]
        start_step = 0
        if existing and not args.smoke:
            saved = torch.load(existing[-1], map_location=device, weights_only=True)
            model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
            start_step = int(saved['step'])

        def evaluate(step):
            dest = folder/(str(origin)+'_step'+str(step)+'.pt')
            torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                            step=step, width=windows.width, origin=origin), dest)
            p = predict(model, windows, origin, device)
            if step == 0:
                expected = data.analog(origin, 61)['term']
                np.testing.assert_allclose(p, expected, rtol=1e-6, atol=1e-3)
            replay = Network(windows.width).to(device)
            replay.load_state_dict(torch.load(dest, map_location=device, weights_only=True)['state_dict'])
            np.testing.assert_allclose(p, predict(replay, windows, origin, device), rtol=1e-6, atol=1e-3)
            del replay
            entry = data.export(folder, 'calendar_step'+str(step), origin, p)
            entry.update(origin=origin, step=step, split='selection' if origin in SELECTION else ('control' if origin==CONTROL else 'submission'))
            fold_records[:] = [r for r in fold_records if r['step'] != step]+[entry]
            save_json(fold_path, fold_records)
            save_json(dest.with_suffix('.json'), dict(origin=origin, step=step, parameters=params,
                features=windows.daily+windows.hourly, mu=windows.mu.tolist(), sd=windows.sd.tolist(),
                scale=windows.scale.tolist(), global_scale=windows.global_scale, training_max_day=origin-1))
            print('EVAL', entry, flush=True)

        if not fold_records:
            evaluate(0)
        start = time.time()
        for step in range(start_step+1, steps+1):
            model.train()
            indices = rng.integers(0, len(windows.samples), size=24)
            past, future, baseline, target, mask, routes = windows.batch([windows.samples[i] for i in indices], device)
            optimizer.zero_grad(set_to_none=True)
            p = model(past, future, baseline, routes)
            denominator = mask.sum().clamp_min(1)
            hourly = ((p-target).abs()*mask).sum()/denominator
            daily = ((p*mask).sum(-1)-(target*mask).sum(-1)).abs().sum()/denominator
            weekly = ((p[:,:56]*mask[:,:56]).reshape(24,8,7,24).sum((2,3))-(target[:,:56]*mask[:,:56]).reshape(24,8,7,24).sum((2,3))).abs().sum()/denominator
            loss = (hourly+.15*daily+.05*weekly)/windows.normalizer
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            if step % 100 == 0:
                status(folder, 'training', origin=origin, step=step, target_steps=steps,
                       loss=float(loss.detach().cpu()), parameters=params, seconds=time.time()-start)
            if step in checkpoints or step == steps or time.time() >= end:
                evaluate(step)
            if time.time() >= end:
                status(folder, 'budget_exhausted', origin=origin, step=step); return
        records.extend(fold_records); save_json(folder/'metrics.json', records)
        del model, optimizer, windows
        if device.type == 'mps':
            torch.mps.empty_cache()
    save_json(folder/'report.json', dict(results=records, selected_step=selected, parameters=params,
        device=str(device), checkpoint_replay=True, public_score=None, active_best_changed=False,
        rationale='Calendar analog improves autumn baseline from 0.84978 to 0.89848; test small early learned corrections, allow zero correction.',
        changes_from_initial_medium='Calendar baseline, cancelled slots masked, learning rate 5e-5, fixed schedule, early checkpoints including zero.',
        limitations=['Overlapping development windows already inspected.', 'No hidden target feedback.', 'No positive route-5 training target.', 'This comparison is not a single-factor ablation.']))
    status(folder, 'smoke_passed' if args.smoke else 'complete', selected_step=selected, results=records)


if __name__ == '__main__':
    main()
