"""v14b: train hourly corrections while preserving the guide's weekly volume.

The guide and all normalization totals are forecasts made at the origin. Actual
hidden volumes never enter this projection. All operations remain differentiable.
"""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

import hourly_consistency as engine


class HourlyShapeResidual(engine.HourlyResidual):
    def finish(self, state, shape, base):
        prediction = super().finish(state, shape, base)
        length = base.shape[2]
        pad = (-length) % 168
        # Forecast weeks start at the cutoff; the last, partial week is separate.
        p = F.pad(prediction, (0, pad)).reshape(*base.shape[:2], -1, 168)
        b = F.pad(base, (0, pad)).reshape_as(p)
        total_p = p.sum(3, keepdim=True)
        total_b = b.sum(3, keepdim=True)
        scaled = p * total_b / total_p.clamp_min(1e-6)
        result = torch.where(total_p > 1e-6, scaled, b)
        return result.reshape(*base.shape[:2], -1)[:, :, :length]


original_fit = engine.fit


def fit(*args, **kwargs):
    model, norm, audit = original_fit(*args, **kwargs)
    audit['projection'] = 'preserve each route and 168-hour guide total; final partial week separately'
    audit['projection_uses_hidden_actuals'] = False
    audit['shape_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return model, norm, audit


def main():
    engine.HourlyResidual = HourlyShapeResidual
    engine.fit = fit
    engine.DEST = engine.ROOT / 'outputs/v14_shape'
    engine.CONFIGS = {'h1': [1], 'h24': [24], 'mixed': engine.STEPS}
    engine.main()


if __name__ == '__main__':
    main()
