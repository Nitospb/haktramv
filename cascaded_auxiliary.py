"""v15 sensitivity check: short transitions as an auxiliary, not dominant loss."""
import hashlib
from pathlib import Path

import cascaded_forecast as engine

original_fit = engine.fit


def fit(*args, **kwargs):
    model, norm, audit = original_fit(*args, **kwargs)
    audit['auxiliary_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return model, norm, audit


def main():
    engine.CONFIGS = {
        'short_aux_c1': dict(short_weight=.1, consistency=1.),
        'short_aux_c5': dict(short_weight=.1, consistency=5.),
    }
    engine.fit = fit
    engine.main()


if __name__ == '__main__':
    main()
