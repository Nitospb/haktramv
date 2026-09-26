# Offline accumulation experiment

This experiment tests whether delayed transaction uploads improve the tram
boarding forecast. It does not treat `input_date_time` as ground truth.

- Only non-negative delays up to seven days are used.
- All target features are historical profiles available before forecast origin.
- Route risk and weighted coordinate-cluster risk from `stop_zone_routes.csv`
  are separate features. These are static route footprints, not live vehicle GPS.
- The whitelist-era flag starts on 2025-09-05.
- No target-hour upload delay or failure observation is available to a future
  November-December submission.

Run from the repository root:

```sh
external_features/.venv/bin/python experiments/offline_accumulation/aggregate_offline.py
external_features/.venv/bin/python experiments/offline_accumulation/train.py
python3 experiments/offline_accumulation/evaluate_penalty.py
```
