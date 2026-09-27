"""Fixed level probes on the rejected direct forecast; no training or blending."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/level_probe_20260927'
BASE = ROOT/'outputs/night_20260927/deliverables/submission_01_direct.csv'
BEST = ROOT/'outputs/submission_best.csv'


def read(path, delimiter=';'):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_total(values, total):
    base = sum(values)
    quotients = [divmod(v*total, base) for v in values]
    result = [q for q, _ in quotients]
    missing = total-sum(result)
    order = sorted(range(len(values)), key=lambda i: (-quotients[i][1], i))
    for index in order[:missing]:
        result[index] += 1
    assert sum(result) == total
    assert all(v != 0 or p == 0 for v, p in zip(values, result))
    return result


def score(target, prediction):
    return max(0., 1-sum(abs(y-p) for y, p in zip(target, prediction))/sum(target))


def main():
    assert sha(BASE) == '1321ca06c74db294310b4f85c5557759242595a93f66b44d745dae8615bb69fe'
    best_hash = 'f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
    assert sha(BEST) == best_hash
    OUT.mkdir(exist_ok=True)
    rows, best = read(BASE), read(BEST)
    key = lambda r: (int(float(r['route'])), r['date'], int(float(r['hour'])))
    assert len(rows) == len(best) == 14640
    assert len({key(r) for r in rows}) == 14640
    assert {key(r) for r in rows} == {key(r) for r in best}
    values = [int(r['prediction']) for r in rows]
    total = sum(int(r['prediction']) for r in best)
    variants = [
        ('submission_direct_plus4.csv', [round(v*1.04) for v in values]),
        ('submission_direct_best_volume.csv', match_total(values, total)),
    ]
    artifacts = []
    for name, prediction in variants:
        path = OUT/name
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['route','date','hour','prediction'], delimiter=';')
            writer.writeheader()
            for row, p in zip(rows, prediction):
                assert isinstance(p, int) and p >= 0
                writer.writerow(dict(row, prediction=p))
        saved = read(path)
        assert [int(r['prediction']) for r in saved] == prediction
        assert [key(r) for r in saved] == [key(r) for r in rows]
        artifacts.append(dict(file=name, sha256=sha(path), public_score=None, rows=len(saved),
            total=sum(prediction), volume_change_vs_direct=sum(prediction)/sum(values)-1,
            volume_change_vs_public_best=sum(prediction)/total-1))
    records = []
    for origin in [181, 212, 243]:
        validation = read(ROOT/'outputs/longer_moscow_external_20260927'/('forecast_'+str(origin)+'.csv'), ',')
        target = [float(r['boardings']) for r in validation]
        base = [round(float(r['direct_1600'])) for r in validation]
        reference = read(ROOT/'outputs/all_moscow_external_20260927'/('forecast_'+str(origin)+'.csv'), ',')
        assert {key(r) for r in validation} == {key(r) for r in reference}
        for scale in [.97, 1., 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10]:
            p = [round(v*scale) for v in base]
            records.append(dict(origin=origin, method='fixed_scale', scale=scale,
                                score=score(target,p), total=sum(p), target_total=sum(target)))
        reference_total = sum(round(float(r['confirmed_v8_up3'])) for r in reference)
        p = match_total(base, reference_total)
        records.append(dict(origin=origin, method='best_forecast_volume', scale=reference_total/sum(base),
                            score=score(target,p), total=sum(p), target_total=sum(target)))
    with (OUT/'validation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    report = dict(base=str(BASE.relative_to(ROOT)), base_public_score=.88232,
        best=str(BEST.relative_to(ROOT)), best_public_score=.89402,
        base_total=sum(values), best_total=total, base_to_best_multiplier=total/sum(values),
        multiplier_for_4percent_above_best=1.04*total/sum(values), candidates=artifacts,
        interpretation='Plus4 is relative to rejected direct candidate. Best-volume probe uses only the total of the preserved best forecast, not its hourly shape. No hidden targets used.',
        limitation='Plus4 worsens all three historical windows. These are public-level diagnostic probes, not locally supported improvements. Actual Nov-Dec volume is unknown.',
        rounding='Base forecasts rounded before scaling, matching inference from the submitted CSV. Equal-volume uses integer largest-remainder allocation.',
        best_changed=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sha(BEST) == best_hash
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
