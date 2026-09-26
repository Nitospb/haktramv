"""Fetch current public route-5 schedules; no historical date is implied."""
import datetime
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parent / 'data/schedule_research_20260927'


def fetch(item):
    direction, day = item
    url = 'https://kudikina.ru/msk/tram/5/' + direction + '?d=' + str(day)
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    response.encoding = 'utf-8'
    path = OUT / 'raw' / ('route5_' + direction + '_day' + str(day) + '.html')
    path.write_text(response.text)
    return dict(file='raw/' + path.name, url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source='Kudikina aggregator; claims public Mosgortrans data',
                effective_date=None, historical_2025_verified=False)


if __name__ == '__main__':
    (OUT / 'raw').mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(fetch, [(direction, day) for direction in ['A', 'B'] for day in [1, 2, 3, 4, 5, 6, 0]]))
    (OUT / 'web_sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
