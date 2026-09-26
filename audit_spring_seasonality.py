"""Audit the linked public dataset; no notebook code is executed."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/seasonality_audit'
DEST.mkdir(exist_ok=True)
MONTHS = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

source = ROOT / 'data/references/moscow_transport/passengersdata.csv'
original = ROOT / 'data/features/08_external_seasonality/moscow_monthly_passengers.csv'
assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()
x = pd.read_csv(source, sep=';')
x = x[pd.to_numeric(x.Year, errors='coerce').notna()].copy()
x['year'] = x.Year.astype(int)
x['month'] = x.Month.map(dict(zip(MONTHS, range(1, 13))))
x['passengers'] = pd.to_numeric(x.PassengerTraffic)
x['date'] = pd.to_datetime(dict(year=x.year, month=x.month, day=1))
x['daily'] = x.passengers / x.date.dt.days_in_month
t = x[x.TransportType == 'Трамвай'].copy()
assert not t.duplicated(['year', 'month']).any()
t[['year', 'month', 'passengers', 'daily']].to_csv(DEST / 'tram_monthly_2019_2025.csv', index=False)

comparison = []
for year in [2023, 2024]:
    y = t[t.year == year].set_index('month')
    spring = y.loc[[3, 4], 'passengers'].sum() / 61
    for month in [1, 2, 3, 4, 11, 12]:
        comparison.append(dict(year=year, month=month, monthly_passengers=int(y.loc[month, 'passengers']),
                               daily_passengers=float(y.loc[month, 'daily']),
                               daily_vs_spring=float(y.loc[month, 'daily'] / spring)))
c = pd.DataFrame(comparison)
c.to_csv(DEST / 'spring_comparison_2023_2024.csv', index=False)

# Reproduce the old prior exactly before discussing what has changed.
old = t[t.year.isin([2022, 2023, 2024])].copy()
old['index'] = old.daily / old.groupby('year').daily.transform('mean')
old_index = old.groupby('month')['index'].median()
saved = json.loads((ROOT / 'data/features/08_external_seasonality/moscow_seasonality_index.json').read_text())['monthly_index']
np.testing.assert_allclose(old_index.to_numpy(), [saved[str(m)] for m in range(1, 13)], rtol=1e-12)
new = t[t.year.isin([2023, 2024])].copy()
new['index'] = new.daily / new.groupby('year').daily.transform('mean')
new_index = new.groupby('month')['index'].mean()
pd.DataFrame({'old_2022_2024': old_index, 'new_2023_2024': new_index}).to_csv(DEST / 'daily_season_indices.csv')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={'width_ratios': [1.65, 1]})
for year, color in [(2023, '#4475bb'), (2024, '#d77536')]:
    yy = t[t.year == year].set_index('month')
    spring = yy.loc[[3, 4], 'passengers'].sum() / 61
    axes[0].plot(yy.index, yy.daily / spring * 100, marker='o', lw=2.3, color=color, label=str(year))
axes[0].axhline(100, color='#7b8592', ls='--', lw=1)
axes[0].axvspan(2.7, 4.3, color='#80959e', alpha=.10)
axes[0].axvspan(10.7, 12.3, color='#559975', alpha=.10)
axes[0].set_xticks(range(1, 13), ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'])
axes[0].set_ylabel('Среднесуточный поток, % к марту–апрелю')
axes[0].set_title('Трамваи: весенний и осенний уровень')
axes[0].legend(frameon=False)
positions = np.arange(4)
for offset, year, color in [(-.19, 2023, '#4475bb'), (.19, 2024, '#d77536')]:
    values = c[(c.year == year) & c.month.isin([1, 2, 11, 12])].daily_vs_spring.to_numpy() * 100
    bars = axes[1].bar(positions + offset, values, width=.36, color=color, label=str(year))
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, value + 1.8, f'{value:.0f}%', ha='center', fontsize=10)
axes[1].axhline(100, color='#7b8592', ls='--', lw=1)
axes[1].set_xticks(positions, ['Янв', 'Фев', 'Ноя', 'Дек'])
axes[1].set_ylim(0, 123)
axes[1].set_title('Март–апрель = 100%')
for ax in axes:
    ax.grid(axis='y', alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Ноябрь–декабрь ближе к весне, чем к январю', fontsize=15, y=1.01)
fig.text(.01, -.015, 'Источник: passengersdata.csv из репозитория ArtNikVal. Только трамваи, 2023–2024. Поправка на длину месяца; состав дней недели не выровнен.', fontsize=9, color='#535d69')
fig.tight_layout()
fig.savefig(DEST / 'tram_spring_analogs.png', dpi=180, bbox_inches='tight')
audit = {
    'source_url': 'https://github.com/ArtNikVal/Analyseofmoscowtaransportproject',
    'commit': 'f4811cd53e482a6330b6c58f948e40282ff5ddac',
    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'same_bytes_as_supplied_bundle': True,
    'dimensions': 'Transport modes, not individual routes. Monthly totals, not hourly data.',
    'old_index_reproduced': 'Median across 2022–2024 of tram average daily passengers / annual mean of monthly daily averages.',
    'new_prior_years': [2023, 2024],
    'comparison': comparison,
    'interpretation_limit': 'Two recent years support a spring-like level, but do not identify Nov–Dec 2025 route/hour values. Workday/calendar correction is implemented separately in the forecasting experiment.',
}
(DEST / 'source_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
print(c.to_string(index=False))
print('Saved', DEST)
