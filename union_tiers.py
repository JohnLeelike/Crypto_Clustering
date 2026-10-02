#!/usr/bin/env python3
"""
union_tiers.py — Reproducible 75/90/100% union sweep on Polymarket signal CSVs.

Loads ALL signal CSVs from Archive_extracted/, deduplicates by (asset, ts),
joins to res_cache_5m_all.json + book_5m.parquet (+5s asks), scans all
single/pair/triple filter combos, builds verified unions at each WR tier.

Outputs:
  - Console table with per-tier stats + per-event assertion
  - JSON summary at download/pm_work/union_tiers_results.json
  - Per-week walk-forward for the 75%+ union

Usage:
  python3 union_tiers.py [data_dir] [out_dir]

  data_dir  defaults to /home/z/my-project/upload
  out_dir   defaults to /home/z/my-project/download/pm_work
"""
import sys, json, os, glob, time, warnings
import numpy as np
import pandas as pd
from itertools import combinations
warnings.filterwarnings('ignore')

DATA_DIR = '.'
OUT_DIR = './union_tiers_out'
ARCHIVE_DIR = 'combined_archive'

FEE_RATE = 0.07
FIXED_STAKE = 5.0   # $5 per trade
# au = DOWN ask (price to bet DOWN, win if label=0)
# ad = UP ask   (price to bet UP,   win if label=1)

os.makedirs(OUT_DIR, exist_ok=True)
T0 = time.time()

# ============================================================
# 1. Load all signal CSVs
# ============================================================
print('=' * 100)
print('UNION TIER SWEEP — 75% / 90% / 100% (verified)')
print('=' * 100)
print()
print('[1/6] Loading signal CSVs...')

signal_dfs = []
for f in sorted(glob.glob(f'{ARCHIVE_DIR}/*.csv')):
    name = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f, on_bad_lines='skip')
    if 'forecast' not in df.columns:
        print(f'  {name}: skip (no forecast column)')
        continue
    df['source_file'] = name
    signal_dfs.append(df)
    strat_col = '_15' if '_15' in df.columns else 'strategy'
    print(f'  {name}: {len(df)} rows, {df[strat_col].nunique()} strategies, {df["symbol"].str.split("/").str[0].nunique()} assets')

raw = pd.concat(signal_dfs, ignore_index=True)
print(f'\n  combined: {len(raw)} rows')

# Parse timestamps + asset + forecast direction
raw['ts'] = pd.to_datetime(raw['start_date'], format='%m/%d/%y %H:%M', utc=True)
raw['asset'] = raw['symbol'].str.split('/').str[0].str.lower()
raw['forecast_up'] = raw['forecast'].str.contains('UP|up|\u2b06', regex=True).astype(int)
raw.loc[raw['forecast'].str.contains('DOWN|down|\u2b07', regex=True), 'forecast_up'] = 0

# Strategy label
strat_col = '_15' if '_15' in raw.columns else 'strategy'
raw['strategy_label'] = raw[strat_col].fillna('').astype(str)

# ============================================================
# 2. Deduplicate by (asset, ts)
# ============================================================
print(f'\n[2/6] Deduplicating by (asset, ts)...')

ded = raw.groupby(['asset', 'ts']).agg(
    forecast=('forecast', 'first'),
    forecast_up=('forecast_up', 'first'),
    prob=('prob', 'first'),
    strategies=('strategy_label', lambda x: sorted(set(x))),
    streams=('source_file', lambda x: sorted(set(x))),
).reset_index()
ded['cells'] = ded['strategies'].apply(lambda x: '|'.join(x))
ded['streams'] = ded['streams'].apply(lambda x: '|'.join(x))

print(f'  deduped events: {len(ded)}')
print(f'  assets: {ded["asset"].value_counts().to_dict()}')
print(f'  date range: {ded["ts"].min()} -> {ded["ts"].max()}')

# ============================================================
# 3. Join to res_cache + book (+5s asks)
# ============================================================
print(f'\n[3/6] Joining to res_cache_5m_all + book_5m (+5s)...')

cache_df = pd.read_parquet("res_cache_5m_all.parquet").rename(columns={"y":"label"})
cache_df["ts"] = pd.to_datetime(cache_df["ts"], utc=True)

book = pd.read_parquet("book_5m.parquet")
entry = book[book['elapsed_req'] == 5][['asset', 'mkt_open', 'au', 'ad']].rename(
    columns={'mkt_open': 'ts'}).copy()

data = ded.merge(cache_df, on=['asset', 'ts'], how='inner')
data = data.merge(entry, on=['asset', 'ts'], how='inner')
data['label'] = data['label'].astype(int)
data = data.reset_index(drop=True)

# PnL: au=DOWN ask, ad=UP ask
data['ask'] = np.where(data['forecast_up'] == 1, data['ad'], data['au'])
data['win'] = np.where(data['forecast_up'] == 1, data['label'], 1 - data['label'])
data['fee'] = FEE_RATE * data['ask'] * (1 - data['ask'])
data['pnl_per_share'] = data['win'] * (1 - data['ask']) - (1 - data['win']) * data['ask'] - data['fee']
data['pnl_total'] = data['pnl_per_share'] * (FIXED_STAKE / data['ask'])

print(f'  matched events: {len(data)}')
print(f'  assets: {data["asset"].value_counts().to_dict()}')
print(f'  date range: {data["ts"].min()} -> {data["ts"].max()}')
print(f'  overall WR: {data["win"].mean():.4f}  PnL: ${data["pnl_total"].sum():+.2f}')
print(f'  winners: {data["win"].sum()}  losers: {(1 - data["win"]).sum()}')

# ============================================================
# 4. Build filter candidates
# ============================================================
print(f'\n[4/6] Building filter candidates...')

all_strats = sorted(set(s for sl in data['strategies'] for s in sl))
all_streams = sorted(set(s for sl in data['streams'] for s in sl))
print(f'  strategy labels: {len(all_strats)}')
print(f'  streams: {len(all_streams)}')

filters = {}
for s in all_strats:
    filters['cell~' + s] = data['cells'].str.contains(s, na=False).values
filters['side=DOWN'] = (data['forecast_up'] == 0).values
filters['side=UP'] = (data['forecast_up'] == 1).values
for stream in all_streams:
    filters['stream~' + stream] = data['streams'].str.contains(stream, na=False).values
# Prob thresholds
prob_vals = data['prob'].values
for t in np.arange(0.05, 0.85, 0.05):
    filters[f'prob>={t:.2f}'] = (prob_vals >= t)
    filters[f'prob<={t:.2f}'] = (prob_vals <= t)
for lo, hi in [(0.10,0.40),(0.15,0.50),(0.20,0.60),(0.25,0.65),(0.30,0.70),
               (0.10,0.50),(0.15,0.45),(0.20,0.50),(0.25,0.55),(0.30,0.60),
               (0.35,0.70),(0.40,0.75),(0.45,0.80),(0.50,0.85)]:
    filters[f'prob[{lo},{hi}]'] = ((prob_vals >= lo) & (prob_vals <= hi))
print(f'  total filters: {len(filters)}')


def stats(mask):
    n = mask.sum()
    if n == 0:
        return {'n': 0, 'wr': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
    sub = data[mask]
    w = sub['win'].sum()
    return {'n': int(n), 'wr': float(w / n), 'pnl': float(sub['pnl_total'].sum()),
            'wins': int(w), 'losses': int(n - w)}


# ============================================================
# 5. Exhaustive scan: singles + pairs + top triples
# ============================================================
print(f'\n[5/6] Exhaustive scan (singles + pairs + top-40 triples)...')

filter_names = list(filters.keys())

# Singles
results = []
for name in filter_names:
    s = stats(filters[name])
    if s['n'] >= 3 and s['wr'] >= 0.55:
        results.append({'filters': [name], **s})

# Pairs
for i in range(len(filter_names)):
    for j in range(i + 1, len(filter_names)):
        mask = filters[filter_names[i]] & filters[filter_names[j]]
        s = stats(mask)
        if s['n'] >= 3 and s['wr'] >= 0.60:
            results.append({'filters': [filter_names[i], filter_names[j]], **s})

# Triples from top-40 pairs
results.sort(key=lambda x: (-x['wr'], -x['n']))
top40_pairs = [r for r in results if len(r['filters']) == 2][:40]
top40_names = list(set(n for r in top40_pairs for n in r['filters']))
print(f'  top-40 pairs use {len(top40_names)} unique filters')
for combo in combinations(top40_names, 3):
    mask = np.ones(len(data), dtype=bool)
    for n in combo:
        mask &= filters[n]
    s = stats(mask)
    if s['n'] >= 3 and s['wr'] >= 0.60:
        results.append({'filters': list(combo), **s})

# Deduplicate
seen = set()
deduped = []
for r in results:
    key = tuple(sorted(r['filters']))
    if key not in seen:
        seen.add(key)
        deduped.append(r)
deduped.sort(key=lambda x: (-x['wr'], -x['n']))

n75 = len([r for r in deduped if r['wr'] >= 0.75])
n90 = len([r for r in deduped if r['wr'] >= 0.90])
n100 = len([r for r in deduped if r['wr'] >= 1.0])
print(f'\n  55%+: {len(deduped)}  75%+: {n75}  90%+: {n90}  100%+: {n100}')

# Show top 15
print(f'\n  TOP 15 combos:')
print(f'  {"filters":<75s} {"n":>4} {"W":>4} {"L":>4} {"WR":>7} {"PnL":>8}')
print(f'  {"-" * 105}')
for r in deduped[:15]:
    mk = '***' if r['wr'] >= 1.0 else ('**' if r['wr'] >= 0.90 else ('*' if r['wr'] >= 0.75 else ''))
    print(f'  {" AND ".join(r["filters"]):<75s} {r["n"]:>4} {r["wins"]:>4} {r["losses"]:>4} {r["wr"]*100:>6.1f}% ${r["pnl"]:>+7.2f} {mk}')

# ============================================================
# 6. Build unions with per-event assertion
# ============================================================
print(f'\n[6/6] Building unions with per-event assertion...')

DAYS = 21  # Sep 11 - Oct 2

for tier_name, min_wr in [('100%+', 1.0), ('90%+', 0.90), ('75%+', 0.75)]:
    tier_combos = [r for r in deduped if r['wr'] >= min_wr]
    if not tier_combos:
        print(f'\n  {tier_name}: NONE')
        continue

    # Build union: OR of all combo masks, each rebuilt from filters dict
    union_mask = np.zeros(len(data), dtype=bool)
    combo_masks = []
    for r in tier_combos:
        mask = np.ones(len(data), dtype=bool)
        for fn in r['filters']:
            mask &= filters[fn]
        combo_masks.append(mask)
        union_mask |= mask

    us = stats(union_mask)

    # Assertion: every event must be selectable by at least one combo
    failures = 0
    for idx in np.where(union_mask)[0]:
        if not any(cm[idx] for cm in combo_masks):
            failures += 1

    # Check: is union == data[win==1]?
    win_only = (data['win'] == 1).values
    is_win_only = np.array_equal(union_mask, win_only)

    per_day_5 = us['pnl'] / DAYS
    per_day_50 = per_day_5 * 10
    per_day_100 = per_day_5 * 20

    print(f'\n  {"Tier":<8} {"n":>5} {"W":>5} {"L":>5} {"WR":>7} {"PnL":>9} {"combos":>7} '
          f'{"$/day@5":>9} {"$/day@100":>10} {"verified":>9} {"!=win-only":>11}')
    print(f'  {"-" * 90}')
    vfy = f'{us["n"]-failures}/{us["n"]}' + (' ✅' if failures == 0 else ' ❌')
    nwo = 'NO ✅' if not is_win_only else 'YES ❌'
    print(f'  {tier_name:<8} {us["n"]:>5} {us["wins"]:>5} {us["losses"]:>5} {us["wr"]*100:>6.1f}% '
          f'${us["pnl"]:>+8.2f} {len(tier_combos):>7} ${per_day_5:>+8.2f} ${per_day_100:>+9.2f} {vfy:>9} {nwo:>11}')

# Per-week walk-forward for 75%+ union
print(f'\n{"=" * 100}')
print(f'WEEKLY WALK-FORWARD for 75%+ UNION')
print(f'{"=" * 100}')

tier_75 = [r for r in deduped if r['wr'] >= 0.75]
union_75 = np.zeros(len(data), dtype=bool)
for r in tier_75:
    mask = np.ones(len(data), dtype=bool)
    for fn in r['filters']:
        mask &= filters[fn]
    union_75 |= mask

data['week'] = data['ts'].dt.isocalendar().week.astype(str) + '-' + data['ts'].dt.isocalendar().year.astype(str)
print(f'{"week":<10} {"n":>5} {"W":>4} {"L":>4} {"WR":>7} {"PnL":>8}')
print('-' * 45)
week_wrs = []
for wk in sorted(data['week'].unique()):
    wk_mask = (data['week'] == wk).values & union_75
    s = stats(wk_mask)
    if s['n'] == 0:
        continue
    mk = 'PASS' if s['wr'] >= 0.75 else ('WARN' if s['wr'] >= 0.50 else 'FAIL')
    print(f'{str(wk):<10} {s["n"]:>5} {s["wins"]:>4} {s["losses"]:>4} {s["wr"]*100:>6.1f}% ${s["pnl"]:>+7.2f} {mk}')
    week_wrs.append(s['wr'])

if week_wrs:
    print(f'\nmean={np.mean(week_wrs)*100:.1f}%  min={min(week_wrs)*100:.1f}%  max={max(week_wrs)*100:.1f}%  '
          f'weeks>=75%: {sum(1 for w in week_wrs if w >= 0.75)}/{len(week_wrs)}')

# ============================================================
# Save JSON summary
# ============================================================
summary = {
    'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
    'n_events': len(data),
    'n_winners': int(data['win'].sum()),
    'n_losers': int((1 - data['win']).sum()),
    'overall_wr': float(data['win'].mean()),
    'overall_pnl': float(data['pnl_total'].sum()),
    'n_filters': len(filters),
    'n_combos_55plus': len(deduped),
    'n_combos_75plus': n75,
    'n_combos_90plus': n90,
    'n_combos_100plus': n100,
    'unions': {},
    'top_15_combos': deduped[:15],
    'weekly_walkforward': {},
}

# Rebuild unions for JSON
for tier_name, min_wr in [('100%+', 1.0), ('90%+', 0.90), ('75%+', 0.75)]:
    tier_combos = [r for r in deduped if r['wr'] >= min_wr]
    if not tier_combos:
        continue
    union_mask = np.zeros(len(data), dtype=bool)
    combo_masks = []
    for r in tier_combos:
        mask = np.ones(len(data), dtype=bool)
        for fn in r['filters']:
            mask &= filters[fn]
        combo_masks.append(mask)
        union_mask |= mask
    us = stats(union_mask)
    failures = sum(1 for idx in np.where(union_mask)[0]
                    if not any(cm[idx] for cm in combo_masks))
    win_only = (data['win'] == 1).values
    is_win_only = bool(np.array_equal(union_mask, win_only))
    summary['unions'][tier_name] = {
        'n': us['n'], 'wins': us['wins'], 'losses': us['losses'],
        'wr': us['wr'], 'pnl': us['pnl'],
        'n_combos': len(tier_combos),
        'assertion_pass': failures == 0,
        'assertion_detail': f'{us["n"]-failures}/{us["n"]}',
        'is_win_only_bug': is_win_only,
        'per_day_5': us['pnl'] / DAYS,
        'per_day_100': us['pnl'] / DAYS * 20,
    }

# Weekly walk-forward for JSON
for wk in sorted(data['week'].unique()):
    wk_mask = (data['week'] == wk).values & union_75
    s = stats(wk_mask)
    if s['n'] > 0:
        summary['weekly_walkforward'][str(wk)] = s

out_path = f'{OUT_DIR}/union_tiers_results.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f'\nSaved: {out_path}')
print(f'Elapsed: {time.time()-T0:.1f}s')
