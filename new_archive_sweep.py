import json, pandas as pd, numpy as np, time, warnings, os, glob
from itertools import combinations
warnings.filterwarnings('ignore')
T0 = time.time()

DATA_DIR = 'backtest_data/new_streams'
FEE = 0.07; FIXED = 5.0

print('[1/6] loading all signal CSVs...', flush=True)
signal_files = sorted(f for f in glob.glob(f'{DATA_DIR}/*.csv') if 'gL_2' not in f)
signal_dfs = []
for f in signal_files:
    name = os.path.basename(f).replace('.csv','')
    df = pd.read_csv(f, on_bad_lines="skip")
    if 'forecast' not in df.columns:
        print(f'  {name}: SKIP (no forecast column)')
        continue
    df['source_file'] = name
    signal_dfs.append(df)
    strat_col = '_15' if '_15' in df.columns else 'strategy'
    print(f'  {name}: {len(df)} rows, {df[strat_col].nunique()} strategies, assets: {df["symbol"].str.split("/").str[0].nunique()}')

raw = pd.concat(signal_dfs, ignore_index=True)
print(f'\n  combined: {len(raw)} rows')

raw['ts'] = pd.to_datetime(raw['start_date'], format='%m/%d/%y %H:%M', utc=True)
raw['asset'] = raw['symbol'].str.split('/').str[0].str.lower()
raw['forecast_up'] = raw['forecast'].astype(str).str.contains('UP|up|⬆', regex=True).astype(int)
raw.loc[raw['forecast'].astype(str).str.contains('DOWN|down|⬇', regex=True), 'forecast_up'] = 0

strat_col = '_15' if '_15' in raw.columns else 'strategy'
raw['strategy_label'] = raw[strat_col].astype(str)
raw['stream'] = raw['source_file']

print(f'\n[2/6] deduplicating by (asset, ts)...', flush=True)
ded = raw.groupby(['asset','ts']).agg(
    forecast=('forecast','first'), forecast_up=('forecast_up','first'),
    prob=('prob','first'),
    strategies=('strategy_label', lambda x: sorted(set(x))),
    streams=('stream', lambda x: sorted(set(x))),
).reset_index()
ded['cells'] = ded['strategies'].apply(lambda x: '|'.join(x))
ded['streams'] = ded['streams'].apply(lambda x: '|'.join(x))
print(f'  deduped: {len(ded)} events')
print(f'  assets: {ded["asset"].value_counts().to_dict()}')
print(f'  date range: {ded["ts"].min()} -> {ded["ts"].max()}')
print(f'  forecast: {ded["forecast"].value_counts().to_dict()}')

print(f'\n[3/6] joining to cache + book...', flush=True)
R = pd.read_parquet("res_cache_5m_all.parquet").rename(columns={"y":"label"})
BK = pd.read_parquet("book_5m.parquet")
BK["ts"]=pd.to_datetime(BK.mkt_open,utc=True); BK["asset"]=BK.asset.astype(str)
entry = BK[BK.elapsed_req==5][['asset','ts','au','ad']].copy()

data = ded.merge(R, on=['asset','ts'], how='inner')
data = data.merge(entry, on=['asset','ts'], how='inner')
data['label'] = data['label'].astype(int)
data['ask'] = np.where(data['forecast_up']==1, data['ad'], data['au'])
data['win'] = np.where(data['forecast_up']==1, data['label'], 1-data['label'])
data['fee'] = FEE * data['ask'] * (1 - data['ask'])
data['pnl_ps'] = data['win']*(1-data['ask']) - (1-data['win'])*data['ask'] - data['fee']
data['pnl_total'] = data['pnl_ps'] * (FIXED / data['ask'])
data = data.reset_index(drop=True)
print(f'  matched: {len(data)} events')
print(f'  assets: {data["asset"].value_counts().to_dict()}')
print(f'  date range: {data["ts"].min()} -> {data["ts"].max()}')
print(f'  overall WR: {data["win"].mean():.4f}  PnL: ${data["pnl_total"].sum():+.2f}')
print(f'  n winners: {data["win"].sum()}  n losers: {(1-data["win"]).sum()}')

print(f'\n[4/6] building filter candidates...', flush=True)
all_strats = sorted(set(s for sl in data['strategies'] for s in sl))
all_streams = sorted(set(s for sl in data['streams'] for s in sl))
print(f'  strategy labels: {len(all_strats)}: {all_strats[:10]}...')
print(f'  streams: {len(all_streams)}: {all_streams}')

filters = {}
for s in all_strats:
    filters['cell~'+s] = data['cells'].str.contains(s, na=False, regex=False).values
filters['side=DOWN'] = (data['forecast_up']==0).values
filters['side=UP'] = (data['forecast_up']==1).values
for stream in all_streams:
    filters['stream~'+stream] = data['streams'].str.contains(stream, na=False, regex=False).values
prob_vals = data['prob'].values
for t in np.arange(0.05, 0.85, 0.05):
    filters[f'prob>={t:.2f}'] = (prob_vals >= t)
    filters[f'prob<={t:.2f}'] = (prob_vals <= t)
for lo, hi in [(0.10,0.40),(0.15,0.50),(0.20,0.60),(0.25,0.65),(0.30,0.70),
               (0.10,0.50),(0.15,0.45),(0.20,0.50),(0.25,0.55),(0.30,0.60),
               (0.35,0.70),(0.40,0.75),(0.45,0.80),(0.50,0.85)]:
    filters[f'prob[{lo},{hi}]'] = ((prob_vals>=lo)&(prob_vals<=hi))
print(f'  total filters: {len(filters)}')

def stats(mask):
    n = mask.sum()
    if n == 0: return {'n':0,'wr':0,'pnl':0,'wins':0,'losses':0}
    sub = data[mask]
    w = sub['win'].sum()
    return {'n':int(n),'wr':float(w/n),'pnl':float(sub['pnl_total'].sum()),'wins':int(w),'losses':int(n-w)}

print(f'\n[5/6] exhaustive scan...', flush=True)
filter_names = list(filters.keys())
results = []
for name in filter_names:
    s = stats(filters[name])
    if s['n'] >= 3 and s['wr'] >= 0.55:
        results.append({'filters': [name], **s})

for i in range(len(filter_names)):
    for j in range(i+1, len(filter_names)):
        mask = filters[filter_names[i]] & filters[filter_names[j]]
        s = stats(mask)
        if s['n'] >= 3 and s['wr'] >= 0.60:
            results.append({'filters': [filter_names[i], filter_names[j]], **s})

results.sort(key=lambda x: (-x['wr'], -x['n']))
top40_pairs = [r for r in results if len(r['filters'])==2][:40]
top40_names = list(set(n for r in top40_pairs for n in r['filters']))
print(f'  top-40 pairs use {len(top40_names)} unique filters')
for combo in combinations(top40_names, 3):
    mask = np.ones(len(data), dtype=bool)
    for n in combo: mask &= filters[n]
    s = stats(mask)
    if s['n'] >= 3 and s['wr'] >= 0.60:
        results.append({'filters': list(combo), **s})

seen = set(); deduped = []
for r in results:
    key = tuple(sorted(r['filters']))
    if key not in seen:
        seen.add(key); deduped.append(r)
deduped.sort(key=lambda x: (-x['wr'], -x['n']))

n66 = len([r for r in deduped if r['wr'] >= 0.66])
n75 = len([r for r in deduped if r['wr'] >= 0.75])
n90 = len([r for r in deduped if r['wr'] >= 0.90])
n100 = len([r for r in deduped if r['wr'] >= 1.0])
print(f'\n  55%+: {len(deduped)}  66%+: {n66}  75%+: {n75}  90%+: {n90}  100%+: {n100}')

print(f'\n  TOP 30 combos:')
print(f'  {"filters":<75s} {"n":>4} {"W":>4} {"L":>4} {"WR":>7} {"PnL":>8}')
for r in deduped[:30]:
    mk = '***' if r['wr']>=1.0 else ('**' if r['wr']>=0.90 else ('*' if r['wr']>=0.75 else ''))
    print(f'  {" AND ".join(r["filters"]):<75s} {r["n"]:>4} {r["wins"]:>4} {r["losses"]:>4} {r["wr"]*100:>6.1f}% ${r["pnl"]:>+7.2f} {mk}')

print(f'\n[6/6] building unions with per-event assertion...', flush=True)
UNION_EVENTS = {}
for tier, min_wr in [('100%+', 1.0), ('90%+', 0.90), ('75%+', 0.75)]:
    tier_combos = [r for r in deduped if r['wr'] >= min_wr]
    if not tier_combos:
        print(f'\n  {tier}: NONE')
        continue
    union_mask = np.zeros(len(data), dtype=bool)
    combo_masks = []
    for r in tier_combos:
        mask = np.ones(len(data), dtype=bool)
        for fn in r['filters']: mask &= filters[fn]
        combo_masks.append(mask)
        union_mask |= mask
    us = stats(union_mask)
    failures = sum(1 for idx in np.where(union_mask)[0] if not any(cm[idx] for cm in combo_masks))
    win_only = (data['win']==1).values
    is_win_only = np.array_equal(union_mask, win_only)
    print(f'\n  {tier} UNION:')
    print(f'    n events: {us["n"]}  W={us["wins"]}  L={us["losses"]}  WR={us["wr"]*100:.1f}%  PnL=${us["pnl"]:+.2f}')
    print(f'    n combos: {len(tier_combos)}')
    print(f'    assertion: {us["n"]-failures}/{us["n"]} verified {"OK" if failures==0 else "BUG"}')
    print(f'    union == data[win==1]? {"YES -- BUG" if is_win_only else "NO -- ok"}')
    UNION_EVENTS[tier] = set(data[union_mask].apply(lambda r: (r['asset'], str(r['ts']), r['forecast_up']), axis=1))
    if us['n'] <= 30 and failures == 0:
        for i, (_, row) in enumerate(data[union_mask].sort_values('ts').iterrows()):
            idx = row.name
            n_combos = sum(1 for cm in combo_masks if cm[idx])
            fc = 'UP' if row['forecast_up']==1 else 'DN'
            print(f'      {i+1:>3}. {str(row["ts"])[:19]} {row["asset"]:<5} {fc} label={row["label"]} win={row["win"]} prob={row["prob"]:.3f} ask={row["ask"]:.3f} in {n_combos} combos')

import pickle
with open("new_archive_sweep_unions.pkl","wb") as f:
    pickle.dump(UNION_EVENTS, f)
print(f"\nelapsed: {time.time()-T0:.1f}s")
