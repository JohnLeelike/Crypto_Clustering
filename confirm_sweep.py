import json, pandas as pd, numpy as np, time, warnings, os, glob
from itertools import combinations
warnings.filterwarnings('ignore')
T0 = time.time()
FEE = 0.07
FIXED = 5.0

print('[1/6] loading all CSVs + dedup (asset,ts only, as in pasted script)...', flush=True)
csv_files = sorted(glob.glob('port_logs/*.csv'))
all_signals = []
for f in csv_files:
    name = os.path.basename(f).replace('.csv','')
    df = pd.read_csv(f, on_bad_lines="skip")
    df['source_file'] = name
    all_signals.append(df)

raw = pd.concat(all_signals, ignore_index=True)
raw['ts'] = pd.to_datetime(raw['start_date'], format='%m/%d/%y %H:%M', utc=True)
sym_map = {'BTC/USDT':'btc','ETH/USDT':'eth','SOL/USDT':'sol','XRP/USDT':'xrp','DOGE/USDT':'doge','BNB/USDT':'bnb','HYPE/USDT':'hype'}
raw['asset'] = raw['symbol'].map(sym_map)
raw['forecast_up'] = raw['forecast'].astype(str).str.contains('UP|up|⬆', regex=True).astype(int)
raw.loc[raw['forecast'].astype(str).str.contains('DOWN|down|⬇', regex=True), 'forecast_up'] = 0

# check for asset+ts collisions with CONFLICTING direction before collapsing
chk = raw.groupby(['asset','ts'])['forecast_up'].nunique()
print(f'  asset+ts groups with CONFLICTING forecast_up (collapsed silently by groupby): {(chk>1).sum()} / {len(chk)}')

ded = raw.groupby(['asset','ts']).agg(
    forecast=('forecast','first'), forecast_up=('forecast_up','first'),
    prob=('prob','first'), confidence=('confidence','first'),
    strategies=('strategy', lambda x: sorted(set(x))),
).reset_index()
ded['cells'] = ded['strategies'].apply(lambda x: '|'.join(x))
print(f'  deduped events (asset,ts): {len(ded)}')

print(f'\n[2/6] joining to cache + book...', flush=True)
R = pd.read_parquet("res_cache_5m_all.parquet").rename(columns={"y":"label"})
BK = pd.read_parquet("book_5m.parquet")
BK["ts"]=pd.to_datetime(BK.mkt_open,utc=True); BK["asset"]=BK.asset.astype(str)
entry = BK[BK.elapsed_req==5][['asset','ts','au','ad']].copy()

data = ded.merge(R, on=['asset','ts'], how='inner')
data = data.merge(entry, on=['asset','ts'], how='inner')
data['label'] = data['label'].astype(int)
print(f'  matched (cache + book): {len(data)}')

data['ask'] = np.where(data['forecast_up']==1, data['ad'], data['au'])
data['win'] = np.where(data['forecast_up']==1, data['label'], 1-data['label'])
data['fee'] = FEE * data['ask'] * (1 - data['ask'])
data['pnl_ps'] = data['win']*(1-data['ask']) - (1-data['win'])*data['ask'] - data['fee']
data['pnl_total'] = data['pnl_ps'] * (FIXED / data['ask'])

print(f'\n[3/6] building candidate filters...', flush=True)
all_strategies = set()
for s in data['strategies']: all_strategies.update(s)
all_strategies = sorted(all_strategies)
candidates = []
for s in all_strategies:
    candidates.append(('cell~' + s, data['cells'].str.contains(s, na=False).values))
candidates.append(('fc=UP', (data['forecast_up']==1).values))
candidates.append(('fc=DN', (data['forecast_up']==0).values))
for t in [0.05,0.10,0.126,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]:
    candidates.append((f'prob>={t}', (data['prob']>=t).values))
    candidates.append((f'prob<={t}', (data['prob']<=t).values))
for lo,hi in [(0.126,0.354),(0.05,0.35),(0.10,0.40),(0.15,0.50),(0.20,0.60)]:
    candidates.append((f'prob in [{lo},{hi}]', ((data['prob']>=lo)&(data['prob']<=hi)).values))
print(f'  total candidates: {len(candidates)}  (n={len(data)} events)')

def compute_stats(mask):
    if mask.sum()==0: return {'n':0,'wr':0,'pnl':0,'wins':0,'losses':0}
    sub = data[mask]; n=len(sub); wins=sub['win'].sum()
    return {'n':n,'wr':float(wins/n),'pnl':float(sub['pnl_total'].sum()),'wins':int(wins),'losses':int(n-wins)}

filter_masks = {name: mask for name, mask in candidates}

print(f'\n[exhaustive scan, EXACTLY as pasted: min_n=3, top-15 singles -> pairs, top-10 pairs -> triples]')
exhaustive_results = []
for name, mask in candidates:
    stats = compute_stats(mask)
    if stats['n']>=3 and stats['wr']>=0.75:
        exhaustive_results.append({'filters':[name], **stats})
exhaustive_results.sort(key=lambda x:-x['wr'])
top15 = [r['filters'][0] for r in exhaustive_results[:15]]
for i in range(len(top15)):
    for j in range(i+1,len(top15)):
        n1,n2=top15[i],top15[j]
        mask=filter_masks[n1]&filter_masks[n2]
        stats=compute_stats(mask)
        if stats['n']>=3 and stats['wr']>=0.75:
            exhaustive_results.append({'filters':[n1,n2],**stats})
exhaustive_results.sort(key=lambda x:(-x['wr'],-x['n']))
top10pairs=[r['filters'] for r in exhaustive_results[:10] if len(r['filters'])==2]
for i in range(len(top10pairs)):
    for j in range(i+1,len(top10pairs)):
        allnames=list(set(top10pairs[i]+top10pairs[j]))
        if len(allnames)<3: continue
        for combo in combinations(allnames,3):
            mask=np.ones(len(data),dtype=bool)
            for nm in combo: mask &= filter_masks[nm]
            stats=compute_stats(mask)
            if stats['n']>=3 and stats['wr']>=0.75:
                exhaustive_results.append({'filters':list(combo),**stats})

seen=set(); deduped=[]
for r in exhaustive_results:
    key=tuple(sorted(r['filters']))
    if key not in seen: seen.add(key); deduped.append(r)
deduped.sort(key=lambda x:(-x['wr'],-x['n']))
print(f'  total 75%+ COMBOS (not deduped by event-set): {len(deduped)}')
print(f'  90%+ combos: {len([r for r in deduped if r["wr"]>=0.90])}')
print(f'  100% combos: {len([r for r in deduped if r["wr"]>=1.0])}')

for tier_name, lo in [('75%+',0.75),('90%+',0.90),('100%+',1.0)]:
    tier_combos=[r for r in deduped if r['wr']>=lo]
    if not tier_combos:
        print(f'\n  {tier_name}: none'); continue
    union_mask=np.zeros(len(data),dtype=bool)
    for r in tier_combos:
        mask=np.ones(len(data),dtype=bool)
        for nm in r['filters']: mask &= filter_masks[nm]
        union_mask |= mask
    st = compute_stats(union_mask)
    print(f'\n  {tier_name} UNION: n={st["n"]} WR={st["wr"]*100:.1f}% pnl=${st["pnl"]:+.2f}  (from {len(tier_combos)} combos)')
