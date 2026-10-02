import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
res = pd.read_parquet("res_flat.parquet")
MIN_N = 15

def metrics(df):
    n=len(df)
    if n==0: return 0,0,0,0
    pnl=df.pnl5.sum(); roi=pnl/(5*n)*100; wr=df.win.mean()*100
    return n,pnl,roi,wr

def build_candidates(df):
    cands=[]
    for a in df.asset.dropna().unique(): cands.append((f"asset=={a}", df.asset==a))
    for s in ["UP","DOWN"]: cands.append((f"side=={s}", df.side==s))
    for c in df.cell.dropna().unique(): cands.append((f"cell=={c}", df.cell==c))
    for st in df.stream.dropna().unique(): cands.append((f"stream=={st}", df.stream==st))
    probs = df.prob.dropna()
    if len(probs)>0:
        qs = np.quantile(probs, np.linspace(0.1,0.9,9))
        for q in sorted(set(round(x,3) for x in qs)):
            cands.append((f"prob>={q}", df.prob>=q))
            cands.append((f"prob<={q}", df.prob<=q))
    return cands

def greedy(df0, target, min_n=MIN_N, max_steps=8):
    current = df0.copy()
    path=[]
    n,pnl,roi,wr = metrics(current)
    cur_val = pnl if target=="pnl" else roi
    for step in range(max_steps):
        cands = build_candidates(current)
        best=None
        for label, mask in cands:
            sub = current[mask]
            n2,pnl2,roi2,wr2 = metrics(sub)
            if n2 < min_n: continue
            val2 = pnl2 if target=="pnl" else roi2
            if val2 > cur_val + 1e-9:
                if best is None or val2 > best[1]:
                    best = (label, val2, sub, n2, pnl2, roi2, wr2)
        if best is None:
            break
        label, val2, sub, n2, pnl2, roi2, wr2 = best
        path.append((label, n2, pnl2, roi2, wr2))
        current = sub
        cur_val = val2
    return path, current

print("="*100)
print("GREEDY SWEEP — maximize TOTAL PnL@$5  (min_n=%d per step)"%MIN_N)
print("="*100)
path,final = greedy(res, "pnl")
n0,pnl0,roi0,wr0 = metrics(res)
print(f"  START: n={n0} pnl={pnl0:+.2f} roi={roi0:+.2f}% wr={wr0:.2f}%")
for label,n2,pnl2,roi2,wr2 in path:
    print(f"  + {label:<22} -> n={n2:<5} pnl={pnl2:+8.2f}  roi={roi2:+7.2f}%  wr={wr2:6.2f}%")
if not path: print("  (no single filter improved total pnl beyond min_n)")

print()
print("="*100)
print("GREEDY SWEEP — maximize ROI%% (pnl per $ staked)  (min_n=%d per step)"%MIN_N)
print("="*100)
path,final = greedy(res, "roi")
print(f"  START: n={n0} pnl={pnl0:+.2f} roi={roi0:+.2f}% wr={wr0:.2f}%")
for label,n2,pnl2,roi2,wr2 in path:
    print(f"  + {label:<22} -> n={n2:<5} pnl={pnl2:+8.2f}  roi={roi2:+7.2f}%  wr={wr2:6.2f}%")
if not path: print("  (no single filter improved roi beyond min_n)")
