import pandas as pd, numpy as np, itertools, warnings; warnings.filterwarnings('ignore')
res = pd.read_parquet("res_flat.parquet")

ded = (res.groupby(["asset","ts","side"])
          .agg(prob=("prob","first"), win=("win","first"), ask=("ask","first"), pnl5=("pnl5","first"),
               cells=("cell", lambda s: frozenset(s)), streams=("stream", lambda s: frozenset(s)))
          .reset_index())
N = len(ded)
print(f"deduped unique events: {N}")

# ---- build single-filter boolean columns ----
filters = {}  # label -> bool array
for a in ded.asset.dropna().unique(): filters[f"asset=={a}"] = (ded.asset==a).values
for s in ["UP","DOWN"]: filters[f"side=={s}"] = (ded.side==s).values
for c in sorted(set().union(*res.cell.apply(lambda x:{x}))):
    filters[f"cell=={c}"] = ded.cells.apply(lambda s: c in s).values
for st in sorted(res.stream.unique()):
    filters[f"stream=={st}"] = ded.streams.apply(lambda s: st in s).values
probs = ded.prob.dropna()
qs = sorted(set(round(x,3) for x in np.quantile(probs, np.linspace(0.1,0.9,9))))
for q in qs:
    filters[f"prob>={q}"] = (ded.prob>=q).values
    filters[f"prob<={q}"] = (ded.prob<=q).values

labels = list(filters.keys())
mats = {k: v for k,v in filters.items()}
win = ded.win.values.astype(bool)
pnl = ded.pnl5.values

MIN_N = 5
results = []  # (labels_tuple, n, wins, wr, pnl_total)

def record(combo_labels, mask):
    n = mask.sum()
    if n < MIN_N: return
    w = win[mask].sum()
    wr = w/n*100
    p = pnl[mask].sum()
    results.append((combo_labels, n, int(w), wr, p))

# singles
for lab in labels:
    record((lab,), mats[lab])

# pairs
for l1, l2 in itertools.combinations(labels, 2):
    m = mats[l1] & mats[l2]
    if m.sum() < MIN_N: continue
    record((l1,l2), m)

# triples
for l1, l2, l3 in itertools.combinations(labels, 3):
    m = mats[l1] & mats[l2] & mats[l3]
    if m.sum() < MIN_N: continue
    record((l1,l2,l3), m)

print(f"total candidate combos evaluated (n>={MIN_N}): {len(results)}")

import pickle
with open("tier_sweep_results.pkl","wb") as f:
    pickle.dump({"results":results, "ded":ded}, f)
print("saved tier_sweep_results.pkl")
