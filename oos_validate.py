import pandas as pd, numpy as np, itertools, warnings; warnings.filterwarnings('ignore')
ded = pd.read_parquet("ded_v2.parquet")
res = pd.read_parquet("res_v2_flat.parquet")
MID = ded.ts.median()
train = ded[ded.ts < MID].reset_index(drop=True)
test  = ded[ded.ts >= MID].reset_index(drop=True)
print(f"train n={len(train)}  test n={len(test)}")

MIN_N_TRAIN = 8  # lower floor since train is half the data

def build_filters(df, all_cells, all_streams, probs_ref=None):
    filters = {}
    for a in df.asset.dropna().unique(): filters[f"asset=={a}"] = (df.asset==a).values
    for s in ["UP","DOWN"]: filters[f"side=={s}"] = (df.side==s).values
    for c in all_cells:
        filters[f"cell=={c}"] = df.cells.str.contains(c, regex=False).values
    for st in all_streams:
        filters[f"stream=={st}"] = df.streams.str.contains(st, regex=False).values
    probs = probs_ref if probs_ref is not None else df.prob.dropna()
    qs = sorted(set(round(x,3) for x in np.quantile(probs, np.linspace(0.1,0.9,9))))
    for q in qs:
        filters[f"prob>={q}"] = (df.prob>=q).values
        filters[f"prob<={q}"] = (df.prob<=q).values
    return filters

all_cells = sorted(set().union(*res.cell.apply(lambda x:{x})))
all_streams = sorted(res.stream.unique())

F_train = build_filters(train, all_cells, all_streams)
labels = list(F_train.keys())
win_tr = train.win.values.astype(bool)

results = []
def record(labs, mask):
    n = mask.sum()
    if n < MIN_N_TRAIN: return
    w = win_tr[mask].sum()
    wr = w/n*100
    results.append((labs, n, int(w), wr, mask.copy()))

for lab in labels: record((lab,), F_train[lab])
for l1,l2 in itertools.combinations(labels,2):
    m = F_train[l1] & F_train[l2]
    if m.sum() < MIN_N_TRAIN: continue
    record((l1,l2), m)
for l1,l2,l3 in itertools.combinations(labels,3):
    m = F_train[l1] & F_train[l2] & F_train[l3]
    if m.sum() < MIN_N_TRAIN: continue
    record((l1,l2,l3), m)

seen={}
for labs,n,w,wr,mask in results:
    key=tuple(np.where(mask)[0])
    if key not in seen or len(labs)<len(seen[key][0]): seen[key]=(labs,n,w,wr,mask)
uniq=list(seen.values())
print(f"TRAIN: candidates evaluated(n>={MIN_N_TRAIN}): {len(results)}  unique result-sets: {len(uniq)}")

# Build matching filters on TEST using identical definitions (prob thresholds from TRAIN quantiles reused)
F_test = {}
for a in test.asset.dropna().unique(): F_test[f"asset=={a}"]=(test.asset==a).values
for s in ["UP","DOWN"]: F_test[f"side=={s}"]=(test.side==s).values
for c in all_cells: F_test[f"cell=={c}"]=test.cells.str.contains(c,regex=False).values
for st in all_streams: F_test[f"stream=={st}"]=test.streams.str.contains(st,regex=False).values
probs_tr = train.prob.dropna()
qs = sorted(set(round(x,3) for x in np.quantile(probs_tr, np.linspace(0.1,0.9,9))))
for q in qs:
    F_test[f"prob>={q}"]=(test.prob>=q).values
    F_test[f"prob<={q}"]=(test.prob<=q).values

win_te = test.win.values.astype(bool)
pnl_te = test.pnl5.values

def eval_on_test(labs):
    m = np.ones(len(test), dtype=bool)
    for l in labs:
        if l not in F_test: return None
        m &= F_test[l]
    n = m.sum()
    if n==0: return (0,0,0,0.0)
    w = win_te[m].sum()
    return (n, int(w), w/n*100, float(pnl_te[m].sum()))

for tier_name, lo, hi in [("100%",100,100), (">=90%",90,99.999), (">=75%",75,89.999)]:
    qual = [x for x in uniq if lo <= x[3] <= hi]
    qual.sort(key=lambda x:(-x[3],-x[1]))
    print(f"\n{'='*110}\nTRAIN TIER {tier_name}: {len(qual)} rules found (n>={MIN_N_TRAIN} on train)\n{'='*110}")
    print(f"  {'rule':<60} {'TRAIN n/WR':<16} {'TEST n':>7} {'TEST W':>7} {'TEST WR':>9} {'TEST pnl':>10}")
    test_union_mask = np.zeros(len(test), dtype=bool)
    for labs,n,w,wr,mask in qual[:15]:
        r = eval_on_test(labs)
        tn,tw,twr,tpnl = r
        test_union_mask |= (eval_mask := np.ones(len(test),dtype=bool))
        m2 = np.ones(len(test),dtype=bool)
        for l in labs: m2 &= F_test[l]
        test_union_mask = test_union_mask  # placeholder, recompute union below properly
        print(f"  {' & '.join(labs)[:58]:<60} {f'{n}/{wr:.1f}%':<16} {tn:>7} {tw:>7} {twr:>8.2f}% {tpnl:>+10.2f}")
    # proper union over ALL qualifying train-rules, applied/tested on TEST data
    union_mask = np.zeros(len(test), dtype=bool)
    for labs,n,w,wr,mask in qual:
        m2 = np.ones(len(test), dtype=bool)
        for l in labs: m2 &= F_test[l]
        union_mask |= m2
    un = union_mask.sum(); uw = win_te[union_mask].sum() if un else 0
    uwr = uw/un*100 if un else 0
    upnl = pnl_te[union_mask].sum() if un else 0
    print(f"\n  TRAIN-discovered rules, UNION applied to TEST: N={un}  wins={uw}  WR={uwr:.2f}%  pnl@5={upnl:+.2f}  ROI={upnl/(5*un)*100 if un else 0:+.2f}%")
