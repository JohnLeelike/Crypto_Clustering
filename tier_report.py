import pickle, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
with open("tier_sweep_results.pkl","rb") as f:
    d = pickle.load(f)
results, ded = d["results"], d["ded"]
win = ded.win.values.astype(bool)
pnl = ded.pnl5.values
N = len(ded)

# recompute masks is expensive; instead re-derive membership by filtering results directly,
# but we need actual event-index sets for union dedup -> rebuild masks from labels
res_full = pd.read_parquet("res_flat.parquet")
filters = {}
for a in ded.asset.dropna().unique(): filters[f"asset=={a}"] = (ded.asset==a).values
for s in ["UP","DOWN"]: filters[f"side=={s}"] = (ded.side==s).values
for c in sorted(set().union(*res_full.cell.apply(lambda x:{x}))):
    filters[f"cell=={c}"] = ded.cells.apply(lambda s: c in s).values
for st in sorted(res_full.stream.unique()):
    filters[f"stream=={st}"] = ded.streams.apply(lambda s: st in s).values
probs = ded.prob.dropna()
qs = sorted(set(round(x,3) for x in np.quantile(probs, np.linspace(0.1,0.9,9))))
for q in qs:
    filters[f"prob>={q}"] = (ded.prob>=q).values
    filters[f"prob<={q}"] = (ded.prob<=q).values

def combo_mask(labels):
    m = np.ones(N, dtype=bool)
    for l in labels: m &= filters[l]
    return m

# dedupe identical result-sets (keep the simplest/shortest label combo)
seen = {}
for labels, n, w, wr, p in results:
    mask = combo_mask(labels)
    key = tuple(np.where(mask)[0])
    if key not in seen or len(labels) < len(seen[key][0]):
        seen[key] = (labels, n, w, wr, p, mask)

uniq = list(seen.values())
print(f"unique result-sets after dedup: {len(uniq)} (from {len(results)} raw combos)")

def tier_report(name, lo, hi):
    qualifying = [x for x in uniq if lo <= x[3] <= hi]
    qualifying.sort(key=lambda x: (-x[3], -x[1]))
    print(f"\n{'='*110}\nTIER: {name}  ({len(qualifying)} qualifying cells, n>=5)\n{'='*110}")
    for labels, n, w, wr, p, mask in qualifying[:25]:
        print(f"  {' & '.join(labels):<70} n={n:<4} w={w:<4} WR={wr:6.2f}%  pnl@5={p:+8.2f}")
    if len(qualifying) > 25: print(f"  ... and {len(qualifying)-25} more")
    # union
    if qualifying:
        union_mask = np.zeros(N, dtype=bool)
        for *_, mask in qualifying: union_mask |= mask
        un = union_mask.sum(); uw = win[union_mask].sum(); uwr = uw/un*100 if un else 0
        upnl = pnl[union_mask].sum()
        print(f"\n  UNION of all {name} cells (deduped events, counted once):")
        print(f"    N={un}  wins={uw}  WR={uwr:.2f}%  total pnl@5={upnl:+.2f}  ROI={upnl/(5*un)*100 if un else 0:+.2f}%")
        return union_mask
    else:
        print("  (none qualify)")
        return np.zeros(N, dtype=bool)

m100 = tier_report("100% WR", 100, 100)
m90  = tier_report(">=90% WR", 90, 99.999)
m75  = tier_report(">=75% WR", 75, 89.999)

print(f"\n{'='*110}\nGRAND UNION: all cells that are >=75% WR (75-89.9) OR >=90% (90-99.9) OR =100%\n{'='*110}")
grand = m100 | m90 | m75
gn=grand.sum(); gw=win[grand].sum(); gwr=gw/gn*100 if gn else 0; gp=pnl[grand].sum()
print(f"  N={gn}  wins={gw}  WR={gwr:.2f}%  total pnl@5={gp:+.2f}  ROI={gp/(5*gn)*100 if gn else 0:+.2f}%")
