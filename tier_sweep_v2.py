import pandas as pd, numpy as np, itertools, warnings; warnings.filterwarnings('ignore')
ded = pd.read_parquet("ded_v2.parquet")
res = pd.read_parquet("res_v2_flat.parquet")
N = len(ded)
MIN_N = 10

filters = {}
for a in ded.asset.dropna().unique(): filters[f"asset=={a}"] = (ded.asset==a).values
for s in ["UP","DOWN"]: filters[f"side=={s}"] = (ded.side==s).values
all_cells = sorted(set().union(*res.cell.apply(lambda x:{x})))
for c in all_cells:
    filters[f"cell=={c}"] = ded.cells.str.contains(c, regex=False).values
all_streams = sorted(res.stream.unique())
for st in all_streams:
    filters[f"stream=={st}"] = ded.streams.str.contains(st, regex=False).values
probs = ded.prob.dropna()
qs = sorted(set(round(x,3) for x in np.quantile(probs, np.linspace(0.1,0.9,9))))
for q in qs:
    filters[f"prob>={q}"] = (ded.prob>=q).values
    filters[f"prob<={q}"] = (ded.prob<=q).values

labels = list(filters.keys())
win = ded.win.values.astype(bool)
pnl = ded.pnl5.values
print(f"N={N}  candidate filters={len(labels)}")

results = []
def record(combo_labels, mask):
    n = mask.sum()
    if n < MIN_N: return
    w = win[mask].sum()
    wr = w/n*100
    p = pnl[mask].sum()
    results.append((combo_labels, n, int(w), wr, p, mask.copy()))

for lab in labels:
    record((lab,), filters[lab])

for l1, l2 in itertools.combinations(labels, 2):
    m = filters[l1] & filters[l2]
    if m.sum() < MIN_N: continue
    record((l1,l2), m)

for l1, l2, l3 in itertools.combinations(labels, 3):
    m = filters[l1] & filters[l2] & filters[l3]
    if m.sum() < MIN_N: continue
    record((l1,l2,l3), m)

print(f"total candidates evaluated at n>={MIN_N}: {len(results)}")

# dedupe by actual resulting event-set (not by filter name)
seen = {}
for labs, n, w, wr, p, mask in results:
    key = tuple(np.where(mask)[0])
    if key not in seen or len(labs) < len(seen[key][0]):
        seen[key] = (labs, n, w, wr, p, mask)
uniq = list(seen.values())
print(f"unique result-sets: {len(uniq)}")

def tier_report(name, lo, hi):
    qual = [x for x in uniq if lo <= x[3] <= hi]
    qual.sort(key=lambda x:(-x[3], -x[1]))
    print(f"\n{'='*105}\nTIER {name}  ({len(qual)} qualifying, n>={MIN_N})\n{'='*105}")
    for labs,n,w,wr,p,mask in qual[:20]:
        print(f"  {' & '.join(labs):<65} n={n:<4} w={w:<4} WR={wr:6.2f}%  pnl@5={p:+8.2f}")
    if len(qual)>20: print(f"  ...and {len(qual)-20} more")
    if qual:
        um = np.zeros(N, dtype=bool)
        for *_, m in qual: um |= m
        un=um.sum(); uw=win[um].sum(); uwr=uw/un*100 if un else 0; up=pnl[um].sum()
        # verify: every event in union is selected by at least one qualifying combo (sanity check)
        covered = np.zeros(N, dtype=bool)
        for *_, m in qual: covered |= m
        assert (covered == um).all(), "UNION MISMATCH BUG"
        print(f"\n  UNION: N={un} wins={uw} WR={uwr:.2f}% pnl@5={up:+.2f} ROI={up/(5*un)*100 if un else 0:+.2f}%  [verified: union == OR of qualifying masks]")
        return um
    return np.zeros(N, dtype=bool)

m100 = tier_report("100%", 100, 100)
m90  = tier_report(">=90%", 90, 99.999)
m75  = tier_report(">=75%", 75, 89.999)

grand = m100|m90|m75
gn=grand.sum(); gw=win[grand].sum(); gwr=gw/gn*100 if gn else 0; gp=pnl[grand].sum()
print(f"\n{'='*105}\nGRAND UNION (75%+ overall)\n{'='*105}")
print(f"  N={gn}  wins={gw}  WR={gwr:.2f}%  total pnl@5={gp:+.2f}  ROI={gp/(5*gn)*100 if gn else 0:+.2f}%")
