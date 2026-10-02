#!/usr/bin/env python3
"""
final_merged_tiers.py
======================
Builds the final 75%/90%/100% WR tier unions by greedily merging three
independently-discovered event sets:
  - "mine"   : exhaustive single/pair/triple filter sweep over ded_v2.parquet
               (all 19 original signal CSVs combined, deduped by asset+ts+side)
  - "theirs" : new_archive_sweep_unions.pkl (10 new streams only, deduped by asset+ts)
  - "run3"   : run3_sets.pkl (union_tiers.py rerun on all 19 files, deduped by asset+ts)

Merge rule per tier: pick whichever of the three sources has the highest
verified WR as the BASE, then add events from the other two sources ONLY if
they are winners (per the canonical ded_v2.parquet lookup) and not already
present. This guarantees N and WR can only increase or stay flat relative to
the base - it is NOT a tradeable filter rule, since it uses realized outcomes
to decide inclusion. It's a measurement of combined upside across all three
sweeps, not a live-deployable strategy.

ded_v2.parquet is used as the single canonical source of truth for win/pnl -
any event from "theirs" or "run3" that cannot be resolved against it is
dropped (see the earlier 77-vs-73 discrepancy: do not patch the lookup table
with a second, independently-built matching pipeline).
"""
import pandas as pd, numpy as np, itertools, pickle, warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1. Canonical lookup: win/pnl for every event, from the 19-file combined sweep
# ----------------------------------------------------------------------------
ded_v2 = pd.read_parquet("ded_v2.parquet")
res_v2 = pd.read_parquet("res_v2_flat.parquet")

ded_lookup = {}
for _, row in ded_v2.iterrows():
    key = (row.asset, str(row.ts), 1 if row.side == "UP" else 0)
    ded_lookup[key] = (bool(row.win), float(row.pnl5))

def stats_of(keyset):
    found = [k for k in keyset if k in ded_lookup]
    n = len(found)
    w = sum(ded_lookup[k][0] for k in found)
    p = sum(ded_lookup[k][1] for k in found)
    return n, w, (w / n * 100 if n else 0.0), p, len(keyset) - n

# ----------------------------------------------------------------------------
# 2. Rebuild "mine": exhaustive filter sweep (asset/side/cell/stream/prob,
#    singles+pairs+triples, min_n=10) over the 19-file combined dataset
# ----------------------------------------------------------------------------
def build_mine_sets():
    N = len(ded_v2); MIN_N = 10
    filt = {}
    for a in ded_v2.asset.dropna().unique(): filt[f"asset=={a}"] = (ded_v2.asset == a).values
    for s in ["UP", "DOWN"]: filt[f"side=={s}"] = (ded_v2.side == s).values
    all_cells = sorted(set().union(*res_v2.cell.apply(lambda x: {x})))
    for c in all_cells: filt[f"cell=={c}"] = ded_v2.cells.str.contains(c, regex=False).values
    all_streams = sorted(res_v2.stream.unique())
    for st in all_streams: filt[f"stream=={st}"] = ded_v2.streams.str.contains(st, regex=False).values
    probs = ded_v2.prob.dropna()
    qs = sorted(set(round(x, 3) for x in np.quantile(probs, np.linspace(0.1, 0.9, 9))))
    for q in qs:
        filt[f"prob>={q}"] = (ded_v2.prob >= q).values
        filt[f"prob<={q}"] = (ded_v2.prob <= q).values

    labels = list(filt.keys())
    win = ded_v2.win.values.astype(bool)
    pnl = ded_v2.pnl5.values
    results = []
    def record(labs, mask):
        n = mask.sum()
        if n < MIN_N: return
        results.append((labs, n, int(win[mask].sum()), win[mask].sum() / n * 100, pnl[mask].sum(), mask.copy()))
    for lab in labels: record((lab,), filt[lab])
    for l1, l2 in itertools.combinations(labels, 2):
        m = filt[l1] & filt[l2]
        if m.sum() >= MIN_N: record((l1, l2), m)
    for l1, l2, l3 in itertools.combinations(labels, 3):
        m = filt[l1] & filt[l2] & filt[l3]
        if m.sum() >= MIN_N: record((l1, l2, l3), m)

    seen = {}
    for labs, n, w, wr, p, mask in results:
        key = tuple(np.where(mask)[0])
        if key not in seen or len(labs) < len(seen[key][0]):
            seen[key] = (labs, n, w, wr, p, mask)
    uniq = list(seen.values())

    def union_keyset(lo, hi):
        qual = [x for x in uniq if lo <= x[3] <= hi]
        um = np.zeros(N, dtype=bool)
        for *_, m in qual: um |= m
        sub = ded_v2[um]
        return set(zip(sub.asset, sub.ts.astype(str), np.where(sub.side == "UP", 1, 0)))

    return {
        "100%+": union_keyset(100, 100),
        "90%+":  union_keyset(90, 99.999),
        "75%+":  union_keyset(75, 89.999),
    }

print("Rebuilding 'mine' (19-file combined exhaustive sweep)...")
mine_sets = build_mine_sets()

# ----------------------------------------------------------------------------
# 3. Load the other two pre-computed sources
# ----------------------------------------------------------------------------
with open("new_archive_sweep_unions.pkl", "rb") as f:
    theirs_sets = pickle.load(f)   # 10-new-streams-only sweep
with open("run3_sets.pkl", "rb") as f:
    run3_sets = pickle.load(f)     # union_tiers.py rerun on all 19 files

SOURCES = {"mine": mine_sets, "theirs": theirs_sets, "run3": run3_sets}

# ----------------------------------------------------------------------------
# 4. Greedy merge per tier: base = highest verified WR, then add wins-only
#    from the other two sources (events present in ded_v2 and are winners).
# ----------------------------------------------------------------------------
print(f"\n{'tier':<8}{'base':<8}{'final N':>10}{'WR':>9}{'pnl@5':>12}   delta vs base")
print("-" * 70)

FINAL_SETS = {}
for tier in ["100%+", "90%+", "75%+"]:
    cands = {name: SOURCES[name].get(tier, set()) for name in SOURCES}
    ranked = sorted(cands.items(), key=lambda kv: -stats_of(kv[1])[2])
    base_name, base_set = ranked[0]
    n0, w0, wr0, p0, _ = stats_of(base_set)

    merged = set(base_set)
    for name, s in ranked[1:]:
        extra = s - merged
        extra_wins = {k for k in extra if ded_lookup.get(k, (False, 0))[0]}
        merged |= extra_wins

    n, w, wr, p, _ = stats_of(merged)
    FINAL_SETS[tier] = merged
    print(f"{tier:<8}{base_name:<8}{n:>10}{wr:>8.2f}%{p:>+12.2f}   "
          f"+{n-n0} events, {wr-wr0:+.2f}pp, ${p-p0:+.2f}")

# ----------------------------------------------------------------------------
# 5. Save final event lists for each tier to CSV
# ----------------------------------------------------------------------------
for tier, keyset in FINAL_SETS.items():
    rows = []
    verified = [k for k in keyset if k in ded_lookup]  # drop any unresolvable base-set keys
    for (asset, ts, up) in sorted(verified, key=lambda k: k[1]):
        win, pnl = ded_lookup[(asset, ts, up)]
        rows.append({"asset": asset, "ts": ts, "side": "UP" if up else "DOWN",
                      "win": win, "pnl5": round(pnl, 2)})
    fname = f"final_tier_{tier.replace('%','pct').replace('+','plus')}.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"saved {fname} ({len(rows)} events)")

print("""
REMINDER: these unions were built by keeping only WINNING events when adding
each non-base source (lookahead). They measure the combined ceiling of the
three sweeps' discoveries, not a tradeable rule. See final_tier_*.csv for the
exact event lists.
""")
