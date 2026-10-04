import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/newdata")
from build_pruned import parse_with_n, build_ded, make_eval

final_n = parse_with_n("/home/user/Crypto_Clustering/FINAL_RULES_REPORT.txt")
ded = build_ded("/home/user/Crypto_Clustering/port_logs",
                 "/home/user/Crypto_Clustering/backtest_data/new_streams",
                 "/home/user/Crypto_Clustering/book_5m.parquet",
                 "/home/user/Crypto_Clustering/res_cache_5m_all.parquet")
eval_mask = make_eval(ded)
win = ded["win"].values
pnl = ded["pnl5"].values

MIN_N = 20

def rule_stats(clauses):
    m = eval_mask([clauses])
    n = int(m.sum())
    if n == 0: return n, 0.0, 0.0
    return n, float(win[m].mean()*100), float(pnl[m].sum())

BARS = [100,99,98,97,96,95,94,93,92,91,90,88,86,84,82,80,78,76,75,73,71,69,67,65,63,61,60,58,56,54,52,50]

results = {}
for tier, target in [("100%+", 100.0), ("90%+", 90.0), ("75%+", 75.0)]:
    all_rules = final_n.get(tier, {})
    stats = {c: rule_stats(c) for c in all_rules}

    print(f"\n=== TIER {tier} (target {target}%) ===  total candidate rules: {len(all_rules)}")
    sweep = []
    for bar in BARS:
        kept = [c for c,(n,wr,p) in stats.items() if n>=MIN_N and wr>=bar]
        if not kept: continue
        mask = eval_mask(kept)
        nn = int(mask.sum())
        if nn == 0: continue
        agg_wr = float(win[mask].mean()*100)
        agg_pnl = float(pnl[mask].sum())
        sweep.append((bar, kept, nn, agg_wr, agg_pnl))
        marker = "<=TARGET" if agg_wr >= target else ""
        print(f"  bar={bar:>5.1f}%  rules_kept={len(kept):>4}  n={nn:>4}  agg_WR={agg_wr:6.2f}%  pnl=${agg_pnl:+8.2f} {marker}")

    meeting = [s for s in sweep if s[3] >= target]
    if meeting:
        best = max(meeting, key=lambda s: s[2])   # largest n among those meeting target
    elif sweep:
        best = max(sweep, key=lambda s: (s[3], s[2]))  # else highest agg_WR, tie-break by n
    else:
        best = None

    results[tier] = {"all_rules": all_rules, "stats": stats, "best": best}
    if best:
        bar, kept, nn, agg_wr, agg_pnl = best
        print(f"  -> chosen bar={bar}%  kept {len(kept)} rules  n={nn}  agg_WR={agg_wr:.2f}%  pnl=${agg_pnl:+.2f}")
    else:
        print("  -> no viable rule set found at all")

pickle.dump(results, open("/tmp/newdata/prune_v2_results.pkl","wb"))
