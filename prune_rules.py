import pandas as pd, numpy as np, warnings, re
warnings.filterwarnings('ignore')
from live_rule_filter import load_current_signals, parse_rules_file

ded = load_current_signals(["port_logs", "backtest_data/new_streams"])
tiers = parse_rules_file("FINAL_RULES_REPORT.txt")

FEE=0.07
R = pd.read_parquet("res_cache_5m_all.parquet")
BK = pd.read_parquet("book_5m.parquet")
BK["ts"]=pd.to_datetime(BK.mkt_open,utc=True); BK["asset"]=BK.asset.astype(str)
A5 = BK[BK.elapsed_req==5][["asset","ts","au","ad"]].rename(columns={"ad":"ask_up","au":"ask_dn"})
ded = ded.merge(R, on=["asset","ts"], how="left").merge(A5, on=["asset","ts"], how="left")
ded["ask"] = np.where(ded.side=="UP", ded.ask_up, ded.ask_dn)
ded["win"] = np.where(ded.side=="UP", ded.y, 1-ded.y)
ded["pps"] = np.where(ded.win==1, 1-ded.ask, -ded.ask) - FEE*ded.ask*(1-ded.ask)
ded["pnl5"] = ded.pps*(5.0/ded.ask)
resolvable = ded["y"].notna().values
win = ded.win.values
pnl = ded.pnl5.values
N = len(ded)

# ---- vectorized clause evaluation: precompute one boolean array per atomic clause ----
clause_cache = {}
def clause_mask(clause):
    if clause in clause_cache: return clause_cache[clause]
    if clause.startswith("asset=="):
        m = (ded.asset == clause.split("==",1)[1]).values
    elif clause.startswith("side=="):
        m = (ded.side == clause.split("==",1)[1]).values
    elif clause.startswith("cell=="):
        v = clause.split("==",1)[1]
        m = ded.cells.apply(lambda fs: v in fs).values
    elif clause.startswith("stream=="):
        v = clause.split("==",1)[1]
        m = ded.streams.apply(lambda fs: v in fs).values
    elif clause.startswith("prob>="):
        v = float(clause.split(">=",1)[1]); m = (ded.prob >= v).values
    elif clause.startswith("prob<="):
        v = float(clause.split("<=",1)[1]); m = (ded.prob <= v).values
    else:
        raise ValueError(clause)
    clause_cache[clause] = m
    return m

def rule_mask(clauses):
    m = np.ones(N, dtype=bool)
    for c in clauses:
        m &= clause_mask(c)
    return m

TIER_THRESH = {"100%+": 100.0, "90%+": 90.0, "75%+": 75.0}

for tier_name, rules in tiers.items():
    thresh = TIER_THRESH[tier_name]
    good_masks, bad_rules, good_rules = [], [], []
    for clauses in rules:
        m = rule_mask(clauses)
        res_m = m & resolvable
        nr = res_m.sum()
        if nr == 0:
            continue  # rule matches nothing resolvable currently; keep it (untested, not "bad")
        w = win[res_m].sum()
        wr = w / nr * 100
        if wr >= thresh:
            good_rules.append((clauses, nr, w, wr))
            good_masks.append(m)
        else:
            bad_rules.append((clauses, nr, w, wr))

    baseline_mask = np.zeros(N, dtype=bool)
    for clauses in rules:
        baseline_mask |= rule_mask(clauses)
    pruned_mask = np.zeros(N, dtype=bool)
    for m in good_masks:
        pruned_mask |= m

    def summarize(mask, label):
        res_m = mask & resolvable
        n = mask.sum(); nr = res_m.sum(); w = win[res_m].sum()
        wr = w/nr*100 if nr else 0; p = pnl[res_m].sum()
        print(f"  {label:<10} n={n:<6} resolvable={nr:<6} W={int(w):<5} WR={wr:6.2f}%  pnl@5={p:+9.2f}")

    print(f"\n{'='*90}\nTIER {tier_name}  ({len(rules)} total rules; {len(bad_rules)} flagged bad, {len(good_rules)} kept)\n{'='*90}")
    summarize(baseline_mask, "baseline")
    summarize(pruned_mask, "pruned")
    print(f"\n  worst bad rules dropped (by n, current-pool WR < {thresh}%):")
    for clauses, nr, w, wr in sorted(bad_rules, key=lambda x: -x[1])[:10]:
        print(f"    {' & '.join(clauses):<65} n={nr:<4} W={int(w):<4} WR={wr:6.2f}%")
