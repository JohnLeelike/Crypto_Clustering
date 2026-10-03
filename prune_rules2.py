import pandas as pd, numpy as np, warnings
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
resolvable = (ded["y"].notna() & ded["ask"].notna()).values   # fixed: also require ask present
win = np.nan_to_num(ded.win.values)
pnl = np.nan_to_num(ded.pnl5.values)
N = len(ded)

clause_cache = {}
def clause_mask(clause):
    if clause in clause_cache: return clause_cache[clause]
    if clause.startswith("asset=="): m = (ded.asset == clause.split("==",1)[1]).values
    elif clause.startswith("side=="): m = (ded.side == clause.split("==",1)[1]).values
    elif clause.startswith("cell=="):
        v = clause.split("==",1)[1]; m = ded.cells.apply(lambda fs: v in fs).values
    elif clause.startswith("stream=="):
        v = clause.split("==",1)[1]; m = ded.streams.apply(lambda fs: v in fs).values
    elif clause.startswith("prob>="): m = (ded.prob >= float(clause.split(">=",1)[1])).values
    elif clause.startswith("prob<="): m = (ded.prob <= float(clause.split("<=",1)[1])).values
    else: raise ValueError(clause)
    clause_cache[clause] = m
    return m

def rule_mask(clauses):
    m = np.ones(N, dtype=bool)
    for c in clauses: m &= clause_mask(c)
    return m

def summarize(mask):
    res_m = mask & resolvable
    n = int(mask.sum()); nr = int(res_m.sum()); w = int(win[res_m].sum())
    wr = w/nr*100 if nr else 0.0; p = float(pnl[res_m].sum())
    return n, nr, w, wr, p

TIER_THRESH = {"100%+": 100.0, "90%+": 90.0, "75%+": 75.0}

all_rule_masks = {}
for tier_name, rules in tiers.items():
    masks = [(clauses, rule_mask(clauses)) for clauses in rules]
    all_rule_masks[tier_name] = masks

print(f"{'tier':<8}{'prune@':>8}{'n':>6}{'resolv':>8}{'WR':>8}{'pnl':>10}   kept/total rules")
for tier_name, masks in all_rule_masks.items():
    baseline_mask = np.zeros(N, dtype=bool)
    for _, m in masks: baseline_mask |= m
    n,nr,w,wr,p = summarize(baseline_mask)
    print(f"{tier_name:<8}{'none':>8}{n:>6}{nr:>8}{wr:>7.2f}%{p:>+10.2f}   {len(masks)}/{len(masks)}")

    for prune_at in [TIER_THRESH[tier_name], 80, 85, 90, 95, 100]:
        if prune_at < TIER_THRESH[tier_name]: continue
        good = []
        for clauses, m in masks:
            res_m = m & resolvable
            nr_r = res_m.sum()
            if nr_r == 0:
                good.append(m); continue
            wr_r = win[res_m].sum()/nr_r*100
            if wr_r >= prune_at:
                good.append(m)
        pm = np.zeros(N, dtype=bool)
        for m in good: pm |= m
        n,nr,w,wr,p = summarize(pm)
        print(f"{tier_name:<8}{prune_at:>7.0f}%{n:>6}{nr:>8}{wr:>7.2f}%{p:>+10.2f}   {len(good)}/{len(masks)}")
    print()
