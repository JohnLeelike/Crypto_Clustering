import re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from live_rule_filter import load_current_signals

RULE_LINE = re.compile(r'^\s*(?P<clauses>.+?)\s{2,}n=(?P<n>\d+)\s+WR=')
TIER_HEADER = re.compile(r'^TIER (?P<tier>\d+%\+)\s')

def parse_with_n(path):
    tiers = {}
    current = None
    with open(path) as f:
        for line in f:
            m = TIER_HEADER.match(line)
            if m:
                current = m.group("tier"); tiers.setdefault(current, {})
                continue
            m = RULE_LINE.match(line)
            if m and current:
                clauses = tuple(c.strip() for c in m.group("clauses").split(" & "))
                n = int(m.group("n"))
                if n > 0:  # keep the real recorded n (PRUNED file has n=0 placeholders)
                    tiers[current][clauses] = n
    return tiers

def parse_clause_set(path):
    tiers = {}
    current = None
    with open(path) as f:
        for line in f:
            m = TIER_HEADER.match(line)
            if m:
                current = m.group("tier"); tiers.setdefault(current, set())
                continue
            m = RULE_LINE.match(line)
            if m and current:
                clauses = tuple(c.strip() for c in m.group("clauses").split(" & "))
                tiers[current].add(clauses)
    return tiers

final_n = parse_with_n("FINAL_RULES_REPORT.txt")
pruned_clauses = parse_clause_set("PRUNED_RULES_REPORT.txt")

ded = load_current_signals(["port_logs", "backtest_data/new_streams"])
FEE = 0.07
R = pd.read_parquet("res_cache_5m_all.parquet")
BK = pd.read_parquet("book_5m.parquet")
BK["ts"] = pd.to_datetime(BK.mkt_open, utc=True); BK["asset"] = BK.asset.astype(str)
A5 = BK[BK.elapsed_req == 5][["asset", "ts", "au", "ad"]].rename(columns={"ad": "ask_up", "au": "ask_dn"})
ded = ded.merge(R, on=["asset", "ts"], how="left").merge(A5, on=["asset", "ts"], how="left")
ded["ask"] = np.where(ded.side == "UP", ded.ask_up, ded.ask_dn)
ded["win"] = np.where(ded.side == "UP", ded.y, 1 - ded.y)
ded["pps"] = np.where(ded.win == 1, 1 - ded.ask, -ded.ask) - FEE * ded.ask * (1 - ded.ask)
ded["pnl5"] = ded.pps * (5.0 / ded.ask)
resolvable = (ded["y"].notna() & ded["ask"].notna()).values
win = np.nan_to_num(ded.win.values)
pnl = np.nan_to_num(ded.pnl5.values)
N = len(ded)

clause_cache = {}
def clause_mask(c):
    if c in clause_cache: return clause_cache[c]
    if c.startswith("asset=="): m = (ded.asset == c.split("==",1)[1]).values
    elif c.startswith("side=="): m = (ded.side == c.split("==",1)[1]).values
    elif c.startswith("cell=="): v=c.split("==",1)[1]; m = ded.cells.apply(lambda fs: v in fs).values
    elif c.startswith("stream=="): v=c.split("==",1)[1]; m = ded.streams.apply(lambda fs: v in fs).values
    elif c.startswith("prob>="): m = (ded.prob >= float(c.split(">=",1)[1])).values
    elif c.startswith("prob<="): m = (ded.prob <= float(c.split("<=",1)[1])).values
    else: raise ValueError(c)
    clause_cache[c] = m
    return m

def rule_mask(clauses):
    m = np.ones(N, dtype=bool)
    for c in clauses: m &= clause_mask(c)
    return m

def eval_set(clause_list):
    mask = np.zeros(N, dtype=bool)
    for clauses in clause_list: mask |= rule_mask(clauses)
    res_m = mask & resolvable
    n = int(mask.sum()); nr = int(res_m.sum()); w = int(win[res_m].sum())
    wr = w/nr*100 if nr else 0.0; p = float(pnl[res_m].sum())
    return n, nr, w, wr, p

print(f"{'tier':<8}{'variant':<14}{'n_rules(>=20)':>14}{'matched':>9}{'resolvable':>12}{'W':>6}{'WR':>8}{'pnl':>10}")
for tier in ["100%+","90%+","75%+"]:
    all_rules = final_n.get(tier, {})
    rules_ge20 = [c for c,n in all_rules.items() if n>=20]
    n,nr,w,wr,p = eval_set(rules_ge20)
    print(f"{tier:<8}{'live':<14}{len(rules_ge20):>14}{n:>9}{nr:>12}{w:>6}{wr:>7.2f}%{p:>+10.2f}")

    pruned_set = pruned_clauses.get(tier, set())
    rules_pruned_ge20 = [c for c in rules_ge20 if c in pruned_set]
    n,nr,w,wr,p = eval_set(rules_pruned_ge20)
    print(f"{tier:<8}{'pruned live':<14}{len(rules_pruned_ge20):>14}{n:>9}{nr:>12}{w:>6}{wr:>7.2f}%{p:>+10.2f}")
    print()
