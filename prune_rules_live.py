#!/usr/bin/env python3
"""
prune_rules_live.py
=====================
Periodic maintenance step (needs res_cache_5m_all.parquet + book_5m.parquet,
i.e. historical resolution data -- run this nightly/weekly, NOT on every
new signal). Scores every rule in FINAL_RULES_REPORT.txt against its own
historical track record and drops rules whose current win rate has fallen
below a tier-specific bar, writing a trimmed PRUNED_RULES_REPORT.txt in the
same format.

This is NOT per-event lookahead: each rule is judged on its own aggregate
past performance (already-resolved signals), not on which future signal it
will match next. live_rule_filter.py then consumes the pruned file with no
resolution data needed at all -- that stays a pure, causal live filter.

Recommended prune bars (found by sweeping): 100%+ -> 100%, 90%+ -> 90%,
75%+ -> 80% (NOT 75% -- see README output below for why).

USAGE
-----
  python3 prune_rules_live.py [--rules-file FILE] [--out FILE]
                               [--prune-100 PCT] [--prune-90 PCT] [--prune-75 PCT]
"""
import argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from live_rule_filter import load_current_signals, parse_rules_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", action="append", default=["port_logs", "backtest_data/new_streams"])
    ap.add_argument("--rules-file", default="FINAL_RULES_REPORT.txt")
    ap.add_argument("--out", default="PRUNED_RULES_REPORT.txt")
    ap.add_argument("--prune-100", type=float, default=100.0)
    ap.add_argument("--prune-90", type=float, default=90.0)
    ap.add_argument("--prune-75", type=float, default=80.0)  # 80, not 75 -- see sweep results
    args = ap.parse_args()
    prune_bar = {"100%+": args.prune_100, "90%+": args.prune_90, "75%+": args.prune_75}

    ded = load_current_signals(args.signals_dir)
    tiers = parse_rules_file(args.rules_file)

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
    def clause_mask(clause):
        if clause in clause_cache: return clause_cache[clause]
        if clause.startswith("asset=="): m = (ded.asset == clause.split("==", 1)[1]).values
        elif clause.startswith("side=="): m = (ded.side == clause.split("==", 1)[1]).values
        elif clause.startswith("cell=="):
            v = clause.split("==", 1)[1]; m = ded.cells.apply(lambda fs: v in fs).values
        elif clause.startswith("stream=="):
            v = clause.split("==", 1)[1]; m = ded.streams.apply(lambda fs: v in fs).values
        elif clause.startswith("prob>="): m = (ded.prob >= float(clause.split(">=", 1)[1])).values
        elif clause.startswith("prob<="): m = (ded.prob <= float(clause.split("<=", 1)[1])).values
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
        wr = w / nr * 100 if nr else 0.0; p = float(pnl[res_m].sum())
        return n, nr, w, wr, p

    with open(args.out, "w") as f:
        f.write("PRUNED RULES -- each rule validated against its own historical track record\n")
        f.write("=" * 100 + "\n\n")
        for tier_name, rules in tiers.items():
            bar = prune_bar[tier_name]
            baseline_mask = np.zeros(N, dtype=bool)
            kept, dropped = [], []
            for clauses in rules:
                m = rule_mask(clauses)
                baseline_mask |= m
                res_m = m & resolvable
                nr = res_m.sum()
                if nr == 0:
                    kept.append((clauses, m)); continue  # untested, not provably bad -- keep
                wr = win[res_m].sum() / nr * 100
                if wr >= bar:
                    kept.append((clauses, m))
                else:
                    dropped.append(clauses)
            pruned_mask = np.zeros(N, dtype=bool)
            for _, m in kept: pruned_mask |= m

            n0, nr0, w0, wr0, p0 = summarize(baseline_mask)
            n1, nr1, w1, wr1, p1 = summarize(pruned_mask)
            print(f"[{tier_name}] prune_bar={bar}%  before: n={n0} WR={wr0:.2f}% pnl={p0:+.2f}  "
                  f"-> after: n={n1} WR={wr1:.2f}% pnl={p1:+.2f}  ({len(dropped)} rules dropped of {len(rules)})")

            f.write(f"TIER {tier_name}  (pruned at {bar}%, {len(kept)}/{len(rules)} rules kept)\n")
            f.write("-" * 100 + "\n")
            for clauses, _ in kept:
                f.write(f"      {' & '.join(clauses):<70}  n=0    WR=  0.00%  pnl@5=    +0.00   covers 0 final event(s)\n")
            f.write("\n")
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
