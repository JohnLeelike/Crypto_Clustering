#!/usr/bin/env python3
"""
live_rule_filter.py
====================
Pure live-trading filter: no backtest scoring, no res_cache/book_5m needed.
Reads whatever Glass-format signal CSVs your signal-gen service is currently
writing, parses the rule list out of FINAL_RULES_REPORT.txt, and reports
which CURRENT signals qualify for the 100%+ / 90%+ / 75%+ tiers.

A signal qualifies for a tier if it matches AT LEAST ONE of that tier's
rules (asset / side / cell / stream / prob conditions -- all of these are
known the instant a signal fires, so this is a legitimate real-time filter,
unlike the backtest's cross-source "win-only" merge step which used
realized outcomes and can't be replayed live).

USAGE
-----
  python3 live_rule_filter.py [--signals-dir DIR ...] [--rules-file FILE]
                              [--pending-only]

  --signals-dir can be passed multiple times (default: port_logs,
  backtest_data/new_streams). Any *.csv in those dirs with a 'forecast'
  column is treated as a signal file.
  --pending-only restricts output to signals whose 'result' column is
  still pending (⏳) -- i.e. signals you could actually still act on.
"""
import argparse, glob, os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SYM = {"BTC":"btc","ETH":"eth","SOL":"sol","XRP":"xrp","DOGE":"doge","BNB":"bnb","HYPE":"hype"}

# ============================================================================
# 1. Parse FINAL_RULES_REPORT.txt into {tier: [ [clause, clause, ...], ... ]}
# ============================================================================
RULE_LINE = re.compile(r'^\s*(?P<clauses>.+?)\s{2,}n=\d+\s+WR=')
TIER_HEADER = re.compile(r'^TIER (?P<tier>\d+%\+)\s')

def parse_rules_file(path):
    tiers = {}
    current_tier = None
    seen_per_tier = {}
    with open(path) as f:
        for line in f:
            m = TIER_HEADER.match(line)
            if m:
                current_tier = m.group("tier")
                tiers.setdefault(current_tier, [])
                seen_per_tier.setdefault(current_tier, set())
                continue
            m = RULE_LINE.match(line)
            if m and current_tier:
                clause_str = m.group("clauses").strip()
                clauses = tuple(c.strip() for c in clause_str.split(" & "))
                if clauses not in seen_per_tier[current_tier]:
                    seen_per_tier[current_tier].add(clauses)
                    tiers[current_tier].append(clauses)
    return tiers

# ============================================================================
# 2. Load current signal CSVs (no res_cache / book join -- not needed to filter)
# ============================================================================
def load_current_signals(signal_dirs):
    files = []
    for d in signal_dirs:
        files += glob.glob(os.path.join(d, "*.csv"))
    rows = []
    for f in sorted(files):
        try:
            d = pd.read_csv(f, on_bad_lines="skip")
        except Exception:
            continue
        if "forecast" not in d.columns:
            continue
        stream = os.path.basename(f).replace(".csv", "")
        strat_col = "strategy" if "strategy" in d.columns else ("_15" if "_15" in d.columns else None)
        d["stream"] = stream
        d["cell"] = d[strat_col].astype(str) if strat_col else ""
        d["asset"] = d.symbol.str.split("/").str[0].str.upper().map(SYM)
        d["ts"] = pd.to_datetime(d.start_date, format="%m/%d/%y %H:%M", errors="coerce", utc=True)
        d["prob"] = pd.to_numeric(d.get("prob"), errors="coerce")
        d["side"] = np.where(d.forecast.astype(str).str.contains("⬆"), "UP",
                    np.where(d.forecast.astype(str).str.contains("⬇"), "DOWN", None))
        d["pending"] = d.get("result", "⏳").astype(str).str.contains("⏳")
        d["symbol_raw"] = d["symbol"]
        rows.append(d[["stream", "cell", "asset", "ts", "side", "prob", "pending", "symbol_raw"]])
    if not rows:
        return pd.DataFrame(columns=["asset","ts","side","prob","cells","streams","pending"])
    S = pd.concat(rows, ignore_index=True).dropna(subset=["ts", "side", "asset"])

    ded = (S.groupby(["asset", "ts", "side"])
             .agg(prob=("prob", "first"),
                  pending=("pending", "max"),
                  symbol_raw=("symbol_raw", "first"),
                  cells=("cell", lambda s: frozenset(s)),
                  streams=("stream", lambda s: frozenset(s)))
             .reset_index())
    return ded

# ============================================================================
# 3. Evaluate one clause against one row
# ============================================================================
def eval_clause(clause, row):
    if clause.startswith("asset=="):
        return row.asset == clause.split("==", 1)[1]
    if clause.startswith("side=="):
        return row.side == clause.split("==", 1)[1]
    if clause.startswith("cell=="):
        return clause.split("==", 1)[1] in row.cells
    if clause.startswith("stream=="):
        return clause.split("==", 1)[1] in row.streams
    if clause.startswith("prob>="):
        return pd.notna(row.prob) and row.prob >= float(clause.split(">=", 1)[1])
    if clause.startswith("prob<="):
        return pd.notna(row.prob) and row.prob <= float(clause.split("<=", 1)[1])
    raise ValueError(f"unrecognized clause: {clause}")

def rule_matches(clauses, row):
    return all(eval_clause(c, row) for c in clauses)

def tier_mask(ded, rules):
    mask = np.zeros(len(ded), dtype=bool)
    matched_rule = [None] * len(ded)
    for i, row in ded.iterrows():
        if mask[i]:
            continue
        for clauses in rules:
            if rule_matches(clauses, row):
                mask[i] = True
                matched_rule[i] = " & ".join(clauses)
                break
    return mask, matched_rule

# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", action="append",
                     default=["port_logs", "backtest_data/new_streams"])
    ap.add_argument("--rules-file", default="FINAL_RULES_REPORT.txt")
    ap.add_argument("--pending-only", action="store_true")
    args = ap.parse_args()

    print(f"Loading signals from: {args.signals_dir}")
    ded = load_current_signals(args.signals_dir)
    print(f"  {len(ded)} deduped current signals (asset+ts+side)")

    print(f"Parsing rules from: {args.rules_file}")
    tiers = parse_rules_file(args.rules_file)
    for t, rules in tiers.items():
        print(f"  {t}: {len(rules)} rules")

    if args.pending_only:
        ded = ded[ded.pending].reset_index(drop=True)
        print(f"\n--pending-only: {len(ded)} signals still pending (not yet resolved)")

    for tier_name, rules in tiers.items():
        mask, matched = tier_mask(ded, rules)
        sub = ded[mask].copy()
        sub["matched_rule"] = [m for m, k in zip(matched, mask) if k]
        sub = sub.sort_values("ts")
        fname = f"live_matches_{tier_name.replace('%','pct').replace('+','plus')}.csv"
        out = sub[["symbol_raw", "ts", "side", "prob", "pending", "matched_rule"]].rename(
            columns={"symbol_raw": "symbol"})
        out.to_csv(fname, index=False)
        print(f"\n[{tier_name}] {len(sub)} current signals match -> saved {fname}")
        if len(sub):
            print(out.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
