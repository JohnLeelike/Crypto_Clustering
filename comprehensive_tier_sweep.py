#!/usr/bin/env python3
"""
comprehensive_tier_sweep.py
=============================
Self-contained: runs only on the raw signal CSVs + res_cache_5m_all.parquet
+ book_5m.parquet.

Builds TWO independent variants, each a 3-source greedy merge, then greedily
merges THOSE two variants together:

  CLASSIC variant  - each source uses its ORIGINAL algorithm/file-set exactly
                      as first built (reproduces 73 / 131 / 518):
        mine   : full exhaustive combo search, min_n=10, dedup by
                 (asset,ts,side). Files = port_logs + new_streams AS THEY
                 EXISTED when mine was first built (no prec.csv).
        theirs : pruned top-40-pairs search (singles n>=3 WR>=55%, pairs
                 n>=3 WR>=60%, triples from top-40 pairs n>=3 WR>=60%),
                 dedup by (asset,ts) only. Files = new_streams only
                 (WITH prec.csv - it existed by the time theirs was built).
        run3   : same pruned search as theirs, over ALL 19 files
                 (WITH prec.csv).

  EXHAUSTIVE variant - all three sources use the SAME full exhaustive combo
                      search (mine's style, no top-40 pruning) over the SAME
                      consistent 19-file set (prec.csv included everywhere).
                      This is the "arguably better methodology" version that
                      won 2 of 3 tiers against CLASSIC.

Each variant's 3 sources are greedily merged (base = highest verified WR,
then add winner-only events from the other two). Then CLASSIC-final and
EXHAUSTIVE-final are THEMSELVES greedily merged per tier, to see whether
combining the two methodologies improves further.

All qualifying rules for every source/tier are written to rules_out/.

Every merge step uses realized win/loss to decide inclusion (lookahead) -
this measures the combined ceiling of what the sweeps found, not a
tradeable rule. The per-source rule files ARE what's tradeable; the merges
are a retrospective analysis on top.
"""
import glob, os, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FEE = 0.07
SYM = {"BTC":"btc","ETH":"eth","SOL":"sol","XRP":"xrp","DOGE":"doge","BNB":"bnb","HYPE":"hype"}

PORT_LOGS_FILES = sorted(glob.glob("port_logs/*.csv"))
NEW_STREAM_FILES_NO_DUP = sorted(f for f in glob.glob("backtest_data/new_streams/*.csv") if "gL_2" not in f)
ALL_19_FILES = PORT_LOGS_FILES + NEW_STREAM_FILES_NO_DUP

# mine's original build predates prec.csv but included the gL_2 duplicate
MINE_CLASSIC_FILES = PORT_LOGS_FILES + sorted(
    f for f in glob.glob("backtest_data/new_streams/*.csv") if "prec.csv" not in f)
THEIRS_CLASSIC_FILES = NEW_STREAM_FILES_NO_DUP
RUN3_CLASSIC_FILES = ALL_19_FILES

R = pd.read_parquet("res_cache_5m_all.parquet")
BK = pd.read_parquet("book_5m.parquet")
BK["ts"] = pd.to_datetime(BK.mkt_open, utc=True)
BK["asset"] = BK.asset.astype(str)
A5 = BK[BK.elapsed_req == 5][["asset", "ts", "au", "ad"]].rename(columns={"ad": "ask_up", "au": "ask_dn"})

# ============================================================================
def load_signal_rows(files):
    rows = []
    for f in files:
        d = pd.read_csv(f, on_bad_lines="skip")
        stream = os.path.basename(f).replace(".csv", "")
        strat_col = "strategy" if "strategy" in d.columns else "_15"
        d["stream"] = stream
        d["cell"] = d[strat_col].astype(str)
        d["asset"] = d.symbol.str.split("/").str[0].str.upper().map(SYM)
        d["ts"] = pd.to_datetime(d.start_date, format="%m/%d/%y %H:%M", errors="coerce", utc=True)
        d["prob"] = pd.to_numeric(d.get("prob"), errors="coerce")
        d["side"] = np.where(d.forecast.astype(str).str.contains("⬆"), "UP",
                    np.where(d.forecast.astype(str).str.contains("⬇"), "DOWN", None))
        rows.append(d[["stream", "cell", "asset", "ts", "side", "prob"]])
    S = pd.concat(rows, ignore_index=True).dropna(subset=["ts", "side", "asset"])
    S = S.merge(R, on=["asset", "ts"], how="left").merge(A5, on=["asset", "ts"], how="left")
    S["win"] = np.where(S.side == "UP", S.y, 1 - S.y)
    S["ask"] = np.where(S.side == "UP", S.ask_up, S.ask_dn)
    S = S.dropna(subset=["y", "ask"]).copy()
    S["pps"] = np.where(S.win == 1, 1 - S.ask, -S.ask) - FEE * S.ask * (1 - S.ask)
    S["pnl5"] = S.pps * (5.0 / S.ask)
    return S

def dedup_with_side(S):
    ded = (S.groupby(["asset", "ts", "side"])
             .agg(prob=("prob", "first"), win=("win", "first"), pnl5=("pnl5", "first"),
                  cells=("cell", lambda s: frozenset(s)), streams=("stream", lambda s: frozenset(s)))
             .reset_index())
    ded["key"] = list(zip(ded.asset, ded.ts.astype(str), np.where(ded.side == "UP", 1, 0)))
    return ded

def dedup_no_side(S):
    ded = (S.groupby(["asset", "ts"])
             .agg(prob=("prob", "first"), win=("win", "first"), pnl5=("pnl5", "first"), side=("side", "first"),
                  cells=("cell", lambda s: frozenset(s)), streams=("stream", lambda s: frozenset(s)))
             .reset_index())
    ded["key"] = list(zip(ded.asset, ded.ts.astype(str), np.where(ded.side == "UP", 1, 0)))
    return ded

def build_candidates(ded, all_cells, all_streams):
    filt = {}
    for a in ded.asset.dropna().unique(): filt[f"asset=={a}"] = (ded.asset == a).values
    for s in ["UP", "DOWN"]: filt[f"side=={s}"] = (ded.side == s).values
    for c in all_cells: filt[f"cell=={c}"] = ded.cells.apply(lambda fs: c in fs).values
    for st in all_streams: filt[f"stream=={st}"] = ded.streams.apply(lambda fs: st in fs).values
    probs = ded.prob.dropna()
    if len(probs):
        qs = sorted(set(round(x, 3) for x in np.quantile(probs, np.linspace(0.1, 0.9, 9))))
        for q in qs:
            filt[f"prob>={q}"] = (ded.prob >= q).values
            filt[f"prob<={q}"] = (ded.prob <= q).values
    return filt

def stats(ded, mask):
    n = mask.sum()
    if n == 0: return {"n":0,"w":0,"wr":0.0,"pnl":0.0}
    win = ded.win.values[mask]; pnl = ded.pnl5.values[mask]
    w = int(win.sum())
    return {"n": int(n), "w": w, "wr": w/n*100, "pnl": float(pnl.sum())}

# ---- MINE-style algorithm: full exhaustive, flat min_n, no WR pre-filter ----
def sweep_full_exhaustive(ded, filt, min_n):
    labels = list(filt.keys())
    results = []
    def record(labs, mask):
        s = stats(ded, mask)
        if s["n"] < min_n: return
        results.append((labs, s, mask.copy()))
    for lab in labels: record((lab,), filt[lab])
    for l1, l2 in itertools.combinations(labels, 2):
        m = filt[l1] & filt[l2]
        if m.sum() >= min_n: record((l1, l2), m)
    for l1, l2, l3 in itertools.combinations(labels, 3):
        m = filt[l1] & filt[l2] & filt[l3]
        if m.sum() >= min_n: record((l1, l2, l3), m)
    seen = {}
    for labs, s, mask in results:
        key = tuple(np.where(mask)[0])
        if key not in seen or len(labs) < len(seen[key][0]):
            seen[key] = (labs, s, mask)
    return list(seen.values())

# ---- THEIRS/RUN3-style algorithm: pruned top-40-pairs, graduated WR gates ----
def sweep_pruned_top40(ded, filt, min_n=3):
    labels = list(filt.keys())
    results = []
    for name in labels:
        s = stats(ded, filt[name])
        if s["n"] >= min_n and s["wr"] >= 55.0:
            results.append(([name], s, filt[name].copy()))
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            m = filt[labels[i]] & filt[labels[j]]
            s = stats(ded, m)
            if s["n"] >= min_n and s["wr"] >= 60.0:
                results.append(([labels[i], labels[j]], s, m.copy()))
    results.sort(key=lambda x: (-x[1]["wr"], -x[1]["n"]))
    top40_pairs = [r for r in results if len(r[0]) == 2][:40]
    top40_names = list(set(n for r in top40_pairs for n in r[0]))
    for combo in itertools.combinations(top40_names, 3):
        m = np.ones(len(ded), dtype=bool)
        for n in combo: m &= filt[n]
        s = stats(ded, m)
        if s["n"] >= min_n and s["wr"] >= 60.0:
            results.append((list(combo), s, m.copy()))
    seen = {}
    for labs, s, mask in results:
        key = tuple(sorted(labs))
        if key not in seen:
            seen[key] = (labs, s, mask)
    return list(seen.values())

def tiers_from_uniq(uniq, ded):
    """Cumulative tiers: 90%+ includes every WR>=90 rule (100% rules too),
    75%+ includes every WR>=75 rule (90%+ and 100% rules too). NOT disjoint
    bands -- a "+" tier must be a superset of every stricter tier above it."""
    out = {}
    for name, lo in [("100%+", 100), ("90%+", 90), ("75%+", 75)]:
        qual = [x for x in uniq if x[1]["wr"] >= lo]
        qual.sort(key=lambda x: (-x[1]["wr"], -x[1]["n"]))
        um = np.zeros(len(ded), dtype=bool)
        for *_, m in qual: um |= m
        keyset = set(ded.loc[um, "key"])
        out[name] = {"rules": qual, "keyset": keyset}
    return out

def run_source(name, files, dedup_fn, sweep_fn, min_n):
    S = load_signal_rows(files)
    ded = dedup_fn(S)
    all_cells = sorted(set().union(*S.cell.apply(lambda x: {x})))
    all_streams = sorted(S.stream.unique())
    filt = build_candidates(ded, all_cells, all_streams)
    uniq = sweep_fn(ded, filt, min_n) if sweep_fn is sweep_full_exhaustive else sweep_fn(ded, filt)
    tiers = tiers_from_uniq(uniq, ded)
    print(f"[{name}] {len(files)} files, {len(ded)} deduped events, WR={ded.win.mean()*100:.2f}%, pnl={ded.pnl5.sum():+.2f}")
    for t, d in tiers.items():
        print(f"       {t}: {len(d['rules'])} rules -> union n={len(d['keyset'])}")
    return ded, tiers

def write_rules(variant, src_name, tiers):
    os.makedirs("rules_out", exist_ok=True)
    for tier_name, d in tiers.items():
        fname = f"rules_out/{variant}_{tier_name.replace('%','pct').replace('+','plus')}_{src_name}.txt"
        with open(fname, "w") as f:
            f.write(f"{variant.upper()} / {src_name.upper()} -- {tier_name} WR tier -- {len(d['rules'])} qualifying rules\n")
            f.write("=" * 100 + "\n")
            for labs, s, _ in d["rules"]:
                f.write(f"  {' & '.join(labs):<70}  n={s['n']:<4} w={s['w']:<4} WR={s['wr']:6.2f}%  pnl@5={s['pnl']:+8.2f}\n")
    return

def greedy_merge(named_sets, ded_lookup):
    """named_sets: list of (name, keyset). Base = highest verified WR."""
    def sc(ks):
        found = [k for k in ks if k in ded_lookup]
        n = len(found); w = sum(ded_lookup[k][0] for k in found); p = sum(ded_lookup[k][1] for k in found)
        return n, w, (w/n*100 if n else 0.0), p
    ranked = sorted(named_sets, key=lambda kv: -sc(kv[1])[2])
    base_name, base_set = ranked[0]
    n0, w0, wr0, p0 = sc(base_set)
    merged = set(base_set)
    for name, s in ranked[1:]:
        extra = s - merged
        extra_wins = {k for k in extra if ded_lookup.get(k, (False, 0))[0]}
        merged |= extra_wins
    n, w, wr, p = sc(merged)
    return base_name, merged, (n0, wr0, p0), (n, wr, p)

print("="*100); print("CLASSIC VARIANT (each source's original algorithm/file-set)"); print("="*100)
mine_c_ded, mine_c_tiers = run_source("mine", MINE_CLASSIC_FILES, dedup_with_side, sweep_full_exhaustive, 10)
theirs_c_ded, theirs_c_tiers = run_source("theirs", THEIRS_CLASSIC_FILES, dedup_no_side, sweep_pruned_top40, 3)
run3_c_ded, run3_c_tiers = run_source("run3", RUN3_CLASSIC_FILES, dedup_no_side, sweep_pruned_top40, 3)
for d in [mine_c_tiers, theirs_c_tiers, run3_c_tiers]:
    pass
write_rules("classic", "mine", mine_c_tiers)
write_rules("classic", "theirs", theirs_c_tiers)
write_rules("classic", "run3", run3_c_tiers)

ded_lookup = {row.key: (bool(row.win), float(row.pnl5)) for _, row in mine_c_ded.iterrows()}

print(f"\n{'tier':<8}{'base':<8}{'N':>6}{'WR':>8}{'pnl':>10}")
classic_final = {}
for tier in ["100%+", "90%+", "75%+"]:
    named = [("mine", mine_c_tiers[tier]["keyset"]), ("theirs", theirs_c_tiers[tier]["keyset"]), ("run3", run3_c_tiers[tier]["keyset"])]
    base_name, merged, (n0,wr0,p0), (n,wr,p) = greedy_merge(named, ded_lookup)
    classic_final[tier] = merged
    print(f"{tier:<8}{base_name:<8}{n:>6}{wr:>7.2f}%{p:>+10.2f}   (base alone: n={n0} wr={wr0:.2f}% pnl={p0:+.2f})")

print("\n"+"="*100); print("EXHAUSTIVE VARIANT (all 3 sources use mine's full exhaustive search, same file set)"); print("="*100)
mine_e_ded, mine_e_tiers = run_source("mine", ALL_19_FILES, dedup_with_side, sweep_full_exhaustive, 10)
theirs_e_ded, theirs_e_tiers = run_source("theirs", NEW_STREAM_FILES_NO_DUP, dedup_no_side, sweep_full_exhaustive, 3)
run3_e_ded, run3_e_tiers = run_source("run3", ALL_19_FILES, dedup_no_side, sweep_full_exhaustive, 3)
write_rules("exhaustive", "mine", mine_e_tiers)
write_rules("exhaustive", "theirs", theirs_e_tiers)
write_rules("exhaustive", "run3", run3_e_tiers)

ded_lookup_e = {row.key: (bool(row.win), float(row.pnl5)) for _, row in mine_e_ded.iterrows()}

print(f"\n{'tier':<8}{'base':<8}{'N':>6}{'WR':>8}{'pnl':>10}")
exhaustive_final = {}
for tier in ["100%+", "90%+", "75%+"]:
    named = [("mine", mine_e_tiers[tier]["keyset"]), ("theirs", theirs_e_tiers[tier]["keyset"]), ("run3", run3_e_tiers[tier]["keyset"])]
    base_name, merged, (n0,wr0,p0), (n,wr,p) = greedy_merge(named, ded_lookup_e)
    exhaustive_final[tier] = merged
    print(f"{tier:<8}{base_name:<8}{n:>6}{wr:>7.2f}%{p:>+10.2f}   (base alone: n={n0} wr={wr0:.2f}% pnl={p0:+.2f})")

print("\n"+"="*100); print("FINAL GREEDY MERGE: CLASSIC vs EXHAUSTIVE"); print("="*100)
# score against the union of both lookups (exhaustive's lookup is a superset; prefer it, fallback to classic's)
combined_lookup = dict(ded_lookup_e)
for k,v in ded_lookup.items():
    combined_lookup.setdefault(k, v)

print(f"{'tier':<8}{'base':<12}{'N':>6}{'WR':>8}{'pnl':>10}   delta")
for tier in ["100%+", "90%+", "75%+"]:
    named = [("classic", classic_final[tier]), ("exhaustive", exhaustive_final[tier])]
    base_name, merged, (n0,wr0,p0), (n,wr,p) = greedy_merge(named, combined_lookup)
    print(f"{tier:<8}{base_name:<12}{n:>6}{wr:>7.2f}%{p:>+10.2f}   +{n-n0} events, {wr-wr0:+.2f}pp, ${p-p0:+.2f}")
    # save final event list
    found = [k for k in merged if k in combined_lookup]
    rows = [{"asset":k[0],"ts":k[1],"side":"UP" if k[2] else "DOWN",
             "win":combined_lookup[k][0], "pnl5":round(combined_lookup[k][1],2)} for k in found]
    pd.DataFrame(rows).sort_values("ts").to_csv(
        f"final_tier_{tier.replace('%','pct').replace('+','plus')}_v2.csv", index=False)
