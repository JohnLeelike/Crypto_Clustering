#!/usr/bin/env python3
"""
signal_only_tier_sweep.py
==========================
Self-contained: runs ONLY on the raw signal CSVs plus the two scoring
inputs (res_cache_5m_all.parquet, book_5m.parquet) that give the real
outcome/ask price for every signal. No cached intermediate files required.

Rebuilds THREE independently-defined event sets from the same raw signal
CSVs (they differ in which files they use and how they deduplicate):

  MINE   : all 19 signal CSVs, deduped by (asset, ts, side) -- a UP and a
           DOWN call on the same bar are kept as two separate events.
  THEIRS : only the 10 "new" stream CSVs, deduped by (asset, ts) only --
           if two files disagree on direction for the same bar, whichever
           row pandas' groupby sees first wins.
  RUN3   : all 19 signal CSVs, deduped by (asset, ts) only (same collapsing
           rule as THEIRS, but over the full file set).

For each source, runs an exhaustive filter sweep (every single/pair/triple
combination of asset / side / cell / stream / prob-threshold, minimum
n>=10 for MINE, n>=3 for THEIRS/RUN3 to match how each was originally
built) and unions every combo whose win rate clears a tier (100% / 90% /
75%). ALL qualifying rules are written out per tier (not just the top few)
to <tier>_<source>_rules.txt.

Final tier = greedy merge across the three sources: whichever source has
the highest WR for that tier becomes the base, then events from the other
two sources are added ONLY if they are winners (scored against a single
canonical lookup built from MINE's own asset+ts+side join). This is a
retrospective "combined ceiling" construction, not a tradeable rule --
the per-source rule lists ARE the tradeable part; the cross-source merge
is not.

USAGE
-----
  python3 signal_only_tier_sweep.py
"""
import glob, os, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FEE = 0.07
SYM = {"BTC":"btc","ETH":"eth","SOL":"sol","XRP":"xrp","DOGE":"doge","BNB":"bnb","HYPE":"hype"}

PORT_LOGS_FILES = sorted(glob.glob("port_logs/*.csv"))
NEW_STREAM_FILES = sorted(f for f in glob.glob("backtest_data/new_streams/*.csv") if "gL_2" not in f)
ALL_19_FILES = PORT_LOGS_FILES + NEW_STREAM_FILES

# NOTE: "mine" was historically built BEFORE prec.csv was uploaded, so its
# file list excludes it to exactly match the original 73/131/518 result.
# theirs/run3 were both built AFTER prec.csv existed, so they keep it.
MINE_FILES = PORT_LOGS_FILES + [f for f in NEW_STREAM_FILES if "prec.csv" not in f]

R = pd.read_parquet("res_cache_5m_all.parquet")
BK = pd.read_parquet("book_5m.parquet")
BK["ts"] = pd.to_datetime(BK.mkt_open, utc=True)
BK["asset"] = BK.asset.astype(str)
A5 = BK[BK.elapsed_req == 5][["asset", "ts", "au", "ad"]].rename(columns={"ad": "ask_up", "au": "ask_dn"})

# ============================================================================
# Shared signal loader
# ============================================================================
def load_signal_rows(files):
    """Read a list of Glass-format signal CSVs into one normalized frame."""
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

# ============================================================================
# Exhaustive filter sweep: builds every (asset/side/cell/stream/prob) combo
# up to 3 clauses, keeps everything that clears a WR tier.
# ============================================================================
def build_candidates(ded, all_cells, all_streams):
    filt = {}
    for a in ded.asset.dropna().unique():
        filt[f"asset=={a}"] = (ded.asset == a).values
    for s in ["UP", "DOWN"]:
        filt[f"side=={s}"] = (ded.side == s).values
    for c in all_cells:
        filt[f"cell=={c}"] = ded.cells.apply(lambda fs: c in fs).values
    for st in all_streams:
        filt[f"stream=={st}"] = ded.streams.apply(lambda fs: st in fs).values
    probs = ded.prob.dropna()
    if len(probs):
        qs = sorted(set(round(x, 3) for x in np.quantile(probs, np.linspace(0.1, 0.9, 9))))
        for q in qs:
            filt[f"prob>={q}"] = (ded.prob >= q).values
            filt[f"prob<={q}"] = (ded.prob <= q).values
    return filt

def exhaustive_sweep(ded, filt, min_n):
    labels = list(filt.keys())
    win = ded.win.values.astype(bool)
    pnl = ded.pnl5.values
    results = []
    def record(labs, mask):
        n = mask.sum()
        if n < min_n:
            return
        results.append((labs, n, int(win[mask].sum()), win[mask].sum() / n * 100, pnl[mask].sum(), mask.copy()))
    for lab in labels:
        record((lab,), filt[lab])
    for l1, l2 in itertools.combinations(labels, 2):
        m = filt[l1] & filt[l2]
        if m.sum() >= min_n:
            record((l1, l2), m)
    for l1, l2, l3 in itertools.combinations(labels, 3):
        m = filt[l1] & filt[l2] & filt[l3]
        if m.sum() >= min_n:
            record((l1, l2, l3), m)

    seen = {}
    for labs, n, w, wr, p, mask in results:
        key = tuple(np.where(mask)[0])
        if key not in seen or len(labs) < len(seen[key][0]):
            seen[key] = (labs, n, w, wr, p, mask)
    return list(seen.values())

def tiers_from_uniq(uniq, ded):
    out = {}
    for name, lo, hi in [("100%+", 100, 100), ("90%+", 90, 99.999), ("75%+", 75, 89.999)]:
        qual = [x for x in uniq if lo <= x[3] <= hi]
        qual.sort(key=lambda x: (-x[3], -x[1]))
        um = np.zeros(len(ded), dtype=bool)
        for *_, m in qual:
            um |= m
        keyset = set(ded.loc[um, "key"])
        out[name] = {"rules": qual, "keyset": keyset}
    return out

def run_source(name, files, dedup_fn, min_n):
    print(f"\n[{name}] loading {len(files)} files, dedup={'asset+ts+side' if dedup_fn is dedup_with_side else 'asset+ts only'}, min_n={min_n}")
    S = load_signal_rows(files)
    ded = dedup_fn(S)
    print(f"[{name}] {len(ded)} deduped events  (overall WR={ded.win.mean()*100:.2f}%  pnl@5={ded.pnl5.sum():+.2f})")
    all_cells = sorted(set().union(*S.cell.apply(lambda x: {x})))
    all_streams = sorted(S.stream.unique())
    filt = build_candidates(ded, all_cells, all_streams)
    uniq = exhaustive_sweep(ded, filt, min_n)
    tiers = tiers_from_uniq(uniq, ded)
    for t, d in tiers.items():
        print(f"[{name}] {t}: {len(d['rules'])} qualifying rules -> union n={len(d['keyset'])}")
    return ded, tiers

# ============================================================================
# Run all three sources
# ============================================================================
mine_ded, mine_tiers = run_source("MINE", MINE_FILES, dedup_with_side, min_n=10)
theirs_ded, theirs_tiers = run_source("THEIRS", NEW_STREAM_FILES, dedup_no_side, min_n=3)
run3_ded, run3_tiers = run_source("RUN3", ALL_19_FILES, dedup_no_side, min_n=3)

# ============================================================================
# Canonical lookup for scoring (built from MINE's own join; see earlier
# 77-vs-73 discrepancy -- do not patch this with a second pipeline's join)
# ============================================================================
ded_lookup = {row.key: (bool(row.win), float(row.pnl5)) for _, row in mine_ded.iterrows()}

def stats_of(keyset):
    found = [k for k in keyset if k in ded_lookup]
    n = len(found)
    w = sum(ded_lookup[k][0] for k in found)
    p = sum(ded_lookup[k][1] for k in found)
    return n, w, (w / n * 100 if n else 0.0), p

SOURCES = {"mine": mine_tiers, "theirs": theirs_tiers, "run3": run3_tiers}

# ============================================================================
# Write out every qualifying rule per source per tier
# ============================================================================
os.makedirs("rules_out", exist_ok=True)
for src_name, tiers in SOURCES.items():
    for tier_name, d in tiers.items():
        fname = f"rules_out/{tier_name.replace('%','pct').replace('+','plus')}_{src_name}_rules.txt"
        with open(fname, "w") as f:
            f.write(f"{src_name.upper()} -- {tier_name} WR tier -- {len(d['rules'])} qualifying rules\n")
            f.write("=" * 100 + "\n")
            for labs, n, w, wr, p, _ in sorted(d["rules"], key=lambda x: (-x[3], -x[1])):
                f.write(f"  {' & '.join(labs):<70}  n={n:<4} w={w:<4} WR={wr:6.2f}%  pnl@5={p:+8.2f}\n")
        print(f"wrote {fname} ({len(d['rules'])} rules)")

# ============================================================================
# Final greedy merge per tier
# ============================================================================
print(f"\n{'='*90}\nFINAL MERGED TIERS\n{'='*90}")
print(f"{'tier':<8}{'base':<8}{'final N':>10}{'WR':>9}{'pnl@5':>12}   delta vs base")
print("-" * 70)
for tier in ["100%+", "90%+", "75%+"]:
    cands = {name: SOURCES[name][tier]["keyset"] for name in SOURCES}
    ranked = sorted(cands.items(), key=lambda kv: -stats_of(kv[1])[2])
    base_name, base_set = ranked[0]
    n0, w0, wr0, p0 = stats_of(base_set)
    merged = set(base_set)
    for name, s in ranked[1:]:
        extra = s - merged
        extra_wins = {k for k in extra if ded_lookup.get(k, (False, 0))[0]}
        merged |= extra_wins
    n, w, wr, p = stats_of(merged)
    print(f"{tier:<8}{base_name:<8}{n:>10}{wr:>8.2f}%{p:>+12.2f}   "
          f"+{n-n0} events, {wr-wr0:+.2f}pp, ${p-p0:+.2f}")
    print(f"         base rules file: rules_out/{tier.replace('%','pct').replace('+','plus')}_{base_name}_rules.txt")
