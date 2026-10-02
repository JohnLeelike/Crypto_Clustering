#!/usr/bin/env python3
"""
ensemble_search_gamma.py
========================
Post-hoc exhaustive ensemble search over saved gamma walk-forward signals.
Loads outputs_gamma_walkforward/gamma_full_signals_long.csv, tries combos of
size 2..5 from a top pool with modes: union, intersection, vote_K (K=2..k).

Reports:
  - Max WR (with min resolved threshold)
  - Max PnL
  - Per-size best (2/3/4/5)

Goal: push gamma WR past 60%.

USAGE
-----
  python3 ensemble_search_gamma.py --pool-size 10 --max-k 5 --min-resolved 30
"""
from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd


HERE = Path(__file__).parent.resolve()
SIG_PATH = HERE / "outputs_gamma_walkforward" / "gamma_full_signals_long.csv"
PER_METHOD_PATH = HERE / "outputs_gamma_walkforward" / "gamma_full_per_method.csv"


def load_signals():
    long_df = pd.read_csv(SIG_PATH, parse_dates=["timestamp"])
    per_m = pd.read_csv(PER_METHOD_PATH)
    return long_df, per_m


def simulate_window_pnl(sigs_in_window, start_bank=100.0, bet_pct=0.10, bet_cap=100.0):
    """sigs_in_window sorted by timestamp; each row has correct_gamma in {0, 1, NaN}."""
    bank = start_bank
    for _, r in sigs_in_window.iterrows():
        if pd.isna(r["correct_gamma"]):
            continue
        stake = min(bank * bet_pct, bet_cap)
        if r["correct_gamma"] == 1:
            bank += stake
        else:
            bank -= stake
        if bank <= 0:
            break
    return bank


def build_vote_matrix(long_df, methods):
    """Pivot long_df into a (event × method) matrix of directions in {-1, 0, +1}.
    Cell is 0 if that method did not fire on that event.
    Returns (events_df with [window, timestamp, gamma_y], votes ndarray of shape (N_events, N_methods)).
    """
    sub = long_df[long_df["method"].isin(methods)][["window", "timestamp", "method", "direction", "gamma_y"]]
    piv = sub.pivot_table(index=["window", "timestamp"], columns="method",
                          values="direction", aggfunc="first")
    # Ensure column order matches methods list; missing methods → all zeros
    for m in methods:
        if m not in piv.columns:
            piv[m] = 0
    piv = piv[list(methods)].fillna(0).astype(np.int8)
    # gamma_y at each event: take first (it's constant per event)
    gamma = sub.groupby(["window", "timestamp"], sort=False)["gamma_y"].first().reindex(piv.index)
    events = piv.reset_index()[["window", "timestamp"]].copy()
    events["gamma_y"] = gamma.values
    return events, piv.values


def combine_vectorized(events, votes, mode, k=None, n_methods=None):
    """Vectorized combo evaluation. Returns fired DataFrame with [window, timestamp, direction, correct_gamma]."""
    up = (votes == 1).sum(axis=1)
    dn = (votes == -1).sum(axis=1)
    methods_fired = (votes != 0).sum(axis=1)
    if n_methods is None:
        n_methods = votes.shape[1]

    if mode == "union":
        mask = up != dn
        direction = np.where(up > dn, 1, -1)
    elif mode == "intersection":
        mask = (methods_fired == n_methods) & ((up == 0) | (dn == 0)) & (methods_fired > 0)
        direction = np.where(up > 0, 1, -1)
    elif mode.startswith("vote_"):
        kk = k if k is not None else int(mode.split("_")[1])
        up_win = (up >= kk) & (up > dn)
        dn_win = (dn >= kk) & (dn > up)
        mask = up_win | dn_win
        direction = np.where(up_win, 1, np.where(dn_win, -1, 0))
    elif mode == "majority":
        need = n_methods // 2 + 1
        up_win = up >= need
        dn_win = dn >= need
        mask = up_win | dn_win
        direction = np.where(up_win, 1, np.where(dn_win, -1, 0))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not mask.any():
        return pd.DataFrame(columns=["window", "timestamp", "direction", "correct_gamma"])
    idx = np.where(mask)[0]
    ev = events.iloc[idx].copy()
    ev["direction"] = direction[idx].astype(np.int8)
    gamma_y = ev["gamma_y"].values
    correct = np.where(
        pd.isna(gamma_y), np.nan,
        ((ev["direction"] == 1) & (gamma_y == 1)) |
        ((ev["direction"] == -1) & (gamma_y == 0))
    )
    ev["correct_gamma"] = correct.astype(float)
    return ev[["window", "timestamp", "direction", "correct_gamma"]]


def combine(long_df, methods, mode, k=None):
    events, votes = build_vote_matrix(long_df, methods)
    return combine_vectorized(events, votes, mode, k=k, n_methods=len(methods))


def evaluate(fired_df, bet_pct=0.10):
    """Total PnL delta (fresh $100 per window) + WR + n_resolved.

    PnL simplification: bet = bank * bet_pct, cap $100 per bet only binds when
    bank > $1000 (bet > $100). For our WR / signal counts this doesn't happen,
    so bank_final = 100 * (1+bet_pct)^wins * (1-bet_pct)^losses. Order-invariant."""
    if len(fired_df) == 0:
        return {"n_signals": 0, "n_resolved": 0, "wr": None, "total_pnl": 0.0}
    resolved = fired_df.dropna(subset=["correct_gamma"])
    total_resolved = len(resolved)
    if total_resolved == 0:
        return {"n_signals": len(fired_df), "n_resolved": 0, "wr": None, "total_pnl": 0.0}
    total_correct = int(resolved["correct_gamma"].sum())
    wr = total_correct / total_resolved
    total_pnl = 0.0
    grp = resolved.groupby("window", sort=False)["correct_gamma"].agg(["sum", "count"])
    for wins, n in zip(grp["sum"].astype(int), grp["count"].astype(int)):
        losses = n - wins
        bank = 100.0 * (1 + bet_pct) ** wins * (1 - bet_pct) ** losses
        total_pnl += (bank - 100.0)
    return {
        "n_signals": len(fired_df),
        "n_resolved": total_resolved,
        "wr": wr,
        "total_pnl": round(total_pnl, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=12, help="Top N methods by WR to include in search pool")
    ap.add_argument("--max-k", type=int, default=5, help="Max ensemble size")
    ap.add_argument("--min-resolved", type=int, default=30, help="Min resolved gamma bars to consider combo")
    ap.add_argument("--out-csv", type=str, default="ensemble_search_results.csv")
    args = ap.parse_args()

    long_df, per_m = load_signals()
    print(f"Loaded {len(long_df)} signals across {long_df['method'].nunique()} methods, "
          f"{long_df['window'].nunique()} windows.")

    per_m_sorted = per_m.sort_values("wr_gamma", ascending=False)
    pool = per_m_sorted["method"].head(args.pool_size).tolist()
    print(f"\nSearch pool (top {args.pool_size} by gamma WR): {pool}")

    all_results = []

    # Build the master vote matrix ONCE over the full pool (fast column subsetting later)
    print("Building master vote matrix ...")
    events, big_votes = build_vote_matrix(long_df, pool)
    print(f"  events: {len(events)}, methods: {big_votes.shape[1]}")
    method_col = {m: i for i, m in enumerate(pool)}

    # Individual baselines
    for m in pool:
        col_idx = [method_col[m]]
        fired = combine_vectorized(events, big_votes[:, col_idx], "union", n_methods=1)
        r = evaluate(fired)
        r.update({"combo": m, "mode": "single", "k_size": 1})
        all_results.append(r)

    modes_2 = ["union", "intersection", "vote_2"]
    modes_3 = ["union", "intersection", "vote_2", "vote_3"]
    modes_4 = ["union", "intersection", "vote_2", "vote_3", "vote_4"]
    modes_5 = ["union", "intersection", "vote_2", "vote_3", "vote_4", "vote_5"]

    combos_by_size = {
        2: (list(combinations(pool, 2)), modes_2),
        3: (list(combinations(pool, 3)), modes_3),
        4: (list(combinations(pool, 4)), modes_4),
        5: (list(combinations(pool, 5)), modes_5),
    }

    import time
    for size in range(2, args.max_k + 1):
        combos, modes = combos_by_size[size]
        t0 = time.time()
        print(f"\nSize {size}: {len(combos)} combos × {len(modes)} modes = {len(combos) * len(modes)} evaluations")
        for combo in combos:
            col_idx = [method_col[m] for m in combo]
            sub_votes = big_votes[:, col_idx]
            for mode in modes:
                fired = combine_vectorized(events, sub_votes, mode, n_methods=len(combo))
                r = evaluate(fired)
                r.update({"combo": "+".join(combo), "mode": mode, "k_size": size})
                all_results.append(r)
        print(f"  size {size} done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_results)
    df["wr_pct"] = df["wr"] * 100
    df = df.sort_values("wr_pct", ascending=False)

    out_path = HERE / "outputs_gamma_walkforward" / args.out_csv
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} combo results to {out_path}")

    # Filter to combos meeting min resolved threshold
    valid = df[df["n_resolved"] >= args.min_resolved].copy()

    print(f"\n{'=' * 115}")
    print(f"TOP 20 BY WR  (min {args.min_resolved} resolved gamma bars)")
    print(f"{'=' * 115}")
    print(f"  {'combo':<80}  {'mode':<12}  {'sigs':>5}  {'gres':>5}  {'wr':>6}  {'PnL($)':>9}")
    for _, r in valid.sort_values("wr_pct", ascending=False).head(20).iterrows():
        mark = ""
        if r["wr"] >= 0.60: mark = " ★★"
        elif r["wr"] >= 0.58: mark = " ★"
        print(f"  {r['combo'][:78]:<80}  {r['mode']:<12}  {r['n_signals']:>5}  "
              f"{r['n_resolved']:>5}  {r['wr']*100:>5.2f}%  ${r['total_pnl']:>+8.2f}{mark}")

    print(f"\n{'=' * 115}")
    print(f"TOP 20 BY PnL  (min {args.min_resolved} resolved gamma bars)")
    print(f"{'=' * 115}")
    print(f"  {'combo':<80}  {'mode':<12}  {'sigs':>5}  {'gres':>5}  {'wr':>6}  {'PnL($)':>9}")
    for _, r in valid.sort_values("total_pnl", ascending=False).head(20).iterrows():
        print(f"  {r['combo'][:78]:<80}  {r['mode']:<12}  {r['n_signals']:>5}  "
              f"{r['n_resolved']:>5}  {r['wr']*100:>5.2f}%  ${r['total_pnl']:>+8.2f}")

    print(f"\n{'=' * 115}")
    print(f"BEST BY (size, mode) — highest WR with n_resolved ≥ {args.min_resolved}")
    print(f"{'=' * 115}")
    print(f"  {'k':>2}  {'mode':<14}  {'best combo':<80}  {'sigs':>5}  {'gres':>5}  {'wr':>6}  {'PnL($)':>9}")
    for (size, mode), g in valid.groupby(["k_size", "mode"], sort=False):
        best = g.sort_values("wr_pct", ascending=False).iloc[0]
        print(f"  {size:>2}  {mode:<14}  {best['combo'][:78]:<80}  {best['n_signals']:>5}  "
              f"{best['n_resolved']:>5}  {best['wr']*100:>5.2f}%  ${best['total_pnl']:>+8.2f}")

    # Highlight anything ≥60%
    over_60 = valid[valid["wr"] >= 0.60].sort_values("wr_pct", ascending=False)
    print(f"\n{'=' * 115}")
    print(f"COMBOS WITH WR ≥ 60% (n_resolved ≥ {args.min_resolved}): {len(over_60)} found")
    print(f"{'=' * 115}")
    if len(over_60) == 0:
        print("  None. Highest achieved:")
        best = valid.sort_values("wr_pct", ascending=False).iloc[0]
        print(f"  {best['combo']}  ({best['mode']})  WR={best['wr']*100:.2f}%  n_resolved={best['n_resolved']}  PnL=${best['total_pnl']:+.2f}")
    else:
        for _, r in over_60.head(30).iterrows():
            print(f"  {r['combo']:<80}  {r['mode']:<12}  sigs={r['n_signals']:>5}  "
                  f"gres={r['n_resolved']:>5}  WR={r['wr']*100:>5.2f}%  PnL=${r['total_pnl']:>+8.2f}")


if __name__ == "__main__":
    main()
