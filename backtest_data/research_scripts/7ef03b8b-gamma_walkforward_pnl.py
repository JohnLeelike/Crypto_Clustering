#!/usr/bin/env python3
"""
gamma_walkforward_pnl.py
========================
Walk-forward evaluation using Polymarket **gamma resolutions** as ground truth
(not 5m close-direction).

Trains on 5m-close-direction labels (since gamma is only 2.5 months) but
evaluates against BOTH:
  - 5m close-direction (old convention, reference for delta)
  - gamma resolution (actual Polymarket outcome — what a trader is paid on)

Per method per window:
  - Signals fired (bar_ts, direction, both labels, correct?)
  - WR under both label sources
  - PnL under gamma with Polymarket rules:
      * $100 fresh bankroll per window (all summed at the end)
      * 10% of current bank per trade, capped at $100
      * Entry price $0.50, redeem $1.00 on win → payout = stake
      * Loss = full stake

Then ensemble search:
  - Pairwise/triple UNION and INTERSECTION of top-K methods
  - Greedy max PnL and max precision selection

USAGE
-----
  python3 gamma_walkforward_pnl.py --smoke  # 2 windows, few methods
  python3 gamma_walkforward_pnl.py          # full sweep
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
spec = importlib.util.spec_from_file_location("wf", HERE / "cmaes_tsfresh_multi_tf.py")
wf = importlib.util.module_from_spec(spec)
sys.modules["wf"] = wf
spec.loader.exec_module(wf)

# Reuse method battery from boost_walkforward_5m_v2
spec2 = importlib.util.spec_from_file_location("boost", HERE / "boost_walkforward_5m_v2.py")
boost = importlib.util.module_from_spec(spec2)
sys.modules["boost"] = boost
spec2.loader.exec_module(boost)

BTC1M_PATH = HERE / "pm5m" / "btc1m_full.parquet"
GAMMA_PATH = HERE / "pm5m" / "gamma_btc.parquet"
OUTPUT_DIR = HERE / "outputs_gamma_walkforward"
CACHE_DIR = HERE / "cache_tsfresh_features"
BPD = 288  # 5m bars per day


# --------------------------------------------------------------------------------------
# Data loading & feature computation
# --------------------------------------------------------------------------------------

def load_5m_from_1m(path=BTC1M_PATH):
    """Load 1m parquet, resample to 5m OHLCV."""
    d = pd.read_parquet(path).reset_index()
    d["timestamp"] = pd.to_datetime(d["t"], unit="s", utc=True)
    d = d.set_index("timestamp").sort_index()
    d = d.rename(columns={"vol": "volume"})
    o5 = d[["open", "high", "low", "close", "volume"]].resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"]).reset_index()
    # Match the schema of prepare_tf_data() output
    o5["timestamp"] = o5["timestamp"].dt.tz_localize(None)
    return o5


def build_features(df_5m, window=100):
    """Compact + tsfresh features on the 5m frame (past-only rolling)."""
    cache_path = CACHE_DIR / f"gamma_5m_win{window}_{_content_hash(df_5m)}.parquet"
    if cache_path.exists():
        print(f"  loading cached features: {cache_path.name}", flush=True)
        return pd.read_parquet(cache_path)
    print(f"  computing tsfresh features (window={window})...", flush=True)
    t0 = time.time()
    compact = wf.compute_compact_features(df_5m)
    tsf = wf.compute_tsfresh_features(df_5m, window=window)
    feats = pd.concat([compact, tsf], axis=1)
    out = pd.concat(
        [df_5m[["timestamp", "close"]].reset_index(drop=True), feats.reset_index(drop=True)],
        axis=1,
    )
    out["log_return_next"] = np.log(out["close"].shift(-1) / out["close"])
    out["year"] = out["timestamp"].dt.year
    out = out.dropna().reset_index(drop=True)
    print(f"  features done in {time.time()-t0:.1f}s ({len(out)} rows, {out.shape[1]-4} feats)", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    return out


def _content_hash(df):
    ts_min = pd.to_datetime(df["timestamp"].min()).value
    ts_max = pd.to_datetime(df["timestamp"].max()).value
    key = f"{len(df)}|{ts_min}|{ts_max}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def load_gamma(path=GAMMA_PATH):
    g = pd.read_parquet(path).copy()
    # bucket is unix seconds; convert to naive UTC datetime to match 5m timestamps
    g["timestamp"] = pd.to_datetime(g["bucket"], unit="s", utc=True).dt.tz_localize(None)
    g = g[["timestamp", "y"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return g


def align_labels(df_feats, gamma, shift_min=-5):
    """Merge gamma with a configurable shift (minutes). Default -5 = correct alignment
    for our model (see comment in main / previous version). shift_min=0 recreates
    the mis-aligned (off-by-one-bar) version."""
    g_shifted = gamma.copy()
    g_shifted["timestamp"] = g_shifted["timestamp"] + pd.Timedelta(minutes=shift_min)
    m = df_feats.merge(g_shifted, on="timestamp", how="left")
    return m


# --------------------------------------------------------------------------------------
# PnL simulation
# --------------------------------------------------------------------------------------

def simulate_pnl(signals, start_bank=100.0, bet_pct=0.10, bet_cap=100.0):
    """signals: DataFrame with columns ['timestamp', 'direction', 'correct_gamma']
    where correct_gamma is 1/0/NaN. NaN rows (no gamma resolution) are skipped.
    Returns (final_bank, per_trade DataFrame, gross_wr_on_resolved)."""
    bank = start_bank
    rows = []
    resolved = 0
    correct = 0
    for _, r in signals.sort_values("timestamp").iterrows():
        if pd.isna(r["correct_gamma"]):
            continue  # no gamma resolution, no PnL
        resolved += 1
        stake = min(bank * bet_pct, bet_cap)
        if r["correct_gamma"] == 1:
            correct += 1
            bank += stake  # buy at 0.50, redeem 1.00 → payout equals stake
            rows.append({**r.to_dict(), "stake": stake, "outcome": "WIN", "bank_after": bank})
        else:
            bank -= stake
            rows.append({**r.to_dict(), "stake": stake, "outcome": "LOSS", "bank_after": bank})
        if bank <= 0:
            break
    wr = correct / resolved if resolved > 0 else None
    return bank, pd.DataFrame(rows), wr, resolved


# --------------------------------------------------------------------------------------
# One-window run: train each method, collect signals + labels
# --------------------------------------------------------------------------------------

def top_k_signals(conf_signed, sub_te, k_pct, mask):
    """Return dataframe of fired signals: timestamp, direction, |confidence|.
    Selection: top k_pct% by |conf| within mask."""
    idx_pool = np.where(mask)[0]
    if len(idx_pool) == 0:
        return pd.DataFrame(columns=["timestamp", "direction", "conf"])
    k = max(3, int(len(idx_pool) * k_pct / 100.0))
    k = min(k, len(idx_pool))
    abs_c = np.abs(conf_signed[idx_pool])
    top = np.argpartition(-abs_c, k - 1)[:k]
    final_idx = idx_pool[top]
    directions = np.sign(conf_signed[final_idx]).astype(int)
    m = directions != 0
    final_idx = final_idx[m]
    directions = directions[m]
    return pd.DataFrame({
        "timestamp": sub_te["timestamp"].values[final_idx],
        "direction": directions,
        "conf": np.abs(conf_signed[final_idx]),
    }).sort_values("timestamp").reset_index(drop=True)


def run_one_window(sub_tr, sub_te, args, method_list):
    """Train each method on sub_tr, score on sub_te, apply atr_low + top-K%,
    return dict method → signals DataFrame. Signals include:
      timestamp, direction (+1/-1), conf,
      close_dir (5m close-based label, +1/-1/0),
      gamma_y (+1/0/NaN), correct_close, correct_gamma."""
    delta = args.delta_bps * 1e-4
    feats_all = [c for c in sub_tr.columns
                 if c not in ("timestamp", "close", "log_return_next", "year", "y")]
    X_tr_full = sub_tr[feats_all].values.astype(np.float64)
    r_tr = sub_tr["log_return_next"].values.astype(np.float64)
    X_te_full = sub_te[feats_all].values.astype(np.float64)
    r_te = sub_te["log_return_next"].values.astype(np.float64)

    good_tr = np.isfinite(X_tr_full).all(axis=1) & np.isfinite(r_tr)
    good_te = np.isfinite(X_te_full).all(axis=1) & np.isfinite(r_te)
    X_tr_full, r_tr = X_tr_full[good_tr], r_tr[good_tr]
    X_te_full, r_te = X_te_full[good_te], r_te[good_te]
    sub_te = sub_te.iloc[good_te].reset_index(drop=True)
    sub_tr_good = sub_tr.iloc[good_tr].reset_index(drop=True)

    if len(X_tr_full) < 500 or len(X_te_full) < 20:
        return None

    sel_idx = wf.select_features_train(X_tr_full, r_tr, delta, args.top_k, feats_all)
    X_tr = X_tr_full[:, sel_idx]
    X_te = X_te_full[:, sel_idx]
    med, scale = wf.fit_scaler_train(X_tr)
    X_tr = wf.apply_scaler(X_tr, med, scale)
    X_te = wf.apply_scaler(X_te, med, scale)

    y_tr = boost.sig_labels(r_tr, delta)
    vs = int(len(X_tr) * 0.8)
    X_va, y_va = X_tr[vs:], y_tr[vs:]
    X_tr_fit, y_tr_fit = X_tr[:vs], y_tr[:vs]

    atr_thresh = np.nanquantile(sub_tr_good["atr_ratio"].values, 0.50)
    atr_mask_te = sub_te["atr_ratio"].values < atr_thresh

    close_dir = boost.sig_labels(r_te, delta)  # +1/-1/0 based on 5m close move
    gamma_y = sub_te["y"].values  # +1/0/NaN

    out = {}
    for name, fn in method_list:
        t0 = time.time()
        try:
            conf = fn(X_tr_fit, y_tr_fit, X_va, y_va, X_te, args)
        except Exception as e:
            print(f"      {name:<18} FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        if getattr(args, "flip_direction", False):
            conf = -conf  # sanity: predict opposite of what model says
        sig = top_k_signals(conf, sub_te, args.k_pct, atr_mask_te)
        # Annotate with both labels
        te_idx_by_ts = {ts: i for i, ts in enumerate(sub_te["timestamp"].values)}
        sig["te_idx"] = sig["timestamp"].map(te_idx_by_ts)
        sig["close_dir"] = close_dir[sig["te_idx"].values]
        sig["gamma_y"] = [gamma_y[i] if not pd.isna(gamma_y[i]) else np.nan
                          for i in sig["te_idx"].values]
        sig["correct_close"] = np.where(sig["close_dir"] == 0, np.nan,
                                         (sig["direction"] == sig["close_dir"]).astype(float))
        # gamma_y: 1 = UP won, 0 = DOWN won
        sig["correct_gamma"] = np.where(
            sig["gamma_y"].isna(), np.nan,
            ((sig["direction"] == 1) & (sig["gamma_y"] == 1)) |
            ((sig["direction"] == -1) & (sig["gamma_y"] == 0))
        )
        sig["correct_gamma"] = sig["correct_gamma"].astype(float)
        elapsed = time.time() - t0
        out[name] = sig.drop(columns=["te_idx"])
        n = len(sig)
        n_close = sig["correct_close"].notna().sum()
        n_gamma = sig["correct_gamma"].notna().sum()
        wr_close = sig["correct_close"].mean() if n_close > 0 else float("nan")
        wr_gamma = sig["correct_gamma"].mean() if n_gamma > 0 else float("nan")
        mark = ""
        if wr_gamma == wr_gamma and wr_gamma >= 0.60: mark = " ★"
        if wr_gamma == wr_gamma and wr_gamma >= 0.65: mark = " ★★"
        print(f"      {name:<18} n={n:>3}  gamma_res={n_gamma:>3}  "
              f"wr_close={100*wr_close:>5.2f}%  wr_gamma={100*wr_gamma:>5.2f}%  ({elapsed:.1f}s){mark}",
              flush=True)
    return out


# --------------------------------------------------------------------------------------
# Ensemble search
# --------------------------------------------------------------------------------------

def signals_to_matrix(signals_per_window, method_names):
    """Flatten to one big DataFrame: [window, timestamp, method, direction, gamma_y].
    Long-format signal table for ensemble analysis."""
    rows = []
    for wi, per_method in signals_per_window.items():
        for name in method_names:
            if name not in per_method:
                continue
            df = per_method[name]
            for _, r in df.iterrows():
                rows.append({
                    "window": wi, "timestamp": r["timestamp"],
                    "method": name, "direction": int(r["direction"]),
                    "gamma_y": r["gamma_y"],
                    "correct_gamma": r["correct_gamma"],
                    "conf": r["conf"],
                })
    return pd.DataFrame(rows)


def combine_signals(long_df, methods, mode, min_agree=None):
    """mode: 'union' | 'intersection' | 'vote_k'
    For each (window, timestamp), aggregate direction votes over `methods`:
      - union: fire if any method fires; direction = sign(sum of votes)
      - intersection: fire only if ALL methods fire same direction
      - vote_k: fire if >= min_agree methods agree on same direction
    Returns DataFrame of fired signals with correct_gamma."""
    sub = long_df[long_df["method"].isin(methods)].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["window", "timestamp", "direction", "gamma_y", "correct_gamma"])

    # For each (window, timestamp), get votes per direction
    grp = sub.groupby(["window", "timestamp"])
    fired_rows = []
    for (win, ts), g in grp:
        dirs = g["direction"].values
        up = int((dirs == 1).sum())
        dn = int((dirs == -1).sum())
        total_methods = len(methods)
        methods_fired = len(g)  # unique methods that fired here

        if mode == "union":
            if up == dn:  # tie → skip
                continue
            direction = 1 if up > dn else -1
        elif mode == "intersection":
            if methods_fired < total_methods:
                continue
            if up > 0 and dn > 0:
                continue
            direction = 1 if up > 0 else -1
        elif mode.startswith("vote_"):
            k = min_agree if min_agree is not None else int(mode.split("_")[1])
            if up >= k:
                direction = 1
            elif dn >= k:
                direction = -1
            else:
                continue
        else:
            raise ValueError(f"Unknown mode: {mode}")

        gamma_y = g["gamma_y"].iloc[0]  # same for all rows at same ts
        if pd.isna(gamma_y):
            correct = np.nan
        else:
            correct = float(
                ((direction == 1) and (gamma_y == 1)) or
                ((direction == -1) and (gamma_y == 0))
            )
        fired_rows.append({"window": win, "timestamp": ts,
                            "direction": direction, "gamma_y": gamma_y,
                            "correct_gamma": correct})
    return pd.DataFrame(fired_rows)


def evaluate_ensemble(fired, per_window_starts=None):
    """Compute total PnL and WR across all windows (each window starts fresh at $100)."""
    if len(fired) == 0:
        return {"total_pnl": 0.0, "wr": None, "n_resolved": 0, "n_signals": 0}
    total_pnl_delta = 0.0
    total_resolved = 0
    total_correct = 0
    for win, g in fired.groupby("window"):
        _, _, wr, resolved = simulate_pnl(g, start_bank=100.0)
        # final_bank - 100 = pnl delta for that window
        _, trades_df, _, _ = simulate_pnl(g, start_bank=100.0)
        final_bank = 100.0 + (trades_df["bank_after"].iloc[-1] - 100.0 if len(trades_df) else 0.0)
        # Simpler: rerun and take final bank
        bank_final, trades, wr_win, res = simulate_pnl(g, start_bank=100.0)
        total_pnl_delta += (bank_final - 100.0)
        total_resolved += res
        if res > 0 and wr_win is not None:
            total_correct += int(round(wr_win * res))
    return {
        "total_pnl_delta": round(total_pnl_delta, 2),
        "wr": (total_correct / total_resolved) if total_resolved > 0 else None,
        "n_resolved": total_resolved,
        "n_signals": len(fired),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2026-05-18")
    ap.add_argument("--end", type=str, default="2026-08-01")
    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--delta-bps", type=float, default=5.0)
    ap.add_argument("--k-pct", type=float, default=5.0)
    ap.add_argument("--pop", type=int, default=15)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--methods", type=str,
                    default="BASELINE,PLATT,ISOTONIC,TEMPERATURE,CMA_ENSEMBLE_3,LR_L2,XGB,LGBM,STACK,MLP,META_LABEL,VENN_ABERS,PCA_LR,ICA_LR,FA_LR,PLS,LDA,CATBOOST,EXTRA_TREES,HIST_GB,NYSTROEM_LR,RIDGE_CLF")
    ap.add_argument("--out-tag", type=str, default="gamma_wf")
    ap.add_argument("--gamma-shift-min", type=int, default=-5,
                    help="Minutes to shift gamma bucket for the merge. -5 = correct model alignment, 0 = off-by-one")
    ap.add_argument("--flip-direction", action="store_true",
                    help="Invert prediction direction before comparing to gamma (sanity test)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 1m data + resampling to 5m ...", flush=True)
    df_5m = load_5m_from_1m(BTC1M_PATH)
    print(f"  5m rows: {len(df_5m)}, range: {df_5m['timestamp'].min()} .. {df_5m['timestamp'].max()}", flush=True)

    df_tf = build_features(df_5m, args.window)
    gamma = load_gamma(GAMMA_PATH)
    gamma["timestamp"] = pd.to_datetime(gamma["timestamp"])
    print(f"  gamma rows: {len(gamma)}, range: {gamma['timestamp'].min()} .. {gamma['timestamp'].max()}", flush=True)
    df_tf["timestamp"] = pd.to_datetime(df_tf["timestamp"])
    df_labeled = align_labels(df_tf, gamma, shift_min=args.gamma_shift_min)
    print(f"  labeled rows (5m ∩ gamma): {df_labeled['y'].notna().sum()}", flush=True)

    # Windows: gamma start → end
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    tr_bars = args.train_days * BPD
    te_bars = args.test_days * BPD
    step_bars = args.step_days * BPD

    df_labeled = df_labeled.sort_values("timestamp").reset_index(drop=True)
    # Find test window boundaries within [start, end]
    windows = []
    cur = start
    while cur + pd.Timedelta(days=args.test_days) <= end:
        te_start_idx = df_labeled.index[df_labeled["timestamp"] >= cur].min()
        if pd.isna(te_start_idx):
            break
        te_end_idx = te_start_idx + te_bars
        tr_start_idx = te_start_idx - tr_bars
        if tr_start_idx < 0 or te_end_idx > len(df_labeled):
            print(f"  skip window at {cur}: not enough data (tr_start={tr_start_idx}, te_end={te_end_idx})")
            cur += pd.Timedelta(days=args.step_days)
            continue
        windows.append((tr_start_idx, te_start_idx, te_end_idx))
        cur += pd.Timedelta(days=args.step_days)

    if args.smoke:
        windows = windows[:2]
    print(f"  windows: {len(windows)} (train={args.train_days}d, test={args.test_days}d, step={args.step_days}d)", flush=True)

    want = set(args.methods.split(","))
    method_list = [(n, f) for n, f in boost.METHODS if n in want]
    print(f"  methods ({len(method_list)}): {[n for n, _ in method_list]}", flush=True)

    signals_per_window = {}
    for wi, (tr_i, te_i, te_end) in enumerate(windows):
        sub_tr = df_labeled.iloc[tr_i:te_i].reset_index(drop=True)
        sub_te = df_labeled.iloc[te_i:te_end].reset_index(drop=True)
        print(f"\n  [w{wi}] train {sub_tr['timestamp'].iloc[0]}..{sub_tr['timestamp'].iloc[-1]}   "
              f"test {sub_te['timestamp'].iloc[0]}..{sub_te['timestamp'].iloc[-1]}", flush=True)
        print(f"    gamma resolutions in test: {sub_te['y'].notna().sum()} / {len(sub_te)}", flush=True)
        per_m = run_one_window(sub_tr, sub_te, args, method_list)
        if per_m:
            signals_per_window[wi] = per_m

    # ---------------- Per-method aggregation ----------------
    print(f"\n{'=' * 115}\nPER-METHOD SUMMARY (all windows, gamma ground truth)\n{'=' * 115}")
    print(f"  {'method':<20}  {'wins':>4}  {'n_sig':>5}  {'n_gamma':>7}  {'wr_close':>8}  {'wr_gamma':>8}  {'Δwr':>6}  {'PnL($)':>8}  {'ROI%':>6}")
    per_method_summary = []
    for name in [n for n, _ in method_list]:
        n_sig_total = 0
        n_close_total = 0
        n_gamma_total = 0
        correct_close = 0
        correct_gamma = 0
        pnl_total = 0.0
        n_wins = 0
        for wi, per_m in signals_per_window.items():
            if name not in per_m:
                continue
            n_wins += 1
            df = per_m[name]
            n_sig_total += len(df)
            n_close_total += df["correct_close"].notna().sum()
            n_gamma_total += df["correct_gamma"].notna().sum()
            correct_close += int(df["correct_close"].fillna(0).sum())
            correct_gamma += int(df["correct_gamma"].fillna(0).sum())
            # PnL for this window (start $100)
            df_pnl = df[["timestamp", "direction", "correct_gamma"]].copy()
            bank, trades, wr_win, _ = simulate_pnl(df_pnl, start_bank=100.0)
            pnl_total += (bank - 100.0)
        wr_close = correct_close / n_close_total if n_close_total else float("nan")
        wr_gamma = correct_gamma / n_gamma_total if n_gamma_total else float("nan")
        dwr = (wr_gamma - wr_close) * 100 if (wr_close == wr_close and wr_gamma == wr_gamma) else float("nan")
        # ROI: total pnl / (n_windows * $100 start)
        roi = pnl_total / max(n_wins * 100.0, 1e-9) * 100
        mark = ""
        if wr_gamma == wr_gamma and wr_gamma >= 0.60: mark = " ★"
        if wr_gamma == wr_gamma and wr_gamma >= 0.65: mark = " ★★"
        per_method_summary.append({
            "method": name, "n_windows": n_wins,
            "n_signals": n_sig_total, "n_gamma_resolved": n_gamma_total,
            "wr_close": wr_close, "wr_gamma": wr_gamma, "delta_wr_pp": dwr,
            "total_pnl": round(pnl_total, 2), "roi_pct": round(roi, 2),
        })
        print(f"  {name:<20}  {n_wins:>4}  {n_sig_total:>5}  {n_gamma_total:>7}  "
              f"{100*wr_close:>6.2f}%  {100*wr_gamma:>6.2f}%  {dwr:>+5.1f}pp  "
              f"${pnl_total:>+7.2f}  {roi:>+5.1f}%{mark}")

    # Sort by PnL
    per_method_summary.sort(key=lambda x: -x["total_pnl"])
    print(f"\n  TOP 5 BY PnL:  " + ", ".join(f"{r['method']}(${r['total_pnl']:+.0f})"
                                              for r in per_method_summary[:5]))
    per_method_summary.sort(key=lambda x: -(x["wr_gamma"] if x["wr_gamma"] == x["wr_gamma"] else 0))
    print(f"  TOP 5 BY WR :  " + ", ".join(f"{r['method']}({100*r['wr_gamma']:.1f}%)"
                                             for r in per_method_summary[:5]))

    # ---------------- Ensemble search ----------------
    long_df = signals_to_matrix(signals_per_window, [n for n, _ in method_list])
    print(f"\nTotal signals in long-format matrix: {len(long_df)}", flush=True)

    # Top 8 methods by gamma WR for ensemble base
    per_method_summary.sort(key=lambda x: -(x["wr_gamma"] if x["wr_gamma"] == x["wr_gamma"] else 0))
    top8 = [r["method"] for r in per_method_summary
            if r["wr_gamma"] == r["wr_gamma"]][:8]
    print(f"\nENSEMBLE SEARCH — base set (top 8 by gamma WR): {top8}\n{'=' * 115}")

    combo_results = []
    # All pairs
    for pair in combinations(top8, 2):
        for mode in ("union", "intersection", "vote_2"):
            fired = combine_signals(long_df, list(pair), mode)
            r = evaluate_ensemble(fired)
            r.update({"combo": "+".join(pair), "mode": mode, "size": 2})
            combo_results.append(r)
    # All triples with vote_2 or intersection
    for tri in combinations(top8[:6], 3):
        for mode in ("union", "intersection", "vote_2"):
            fired = combine_signals(long_df, list(tri), mode)
            r = evaluate_ensemble(fired)
            r.update({"combo": "+".join(tri), "mode": mode, "size": 3})
            combo_results.append(r)

    # Add individual baselines
    for name in top8:
        fired = combine_signals(long_df, [name], "union")
        r = evaluate_ensemble(fired)
        r.update({"combo": name, "mode": "single", "size": 1})
        combo_results.append(r)

    print(f"\n{'-' * 115}\nMAX PnL COMBOS:\n{'-' * 115}")
    print(f"  {'combo':<70}  {'mode':<14}  {'sigs':>5}  {'gres':>5}  {'wr':>7}  {'PnL($)':>8}")
    for r in sorted(combo_results, key=lambda x: -x["total_pnl_delta"])[:15]:
        wr_s = f"{100*r['wr']:.1f}%" if r['wr'] is not None else "n/a"
        print(f"  {r['combo'][:68]:<70}  {r['mode']:<14}  {r['n_signals']:>5}  {r['n_resolved']:>5}  "
              f"{wr_s:>7}  ${r['total_pnl_delta']:>+7.2f}")

    print(f"\n{'-' * 115}\nMAX WR COMBOS (min 20 resolved):\n{'-' * 115}")
    print(f"  {'combo':<70}  {'mode':<14}  {'sigs':>5}  {'gres':>5}  {'wr':>7}  {'PnL($)':>8}")
    survivors = [r for r in combo_results if r["n_resolved"] >= 20 and r["wr"] is not None]
    for r in sorted(survivors, key=lambda x: -x["wr"])[:15]:
        print(f"  {r['combo'][:68]:<70}  {r['mode']:<14}  {r['n_signals']:>5}  {r['n_resolved']:>5}  "
              f"{100*r['wr']:>5.1f}%  ${r['total_pnl_delta']:>+7.2f}")

    # Save everything
    tag = args.out_tag
    per_method_df = pd.DataFrame(per_method_summary)
    per_method_df.to_csv(OUTPUT_DIR / f"{tag}_per_method.csv", index=False)
    long_df.to_csv(OUTPUT_DIR / f"{tag}_signals_long.csv", index=False)
    pd.DataFrame(combo_results).to_csv(OUTPUT_DIR / f"{tag}_combos.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
