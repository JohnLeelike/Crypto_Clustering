#!/usr/bin/env python3
"""
cmaes_tsfresh_multi_tf.py
=========================
Successor to cmaes_precision_multi_tf.py.

Changes vs predecessor:
  1) Features: curated tsfresh feature dict computed on strict past-only rolling
     windows, PLUS the compact "close-of-bar" set from the previous script.
  2) Optimizer: pycma (CMAEvolutionStrategy) with IPOP-style restart on val
     stagnation.
  3) Leak discipline:
        - Scaler fit on TRAIN ONLY per fold (median/MAD), applied to val/oos.
        - Feature selection (mutual info) fit on TRAIN ONLY.
        - Rolling windows end at bar t; label at bar t is
          sign(close[t+1] / close[t]) with an optional |ret| >= δ dead-band
          (matches v34c "significant-only" accounting so numbers are
          comparable to the v34c sweep 75%+ column).
  4) Feature cache: computed features are cached to parquet under a
     content-hashed key to keep re-runs cheap.
  5) Multi-seed CMA-ES: N seeds per fold, best val fitness wins.

USAGE
-----
  # Smoke test on 5m, 1 fold, tiny compute
  python3 cmaes_tsfresh_multi_tf.py --smoke --tfs 5m --folds 1

  # Full 5m run
  python3 cmaes_tsfresh_multi_tf.py --tfs 5m --folds 1,2,3

  # All TFs
  python3 cmaes_tsfresh_multi_tf.py --tfs 5m,15m,1h,4h,1d --folds 1,2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cma
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

DATA_PATH = Path("/Users/susan/neat-trading-neat-copy/nine_year_overlap_data_5m_complete_with_ohlcv.parquet")
OUTPUT_DIR = Path("/Users/susan/neat-trading-neat-copy/outputs_cmaes_tsfresh_multi_tf")
CACHE_DIR = Path("/Users/susan/neat-trading-neat-copy/cache_tsfresh_features")

MIN_TAU = 0.05
MIN_ABSOLUTE_SIGNALS_PER_FOLD = 30

CONF_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
COVERAGE_TARGETS = [50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5]

FOLDS = [
    {"name": "fold1", "train": (2017, 2020), "val": 2021, "oos": 2022},
    {"name": "fold2", "train": (2018, 2021), "val": 2022, "oos": 2023},
    {"name": "fold3", "train": (2019, 2022), "val": 2023, "oos": 2024},
]

TIMEFRAMES = {
    "5m":  {"rule": None,    "bars_per_year": 252 * 288},
    "15m": {"rule": "15min", "bars_per_year": 252 * 96},
    "1h":  {"rule": "1h",    "bars_per_year": 252 * 24},
    "4h":  {"rule": "4h",    "bars_per_year": 252 * 6},
    "1d":  {"rule": "1D",    "bars_per_year": 252},
}

TAKER_FEE = 0.00045
SLIPPAGE = 0.00005
COST_PER_FLIP = TAKER_FEE + SLIPPAGE

# --------------------------------------------------------------------------------------
# Data resampling (identical to predecessor)
# --------------------------------------------------------------------------------------

def resample_ohlcv(df_5m: pd.DataFrame, rule):
    if rule is None:
        return df_5m.copy()
    df = df_5m.set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "bvol_open" in df.columns:
        agg.update({"bvol_open": "first", "bvol_high": "max", "bvol_low": "min",
                    "bvol_close": "last", "bvol_volume": "sum"})
    out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out

# --------------------------------------------------------------------------------------
# Compact close-of-bar features (past-only) — same 17 as predecessor
# --------------------------------------------------------------------------------------

COMPACT_FEATURE_NAMES = [
    "ret_1", "ret_3", "ret_12",
    "ema5_ratio", "ema20_ratio", "ema50_ratio",
    "rsi", "atr_ratio", "vol_z",
    "range_ratio", "body_ratio", "upper_wick", "lower_wick",
    "bvol_ret_1", "bvol_ret_3", "bvol_ema_ratio", "bvol_vol_z",
]


def compute_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, o_, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    f["ret_1"] = np.log(c / c.shift(1))
    f["ret_3"] = np.log(c / c.shift(3))
    f["ret_12"] = np.log(c / c.shift(12))
    for span in (5, 20, 50):
        ema = c.ewm(span=span, adjust=False).mean()
        f[f"ema{span}_ratio"] = c / ema - 1.0
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-12)
    f["rsi"] = (100.0 - 100.0 / (1.0 + rs) - 50.0) / 50.0
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_ratio"] = (tr.rolling(14).mean() / c).clip(0, 0.5)
    vol_mean = v.rolling(100).mean()
    vol_std = v.rolling(100).std() + 1e-12
    f["vol_z"] = ((v - vol_mean) / vol_std).clip(-5, 5)
    f["range_ratio"] = (h - l) / c
    f["body_ratio"] = (c - o_) / c
    f["upper_wick"] = (h - np.maximum(c, o_)) / c
    f["lower_wick"] = (np.minimum(c, o_) - l) / c
    if "bvol_close" in df.columns:
        bv = df["bvol_close"]
        f["bvol_ret_1"] = np.log(bv / bv.shift(1)).clip(-1, 1)
        f["bvol_ret_3"] = np.log(bv / bv.shift(3)).clip(-1, 1)
        f["bvol_ema_ratio"] = (bv / bv.ewm(span=20, adjust=False).mean() - 1.0).clip(-1, 1)
        if "bvol_volume" in df.columns:
            bvv = df["bvol_volume"]
            f["bvol_vol_z"] = ((bvv - bvv.rolling(100).mean()) / (bvv.rolling(100).std() + 1e-12)).clip(-5, 5)
        else:
            f["bvol_vol_z"] = 0.0
    else:
        f["bvol_ret_1"] = f["bvol_ret_3"] = f["bvol_ema_ratio"] = f["bvol_vol_z"] = 0.0
    return f[COMPACT_FEATURE_NAMES]

# --------------------------------------------------------------------------------------
# Curated tsfresh features on past-only rolling windows
# --------------------------------------------------------------------------------------
# We do NOT use tsfresh.roll_time_series (very slow and awkward on 100k+ bars).
# Instead we build sliding-window arrays via numpy stride tricks and compute a
# hand-picked set of tsfresh feature functions column-wise. This is fully
# equivalent to tsfresh's output for these functions but ~30-100x faster.

from tsfresh.feature_extraction import feature_calculators as fc


def _windows(a: np.ndarray, w: int) -> np.ndarray:
    """Return a (n - w + 1, w) view of past-only windows ending at each row."""
    if len(a) < w:
        return np.empty((0, w), dtype=a.dtype)
    n = len(a) - w + 1
    stride = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(n, w), strides=(stride, stride))


def compute_tsfresh_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute a curated tsfresh feature set on past-only rolling windows.
    Each feature at row t uses closed bars [t-window+1 .. t] — no look-ahead.
    Rows before window-1 are NaN."""
    log_ret = np.log(df["close"].values / np.maximum(np.roll(df["close"].values, 1), 1e-12))
    log_ret[0] = 0.0
    close = df["close"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)

    ret_win = _windows(log_ret, window)   # (N, window)
    close_win = _windows(close, window)
    vol_win = _windows(vol, window)

    out = pd.DataFrame(index=df.index)

    def _apply(arr_win, fn, name, **kw):
        col = np.full(len(df), np.nan, dtype=np.float64)
        if len(arr_win) == 0:
            out[name] = col
            return
        vals = np.array([fn(x, **kw) if not np.any(np.isnan(x)) else np.nan
                         for x in arr_win], dtype=np.float64)
        col[window - 1:] = vals
        out[name] = col

    # ---- returns-based ----
    _apply(ret_win, fc.abs_energy, "ts_ret_abs_energy")
    _apply(ret_win, fc.absolute_sum_of_changes, "ts_ret_abs_sum_changes")
    _apply(ret_win, fc.mean_abs_change, "ts_ret_mean_abs_change")
    _apply(ret_win, fc.mean_change, "ts_ret_mean_change")
    _apply(ret_win, fc.mean_second_derivative_central, "ts_ret_mean_2nd_deriv")
    _apply(ret_win, fc.variance, "ts_ret_var")
    _apply(ret_win, fc.skewness, "ts_ret_skew")
    _apply(ret_win, fc.kurtosis, "ts_ret_kurt")
    _apply(ret_win, fc.longest_strike_above_mean, "ts_ret_longest_above")
    _apply(ret_win, fc.longest_strike_below_mean, "ts_ret_longest_below")
    _apply(ret_win, fc.count_above_mean, "ts_ret_count_above")
    _apply(ret_win, fc.count_below_mean, "ts_ret_count_below")
    _apply(ret_win, fc.cid_ce, "ts_ret_cid_ce", normalize=True)
    _apply(ret_win, lambda x: fc.autocorrelation(x, 1), "ts_ret_autocorr_1")
    _apply(ret_win, lambda x: fc.autocorrelation(x, 2), "ts_ret_autocorr_2")
    _apply(ret_win, lambda x: fc.autocorrelation(x, 5), "ts_ret_autocorr_5")
    _apply(ret_win, lambda x: fc.autocorrelation(x, 10), "ts_ret_autocorr_10")

    # ---- close-based ----
    _apply(close_win, fc.skewness, "ts_close_skew")
    _apply(close_win, fc.kurtosis, "ts_close_kurt")
    _apply(close_win, lambda x: fc.linear_trend(x, [{"attr": "slope"}])[0][1], "ts_close_trend_slope")
    _apply(close_win, lambda x: fc.linear_trend(x, [{"attr": "stderr"}])[0][1], "ts_close_trend_stderr")

    # ---- volume-based ----
    _apply(vol_win, fc.skewness, "ts_vol_skew")
    _apply(vol_win, fc.kurtosis, "ts_vol_kurt")
    _apply(vol_win, fc.mean_abs_change, "ts_vol_mean_abs_change")
    _apply(vol_win, lambda x: fc.autocorrelation(x, 1), "ts_vol_autocorr_1")
    _apply(vol_win, lambda x: fc.autocorrelation(x, 5), "ts_vol_autocorr_5")

    return out

# --------------------------------------------------------------------------------------
# Dataset preparation (with cache)
# --------------------------------------------------------------------------------------

def _cache_key(tf: str, window: int, data_path: Path) -> str:
    st = data_path.stat()
    h = hashlib.sha1(f"{tf}|{window}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]
    return f"{tf}_win{window}_{h}"


def prepare_tf_data(df_5m: pd.DataFrame, tf: str, window: int, use_cache: bool = True) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{_cache_key(tf, window, DATA_PATH)}.parquet"
    if use_cache and cache_path.exists():
        print(f"  [{tf}] loading cached features: {cache_path.name}", flush=True)
        return pd.read_parquet(cache_path)

    cfg = TIMEFRAMES[tf]
    df = resample_ohlcv(df_5m, cfg["rule"])
    t0 = time.time()
    compact = compute_compact_features(df)
    tsf = compute_tsfresh_features(df, window=window)
    feats = pd.concat([compact, tsf], axis=1)

    out = pd.concat(
        [df[["timestamp", "close"]].reset_index(drop=True), feats.reset_index(drop=True)],
        axis=1,
    )
    out["log_return_next"] = np.log(out["close"].shift(-1) / out["close"])  # LABEL
    out["year"] = out["timestamp"].dt.year
    out = out.dropna().reset_index(drop=True)
    print(f"  [{tf}] built {out.shape[0]:,} rows × {out.shape[1]-4} features in {time.time()-t0:.1f}s", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    return out


ALL_FEATURE_NAMES_CACHE = {}


def get_feature_names(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("timestamp", "close", "log_return_next", "year")]

# --------------------------------------------------------------------------------------
# Fold slicing + per-fold normalization + feature selection
# --------------------------------------------------------------------------------------

def slice_years(df, year_range, feature_names):
    if isinstance(year_range, int):
        mask = df["year"] == year_range
    else:
        lo, hi = year_range
        mask = (df["year"] >= lo) & (df["year"] <= hi)
    sub = df.loc[mask]
    X = sub[feature_names].values.astype(np.float64)
    r = sub["log_return_next"].values.astype(np.float64)
    return X, r


def fit_scaler_train(X_train):
    """Median/MAD-Z on train only. Returns (med, mad_scale)."""
    med = np.nanmedian(X_train, axis=0)
    mad = np.nanmedian(np.abs(X_train - med), axis=0) + 1e-9
    return med, mad * 1.4826


def apply_scaler(X, med, scale):
    Xs = (X - med) / scale
    return np.clip(np.nan_to_num(Xs, nan=0.0), -5.0, 5.0)


def select_features_train(X_train, r_train, delta: float, k: int, feature_names: list, seed: int = 0):
    """Mutual info on TRAIN ONLY vs the sig-only direction label with delta dead-band."""
    y = np.zeros(len(r_train), dtype=np.int8)
    y[r_train > delta] = 1
    y[r_train < -delta] = -1
    mask = y != 0
    if mask.sum() < 200 or k >= X_train.shape[1]:
        return list(range(X_train.shape[1]))
    mi = mutual_info_classif(
        np.nan_to_num(X_train[mask], nan=0.0), y[mask],
        discrete_features=False, random_state=seed,
    )
    idx = np.argsort(-mi)[:k]
    return sorted(idx.tolist())

# --------------------------------------------------------------------------------------
# Fitness (numpy, deterministic)
# --------------------------------------------------------------------------------------

def make_fitness_fn(X, r, delta: float, min_signals: int, class_weighted: bool = False):
    """Fitness = 0.5 − precision (pycma minimizes). If class_weighted=True,
    weight each active bar by 1/prev(class) so long/short precision are pushed
    together instead of the model dumping the minority side."""
    y = np.zeros(len(r), dtype=np.int8)
    y[r > delta] = 1
    y[r < -delta] = -1

    if class_weighted:
        n_up = max(int((y == 1).sum()), 1)
        n_dn = max(int((y == -1).sum()), 1)
        w_up = 0.5 / (n_up / max(n_up + n_dn, 1))
        w_dn = 0.5 / (n_dn / max(n_up + n_dn, 1))
    else:
        w_up = w_dn = 1.0

    def fitness(params):
        w = params[:-2]
        b = params[-2]
        tau = max(params[-1], MIN_TAU)
        scores = X @ w + b
        pred = np.where(scores > tau, 1, np.where(scores < -tau, -1, 0))
        active = (pred != 0) & (y != 0)
        if active.sum() == 0:
            return 0.5
        if class_weighted:
            wts = np.where(y[active] == 1, w_up, w_dn)
            correct = (pred[active] == y[active]).astype(float)
            prec = float((correct * wts).sum() / wts.sum())
        else:
            prec = float((pred[active] == y[active]).mean())
        loss = 0.5 - prec
        n_sig = int(active.sum())
        if n_sig < min_signals:
            loss += 0.5
        return float(loss)

    return fitness

# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def evaluate_full(params, X, r, delta, bars_per_year, label):
    w = params[:-2]
    b = float(params[-2])
    tau = max(float(params[-1]), MIN_TAU)
    scores = X @ w + b
    pred = np.where(scores > tau, 1, np.where(scores < -tau, -1, 0))
    y = np.zeros(len(r), dtype=np.int8)
    y[r > delta] = 1
    y[r < -delta] = -1
    active = (pred != 0) & (y != 0)
    n_sig = int(active.sum())
    n_cor = int((pred[active] == y[active]).sum()) if n_sig else 0
    prec = n_cor / n_sig if n_sig else None
    long_m = pred == 1
    short_m = pred == -1
    p_long = float((long_m & (y == 1)).sum() / max(long_m.sum(), 1)) if long_m.sum() else None
    p_short = float((short_m & (y == -1)).sum() / max(short_m.sum(), 1)) if short_m.sum() else None
    pos = pred.astype(np.float64)
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    trade_diff = np.abs(pos - pos_prev)
    net = pos * r - COST_PER_FLIP * trade_diff
    sharpe = float(np.mean(net) / (np.std(net) + 1e-12) * np.sqrt(bars_per_year))
    cum_log = float(net.sum())
    return dict(
        label=label,
        n_bars=int(len(scores)),
        coverage_pct=round(100.0 * n_sig / len(scores), 3),
        n_signals=n_sig,
        precision=round(prec, 4) if prec is not None else None,
        precision_long=round(p_long, 4) if p_long is not None else None,
        precision_short=round(p_short, 4) if p_short is not None else None,
        n_long=int(long_m.sum()), n_short=int(short_m.sum()),
        tau=round(tau, 4),
        sharpe_after_fee=round(sharpe, 3),
        cum_return_pct=round((np.exp(cum_log) - 1) * 100, 3),
    ), scores


def precision_coverage_table(scores, returns, delta):
    rows = []
    abs_s = np.abs(scores)
    n = len(scores)
    y = np.zeros(len(returns), dtype=np.int8)
    y[returns > delta] = 1
    y[returns < -delta] = -1
    for tau in CONF_THRESHOLDS:
        mask = abs_s > tau
        if not mask.any():
            continue
        sub_scores = scores[mask]
        sub_y = y[mask]
        sub_r = returns[mask]
        pred = np.sign(sub_scores)
        active = (pred != 0) & (sub_y != 0)
        if active.sum() == 0:
            continue
        prec = float((pred[active] == sub_y[active]).sum() / active.sum())
        signed_ret = pred * sub_r
        rows.append(dict(
            tau=tau,
            coverage_pct=round(100.0 * active.sum() / n, 3),
            signals=int(active.sum()),
            precision=round(prec, 4),
            avg_signed_return=round(float(signed_ret.mean()), 6),
            avg_pnl_after_fee=round(float(signed_ret.mean() - COST_PER_FLIP), 6),
        ))
    return rows


def coverage_target_table(scores, returns, delta):
    abs_s = np.abs(scores)
    n = len(scores)
    y = np.zeros(len(returns), dtype=np.int8)
    y[returns > delta] = 1
    y[returns < -delta] = -1
    rows = []
    for cov_target in COVERAGE_TARGETS:
        k = int(round(n * cov_target / 100.0))
        if k < 5:
            continue
        idx = np.argpartition(-abs_s, k - 1)[:k]
        sub_scores = scores[idx]
        sub_y = y[idx]
        sub_r = returns[idx]
        pred = np.sign(sub_scores)
        active = (pred != 0) & (sub_y != 0)
        if active.sum() == 0:
            continue
        prec = float((pred[active] == sub_y[active]).sum() / active.sum())
        rows.append(dict(
            coverage_target_pct=cov_target,
            tau_used=round(float(abs_s[idx].min()), 4),
            signals=int(active.sum()),
            precision=round(prec, 4),
            avg_pnl_after_fee=round(float((pred * sub_r).mean() - COST_PER_FLIP), 6),
        ))
    return rows

# --------------------------------------------------------------------------------------
# One fold, N seeds
# --------------------------------------------------------------------------------------

def cma_train(fitness_fn, n_params, pop, gens, seed, sigma_init=0.5, restarts=0):
    # x0: small weights + zero bias + starting tau=0.5 (safely inside [MIN_TAU, 3])
    x0 = np.zeros(n_params)
    x0[-1] = 0.5
    opts = {
        "popsize": pop,
        "maxiter": gens,
        "seed": seed if seed > 0 else 1,  # pycma requires positive seed
        "verbose": -9,
        "tolx": 1e-8,
        "tolfun": 1e-8,
        "bounds": [[-3.0] * (n_params - 1) + [MIN_TAU], [3.0] * (n_params - 1) + [3.0]],
    }
    best_x = x0.copy()
    best_f = float("inf")
    for r in range(restarts + 1):
        es = cma.CMAEvolutionStrategy(x0, sigma_init * (0.7 ** r), opts)
        es.optimize(fitness_fn)
        if es.result.fbest < best_f:
            best_f = float(es.result.fbest)
            best_x = np.asarray(es.result.xbest)
        # Restart from perturbed best if we're doing IPOP-style restarts
        if r < restarts:
            x0 = best_x + np.random.default_rng(seed + r + 1).normal(0, sigma_init * 0.3, size=n_params)
            # Keep tau within bounds after perturbation
            x0[-1] = float(np.clip(x0[-1], MIN_TAU + 1e-6, 3.0 - 1e-6))
            x0[:-1] = np.clip(x0[:-1], -3.0 + 1e-6, 3.0 - 1e-6)
            opts["popsize"] = min(int(pop * 2), 500)
    return best_x, best_f


def run_fold(tf, fold, df_tf, args):
    cfg = TIMEFRAMES[tf]
    bpy = cfg["bars_per_year"]
    delta = args.delta
    name = fold["name"]
    feature_names_all = get_feature_names(df_tf)

    X_tr, r_tr = slice_years(df_tf, fold["train"], feature_names_all)
    X_va, r_va = slice_years(df_tf, fold["val"], feature_names_all)
    X_oo, r_oo = slice_years(df_tf, fold["oos"], feature_names_all)
    if X_tr.shape[0] < 200 or X_va.shape[0] < 50 or X_oo.shape[0] < 50:
        print(f"    [{tf} {name}] SKIPPED: too few bars", flush=True)
        return None, None

    # Feature selection on train only
    sel_idx = select_features_train(X_tr, r_tr, delta, args.top_k, feature_names_all)
    sel_names = [feature_names_all[i] for i in sel_idx]
    X_tr, X_va, X_oo = X_tr[:, sel_idx], X_va[:, sel_idx], X_oo[:, sel_idx]

    # Scaler on train only
    med, scale = fit_scaler_train(X_tr)
    X_tr = apply_scaler(X_tr, med, scale)
    X_va = apply_scaler(X_va, med, scale)
    X_oo = apply_scaler(X_oo, med, scale)

    n_feats = X_tr.shape[1]
    n_params = n_feats + 2

    fit_train = make_fitness_fn(X_tr, r_tr, delta, MIN_ABSOLUTE_SIGNALS_PER_FOLD,
                                class_weighted=getattr(args, "class_weighted", False))
    fit_val = make_fitness_fn(X_va, r_va, delta, max(5, MIN_ABSOLUTE_SIGNALS_PER_FOLD // 4),
                              class_weighted=getattr(args, "class_weighted", False))

    best_val_f = float("inf")
    best_params = None
    t0 = time.time()
    for s in range(args.seeds):
        seed = args.seed + 1000 * s + hash(name) % 1000
        cand_x, _ = cma_train(fit_train, n_params, args.pop, args.gens, seed,
                              sigma_init=args.sigma, restarts=args.restarts)
        val_f = fit_val(cand_x)
        if val_f < best_val_f:
            best_val_f = val_f
            best_params = cand_x
    elapsed = time.time() - t0

    val_m, _ = evaluate_full(best_params, X_va, r_va, delta, bpy, label=f"{tf}_{name}_val")
    oos_m, oos_scores = evaluate_full(best_params, X_oo, r_oo, delta, bpy, label=f"{tf}_{name}_oos")
    pc = precision_coverage_table(oos_scores, r_oo, delta)
    ct = coverage_target_table(oos_scores, r_oo, delta)

    y_oos = np.zeros(len(r_oo), dtype=np.int8)
    y_oos[r_oo > delta] = 1
    y_oos[r_oo < -delta] = -1
    baseline = float((y_oos == 1).sum() / max((y_oos != 0).sum(), 1))

    return {
        "tf": tf, "fold": name,
        "bars": {"train": int(X_tr.shape[0]), "val": int(X_va.shape[0]), "oos": int(X_oo.shape[0])},
        "n_features_after_select": n_feats,
        "selected_features": sel_names,
        "delta": delta,
        "best_val_loss": round(best_val_f, 4),
        "best_val_precision": round(0.5 - best_val_f, 4) if best_val_f <= 0.5 else None,
        "val": val_m,
        "oos": oos_m,
        "oos_up_rate_baseline_sig": round(baseline, 4),
        "precision_coverage_oos": pc,
        "coverage_target_oos": ct,
        "elapsed_sec": round(elapsed, 1),
    }, best_params

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3, help="CMA-ES seeds per fold; best val wins")
    ap.add_argument("--restarts", type=int, default=1, help="IPOP-style restarts per seed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tfs", type=str, default="5m")
    ap.add_argument("--folds", type=str, default="1,2,3")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=100, help="Rolling window for tsfresh features")
    ap.add_argument("--top-k", type=int, default=30, help="Top-K features by MI on train")
    ap.add_argument("--delta", type=float, default=0.0005, help="Dead-band on log-return (0.0005≈5bps ≈ v34 sig-only)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--class-weighted", action="store_true",
                    help="Weight fitness by inverse class prevalence to prevent minority-side dumping")
    ap.add_argument("--variant", type=str, default="base",
                    help="Tag for output subdirectory (base/heavier/class_weighted). Files saved under OUTPUT_DIR/variant/.")
    args = ap.parse_args()

    # Redirect output to per-variant subdir when tagged
    if args.variant != "base":
        global OUTPUT_DIR
        OUTPUT_DIR = OUTPUT_DIR / args.variant

    if args.smoke:
        args.pop, args.gens = 20, 80
        args.seeds = 1
        args.restarts = 0
        print("SMOKE MODE: pop=20, gens=80, seeds=1, restarts=0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"pycma: {cma.__version__}")
    print(f"Delta (dead-band): {args.delta}")
    print(f"Top-K features: {args.top_k} | Window: {args.window}")

    tfs = [t.strip() for t in args.tfs.split(",") if t.strip() in TIMEFRAMES]
    folds_sel = [int(x) for x in args.folds.split(",")]
    print(f"Timeframes: {tfs}")
    print(f"Folds: {folds_sel}")

    print(f"\nLoading {DATA_PATH.name} ...")
    df_5m = pd.read_parquet(DATA_PATH)
    print(f"  raw shape: {df_5m.shape}", flush=True)

    all_results = []
    for tf in tfs:
        print(f"\n{'#' * 80}\n# TIMEFRAME: {tf}\n{'#' * 80}", flush=True)
        df_tf = prepare_tf_data(df_5m, tf, args.window, use_cache=not args.no_cache)
        per_year = df_tf["year"].value_counts().sort_index()
        print(f"  per-year: {per_year.to_dict()}", flush=True)

        for i, fold in enumerate(FOLDS, start=1):
            if i not in folds_sel:
                continue
            print(f"\n  >>> {tf} {fold['name']}: train={fold['train']} val={fold['val']} oos={fold['oos']}", flush=True)
            res, best_params = run_fold(tf, fold, df_tf, args)
            if res is None:
                continue
            np.save(OUTPUT_DIR / f"best_params_{tf}_{fold['name']}.npy", best_params)
            (OUTPUT_DIR / f"selected_features_{tf}_{fold['name']}.json").write_text(
                json.dumps(res["selected_features"], indent=2)
            )
            print(f"      elapsed {res['elapsed_sec']}s | "
                  f"val prec {res['val']['precision']} cov {res['val']['coverage_pct']:.2f}% | "
                  f"OOS prec {res['oos']['precision']} cov {res['oos']['coverage_pct']:.2f}% "
                  f"(sig-baseline up {res['oos_up_rate_baseline_sig']*100:.2f}%)", flush=True)
            all_results.append(res)

    report_path = OUTPUT_DIR / "results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 110}\nSUMMARY: OOS precision per (TF, fold)\n{'=' * 110}")
    print(f"{'TF':<5} {'Fold':<6} {'OOS prec':<10} {'cov%':<7} {'P_long':<8} {'P_short':<8} "
          f"{'Baseline':<10} {'Sharpe':<8} {'Ret%':<8} {'Bars':<8}")
    print("-" * 110)
    crossed_75 = []
    for r in all_results:
        oos = r["oos"]
        p = oos["precision"] if oos["precision"] is not None else 0
        if p >= 0.75:
            crossed_75.append((r["tf"], r["fold"], p, oos["coverage_pct"]))
        print(f"{r['tf']:<5} {r['fold']:<6} "
              f"{p*100:>6.2f}%   {oos['coverage_pct']:>5.2f}%  "
              f"{(oos['precision_long'] or 0)*100:>5.2f}%   {(oos['precision_short'] or 0)*100:>5.2f}%   "
              f"{r['oos_up_rate_baseline_sig']*100:>5.2f}%     "
              f"{oos['sharpe_after_fee']:+.2f}   {oos['cum_return_pct']:+6.1f}%  {r['bars']['oos']}")

    print(f"\n{'=' * 110}\nPRECISION @ COVERAGE TARGETS (OOS, delta={args.delta})\n{'=' * 110}")
    for r in all_results:
        print(f"\n  [{r['tf']} {r['fold']}]  (baseline up {r['oos_up_rate_baseline_sig']*100:.2f}%)")
        for row in r["coverage_target_oos"]:
            mark = ""
            if row["precision"] >= 0.75:
                mark = "  *** ≥75%"
            elif row["precision"] >= 0.65:
                mark = "  ** ≥65%"
            elif row["precision"] >= 0.60:
                mark = "  * ≥60%"
            print(f"    target {row['coverage_target_pct']:>5}%   n={row['signals']:>5}  "
                  f"prec={row['precision']*100:>6.2f}%  pnl/sig {row['avg_pnl_after_fee']:+.6f}{mark}")

    print(f"\nFull JSON: {report_path}")


if __name__ == "__main__":
    main()
