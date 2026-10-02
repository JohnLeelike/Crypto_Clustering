#!/usr/bin/env python3
"""
boost_walkforward_5m_v2.py
==========================
Extended method battery on tr120d/te14d walk-forward (5m, δ=5bps, atr_low, top-5% conf).

Fixes vs v1:
  - BLAS/OMP thread limits (set BEFORE numpy import) to avoid CPU contention lockup.
  - --smoke flag: 3 windows for quick sanity check.
  - --methods flag: comma-list of method names to run (default all).
  - Progress prints per-method-per-window.
  - Dim-reduce methods (PCA/KPCA/ICA/FA/PLS/LDA/Isomap) each paired with LR downstream.
  - SOTA classifiers (CatBoost/ExtraTrees/HistGB/RandomForest/Nystroem+LR/SGDClassifier/PassiveAggressive/RidgeClassifier).
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier,
)
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import Isomap
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
spec = importlib.util.spec_from_file_location("wf", HERE / "cmaes_tsfresh_multi_tf.py")
wf = importlib.util.module_from_spec(spec)
sys.modules["wf"] = wf
spec.loader.exec_module(wf)

BPD = 288


def sig_labels(r, delta):
    y = np.zeros(len(r), dtype=np.int8)
    y[r > delta] = 1
    y[r < -delta] = -1
    return y


def top_k_prec(conf_signed, r, mask, k_pct, delta):
    y = sig_labels(r, delta)
    idx_pool = np.where(mask)[0]
    if len(idx_pool) == 0:
        return None, 0
    k = max(3, int(len(idx_pool) * k_pct / 100.0))
    k = min(k, len(idx_pool))
    abs_c = np.abs(conf_signed[idx_pool])
    top = np.argpartition(-abs_c, k - 1)[:k]
    final_idx = idx_pool[top]
    pred = np.sign(conf_signed[final_idx])
    sub_y = y[final_idx]
    active = (pred != 0) & (sub_y != 0)
    if active.sum() == 0:
        return None, 0
    prec = float((pred[active] == sub_y[active]).sum() / active.sum())
    return prec, int(active.sum())


# --- Original 14 methods (compact versions) ---

def m_baseline(X_tr, y_tr, X_va, y_va, X_te, args, seed=42):
    n_params = X_tr.shape[1] + 2
    r_proxy = y_tr.astype(np.float64)
    fit = wf.make_fitness_fn(X_tr.astype(np.float64), r_proxy, delta=1e-9,
                             min_signals=max(5, len(y_tr) // 500))
    best_x, _ = wf.cma_train(fit, n_params, args.pop, args.gens, seed, sigma_init=0.5, restarts=0)
    w, b = best_x[:-2], float(best_x[-2])
    return X_te @ w + b


def _cmaes(X_tr, y_tr, X_te, args, seed):
    n_params = X_tr.shape[1] + 2
    r_proxy = y_tr.astype(np.float64)
    fit = wf.make_fitness_fn(X_tr.astype(np.float64), r_proxy, delta=1e-9,
                             min_signals=max(5, len(y_tr) // 500))
    best_x, _ = wf.cma_train(fit, n_params, args.pop, args.gens, seed, sigma_init=0.5, restarts=0)
    w, b = best_x[:-2], float(best_x[-2])
    return X_te @ w + b


def m_platt(X_tr, y_tr, X_va, y_va, X_te, args):
    s_tr = _cmaes(X_tr, y_tr, X_tr, args, seed=42)
    s_te = _cmaes(X_tr, y_tr, X_te, args, seed=42)
    mask = y_tr != 0
    if mask.sum() < 20:
        return s_te
    lr = LogisticRegression(class_weight="balanced", max_iter=200)
    lr.fit(s_tr[mask].reshape(-1, 1), (y_tr[mask] == 1).astype(int))
    return lr.predict_proba(s_te.reshape(-1, 1))[:, 1] - 0.5


def m_isotonic(X_tr, y_tr, X_va, y_va, X_te, args):
    s_tr = _cmaes(X_tr, y_tr, X_tr, args, seed=42)
    s_te = _cmaes(X_tr, y_tr, X_te, args, seed=42)
    mask = y_tr != 0
    if mask.sum() < 20:
        return s_te
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s_tr[mask], (y_tr[mask] == 1).astype(int))
    return iso.predict(s_te) - 0.5


def m_temperature(X_tr, y_tr, X_va, y_va, X_te, args):
    from scipy.optimize import minimize_scalar
    s_va = _cmaes(X_tr, y_tr, X_va, args, seed=42)
    s_te = _cmaes(X_tr, y_tr, X_te, args, seed=42)
    mask_va = y_va != 0
    if mask_va.sum() < 10:
        return s_te
    y_va_bin = (y_va[mask_va] == 1).astype(int)
    def nll(T):
        p = 1.0 / (1.0 + np.exp(-s_va[mask_va] / max(T, 0.01)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y_va_bin * np.log(p) + (1 - y_va_bin) * np.log(1 - p))
    T_opt = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded").x
    return 1.0 / (1.0 + np.exp(-s_te / T_opt)) - 0.5


def m_cma_ensemble(X_tr, y_tr, X_va, y_va, X_te, args, n_seeds=3):
    scores = np.zeros(len(X_te))
    for i in range(n_seeds):
        scores += _cmaes(X_tr, y_tr, X_te, args, seed=42 + 100 * i)
    return scores / n_seeds


def _fit_prob_clf(clf, X_tr, y_tr, X_te):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    clf.fit(X_tr[mask], (y_tr[mask] == 1).astype(int))
    return clf.predict_proba(X_te)[:, 1] - 0.5


def m_lr(X_tr, y_tr, X_va, y_va, X_te, args):
    return _fit_prob_clf(LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", n_jobs=1),
                          X_tr, y_tr, X_te)


def m_xgb(X_tr, y_tr, X_va, y_va, X_te, args):
    return _fit_prob_clf(xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=1.0, reg_lambda=2.0, objective="binary:logistic",
        eval_metric="logloss", n_jobs=1, random_state=42, verbosity=0,
    ), X_tr, y_tr, X_te)


def m_xgb_rank(X_tr, y_tr, X_va, y_va, X_te, args):
    mask = y_tr != 0
    if mask.sum() < 100:
        return np.zeros(len(X_te))
    y_bin = (y_tr[mask] == 1).astype(int)
    d_tr = xgb.DMatrix(X_tr[mask], label=y_bin)
    d_tr.set_group([len(y_bin)])
    params = {"objective": "rank:pairwise", "eta": 0.05, "max_depth": 4,
              "subsample": 0.8, "colsample_bytree": 0.7,
              "min_child_weight": 10, "verbosity": 0, "nthread": 1}
    booster = xgb.train(params, d_tr, num_boost_round=150)
    raw = booster.predict(xgb.DMatrix(X_te))
    return raw - np.median(raw)


def m_lgbm(X_tr, y_tr, X_va, y_va, X_te, args):
    return _fit_prob_clf(lgb.LGBMClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        reg_alpha=1.0, reg_lambda=2.0, objective="binary",
        n_jobs=1, random_state=42, verbose=-1,
    ), X_tr, y_tr, X_te)


def m_stack(X_tr, y_tr, X_va, y_va, X_te, args):
    p1 = m_lr(X_tr, y_tr, X_va, y_va, X_te, args) + 0.5
    p2 = m_xgb(X_tr, y_tr, X_va, y_va, X_te, args) + 0.5
    p3 = m_lgbm(X_tr, y_tr, X_va, y_va, X_te, args) + 0.5
    return (p1 + p2 + p3) / 3.0 - 0.5


def m_mlp(X_tr, y_tr, X_va, y_va, X_te, args):
    return _fit_prob_clf(MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
        batch_size=256, learning_rate_init=1e-3,
        early_stopping=True, validation_fraction=0.15,
        max_iter=80, random_state=42,
    ), X_tr, y_tr, X_te)


def m_meta_label(X_tr, y_tr, X_va, y_va, X_te, args):
    s_tr = _cmaes(X_tr, y_tr, X_tr, args, seed=42)
    s_te = _cmaes(X_tr, y_tr, X_te, args, seed=42)
    active_tr = (y_tr != 0)
    if active_tr.sum() < 100:
        return s_te
    pred_tr = np.sign(s_tr[active_tr])
    correct = (pred_tr == y_tr[active_tr]).astype(int)
    X_meta_tr = np.hstack([X_tr[active_tr], s_tr[active_tr].reshape(-1, 1)])
    X_meta_te = np.hstack([X_te, s_te.reshape(-1, 1)])
    meta = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=1, random_state=42, verbosity=0,
    )
    meta.fit(X_meta_tr, correct)
    p_correct = meta.predict_proba(X_meta_te)[:, 1]
    return np.sign(s_te) * p_correct


def m_venn_abers(X_tr, y_tr, X_va, y_va, X_te, args):
    n = len(X_tr)
    half = n // 2
    order = np.random.default_rng(42).permutation(n)
    a_idx, b_idx = order[:half], order[half:]
    s_te_A = _cmaes(X_tr[a_idx], y_tr[a_idx], X_te, args, seed=17)
    s_b_A = _cmaes(X_tr[a_idx], y_tr[a_idx], X_tr[b_idx], args, seed=17)
    s_te_B = _cmaes(X_tr[b_idx], y_tr[b_idx], X_te, args, seed=29)
    s_a_B = _cmaes(X_tr[b_idx], y_tr[b_idx], X_tr[a_idx], args, seed=29)
    mask_a = y_tr[a_idx] != 0
    mask_b = y_tr[b_idx] != 0
    if mask_a.sum() < 30 or mask_b.sum() < 30:
        avg = (s_te_A + s_te_B) / 2
        return avg - np.median(avg)
    iso_A = IsotonicRegression(out_of_bounds="clip")
    iso_A.fit(s_b_A[mask_b], (y_tr[b_idx][mask_b] == 1).astype(int))
    iso_B = IsotonicRegression(out_of_bounds="clip")
    iso_B.fit(s_a_B[mask_a], (y_tr[a_idx][mask_a] == 1).astype(int))
    return (iso_A.predict(s_te_A) + iso_B.predict(s_te_B)) / 2.0 - 0.5


# ---------------------------------------------------------------
# Dim-reduce → LR downstream
# ---------------------------------------------------------------

def _pipeline_dr_lr(dr, X_tr, y_tr, X_te):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    y_bin = (y_tr[mask] == 1).astype(int)
    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", n_jobs=1)
    try:
        Z_tr = dr.fit_transform(X_tr[mask], y_bin) if _dr_supports_y(dr) else dr.fit_transform(X_tr[mask])
        Z_te = dr.transform(X_te)
    except Exception as e:
        raise RuntimeError(f"dim-reduce failed: {type(dr).__name__}: {e}")
    lr.fit(Z_tr, y_bin)
    return lr.predict_proba(Z_te)[:, 1] - 0.5


def _dr_supports_y(dr):
    return isinstance(dr, (LinearDiscriminantAnalysis, PLSRegression))


def m_pca_lr(X_tr, y_tr, X_va, y_va, X_te, args):
    return _pipeline_dr_lr(PCA(n_components=min(15, X_tr.shape[1]), random_state=42),
                            X_tr, y_tr, X_te)


def m_kpca_lr(X_tr, y_tr, X_va, y_va, X_te, args):
    return _pipeline_dr_lr(KernelPCA(n_components=min(15, X_tr.shape[1]), kernel="rbf",
                                     gamma=0.05, random_state=42, n_jobs=1),
                            X_tr, y_tr, X_te)


def m_ica_lr(X_tr, y_tr, X_va, y_va, X_te, args):
    return _pipeline_dr_lr(FastICA(n_components=min(15, X_tr.shape[1]), random_state=42, max_iter=200),
                            X_tr, y_tr, X_te)


def m_fa_lr(X_tr, y_tr, X_va, y_va, X_te, args):
    return _pipeline_dr_lr(FactorAnalysis(n_components=min(15, X_tr.shape[1]), random_state=42),
                            X_tr, y_tr, X_te)


def m_pls(X_tr, y_tr, X_va, y_va, X_te, args):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    y_signed = y_tr[mask].astype(np.float64)  # -1, +1
    pls = PLSRegression(n_components=min(5, X_tr.shape[1]))
    pls.fit(X_tr[mask], y_signed)
    return pls.predict(X_te).ravel()


def m_lda_direct(X_tr, y_tr, X_va, y_va, X_te, args):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    y_bin = (y_tr[mask] == 1).astype(int)
    lda.fit(X_tr[mask], y_bin)
    return lda.decision_function(X_te)


def m_isomap_lr(X_tr, y_tr, X_va, y_va, X_te, args):
    """Isomap can be SLOW (O(n^2) memory). Cap train samples for speed."""
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    idx = np.where(mask)[0]
    if len(idx) > 3000:
        idx = np.random.default_rng(42).choice(idx, 3000, replace=False)
    X_sub, y_bin = X_tr[idx], (y_tr[idx] == 1).astype(int)
    iso = Isomap(n_components=min(10, X_sub.shape[1]), n_neighbors=15, n_jobs=1)
    Z_tr = iso.fit_transform(X_sub)
    Z_te = iso.transform(X_te)
    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", n_jobs=1)
    lr.fit(Z_tr, y_bin)
    return lr.predict_proba(Z_te)[:, 1] - 0.5


# ---------------------------------------------------------------
# SOTA classifiers
# ---------------------------------------------------------------

def m_catboost(X_tr, y_tr, X_va, y_va, X_te, args):
    if not HAS_CATBOOST:
        return np.zeros(len(X_te))
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    clf = cb.CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                 l2_leaf_reg=3.0, verbose=0, thread_count=2,
                                 random_seed=42, class_weights=[1.0, 1.0])
    clf.fit(X_tr[mask], (y_tr[mask] == 1).astype(int))
    return clf.predict_proba(X_te)[:, 1] - 0.5


def m_extra_trees(X_tr, y_tr, X_va, y_va, X_te, args):
    return _fit_prob_clf(ExtraTreesClassifier(n_estimators=200, max_depth=8,
                                                min_samples_leaf=20, n_jobs=1,
                                                random_state=42),
                          X_tr, y_tr, X_te)


def m_hist_gb(X_tr, y_tr, X_va, y_va, X_te, args):
    return _fit_prob_clf(HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.05, l2_regularization=1.0,
        random_state=42,
    ), X_tr, y_tr, X_te)


def m_random_forest(X_tr, y_tr, X_va, y_va, X_te, args):
    return _fit_prob_clf(RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  min_samples_leaf=20, n_jobs=1,
                                                  random_state=42),
                          X_tr, y_tr, X_te)


def m_nystroem_lr(X_tr, y_tr, X_va, y_va, X_te, args):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    ny = Nystroem(kernel="rbf", gamma=0.05, n_components=200, random_state=42, n_jobs=1)
    Z_tr = ny.fit_transform(X_tr[mask])
    Z_te = ny.transform(X_te)
    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", n_jobs=1)
    lr.fit(Z_tr, (y_tr[mask] == 1).astype(int))
    return lr.predict_proba(Z_te)[:, 1] - 0.5


def m_sgd_huber(X_tr, y_tr, X_va, y_va, X_te, args):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    clf = SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=200,
                        class_weight="balanced", random_state=42, n_jobs=1)
    clf.fit(X_tr[mask], (y_tr[mask] == 1).astype(int))
    return clf.predict_proba(X_te)[:, 1] - 0.5


def m_passive_aggressive(X_tr, y_tr, X_va, y_va, X_te, args):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    clf = PassiveAggressiveClassifier(C=0.5, max_iter=200, class_weight="balanced",
                                       random_state=42, n_jobs=1)
    clf.fit(X_tr[mask], (y_tr[mask] == 1).astype(int))
    # No predict_proba — use decision_function as signed conf
    return clf.decision_function(X_te)


def m_ridge_classifier(X_tr, y_tr, X_va, y_va, X_te, args):
    mask = y_tr != 0
    if mask.sum() < 50:
        return np.zeros(len(X_te))
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(X_tr[mask], (y_tr[mask] == 1).astype(int))
    return clf.decision_function(X_te)


METHODS = [
    # Baselines
    ("BASELINE",        m_baseline),
    ("PLATT",           m_platt),
    ("ISOTONIC",        m_isotonic),
    ("TEMPERATURE",     m_temperature),
    ("CMA_ENSEMBLE_3",  m_cma_ensemble),
    # Standard classifiers
    ("LR_L2",           m_lr),
    ("XGB",             m_xgb),
    ("XGB_RANK",        m_xgb_rank),
    ("LGBM",            m_lgbm),
    ("STACK",           m_stack),
    ("MLP",             m_mlp),
    ("META_LABEL",      m_meta_label),
    ("VENN_ABERS",      m_venn_abers),
    # Dim-reduce → LR
    ("PCA_LR",          m_pca_lr),
    ("KPCA_LR",         m_kpca_lr),
    ("ICA_LR",          m_ica_lr),
    ("FA_LR",           m_fa_lr),
    ("PLS",             m_pls),
    ("LDA",             m_lda_direct),
    ("ISOMAP_LR",       m_isomap_lr),
    # SOTA classifiers
    ("CATBOOST",        m_catboost),
    ("EXTRA_TREES",     m_extra_trees),
    ("HIST_GB",         m_hist_gb),
    ("RF",              m_random_forest),
    ("NYSTROEM_LR",     m_nystroem_lr),
    ("SGD_HUBER",       m_sgd_huber),
    ("PASSIVE_AGGR",    m_passive_aggressive),
    ("RIDGE_CLF",       m_ridge_classifier),
]


def run_battery(df_tf, start, end, args, method_list):
    delta = args.delta_bps * 1e-4
    feats_all = wf.get_feature_names(df_tf)
    df_tf = df_tf.copy()
    df_tf["timestamp"] = pd.to_datetime(df_tf["timestamp"])
    df_slice = df_tf[(df_tf["timestamp"] >= pd.Timestamp(start)) &
                     (df_tf["timestamp"] < pd.Timestamp(end))].reset_index(drop=True)

    tr_len = args.train_days * BPD
    te_len = args.test_days * BPD
    step_len = args.step_days * BPD
    windows = []
    t0 = 0
    while t0 + tr_len + te_len <= len(df_slice):
        windows.append(t0)
        t0 += step_len
    if args.smoke:
        windows = windows[:3]
    print(f"windows: {len(windows)} (train={args.train_days}d, test={args.test_days}d, step={args.step_days}d)", flush=True)
    print(f"methods to test ({len(method_list)}): {[n for n, _ in method_list]}", flush=True)

    all_results = {name: [] for name, _ in method_list}
    per_window_details = []

    for wi, t_start in enumerate(windows):
        tr_start, tr_end = t_start, t_start + tr_len
        te_start, te_end = tr_end, tr_end + te_len
        sub_tr = df_slice.iloc[tr_start:tr_end]
        sub_te = df_slice.iloc[te_start:te_end]

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
            continue

        # Feature selection + scaler on train only
        sel_idx = wf.select_features_train(X_tr_full, r_tr, delta, args.top_k, feats_all)
        X_tr = X_tr_full[:, sel_idx]
        X_te = X_te_full[:, sel_idx]
        med, scale = wf.fit_scaler_train(X_tr)
        X_tr = wf.apply_scaler(X_tr, med, scale)
        X_te = wf.apply_scaler(X_te, med, scale)

        y_tr = sig_labels(r_tr, delta)
        vs = int(len(X_tr) * 0.8)
        X_va, y_va = X_tr[vs:], y_tr[vs:]
        X_tr_fit, y_tr_fit = X_tr[:vs], y_tr[:vs]

        atr_thresh = np.nanquantile(sub_tr_good["atr_ratio"].values, 0.50)
        atr_mask_te = sub_te["atr_ratio"].values < atr_thresh

        row = {"window": wi, "test_start": str(sub_te["timestamp"].iloc[0])[:10],
               "test_end": str(sub_te["timestamp"].iloc[-1])[:10]}

        print(f"\n  [w{wi}] {row['test_start']}..{row['test_end']}", flush=True)
        for name, fn in method_list:
            t0m = time.time()
            try:
                conf = fn(X_tr_fit, y_tr_fit, X_va, y_va, X_te, args)
            except Exception as e:
                print(f"    {name:<18} FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            prec, n = top_k_prec(conf, r_te, atr_mask_te, args.k_pct, delta)
            elapsed = time.time() - t0m
            if prec is None:
                print(f"    {name:<18} no fires ({elapsed:.1f}s)", flush=True)
                continue
            all_results[name].append({"window": wi, "precision": prec, "n": n,
                                       "elapsed_sec": round(elapsed, 1)})
            mark = " ★★" if prec >= 0.65 else " ★" if prec >= 0.60 else ""
            print(f"    {name:<18} prec={100*prec:>6.2f}% n={n:>3} ({elapsed:.1f}s){mark}", flush=True)
            row[name] = prec
        per_window_details.append(row)

    return all_results, per_window_details


def summarize(all_results):
    rows = []
    for name, results in all_results.items():
        if not results:
            rows.append({"method": name, "windows": 0})
            continue
        precs = np.array([r["precision"] for r in results])
        ns = np.array([r["n"] for r in results])
        elapsed = np.array([r["elapsed_sec"] for r in results])
        rows.append({
            "method": name,
            "windows": len(results),
            "mean_prec": float(precs.mean()),
            "median_prec": float(np.median(precs)),
            "min_prec": float(precs.min()),
            "max_prec": float(precs.max()),
            "std_prec": float(precs.std()),
            "pct_ge_60": float((precs >= 0.60).mean() * 100),
            "pct_ge_65": float((precs >= 0.65).mean() * 100),
            "avg_sig_per_win": float(ns.mean()),
            "total_signals": int(ns.sum()),
            "avg_sec": float(elapsed.mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2023-01-01")
    ap.add_argument("--end", type=str, default="2024-01-01")
    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--delta-bps", type=float, default=5.0)
    ap.add_argument("--k-pct", type=float, default=5.0)
    ap.add_argument("--pop", type=int, default=15)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--smoke", action="store_true", help="Run only 3 windows")
    ap.add_argument("--methods", type=str, default="",
                    help="Comma-list of method names to run (default all)")
    ap.add_argument("--out-tag", type=str, default="boost_v2")
    args = ap.parse_args()

    if args.methods:
        want = set(args.methods.split(","))
        method_list = [(n, f) for n, f in METHODS if n in want]
    else:
        method_list = METHODS

    print("Loading 5m parquet + features ...", flush=True)
    df_5m = pd.read_parquet(wf.DATA_PATH)
    df_tf = wf.prepare_tf_data(df_5m, "5m", 100, use_cache=True)

    all_results, per_window = run_battery(df_tf, args.start, args.end, args, method_list)
    summary = summarize(all_results)

    summary_sorted = sorted([s for s in summary if s.get("windows", 0) > 0],
                             key=lambda x: -x["mean_prec"])
    print(f"\n{'=' * 115}\nBOOST V2 METHOD BATTERY  ({args.start[:4]}, tr{args.train_days}d/te{args.test_days}d, δ={args.delta_bps}bps, atr_low, top-{args.k_pct}%)\n{'=' * 115}")
    print(f"  {'method':<20}  {'wins':>4}  {'mean':>6}  {'median':>7}  {'min':>6}  {'std':>5}  {'≥60%':>5}  {'≥65%':>5}  {'sig/w':>6}  {'sec/w':>5}")
    for s in summary_sorted:
        mark = ""
        if s["mean_prec"] >= 0.63: mark = " ← beats WF baseline"
        elif s["mean_prec"] >= 0.60: mark = " ★"
        print(f"  {s['method']:<20}  {s['windows']:>4}  "
              f"{s['mean_prec']*100:>5.1f}%  {s['median_prec']*100:>6.1f}%  "
              f"{s['min_prec']*100:>5.1f}%  {s['std_prec']*100:>4.1f}pp  "
              f"{s['pct_ge_60']:>4.0f}%  {s['pct_ge_65']:>4.0f}%  "
              f"{s['avg_sig_per_win']:>6.1f}  {s['avg_sec']:>4.1f}s{mark}")

    out_dir = wf.OUTPUT_DIR
    (out_dir / f"{args.out_tag}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / f"{args.out_tag}_per_window.json").write_text(json.dumps(per_window, indent=2, default=str))
    print(f"\nSaved: {out_dir/f'{args.out_tag}_summary.json'}")


if __name__ == "__main__":
    main()
