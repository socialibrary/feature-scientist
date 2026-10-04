"""Ablation harness: baseline + one candidate feature (Days 3-4).

Two backends, selected by run_scientist.py --model:

  * torch (default): PyTorch two-tower. The candidate feature is routed to
    the appropriate tower's input (provenance["tower"]: "user" | "movie" |
    "wide") via two_tower.train_two_tower(..., extra={tower: cols}).
  * lgbm: LightGBM on the flat baseline feature matrix + the candidate
    column (the original Days 1-4 path).

Deltas are always computed vs the matching baseline's published metrics.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from baseline import TARGET, featurize, make_model
from features import Feature


def _candidate_cols(feature: Feature, train: pd.DataFrame,
                    valid: pd.DataFrame, ctx: dict):
    tower = (feature.provenance or {}).get("tower", "wide")
    ctr = feature.compute(train, ctx).astype("float32").fillna(0) \
        .to_numpy().reshape(-1, 1)
    cva = feature.compute(valid, ctx).astype("float32").fillna(0) \
        .to_numpy().reshape(-1, 1)
    return tower, ctr, cva


def ablate_torch(train: pd.DataFrame, valid: pd.DataFrame, stats: dict,
                 feature: Feature, ctx: dict, baseline_auc: float,
                 baseline_ll: float, prep: dict) -> dict:
    """Two-tower ablation: route the candidate to its tower, retrain."""
    from two_tower import train_two_tower

    tower, ctr, cva = _candidate_cols(feature, train, valid, ctx)
    print(f"   routing '{feature.name}' -> {tower} tower")
    res = train_two_tower(prep, extra={tower: (ctr, cva)}, verbose=False)
    return {
        "auc": round(res["auc"], 4),
        "log_loss": round(res["log_loss"], 4),
        "delta_auc": round(res["auc"] - baseline_auc, 4),
        "delta_log_loss": round(res["log_loss"] - baseline_ll, 4),
        "backend": "two_tower",
        "tower": tower,
        "d_user": res["d_user"],
        "d_movie": res["d_movie"],
        "d_wide": res["d_wide"],
    }


def ablate_lgbm(train: pd.DataFrame, valid: pd.DataFrame, stats: dict,
                feature: Feature, ctx: dict, baseline_auc: float,
                baseline_ll: float,
                X_train_base: pd.DataFrame | None = None,
                X_valid_base: pd.DataFrame | None = None) -> dict:
    """LightGBM ablation: flat feature matrix + candidate column."""
    if X_train_base is None:
        X_train_base = featurize(train, stats)
    if X_valid_base is None:
        X_valid_base = featurize(valid, stats)

    X_train = X_train_base.copy()
    X_valid = X_valid_base.copy()
    X_train[feature.name] = feature.compute(train, ctx).astype("float32") \
        .fillna(0).to_numpy()
    X_valid[feature.name] = feature.compute(valid, ctx).astype("float32") \
        .fillna(0).to_numpy()

    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()

    model = make_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1]

    auc = float(roc_auc_score(y_valid, proba))
    ll = float(log_loss(y_valid, proba))
    return {
        "auc": round(auc, 4),
        "log_loss": round(ll, 4),
        "delta_auc": round(auc - baseline_auc, 4),
        "delta_log_loss": round(ll - baseline_ll, 4),
        "backend": "lgbm",
        "n_features": int(X_train.shape[1]),
    }


def ablate(train, valid, stats, feature, ctx, baseline_auc, baseline_ll,
           prep=None, X_train_base=None, X_valid_base=None,
           model: str = "torch") -> dict:
    if model == "torch":
        if prep is None:
            raise ValueError("two-tower ablation needs prep=prepare_tower_inputs(...)")
        return ablate_torch(train, valid, stats, feature, ctx,
                            baseline_auc, baseline_ll, prep)
    if model == "lgbm":
        return ablate_lgbm(train, valid, stats, feature, ctx, baseline_auc,
                           baseline_ll, X_train_base, X_valid_base)
    raise ValueError(f"unknown model backend: {model}")
