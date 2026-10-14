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
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from baseline import TARGET, featurize, make_model, make_regressor, \
    transform_target
from features import Feature


def _feature_cols(feature: Feature, train: pd.DataFrame,
                    valid: pd.DataFrame, ctx: dict):
    """(tower, name, train_col_2d, valid_col_2d) for one Feature."""
    tower = (feature.provenance or {}).get("tower", "wide")
    ctr = feature.compute(train, ctx).astype("float32").fillna(0) \
        .to_numpy().reshape(-1, 1)
    cva = feature.compute(valid, ctx).astype("float32").fillna(0) \
        .to_numpy().reshape(-1, 1)
    return tower, feature.name, ctr, cva


def _extra_for(features: list[Feature], train: pd.DataFrame,
               valid: pd.DataFrame, ctx: dict) -> dict:
    """Build the multi-feature `extra` spec: {tower: [(name, ctr, cva)]}."""
    extra: dict = {}
    for f in features:
        tower, name, ctr, cva = _feature_cols(f, train, valid, ctx)
        extra.setdefault(tower, []).append((name, ctr, cva))
    return extra


def _candidate_cols(feature: Feature, train: pd.DataFrame,
                    valid: pd.DataFrame, ctx: dict):
    tower = (feature.provenance or {}).get("tower", "wide")
    ctr = feature.compute(train, ctx).astype("float32").fillna(0) \
        .to_numpy().reshape(-1, 1)
    cva = feature.compute(valid, ctx).astype("float32").fillna(0) \
        .to_numpy().reshape(-1, 1)
    return tower, ctr, cva


def check_degenerate(ctr: np.ndarray, name: str,
                     std_tol: float = 1e-9) -> str | None:
    """Reject features that are (near-)constant on train.

    A train-constant feature is unlearnable: the model cannot fit a weight
    for a column with no variance, and the per-feature standardizer divides
    by ~zero, which can diverge training (observed: AUC 0.53, ll 3.36).
    Returns a rejection reason, or None if the column is viable.
    """
    std = float(np.std(ctr))
    if not np.isfinite(std) or std < std_tol:
        return (f"degenerate train column (std={std:.2e}): constant on train, "
                f"unlearnable — no weight can be fit")
    if not np.all(np.isfinite(ctr)):
        return "non-finite values in train column (inf/nan survived fillna)"
    return None


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
                X_valid_base: pd.DataFrame | None = None,
                target: str | None = None,
                metric: str = "auc", baseline_metric: float | None = None,
                target_transform: str | None = None) -> dict:
    """LightGBM ablation: flat feature matrix + candidate column.

    metric="auc" (default): binary classification, AUC/log-loss deltas.
    metric="logloss": multiclass (or binary) classification gated on the
      log-loss delta -- the text competitions' metric. predict_proba's
      columns align with encode_labels()' sorted-unique classes.
    metric="rmse": regression -- LGBMRegressor on the (optionally
    transformed) target, RMSE delta. baseline_metric (the matching
    baseline's RMSE, in the same transformed space) is required; the
    comparison is only honest against the identical baseline.
    """
    _target = target or TARGET
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

    y_train_raw = train[_target].to_numpy()
    y_valid_raw = valid[_target].to_numpy()

    if metric == "logloss":
        if baseline_metric is None:
            raise ValueError(
                "logloss ablation needs baseline_metric= (the matching "
                "baseline's log-loss)")
        from baseline import encode_labels
        y_train, y_valid, classes = encode_labels(y_train_raw, y_valid_raw)
        n_classes = len(classes)
        model = make_model(n_classes if n_classes > 2 else None)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)
        ll = float(log_loss(y_valid, proba, labels=list(range(n_classes))))
        return {
            "metric": "logloss",
            "log_loss": round(ll, 4),
            "delta_log_loss": round(ll - baseline_metric, 4),
            "baseline_log_loss": round(baseline_metric, 4),
            "n_classes": n_classes,
            "backend": "lgbm",
            "n_features": int(X_train.shape[1]),
        }

    if metric == "rmse":
        if baseline_metric is None:
            raise ValueError(
                "regression ablation needs baseline_metric= (the matching "
                "baseline's RMSE in the same target space)")
        y_train = transform_target(y_train_raw, target_transform)
        y_valid = transform_target(y_valid_raw, target_transform)
        model = make_regressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
        return {
            "metric": "rmse",
            "rmse": round(rmse, 4),
            "delta_rmse": round(rmse - baseline_metric, 4),
            "baseline_rmse": round(baseline_metric, 4),
            "target_transform": target_transform,
            "backend": "lgbm",
            "n_features": int(X_train.shape[1]),
        }

    y_train = y_train_raw
    y_valid = y_valid_raw
    # Encode once (sorted-unique) so binary string targets work and the
    # encoding matches train_baseline_predict's.
    from baseline import encode_labels as _enc
    y_train, y_valid, _ = _enc(y_train, y_valid)
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
           model: str = "torch", target: str | None = None,
           metric: str = "auc", baseline_metric: float | None = None,
           target_transform: str | None = None) -> dict:
    if model == "torch":
        if prep is None:
            raise ValueError("two-tower ablation needs prep=prepare_tower_inputs(...)")
        return ablate_torch(train, valid, stats, feature, ctx,
                            baseline_auc, baseline_ll, prep)
    if model == "lgbm":
        return ablate_lgbm(train, valid, stats, feature, ctx, baseline_auc,
                           baseline_ll, X_train_base, X_valid_base,
                           target=target, metric=metric,
                           baseline_metric=baseline_metric,
                           target_transform=target_transform)
    raise ValueError(f"unknown model backend: {model}")


def ablate_cumulative(train: pd.DataFrame, valid: pd.DataFrame, stats: dict,
                      accepted: list[Feature], candidate: Feature, ctx: dict,
                      base_auc: float, base_ll: float,
                      champion_auc: float, champion_ll: float,
                      prep: dict | None = None,
                      X_train_base: pd.DataFrame | None = None,
                      X_valid_base: pd.DataFrame | None = None,
                      model: str = "torch", seed: int = 42,
                      verbose: bool = False, target: str | None = None,
                      **train_kwargs) -> dict:
    """Cumulative ablation: champion (= baseline + accepted) + candidate.

    Trains ONE model (baseline + accepted + candidate) and reports both:
      * marginal deltas vs the champion (baseline + accepted), whose
        single-seed metrics are carried in champion_auc/champion_ll --
        no retraining of the champion needed;
      * total deltas vs the base baseline.

    The driver carries champion metrics forward: when a candidate is
    accepted, this run's auc/log_loss become the next champion_auc/ll.
    """
    features = list(accepted) + [candidate]
    if model == "torch":
        from two_tower import train_two_tower
        if prep is None:
            raise ValueError(
                "two-tower cumulative ablation needs prep=prepare_tower_inputs(...)")
        extra = _extra_for(features, train, valid, ctx)
        if verbose:
            towers = {t: [n for n, _, _ in v] for t, v in extra.items()}
            print(f"   champion stack ({len(features)}): {towers}")
        res = train_two_tower(prep, extra=extra, seed=seed, verbose=verbose,
                              **train_kwargs)
        auc, ll = float(res["auc"]), float(res["log_loss"])
        return {
            "auc": round(auc, 4),
            "log_loss": round(ll, 4),
            "proba": res["proba"],
            "marginal_delta_auc": round(auc - champion_auc, 4),
            "marginal_delta_ll": round(ll - champion_ll, 4),
            "total_delta_auc": round(auc - base_auc, 4),
            "total_delta_ll": round(ll - base_ll, 4),
            "champion_auc": round(champion_auc, 4),
            "champion_ll": round(champion_ll, 4),
            "n_accepted": len(accepted),
            "backend": "two_tower",
            "d_user": res["d_user"],
            "d_movie": res["d_movie"],
            "d_wide": res["d_wide"],
        }
    if model == "lgbm":
        _target = target or TARGET
        if X_train_base is None:
            X_train_base = featurize(train, stats)
        if X_valid_base is None:
            X_valid_base = featurize(valid, stats)
        X_train = X_train_base.copy()
        X_valid = X_valid_base.copy()
        for f in features:
            X_train[f.name] = f.compute(train, ctx).astype("float32") \
                .fillna(0).to_numpy()
            X_valid[f.name] = f.compute(valid, ctx).astype("float32") \
                .fillna(0).to_numpy()
        y_train = train[_target].to_numpy()
        y_valid = valid[_target].to_numpy()
        m = make_model()
        m.fit(X_train, y_train)
        proba = m.predict_proba(X_valid)[:, 1]
        auc = float(roc_auc_score(y_valid, proba))
        ll = float(log_loss(y_valid, proba))
        return {
            "auc": round(auc, 4),
            "log_loss": round(ll, 4),
            "proba": proba,
            "marginal_delta_auc": round(auc - champion_auc, 4),
            "marginal_delta_ll": round(ll - champion_ll, 4),
            "total_delta_auc": round(auc - base_auc, 4),
            "total_delta_ll": round(ll - base_ll, 4),
            "champion_auc": round(champion_auc, 4),
            "champion_ll": round(champion_ll, 4),
            "n_accepted": len(accepted),
            "backend": "lgbm",
            "n_features": int(X_train.shape[1]),
        }
    raise ValueError(f"unknown model backend: {model}")


def evaluate_champion(train: pd.DataFrame, valid: pd.DataFrame, stats: dict,
                      accepted: list[Feature], ctx: dict,
                      prep: dict | None = None,
                      X_train_base: pd.DataFrame | None = None,
                      X_valid_base: pd.DataFrame | None = None,
                      model: str = "torch",
                      seeds: tuple[int, ...] = (42, 43, 44),
                      verbose: bool = False, **train_kwargs) -> dict:
    """Multi-seed evaluation of the champion (baseline + accepted).

    Trains one model per seed and reports mean +/- std for AUC and log-loss.
    Used at each champion milestone so the cumulative claim carries error
    bars; day-to-day ablations stay single-seed for speed.
    """
    aucs, lls, probas = [], [], []
    for s in seeds:
        if model == "torch":
            from two_tower import train_two_tower
            if prep is None:
                raise ValueError("two-tower champion eval needs prep=...")
            extra = _extra_for(accepted, train, valid, ctx)
            res = train_two_tower(prep, extra=extra, seed=s, verbose=False,
                                  **train_kwargs)
            aucs.append(float(res["auc"]))
            lls.append(float(res["log_loss"]))
            probas.append(res["proba"])
        elif model == "lgbm":
            if X_train_base is None:
                X_train_base = featurize(train, stats)
            if X_valid_base is None:
                X_valid_base = featurize(valid, stats)
            X_train = X_train_base.copy()
            X_valid = X_valid_base.copy()
            for f in accepted:
                X_train[f.name] = f.compute(train, ctx).astype("float32") \
                    .fillna(0).to_numpy()
                X_valid[f.name] = f.compute(valid, ctx).astype("float32") \
                    .fillna(0).to_numpy()
            y_train = train[TARGET].to_numpy()
            y_valid = valid[TARGET].to_numpy()
            m = make_model()
            # LightGBM is deterministic given data; emulate seeds by
            # bootstrap-resampling the training rows.
            rng = np.random.RandomState(s)
            idx = rng.choice(len(X_train), len(X_train), replace=True)
            m.fit(X_train.iloc[idx], y_train[idx])
            proba = m.predict_proba(X_valid)[:, 1]
            aucs.append(float(roc_auc_score(y_valid, proba)))
            lls.append(float(log_loss(y_valid, proba)))
            probas.append(proba)
        else:
            raise ValueError(f"unknown model backend: {model}")
        if verbose:
            print(f"   seed {s}: AUC={aucs[-1]:.4f} ll={lls[-1]:.4f}")
    aucs = np.array(aucs)
    lls = np.array(lls)
    return {
        "seeds": list(seeds),
        "auc_mean": round(float(aucs.mean()), 4),
        "auc_std": round(float(aucs.std(ddof=1)), 4) if len(aucs) > 1 else 0.0,
        "ll_mean": round(float(lls.mean()), 4),
        "ll_std": round(float(lls.std(ddof=1)), 4) if len(lls) > 1 else 0.0,
        "auc_seeds": [round(float(a), 4) for a in aucs],
        "ll_seeds": [round(float(l), 4) for l in lls],
        "proba_mean": np.mean(probas, axis=0),
        "n_accepted": len(accepted),
        "backend": model,
    }
