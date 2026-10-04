"""Baseline model: predict P(liked) for MovieLens-1M.

Features are deliberately simple and fast -- the point of the baseline is to
give the Feature Scientist agent something to beat, and to establish the
train-period-only discipline every later feature must follow:

  * user aggregates (count, mean rating) -- computed on TRAIN rows only,
    mapped onto both train and valid; unseen users fall back to global stats
  * movie aggregates (count, mean rating) -- same train-only discipline
  * genre one-hots (static movie metadata -- no leakage possible)
  * user demographics: gender / age / occupation (static -- no leakage)

Model: LightGBM binary classifier (falls back to sklearn's
HistGradientBoostingClassifier if LightGBM is unavailable).
Reports validation AUC and log-loss. Trains in ~1-2 minutes on CPU.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from data import load_all
from split import temporal_split

TARGET = "liked"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "baseline_metrics.json"
)


def build_train_only_stats(train: pd.DataFrame) -> dict:
    """Aggregate lookup tables built STRICTLY from train-period rows."""
    g = train.groupby("user_id")["rating"]
    user_count = g.count()
    user_mean = g.mean()

    g = train.groupby("movie_id")["rating"]
    movie_count = g.count()
    movie_mean = g.mean()

    return {
        "user_count": user_count,
        "user_mean": user_mean,
        "movie_count": movie_count,
        "movie_mean": movie_mean,
        "global_mean": float(train["rating"].mean()),
    }


def featurize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Build the baseline feature matrix for any dataframe (train or valid)."""
    X = pd.DataFrame(index=df.index)

    # --- train-period-only aggregates (unseen ids -> global fallback) ---
    X["user_rating_count"] = df["user_id"].map(stats["user_count"]).fillna(0)
    X["user_mean_rating"] = df["user_id"].map(stats["user_mean"]).fillna(stats["global_mean"])
    X["movie_rating_count"] = df["movie_id"].map(stats["movie_count"]).fillna(0)
    X["movie_mean_rating"] = df["movie_id"].map(stats["movie_mean"]).fillna(stats["global_mean"])

    # --- static side information: no leakage possible ---
    X["gender_m"] = (df["gender"] == "M").astype("int8")
    # age / occupation are coded ints in users.dat; treat as categories
    X["age"] = pd.Categorical(df["age"]).codes.astype("int16")
    X["occupation"] = pd.Categorical(df["occupation"]).codes.astype("int16")

    # genre one-hots (18 genres in MovieLens-1M)
    genres = df["genres"].str.get_dummies(sep="|")
    for col in genres.columns:
        X[f"genre_{col}"] = genres[col].astype("int8")

    return X.astype("float32")


def make_model():
    """LightGBM if available, else sklearn HistGradientBoosting."""
    try:
        import lightgbm as lgb

        print("model: LightGBM")
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        print("model: sklearn HistGradientBoosting (LightGBM not installed)")
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=50, random_state=42,
        )


def main() -> dict:
    t0 = time.time()
    print("loading data...")
    df = load_all()
    train, valid, cutoff_ts = temporal_split(df)
    print(f"train={len(train):,} valid={len(valid):,}")

    print("building train-only aggregates...")
    stats = build_train_only_stats(train)

    print("featurizing...")
    X_train = featurize(train, stats)
    y_train = train[TARGET].to_numpy()
    X_valid = featurize(valid, stats)
    y_valid = valid[TARGET].to_numpy()
    print(f"features: {X_train.shape[1]} ({', '.join(X_train.columns[:6])}, ...)")

    print("training...")
    model = make_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, proba))
    ll = float(log_loss(y_valid, proba))
    elapsed = time.time() - t0

    print(f"\nvalidation AUC      = {auc:.4f}")
    print(f"validation log-loss = {ll:.4f}")
    print(f"elapsed             = {elapsed:.1f}s")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    metrics = {
        "valid_auc": round(auc, 4),
        "valid_log_loss": round(ll, 4),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_features": X_train.shape[1],
        "elapsed_s": round(elapsed, 1),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics written to {RESULTS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
