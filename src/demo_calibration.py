"""Standalone demo of the calibration screening stage.

Uses the EXISTING LightGBM baseline's validation predictions (Days 1-2
harness, AUC 0.7439) -- it never trains a model itself. Predictions are
cached at results/lgbm_valid_proba.npz; on first run the cache is generated
by the deterministic baseline procedure (random_state=42, reproduces the
committed metrics), then reused.

(The two-tower model is mid-migration in a parallel track, so the demo
uses the stable LightGBM baseline. The calibration screen is
model-agnostic: it only consumes (y_true, y_pred).)

Demonstrates:
  1. movie_age_days (accepted earlier at +0.0005 AUC) shows a significant
     calibration gap -- the screen detects the same signal as ablation,
     with zero retraining.
  2. A random-noise column shows no significant gap.
  3. Sparse case: movies bucketed by train-period rating count -- the
     cold-start diagnostic on the rare-movie segment.
  4. A sparse binary flag with an underpowered positive bucket reports
     "insufficient evidence", not "no signal".
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calibration import (
    calibration_screen,
    format_table,
    screen_candidates,
    screen_sparse_id,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results")
CACHE_PATH = os.path.join(RESULTS_DIR, "lgbm_valid_proba.npz")
DAY = 86400


def load_frames():
    from data import load_all
    from split import temporal_split
    df = load_all()
    train, valid, cutoff = temporal_split(df)
    return train, valid, cutoff


def load_or_generate_predictions(train, valid):
    """Return (y_valid, proba) for the validation frame.

    Loads the cache when present; otherwise runs the deterministic
    baseline training procedure ONCE (seed 42 -- reproduces the committed
    two_tower_baseline.json metrics) and caches the predictions.
    """
    if os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH)
        proba, y_valid = z["proba"], z["y_valid"]
        assert len(proba) == len(valid) == len(y_valid), \
            "cache/valid misalignment -- delete the cache and re-run"
        print(f"loaded cached validation predictions ({len(proba):,} rows)")
        return y_valid, proba

    print("no prediction cache -- running deterministic LightGBM baseline once...")
    from sklearn.metrics import roc_auc_score
    from baseline import TARGET, build_train_only_stats, featurize, make_model
    stats = build_train_only_stats(train)
    X_train = featurize(train, stats)
    y_train = train[TARGET].to_numpy()
    X_valid = featurize(valid, stats)
    y_valid = valid[TARGET].to_numpy()
    model = make_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, proba))
    print(f"  baseline valid AUC={auc:.4f}")
    assert abs(auc - 0.7439) < 0.002, \
        f"baseline drift: AUC={auc:.4f} != committed 0.7439"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(CACHE_PATH, proba=proba, y_valid=y_valid)
    print(f"cached predictions -> {CACHE_PATH}")
    return y_valid, proba


def movie_age_days(valid: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Replicates the release_ts logic from scientist.build_ctx (source of
    truth); kept local so this demo stays decoupled from the agent loop."""
    imdb = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "data", "imdb",
                                    "movie_enrichment.csv"))
    imdb_idx = imdb.set_index("movieId")
    ml_year = train.drop_duplicates("movie_id").set_index("movie_id")["title"] \
        .str.extract(r"\((\d{4})\)")[0].astype(float)
    release_ts = {}
    for mid in train["movie_id"].unique():
        y = None
        if mid in imdb_idx.index and pd.notna(imdb_idx.loc[mid, "startYear"]):
            y = float(imdb_idx.loc[mid, "startYear"])
        elif mid in ml_year.index and pd.notna(ml_year.loc[mid]):
            y = float(ml_year.loc[mid])
        if y is not None:
            release_ts[int(mid)] = int(
                pd.Timestamp(int(y), 1, 1, tz="UTC").timestamp())
    ages = (train["timestamp"] - train["movie_id"].map(release_ts)) / DAY
    median_age = float(ages[ages >= 0].median())
    out = (valid["timestamp"] - valid["movie_id"].map(release_ts)) / DAY
    return pd.Series(out.fillna(median_age).clip(lower=0),
                     index=valid.index, name="movie_age_days")


def main():
    train, valid, _ = load_frames()
    y_valid, proba = load_or_generate_predictions(train, valid)
    print(f"valid rows: {len(valid):,}, baseline AUC: "
          f"{roc_auc_score(y_valid, proba):.4f}")

    rng = np.random.default_rng(7)
    candidates = {
        "movie_age_days": movie_age_days(valid, train),
        "random_noise": pd.Series(rng.normal(size=len(valid)),
                                  index=valid.index),
    }
    # Sparse binary flag: ~0.01% positives -> underpowered positive bucket.
    rare = pd.Series((rng.random(len(valid)) < 0.0001).astype(int),
                     index=valid.index, name="rare_flag_synth")
    candidates["rare_flag_synth"] = rare

    results = screen_candidates(
        y_valid, proba, candidates,
        key_buckets={"rare_flag_synth": "1"})
    print("\n" + format_table(results) + "\n")
    for r in results:
        print(" *", r["summary"])

    # --- sparse-ID case: movies by train-period rating count ---------------
    train_counts = train.groupby("movie_id").size()
    train_last = train.groupby("movie_id")["timestamp"].max()
    sparse = screen_sparse_id(
        y_valid, proba, valid["movie_id"], train_counts, "movie_id",
        train_id_last_ts=train_last, valid_ts=valid["timestamp"])
    print("\n--- sparse: movie train-period popularity ---")
    print(" *", sparse["popularity"]["summary"])
    if sparse["recency"] is not None:
        print("--- sparse: movie recency (days since last train rating) ---")
        print(" *", sparse["recency"]["summary"])
    for n_ in sparse["notes"]:
        print("note:", n_)

    # --- checks -------------------------------------------------------------
    by_name = {r["feature"]: r for r in results}
    assert by_name["movie_age_days"]["verdict"] == "signal", \
        "movie_age_days should show a calibration gap (it gained +0.0005 AUC)"
    assert by_name["random_noise"]["verdict"] == "no_signal", \
        "noise column must not flag"
    assert by_name["rare_flag_synth"]["verdict"] == "insufficient_evidence", \
        "underpowered positive bucket must not read as 'no signal'"
    print("\nall demo checks passed")


if __name__ == "__main__":
    main()
