"""Error-slice analysis of the baseline model (Days 3-4).

Trains the baseline once, scores the validation ("future") period, then
breaks the errors down by slice:

  * user activity decile   (how many ratings the user gave in train)
  * movie popularity decile (how many ratings the movie got in train)
  * genre                   (per-genre membership -- a row can sit in many)
  * time                    (calendar month of the validation rating)

Writes results/error_slices.json and prints the worst slices. The
RuleBasedEngine in scientist.py reads this file to decide *what kind of
feature to invent next* -- e.g. "cold movies: AUC 0.61" triggers
cold-start / momentum hypotheses.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from baseline import TARGET, build_train_only_stats, featurize, make_model
from data import load_all
from split import temporal_split

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "error_slices.json"
)

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def train_baseline_predict(model_backend: str = "torch"):
    """Train baseline on the temporal split; return everything downstream needs.

    model_backend: "torch" (two-tower, default) or "lgbm" (LightGBM).
    Returns (train, valid, stats, model_info, valid_proba, cutoff_ts).
    """
    print("loading data + temporal split...")
    df = load_all()
    train, valid, cutoff_ts = temporal_split(df)

    print("building train-only stats + featurizing...")
    stats = build_train_only_stats(train)

    if model_backend == "torch":
        from two_tower import prepare_tower_inputs, train_two_tower
        prep = prepare_tower_inputs(train, valid, stats)
        print(f"training two-tower baseline "
              f"(user_in={prep['Xu_train'].shape[1]}, "
              f"movie_in={prep['Xm_train'].shape[1]})...")
        res = train_two_tower(prep)
        return train, valid, stats, {"backend": "torch", "prep": prep,
                                     "model": res["model"]}, \
            res["proba"], cutoff_ts

    X_train = featurize(train, stats)
    y_train = train[TARGET].to_numpy()
    X_valid = featurize(valid, stats)
    print("training baseline...")
    model = make_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1]
    return train, valid, stats, {"backend": "lgbm", "model": model}, \
        proba, cutoff_ts


def _slice_stats(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC / log-loss / size for one slice; None when the slice is degenerate."""
    n = len(y)
    out = {"n": int(n), "pos_rate": round(float(y.mean()), 4)}
    if n == 0 or len(np.unique(y)) < 2:
        out.update({"auc": None, "log_loss": None})
    else:
        out.update({
            "auc": round(float(roc_auc_score(y, p)), 4),
            "log_loss": round(float(log_loss(y, p)), 4),
        })
    return out


def analyze(train: pd.DataFrame, valid: pd.DataFrame, proba: np.ndarray,
            stats: dict) -> dict:
    """Compute per-slice metrics on the validation predictions."""
    y = valid[TARGET].to_numpy()
    frame = pd.DataFrame({"y": y, "p": proba})
    slices: dict[str, list[dict]] = {}

    # --- user activity decile (train-period rating count) ---
    ucount = valid["user_id"].map(stats["user_count"]).fillna(0).to_numpy()
    dec = pd.qcut(ucount, 10, labels=False, duplicates="drop")
    rows = []
    for d in sorted(pd.Series(dec).dropna().unique()):
        m = dec == d
        s = _slice_stats(y[m], proba[m])
        s["slice"] = f"user_activity_decile_{int(d)}"
        rows.append(s)
    slices["user_activity"] = rows

    # --- movie popularity decile (train-period rating count) ---
    mcount = valid["movie_id"].map(stats["movie_count"]).fillna(0).to_numpy()
    dec = pd.qcut(mcount, 10, labels=False, duplicates="drop")
    rows = []
    for d in sorted(pd.Series(dec).dropna().unique()):
        m = dec == d
        s = _slice_stats(y[m], proba[m])
        s["slice"] = f"movie_popularity_decile_{int(d)}"
        rows.append(s)
    slices["movie_popularity"] = rows

    # --- genre (membership; rows appear in every genre they carry) ---
    rows = []
    for g in GENRES:
        m = valid["genres"].str.contains(g, regex=False).to_numpy()
        if m.sum() == 0:
            continue
        s = _slice_stats(y[m], proba[m])
        s["slice"] = f"genre_{g}"
        rows.append(s)
    slices["genre"] = rows

    # --- time (calendar month of the validation rating) ---
    month = valid["datetime"].dt.to_period("M").astype(str).to_numpy()
    rows = []
    for mo in sorted(np.unique(month)):
        m = month == mo
        s = _slice_stats(y[m], proba[m])
        s["slice"] = f"month_{mo}"
        rows.append(s)
    slices["time"] = rows

    # --- worst slices overall (n >= 2000, ranked by AUC) ---
    flat = [s for group in slices.values() for s in group
            if s["auc"] is not None and s["n"] >= 2000]
    worst = sorted(flat, key=lambda s: s["auc"])[:8]
    return {"slices": slices, "worst": worst,
            "overall": _slice_stats(y, proba)}


def main(model_backend: str = "torch") -> dict:
    train, valid, stats, model_info, proba, cutoff_ts = \
        train_baseline_predict(model_backend)
    result = analyze(train, valid, proba, stats)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== worst error slices (validation) ===")
    for s in result["worst"]:
        print(f"  {s['slice']:<32} n={s['n']:<7,} auc={s['auc']:.4f} "
              f"pos_rate={s['pos_rate']:.3f}")
    print(f"\noverall valid auc={result['overall']['auc']:.4f}")
    print(f"slices written to {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="torch", choices=["torch", "lgbm"])
    args = ap.parse_args()
    main(args.model)
