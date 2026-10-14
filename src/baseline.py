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


def make_model(n_classes: int | None = None):
    """LightGBM if available, else sklearn HistGradientBoosting.

    Binary by default. Pass n_classes (> 2) for multiclass briefs
    (text competitions like spooky-author); LightGBM sorts classes, so
    targets must be encoded with encode_labels() (sorted-unique) for the
    predict_proba columns to line up.
    """
    if n_classes is not None and n_classes > 2:
        try:
            import lightgbm as lgb

            print(f"model: LightGBM (multiclass, {n_classes} classes)")
            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
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

            print("model: sklearn HistGradientBoosting "
                  f"(multiclass, {n_classes} classes)")
            return HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                min_samples_leaf=50, random_state=42,
            )
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


def make_regressor():
    """LightGBM regressor if available, else sklearn HistGradientBoosting.

    Mirrors make_model()'s hyperparameters so the classification and
    regression baselines are configured symmetrically; the generic-tabular
    path (briefs) trains this on the brief's target and reports RMSE.
    """
    try:
        import lightgbm as lgb

        print("model: LightGBM regressor")
        return lgb.LGBMRegressor(
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
        from sklearn.ensemble import HistGradientBoostingRegressor

        print("model: sklearn HistGradientBoostingRegressor "
              "(LightGBM not installed)")
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=50, random_state=42,
        )


def encode_labels(y_train_raw, y_valid_raw=None):
    """Encode classification labels as 0..K-1 via sorted unique values.

    LightGBM's sklearn wrapper sorts classes internally, so this encoding
    keeps predict_proba columns aligned with the classes for both binary
    and multiclass targets. Returns (y_train, y_valid_or_None, classes).
    """
    classes = np.sort(pd.unique(pd.Series(y_train_raw).dropna()))
    y_train = pd.Categorical(y_train_raw, categories=classes).codes
    y_valid = None
    if y_valid_raw is not None:
        y_valid = pd.Categorical(y_valid_raw, categories=classes).codes
        if (y_valid == -1).any():
            raise ValueError("validation target has classes unseen in train")
    return y_train.astype(int), (None if y_valid is None else y_valid.astype(int)), classes


def transform_target(y, kind: str | None):
    """Apply the brief's target transform before train/eval.

    The metric is computed in the transformed space, so e.g. metric=rmse
    with target_transform=log1p is exactly the Kaggle house-price metric
    (RMSE of log predictions). "none"/None leaves the target untouched.
    """
    import numpy as np

    if not kind or str(kind).lower() in ("none", "identity"):
        return np.asarray(y, dtype=float)
    if str(kind).lower() == "log1p":
        y = np.asarray(y, dtype=float)
        if np.any(y <= -1):
            raise ValueError("log1p target transform needs all y > -1")
        return np.log1p(y)
    raise ValueError(f"unknown target_transform: {kind!r}")


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


# Text columns: free-text fields are useless as top-20 one-hots, so the
# generic path vectorizes them with TF-IDF (train-fit vocabulary, same
# train-only discipline as the numeric medians and categorical levels).
TFIDF_MAX_FEATURES = 200


def detect_text_columns(df: pd.DataFrame, target: str,
                        min_avg_len: float = 30.0,
                        min_avg_tokens: float = 4.0) -> list[str]:
    """Heuristic: object columns that look like free text, not categoricals.

    A column counts as text when its values are long on average (>= 30
    chars) or multi-token (>= 4 words). Short-string columns stay on the
    categorical one-hot path. Overridable per-brief via `text_columns`.
    """
    text: list[str] = []
    for col in df.columns:
        if col == target or col.lower() in ("id", "passengerid"):
            continue
        s = df[col]
        # pandas 3 infers `str` dtype for string columns (not object).
        if not pd.api.types.is_string_dtype(s.dtype):
            continue
        s_str = s.dropna().astype(str)
        if len(s_str) == 0:
            continue
        avg_len = float(s_str.str.len().mean())
        avg_tokens = float(s_str.str.split().str.len().mean())
        if avg_len >= min_avg_len or avg_tokens >= min_avg_tokens:
            text.append(col)
    return text


def _fit_tfidf(docs: pd.Series):
    """Fit a TF-IDF vectorizer on train documents (train-only discipline).

    Short-token columns (avg < 2 tokens, e.g. text-normalization "before"
    tokens) get character n-grams; real documents get word n-grams.
    Returns None when the vocabulary is empty (falls back to categorical).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    s = docs.fillna("").astype(str)
    avg_tokens = float(s.str.split().str.len().mean()) if len(s) else 0.0
    if avg_tokens < 2.0:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                              max_features=TFIDF_MAX_FEATURES,
                              sublinear_tf=True, min_df=2)
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2),
                              max_features=TFIDF_MAX_FEATURES,
                              sublinear_tf=True, stop_words="english",
                              min_df=2)
    try:
        vec.fit(s)
    except ValueError:
        return None  # empty vocabulary
    if not vec.vocabulary_:
        return None
    return vec


def fit_tabular_schema(train: pd.DataFrame, target: str,
                       text_columns: list[str] | None = None,
                       exclude: tuple | list = ()) -> dict:
    """Learn the generic-tabular feature schema from TRAIN rows only.

    The train-only discipline matters here: one-hot levels, numeric
    medians and TF-IDF vocabularies must be learned on train and applied
    unchanged to valid. Learning them per-frame gives train/valid
    different columns (observed: 296 train features vs 256 valid features
    -> LightGBM shape-check error) and leaks valid statistics into the
    features.

    text_columns: explicit text columns (None = auto-detect via
    detect_text_columns). exclude: extra columns dropped from features
    entirely -- used for sibling target columns in multi-target
    competitions (the nomad leakage lesson: never let one target be a
    feature for another).
    """
    exclude = set(exclude or ())
    if text_columns is None:
        text_columns = detect_text_columns(train, target)
    text_columns = [c for c in text_columns if c not in exclude]
    schema: dict = {"categorical": {}, "numeric_median": {},
                    "text": {}, "text_columns": list(text_columns)}
    for col in train.columns:
        if col == target or col in exclude:
            continue
        if col.lower() in ("id", "passengerid", "key"):
            continue
        s = train[col]
        if col in text_columns:
            vec = _fit_tfidf(s)
            if vec is not None:
                schema["text"][col] = vec
            else:
                # Degenerate text (empty vocab): categorical fallback.
                vc = s.fillna("missing").astype(str).value_counts()
                schema["categorical"][col] = vc.head(20).index.tolist()
            continue
        if pd.api.types.is_numeric_dtype(s.dtype):
            schema["numeric_median"][col] = float(s.median())
        else:
            vc = s.fillna("missing").astype(str).value_counts()
            schema["categorical"][col] = vc.head(20).index.tolist()
    return schema


def featurize_tabular(df: pd.DataFrame, target: str,
                      schema: dict | None = None,
                      exclude: tuple | list = ()) -> pd.DataFrame:
    """Generic tabular featurizer: numeric passthrough + categorical one-hot
    + TF-IDF text columns.

    Excludes the target column, ID-like columns, and any `exclude` columns
    (sibling targets). With schema=None the schema is learned from df
    itself (training frame only); pass the train-fit schema
    (fit_tabular_schema) for validation frames so the columns always
    match the baseline's. Categories unseen in train are simply all-zero
    -- no leakage, no shape mismatch. Text terms unseen in train are
    ignored by the fitted vocabulary -- same discipline.
    """
    exclude = set(exclude or ())
    if schema is None:
        schema = fit_tabular_schema(df, target, exclude=exclude)
    text_vecs = schema.get("text", {})
    X = pd.DataFrame(index=df.index)
    tfidf_blocks = []
    for col in df.columns:
        if col == target or col in exclude:
            continue
        if col.lower() in ("id", "passengerid", "key"):
            continue
        s = df[col]
        if col in text_vecs:
            docs = s.fillna("").astype(str)
            mat = text_vecs[col].transform(docs).toarray().astype("float32")
            terms = text_vecs[col].get_feature_names_out()
            # One concat, not per-column insert (avoids DataFrame
            # fragmentation warnings at 100+ TF-IDF columns).
            tfidf_blocks.append(pd.DataFrame(
                mat, index=df.index,
                columns=[f"tfidf_{col}_{t}" for t in terms],
                dtype="float32"))
            continue
        if pd.api.types.is_numeric_dtype(s.dtype):
            med = schema["numeric_median"].get(col, float(s.median()))
            X[col] = s.fillna(med).astype("float32")
        else:
            top = schema["categorical"].get(col)
            if top is None:  # novel column; learn locally as a fallback
                vc = s.fillna("missing").astype(str).value_counts()
                top = vc.head(20).index.tolist()
            s_str = s.fillna("missing").astype(str)
            for cat in top:
                # Sanitize: LightGBM rejects JSON-special chars in names.
                _safe = "".join(c if (c.isalnum() or c == "_") else "_"
                                for c in f"{col}_{cat}")
                X[_safe] = (s_str == cat).astype("int8")
    if tfidf_blocks:
        X = pd.concat([X] + tfidf_blocks, axis=1)
    return X.astype("float32")
