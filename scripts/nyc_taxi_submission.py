#!/usr/bin/env python3
"""Build the MLE-bench nyc-taxi submission.

Regression (RMSE on fare_amount). The loop accepted 0 features (2 leaky
proposals correctly rejected by the target-leakage guard), so this is a
plain LightGBM baseline. Writes submission.csv, grades locally with RMSE.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from baseline import make_regressor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "kaggle", "nyc-taxi")
CACHE = os.path.expanduser("~/.cache/mle-bench/data/new-york-city-taxi-fare-prediction/prepared")

TARGET = "fare_amount"
ID_COL = "key"

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
FEATURE_COLS = [c for c in train.columns if c not in (ID_COL, TARGET)]
print(f"train {train.shape}, test {test.shape}, {len(FEATURE_COLS)} feature cols", flush=True)


def build_matrix(df, medians, cat_levels):
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s.dtype):
            X[col] = s.fillna(medians[col]).astype("float32")
        else:
            # Sanitize names (LightGBM rejects JSON-special chars).
            s_str = s.fillna("missing").astype(str)
            for cat in cat_levels.get(col, []):
                safe = "".join(c if (c.isalnum() or c == "_") else "_"
                               for c in f"{col}_{cat}")
                X[safe] = (s_str == cat).astype("int8")
    return X.astype("float32")


medians = {c: float(train[c].median()) for c in FEATURE_COLS
           if pd.api.types.is_numeric_dtype(train[c].dtype)}
cat_levels = {}
for c in FEATURE_COLS:
    if not pd.api.types.is_numeric_dtype(train[c].dtype):
        vc = train[c].fillna("missing").astype(str).value_counts()
        cat_levels[c] = vc.head(20).index.tolist()

print("building matrices...", flush=True)
X_train = build_matrix(train, medians, cat_levels)
X_test = build_matrix(test, medians, cat_levels)
print(f"matrix: {X_train.shape[1]} cols", flush=True)

model = make_regressor()
print("training...", flush=True)
model.fit(X_train, train[TARGET].to_numpy())
preds = model.predict(X_test)
# Fares can't be negative; clip.
preds = np.maximum(preds, 0)

sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: preds})
out = os.path.join(BASE, "results", "new-york-city-taxi-fare-prediction_submission.csv")
sub.to_csv(out, index=False)
print(f"wrote {out}", flush=True)

from sklearn.metrics import root_mean_squared_error
answers = pd.read_csv(os.path.join(CACHE, "private", "test.csv"))
merged = sub.merge(answers[[ID_COL, TARGET]], on=ID_COL, suffixes=("", "_true"))
score = root_mean_squared_error(merged[f"{TARGET}_true"], merged[TARGET])
print(f"\nlocal RMSE: {score:.4f}")
