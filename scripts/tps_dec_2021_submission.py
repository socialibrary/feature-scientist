#!/usr/bin/env python3
"""Build the MLE-bench tps-dec-2021 submission.

Multiclass (7 forest cover types), accuracy metric. The loop accepted
0 features (baseline logloss 0.1462 already excellent), so this is a
plain LightGBM multiclass baseline retrained on the full 3.6M train rows.
Writes submission.csv, grades locally with accuracy_score.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from baseline import make_model

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "kaggle", "tps-dec-2021")
CACHE = os.path.expanduser("~/.cache/mle-bench/data/tabular-playground-series-dec-2021/prepared")

TARGET = "Cover_Type"

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
FEATURE_COLS = [c for c in train.columns if c not in ("Id", TARGET)]
print(f"train {train.shape}, test {test.shape}, {len(FEATURE_COLS)} feature cols", flush=True)


def build_matrix(df, medians, cat_levels):
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s.dtype):
            X[col] = s.fillna(medians[col]).astype("float32")
        else:
            s_str = s.fillna("missing").astype(str)
            for cat in cat_levels.get(col, []):
                X[f"{col}_{cat}"] = (s_str == cat).astype("int8")
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

# Encode labels 0..K-1 via sorted unique (matches LightGBM column order)
classes = np.sort(train[TARGET].unique())
y_train = pd.Categorical(train[TARGET], categories=classes).codes.astype(int)
print(f"classes: {classes}", flush=True)

model = make_model(len(classes))
print("training...", flush=True)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
pred_codes = proba.argmax(axis=1)
preds = classes[pred_codes]

sub = pd.DataFrame({"Id": test["Id"].values, "Cover_Type": preds})
out = os.path.join(BASE, "results", "tabular-playground-series-dec-2021_submission.csv")
sub.to_csv(out, index=False)
print(f"wrote {out}", flush=True)

from sklearn.metrics import accuracy_score
answers = pd.read_csv(os.path.join(CACHE, "private", "test.csv"))
merged = sub.merge(answers[["Id", TARGET]], on="Id", suffixes=("", "_true"))
score = accuracy_score(merged[f"{TARGET}_true"], merged[TARGET])
print(f"\nlocal accuracy: {score:.4f}")
