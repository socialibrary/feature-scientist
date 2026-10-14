#!/usr/bin/env python3
"""Build the MLE-bench tps-may-2022 submission.

Binary classification (AUC): baseline + 1 accepted feature
(f_00_x_f_01_interaction, +0.0011 AUC). Retrained on full train data.
Writes submission.csv, grades locally with roc_auc_score.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from baseline import make_model

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "kaggle", "tps-may-2022")
CACHE = os.path.expanduser("~/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared")

TARGET = "target"

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
FEATURE_COLS = [c for c in train.columns if c not in ("id", TARGET)]
print(f"train {train.shape}, test {test.shape}, {len(FEATURE_COLS)} feature cols")

with open(os.path.join(BASE, "results", "champion_tps_may_2022.pkl"), "rb") as f:
    champ = pickle.load(f)
ACCEPTED = champ["accepted_feature_code"]
print(f"accepted features: {sorted(ACCEPTED)}")


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

X_train = build_matrix(train, medians, cat_levels)
X_test = build_matrix(test, medians, cat_levels)
for name, code in ACCEPTED.items():
    ns = {"pd": pd, "np": np}
    exec(code, ns)
    X_train[name] = ns["compute"](train, {}).astype("float32").fillna(0).to_numpy()
    X_test[name] = ns["compute"](test, {}).astype("float32").fillna(0).to_numpy()
print(f"matrix: {X_train.shape[1]} cols")

model = make_model()
model.fit(X_train, train[TARGET].to_numpy())
proba = model.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"id": test["id"].values, "target": proba})
out = os.path.join(BASE, "results", "tabular-playground-series-may-2022_submission.csv")
sub.to_csv(out, index=False)
print(f"wrote {out}")

from sklearn.metrics import roc_auc_score
answers = pd.read_csv(os.path.join(CACHE, "private", "test.csv"))
merged = sub.merge(answers[["id", TARGET]], on="id", suffixes=("", "_true"))
score = roc_auc_score(merged[f"{TARGET}_true"], merged[TARGET])
print(f"\nlocal AUC: {score:.4f}")
