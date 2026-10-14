#!/usr/bin/env python3
"""Build the MLE-bench nomad2018 submission from Feature Scientist outputs.

Target-leakage discipline: neither target is ever a feature for the other
(the saved bandgap champion used formation_energy as its #1 feature, which
is unavailable at submission time, so we retrain without it).

- bandgap_energy_ev: baseline + 2 accepted features (spacegroup interactions)
- formation_energy_ev_natom: plain baseline (0 accepted; honest)
Writes submission.csv, then grades locally with mlebench's grade.py.
"""
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from baseline import make_regressor, transform_target

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "kaggle", "nomad2018")
CACHE = os.path.expanduser("~/.cache/mle-bench/data/nomad2018-predict-transparent-conductors/prepared")

TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
FEATURE_COLS = [c for c in train.columns if c not in ("id", *TARGETS)]
print(f"train {train.shape}, test {test.shape}, {len(FEATURE_COLS)} feature cols")

# Accepted feature code from the bandgap brief (re-executed, not re-derived).
with open(os.path.join(BASE, "results", "champion_nomad2018_bandgap.pkl"), "rb") as f:
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

X_train_base = build_matrix(train, medians, cat_levels)
X_test_base = build_matrix(test, medians, cat_levels)
print(f"base matrix: {X_train_base.shape[1]} cols")


def add_accepted(X, df):
    X = X.copy()
    for name, code in ACCEPTED.items():
        ns = {"pd": pd, "np": np}
        exec(code, ns)
        X[name] = ns["compute"](df, {}).astype("float32").fillna(0).to_numpy()
    return X


preds = {"id": test["id"].values}

# --- bandgap: baseline + accepted features (no target leakage) ---
Xtr = add_accepted(X_train_base, train)
Xte = add_accepted(X_test_base, test)
m_bg = make_regressor()
m_bg.fit(Xtr, transform_target(train["bandgap_energy_ev"].to_numpy(), "log1p"))
preds["bandgap_energy_ev"] = np.expm1(m_bg.predict(Xte))
print(f"bandgap: trained on {Xtr.shape[1]} features")

# --- formation: plain baseline ---
m_fm = make_regressor()
m_fm.fit(X_train_base, transform_target(train["formation_energy_ev_natom"].to_numpy(), "log1p"))
preds["formation_energy_ev_natom"] = np.expm1(m_fm.predict(X_test_base))
print("formation: baseline trained")

sub = pd.DataFrame({
    "id": preds["id"],
    "formation_energy_ev_natom": preds["formation_energy_ev_natom"],
    "bandgap_energy_ev": preds["bandgap_energy_ev"],
})
out = os.path.join(BASE, "results", "nomad2018_submission.csv")
sub.to_csv(out, index=False)
print(f"wrote {out}")

# --- grade locally: mean column-wise RMSLE (same as mlebench grade.py) ---
from sklearn.metrics import root_mean_squared_log_error
answers = pd.read_csv(os.path.join(CACHE, "private", "test.csv"))
merged = sub.merge(answers[["id"] + TARGETS], on="id", suffixes=("", "_true"))
score = float(np.mean([
    root_mean_squared_log_error(merged[f"{t}_true"], merged[t]) for t in TARGETS
]))
print(f"\nlocal RMSLE (mean column-wise): {score:.4f}")
for t in TARGETS:
    s = root_mean_squared_log_error(merged[f"{t}_true"], merged[t])
    print(f"  {t}: {s:.4f}")
