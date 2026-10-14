"""3-seed honesty check for the spaceship-titanic champion.

Replicates the brief-run ablation (featurize_tabular + LightGBM, same
hyperparams) for the baseline and the champion (baseline + cabin_deck +
cabin_side + cabin_num) across 3 data-split/model seeds. Reports the mean
marginal vs baseline; the champion stands only if the mean marginal exceeds
+0.0005 (same bar as the iterate revert gate).

Run:  source .venv/bin/activate && python3 scripts/verify_st_3seed.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from sklearn.metrics import roc_auc_score

from baseline import featurize_tabular, fit_tabular_schema
from brief_data import load_brief_data, parse_target

SEEDS = [42, 43, 44]
FEATURE_FILES = ["cabin_deck", "cabin_side", "cabin_num"]


def load_compute(name):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "results", "gen_features", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import pandas as pd  # noqa: F401  (feature code expects pd/np in ns)
    import numpy as np  # noqa: F401
    mod.pd = pd
    mod.np = np
    spec.loader.exec_module(mod)
    return mod.compute


def make_lgbm(seed):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=seed,
        n_jobs=-1, verbose=-1)


def main():
    brief = {
        "tables": [{"name": "st_train"}],
        "label": "Transported BOOL, in train.csv",
    }
    computes = [load_compute(n) for n in FEATURE_FILES]
    rows = []
    for seed in SEEDS:
        train, valid, target = load_brief_data(brief, seed=seed)
        schema = fit_tabular_schema(train, target)
        Xtr = featurize_tabular(train, target, schema)
        Xva = featurize_tabular(valid, target, schema)
        ytr = train[target].to_numpy().astype(int)
        yva = valid[target].to_numpy().astype(int)

        m = make_lgbm(seed)
        m.fit(Xtr, ytr)
        base_auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])

        Xtr_c, Xva_c = Xtr.copy(), Xva.copy()
        for name, compute in zip(FEATURE_FILES, computes):
            Xtr_c[name] = compute(train, {}).astype("float32").fillna(0).to_numpy()
            Xva_c[name] = compute(valid, {}).astype("float32").fillna(0).to_numpy()
        m2 = make_lgbm(seed)
        m2.fit(Xtr_c, ytr)
        champ_auc = roc_auc_score(yva, m2.predict_proba(Xva_c)[:, 1])

        marginal = champ_auc - base_auc
        rows.append((seed, base_auc, champ_auc, marginal))
        print(f"seed {seed}: baseline={base_auc:.4f} champion={champ_auc:.4f} "
              f"marginal={marginal:+.4f}")

    marg = np.array([r[3] for r in rows])
    mean, std = marg.mean(), marg.std(ddof=1) if len(marg) > 1 else 0.0
    verdict = "STANDS" if mean > 0.0005 else "REVERTED"
    print(f"\nmean marginal {mean:+.4f} +/- {std:.4f} over {len(SEEDS)} seeds "
          f"-> champion {verdict}")
    return 0 if verdict == "STANDS" else 1


if __name__ == "__main__":
    sys.exit(main())
