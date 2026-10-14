#!/usr/bin/env python3
"""Generic MLE-bench competition submitter for Feature Scientist champions.

Replaces the hardcoded scripts/nomad_submission.py with a config-driven
flow: any brief whose champion pickles live in results/ can produce a
submission.csv.

Config (YAML, e.g. briefs/submit_nomad2018.yaml):

    dataset_dir: data/kaggle/nomad2018        # repo-relative, train/test.csv
    id_column: id
    output: results/nomad2018_submission.csv
    mlebench_competition: nomad2018-predict-transparent-conductors  # for local grading
    targets:
      - name: bandgap_energy_ev
        champion: results/champion_nomad2018_bandgap.pkl  # null = baseline only
        target_transform: log1p        # regression only
        grade: rmsle                   # rmsle | rmse | logloss | auc
      - name: formation_energy_ev_natom
        champion: null
        target_transform: log1p
        grade: rmsle

Run:  python3 scripts/mlebench_submit.py briefs/submit_nomad2018.yaml

Target-leakage discipline (the nomad lesson): when a competition has
several targets, EVERY target is excluded from the feature matrix for
each per-target model -- neither target is ever a feature for the other.
Accepted features are re-executed from the champion's saved source (not
re-derived), on raw train/test frames, exactly as the ablations measured.

Classification targets are label-encoded with encode_labels()
(sorted-unique); the champion's saved label_classes map argmax predictions
back to the original labels for the submission file.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from baseline import (encode_labels, featurize_tabular, fit_tabular_schema,
                      make_model, make_regressor, transform_target)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_dir"] = os.path.join(BASE, cfg["dataset_dir"])
    if cfg.get("output"):
        cfg["output"] = os.path.join(BASE, cfg["output"])
    else:
        cfg["output"] = os.path.join(
            BASE, "results",
            os.path.splitext(os.path.basename(path))[0]
            .replace("submit_", "") + "_submission.csv")
    return cfg


def build_target_predictions(cfg, train, test):
    id_col = cfg.get("id_column", "id")
    all_targets = [t["name"] for t in cfg["targets"]]
    preds = {"id": test[id_col].to_numpy()}
    for spec in cfg["targets"]:
        target = spec["name"]
        tform = spec.get("target_transform")
        champion_path = spec.get("champion")
        code = {}
        label_classes = None
        if champion_path:
            with open(os.path.join(BASE, champion_path), "rb") as f:
                champ = pickle.load(f)
            code = champ.get("accepted_feature_code") or {}
            label_classes = champ.get("label_classes")
            print(f"{target}: {len(code)} accepted features from {champion_path}")
        else:
            print(f"{target}: baseline only (no champion)")

        # Exclude ALL targets: sibling-target leakage guard.
        schema = fit_tabular_schema(train, target, exclude=all_targets)
        X_train = featurize_tabular(train, target, schema, exclude=all_targets)
        X_test = featurize_tabular(test, target, schema, exclude=all_targets)
        print(f"  base matrix: {X_train.shape[1]} cols "
              f"(excluded: {all_targets})")

        for name, src in code.items():
            ns = {"pd": pd, "np": np}
            exec(src, ns)
            X_train[name] = ns["compute"](train, {}).astype("float32") \
                .fillna(0).to_numpy()
            X_test[name] = ns["compute"](test, {}).astype("float32") \
                .fillna(0).to_numpy()

        y = train[target].to_numpy()
        is_regression = y.dtype.kind == "f" or tform is not None
        if is_regression:
            model = make_regressor()
            model.fit(X_train, transform_target(y, tform))
            p = model.predict(X_test)
            if tform == "log1p":
                p = np.expm1(p)
            preds[target] = p
        else:
            y_enc, _, classes = encode_labels(y)
            n = len(classes)
            model = make_model(n if n > 2 else None)
            model.fit(X_train, y_enc)
            proba = model.predict_proba(X_test)
            idx = proba.argmax(axis=1)
            if n > 2:
                labels = label_classes or [str(c) for c in classes]
                preds[target] = np.array(labels)[idx]
            else:
                preds[target] = idx
        print(f"  trained on {X_train.shape[1]} features")
    return preds


def grade_locally(cfg, sub):
    """Grade against mlebench's private answers when the data is prepared."""
    comp = cfg.get("mlebench_competition")
    if not comp:
        return
    cache = os.path.expanduser(
        f"~/.cache/mle-bench/data/{comp}/prepared/private/test.csv")
    if not os.path.exists(cache):
        print(f"\nlocal grading skipped: {cache} not found")
        return
    from sklearn.metrics import (log_loss, roc_auc_score,
                                 root_mean_squared_log_error,
                                 mean_squared_error)
    answers = pd.read_csv(cache)
    id_col = cfg.get("id_column", "id")
    merged = sub.merge(answers, on=id_col, suffixes=("", "_true"))
    print("\nlocal grade (mlebench private answers):")
    scores = []
    for spec in cfg["targets"]:
        t, kind = spec["name"], spec.get("grade", "rmsle")
        yt, yp = merged[f"{t}_true"], merged[t]
        if kind == "rmsle":
            s = float(root_mean_squared_log_error(yt, yp))
        elif kind == "rmse":
            s = float(np.sqrt(mean_squared_error(yt, yp)))
        elif kind == "logloss":
            s = float(log_loss(yt, yp))
        elif kind == "auc":
            s = float(roc_auc_score(yt, yp))
        else:
            raise ValueError(f"unknown grade kind: {kind}")
        scores.append(s)
        print(f"  {t}: {kind}={s:.4f}")
    if scores:
        print(f"  mean: {np.mean(scores):.4f}")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: mlebench_submit.py <submit-config.yaml>")
    cfg = load_config(sys.argv[1])
    train = pd.read_csv(os.path.join(cfg["dataset_dir"], "train.csv"))
    test = pd.read_csv(os.path.join(cfg["dataset_dir"], "test.csv"))
    print(f"train {train.shape}, test {test.shape}")
    preds = build_target_predictions(cfg, train, test)
    cols = [cfg.get("id_column", "id")] + [t["name"] for t in cfg["targets"]]
    sub = pd.DataFrame({c: preds[c] for c in cols})
    sub.to_csv(cfg["output"], index=False)
    print(f"\nwrote {cfg['output']} ({len(sub)} rows)")
    grade_locally(cfg, sub)


if __name__ == "__main__":
    main()
