#!/usr/bin/env python3
"""3-seed @ 8-epoch verification of the v2 brief's two single-seed acceptances.

The v2 brief (brief.py path) accepted movie_age_days_llm (+0.0018) and
movie_release_year_llm (+0.0021) single-seed at 4 epochs -- bypassing the
hardened 3-seed gate from iterate.py. This script enforces that gate:
an acceptance stands only if the 3-seed mean marginal vs the previous
champion exceeds +0.0005 (8 epochs, seeds 42/43/44).

Resumable: per-stage results append to results/verify_v2_3seed.jsonl;
stages already present are skipped on re-run.
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ablation import evaluate_champion  # noqa: E402
from baseline import build_train_only_stats  # noqa: E402
from data import load_all  # noqa: E402
from features import Feature  # noqa: E402
from iter_features import merge_wikidata_ctx  # noqa: E402
from scientist import build_ctx  # noqa: E402
from split import temporal_split  # noqa: E402
from two_tower import prepare_tower_inputs  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results", "verify_v2_3seed.jsonl")
SEEDS = (42, 43, 44)
EPOCHS = 8
GATE = 0.0005

FEATURE_FILES = [
    ("movie_age_days_llm", "results/gen_features/movie_age_days_llm.py"),
    ("movie_release_year_llm", "results/gen_features/movie_release_year_llm.py"),
]


def load_feature(name, relpath):
    import pandas as _pd
    import numpy as _np
    path = os.path.join(BASE, relpath)
    spec = importlib.util.spec_from_file_location(f"gen_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    # gen files reference pd/np without importing them; the brief path execs
    # them with those names in scope, so provide them here too.
    mod.__dict__["pd"] = _pd
    mod.__dict__["np"] = _np
    spec.loader.exec_module(mod)
    return Feature(name=name, compute=mod.compute)


def done_stages():
    if not os.path.exists(RESULTS):
        return set()
    out = set()
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["stage"])
    return out


def main():
    print("== 3-seed @ 8-epoch verification of v2 acceptances ==")
    df = load_all()
    train, valid, cutoff_ts = temporal_split(df)
    stats = build_train_only_stats(train)
    prep = prepare_tower_inputs(train, valid, stats)
    ctx = build_ctx(train, valid, stats, cutoff_ts)
    ctx = merge_wikidata_ctx(ctx, train)

    feats = [load_feature(n, p) for n, p in FEATURE_FILES]
    stages = [
        ("base", []),
        ("age", [feats[0]]),
        ("age+release_year", feats),
    ]
    have = done_stages()
    results = {}
    for stage, accepted in stages:
        if stage in have:
            print(f"-- stage '{stage}' already done, skipping")
            with open(RESULTS) as f:
                for line in f:
                    d = json.loads(line)
                    if d["stage"] == stage:
                        results[stage] = d
            continue
        print(f"-- stage '{stage}': training 3 seeds x 8 epochs ...", flush=True)
        r = evaluate_champion(train, valid, stats, accepted, ctx, prep=prep,
                              seeds=SEEDS, verbose=True, epochs=EPOCHS)
        rec = {"stage": stage,
               "auc_mean": r["auc_mean"], "auc_std": r["auc_std"],
               "ll_mean": r["ll_mean"], "ll_std": r["ll_std"],
               "auc_seeds": r["auc_seeds"], "ll_seeds": r["ll_seeds"],
               "epochs": EPOCHS, "seeds": list(SEEDS)}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        results[stage] = rec

    base = results["base"]
    age = results["age"]
    both = results["age+release_year"]
    m_age = age["auc_mean"] - base["auc_mean"]
    m_ry = both["auc_mean"] - age["auc_mean"]
    print("\n== VERDICT (gate: 3-seed mean marginal > +0.0005) ==")
    print(f"base 8-epoch 3-seed: AUC={base['auc_mean']:.4f}±{base['auc_std']:.4f} "
          f"ll={base['ll_mean']:.4f}±{base['ll_std']:.4f}")
    print(f"movie_age_days_llm:      marginal {m_age:+.4f} -> "
          f"{'STANDS' if m_age > GATE else 'REVERTED'}")
    print(f"movie_release_year_llm:  marginal {m_ry:+.4f} -> "
          f"{'STANDS' if m_ry > GATE else 'REVERTED'}")
    print(f"(1% target needs +0.0100 AUC on base {base['auc_mean']:.4f})")


if __name__ == "__main__":
    main()
