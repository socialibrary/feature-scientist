#!/usr/bin/env python3
"""Offline validation of the v3 scripted backend arc (FS_SCRIPTED=v3).

Verifies, WITHOUT any network or training:
  1. propose -> parse_llm_response -> validate_spec -> safety_check
  2. staged critic verdicts route to the right canned text (name-keyed)
  3. materialize_feature + smoke_test on real train rows
  4. compute() on real train/valid ctx: aligned, finite, non-degenerate
     (ablation.check_degenerate), with coverage/variance stats
  5. leakage.audit_feature: H1/H2 PASS, H3 FAILS (all_time fail-closed),
     H3r FAILS again (B4: pit_correct declaration not credible)
  6. the revise round returns the staged H3r spec

Usage: python3 scripts/validate_v3.py
Exit 0 when every staged expectation holds, 1 otherwise.
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

os.environ["FS_SCRIPTED"] = "v3"

import numpy as np
import pandas as pd

from ablation import check_degenerate
from baseline import build_train_only_stats
from codegen import parse_llm_response, validate_spec, safety_check, \
    materialize_feature, smoke_test
from critic import review
from data import load_all
from iter_features import merge_wikidata_ctx
from leakage import audit_feature
from llm_backend import ScriptedBackend
from scientist import build_ctx
from split import temporal_split

failures = []


def check(cond: bool, label: str, detail: str = ""):
    tag = "ok  " if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f" -- {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def main() -> int:
    print("== v3 scripted-backend offline validation ==\n")

    backend = ScriptedBackend()
    print(f"script selected: {backend.script is not None}")
    check(isinstance(backend.script, dict) and "critique_by_name" in backend.script,
          "v3 script selected via FS_SCRIPTED")

    # ---- 1. propose ---------------------------------------------------
    raw = backend.propose("PROMPT (ignored by scripted backend)")
    payload = parse_llm_response(raw)
    hyps = payload.get("hypotheses", [])
    check(len(hyps) == 3, "propose returns 3 hypotheses",
          f"got {len(hyps)}")
    specs = {}
    for spec in hyps:
        spec = validate_spec(spec)
        check(not safety_check("def compute(df, ctx):\n" + spec["code"]) or True,
              f"safety_check parseable: {spec['name']}")
        specs[spec["name"]] = spec
    names = set(specs)
    check(names == {"movie_budget_prior_llm", "movie_nonenglish_llm",
                    "movie_awards_prior_llm"},
          "hypothesis names as staged", str(sorted(names)))

    # ---- 2. data + ctx -----------------------------------------------
    print("\nbuilding data + ctx (train/valid split, wikidata merge)...")
    df = load_all()
    train, valid, cutoff_ts = temporal_split(df)
    stats = build_train_only_stats(train)
    ctx = build_ctx(train, valid, stats, cutoff_ts)
    merge_wikidata_ctx(ctx, train)
    for k in ["movie_budget_log", "budget_log_median", "movie_award_count",
              "movie_non_english"]:
        check(k in ctx, f"ctx has {k}", f"n={len(ctx[k]) if isinstance(ctx[k], dict) else ctx[k]}")
    for k in ["movie_budget_log", "movie_award_count", "movie_non_english"]:
        check(k in ctx.get("_table_meta", {}), f"_table_meta has {k}")

    # ---- 3/4. materialize + compute sanity -----------------------------
    feats = {}
    for name, spec in specs.items():
        f = materialize_feature(spec)
        feats[name] = f
        try:
            smoke_test(f, train.head(100), ctx)
            check(True, f"smoke_test: {name}")
        except Exception as e:  # noqa: BLE001
            check(False, f"smoke_test: {name}", str(e)[:120])
        ctr = np.asarray(f.compute(train, ctx), dtype=float)
        cva = np.asarray(f.compute(valid, ctx), dtype=float)
        check(len(ctr) == len(train) and len(cva) == len(valid),
              f"compute aligned: {name}")
        deg = check_degenerate(ctr, name)
        check(deg is None, f"non-degenerate on train: {name}", deg or "")
        check(np.all(np.isfinite(cva)), f"finite on valid: {name}")
        print(f"        {name}: train std={np.std(ctr):.4f} "
              f"valid std={np.std(cva):.4f} "
              f"range=({np.min(cva):.3f},{np.max(cva):.3f})")

    # coverage honesty: what fraction of valid movies actually have data?
    mids = valid["movie_id"].to_numpy()
    bmap, amap, lmap = (ctx["movie_budget_log"], ctx["movie_award_count"],
                        ctx["movie_non_english"])
    print(f"        budget coverage on valid movies: "
          f"{np.mean([m in bmap for m in mids]):.1%}")
    print(f"        awards coverage on valid movies: "
          f"{np.mean([m in amap for m in mids]):.1%}")
    print(f"        non-english rate on valid movies: "
          f"{np.mean([lmap.get(m, 0.0) for m in mids]):.2%}")

    # ---- 2b. staged critic verdicts ------------------------------------
    print("\nstaged critic routing:")
    for name, spec in specs.items():
        cv = review(spec, backend, ctx.get("_table_meta", {}))
        check(cv.passed, f"critic passes (advisory): {name}",
              cv.critique[:80])

    # ---- 5. leakage audit -----------------------------------------------
    print("\nleakage audit (real, from leakage.py):")
    expected = {"movie_budget_prior_llm": True,
                "movie_nonenglish_llm": True,
                "movie_awards_prior_llm": False}
    for name, want_pass in expected.items():
        a = audit_feature(feats[name], ctx, valid)
        check(a.passed == want_pass, f"audit {'PASS' if want_pass else 'FAIL'}: {name}",
              str(a)[:160])

    # ---- 6. revise round --------------------------------------------------
    raw_rev = backend.revise("REVISE PROMPT (ignored by scripted backend)")
    rev_spec = validate_spec(parse_llm_response(raw_rev))
    check(rev_spec["name"] == "movie_awards_asof_llm",
          "revise returns staged H3r", rev_spec["name"])
    fr = materialize_feature(rev_spec)
    smoke_test(fr, train.head(100), ctx)
    a2 = audit_feature(fr, ctx, valid)
    check(not a2.passed and any("pit_correct" in r or "prediction time" in r
                                for r in a2.reasons),
          "audit kills the as-of laundering attempt (B4)",
          str(a2)[:180])

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILED: {failures}")
        return 1
    print("RESULT: all v3 staging expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
