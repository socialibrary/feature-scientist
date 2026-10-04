"""Run the Feature Scientist loop (Days 3-4).

  python src/run_scientist.py --rounds 3 [--engine rule|llm]

Each round: propose hypotheses -> build executable features -> leakage audit
-> serving-cost check -> ablate survivors -> ledger verdict. Prints a summary
table at the end.

A feature is ACCEPTED only if it passes the leakage audit, fits the serving
budget, AND lifts validation AUC by >= ACCEPT_DELTA_AUC. Everything else is
rejected with a reason -- including the planted traps, which die at the audit
or cost step before wasting any training time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation import ablate
from agentic import process_agentic
from baseline import TARGET, RESULTS_PATH as LGBM_RESULTS_PATH, featurize
from error_analysis import analyze, train_baseline_predict
from features import FeatureRegistry
from leakage import audit_feature
from ledger import append_entry, build_entry, load_entries
from llm_backend import LLMNotConfiguredError
from scientist import DATA_CATALOG, LLMEngine, build_ctx, get_engine
from serving_cost import MAX_TOTAL_UNITS, check_budget
from two_tower import RESULTS_PATH as TORCH_RESULTS_PATH

ACCEPT_DELTA_AUC = 0.0005


def run(rounds: int, engine_name: str, model_backend: str = "torch",
        llm_backend: str = "http") -> list[dict]:
    # --- one-time setup: data, baseline model, error slices, ctx -----------
    train, valid, stats, model_info, proba, cutoff_ts = \
        train_baseline_predict(model_backend)
    results_path = TORCH_RESULTS_PATH if model_backend == "torch" \
        else LGBM_RESULTS_PATH
    with open(results_path) as f:
        base = json.load(f)
    base_auc, base_ll = base["valid_auc"], base["valid_log_loss"]
    print(f"\nbaseline to beat [{model_backend}]: "
          f"AUC={base_auc:.4f} logloss={base_ll:.4f}")

    print("\nanalyzing error slices...")
    slices = analyze(train, valid, proba, stats)

    print("building feature-builder context (PIT structures, IMDb join)...")
    ctx = build_ctx(train, valid, stats, cutoff_ts)

    prep = None
    X_train_base = X_valid_base = None
    if model_backend == "torch":
        from two_tower import prepare_tower_inputs
        print("caching two-tower inputs...")
        prep = prepare_tower_inputs(train, valid, stats)
    else:
        print("caching baseline feature matrices...")
        X_train_base = featurize(train, stats)
        X_valid_base = featurize(valid, stats)

    try:
        engine = get_engine(engine_name, backend=llm_backend)
    except ValueError as e:
        print(e)
        sys.exit(2)

    agentic = isinstance(engine, LLMEngine) and engine.is_live()
    if engine_name == "llm":
        if agentic:
            print(f"agentic loop live via '{engine.backend.name}' backend: "
                  "propose -> critic -> revise -> audit -> cost -> ablate, "
                  "with one revision round after each rejection.")
        else:
            print("LLM backend not configured; will fall back per round.")

    registry = FeatureRegistry()
    ledger_entries: list[dict] = []
    accepted_cost_units = 0.0  # cumulative serving budget of accepted features

    # --- rounds ------------------------------------------------------------
    for r in range(1, rounds + 1):
        print(f"\n{'=' * 60}\nROUND {r}\n{'=' * 60}")
        try:
            hypotheses = engine.propose(slices, DATA_CATALOG, r)
        except (NotImplementedError, LLMNotConfiguredError) as e:
            print(f"engine unavailable: {e}\nfalling back to RuleBasedEngine")
            hypotheses = get_engine("rule").propose(slices, DATA_CATALOG, r)
            agentic = False

        if not hypotheses:
            print("no hypotheses proposed this round.")
            continue

        run_ctx = {
            "ctx": ctx, "train": train, "valid": valid, "stats": stats,
            "base_auc": base_auc, "base_ll": base_ll,
            "prep": prep, "X_train_base": X_train_base,
            "X_valid_base": X_valid_base, "model_backend": model_backend,
            "engine": engine, "accepted_cost_units": accepted_cost_units,
            "accept_delta": ACCEPT_DELTA_AUC,
        }
        for h in hypotheses:
            if agentic and getattr(h, "llm_spec", None):
                print(f"\n-- agentic hypothesis: {h.text}")
                print(f"   sources: {', '.join(h.sources)}")
                entry, accepted_cost_units = process_agentic(r, h, run_ctx)
                run_ctx["accepted_cost_units"] = accepted_cost_units
                ledger_entries.append(entry)
                continue
            print(f"\n-- hypothesis: {h.text}")
            print(f"   sources: {', '.join(h.sources)}")
            feature = h.build(ctx)
            registry.register(feature)
            print(f"   built feature '{feature.name}' "
                  f"(scope={feature.provenance.get('temporal_scope')})")

            audit = audit_feature(feature, ctx, valid)
            print(f"   leakage audit: {audit}")
            metrics = None
            cost = None
            if not audit.passed:
                verdict, reason = "rejected", \
                    "leakage: " + "; ".join(audit.reasons)
                print(f"   verdict: REJECTED ({reason})")
            else:
                cost = check_budget(feature, accepted_cost_units)
                print(f"   serving cost: {cost}")
                if not cost.passed:
                    verdict, reason = "rejected", \
                        "serving cost: " + "; ".join(cost.reasons)
                    print(f"   verdict: REJECTED ({reason})")
                else:
                    print("   ablating (baseline + feature)...")
                    metrics = ablate(train, valid, stats, feature, ctx,
                                     base_auc, base_ll, prep=prep,
                                     X_train_base=X_train_base,
                                     X_valid_base=X_valid_base,
                                     model=model_backend)
                    print(f"   ablation: AUC={metrics['auc']:.4f} "
                          f"(delta {metrics['delta_auc']:+.4f}), "
                          f"logloss={metrics['log_loss']:.4f}")
                    if metrics["delta_auc"] >= ACCEPT_DELTA_AUC:
                        verdict = "accepted"
                        reason = (f"AUC lift {metrics['delta_auc']:+.4f} >= "
                                  f"{ACCEPT_DELTA_AUC}")
                        accepted_cost_units = cost.new_total_units
                    else:
                        verdict, reason = "rejected", \
                            f"no lift (delta {metrics['delta_auc']:+.4f} < {ACCEPT_DELTA_AUC})"
                    print(f"   verdict: {verdict.upper()} ({reason})")

            entry = build_entry(r, h, feature, audit, cost, metrics,
                                verdict, reason)
            append_entry(entry)
            ledger_entries.append(entry)

    return ledger_entries


def print_summary(entries: list[dict]) -> None:
    print(f"\n{'=' * 60}\nEXPERIMENT SUMMARY\n{'=' * 60}")
    print(f"{'round':<6}{'feature':<28}{'verdict':<10}{'delta_auc':<10}reason")
    for e in entries:
        d = e["metrics"]["delta_auc"] if e["metrics"] else float("nan")
        dstr = f"{d:+.4f}" if e["metrics"] else "n/a"
        print(f"{e['round']:<6}{e['feature_name']:<28}{e['verdict']:<10}"
              f"{dstr:<10}{e['reason'][:60]}")
    accepted = [e for e in entries if e["verdict"] == "accepted"]
    print(f"\naccepted {len(accepted)}/{len(entries)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature Scientist science loop")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--engine", type=str, default="rule",
                    choices=["rule", "llm"])
    ap.add_argument("--backend", type=str, default="http",
                    choices=["http", "scripted"],
                    help="LLM backend for --engine llm: OpenAI-compatible "
                         "HTTP (needs FS_LLM_* env) or the offline scripted "
                         "test double")
    ap.add_argument("--model", type=str, default="torch",
                    choices=["torch", "lgbm"],
                    help="two-tower (default) or LightGBM baseline+ablations")
    args = ap.parse_args()
    entries = run(args.rounds, args.engine, args.model,
                  llm_backend=args.backend)
    print_summary(entries)
    print(f"\nledger: results/experiments.jsonl ({len(load_entries())} entries total)")


if __name__ == "__main__":
    main()
