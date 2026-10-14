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
import inspect
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
# NOTE: two_tower (torch) is imported lazily inside the torch branches --
# lgbm-only generic runs must not require torch.

ACCEPT_DELTA_AUC = 0.0005
# Regression acceptance: RMSE improvement (lower is better) must clear this
# bar. Same magnitude convention as the AUC gate.
ACCEPT_DELTA_RMSE = 0.0005
# Log-loss acceptance (multiclass text briefs): same 0.0005 convention,
# lower is better.
ACCEPT_DELTA_LOGLOSS = 0.0005


def run(rounds: int, engine_name: str, model_backend: str = "torch",
        llm_backend: str = "http",
        calibration_screen: bool = False,
        screen_battery_fn=None,
        stop_on_no_signal: bool = False,
        problem: dict | None = None,
        skip_accepted: bool = False,
        catalog: dict | None = None,
        brief: dict | None = None,
        save_champion: str | None = None) -> list[dict]:
    """The science loop.

    The extra kwargs are the brief-driven mode (src/brief.py); all default
    to off so the classic paths (--rounds N, --engine llm --backend
    scripted) behave exactly as before:
      calibration_screen - gate each surviving candidate through the
        calibration screen before ablation; non-signals are "screened_out"
        without spending a training run (docs/calibration_integration.md).
      screen_battery_fn  - (ctx, train, valid, stats, proba) -> [summaries];
        cheap pre-loop screen whose significant gaps guide proposing.
      stop_on_no_signal  - stop after a round with no significant
        calibration gap (the screen as convergence criterion).
      problem            - brief dict (problem/label/prediction_time/...);
        threaded into the LLM prompt and stamped on ledger entries.
      skip_accepted      - don't re-ablate already-accepted features.
      catalog            - data catalog for the engine (default DATA_CATALOG).
      save_champion      - when set: after the loop, retrain the champion
        (baseline + accepted features) and save it to this path for
        deployment. lgbm only in v1; torch prints an honest skip.
    """
    from serving_cost import MAX_TOTAL_UNITS
    cat = catalog if catalog is not None else DATA_CATALOG
    problem = dict(problem or {})
    # --- one-time setup: data, baseline model, error slices, ctx -----------
    train, valid, stats, model_info, proba, cutoff_ts = \
        train_baseline_predict(model_backend, brief=brief)
    # Task metric: classification reports AUC/log-loss; regression briefs
    # report RMSE (computed in the brief's target-transform space);
    # multiclass briefs (metric: logloss) report log-loss only.
    metric_name = "auc"
    base_metric = None
    base_auc = base_ll = None
    _target_transform = model_info.get("target_transform")
    if brief is not None:
        # Generic task: compute baseline metrics from proba directly.
        from brief_data import parse_target
        from baseline import encode_labels, transform_target
        _bt = parse_target(brief.get("label", ""))
        _by = valid[_bt].to_numpy()
        if model_info.get("is_regression"):
            from sklearn.metrics import mean_squared_error
            import numpy as np
            metric_name = model_info.get("metric") or "rmse"
            _yt = transform_target(_by, _target_transform)
            base_metric = float(np.sqrt(mean_squared_error(_yt, proba)))
        elif (model_info.get("metric") or "auc") == "logloss":
            from sklearn.metrics import log_loss
            metric_name = "logloss"
            # Encode valid with TRAIN's full class set: a rare class can be
            # absent from valid, and valid-only codes would misalign with
            # the model's probability columns.
            _, _yv, _classes = encode_labels(train[_bt].to_numpy(), _by)
            _n_prob = proba.shape[1] if hasattr(proba, "shape") else len(_classes)
            base_ll = float(log_loss(_yv, proba,
                                     labels=list(range(_n_prob))))
        else:
            from sklearn.metrics import roc_auc_score, log_loss
            # Encode the target exactly as the baseline was trained
            # (sorted-unique), so proba columns line up for string labels.
            _yv, _, _ = encode_labels(_by)
            try:
                base_auc = float(roc_auc_score(_yv, proba))
                base_ll = float(log_loss(_yv, proba))
            except Exception:
                base_auc, base_ll = 0.5, 1.0
    else:
        if model_backend == "torch":
            from two_tower import RESULTS_PATH as TORCH_RESULTS_PATH
            results_path = TORCH_RESULTS_PATH
        else:
            results_path = LGBM_RESULTS_PATH
        with open(results_path) as f:
            base = json.load(f)
        base_auc, base_ll = base["valid_auc"], base["valid_log_loss"]
    if metric_name == "rmse":
        print(f"\nbaseline to beat [{model_backend}]: RMSE={base_metric:.4f} "
              f"(regression; target_transform={_target_transform})")
    elif metric_name == "logloss":
        print(f"\nbaseline to beat [{model_backend}]: "
              f"logloss={base_ll:.4f} (multiclass; "
              f"{model_info.get('n_classes')} classes)")
    else:
        print(f"\nbaseline to beat [{model_backend}]: "
              f"AUC={base_auc:.4f} logloss={base_ll:.4f}")

    print("\nanalyzing error slices...")
    # Generic target for non-MovieLens briefs.
    _analyze_target = None
    _is_generic = False
    if brief is not None:
        from brief_data import parse_target
        _analyze_target = parse_target(brief.get("label", ""))
        # Generic if tables are *_train (not MovieLens).
        _is_generic = any(
            (t.get("name", "").endswith("_train"))
            for t in (brief.get("tables") or [])
        )
    if _is_generic:
        # Use the generic problem finder: rank slices by impact.
        from problem_finder import find_problems
        import pandas as pd
        _valid_for_slices = valid.copy()
        # proba is array-like; ensure it's numpy
        import numpy as np
        _proba = np.asarray(proba)
        _problems = find_problems(_valid_for_slices, _proba,
                                  _analyze_target, max_slices=5,
                                  is_regression=(metric_name == "rmse"))
        # Convert to the error_slices format the proposer expects.
        slices = {"worst": [
            {"slice": p["slice"], "error_rate": p["error_rate"],
             "lift_pp": p["lift_pp"], "n": p["n"],
             "fraction": p["fraction"]}
            for p in _problems
        ]}
        if slices["worst"]:
            print(f"  top problem: {slices['worst'][0]['slice']} "
                  f"(+{slices['worst'][0]['lift_pp']:.1f}pp, "
                  f"{slices['worst'][0]['fraction']:.1%} of data)")
        else:
            print("  (no problem slices found; proposer runs untargeted)")
    else:
        slices = analyze(train, valid, proba, stats, target=_analyze_target)

    print("building feature-builder context (PIT structures, IMDb join)...")
    if brief is not None:
        # Generic tabular task: empty ctx (features are row-local).
        ctx = {}
        # Target-leakage discipline: the audit needs the target name.
        if _analyze_target:
            ctx["_target_column"] = _analyze_target
    else:
        ctx = build_ctx(train, valid, stats, cutoff_ts)

    # Wikidata-backed ctx tables (budget/language/awards maps, cast prefix
    # sums, user×director rates). Guarded: brief runs keep working when the
    # crawl CSV isn't present yet.
    try:
        from iter_features import merge_wikidata_ctx
        merge_wikidata_ctx(ctx, train)
        print("   + wikidata ctx merged "
              f"(budget_log={len(ctx.get('movie_budget_log', {}))} movies, "
              f"non_english={len(ctx.get('movie_non_english', {}))} movies)")
    except Exception as e:  # noqa: BLE001
        print(f"   wikidata ctx skipped: {e}")

    prep = None
    X_train_base = X_valid_base = None
    if model_backend == "torch":
        from two_tower import prepare_tower_inputs
        print("caching two-tower inputs...")
        prep = prepare_tower_inputs(train, valid, stats)
    else:
        print("caching baseline feature matrices...")
        if brief is not None:
            # Generic tabular path: featurize with the train-fit schema so
            # the ablation matrices match the baseline's column-for-column.
            from brief_data import parse_target
            from baseline import featurize_tabular, fit_tabular_schema
            _target = parse_target((brief.get("label", "")))
            _schema = model_info.get("tabular_schema") \
                or fit_tabular_schema(train, _target)
            X_train_base = featurize_tabular(train, _target, _schema)
            X_valid_base = featurize_tabular(valid, _target, _schema)
        else:
            X_train_base = featurize(train, stats)
            X_valid_base = featurize(valid, stats)

    # --- calibration battery: cheap pre-loop screen (no training) ----------
    # Its significant gaps guide proposing (they become part of the LLM
    # prompt); with stop_on_no_signal, an empty battery stops the run
    # before round 1 -- the screen as convergence criterion.
    if screen_battery_fn is not None:
        print("\nrunning calibration screen battery (no training)...")
        battery_summaries = screen_battery_fn(ctx, train, valid, stats,
                                              proba)
        for s in battery_summaries:
            print(f"  - {s}")
        if not battery_summaries:
            print("battery found no significant calibration gaps.")
        problem = {**problem, "calibration_gaps": battery_summaries}
        if stop_on_no_signal and not battery_summaries:
            print("STOPPING before round 1: no significant calibration "
                  "gaps remain.")
            return []

    try:
        engine = get_engine(engine_name, backend=llm_backend,
                            problem_context=problem if engine_name == "llm"
                            else None)
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
    accepted_names: set[str] = set()  # brief mode: skip re-ablating these
    accepted_features: dict = {}  # name -> Feature, for champion retraining
    tested_names: set[str] = set()  # brief mode: already ablated this run
    brief_stamp = None
    if problem:
        brief_stamp = {k: problem[k] for k in ("problem", "label",
                                               "prediction_time")
                       if k in problem} or None

    # --- rounds ------------------------------------------------------------
    for r in range(1, rounds + 1):
        print(f"\n{'=' * 60}\nROUND {r}\n{'=' * 60}")
        try:
            hypotheses = engine.propose(slices, cat, r)
        except (NotImplementedError, LLMNotConfiguredError) as e:
            print(f"engine unavailable: {e}\nfalling back to RuleBasedEngine")
            hypotheses = get_engine("rule").propose(slices, cat, r)
            agentic = False

        if not hypotheses:
            print("no hypotheses proposed this round.")
            continue

        round_had_signal = False  # any fresh hypothesis ablated this round?
        round_had_fresh = False  # any non-duplicate hypothesis?
        run_ctx = {
            "ctx": ctx, "train": train, "valid": valid, "stats": stats,
            "base_auc": base_auc, "base_ll": base_ll, "proba": proba,
            "prep": prep, "X_train_base": X_train_base,
            "X_valid_base": X_valid_base, "model_backend": model_backend,
            "engine": engine, "accepted_cost_units": accepted_cost_units,
            "accepted_names": accepted_names,
            "tested_names": tested_names,
            "accept_delta": ACCEPT_DELTA_AUC,
            "accept_delta_rmse": ACCEPT_DELTA_RMSE,
            "accept_delta_logloss": ACCEPT_DELTA_LOGLOSS,
            "accepted_features": accepted_features,
            "metric_name": metric_name,
            "base_metric": (base_ll if metric_name == "logloss"
                            else base_metric),
            "target_transform": _target_transform,
            "calibration_screen": calibration_screen,
            "skip_accepted": skip_accepted,
            "brief_stamp": brief_stamp,
            "brief": brief,
        }
        for h in hypotheses:
            if agentic and getattr(h, "llm_spec", None):
                print(f"\n-- agentic hypothesis: {h.text}")
                print(f"   sources: {', '.join(h.sources)}")
                entry, accepted_cost_units = process_agentic(r, h, run_ctx)
                run_ctx["accepted_cost_units"] = accepted_cost_units
                for a in entry.get("agentic", {}).get("attempts", []):
                    if (a.get("stage") == "screen"
                            and a.get("verdict") == "signal") or \
                            a.get("stage") == "ablation":
                        # A hypothesis that reached the ablation stage is
                        # signal; the screen-stage check alone would
                        # wrongly stop no-screen runs (generic tabular
                        # briefs) after round 1.
                        round_had_signal = True
                if entry.get("verdict") != "duplicate":
                    round_had_fresh = True
                ledger_entries.append(entry)
                continue
            print(f"\n-- hypothesis: {h.text}")
            print(f"   sources: {', '.join(h.sources)}")
            feature = h.build(ctx)
            registry.register(feature)
            print(f"   built feature '{feature.name}' "
                  f"(scope={feature.provenance.get('temporal_scope')})")

            audit = cost = metrics = None
            verdict = reason = None
            if skip_accepted and feature.name in accepted_names:
                verdict, reason = "duplicate", \
                    "already accepted in an earlier round; not re-ablated"
                print(f"   verdict: DUPLICATE ({reason})")
            elif skip_accepted and feature.name in tested_names:
                verdict, reason = "duplicate", \
                    ("already ablated in an earlier round of this brief; "
                     "not re-tested")
                print(f"   verdict: DUPLICATE ({reason})")
            else:
                round_had_fresh = True
                audit = audit_feature(feature, ctx, valid)
                print(f"   leakage audit: {audit}")
            if verdict is None:
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
                    elif calibration_screen:
                        # Screen gate (docs/calibration_integration.md):
                        # ablate only significant miscalibrations; the rest
                        # never cost a training run.
                        from calibration import calibration_screen as _screen
                        vals = feature.compute(valid, ctx)
                        scr = _screen(valid[TARGET].to_numpy(), proba,
                                      vals, feature.name)
                        print(f"   calibration screen: {scr['summary']}")
                        if scr["verdict"] == "signal":
                            round_had_signal = True
                        else:
                            verdict, reason = "screened_out", \
                                "calibration screen: " + scr["summary"]
                            print(f"   verdict: SCREENED OUT ({reason})")
                    if verdict is None:
                        if not calibration_screen:
                            # Generic tabular briefs skip the calibration
                            # battery, so the gap-based signal can never
                            # fire; reaching the ablation stage (passed
                            # audit + cost) is the signal here.
                            round_had_signal = True
                        print("   ablating (baseline + feature)...")
                        _ablate_target = None
                        if brief is not None:
                            from brief_data import parse_target
                            _ablate_target = parse_target(brief.get("label", ""))
                        metrics = ablate(train, valid, stats, feature, ctx,
                                         base_auc, base_ll, prep=prep,
                                         X_train_base=X_train_base,
                                         X_valid_base=X_valid_base,
                                         model=model_backend,
                                         target=_ablate_target,
                                         metric=metric_name,
                                         baseline_metric=(base_ll
                                                          if metric_name
                                                          == "logloss"
                                                          else base_metric),
                                         target_transform=_target_transform)
                        # Brief mode: never re-ablate the same feature name
                        # in a later round of this run (scripted backends may
                        # repeat proposals).
                        tested_names.add(feature.name)
                        if metric_name == "rmse":
                            print(f"   ablation: RMSE={metrics['rmse']:.4f} "
                                  f"(delta {metrics['delta_rmse']:+.4f})")
                            if metrics["delta_rmse"] <= -ACCEPT_DELTA_RMSE:
                                verdict = "accepted"
                                reason = (f"RMSE improvement "
                                          f"{metrics['delta_rmse']:+.4f} "
                                          f"<= {-ACCEPT_DELTA_RMSE}")
                                accepted_cost_units = cost.new_total_units
                                accepted_names.add(feature.name)
                                accepted_features[feature.name] = feature
                            else:
                                verdict, reason = "rejected", \
                                    (f"no improvement (delta_rmse "
                                     f"{metrics['delta_rmse']:+.4f} > "
                                     f"{-ACCEPT_DELTA_RMSE})")
                        elif metric_name == "logloss":
                            print(f"   ablation: logloss={metrics['log_loss']:.4f} "
                                  f"(delta {metrics['delta_log_loss']:+.4f})")
                            if metrics["delta_log_loss"] <= -ACCEPT_DELTA_LOGLOSS:
                                verdict = "accepted"
                                reason = (f"log-loss improvement "
                                          f"{metrics['delta_log_loss']:+.4f} "
                                          f"<= {-ACCEPT_DELTA_LOGLOSS}")
                                accepted_cost_units = cost.new_total_units
                                accepted_names.add(feature.name)
                                accepted_features[feature.name] = feature
                            else:
                                verdict, reason = "rejected", \
                                    (f"no improvement (delta_log_loss "
                                     f"{metrics['delta_log_loss']:+.4f} > "
                                     f"{-ACCEPT_DELTA_LOGLOSS})")
                        else:
                            print(f"   ablation: AUC={metrics['auc']:.4f} "
                                  f"(delta {metrics['delta_auc']:+.4f}), "
                                  f"logloss={metrics['log_loss']:.4f}")
                            if metrics["delta_auc"] >= ACCEPT_DELTA_AUC:
                                verdict = "accepted"
                                reason = (f"AUC lift "
                                          f"{metrics['delta_auc']:+.4f} "
                                          f">= {ACCEPT_DELTA_AUC}")
                                accepted_cost_units = cost.new_total_units
                                accepted_names.add(feature.name)
                                accepted_features[feature.name] = feature
                            else:
                                verdict, reason = "rejected", \
                                    f"no lift (delta {metrics['delta_auc']:+.4f} < {ACCEPT_DELTA_AUC})"
                        print(f"   verdict: {verdict.upper()} ({reason})")

            entry = build_entry(r, h, feature, audit, cost, metrics,
                                verdict, reason, brief=brief_stamp)
            append_entry(entry)
            ledger_entries.append(entry)

        # --- stopping rules (brief mode) -----------------------------------
        if accepted_cost_units >= MAX_TOTAL_UNITS:
            print(f"\nSTOPPING after round {r}: serving budget exhausted "
                  f"({accepted_cost_units:.1f}/{MAX_TOTAL_UNITS} units).")
            break
        if stop_on_no_signal and hypotheses:
            if not round_had_fresh:
                print(f"\nSTOPPING after round {r}: no fresh hypotheses to "
                      f"test (all already accepted or ablated).")
                break
            if not round_had_signal:
                print(f"\nSTOPPING after round {r}: no candidate reached "
                      f"the ablation stage.")
                break

    # --- champion model: the 0-1 artifact ----------------------------------
    # Retrain baseline + accepted features as one model and save it for
    # deployment. v1 supports the lgbm backend; torch is an honest skip.
    if save_champion:
        if not accepted_features:
            print("\nno accepted features; champion model not saved.")
        elif model_backend == "lgbm":
            print("\nretraining champion (baseline + "
                  f"{len(accepted_features)} accepted features)...")
            model, feat_names, target_name, is_reg, label_classes = \
                train_champion_lgbm(
                    train, valid, stats, ctx, accepted_features,
                    X_train_base, X_valid_base, brief,
                    metric_name, _target_transform)
            code = {}
            for _n, _f in accepted_features.items():
                try:
                    code[_n] = inspect.getsource(_f.compute)
                except (OSError, TypeError):
                    code[_n] = "<source unavailable>"
            from deploy import save_champion_model
            parent = os.path.dirname(save_champion)
            if parent:
                os.makedirs(parent, exist_ok=True)
            save_champion_model(
                model, feat_names, save_champion,
                extra={
                    "framework": "lightgbm-regressor"
                    if is_reg else "lightgbm",
                    "target": target_name,
                    "metric": metric_name,
                    "label": (brief or {}).get("label", ""),
                    "problem": (brief or {}).get("problem", ""),
                    "accepted_feature_code": code,
                    "label_classes": label_classes,
                })
            print(f"champion saved: {save_champion} "
                  f"({len(feat_names)} features)")
        else:
            print(f"\nchampion saving for the '{model_backend}' backend is "
                  f"not implemented in v1; skipping (model not saved).")

    return ledger_entries


def train_champion_lgbm(train, valid, stats, ctx, accepted: dict,
                        X_train_base, X_valid_base, brief,
                        metric_name: str, target_transform):
    """Retrain one LightGBM model on baseline + all accepted features.

    Mirrors ablate_lgbm's matrix construction exactly (same base columns,
    accepted features appended in acceptance order), so the saved champion
    matches what the ablations measured. Returns
    (model, feature_names, target_name, is_regression).
    """
    from baseline import (encode_labels, make_model, make_regressor,
                          transform_target)
    if brief is not None:
        from brief_data import parse_target
        target = parse_target(brief.get("label", ""))
    else:
        target = TARGET
    X_train = X_train_base.copy()
    X_valid = X_valid_base.copy()
    for name, f in accepted.items():
        X_train[name] = f.compute(train, ctx).astype("float32") \
            .fillna(0).to_numpy()
        X_valid[name] = f.compute(valid, ctx).astype("float32") \
            .fillna(0).to_numpy()
    y_train_raw = train[target].to_numpy()
    n_classes = None
    if metric_name == "rmse":
        y_train = transform_target(y_train_raw, target_transform)
        model = make_regressor()
        is_regression = True
    elif metric_name == "logloss":
        y_train, _, classes = encode_labels(y_train_raw)
        n_classes = len(classes)
        model = make_model(n_classes if n_classes > 2 else None)
        is_regression = False
    else:
        y_train = y_train_raw
        if getattr(y_train, "dtype", None) == bool:
            y_train = y_train.astype(int)
        model = make_model()
        is_regression = False
    model.fit(X_train, y_train)
    return model, list(X_train.columns), target, is_regression, \
        (None if metric_name == "rmse"
         else [str(c) for c in encode_labels(y_train_raw)[2]])


def print_summary(entries: list[dict]) -> None:
    print(f"\n{'=' * 60}\nEXPERIMENT SUMMARY\n{'=' * 60}")
    print(f"{'round':<6}{'feature':<28}{'verdict':<10}{'delta':<10}reason")
    for e in entries:
        m = e.get("metrics") or {}
        d = m.get("delta_auc", m.get("delta_log_loss", m.get("delta_rmse")))
        dstr = f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"
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
