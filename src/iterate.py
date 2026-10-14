"""Cumulative feature iteration driver: stack accepted features toward
+0.01 AUC (0.7410 -> >=0.7510) and -1% NE (log-loss 0.5919 -> <=0.5860)
on MovieLens-1M.

How it differs from run_scientist.py / brief.py:
  * Ablations are CUMULATIVE: every candidate is measured marginally against
    the champion (baseline + all accepted features), not against the bare
    baseline. The champion's single-seed metrics are carried forward, so each
    candidate costs exactly one training run.
  * The calibration screen re-runs against the CHAMPION's residuals after
    every acceptance -- that is how the next 0.2pp is found instead of
    rediscovering the same gap.
  * Candidates come from a deterministic gap-driven pool (proven builders +
    new Wikidata-backed builders in iter_features.py), each passing:
    leakage audit -> serving-cost gate -> deterministic skeptic
    (redundancy vs accepted features) -> calibration screen gate ->
    cumulative ablation. Accept on positive marginal AUC gain.
  * Every acceptance triggers a 3-seed champion milestone (mean +/- std);
    the final 1% claim carries error bars. Day-to-day ablations stay
    single-seed for speed.
  * Writes to results/iteration_ledger.jsonl -- NEVER to the shared
    results/experiments.jsonl.

Note on the LLM proposer: the scripted backend's canned script contains
only two hypotheses, both already covered by the deterministic pool
(movie_age_days). Re-proposing them adds no novelty after dedup, so this
driver uses the gap-driven pool as its proposer and documents the choice.
The deterministic gates (audit, cost, screen, skeptic) are the filters
that matter for a metric target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ablation import ablate_cumulative, evaluate_champion, \
    check_degenerate, _candidate_cols
from baseline import TARGET, build_train_only_stats
from calibration import calibration_screen
from data import load_all
from features import Feature
from leakage import audit_feature
from ledger import append_entry, build_entry
from scientist import (
    build_ctx,
    build_director_track_record_pit,
    build_movie_age_days,
    build_movie_velocity_30d,
    build_user_genre_affinity,
)
from iter_features import (
    CANDIDATE_BUILDERS,
    merge_wikidata_ctx,
)
from serving_cost import check_budget
from split import temporal_split
from two_tower import prepare_tower_inputs, train_two_tower

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results")
ITER_LEDGER = os.path.join(RESULTS_DIR, "iteration_ledger.jsonl")

TARGET_AUC = 0.7596  # +0.01 absolute on the converged 8-epoch base (0.7496)
TARGET_LL = 0.5780  # 1% NE under the converged 8-epoch base ll (0.5838)
MILESTONE_SEEDS = (42, 43, 44)
BASE_SEED = 42
MAX_CORR_REDUNDANT = 0.95


def _hyp(text: str, sources: list[str]) -> SimpleNamespace:
    return SimpleNamespace(text=text, sources=sources, trap=False)


def _champion_snapshot(champion: dict, n_accepted: int, base: dict) -> dict:
    return {
        "auc": round(champion["auc"], 4),
        "log_loss": round(champion["log_loss"], 4),
        "delta_auc_vs_base": round(champion["auc"] - base["auc"], 4),
        "delta_ll_vs_base": round(champion["log_loss"] - base["ll"], 4),
        "n_accepted": n_accepted,
    }


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--max-ablations-per-round", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ledger", default=ITER_LEDGER)
    ap.add_argument("--dry-run", action="store_true",
                    help="epochs=1, max 1 round, no ledger writes (smoke test)")
    args = ap.parse_args()
    if args.dry_run:
        args.epochs = 1
        args.rounds = 1

    t0 = time.time()
    print("== cumulative iteration: target AUC>=%.4f, logloss<=%.4f =="
          % (TARGET_AUC, TARGET_LL))
    df = load_all()
    train, valid, cutoff_ts = temporal_split(df)
    stats = build_train_only_stats(train)
    prep = prepare_tower_inputs(train, valid, stats)
    ctx = build_ctx(train, valid, stats, cutoff_ts)
    print("building wikidata ctx tables...")
    ctx = merge_wikidata_ctx(ctx, train)
    print("ctx ready: actor_prefix=%d actors, movie_to_cast=%d movies" % (
        len(ctx["actor_prefix"]), len(ctx["movie_to_cast"])))

    yv = valid[TARGET].to_numpy()
    train_kwargs = {"epochs": args.epochs}

    # ---- base baseline: single-seed (for marginals) + 3-seed (for claims)
    print("base baseline: single-seed train...")
    bres = train_two_tower(prep, seed=BASE_SEED, verbose=False,
                           **train_kwargs)
    base_single = {"auc": float(bres["auc"]), "ll": float(bres["log_loss"]),
                   "proba": bres["proba"]}
    print("base baseline: 3-seed eval...")
    base_multi = evaluate_champion(train, valid, stats, [], ctx, prep=prep,
                                   seeds=MILESTONE_SEEDS, verbose=True,
                                   **train_kwargs)
    base = {"auc": base_single["auc"], "ll": base_single["ll"]}
    print(f"base: single AUC={base['auc']:.4f} ll={base['ll']:.4f} | "
          f"3-seed AUC={base_multi['auc_mean']:.4f}±{base_multi['auc_std']:.4f} "
          f"ll={base_multi['ll_mean']:.4f}±{base_multi['ll_std']:.4f}")

    champion = {"auc": base_single["auc"], "log_loss": base_single["ll"],
                "proba": base_single["proba"]}
    accepted: list[Feature] = []
    accepted_values: dict[str, np.ndarray] = {}  # name -> valid values
    accepted_cost = 0.0
    tried: set[str] = set()
    progression: list[dict] = []

    builders = list(CANDIDATE_BUILDERS) + [
        build_movie_age_days,
        build_director_track_record_pit,
        build_user_genre_affinity,
        build_movie_velocity_30d,
    ]
    # dedupe builders by name (built once to learn names)
    seen, pool = set(), []
    for b in builders:
        f = b(ctx)
        if f.name not in seen:
            seen.add(f.name)
            pool.append((b, f))
    print(f"candidate pool: {len(pool)} builders: "
          f"{[f.name for _, f in pool]}")

    def log_entry(round_no, feature, hyp_text, sources, audit, cost,
                  metrics, verdict, reason, attempts):
        entry = build_entry(
            round_no, _hyp(hyp_text, sources), feature, audit, cost,
            metrics, verdict, reason,
            agentic={"driver": "iterate.py", "attempts": attempts},
            champion=_champion_snapshot(champion, len(accepted), base))
        if not args.dry_run:
            append_entry(entry, path=args.ledger)
        return entry

    stop_reason = None
    for r in range(1, args.rounds + 1):
        print(f"\n===== round {r} (champion: "
              f"AUC={champion['auc']:.4f} "
              f"total {champion['auc'] - base['auc']:+.4f}, "
              f"ll={champion['log_loss']:.4f}) =====")
        # ---- calibration battery vs CHAMPION residuals ----
        candidates = []
        for b, f in pool:
            if f.name in tried or f.name in {a.name for a in accepted}:
                continue
            vals = f.compute(valid, ctx).astype("float32").fillna(0).to_numpy()
            scr = calibration_screen(yv, champion["proba"], vals, f.name)
            if scr["verdict"] == "signal":
                candidates.append((scr["max_significant_gap_pp"] or 0.0, b, f,
                                   vals, scr["summary"]))
        candidates.sort(key=lambda t: -t[0])
        print(f"screen vs champion: {len(candidates)} signals / "
              f"{len(pool) - len(tried)} tried-pool")
        for gap, _, f, _, summ in candidates[:5]:
            print(f"   {f.name}: gap {gap:.2f}pp -- {summ[:100]}")

        if not candidates:
            stop_reason = "no significant calibration gaps vs champion"
            print("STOP:", stop_reason)
            break

        ablated_this_round = 0
        for gap, b, f, vals, summ in candidates:
            if ablated_this_round >= args.max_ablations_per_round:
                break
            tried.add(f.name)
            attempts = [{"stage": "screen", "verdict": "signal",
                         "gap_pp": round(gap, 2), "summary": summ}]
            print(f"\n-- candidate: {f.name} "
                  f"(tower={(f.provenance or {}).get('tower')})")

            # deterministic skeptic: redundancy vs accepted features
            redundant_with = None
            for aname, avals in accepted_values.items():
                c = np.corrcoef(vals, avals)[0, 1]
                if abs(c) > MAX_CORR_REDUNDANT:
                    redundant_with = (aname, c)
                    break
            if redundant_with:
                reason = (f"skeptic: redundant with accepted "
                          f"'{redundant_with[0]}' (|corr|={redundant_with[1]:.3f})")
                print(f"   verdict: REJECTED ({reason})")
                attempts.append({"stage": "skeptic", "verdict": "reject",
                                 "reason": reason})
                log_entry(r, f, f"gap-driven: {summ}", ["calibration"],
                          None, None, None, "rejected", reason, attempts)
                continue
            attempts.append({"stage": "skeptic", "verdict": "pass"})

            audit = audit_feature(f, ctx, valid)
            print(f"   audit: {'PASS' if audit.passed else 'FAIL'}")
            attempts.append({"stage": "audit", "verdict":
                             "pass" if audit.passed else "fail",
                             "reasons": audit.reasons})
            if not audit.passed:
                reason = "leakage: " + "; ".join(audit.reasons)
                print(f"   verdict: REJECTED ({reason})")
                log_entry(r, f, f"gap-driven: {summ}", ["calibration"],
                          audit, None, None, "rejected", reason, attempts)
                continue

            cost = check_budget(f, accepted_cost)
            print(f"   cost: {'PASS' if cost.passed else 'FAIL'} "
                  f"({cost.units} units)")
            attempts.append({"stage": "cost", "verdict":
                             "pass" if cost.passed else "fail",
                             "units": cost.units})
            if not cost.passed:
                reason = "serving cost: " + "; ".join(cost.reasons)
                print(f"   verdict: REJECTED ({reason})")
                log_entry(r, f, f"gap-driven: {summ}", ["calibration"],
                          audit, cost, None, "rejected", reason, attempts)
                continue

            # Degenerate-column guard: a train-constant feature is unlearnable
            # (the per-feature standardizer divides by ~zero and training can
            # diverge). Fail fast with a clear verdict instead of training.
            _, ctr, _ = _candidate_cols(f, train, valid, ctx)
            deg_reason = check_degenerate(ctr, f.name)
            if deg_reason:
                print(f"   verdict: REJECTED ({deg_reason})")
                attempts.append({"stage": "degenerate_check",
                                 "verdict": "fail", "reason": deg_reason})
                log_entry(r, f, f"gap-driven: {summ}", ["calibration"],
                          audit, cost, None, "rejected", deg_reason, attempts)
                continue

            print(f"   cumulative ablation (champion + {f.name})...")
            m = ablate_cumulative(train, valid, stats, accepted, f, ctx,
                                  base["auc"], base["ll"],
                                  champion["auc"], champion["log_loss"],
                                  prep=prep, seed=BASE_SEED, verbose=False,
                                  **train_kwargs)
            ablated_this_round += 1
            attempts.append({"stage": "ablation", "passed": True,
                             "marginal_delta_auc": m["marginal_delta_auc"],
                             "total_delta_auc": m["total_delta_auc"]})
            print(f"   ablation: AUC={m['auc']:.4f} "
                  f"(marginal {m['marginal_delta_auc']:+.4f}, "
                  f"total {m['total_delta_auc']:+.4f}), "
                  f"ll={m['log_loss']:.4f} "
                  f"(marginal {m['marginal_delta_ll']:+.4f})")
            if m["marginal_delta_auc"] > 0:
                verdict, reason = "accepted", (
                    f"marginal AUC {m['marginal_delta_auc']:+.4f} > 0 "
                    f"(total {m['total_delta_auc']:+.4f})")
                prev_champion = dict(champion)
                accepted.append(f)
                accepted_values[f.name] = vals
                accepted_cost = cost.new_total_units \
                    if hasattr(cost, "new_total_units") else accepted_cost
                champion = {"auc": m["auc"], "log_loss": m["log_loss"],
                            "proba": m["proba"]}
                print(f"   verdict: ACCEPTED ({reason})")
                print("   3-seed champion milestone...")
                ms = evaluate_champion(train, valid, stats, accepted, ctx,
                                       prep=prep, seeds=MILESTONE_SEEDS,
                                       verbose=True, **train_kwargs)
                # Gate on the 3-seed mean: a lucky single seed must not lock in.
                prev_mean = (progression[-1]["auc_mean"] if progression
                             else base_multi["auc_mean"])
                milestone_marginal = ms["auc_mean"] - prev_mean
                if milestone_marginal <= 0.0005:
                    # Revert: the milestone overturned the single-seed accept.
                    print(f"   REVERTED: 3-seed mean marginal "
                          f"{milestone_marginal:+.4f} <= 0.0005 "
                          f"(single-seed was {m['marginal_delta_auc']:+.4f})")
                    accepted.pop()
                    accepted_values.pop(f.name, None)
                    champion = prev_champion
                    verdict, reason = "reverted", (
                        f"3-seed milestone overturned single-seed accept "
                        f"(mean marginal {milestone_marginal:+.4f})")
                    log_entry(r, f, f"gap-driven: {summ}", ["calibration"],
                              audit, cost, {**m, "milestone_3seed": {
                                  k: ms[k] for k in
                                  ("auc_mean", "auc_std",
                                   "ll_mean", "ll_std")}},
                              verdict, reason, attempts)
                    continue
                progression.append({
                    "n_accepted": len(accepted),
                    "feature": f.name,
                    "marginal_delta_auc": m["marginal_delta_auc"],
                    "total_delta_auc": m["total_delta_auc"],
                    "total_delta_ll": m["total_delta_ll"],
                    "auc_mean": ms["auc_mean"], "auc_std": ms["auc_std"],
                    "ll_mean": ms["ll_mean"], "ll_std": ms["ll_std"],
                })
                log_entry(r, f, f"gap-driven: {summ}", ["calibration"],
                          audit, cost, {**m, "milestone_3seed": {
                              k: ms[k] for k in
                              ("auc_mean", "auc_std", "ll_mean", "ll_std")}},
                          verdict, reason, attempts)
                # champion changed -> re-screen remaining candidates next round
                break
            else:
                verdict, reason = "rejected", (
                    f"no marginal lift "
                    f"(marginal {m['marginal_delta_auc']:+.4f} <= 0)")
                print(f"   verdict: REJECTED ({reason})")
                log_entry(r, f, f"gap-driven: {summ}", ["calibration"],
                          audit, cost, m, verdict, reason, attempts)

        # ---- stop checks ----
        if champion["auc"] >= TARGET_AUC and champion["log_loss"] <= TARGET_LL:
            stop_reason = (f"target reached: AUC={champion['auc']:.4f} "
                           f"ll={champion['log_loss']:.4f}")
            print("STOP:", stop_reason)
            break
    else:
        stop_reason = f"max rounds ({args.rounds}) reached"

    # ---- final 3-seed champion eval ----
    print("\n== final champion 3-seed eval ==")
    final = evaluate_champion(train, valid, stats, accepted, ctx, prep=prep,
                              seeds=MILESTONE_SEEDS, verbose=True,
                              **train_kwargs)
    total_auc = final["auc_mean"] - base_multi["auc_mean"]
    total_ll = final["ll_mean"] - base_multi["ll_mean"]
    print(f"\nCHAMPION: {len(accepted)} features stacked: "
          f"{[a.name for a in accepted]}")
    print(f"  3-seed AUC: {final['auc_mean']:.4f}±{final['auc_std']:.4f} "
          f"(base {base_multi['auc_mean']:.4f}±{base_multi['auc_std']:.4f}) "
          f"=> total {total_auc:+.4f}")
    print(f"  3-seed ll:  {final['ll_mean']:.4f}±{final['ll_std']:.4f} "
          f"(base {base_multi['ll_mean']:.4f}±{base_multi['ll_std']:.4f}) "
          f"=> total {total_ll:+.4f}")
    print(f"  target: AUC>=0.7510 ({'HIT' if final['auc_mean'] >= TARGET_AUC else 'miss'}), "
          f"ll<=0.5860 ({'HIT' if final['ll_mean'] <= TARGET_LL else 'miss'})")
    print(f"  stop: {stop_reason} ({time.time() - t0:.0f}s total)")

    summary = {
        "n_accepted": len(accepted),
        "accepted": [a.name for a in accepted],
        "progression": progression,
        "final_auc_mean": final["auc_mean"], "final_auc_std": final["auc_std"],
        "final_ll_mean": final["ll_mean"], "final_ll_std": final["ll_std"],
        "base_auc_mean": base_multi["auc_mean"],
        "base_ll_mean": base_multi["ll_mean"],
        "total_delta_auc": round(total_auc, 4),
        "total_delta_ll": round(total_ll, 4),
        "target_hit": bool(final["auc_mean"] >= TARGET_AUC
                           and final["ll_mean"] <= TARGET_LL),
        "stop_reason": stop_reason,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "iteration_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    main()
