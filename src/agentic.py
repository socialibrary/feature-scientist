"""Agentic orchestration for one hypothesis.

Full path: propose (already done by the engine) -> critic -> at most one
critic-driven revision -> materialize (AST safety check) -> deterministic
leakage audit -> at most one audit-driven revision -> serving-cost check ->
at most one cost-driven revision -> ablate survivors -> ledger.

Every stage is recorded in the entry's "agentic" block: the free-form
reasoning, each critique, each revision (with the feedback that caused it
and the new code), and each attempt's verdict. The ledger is the reasoning
trail.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation import ablate
from codegen import (CodegenError, materialize_feature, parse_llm_response,
                     smoke_test)
from critic import review, revise_after_critique
from leakage import audit_feature
from ledger import append_entry, build_entry
from llm_backend import LLMError
from serving_cost import check_budget


def revise_with_feedback(spec: dict, stage: str, feedback: str,
                         backend, ctx_meta: dict) -> dict:
    """Feed a deterministic rejection (audit/cost) back to the proposer for
    one revision attempt. Returns the revised full spec."""
    prompt = (
        f"Your proposed feature was REJECTED by the deterministic {stage}:\n\n"
        f"Current spec:\n{json.dumps(spec, indent=2)}\n\n"
        f"Rejection reason (exact, from the checker — address it literally):\n"
        f"{feedback}\n\n"
        "Revise the feature to fix the stated problem. If the approach cannot "
        "be saved (e.g. it fundamentally needs future data), propose a "
        "different, honest signal instead. Return the FULL revised spec as "
        "JSON with exactly these keys: name, text, sources, point_in_time, "
        "temporal_scope, tower, serving, serving_notes, code (a complete "
        "`def compute(df, ctx):`). The code runs with only pd, np and safe "
        "builtins — no imports, no I/O. temporal_scope and serving must "
        "honestly describe what the code does."
    )
    raw = backend.revise(prompt)
    payload = parse_llm_response(raw)
    if isinstance(payload, dict) and "hypotheses" in payload:
        hyps = payload["hypotheses"]
        if not hyps:
            raise LLMError("revise returned no hypotheses")
        return hyps[0]
    return payload


def _materialize_or_fail(spec, ctx, train):
    """Build the Feature; returns (feature|None, error|None)."""
    try:
        feature = materialize_feature(spec)
    except CodegenError as e:
        return None, f"codegen failed: {e}"
    try:
        smoke_test(feature, train.head(100), ctx)
    except CodegenError as e:
        return None, f"smoke test failed: {e}"
    except Exception as e:  # noqa: BLE001 - generated code, anything can happen
        return None, f"smoke test raised {type(e).__name__}: {e}"
    return feature, None


def process_agentic(r: int, h, run_ctx: dict) -> tuple[dict, float]:
    """Run one LLM-proposed hypothesis through the full agentic pipeline.

    Returns (ledger_entry, updated_accepted_cost_units).
    """
    ctx = run_ctx["ctx"]
    train, valid, stats = run_ctx["train"], run_ctx["valid"], run_ctx["stats"]
    base_auc, base_ll = run_ctx["base_auc"], run_ctx["base_ll"]
    engine = run_ctx["engine"]
    backend = engine.backend
    ctx_meta = ctx.get("_table_meta", {})
    model_backend = run_ctx["model_backend"]
    accept_delta = run_ctx["accept_delta"]

    spec = dict(h.llm_spec or {})
    agentic = {
        "reasoning": h.reasoning or "",
        "critiques": [],
        "revisions": [],
        "attempts": [],
    }
    print(f"   reasoning: {(h.reasoning or '')[:160]}")

    # ---- 1. critic (advisory) -------------------------------------------
    cv = review(spec, backend, ctx_meta)
    agentic["critiques"].append(
        {"passed": cv.passed, "critique": cv.critique,
         "suggested_fix": cv.suggested_fix})
    print(f"   critic: {'PASS' if cv.passed else 'REJECT'} — "
          f"{cv.critique[:120]}")
    if not cv.passed:
        old_name = spec.get("name")
        spec = revise_after_critique(spec, cv, backend)
        agentic["revisions"].append(
            {"stage": "critic",
             "feedback": cv.critique,
             "from": old_name, "to": spec.get("name"),
             "code": spec.get("code", "")})
        print(f"   revised after critic -> '{spec.get('name')}'")
        cv2 = review(spec, backend, ctx_meta)
        agentic["critiques"].append(
            {"passed": cv2.passed, "critique": cv2.critique,
             "suggested_fix": cv2.suggested_fix})
        print(f"   re-critic: {'PASS' if cv2.passed else 'REJECT'}")
        # Advisory: proceed regardless; the deterministic audit decides.

    # ---- 2. materialize (safety-checked) ----------------------------------
    feature, err = _materialize_or_fail(spec, ctx, train)
    audit = cost = metrics = None
    verdict = reason = None
    if err:
        verdict, reason = "rejected", err
        print(f"   verdict: REJECTED ({reason})")
    else:
        print(f"   built feature '{feature.name}' "
              f"(scope={feature.provenance.get('temporal_scope')}, "
              f"serving={feature.serving})")

        # ---- 3. leakage audit, with ONE revision on reject ---------------
        audit = audit_feature(feature, ctx, valid)
        print(f"   leakage audit: {audit}")
        agentic["attempts"].append(
            {"stage": "audit", "passed": audit.passed,
             "reasons": list(audit.reasons)})
        if not audit.passed:
            fb = "leakage audit rejected: " + "; ".join(audit.reasons)
            old_name = spec.get("name")
            try:
                spec = revise_with_feedback(spec, "leakage audit", fb,
                                            backend, ctx_meta)
                agentic["revisions"].append(
                    {"stage": "audit", "feedback": fb,
                     "from": old_name, "to": spec.get("name"),
                     "code": spec.get("code", "")})
                print(f"   revised after audit -> '{spec.get('name')}'")
                feature, err = _materialize_or_fail(spec, ctx, train)
                if err:
                    verdict, reason = "rejected", err
                else:
                    audit = audit_feature(feature, ctx, valid)
                    print(f"   re-audit: {audit}")
                    agentic["attempts"].append(
                        {"stage": "audit-retry", "passed": audit.passed,
                         "reasons": list(audit.reasons)})
            except (LLMError, CodegenError) as e:
                verdict, reason = "rejected", f"revision failed: {e}"

        # ---- 4. serving-cost check, with ONE revision on reject ----------
        if verdict is None and audit is not None and audit.passed:
            cost = check_budget(feature, run_ctx["accepted_cost_units"])
            print(f"   serving cost: {cost}")
            agentic["attempts"].append(
                {"stage": "cost", "passed": cost.passed,
                 "reasons": list(cost.reasons)})
            if not cost.passed:
                fb = "serving-cost gate rejected: " + "; ".join(cost.reasons)
                old_name = spec.get("name")
                try:
                    spec = revise_with_feedback(spec, "serving-cost gate",
                                                fb, backend, ctx_meta)
                    agentic["revisions"].append(
                        {"stage": "cost", "feedback": fb,
                         "from": old_name, "to": spec.get("name"),
                         "code": spec.get("code", "")})
                    print(f"   revised after cost gate -> '{spec.get('name')}'")
                    feature, err = _materialize_or_fail(spec, ctx, train)
                    if err:
                        verdict, reason = "rejected", err
                    else:
                        # A revision can introduce new leakage: re-audit.
                        audit = audit_feature(feature, ctx, valid)
                        print(f"   re-audit: {audit}")
                        agentic["attempts"].append(
                            {"stage": "audit-retry", "passed": audit.passed,
                             "reasons": list(audit.reasons)})
                        if audit.passed:
                            cost = check_budget(
                                feature, run_ctx["accepted_cost_units"])
                            print(f"   re-cost: {cost}")
                            agentic["attempts"].append(
                                {"stage": "cost-retry", "passed": cost.passed,
                                 "reasons": list(cost.reasons)})
                except (LLMError, CodegenError) as e:
                    verdict, reason = "rejected", f"revision failed: {e}"

        # ---- 5. ablate survivors ------------------------------------------
        if (verdict is None and audit is not None and audit.passed
                and cost is not None and cost.passed):
            print("   ablating (baseline + feature)...")
            metrics = ablate(train, valid, stats, feature, ctx,
                             base_auc, base_ll,
                             prep=run_ctx.get("prep"),
                             X_train_base=run_ctx.get("X_train_base"),
                             X_valid_base=run_ctx.get("X_valid_base"),
                             model=model_backend)
            print(f"   ablation: AUC={metrics['auc']:.4f} "
                  f"(delta {metrics['delta_auc']:+.4f}), "
                  f"logloss={metrics['log_loss']:.4f}")
            agentic["attempts"].append(
                {"stage": "ablation", "passed": True,
                 "delta_auc": metrics["delta_auc"]})
            if metrics["delta_auc"] >= accept_delta:
                verdict = "accepted"
                reason = (f"AUC lift {metrics['delta_auc']:+.4f} >= "
                          f"{accept_delta}")
                run_ctx["accepted_cost_units"] = cost.new_total_units
            else:
                verdict, reason = "rejected", (
                    f"no lift (delta {metrics['delta_auc']:+.4f} < "
                    f"{accept_delta})")
            print(f"   verdict: {verdict.upper()} ({reason})")
        elif verdict is None:
            # Rejected at audit or cost without a successful revision.
            if audit is not None and not audit.passed:
                verdict, reason = "rejected", \
                    "leakage: " + "; ".join(audit.reasons)
            elif cost is not None and not cost.passed:
                verdict, reason = "rejected", \
                    "serving cost: " + "; ".join(cost.reasons)
            else:
                verdict, reason = "rejected", "pipeline did not complete"
            print(f"   verdict: REJECTED ({reason})")

    # ---- 6. ledger ----------------------------------------------------------
    if feature is not None:
        entry = build_entry(r, h, feature, audit, cost, metrics,
                            verdict, reason, agentic=agentic)
        # build_entry can't inspect.getsource an exec'd function reliably;
        # the exact generated code is authoritative.
        entry["feature_code"] = spec.get("code", entry.get("feature_code"))
    else:
        # Nothing materialized: hand-rolled entry so the trail isn't lost.
        entry = {
            "round": r,
            "hypothesis": h.text,
            "feature_name": spec.get("name", h.feature_name),
            "feature_code": spec.get("code", ""),
            "sources": h.sources,
            "verdict": verdict, "reason": reason,
            "agentic": agentic,
        }
    append_entry(entry)
    return entry, run_ctx["accepted_cost_units"]
