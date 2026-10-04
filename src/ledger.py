"""Append-only experiment ledger (Days 3-4).

Every hypothesis the agent tests -- accepted or rejected -- lands here as
one JSON line in results/experiments.jsonl: the idea, the exact feature
code, the sources used, the leakage audit, the ablation metrics, and the
verdict with its reason. The demo UI (Days 7-8) reads this file to render
the experiment tree.
"""

from __future__ import annotations

import inspect
import json
import os
import time

LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiments.jsonl"
)


def append_entry(entry: dict, path: str = LEDGER_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_entries(path: str = LEDGER_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_entry(round_no: int, hypothesis, feature, audit_result,
                cost_verdict, metrics: dict | None, verdict: str,
                reason: str, agentic: dict | None = None,
                brief: dict | None = None) -> dict:
    try:
        code = inspect.getsource(feature.compute)
    except (OSError, TypeError):
        code = "<source unavailable>"
    return {
        "round": round_no,
        "hypothesis": hypothesis.text,
        "feature_name": feature.name,
        "feature_code": code,
        "sources": hypothesis.sources,
        "temporal_scope": (feature.provenance or {}).get("temporal_scope"),
        "serving": getattr(feature, "serving", "lookup"),
        "trap": bool(hypothesis.trap),
        "leakage_audit": ({
            "passed": audit_result.passed,
            "reasons": audit_result.reasons,
        } if audit_result is not None else None),
        "serving_cost": ({
            "passed": cost_verdict.passed,
            "units": cost_verdict.units,
            "reasons": cost_verdict.reasons,
        } if cost_verdict is not None else None),
        "metrics": metrics,
        "verdict": verdict,   # accepted | rejected | screened_out | duplicate
        "reason": reason,
        # Agentic trail (LLM engine only): free-form reasoning, critiques,
        # revisions with the feedback that caused them, per-attempt verdicts.
        "agentic": agentic,
        # Brief stamp (brief-driven runs): the problem + temporal contract
        # this verdict was produced under.
        "brief": brief,
    }
