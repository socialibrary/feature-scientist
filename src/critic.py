"""The skeptic: a second LLM role that tries to kill hypotheses BEFORE they
cost audit/ablation time.

review() sends the proposed spec (text + code + declared provenance +
serving + the ctx table metadata + PIT rules + budgets) to the backend's
critique role and returns a CriticVerdict. revise_after_critique() feeds a
rejection back to the proposer for one revision attempt.

The critic is advisory -- the deterministic leakage audit remains the hard
gate. A good critic catch saves an audit cycle; a missed catch is still
caught downstream (defense in depth).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from codegen import parse_llm_response
from llm_backend import LLMBackend, LLMError


@dataclass
class CriticVerdict:
    passed: bool
    critique: str
    suggested_fix: str = ""


CRITIC_RULES = """\
You are reviewing a proposed ML feature for a movie recommender. Be adversarial:
your job is to find reasons this feature should NOT be built.

Point-in-time (PIT) rules — REJECT if violated:
- For each scored row with prediction time T, the feature may only use source
  data with time <= T, time-invariant metadata, or train-period aggregates.
- These ctx sources are DANGEROUS (future data): ctx["movie_mean_all"]
  (train+validation means), ctx["imdb_map_avg"] / the "imdb_averageRating" /
  "imdb_numVotes" columns (all-time IMDb aggregates through Oct 2026).
- A feature claiming temporal_scope="pit_correct" whose code shows no as-of
  logic (no filtering by prediction time, no searchsorted/prefix lookups) is
  lying — reject it.

Serving rules — REJECT if violated:
- Declared patterns and budgets: row_local 0.5u, lookup 1.0u,
  history_scan 25.0u, external 60.0u. Per-feature budget: 8.0u.
- A history_scan feature can essentially never ship; demand a row_local or
  lookup reformulation unless the signal is extraordinary (it isn't).

Plausibility — REJECT if:
- The "signal" is circular with the target (e.g. aggregates of the label).
- The computation cannot produce what the text claims.

Respond ONLY with JSON:
{"verdict": "pass" | "reject", "critique": "<specific reasons>",
 "suggested_fix": "<concrete revision, empty if passing>"}
"""


def build_critique_prompt(spec: dict, ctx_meta: dict) -> str:
    return (
        CRITIC_RULES
        + "\n\nProposed feature spec:\n"
        + json.dumps({k: spec.get(k) for k in
                      ("name", "text", "sources", "point_in_time",
                       "temporal_scope", "tower", "serving", "serving_notes")},
                     indent=2)
        + "\n\nGenerated code:\n```python\n" + spec.get("code", "") + "\n```"
        + "\n\nCtx table metadata (built_from tells you what each table may read):\n"
        + json.dumps(ctx_meta, indent=2, default=str)[:2000]
    )


def review(spec: dict, backend: LLMBackend, ctx_meta: dict) -> CriticVerdict:
    """One skeptic pass over a proposed spec."""
    raw = backend.critique(build_critique_prompt(spec, ctx_meta))
    try:
        payload = parse_llm_response(raw)
    except LLMError as e:
        # Unparseable critique: fail open (advisory role) but record it.
        return CriticVerdict(
            passed=True,
            critique=f"critic response unparseable, skipped: {e}",
            suggested_fix="",
        )
    verdict = str(payload.get("verdict", "pass")).strip().lower()
    return CriticVerdict(
        passed=(verdict == "pass"),
        critique=str(payload.get("critique", "")),
        suggested_fix=str(payload.get("suggested_fix", "")),
    )


def revise_after_critique(spec: dict, verdict: CriticVerdict,
                          backend: LLMBackend) -> dict:
    """Feed the critic's rejection back to the proposer; return the revised
    full spec (same schema as propose output)."""
    prompt = (
        "Your proposed feature was REJECTED by the skeptic reviewer:\n\n"
        f"Original spec:\n{json.dumps(spec, indent=2)}\n\n"
        f"Critique:\n{verdict.critique}\n\n"
        f"Suggested fix:\n{verdict.suggested_fix}\n\n"
        "Revise the feature to address the critique. You may change the "
        "approach entirely if the original cannot be saved (e.g. replace an "
        "unshippable history_scan with a row_local signal). Return the FULL "
        "revised spec as JSON with exactly these keys: name, text, sources, "
        "point_in_time, temporal_scope, tower, serving, serving_notes, code "
        "(a complete `def compute(df, ctx):`). The code runs with only "
        "pd, np and safe builtins available — no imports, no I/O. "
        "temporal_scope must honestly describe what the code does."
    )
    raw = backend.revise(prompt)
    payload = parse_llm_response(raw)
    # The revise role returns the spec directly (not wrapped in hypotheses).
    if isinstance(payload, dict) and "hypotheses" in payload:
        hyps = payload["hypotheses"]
        if not hyps:
            raise LLMError("revise returned no hypotheses")
        return hyps[0]
    return payload
