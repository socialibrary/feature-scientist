"""Serving-cost verdict (Days 5-6, judgment layer).

Every feature declares a *serving pattern* -- how the feature would be
computed at request time in production -- via Feature.serving. The cost
model assigns each pattern a relative latency in ms-equivalents:

  row_local     0.5   pure function of the scored row (no extra IO)
  lookup        1.0   O(1) key lookup in a precomputed table (user/movie id)
  history_scan 25.0   scans the user's (or movie's) full event history per request
  external     60.0   network call to an outside service per request

Budgets (demo-tuned, deliberately tight so the trap trips):
  MAX_FEATURE_UNITS  8.0   no single feature may cost more than this
  MAX_TOTAL_UNITS   15.0   accepted features' cumulative cost may not exceed this

A feature can be rejected for cost even with a positive AUC delta -- the
verdict pipeline runs leakage audit -> cost check -> ablation, so an
expensive feature never wastes training time. The ledger records the cost
verdict alongside the leakage audit and the ablation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SERVING_PATTERNS = {
    "row_local": {
        "units": 0.5,
        "desc": "pure function of the scored row; no extra IO at request time",
    },
    "lookup": {
        "units": 1.0,
        "desc": "O(1) lookup of a precomputed value by user/movie id",
    },
    "history_scan": {
        "units": 25.0,
        "desc": "scans the full event history at request time",
    },
    "external": {
        "units": 60.0,
        "desc": "network call to an outside service at request time",
    },
}

MAX_FEATURE_UNITS = 8.0
MAX_TOTAL_UNITS = 15.0


@dataclass
class CostEstimate:
    feature_name: str
    pattern: str
    units: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class CostVerdict:
    feature_name: str
    passed: bool
    units: float
    new_total_units: float
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = "; ".join(self.reasons) if self.reasons else "ok"
        return (f"[{status}] {self.feature_name}: cost {self.units} units "
                f"(total {self.new_total_units:.1f}/{MAX_TOTAL_UNITS}) -- {detail}")


def estimate_cost(feature) -> CostEstimate:
    """Map a feature's declared serving pattern to latency units."""
    pattern = getattr(feature, "serving", None) or "lookup"
    spec = SERVING_PATTERNS.get(pattern)
    if spec is None:
        return CostEstimate(
            feature_name=feature.name,
            pattern=pattern,
            units=float("inf"),
            reasons=[f"unknown serving pattern '{pattern}'; declare one of "
                     f"{sorted(SERVING_PATTERNS)}"],
        )
    reasons = [f"serving='{pattern}': {spec['desc']}"]
    notes = getattr(feature, "serving_notes", "") or ""
    if notes:
        reasons.append(notes)
    return CostEstimate(
        feature_name=feature.name,
        pattern=pattern,
        units=float(spec["units"]),
        reasons=reasons,
    )


def check_budget(feature, accepted_total_units: float = 0.0) -> CostVerdict:
    """Verdict: does this feature fit the serving budget?

    Fail-closed: unknown patterns and per-feature overruns are rejected
    outright; the cumulative budget protects the accepted feature set.
    """
    est = estimate_cost(feature)
    reasons = list(est.reasons)
    passed = True

    if est.units == float("inf"):
        passed = False
    elif est.units > MAX_FEATURE_UNITS:
        passed = False
        reasons.append(
            f"{est.units} units > per-feature budget {MAX_FEATURE_UNITS} -- "
            f"SERVING TOO EXPENSIVE"
        )

    new_total = accepted_total_units + (0.0 if est.units == float("inf")
                                        else est.units)
    if passed and new_total > MAX_TOTAL_UNITS:
        passed = False
        reasons.append(
            f"cumulative {new_total:.1f} units > total budget {MAX_TOTAL_UNITS}"
        )

    return CostVerdict(
        feature_name=feature.name,
        passed=passed,
        units=est.units,
        new_total_units=new_total,
        reasons=reasons,
    )
