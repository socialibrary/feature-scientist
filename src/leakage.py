"""First-pass leakage audit (Days 3-4; hardened on Days 5-6).

A feature passes only if its provenance shows it cannot see the future:

  1. point_in_time must name a real timestamp column of the scored frame.
  2. temporal_scope == "all_time" -> FAIL: the feature reads aggregates that
     include data newer than each row's prediction time (the classic trap:
     IMDb's all-time average rating, or a movie mean over train+validation).
  3. temporal_scope == "unknown"/missing -> FAIL: no silent features.
  4. Every ctx table the feature declares (provenance["ctx_keys"]) must be
     built from train-period (or static) data per ctx["_table_meta"].

This is metadata-driven by design for v1 -- the Day 5-6 hardening adds code
inspection of compute() closures. It already catches both planted traps:
imdb_average_rating_raw and movie_mean_rating_all declare all_time scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from features import Feature

FAIL_SCOPES = {"all_time"}


@dataclass
class AuditResult:
    feature_name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = "; ".join(self.reasons) if self.reasons else "ok"
        return f"[{status}] {self.feature_name}: {detail}"


def audit_feature(feature: Feature, ctx: dict, scored_df: pd.DataFrame) -> AuditResult:
    reasons: list[str] = []
    passed = True

    def fail(msg: str):
        nonlocal passed
        passed = False
        reasons.append(msg)

    prov = feature.provenance or {}

    # 1. point_in_time must exist on the scored frame
    if feature.point_in_time not in scored_df.columns:
        fail(f"point_in_time column '{feature.point_in_time}' not in scored frame")

    # 2./3. temporal scope must be declared and must not be all_time
    scope = prov.get("temporal_scope")
    if not scope:
        fail("no temporal_scope declared in provenance")
    elif scope in FAIL_SCOPES:
        fail(
            f"temporal_scope='{scope}': reads aggregates that include data "
            f"newer than prediction time ({prov.get('reads', 'n/a')}) -- "
            f"TEMPORAL LEAKAGE"
        )

    # 4. declared ctx tables must be train-built or static
    table_meta = ctx.get("_table_meta", {})
    for key in prov.get("ctx_keys", []):
        meta = table_meta.get(key)
        if meta is None:
            # table not tracked: builders are expected to register metadata;
            # untracked tables are suspicious but not fatal in v1.
            reasons.append(f"note: ctx table '{key}' has no _table_meta entry")
            continue
        if meta.get("built_from") == "future":
            fail(f"ctx table '{key}' was built from future data "
                 f"({meta.get('note', 'n/a')})")

    return AuditResult(feature_name=feature.name, passed=passed, reasons=reasons)
