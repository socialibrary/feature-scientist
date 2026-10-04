"""Leakage audit, hardened (Days 5-6): verify, don't trust.

Two layers:

  A. Declaration checks (v1): point_in_time names a real column,
     temporal_scope is sane, declared ctx tables are train-built/static.

  B. Code inspection (new): parse compute()'s AST and closure variables and
     check what the code ACTUALLY touches, regardless of what the
     declaration claims:
       1. ctx keys read off the ctx parameter vs DANGEROUS_CTX_KEYS
          (e.g. c["movie_mean_all"], c["imdb_map_avg"]).
       2. closure/free variables that ARE dangerous ctx objects
          (catches `s = ctx["movie_mean_all"]` laundered through a local
          name, with the declaration scrubbed to look clean).
       3. string literals naming dangerous columns
          (imdb_averageRating / imdb_numVotes all-time aggregates).
       4. temporal-guard check: a feature claiming temporal_scope=
          "pit_correct" whose code never references the prediction
          timestamp (or as-of/searchsorted logic) is lying -> FAIL.

Fail-closed: if the source is unavailable for inspection, the feature is
rejected -- a feature we cannot verify is a feature we cannot trust.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field

import pandas as pd

from features import Feature

FAIL_SCOPES = {"all_time"}

# ---------------------------------------------------------------------------
# Known-dangerous sources. The audit fails any feature whose CODE touches
# these, no matter what its provenance declaration claims.
# ---------------------------------------------------------------------------
DANGEROUS_CTX_KEYS = {
    # ctx key -> why it is dangerous
    "movie_mean_all":
        "ctx table built from train+validation rows (future data)",
    "imdb_map_avg":
        "IMDb all-time averageRating (votes through Oct 2026, incl. "
        "post-prediction)",
}
DANGEROUS_COLUMNS = {
    # string literal -> why it is dangerous
    "imdb_averageRating": "IMDb all-time aggregate column",
    "imdb_numVotes": "IMDb all-time aggregate column",
}
# Free-variable names that are suspicious when bound to data objects.
DANGEROUS_NAMES = set(DANGEROUS_CTX_KEYS)


@dataclass
class AuditResult:
    feature_name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = "; ".join(self.reasons) if self.reasons else "ok"
        return f"[{status}] {self.feature_name}: {detail}"


# ---------------------------------------------------------------------------
# AST helpers: what does compute() actually do?
# ---------------------------------------------------------------------------

def _get_tree(fn):
    """(ast.Module, param_names) for fn, or (None, []) if unavailable."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None, []
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None, []
    params: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args]
            break
    return tree, params


def _ctx_param_names(params: list[str]) -> set[str]:
    """Names that plausibly refer to the ctx argument."""
    names = set(params[1:2])  # conventional second positional arg
    names |= {p for p in params if p in {"ctx", "c", "context"}}
    return names


def _ctx_subscript_keys(tree: ast.AST, ctx_names: set[str]) -> set[str]:
    """Literal keys in ctx['key'] / c['key'] subscripts."""
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in ctx_names):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                keys.add(sl.value)
    return keys


def _string_literals(tree: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _called_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _name_ids(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _closure_vars(fn) -> dict[str, object]:
    try:
        cv = inspect.getclosurevars(fn)
    except TypeError:
        return {}
    merged: dict[str, object] = {}
    merged.update(cv.nonlocals)
    merged.update(cv.globals)
    return merged


def _is_data_like(val: object) -> bool:
    return isinstance(val, (pd.Series, pd.DataFrame, dict, list, tuple)) \
        or type(val).__name__ == "ndarray"


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def audit_feature(feature: Feature, ctx: dict, scored_df: pd.DataFrame) -> AuditResult:
    reasons: list[str] = []
    passed = True

    def fail(msg: str):
        nonlocal passed
        passed = False
        reasons.append(msg)

    prov = feature.provenance or {}

    # ---- A. declaration checks (v1) -------------------------------------
    if feature.point_in_time not in scored_df.columns:
        fail(f"point_in_time column '{feature.point_in_time}' not in scored frame")

    scope = prov.get("temporal_scope")
    if not scope:
        fail("no temporal_scope declared in provenance")
    elif scope in FAIL_SCOPES:
        fail(
            f"temporal_scope='{scope}': reads aggregates that include data "
            f"newer than prediction time ({prov.get('reads', 'n/a')}) -- "
            f"TEMPORAL LEAKAGE"
        )

    table_meta = ctx.get("_table_meta", {})
    for key in prov.get("ctx_keys", []):
        meta = table_meta.get(key)
        if meta is None:
            reasons.append(f"note: ctx table '{key}' has no _table_meta entry")
            continue
        if meta.get("built_from") == "future":
            fail(f"ctx table '{key}' was built from future data "
                 f"({meta.get('note', 'n/a')})")

    # ---- B. code inspection: verify, don't trust ------------------------
    tree, params = _get_tree(feature.compute)
    if tree is None:
        fail("source code unavailable for inspection -- cannot verify; "
             "rejected (fail-closed)")
        return AuditResult(feature_name=feature.name, passed=passed,
                           reasons=reasons)

    ctx_names = _ctx_param_names(params)

    # B1. dangerous ctx keys read off the ctx parameter in code
    touched = _ctx_subscript_keys(tree, ctx_names)
    for key in sorted(touched & set(DANGEROUS_CTX_KEYS)):
        fail(f"code reads ctx['{key}']: {DANGEROUS_CTX_KEYS[key]} -- "
             f"TEMPORAL LEAKAGE (code inspection)")

    # B2. closure/free variables that ARE dangerous objects (laundering check)
    danger_ids = {id(ctx[k]): (k, DANGEROUS_CTX_KEYS[k])
                  for k in DANGEROUS_CTX_KEYS if k in ctx}
    for name, val in _closure_vars(feature.compute).items():
        if id(val) in danger_ids:
            key, why = danger_ids[id(val)]
            fail(f"closure variable '{name}' IS ctx['{key}'] ({why}) -- "
                 f"TEMPORAL LEAKAGE (code inspection)")
        elif name in DANGEROUS_NAMES and _is_data_like(val):
            fail(f"closure variable '{name}' looks like a dangerous source -- "
                 f"TEMPORAL LEAKAGE (code inspection)")

    # B3. dangerous column literals anywhere in the code
    lits = _string_literals(tree)
    for col in sorted(lits & set(DANGEROUS_COLUMNS)):
        fail(f"code references '{col}': {DANGEROUS_COLUMNS[col]} -- "
             f"TEMPORAL LEAKAGE (code inspection)")

    # B4. temporal-guard check: pit_correct claims must show their work
    if scope == "pit_correct":
        pit = feature.point_in_time
        names = _name_ids(tree)
        calls = _called_names(tree)
        has_pit_ref = pit in names or pit in lits
        has_asof = "searchsorted" in calls
        if not (has_pit_ref or has_asof):
            fail(f"claims temporal_scope='pit_correct' but the code never "
                 f"references prediction time ('{pit}') or as-of logic -- "
                 f"declaration not credible (code inspection)")

    if passed:
        reasons.append("code inspection: no dangerous sources, "
                       "temporal claims check out")
    return AuditResult(feature_name=feature.name, passed=passed,
                       reasons=reasons)
