"""Turn LLM-produced specs into executable Features, safely.

Pipeline: parse_llm_response -> validate_spec -> safety_check (AST) ->
materialize_feature (writes a real .py file under results/gen_features/ so
the hardened leakage audit can inspect.getsource() it, then execs it in a
restricted namespace).

Fail-closed at every step: unparseable output, invalid spec, or unsafe code
raises; nothing generated ever executes before the AST safety check AND the
deterministic leakage audit both pass.
"""

from __future__ import annotations

import ast
import json
import os
import re

import numpy as np
import pandas as pd

from features import Feature
from llm_backend import LLMError
from serving_cost import SERVING_PATTERNS

GEN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "gen_features"
)

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TOWERS = {"user", "movie", "wide"}
TEMPORAL_SCOPES = {"train_only", "pit_correct", "static", "all_time"}

# Builtins available to generated code: enough for numeric/pandas work,
# nothing that touches the system.
SAFE_BUILTINS = {
    "float": float, "int": int, "bool": bool, "str": str,
    "len": len, "zip": zip, "range": range, "enumerate": enumerate,
    "min": min, "max": max, "abs": abs, "round": round,
    "dict": dict, "list": list, "set": set, "tuple": tuple,
    "sorted": sorted, "isinstance": isinstance,
}

# Attribute roots that must never appear in generated code.
FORBIDDEN_ROOTS = {"os", "sys", "subprocess", "pathlib", "shutil",
                   "socket", "importlib", "builtins"}
FORBIDDEN_CALLS = {"eval", "exec", "open", "__import__", "compile", "input"}


class CodegenError(LLMError):
    """Unparseable response, invalid spec, or failed materialization."""


class CodegenSafetyError(CodegenError):
    """Generated code failed the AST safety check."""


def parse_llm_response(text: str) -> dict:
    """Parse the model's response: JSON mode first, ```json fence fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    raise CodegenError(
        "could not parse LLM response as JSON (tried raw + ```json fence); "
        f"first 200 chars: {text[:200]!r}")


def validate_spec(spec: dict) -> dict:
    """Check a hypothesis spec dict has everything the pipeline needs."""
    required = ["name", "text", "sources", "point_in_time", "temporal_scope",
                "tower", "serving", "code"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise CodegenError(f"spec missing keys: {missing}")
    if not NAME_RE.match(spec["name"]):
        raise CodegenError(f"invalid feature name: {spec['name']!r}")
    if spec["tower"] not in TOWERS:
        raise CodegenError(f"unknown tower {spec['tower']!r}; want {TOWERS}")
    if spec["serving"] not in SERVING_PATTERNS:
        raise CodegenError(
            f"unknown serving pattern {spec['serving']!r}; "
            f"want {sorted(SERVING_PATTERNS)}")
    if spec["temporal_scope"] not in TEMPORAL_SCOPES:
        raise CodegenError(
            f"unknown temporal_scope {spec['temporal_scope']!r}; "
            f"want {TEMPORAL_SCOPES}")
    if spec["point_in_time"] != "timestamp":
        raise CodegenError(
            f"unsupported point_in_time {spec['point_in_time']!r}; "
            "only 'timestamp' is supported")
    if not isinstance(spec["code"], str) or "def " not in spec["code"]:
        raise CodegenError("spec['code'] must be a non-empty function definition")
    return spec


def safety_check(code: str) -> list[str]:
    """AST safety scan of generated code. Returns a list of violations
    (empty = safe). No imports, no I/O, no dynamic execution, no system
    modules -- the code may only compute with pd/np and safe builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("import statements are forbidden")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                violations.append(f"forbidden call: {f.id}()")
            elif isinstance(f, ast.Attribute):
                root = f
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in FORBIDDEN_ROOTS:
                    violations.append(
                        f"forbidden module access: {root.id}.{f.attr}")
    return violations


def _as_function(code: str) -> str:
    """Accept a full `def compute(df, ctx):` or a bare body; return source
    with the def at column 0."""
    if re.search(r"^def\s+compute\s*\(", code, re.MULTILINE):
        return code
    body = "\n".join("    " + ln if ln.strip() else ln
                     for ln in code.splitlines())
    return f"def compute(df, ctx):\n{body}\n"


def materialize_feature(spec: dict, gen_dir: str = GEN_DIR) -> Feature:
    """Validate + safety-check + write-to-file + exec a spec into a Feature.

    The code is written to a real file (results/gen_features/<name>.py) so
    the hardened leakage audit's inspect.getsource() works on it -- the
    audit inspects the exact code the agent wrote, not a proxy.
    """
    spec = validate_spec(spec)
    src = _as_function(spec["code"])
    violations = safety_check(src)
    if violations:
        raise CodegenSafetyError(
            "generated code failed safety check: " + "; ".join(violations))

    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, f"{spec['name']}.py")
    with open(path, "w") as f:
        f.write(src)

    namespace: dict = {"pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
    try:
        code_obj = compile(src, path, "exec")
        exec(code_obj, namespace)
    except Exception as e:
        raise CodegenError(f"exec of generated code failed: {e}")
    compute = namespace.get("compute")
    if not callable(compute):
        raise CodegenError("generated code did not define compute(df, ctx)")

    return Feature(
        name=spec["name"],
        compute=compute,
        point_in_time=spec["point_in_time"],
        description=spec.get("text", ""),
        provenance={
            "sources": spec["sources"],
            "temporal_scope": spec["temporal_scope"],
            "tower": spec["tower"],
            "reads": "LLM-generated; audited by leakage.py code inspection",
            "llm_generated": True,
        },
        serving=spec["serving"],
        serving_notes=spec.get("serving_notes", ""),
    )


def smoke_test(feature: Feature, df_head, ctx) -> None:
    """Run compute on a few rows; must return an aligned Series."""
    out = feature.compute(df_head, ctx)
    if not isinstance(out, pd.Series):
        raise CodegenError(
            f"compute returned {type(out).__name__}, want pd.Series")
    if len(out) != len(df_head):
        raise CodegenError(
            f"compute returned {len(out)} rows for {len(df_head)} input rows")
