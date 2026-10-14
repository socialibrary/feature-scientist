"""Full-text discovery API over the feature catalog.

The agent calls search_datasets() to discover data -- e.g.
search_datasets("director awards") -> wikidata_movies.awards with its
temporal_scope (all_time: fail-closed) and serving class. This is the
"what information exists?" counterpart to the hypothesis engine's
"what information is missing?".
"""

from __future__ import annotations

import json
import os
import re
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_JSON = os.path.join(_HERE, "registry.json")
CATALOG_DB = os.path.join(_HERE, "catalog.db")


def load_registry() -> dict:
    """Full registry: tables -> {description, join_keys, columns{...}}."""
    with open(REGISTRY_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_catalog_dict() -> dict:
    """Backwards-compatible DATA_CATALOG shape for src/scientist.py.

    Same keys as the historic literal (description / columns / temporal /
    note) plus column_meta with the per-column metadata the LLM reasons
    over. Engines import search_datasets from the catalog package (also
    re-exported at scientist module level) for discovery.
    """
    reg = load_registry()
    out = {}
    for table, meta in reg.items():
        scopes = {c["temporal_scope"] for c in meta["columns"].values()}
        temporal = ("mixed" if len(scopes) > 1
                    else ("temporal" if "point_in_time_safe" in scopes
                          else "static"))
        notes = []
        leaky = [c for c, cm in meta["columns"].items()
                 if cm["temporal_scope"] == "all_time"]
        if leaky:
            notes.append("ALL-TIME columns (temporal leakage if used raw): "
                         + ", ".join(leaky) + ".")
        if meta.get("pending_note"):
            notes.append(meta["pending_note"])
        out[table] = {
            "description": meta["description"],
            "columns": list(meta["columns"].keys()),
            "temporal": temporal,
            "note": " ".join(notes) or "See column_meta for per-column detail.",
            "join_keys": meta["join_keys"],
            "column_meta": meta["columns"],
        }
    return out


def search_datasets(query: str, k: int = 5) -> list[dict]:
    """Ranked full-text search over table + column descriptions.

    Returns up to k hits: each has table, column (None for table-level
    hits), kind, description, temporal_scope, serving, coverage, join_key,
    and the bm25 rank. Raises RuntimeError if the index hasn't been built
    (run: python3 -m catalog.build_catalog from src/).
    """
    if not os.path.exists(CATALOG_DB):
        raise RuntimeError(
            "catalog index missing; run: python3 -m catalog.build_catalog")
    # Forgiving matching: OR of per-term prefixes, so "box office revenue"
    # finds "box-office gross" and "director awards" finds both the
    # directors column and the awards column.
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not tokens:
        return []
    match_q = " OR ".join(t + "*" for t in tokens)
    con = sqlite3.connect(CATALOG_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT table_name, kind, col_name, text, bm25(catalog_fts) AS r "
        "FROM catalog_fts WHERE catalog_fts MATCH ? "
        "ORDER BY r LIMIT ?", (match_q, k)).fetchall()
    con.close()

    reg = load_registry()
    hits = []
    for r in rows:
        table, kind, col = r["table_name"], r["kind"], r["col_name"]
        tmeta = reg[table]
        if kind == "column":
            cm = tmeta["columns"][col]
            hits.append({
                "table": table, "column": col, "kind": kind,
                "description": cm["description"],
                "dtype": cm["dtype"],
                "temporal_scope": cm["temporal_scope"],
                "serving": cm["serving"],
                "coverage": cm["coverage"],
                "join_key": cm["join_key"],
                "table_description": tmeta["description"],
                "rank": r["r"],
            })
        else:
            hits.append({
                "table": table, "column": None, "kind": kind,
                "description": tmeta["description"],
                "join_keys": tmeta["join_keys"],
                "n_columns": tmeta["n_columns"],
                "rank": r["r"],
            })
    return hits
