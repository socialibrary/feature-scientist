"""The brief: the front door of Feature Scientist.

Usage:
    python3 src/brief.py briefs/movielens.yaml [--backend scripted]

A brief is a YAML file describing an ML problem in plain language plus the
datasets as free-form schema text (think: a Hive table dump pasted into a
doc). The agent:

  1. profiles the ACTUAL tables behind the schema text ("text describes,
     data verifies"): row counts, null rates, time ranges, join-key
     overlap/coverage -- and writes the profile into the data-catalog
     registry format (src/catalog/profiling.json).
  2. trains the baseline and analyzes error slices.
  3. runs the calibration screen battery (no training): cheap candidate
     signals whose miscalibration gaps guide proposing.
  4. proposes -> critic -> leakage audit -> cost gate -> calibration screen
     -> ablate, stopping when no candidate shows a significant calibration
     gap, max_rounds is hit, or the serving budget is exhausted.
  5. appends every verdict to the experiment ledger.

The brief's `prediction_time` is the load-bearing temporal contract: it is
quoted into every LLM prompt, and the deterministic leakage audit enforces
each generated feature's PIT declaration against the actual code.

Table loading is behind a small TableSource abstraction; the MovieLens /
IMDb / Wikidata loaders below are one backend. A future warehouse backend
(Hive/Spark/Trino) implements the same interface: name, schema_text,
registry_key, load() -> pd.DataFrame.

YAML subset supported (stdlib only, no PyYAML): nested maps via
indentation, lists of maps, inline {k: v} flow maps, quoted strings,
# comments, and bool/int/float/null scalars. JSON briefs are accepted too.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILING_JSON = os.path.join(BASE, "src", "catalog", "profiling.json")


# ---------------------------------------------------------------------------
# Minimal YAML-subset parser (stdlib only)
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    in_s = in_d = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return line[:i]
    return line


def _split_top(s: str, delim: str) -> list[str]:
    parts, depth, in_s, in_d, cur = [], 0, False, False, ""
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif not in_s and not in_d:
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
            elif ch == delim and depth == 0:
                parts.append(cur)
                cur = ""
                continue
        cur += ch
    parts.append(cur)
    return parts


def _parse_scalar(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    if s.startswith("{") and s.endswith("}"):
        out = {}
        for part in _split_top(s[1:-1], ","):
            if not part.strip():
                continue
            k, sep, v = part.partition(":")
            if not sep:
                raise ValueError(f"bad flow-map entry: {part!r}")
            out[k.strip()] = _parse_scalar(v)
        return out
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_block(lines: list[tuple[int, str]], i: int, ind: int):
    if i < len(lines):
        c = lines[i][1]
        if c == "-" or c.startswith("- "):
            return _parse_list(lines, i, ind)
    return _parse_map(lines, i, ind)


def _parse_map(lines: list[tuple[int, str]], i: int, ind: int):
    d: dict = {}
    while i < len(lines):
        cur_ind, content = lines[i]
        if cur_ind != ind:
            break
        key, sep, val = content.partition(":")
        if not sep:
            raise ValueError(f"bad YAML line: {content!r}")
        key, val = key.strip(), val.strip()
        if val == "":
            if i + 1 < len(lines) and lines[i + 1][0] > ind:
                sub, i = _parse_block(lines, i + 1, lines[i + 1][0])
                d[key] = sub
            else:
                d[key] = None
                i += 1
        else:
            d[key] = _parse_scalar(val)
            i += 1
    return d, i


def _parse_list(lines: list[tuple[int, str]], i: int, ind: int):
    lst: list = []
    while i < len(lines):
        cur_ind, content = lines[i]
        if cur_ind != ind or not (content == "-" or content.startswith("- ")):
            break
        item = content[1:].strip()
        if item == "":
            sub, i = _parse_block(lines, i + 1, lines[i + 1][0])
            lst.append(sub)
        elif ":" in item:
            k, _, v = item.partition(":")
            d = {k.strip(): _parse_scalar(v.strip())}
            if i + 1 < len(lines) and lines[i + 1][0] > ind:
                sub, i = _parse_map(lines, i + 1, lines[i + 1][0])
                if not isinstance(sub, dict):
                    raise ValueError("mixed list item")
                d.update(sub)
            else:
                i += 1
            lst.append(d)
        else:
            lst.append(_parse_scalar(item))
            i += 1
    return lst, i


def _parse_yaml_subset(text: str) -> dict:
    raw = [_strip_comment(l).rstrip() for l in text.splitlines()]
    lines = [(len(l) - len(l.lstrip(" ")), l.strip())
             for l in raw if l.strip()]
    if not lines:
        return {}
    obj, nxt = _parse_block(lines, 0, lines[0][0])
    if nxt != len(lines):
        raise ValueError(f"trailing content at line: {lines[nxt]!r}")
    if not isinstance(obj, dict):
        raise ValueError("brief must be a top-level mapping")
    return obj


def _validate_brief(b: dict, path: str) -> None:
    if "problem" not in b:
        raise ValueError(f"{path}: brief needs a 'problem' field")
    tables = b.get("tables") or []
    if not isinstance(tables, list):
        raise ValueError(f"{path}: 'tables' must be a list")
    for t in tables:
        if not isinstance(t, dict) or "name" not in t:
            raise ValueError(f"{path}: each table needs a 'name': {t!r}")


def parse_brief(path: str) -> dict:
    """Parse a brief YAML (subset) or JSON file; validate the shape."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    brief = json.loads(text) if text.lstrip().startswith("{") \
        else _parse_yaml_subset(text)
    _validate_brief(brief, path)
    return brief


# ---------------------------------------------------------------------------
# Table abstraction. One backend today (local MovieLens/IMDb/Wikidata files);
# a warehouse backend implements the same four attributes.
# ---------------------------------------------------------------------------

class TableSource:
    """A logical table: schema text (what the user wrote) + a loader."""

    def __init__(self, name: str, schema_text: str, registry_key: str,
                 load):
        self.name = name
        self.schema_text = schema_text
        self.registry_key = registry_key
        self._load = load

    def load(self) -> pd.DataFrame | None:
        """The data; None when the source isn't available (e.g. crawl
        still running) -- the profiler reports it as pending, honestly."""
        return self._load()


def _load_imdb() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, "data", "imdb",
                                    "movie_enrichment.csv"))


def _load_wikidata() -> pd.DataFrame | None:
    p = os.path.join(BASE, "data", "wikidata", "movie_wikidata.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _ml1m_loaders():
    from data import load_movies, load_ratings, load_users
    return load_ratings, load_users, load_movies


# brief table name -> (catalog registry key, loader)
TABLE_DEFS: dict[str, tuple[str, object]] = {}


def _table_defs() -> dict[str, tuple[str, object]]:
    global TABLE_DEFS
    if not TABLE_DEFS:
        load_ratings, load_users, load_movies = _ml1m_loaders()
        TABLE_DEFS = {
            "ratings": ("ml1m_ratings", load_ratings),
            "users": ("ml1m_users", load_users),
            "movies": ("ml1m_movies", load_movies),
            "imdb_enrichment": ("imdb_enrichment", _load_imdb),
            "wikidata_movies": ("wikidata_movies", _load_wikidata),
        }
    return TABLE_DEFS


def table_source(name: str, schema_text: str = "") -> TableSource | None:
    defs = _table_defs()
    if name not in defs:
        return None
    reg_key, loader = defs[name]
    return TableSource(name, schema_text, reg_key, loader)


# v1 shim for the known id-space rename between the MovieLens .dat files
# (movie_id) and the IMDb bulk data (movieId). A warehouse backend declares
# join keys explicitly in the brief instead of relying on this.
COLUMN_EQUIV: dict[tuple[str, str], tuple[str, str]] = {
    ("imdb_enrichment", "movieId"): ("movies", "movie_id"),
    ("imdb_enrichment", "movieId"): ("ratings", "movie_id"),
}


# ---------------------------------------------------------------------------
# Profiler: text describes, data verifies.
# ---------------------------------------------------------------------------

def _jsonable(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def profile_frame(name: str, df: pd.DataFrame,
                  schema_text: str, registry_key: str) -> dict:
    cols = {}
    for c in df.columns:
        s = df[c]
        info: dict = {
            "dtype": str(s.dtype),
            "null_rate": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s.dtype):
            info["min"] = _jsonable(s.min())
            info["max"] = _jsonable(s.max())
        cols[str(c)] = info
    time_range = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("timestamp", "ts") \
                and pd.api.types.is_numeric_dtype(df[c].dtype):
            lo, hi = df[c].min(), df[c].max()
            time_range[str(c)] = [
                pd.to_datetime(lo, unit="s", utc=True).isoformat(),
                pd.to_datetime(hi, unit="s", utc=True).isoformat(),
            ]
    return {"table": name, "registry_key": registry_key,
            "schema_text": schema_text,
            "n_rows": int(len(df)), "n_columns": len(cols),
            "columns": cols, "time_range": time_range}


def _key_pairs(frames: dict[str, pd.DataFrame]) -> list[tuple]:
    """Candidate join-key pairs: shared column names + the v1 equiv shim."""
    pairs = []
    names = list(frames)
    seen = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = set(frames[a].columns) & set(frames[b].columns)
            for c in sorted(shared):
                key = tuple(sorted([(a, str(c)), (b, str(c))]))
                if key not in seen:
                    seen.add(key)
                    pairs.append(((a, str(c)), (b, str(c))))
    for (ta, ca), (tb, cb) in COLUMN_EQUIV.items():
        if ta in frames and tb in frames \
                and ca in frames[ta].columns and cb in frames[tb].columns:
            pairs.append(((ta, ca), (tb, cb)))
    return pairs


def join_coverage(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """For each candidate key pair: key overlap + row-weighted coverage."""
    out = []
    for (ta, ca), (tb, cb) in _key_pairs(frames):
        fa, fb = frames[ta], frames[tb]
        sa = set(fa[ca].dropna().unique())
        sb = set(fb[cb].dropna().unique())
        inter = len(sa & sb)
        # row-weighted: what fraction of each table's ROWS join?
        ra = float(fa[ca].isin(sb).mean()) if len(fa) else 0.0
        rb = float(fb[cb].isin(sa).mean()) if len(fb) else 0.0
        out.append({
            "left": f"{ta}.{ca}", "right": f"{tb}.{cb}",
            "n_left_keys": len(sa), "n_right_keys": len(sb),
            "n_overlap": inter,
            "left_keys_in_right": round(inter / max(len(sa), 1), 4),
            "right_keys_in_left": round(inter / max(len(sb), 1), 4),
            "left_rows_join": round(ra, 4),
            "right_rows_join": round(rb, 4),
        })
    return out


def profile_brief_tables(brief: dict) -> dict[str, dict]:
    """Load + profile every known table in the brief; print the report.

    Returns {table_name: profile}; also writes src/catalog/profiling.json.
    Unknown tables get a warning (a warehouse backend would handle them).
    """
    profiles: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    pending: list[str] = []
    for t in brief.get("tables") or []:
        name = t["name"]
        src = table_source(name, t.get("schema_text", ""))
        if src is None:
            print(f"  ! table '{name}': no loader in this backend "
                  f"(warehouse tables plug in here)")
            pending.append(name)
            continue
        try:
            df = src.load()
        except FileNotFoundError as e:
            print(f"  ! table '{name}': source file missing ({e.filename})")
            pending.append(name)
            continue
        if df is None:
            print(f"  ! table '{name}': source not available yet (pending)")
            pending.append(name)
            continue
        prof = profile_frame(name, df, src.schema_text, src.registry_key)
        profiles[name] = prof
        frames[name] = df
        print(f"  table '{name}': {prof['n_rows']:,} rows x "
              f"{prof['n_columns']} cols")
        for c, ci in prof["columns"].items():
            print(f"    {c}: {ci['dtype']}, null {ci['null_rate']:.1%}, "
                  f"unique {ci['n_unique']:,}")
        for c, (lo, hi) in prof["time_range"].items():
            print(f"    time range [{c}]: {lo} -> {hi}")

    coverage = join_coverage(frames)
    if coverage:
        print("  join coverage:")
        for jc in coverage:
            print(f"    {jc['left']} <-> {jc['right']}: "
                  f"{jc['n_overlap']:,}/{jc['n_left_keys']:,} left keys "
                  f"({jc['left_keys_in_right']:.1%}), "
                  f"{jc['left_rows_join']:.1%} of left rows join")
    for p in profiles.values():
        p["join_coverage"] = [jc for jc in coverage
                              if jc["left"].startswith(p["table"] + ".")
                              or jc["right"].startswith(p["table"] + ".")]
    if pending:
        print(f"  pending tables (not profiled): {', '.join(pending)}")

    doc = {"generated_at": datetime.datetime.now(datetime.timezone.utc)
           .isoformat(),
           "tables": profiles, "pending": pending}
    os.makedirs(os.path.dirname(PROFILING_JSON), exist_ok=True)
    with open(PROFILING_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"  profile written to {PROFILING_JSON}")
    return profiles


def load_profiled_catalog() -> dict:
    """DATA_CATALOG shape extended with profiling fields (not a fork).

    Starts from the registry-backed load_catalog_dict() the agentic loop
    uses, then merges profiling.json: per-table n_rows/join_coverage and
    per-column null_rate/n_unique/min/max. Rebuild the profile with:
    python3 src/brief.py briefs/movielens.yaml (profiling runs first).
    """
    from catalog import load_catalog_dict
    cat = load_catalog_dict()
    if not os.path.exists(PROFILING_JSON):
        return cat
    with open(PROFILING_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    defs = _table_defs()
    for name, prof in doc.get("tables", {}).items():
        reg_key = defs.get(name, (None,))[0]
        if reg_key not in cat:
            continue
        cat[reg_key]["profile"] = {
            "n_rows": prof["n_rows"],
            "time_range": prof.get("time_range", {}),
            "join_coverage": prof.get("join_coverage", []),
        }
        for col, ci in prof["columns"].items():
            cm = cat[reg_key]["column_meta"].get(col)
            if cm is not None:
                for k in ("null_rate", "n_unique", "min", "max"):
                    if k in ci and ci[k] is not None:
                        cm[k] = ci[k]
    return cat


# ---------------------------------------------------------------------------
# Calibration screen battery: cheap pre-loop screen over pre-computable
# candidate signals. Its significant gaps guide proposing (they become part
# of the LLM prompt). No training -- inference only.
# ---------------------------------------------------------------------------

def build_screen_battery(ctx: dict, train: pd.DataFrame, valid: pd.DataFrame,
                         stats: dict, proba: np.ndarray) -> list[str]:
    """Return plain-English summaries of significant miscalibrations."""
    from baseline import TARGET
    from calibration import screen_candidates, screen_sparse_id
    from error_analysis import GENRES

    y = valid[TARGET].to_numpy()
    idx = valid.index
    cands: dict[str, pd.Series] = {}

    # movie age at prediction time (release timestamps are time-invariant)
    rel = ctx.get("movie_release_ts") or {}
    if rel:
        age = (valid["timestamp"].to_numpy()
               - valid["movie_id"].map(rel).to_numpy()) / 86400.0
        cands["battery/movie_age_days"] = pd.Series(age, index=idx)
        cands["battery/release_year"] = pd.Series(
            valid["movie_id"].map(
                {m: int(pd.to_datetime(t, unit="s", utc=True).year)
                 for m, t in rel.items()}),
            index=idx)
    # genre flags
    if "genres" in valid.columns:
        g = valid["genres"].fillna("")
        for genre in GENRES:
            cands[f"battery/genre_{genre}"] = \
                g.str.contains(genre, regex=False).astype("int8")
    # IMDb time-invariant columns
    imdb = ctx.get("imdb")
    if imdb is not None:
        im = imdb.set_index("movieId")
        for col in ("runtimeMinutes", "startYear"):
            if col in im.columns:
                cands[f"battery/imdb_{col}"] = pd.Series(
                    valid["movie_id"].map(im[col]), index=idx)
    # train-period activity: dense proxies for the sparse IDs
    cands["battery/user_train_count"] = pd.Series(
        valid["user_id"].map(stats["user_count"]).fillna(0), index=idx)
    cands["battery/movie_train_count"] = pd.Series(
        valid["movie_id"].map(stats["movie_count"]).fillna(0), index=idx)

    results = screen_candidates(y, proba, cands)
    summaries = [r["summary"] for r in results
                 if r["verdict"] == "signal"]

    # sparse-ID screens: popularity + recency proxies (never raw IDs)
    for id_col, id_name in (("movie_id", "movie"), ("user_id", "user")):
        try:
            sp = screen_sparse_id(
                y, proba, valid[id_col],
                train.groupby(id_col).size(),
                f"{id_name}_id",
                train_id_last_ts=train.groupby(id_col)["timestamp"].max(),
                valid_ts=valid["timestamp"])
            for part in ("popularity", "recency"):
                s = sp.get(part) or {}
                if s.get("verdict") == "signal":
                    summaries.append(
                        f"battery/{id_name}_{part}: {s.get('summary')}")
        except Exception as e:  # noqa: BLE001 -- battery must not kill the run
            print(f"  battery sparse screen ({id_name}) skipped: {e}")
    return summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Feature Scientist: run an ML problem from a brief YAML")
    ap.add_argument("brief", help="path to the brief YAML (or JSON)")
    ap.add_argument("--backend", choices=["http", "scripted"], default=None,
                    help="LLM backend (default: brief's, else scripted)")
    ap.add_argument("--engine", choices=["rule", "llm"], default=None,
                    help="hypothesis engine (default: brief's, else llm)")
    ap.add_argument("--model", choices=["torch", "lgbm"], default=None,
                    help="model backend (default: brief's, else torch)")
    ap.add_argument("--max-rounds", type=int, default=None)
    args = ap.parse_args()

    brief = parse_brief(args.brief)
    engine = args.engine or brief.get("engine", "llm")
    llm_backend = args.backend or brief.get("backend", "scripted")
    model = args.model or brief.get("model", "torch")
    stop_cfg = brief.get("stop_when", {}) or {}
    max_rounds = args.max_rounds or stop_cfg.get("max_rounds", 3)
    stop_on_no_signal = bool(stop_cfg.get("stop_on_no_signal", True))

    print("=" * 60)
    print("BRIEF")
    print("=" * 60)
    print(f"problem: {brief['problem']}")
    if brief.get("label"):
        print(f"label:   {brief['label']}")
    if brief.get("prediction_time"):
        print(f"TEMPORAL CONTRACT: {brief['prediction_time']}")

    # 1. text describes, data verifies: profile the actual tables ---------
    print("\nprofiling tables...")
    profile_brief_tables(brief)

    # 2. catalog = registry + profiling fields (extends, doesn't fork) -----
    catalog = load_profiled_catalog()

    # 3. serving budgets from the brief (module constants, restored after)
    import serving_cost
    serving = brief.get("serving", {}) or {}
    old_feat, old_total = serving_cost.MAX_FEATURE_UNITS, \
        serving_cost.MAX_TOTAL_UNITS
    serving_cost.MAX_FEATURE_UNITS = float(
        serving.get("per_feature_budget", old_feat))
    serving_cost.MAX_TOTAL_UNITS = float(
        serving.get("total_budget", old_total))
    print(f"\nserving budget: {serving_cost.MAX_FEATURE_UNITS}/feature, "
          f"{serving_cost.MAX_TOTAL_UNITS} total")

    # 4. the temporal contract is load-bearing: it travels with the run ---
    problem = {"problem": brief["problem"],
               "label": brief.get("label", ""),
               "prediction_time": brief.get("prediction_time", "")}

    from run_scientist import print_summary, run
    try:
        entries = run(
            rounds=max_rounds, engine_name=engine, model_backend=model,
            llm_backend=llm_backend,
            calibration_screen=True,
            screen_battery_fn=build_screen_battery,
            stop_on_no_signal=stop_on_no_signal,
            problem=problem, skip_accepted=True, catalog=catalog)
    finally:
        serving_cost.MAX_FEATURE_UNITS = old_feat
        serving_cost.MAX_TOTAL_UNITS = old_total
    print_summary(entries)
    from ledger import load_entries
    print(f"\nledger: results/experiments.jsonl "
          f"({len(load_entries())} entries total)")


if __name__ == "__main__":
    main()
