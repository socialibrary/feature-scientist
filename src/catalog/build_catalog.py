"""Build the searchable feature catalog.

Reads every source table, computes per-column coverage, and writes:
  src/catalog/registry.json  - full metadata (what the LLM reasons over)
  src/catalog/catalog.db     - SQLite FTS5 index over table/column text

Re-runnable: coverage is recomputed from whatever CSVs exist. Wikidata
columns report 0.0 coverage until the crawl lands.

Usage:  python3 -m catalog.build_catalog   (from src/)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))

OUT_DIR = os.path.join(BASE, "src", "catalog")
REGISTRY_JSON = os.path.join(OUT_DIR, "registry.json")
CATALOG_DB = os.path.join(OUT_DIR, "catalog.db")

# ---------------------------------------------------------------------------
# Static metadata. temporal_scope:
#   time_invariant    - true at any prediction time (title, runtime, gender)
#   point_in_time_safe- the event/anchor itself (rating, timestamp); aggregates
#                       over it must still be as-of
#   all_time          - aggregate over data newer than prediction time.
#                       FAIL-CLOSED: raw use is temporal leakage.
# serving classes mirror src/serving_cost.py.
# ---------------------------------------------------------------------------

TABLES: dict = {
    "ml1m_ratings": {
        "description": "1,000,209 user-movie ratings with timestamps (MovieLens-1M). "
                       "The event stream every behavioral feature is built from.",
        "join_keys": ["user_id", "movie_id"],
        "path": os.path.join(BASE, "data", "ml-1m", "ratings.dat"),
        "columns": {
            "user_id": {"description": "MovieLens user id (1-6040). Join key to ml1m_users.",
                        "dtype": "int", "temporal_scope": "time_invariant",
                        "serving": "row_local", "join_key": True},
            "movie_id": {"description": "MovieLens movie id (1-3952). Join key to ml1m_movies.",
                         "dtype": "int", "temporal_scope": "time_invariant",
                         "serving": "row_local", "join_key": True},
            "rating": {"description": "Whole-star rating 1-5. The observed event itself; "
                                      "aggregates over ratings must be as-of prediction time.",
                        "dtype": "int", "temporal_scope": "point_in_time_safe",
                        "serving": "history_scan", "join_key": False},
            "timestamp": {"description": "Seconds since epoch. THE prediction-time anchor: "
                                        "a feature is PIT-correct only if it reads rows with "
                                        "timestamp <= the scored row's timestamp.",
                          "dtype": "int", "temporal_scope": "point_in_time_safe",
                          "serving": "row_local", "join_key": False},
            "liked": {"description": "Derived label: rating >= 4. The prediction target.",
                      "dtype": "int", "temporal_scope": "point_in_time_safe",
                      "serving": "row_local", "join_key": False},
        },
    },
    "ml1m_users": {
        "description": "6,040 users: volunteered demographics (gender, age bucket, "
                       "occupation code, zip). Static per user.",
        "join_keys": ["user_id"],
        "path": os.path.join(BASE, "data", "ml-1m", "users.dat"),
        "columns": {
            "user_id": {"description": "Join key to ml1m_ratings.", "dtype": "int",
                        "temporal_scope": "time_invariant", "serving": "row_local",
                        "join_key": True},
            "gender": {"description": "M/F, volunteered.", "dtype": "str",
                       "temporal_scope": "time_invariant", "serving": "lookup",
                       "join_key": False},
            "age": {"description": "Bucket code: 1=Under 18, 18=18-24, 25=25-34, "
                                  "35=35-44, 45=45-49, 50=50-55, 56=56+.",
                     "dtype": "int", "temporal_scope": "time_invariant",
                     "serving": "lookup", "join_key": False},
            "occupation": {"description": "Coded 0-20 (student, engineer, educator, ...).",
                           "dtype": "int", "temporal_scope": "time_invariant",
                           "serving": "lookup", "join_key": False},
            "zip": {"description": "Zip code, volunteered, noisy.", "dtype": "str",
                    "temporal_scope": "time_invariant", "serving": "lookup",
                    "join_key": False},
        },
    },
    "ml1m_movies": {
        "description": "~3,900 movies: title (with release year), pipe-separated genres.",
        "join_keys": ["movie_id"],
        "path": os.path.join(BASE, "data", "ml-1m", "movies.dat"),
        "columns": {
            "movie_id": {"description": "Join key to ml1m_ratings and imdb_enrichment.movieId.",
                         "dtype": "int", "temporal_scope": "time_invariant",
                         "serving": "row_local", "join_key": True},
            "title": {"description": "IMDb title including release year, e.g. 'Toy Story (1995)'.",
                      "dtype": "str", "temporal_scope": "time_invariant",
                      "serving": "lookup", "join_key": False},
            "genres": {"description": "Pipe-separated genres (Action|Comedy|...). 18 possible values.",
                       "dtype": "str", "temporal_scope": "time_invariant",
                       "serving": "lookup", "join_key": False},
        },
    },
    "imdb_enrichment": {
        "description": "IMDb bulk-data join: 86% of movies, 94% of ratings. "
                       "Downloaded offline from datasets.imdbws.com; refresh = re-download.",
        "join_keys": ["movieId", "tconst"],
        "path": os.path.join(BASE, "data", "imdb", "movie_enrichment.csv"),
        "columns": {
            "movieId": {"description": "Join key = ml1m_movies.movie_id.", "dtype": "int",
                        "temporal_scope": "time_invariant", "serving": "row_local",
                        "join_key": True},
            "tconst": {"description": "IMDb title id (tt0114709). Join key to wikidata_movies.",
                       "dtype": "str", "temporal_scope": "time_invariant",
                       "serving": "row_local", "join_key": True},
            "primaryTitle": {"description": "Canonical IMDb title.", "dtype": "str",
                             "temporal_scope": "time_invariant", "serving": "lookup",
                             "join_key": False},
            "startYear": {"description": "Release year. Time-invariant, safe raw.",
                          "dtype": "float", "temporal_scope": "time_invariant",
                          "serving": "lookup", "join_key": False},
            "runtimeMinutes": {"description": "Runtime in minutes. Time-invariant, safe raw. "
                                              "Sparse for obscure titles.",
                               "dtype": "float", "temporal_scope": "time_invariant",
                               "serving": "lookup", "join_key": False},
            "directors": {"description": "IMDb name ids (nm...) pipe-separated, from title.crew. "
                                        "Time-invariant. Overlaps wikidata_movies.directors.",
                          "dtype": "str", "temporal_scope": "time_invariant",
                          "serving": "lookup", "join_key": False},
            "imdb_averageRating": {"description": "ALL-TIME aggregate (votes through crawl date). "
                                                 "LEAKAGE if used raw to score a 2001 rating.",
                                   "dtype": "float", "temporal_scope": "all_time",
                                   "serving": "lookup", "join_key": False},
            "imdb_numVotes": {"description": "ALL-TIME vote count. Same leakage warning as averageRating.",
                              "dtype": "float", "temporal_scope": "all_time",
                              "serving": "lookup", "join_key": False},
        },
    },
    "wikidata_movies": {
        "description": "Wikidata crawl keyed by IMDb tconst (P345): directors, cast, awards, "
                       "box office, budget, release date, language. Crawled offline by "
                       "src/crawl_wikidata.py; entity refs are QIDs (labels resolved for "
                       "directors/awards/languages).",
        "join_keys": ["tconst"],
        "path": os.path.join(BASE, "data", "wikidata", "movie_wikidata.csv"),
        "pending_note": "Populated by the offline crawl; coverage 0.0 until it lands.",
        "columns": {
            "tconst": {"description": "Join key = imdb_enrichment.tconst.", "dtype": "str",
                       "temporal_scope": "time_invariant", "serving": "row_local",
                       "join_key": True},
            "wikidata_qid": {"description": "Wikidata entity id (Q...).", "dtype": "str",
                             "temporal_scope": "time_invariant", "serving": "lookup",
                             "join_key": False},
            "directors": {"description": "Director QIDs, |-separated. Time-invariant. "
                                        "Overlaps imdb_enrichment.directors (different id space).",
                          "dtype": "str", "temporal_scope": "time_invariant",
                          "serving": "lookup", "join_key": False},
            "cast": {"description": "Cast-member QIDs, |-separated (can be dozens per film). "
                                   "Time-invariant. Useful for cast-track-record features.",
                      "dtype": "str", "temporal_scope": "time_invariant",
                      "serving": "lookup", "join_key": False},
            "awards": {"description": "Award QIDs (P166), |-separated, CURRENT as of crawl date. "
                                     "FAIL-CLOSED all_time: a film may have won awards AFTER "
                                     "prediction time, so raw use leaks the future. Needs "
                                     "award-date filtering for PIT-correct use.",
                       "dtype": "str", "temporal_scope": "all_time",
                       "serving": "lookup", "join_key": False},
            "box_office_amount": {"description": "Box-office gross (P2142), raw amount. ALL-TIME "
                                                "final gross; MIXED CURRENCIES across films. "
                                                "Leakage if raw; needs normalization + PIT care.",
                                    "dtype": "float", "temporal_scope": "all_time",
                                    "serving": "lookup", "join_key": False},
            "budget_amount": {"description": "Production budget (P2130), raw amount. Set pre-release "
                                            "so time-invariant in principle, but VERY sparse on "
                                            "Wikidata. Mixed currencies.",
                             "dtype": "float", "temporal_scope": "time_invariant",
                             "serving": "lookup", "join_key": False},
            "release_date": {"description": "Publication/release date (P577), YYYY-MM-DD.",
                            "dtype": "str", "temporal_scope": "time_invariant",
                            "serving": "lookup", "join_key": False},
            "original_language": {"description": "Original-language QID (P364). Time-invariant.",
                                  "dtype": "str", "temporal_scope": "time_invariant",
                                  "serving": "lookup", "join_key": False},
        },
    },
}


def _coverage_for(table: str) -> dict[str, float]:
    """Fraction of non-empty values per column, from whatever CSVs exist."""
    import pandas as pd

    meta = TABLES[table]
    path = meta["path"]
    if not os.path.exists(path):
        return {c: 0.0 for c in meta["columns"]}
    try:
        if path.endswith(".dat"):
            sep = "::"
            if table == "ml1m_ratings":
                names = ["user_id", "movie_id", "rating", "timestamp"]
            elif table == "ml1m_users":
                names = ["user_id", "gender", "age", "occupation", "zip"]
            else:
                names = ["movie_id", "title", "genres"]
            df = pd.read_csv(path, sep=sep, names=names, engine="python",
                             encoding="latin-1")
            if table == "ml1m_ratings":
                df["liked"] = (df["rating"] >= 4).astype(int)
        else:
            df = pd.read_csv(path, dtype=str)
            if table == "wikidata_movies":
                # coverage vs the FULL tconst universe, not crawled rows:
                # a partial crawl must honestly report partial coverage.
                enrich = os.path.join(
                    BASE, "data", "imdb", "movie_enrichment.csv")
                uni = pd.read_csv(enrich, dtype=str)[["tconst"]].dropna()
                df = uni.merge(df, on="tconst", how="left")
    except Exception:
        return {c: 0.0 for c in meta["columns"]}
    n = max(len(df), 1)
    cov = {}
    for c in meta["columns"]:
        if c not in df.columns:
            cov[c] = 0.0
        else:
            # notna() first: pandas 3 StringDtype renders NA as "<NA>",
            # and merged/empty cells must not count as coverage.
            s = df[c]
            mask = s.notna() & (s.astype(str) != "") & (~s.astype(str).str.fullmatch(r"(?i)nan|none|<na>|nat"))
            cov[c] = float(mask.mean())
    return cov


def build_registry() -> dict:
    registry = {}
    for table, meta in TABLES.items():
        cov = _coverage_for(table)
        cols = {}
        for cname, cm in meta["columns"].items():
            cols[cname] = {**cm, "coverage": round(cov.get(cname, 0.0), 4)}
        registry[table] = {
            "description": meta["description"],
            "join_keys": meta["join_keys"],
            "n_columns": len(cols),
            "columns": cols,
        }
        if "pending_note" in meta:
            nonkey = [v for k, v in cov.items()
                      if k not in meta["join_keys"]]
            if nonkey and max(nonkey) < 0.5:
                registry[table]["pending_note"] = meta["pending_note"]
    return registry


def build_fts(registry: dict) -> None:
    if os.path.exists(CATALOG_DB):
        os.remove(CATALOG_DB)
    con = sqlite3.connect(CATALOG_DB)
    con.execute("CREATE VIRTUAL TABLE catalog_fts USING fts5("
                "table_name, kind, col_name, text)")
    rows = []
    for table, meta in registry.items():
        rows.append((table, "table", "",
                     f"{table} {meta['description']} "
                     f"join keys: {', '.join(meta['join_keys'])}"))
        for cname, cm in meta["columns"].items():
            rows.append((table, "column", cname,
                         f"{table} {cname} {cm['description']} "
                         f"temporal scope {cm['temporal_scope']} "
                         f"serving {cm['serving']} dtype {cm['dtype']}"))
    con.executemany("INSERT INTO catalog_fts VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    registry = build_registry()
    with open(REGISTRY_JSON, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=1)
    build_fts(registry)
    print(f"wrote {REGISTRY_JSON} "
          f"({len(registry)} tables, "
          f"{sum(m['n_columns'] for m in registry.values())} columns)")
    print(f"wrote {CATALOG_DB}")


if __name__ == "__main__":
    main()
