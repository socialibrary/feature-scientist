"""The science loop: hypothesis engines + executable feature builders.

The loop's job is to answer "what information is the model missing?" --
not "what hyperparameters should I try?". Each Hypothesis becomes an
executable Feature auto-registered in the FeatureRegistry with a
point_in_time declaration and a provenance record the leakage audit reads.

v1 data catalog (the agent reasons over BOTH sources from the start):
  * ml-1m   - ratings/users/movies, fully timestamped
  * imdb    - title enrichment; imdb_averageRating/numVotes are ALL-TIME
              aggregates (through Oct 2026) -> raw use is temporal leakage.
              startYear/runtimeMinutes/directors are time-invariant -> safe.

Engines:
  * RuleBasedEngine - default, no API keys. Template hypotheses triggered by
    the worst error slices (momentum, affinity, cold-start, recency...).
    Deliberately includes two tempting-but-leaky hypotheses so the demo can
    show the leakage audit catching them on camera.
  * LLMEngine - STUB. This is the seam for the Meta Model API on hackathon
    day: set META_MODEL_API_URL / META_MODEL_API_KEY and implement propose()
    using _build_prompt(). Raises NotImplementedError until then.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features import Feature, FeatureRegistry
from traps import build_sneaky_movie_mean, build_user_history_entropy

# ---------------------------------------------------------------------------
# Data catalog: what the agent is allowed to reason about.
# ---------------------------------------------------------------------------

DATA_CATALOG = {
    "ml1m_ratings": {
        "description": "1,000,209 user-movie ratings with timestamps",
        "columns": ["user_id", "movie_id", "rating", "timestamp", "liked"],
        "temporal": True,
        "note": "Timestamped. Aggregates must be train-period-only or as-of "
                "each row's prediction time, never all-time.",
    },
    "ml1m_users": {
        "description": "6,040 users: gender, age bucket, occupation, zip",
        "columns": ["user_id", "gender", "age", "occupation", "zip"],
        "temporal": False,
        "note": "Static demographics. Safe to use raw.",
    },
    "ml1m_movies": {
        "description": "~3,900 movies: title (with year), pipe-separated genres",
        "columns": ["movie_id", "title", "genres"],
        "temporal": False,
        "note": "Static metadata. Safe to use raw.",
    },
    "imdb_enrichment": {
        "description": "IMDb join on 86% of movies (94% of ratings)",
        "columns": ["movieId", "tconst", "primaryTitle", "startYear",
                    "directors", "runtimeMinutes",
                    "imdb_averageRating", "imdb_numVotes"],
        "temporal": "mixed",
        "note": "imdb_averageRating / imdb_numVotes are ALL-TIME aggregates "
                "(votes through Oct 2026). Using them raw to score a 2001 "
                "rating is TEMPORAL LEAKAGE. startYear / runtimeMinutes / "
                "directors are time-invariant and safe.",
    },
}

IMDB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "imdb",
    "movie_enrichment.csv",
)

DAY = 86400


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """One falsifiable idea about what information the model is missing."""
    text: str                    # plain-English statement of the idea
    feature_name: str            # registry name the builder will use
    sources: list[str]           # catalog keys, e.g. ["ml1m_ratings", "imdb_enrichment"]
    rationale: str               # why this should help, given the error slices
    build: object = None         # (ctx) -> Feature; set by the engine
    # trap=True marks the deliberate leakage traps (demo only). A real LLM
    # engine would propose risky ideas on its own; here we plant two so the
    # audit has something to catch on camera.
    trap: bool = False


class HypothesisEngine:
    """Interface: error slices + catalog in, hypotheses out."""

    def propose(self, error_slices: dict, catalog: dict,
                round_no: int) -> list[Hypothesis]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Feature builders. Each returns a Feature with provenance; compute(df, ctx)
# is vectorized and point-in-time correct (except the deliberate traps).
# ---------------------------------------------------------------------------

def _events_by_key(df: pd.DataFrame, key: str):
    """{key_value: (sorted ts array, ratings array)} from a timestamped frame."""
    out = {}
    for k, g in df.groupby(key, sort=False):
        o = np.argsort(g["timestamp"].to_numpy(), kind="mergesort")
        out[k] = (g["timestamp"].to_numpy()[o], g["rating"].to_numpy()[o])
    return out


def _asof_mean(key_ids: np.ndarray, row_ts: np.ndarray, events: dict,
               window_s: int, fallback: dict | float) -> np.ndarray:
    """Mean of values in [row_ts - window, row_ts) per key. Vectorized per key."""
    n = len(key_ids)
    if isinstance(fallback, dict):
        fb_global = float(np.mean(list(fallback.values()))) \
            if fallback else 0.0
    else:
        fb_global = float(fallback)
    out = np.full(n, fb_global, dtype=float)
    for k in np.unique(key_ids):
        ev = events.get(k)
        m = key_ids == k
        rts = row_ts[m]
        if ev is None:
            out[m] = fallback.get(k, fb_global) if isinstance(fallback, dict) \
                else fb_global
            continue
        ts_arr, r_arr = ev
        lo = np.searchsorted(ts_arr, rts - window_s, side="left")
        hi = np.searchsorted(ts_arr, rts, side="left")  # strictly < rts
        cs = np.concatenate([[0.0], np.cumsum(r_arr, dtype=float)])
        cnt = hi - lo
        s = cs[hi] - cs[lo]
        fb = fallback.get(k, fb_global) if isinstance(fallback, dict) else fb_global
        out[m] = np.where(cnt > 0, s / np.maximum(cnt, 1), fb)
    return out


def build_movie_velocity_30d(ctx) -> Feature:
    """A movie's mean rating over the 30 days before each prediction (as-of)."""
    gm = ctx["global_mean"]
    movie_mean = ctx["stats"]["movie_mean"].to_dict()

    def compute(df, c):
        return pd.Series(
            _asof_mean(df["movie_id"].to_numpy(), df["timestamp"].to_numpy(),
                       c["movie_events"], 30 * DAY, movie_mean),
            index=df.index,
        )

    return Feature(
        name="movie_velocity_30d",
        compute=compute,
        point_in_time="timestamp",
        description="Mean rating of the movie over the 30 days strictly "
                    "before each row's timestamp (as-of). Captures momentum / "
                    "cooling for new releases.",
        provenance={"sources": ["ml1m_ratings"], "temporal_scope": "pit_correct", "tower": "movie",
                    "reads": "train-period ratings with ts in [row_ts-30d, row_ts)",
                    "ctx_keys": ["movie_events"]},
        serving="history_scan",
        serving_notes="30d as-of window needs the movie event stream at request time",
    )


def build_user_recency_30d(ctx) -> Feature:
    """The user's mean rating over the 30 days before each prediction."""
    gm = ctx["global_mean"]
    user_mean = ctx["stats"]["user_mean"].to_dict()

    def compute(df, c):
        return pd.Series(
            _asof_mean(df["user_id"].to_numpy(), df["timestamp"].to_numpy(),
                       c["user_events"], 30 * DAY, user_mean),
            index=df.index,
        )

    return Feature(
        name="user_recency_30d",
        compute=compute,
        point_in_time="timestamp",
        description="User's mean rating over the prior 30 days (as-of). "
                    "Captures taste drift and rating-scale drift.",
        provenance={"sources": ["ml1m_ratings"], "temporal_scope": "pit_correct", "tower": "user",
                    "reads": "train-period ratings with ts in [row_ts-30d, row_ts)",
                    "ctx_keys": ["user_events"]},
        serving="history_scan",
        serving_notes="30d as-of window needs the user event stream at request time",
    )


def build_user_genre_affinity(ctx) -> Feature:
    """User's train-period like-rate in the target movie's genres (averaged)."""
    table: pd.DataFrame = ctx["user_genre_rate"]  # MultiIndex (user_id, genre)
    global_like = ctx["global_like"]

    def compute(df, c):
        genres = df["genres"].str.split("|")
        # explode to (row, genre), map the like-rate, average back per row
        rep = np.repeat(np.arange(len(df)), genres.str.len().to_numpy())
        g = pd.Series([gg for sub in genres for gg in sub])
        keys = list(zip(df["user_id"].to_numpy()[rep], g.to_numpy()))
        vals = table["like_rate"].reindex(
            pd.MultiIndex.from_tuples(keys, names=["user_id", "genre"])
        ).to_numpy()
        vals = np.where(np.isnan(vals), global_like, vals)
        out = np.bincount(rep, weights=vals, minlength=len(df)) / \
            np.maximum(np.bincount(rep, minlength=len(df)), 1)
        return pd.Series(out, index=df.index)

    return Feature(
        name="user_genre_affinity",
        compute=compute,
        point_in_time="timestamp",
        description="Mean of the user's train-period like-rate across the "
                    "target movie's genres. Content signal from behavior.",
        provenance={"sources": ["ml1m_ratings", "ml1m_movies"],
                    "temporal_scope": "train_only", "tower": "wide",
                    "reads": "user x genre like-rate table built from train rows only",
                    "ctx_keys": ["user_genre_rate"]},
        serving="lookup",
        serving_notes="precomputed user x genre table, O(1) by (user_id, genre)",
    )


def build_director_track_record_pit(ctx) -> Feature:
    """Director's track record using only movies released before prediction time.

    Point-in-time correct: for a row at time t, eligible movies are the
    director's movies with release_ts < t, scored by their TRAIN-period
    ratings (all train timestamps precede every validation timestamp, and
    rating ts < t is enforced anyway). Prefix sums over release-sorted movies
    make it vectorized per director.
    """
    prefix: dict = ctx["director_prefix"]  # nconst -> (rel_ts, cum_count, cum_sum)
    movie_to_director: dict = ctx["movie_to_director"]
    gm = ctx["global_mean"]

    def compute(df, c):
        n = len(df)
        out = np.full(n, gm, dtype=float)
        mids = df["movie_id"].to_numpy()
        rts = df["timestamp"].to_numpy()
        darr = np.array([movie_to_director.get(m) for m in mids], dtype=object)
        for d in set(darr) - {None}:
            ev = prefix.get(d)
            if ev is None:
                continue  # director has no train-period movies with release dates
            rel, pc, ps = ev
            pm = darr == d
            k = np.searchsorted(rel, rts[pm], side="left")
            cnt = pc[k]
            s = ps[k]
            out[pm] = np.where(cnt > 0, s / np.maximum(cnt, 1), gm)
        return pd.Series(out, index=df.index)

    return Feature(
        name="director_track_record_pit",
        compute=compute,
        point_in_time="timestamp",
        description="Mean train-period rating of the director's movies "
                    "released strictly before each row's timestamp. Cold-start "
                    "signal from IMDb, point-in-time correct.",
        provenance={"sources": ["ml1m_ratings", "imdb_enrichment"],
                    "temporal_scope": "pit_correct", "tower": "movie",
                    "reads": "train ratings of director's movies with release_ts < row_ts",
                    "ctx_keys": ["director_prefix", "movie_to_director"]},
        serving="lookup",
        serving_notes="precomputable per (director, month); O(1) at request time",
    )


def build_movie_age_days(ctx) -> Feature:
    """Days since the movie's release. Time-invariant metadata -> safe."""
    release_ts: dict = ctx["movie_release_ts"]

    def compute(df, c):
        rel = df["movie_id"].map(release_ts)
        # rows without a release date get the median age (neutral)
        med = c["median_movie_age_days"]
        out = (df["timestamp"] - rel) / DAY
        return pd.Series(out.fillna(med).clip(lower=0), index=df.index)

    return Feature(
        name="movie_age_days",
        compute=compute,
        point_in_time="timestamp",
        description="Days from movie release to prediction time. Captures "
                    "novelty decay / catalog staleness.",
        provenance={"sources": ["imdb_enrichment", "ml1m_movies"],
                    "temporal_scope": "static", "tower": "movie",
                    "reads": "release year only (time-invariant)",
                    "ctx_keys": ["movie_release_ts"]},
        serving="row_local",
        serving_notes="timestamp minus release_ts; pure row function",
    )


# --- deliberate leakage traps (demo) ---------------------------------------

def build_imdb_average_rating_raw(ctx) -> Feature:
    """TRAP: IMDb's all-time average rating used raw. Temporal leakage."""
    s = ctx["imdb"]["imdb_averageRating"]
    gm = ctx["global_mean"]

    def compute(df, c):
        return pd.Series(
            df["movie_id"].map(ctx["imdb_map_avg"]).fillna(gm), index=df.index)

    _ = s  # documented read of the all-time aggregate
    return Feature(
        name="imdb_average_rating_raw",
        compute=compute,
        point_in_time="timestamp",
        description="TRAP (demo): IMDb all-time average rating mapped raw. "
                    "Includes votes from years AFTER the prediction -- leakage.",
        provenance={"sources": ["imdb_enrichment"], "temporal_scope": "all_time", "tower": "movie",
                    "reads": "imdb_averageRating: all-time aggregate through Oct 2026",
                    "ctx_keys": []},
    )


def build_movie_mean_rating_all(ctx) -> Feature:
    """TRAP: movie mean rating over ALL data incl. the validation future."""
    s: pd.Series = ctx["movie_mean_all"]
    gm = ctx["global_mean"]

    def compute(df, c):
        return pd.Series(df["movie_id"].map(s).fillna(gm), index=df.index)

    return Feature(
        name="movie_mean_rating_all",
        compute=compute,
        point_in_time="timestamp",
        description="TRAP (demo): movie mean rating computed over train AND "
                    "validation periods. Peeks at the future.",
        provenance={"sources": ["ml1m_ratings"], "temporal_scope": "all_time", "tower": "movie",
                    "reads": "mean over ALL ratings incl. validation period",
                    "ctx_keys": ["movie_mean_all"]},
    )


# ---------------------------------------------------------------------------
# Context: everything feature builders may touch. NOTE: ctx deliberately
# contains NO validation-period rating table -- compute() cannot peek at the
# future even if it wanted to; the audit additionally checks provenance.
# ---------------------------------------------------------------------------

def build_ctx(train: pd.DataFrame, valid: pd.DataFrame, stats: dict,
              cutoff_ts: int) -> dict:
    ctx: dict = {"stats": stats, "cutoff_ts": cutoff_ts,
                 "global_mean": float(stats["global_mean"]),
                 "global_like": float(train["liked"].mean())}
    ctx["_table_meta"] = {}

    # as-of event streams (train period only)
    ctx["movie_events"] = _events_by_key(train, "movie_id")
    ctx["user_events"] = _events_by_key(train, "user_id")
    ctx["_table_meta"]["movie_events"] = {"built_from": "train"}
    ctx["_table_meta"]["user_events"] = {"built_from": "train"}

    # user x genre like-rate (train only, min 5 ratings)
    tr = train[["user_id", "genres", "liked"]].copy()
    tr["genre"] = tr["genres"].str.split("|")
    tr = tr.explode("genre")
    ug = tr.groupby(["user_id", "genre"])["liked"].agg(["mean", "count"])
    ug = ug[ug["count"] >= 5][["mean"]].rename(columns={"mean": "like_rate"})
    ctx["user_genre_rate"] = ug
    ctx["_table_meta"]["user_genre_rate"] = {"built_from": "train"}

    # --- IMDb enrichment ---
    imdb = pd.read_csv(IMDB_PATH)
    imdb_map_avg = imdb.set_index("movieId")["imdb_averageRating"]
    ctx["imdb"] = imdb
    ctx["imdb_map_avg"] = imdb_map_avg

    # movie -> director (first listed), director -> release-sorted prefix sums
    imdb_idx = imdb.set_index("movieId")
    movie_to_director = {}
    for mid, row in imdb_idx.iterrows():
        d = row["directors"]
        if isinstance(d, str) and d.startswith("nm"):
            movie_to_director[int(mid)] = d.split(",")[0]
    ctx["movie_to_director"] = movie_to_director
    ctx["_table_meta"]["movie_to_director"] = {"built_from": "static",
                                               "note": "IMDb credits are time-invariant"}

    # release timestamps: IMDb startYear (Jan 1) else MovieLens title year
    import re
    ml_year = train.drop_duplicates("movie_id").set_index("movie_id")["title"] \
        .str.extract(r"\((\d{4})\)")[0].astype(float)
    release_ts = {}
    for mid in train["movie_id"].unique():
        y = None
        if mid in imdb_idx.index and pd.notna(imdb_idx.loc[mid, "startYear"]):
            y = float(imdb_idx.loc[mid, "startYear"])
        elif mid in ml_year.index and pd.notna(ml_year.loc[mid]):
            y = float(ml_year.loc[mid])
        if y is not None:
            release_ts[int(mid)] = int(pd.Timestamp(int(y), 1, 1, tz="UTC").timestamp())
    ctx["movie_release_ts"] = release_ts
    ctx["_table_meta"]["movie_release_ts"] = {"built_from": "static",
                                              "note": "release year is time-invariant"}
    ages = (train["timestamp"] - train["movie_id"].map(release_ts)) / DAY
    ctx["median_movie_age_days"] = float(ages[ages >= 0].median())

    # per-director prefix sums over release-sorted movies (train stats only)
    mstats = train.groupby("movie_id")["rating"].agg(["count", "sum"])
    dir_movies: dict[str, list] = {}
    for mid, d in movie_to_director.items():
        if mid in release_ts and mid in mstats.index:
            dir_movies.setdefault(d, []).append(
                (release_ts[mid], mstats.loc[mid, "count"], mstats.loc[mid, "sum"]))
    prefix = {}
    for d, lst in dir_movies.items():
        lst.sort(key=lambda t: t[0])
        rel = np.array([t[0] for t in lst])
        pc = np.concatenate([[0], np.cumsum([t[1] for t in lst])])
        ps = np.concatenate([[0.0], np.cumsum([t[2] for t in lst])])
        prefix[d] = (rel, pc, ps)
    ctx["director_prefix"] = prefix
    ctx["_table_meta"]["director_prefix"] = {"built_from": "train+imdb_static"}

    # LEAKY table, built deliberately for the trap demo (never used by legit features)
    full = pd.concat([train[["movie_id", "rating"]], valid[["movie_id", "rating"]]])
    ctx["movie_mean_all"] = full.groupby("movie_id")["rating"].mean()
    ctx["_table_meta"]["movie_mean_all"] = {"built_from": "future",
                                            "note": "includes validation period"}

    return ctx


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class RuleBasedEngine(HypothesisEngine):
    """Default engine: template hypotheses scheduled across rounds.

    Round 1: behavioral momentum + taste signals (internal data).
    Round 2: cold-start via IMDb (PIT-correct director signal) + the IMDb trap.
    Round 3: recency + the all-time-mean trap.
    Round 4: judgment layer -- legit movie_age_days + the liar trap + the
        budget-buster trap (see traps.py).
    """

    _SCHEDULE = {
        1: [
            ("movie_velocity_30d",
             "The model lacks any notion of momentum: a movie heating up (or "
             "cooling off) in the last 30 days should shift its predicted "
             "like-rate. Worst slices: cold/new movies.",
             ["ml1m_ratings"], build_movie_velocity_30d, False),
            ("user_genre_affinity",
             "The model knows coarse genre tags but not the user's personal "
             "taste per genre. A user who loves Drama but hates Horror should "
             "get different predictions for the two.",
             ["ml1m_ratings", "ml1m_movies"], build_user_genre_affinity, False),
        ],
        2: [
            ("director_track_record_pit",
             "Cold movies have no rating history, but IMDb tells us the "
             "director. A director's track record -- restricted to movies "
             "released BEFORE each prediction -- is a legitimate cold-start "
             "signal.",
             ["ml1m_ratings", "imdb_enrichment"], build_director_track_record_pit,
             False),
            ("imdb_average_rating_raw",
             "IMDb's average rating aggregates over a million votes -- it must "
             "encode true movie quality better than our sparse 2001 data.",
             ["imdb_enrichment"], build_imdb_average_rating_raw, True),
        ],
        3: [
            ("user_recency_30d",
             "Taste drifts: what a user liked last month predicts better than "
             "their lifetime average. As-of recency captures scale drift too.",
             ["ml1m_ratings"], build_user_recency_30d, False),
            ("movie_mean_rating_all",
             "The single strongest predictor of a rating should be the movie's "
             "overall mean -- computed over the complete dataset for maximum "
             "statistical power.",
             ["ml1m_ratings"], build_movie_mean_rating_all, True),
        ],
        # Round 4 (Days 5-6, judgment layer): one legit feature plus the two
        # new demo traps -- the liar and the budget-buster (see traps.py).
        4: [
            ("movie_age_days",
             "Novelty decays: a movie's age at prediction time should shift "
             "its like-rate. Time-invariant metadata, cheap to serve.",
             ["imdb_enrichment", "ml1m_movies"], build_movie_age_days, False),
            ("sneaky_movie_mean_pit",
             "An as-of movie mean rating should be a strong predictor -- "
             "computed point-in-time-correct per row.",
             ["ml1m_ratings"], build_sneaky_movie_mean, True),
            ("user_history_entropy",
             "Rater behavior matters: the entropy of a user's rating history "
             "captures how discriminating they are, which shifts P(liked).",
             ["ml1m_ratings"], build_user_history_entropy, True),
        ],
    }

    def propose(self, error_slices: dict, catalog: dict,
                round_no: int) -> list[Hypothesis]:
        specs = self._SCHEDULE.get(round_no, [])
        hyps = []
        for name, text, sources, builder, trap in specs:
            hyps.append(Hypothesis(
                text=text,
                feature_name=name,
                sources=sources,
                rationale=f"round {round_no} template; worst slices: "
                          f"{', '.join(s['slice'] for s in error_slices.get('worst', [])[:3])}",
                build=builder,
                trap=trap,
            ))
        return hyps


class LLMEngine(HypothesisEngine):
    """STUB -- seam for the Meta Model API (hackathon day).

    To activate: set META_MODEL_API_URL and META_MODEL_API_KEY, then
    implement propose() to POST _build_prompt(...) and parse the returned
    hypotheses into Hypothesis objects whose .build closures construct
    Features (the agent writes the builder code; the audit still judges it).

    Until then, RuleBasedEngine is the default. propose() raises
    NotImplementedError with setup instructions.
    """

    def __init__(self):
        self.url = os.environ.get("META_MODEL_API_URL")
        self.key = os.environ.get("META_MODEL_API_KEY")

    def _build_prompt(self, error_slices: dict, catalog: dict,
                      round_no: int) -> str:
        import json as _json
        worst = error_slices.get("worst", [])[:5]
        return (
            "You are a feature-discovery scientist for a movie recommender. "
            "Baseline AUC 0.7439. Propose 2 falsifiable hypotheses about what "
            "information the model is missing.\n\n"
            f"Worst error slices: {_json.dumps(worst)}\n\n"
            f"Data catalog: {_json.dumps(catalog, default=str)[:3000]}\n\n"
            "For each hypothesis return: name, plain-English text, which "
            "catalog sources it uses, and a point-in-time-correct computation "
            "plan. WARNING: imdb_averageRating is an all-time aggregate -- "
            "using it raw is temporal leakage and will be rejected."
        )

    def propose(self, error_slices: dict, catalog: dict,
                round_no: int) -> list[Hypothesis]:
        raise NotImplementedError(
            "LLMEngine is a stub. Set META_MODEL_API_URL and META_MODEL_API_KEY "
            "and implement the API call in LLMEngine.propose() (see "
            "_build_prompt for the prompt). Falling back to RuleBasedEngine."
        )


def get_engine(name: str) -> HypothesisEngine:
    if name == "rule":
        return RuleBasedEngine()
    if name == "llm":
        return LLMEngine()
    raise ValueError(f"unknown engine: {name}")
