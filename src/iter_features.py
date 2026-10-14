"""Feature builders for the cumulative iteration track (1% AUC / 1% NE target).

Extends the context built by scientist.build_ctx with Wikidata-backed tables
and adds new PIT-correct builders aimed at the biggest measured calibration
gaps (movie recency/staleness) and untapped catalog signals (cast track
records, user x director affinity, precomputed recency lookups).

All builders follow the same discipline as scientist.py: train-period-only
or as-of tables, code references the prediction timestamp, serving patterns
that pass the cost gate (lookup / row_local).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from features import Feature
from scientist import DAY, IMDB_PATH

WIKIDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "wikidata", "movie_wikidata.csv",
)

TOP_CAST_N = 5       # cast members kept per movie (billing order)
TRACK_TOP_K = 3      # top-billed actors averaged in the track-record feature
MIN_ACTOR_MOVIES = 2  # min train movies for an actor prefix table


def _qid_list(s) -> list[str]:
    if not isinstance(s, str) or not s:
        return []
    return [q for q in s.split("|") if q.startswith("Q")]


def build_wikidata_ctx(train: pd.DataFrame) -> dict:
    """Build Wikidata-backed ctx tables from train-period data only.

    Returns a dict of ctx entries (merge into the ctx from build_ctx):
      * tconst_to_movie: tconst -> ml movie_id
      * movie_to_cast: movie_id -> [actor QIDs] (billing order, top 5)
      * actor_prefix: actor QID -> (release_ts, cum_count, cum_rating_sum)
        over release-sorted TRAIN movies (same pattern as director_prefix)
      * movie_last_train_ts: movie_id -> max train timestamp (staleness)
      * user_director_rate: DataFrame indexed (user_id, director) with
        like_rate, min 5 train ratings
      * user_recency_lookup: DataFrame indexed (user_id, day) with the
        mean rating over the 30 days before that day (train events only)
      * movie_budget_log: movie_id -> log10(budget_amount), 32% coverage
      * budget_log_median: train-period median of movie_budget_log (fill)
      * movie_award_count: movie_id -> award count, 24% (ALL-TIME, trap)
      * movie_non_english: movie_id -> 1.0 iff original_language != Q1860
      * _table_meta: provenance for the leakage audit (merge, don't clobber)
    """
    ctx: dict = {}
    imdb = pd.read_csv(IMDB_PATH, usecols=["movieId", "tconst"])
    tconst_to_movie = dict(zip(imdb["tconst"], imdb["movieId"].astype(int)))
    ctx["tconst_to_movie"] = tconst_to_movie

    wd = pd.read_csv(WIKIDATA_PATH, usecols=["tconst", "cast"])
    movie_to_cast: dict[int, list[str]] = {}
    for tconst, cast in zip(wd["tconst"], wd["cast"]):
        mid = tconst_to_movie.get(tconst)
        if mid is not None:
            qids = _qid_list(cast)[:TOP_CAST_N]
            if qids:
                movie_to_cast[int(mid)] = qids
    ctx["movie_to_cast"] = movie_to_cast

    # release timestamps come from the main ctx; rebuild cheaply here from
    # train titles (same source as scientist.build_ctx)
    ml_year = train.drop_duplicates("movie_id").set_index("movie_id")["title"] \
        .str.extract(r"\((\d{4})\)")[0].astype(float)
    imdb_idx = pd.read_csv(IMDB_PATH, usecols=["movieId", "startYear"]) \
        .set_index("movieId")["startYear"]
    release_ts: dict[int, int] = {}
    for mid in train["movie_id"].unique():
        mid = int(mid)
        y = None
        if mid in imdb_idx.index and pd.notna(imdb_idx.loc[mid]):
            y = float(imdb_idx.loc[mid])
        elif mid in ml_year.index and pd.notna(ml_year.loc[mid]):
            y = float(ml_year.loc[mid])
        if y is not None:
            release_ts[mid] = int(pd.Timestamp(int(y), 1, 1, tz="UTC").timestamp())
    ctx["wd_release_ts"] = release_ts

    # per-actor prefix sums over release-sorted train movies
    mstats = train.groupby("movie_id")["rating"].agg(["count", "sum"])
    actor_movies: dict[str, list] = {}
    for mid, qids in movie_to_cast.items():
        if mid in release_ts and mid in mstats.index:
            for q in qids:
                actor_movies.setdefault(q, []).append(
                    (release_ts[mid], mstats.loc[mid, "count"],
                     mstats.loc[mid, "sum"]))
    actor_prefix = {}
    for q, lst in actor_movies.items():
        if len(lst) < MIN_ACTOR_MOVIES:
            continue
        lst.sort(key=lambda t: t[0])
        rel = np.array([t[0] for t in lst])
        pc = np.concatenate([[0], np.cumsum([t[1] for t in lst])])
        ps = np.concatenate([[0.0], np.cumsum([t[2] for t in lst])])
        actor_prefix[q] = (rel, pc, ps)
    ctx["actor_prefix"] = actor_prefix

    # --- fresh Wikidata territory (v3): budget / awards / language --------
    # budget_amount is locked at release, so a log-budget prior is
    # time-invariant (PIT-safe); it is sparse (32% coverage) and missing
    # values are filled with the train-period median, honestly logged.
    # original_language is time-invariant (98% coverage; Q1860 = English).
    # awards are ALL-TIME as of the crawl (24% coverage) -- FAIL-CLOSED:
    # the leakage audit must reject any feature built on them.
    _wd2 = pd.read_csv(WIKIDATA_PATH,
                       usecols=["tconst", "budget_amount", "awards",
                                "original_language"])
    budget_log: dict[int, float] = {}
    award_count: dict[int, float] = {}
    non_english: dict[int, float] = {}
    for tconst, budget, awards, lang in zip(
            _wd2["tconst"], _wd2["budget_amount"],
            _wd2["awards"], _wd2["original_language"]):
        mid = tconst_to_movie.get(tconst)
        if mid is None:
            continue
        mid = int(mid)
        if pd.notna(budget) and float(budget) > 0:
            budget_log[mid] = float(np.log10(float(budget)))
        if isinstance(awards, str) and awards:
            award_count[mid] = float(
                len([q for q in awards.split("|") if q.startswith("Q")]))
        if isinstance(lang, str) and lang:
            non_english[mid] = 0.0 if lang == "Q1860" else 1.0
    ctx["movie_budget_log"] = budget_log
    ctx["budget_log_median"] = float(np.median(list(budget_log.values()))) \
        if budget_log else 0.0
    ctx["movie_award_count"] = award_count
    ctx["movie_non_english"] = non_english

    # staleness: last train sighting per movie
    ctx["movie_last_train_ts"] = train.groupby("movie_id")["timestamp"].max()

    # user x director like-rate (train only, min 5)
    movie_to_director: dict[int, str] = {}
    imdb_d = pd.read_csv(IMDB_PATH, usecols=["movieId", "directors"])
    for mid, d in zip(imdb_d["movieId"], imdb_d["directors"]):
        if isinstance(d, str) and d.startswith("nm"):
            movie_to_director[int(mid)] = d.split(",")[0]
    tr = train[["user_id", "movie_id", "liked"]].copy()
    tr["director"] = tr["movie_id"].map(movie_to_director)
    tr = tr.dropna(subset=["director"])
    ud = tr.groupby(["user_id", "director"])["liked"].agg(["mean", "count"])
    ud = ud[ud["count"] >= 5][["mean"]].rename(columns={"mean": "like_rate"})
    ctx["user_director_rate"] = ud

    # precomputed 30d recency lookup: (user_id, day) -> mean rating over the
    # 30 days before that day, from TRAIN events only. Day range covers the
    # full train+validation span so validation lookups hit precomputed rows.
    # PIT-correct: every window reads train events with ts < day boundary.
    user_ev = train[["user_id", "timestamp", "rating"]].sort_values("timestamp")
    day0 = int(user_ev["timestamp"].min() // DAY)
    day1 = int(train["timestamp"].max() // DAY) + 400  # covers validation span
    win = 30 * DAY
    rows = []
    for uid, g in user_ev.groupby("user_id"):
        ts = g["timestamp"].to_numpy()
        rt = g["rating"].to_numpy()
        cs = np.concatenate([[0.0], np.cumsum(rt)])
        days = np.arange(day0, day1 + 1)
        bounds = days * DAY
        lo = np.searchsorted(ts, bounds - win, side="left")
        hi = np.searchsorted(ts, bounds, side="left")
        cnt = hi - lo
        tot = cs[hi] - cs[lo]
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        have = cnt > 0
        rows.append(pd.DataFrame(
            {"user_id": uid, "day": days[have], "rec30": mean[have]}))
    ctx["user_recency_lookup"] = pd.concat(rows, ignore_index=True) \
        .set_index(["user_id", "day"])[["rec30"]]

    # provenance for the leakage audit (merged into ctx["_table_meta"] by
    # callers; kept separate so a plain dict.update() can't clobber it)
    ctx["_table_meta"] = {
        "movie_budget_log": {
            "built_from": "static",
            "note": "log10(budget_amount), 32% coverage, missing->median; "
                    "budgets are locked pre-release (PIT-safe)"},
        "budget_log_median": {
            "built_from": "train",
            "note": "scalar fill for missing budgets"},
        "movie_award_count": {
            "built_from": "static",
            "note": "ALL-TIME awards as of crawl, 24% coverage -- FAIL-CLOSED: "
                    "wins may postdate prediction time"},
        "movie_non_english": {
            "built_from": "static",
            "note": "original_language != English(Q1860), 98% coverage, "
                    "missing->0; time-invariant"},
    }
    return ctx


def merge_wikidata_ctx(ctx: dict, train: pd.DataFrame) -> dict:
    """Merge build_wikidata_ctx(train) into ctx without clobbering the base
    _table_meta (a plain dict.update would overwrite it). Returns ctx."""
    wd = build_wikidata_ctx(train)
    meta = wd.pop("_table_meta", {})
    ctx.update(wd)
    ctx.setdefault("_table_meta", {}).update(meta)
    return ctx


# ---------------------------------------------------------------------------
# New builders
# ---------------------------------------------------------------------------

def build_actor_track_record_pit(ctx) -> Feature:
    """Mean track record of the top-billed cast, PIT-correct.

    For each row, eligible movies are the cast members' movies with
    release_ts < row_ts, scored by TRAIN-period ratings (prefix sums).
    Averages the top-3 billed actors' records. Targets cold-start: cast
    has 95% coverage vs directors' 99%, and the signals are complementary.
    """
    actor_prefix: dict = ctx["actor_prefix"]
    movie_to_cast: dict = ctx["movie_to_cast"]
    gm = ctx["global_mean"]

    def compute(df, c):
        mids = df["movie_id"].to_numpy()
        rts = df["timestamp"].to_numpy()
        n = len(df)
        acc = np.zeros(n)
        cnt = np.zeros(n)
        for m in np.unique(mids):
            qids = movie_to_cast.get(int(m), [])[:TRACK_TOP_K]
            if not qids:
                continue
            pm = mids == m
            mrt = rts[pm]
            for q in qids:
                ev = actor_prefix.get(q)
                if ev is None:
                    continue
                rel, pc, ps = ev
                k = np.searchsorted(rel, mrt, side="left")
                cc = pc[k]
                ss = ps[k]
                ok = cc > 0
                # accumulate only where the actor has prior movies
                idx = np.flatnonzero(pm)
                acc[idx[ok]] += ss[ok] / np.maximum(cc[ok], 1)
                cnt[idx[ok]] += 1
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1), gm)
        return pd.Series(out, index=df.index)

    return Feature(
        name="actor_track_record_pit",
        compute=compute,
        point_in_time="timestamp",
        description="Mean train-period rating of the top-3 billed cast's "
                    "movies released strictly before each row's timestamp. "
                    "Cold-start signal from Wikidata, point-in-time correct.",
        provenance={"sources": ["ml1m_ratings", "wikidata"],
                    "temporal_scope": "pit_correct", "tower": "movie",
                    "reads": "train ratings of cast movies with release_ts < row_ts",
                    "ctx_keys": ["actor_prefix", "movie_to_cast"]},
        serving="lookup",
        serving_notes="precomputable per (actor, month); O(1) at request time",
    )


def build_movie_staleness_days(ctx) -> Feature:
    """Days since the movie's last train-period rating, at prediction time.

    Directly targets the strongest measured calibration gap: stale movies
    are over-predicted by ~4pp, fresh movies under-predicted by ~3.3pp.
    Train-only table + row timestamp -> PIT-safe, row-local serving.
    """
    last_ts: pd.Series = ctx["movie_last_train_ts"]
    med = float((ctx["cutoff_ts"] - last_ts).median() / DAY)

    def compute(df, c):
        lt = df["movie_id"].map(last_ts)
        out = (df["timestamp"] - lt) / DAY
        return pd.Series(out.fillna(med).clip(lower=0), index=df.index)

    return Feature(
        name="movie_staleness_days",
        compute=compute,
        point_in_time="timestamp",
        description="Days from the movie's last train-period rating to "
                    "prediction time. Captures catalog staleness; targets "
                    "the measured 4pp over-prediction on stale movies.",
        provenance={"sources": ["ml1m_ratings"],
                    "temporal_scope": "pit_correct", "tower": "movie",
                    "reads": "max train timestamp per movie (train only)",
                    "ctx_keys": ["movie_last_train_ts"]},
        serving="row_local",
        serving_notes="timestamp minus static per-movie table; pure row function",
    )


def build_user_director_affinity(ctx) -> Feature:
    """User's train-period like-rate for the movie's director."""
    table: pd.DataFrame = ctx["user_director_rate"]
    movie_to_director: dict = ctx.get("movie_to_director", {})
    global_like = ctx["global_like"]

    def compute(df, c):
        d = df["movie_id"].map(movie_to_director)
        keys = pd.MultiIndex.from_arrays(
            [df["user_id"].to_numpy(), d.to_numpy()],
            names=["user_id", "director"])
        vals = table["like_rate"].reindex(keys).to_numpy()
        out = np.where(np.isnan(vals), global_like, vals)
        return pd.Series(out, index=df.index)

    return Feature(
        name="user_director_affinity",
        compute=compute,
        point_in_time="timestamp",
        description="The user's train-period like-rate for movies by this "
                    "movie's director (min 5 ratings). User x movie taste "
                    "interaction from behavior + IMDb.",
        provenance={"sources": ["ml1m_ratings", "imdb_enrichment"],
                    "temporal_scope": "train_only", "tower": "wide",
                    "reads": "user x director like-rate table (train only)",
                    "ctx_keys": ["user_director_rate", "movie_to_director"]},
        serving="lookup",
        serving_notes="precomputed user x director table, O(1)",
    )


def build_user_recency_30d_lookup(ctx) -> Feature:
    """User's mean rating over the prior 30 days, as a precomputed lookup.

    Same signal as the history-scan version, redesigned as an O(1) lookup
    so it passes the serving-cost gate. PIT-correct: every window reads
    train events strictly before the day boundary.
    """
    table: pd.DataFrame = ctx["user_recency_lookup"]
    user_mean = ctx["stats"]["user_mean"].to_dict()
    gm = ctx["global_mean"]

    def compute(df, c):
        keys = pd.MultiIndex.from_arrays(
            [df["user_id"].to_numpy(),
             (df["timestamp"] // DAY).astype(int).to_numpy()],
            names=["user_id", "day"])
        vals = table["rec30"].reindex(keys).to_numpy()
        um = df["user_id"].map(user_mean).fillna(gm).to_numpy()
        out = np.where(np.isnan(vals), um, vals)
        return pd.Series(out, index=df.index)

    return Feature(
        name="user_recency_30d_lookup",
        compute=compute,
        point_in_time="timestamp",
        description="User's mean rating over the 30 days before prediction, "
                    "precomputed per (user, day) from train events. Taste "
                    "drift without the history-scan serving cost.",
        provenance={"sources": ["ml1m_ratings"],
                    "temporal_scope": "pit_correct", "tower": "user",
                    "reads": "train events with ts in [day-30d, day)",
                    "ctx_keys": ["user_recency_lookup"]},
        serving="lookup",
        serving_notes="precomputed (user_id, day) table, O(1)",
    )


# Candidate pool for the iteration driver, ordered by expected value:
# strongest measured gaps / fresh signals first.
CANDIDATE_BUILDERS = [
    build_movie_staleness_days,      # 4.0pp recency calibration gap
    build_actor_track_record_pit,    # cast 95% coverage, new signal
    build_user_recency_30d_lookup,   # taste drift, lookup-redesigned
    build_user_director_affinity,    # user x movie interaction
]
