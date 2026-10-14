#!/usr/bin/env python3
"""Propose-only Nemotron run: fresh MovieLens hypotheses from the Wikidata catalog.

Validates the tightened LLM output schema (commit 33937e3: EXACTLY-ONE-OF
enums for temporal_scope/tower/serving) end-to-end through LLMEngine.propose
-> codegen.parse_llm_response -> codegen.validate_spec, WITHOUT training or
ablations. Output is staged at results/nemotron_fresh_proposals_<date>.json
for the ablate decision.

The problem context tells Nemotron the exhausted 8-candidate pool so it does
not re-propose them, and points it at the Wikidata-enriched catalog territory
(directors 99%, cast 95%, awards 24%, box office 27%, budget 32%, release_date
99%, original_language 98% coverage) that the iteration driver never explored
with the LLM proposer.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scientist import LLMEngine  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXHAUSTED_POOL = [
    "movie_staleness_days",      # degenerate (train-constant) -- do not re-propose
    "actor_track_record_pit",
    "user_recency_30d_lookup",
    "user_director_affinity",
    "movie_age_days",
    "director_track_record_pit",
    "user_genre_affinity",       # single-seed phantom, 3-seed overturn -- noise
    "movie_velocity_30d",
]

ERROR_SLICES = {
    "worst": [
        {"slice": "movie recency: days between release_date and rating timestamp",
         "val_auc": 0.71, "base_auc": 0.7496,
         "gap": -0.04,
         "note": "top significant gap from the calibration screen (~4pp)"},
        {"slice": "cold movies: < 20 train ratings",
         "val_auc": 0.73, "base_auc": 0.7496,
         "gap": -0.02,
         "note": "cold-start segment under-predicted"},
        {"slice": "movies with budget_amount present vs absent",
         "val_auc": None, "base_auc": 0.7496,
         "gap": None,
         "note": "unexplored segment; Wikidata coverage signal never proposed"},
    ]
}

CATALOG = {
    "ml1m_ratings": "1,000,209 rows; user_id, movie_id, rating 1-5, timestamp (prediction time T)",
    "ml1m_users": "6,040 rows; user_id, gender, age bucket, occupation code, zip -- static demographics",
    "ml1m_movies": "3,900 rows; movie_id, title+year, pipe-separated genres",
    "imdb_enrichment": "86% of movies; tconst, startYear, runtimeMinutes, directors, "
                       "imdb_averageRating / imdb_numVotes = ALL-TIME aggregates (leakage if raw)",
    "wikidata_movies": {
        "join": "tconst == imdb_enrichment.tconst (99% matched)",
        "columns": {
            "directors": "QIDs |-sep, 99% cov, time_invariant, lookup",
            "cast": "QIDs |-sep, 95% cov, time_invariant, lookup",
            "awards": "QIDs |-sep, 24% cov, ALL_TIME as of crawl (FAIL-CLOSED: wins may postdate T)",
            "box_office_amount": "float, 27% cov, ALL_TIME final gross, MIXED CURRENCIES (leakage if raw)",
            "budget_amount": "float, 32% cov, time_invariant in principle (pre-release), very sparse",
            "release_date": "P577 YYYY-MM-DD, 99% cov, time_invariant, lookup",
            "original_language": "P364 QID, 98% cov, time_invariant, lookup",
            "wikidata_qid": "entity id, 99% cov, time_invariant",
        },
    },
}

PROBLEM_CONTEXT = {
    "problem": "predict P(user likes movie) for ranking",
    "label": "liked = rating >= 4, observed at rating timestamp",
    "prediction_time": "score at impression time T; features may only use data with timestamp <= T",
    "calibration_gaps": [
        "-4pp on movie-recency slice (days since release); cold movies under-predicted ~2pp",
        f"EXHAUSTED CANDIDATE POOL (do not re-propose): {', '.join(EXHAUSTED_POOL)}",
        "The deterministic iteration driver accepted 0 features over 8 candidates. "
        "Prefer genuinely new information, especially from wikidata_movies columns "
        "the pool never used with the LLM proposer (awards, box_office_amount, "
        "budget_amount, release_date, original_language).",
    ],
}


def main() -> int:
    engine = LLMEngine(backend="nebius", problem_context=PROBLEM_CONTEXT)
    print(f"backend configured: {engine.is_live()} (model: {engine.backend.model})")
    if not engine.is_live():
        print("Nebius backend not configured; aborting.")
        return 2
    hyps = engine.propose(ERROR_SLICES, CATALOG, round_no=1)
    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": engine.backend.model,
        "backend": "nebius",
        "note": "propose-only: schema validation + fresh hypotheses; NO training, NO ablations",
        "hypotheses": [
            {
                "name": h.feature_name,
                "text": h.text,
                "sources": h.sources,
                "reasoning": h.reasoning,
                "llm_spec": h.llm_spec,
            }
            for h in hyps
        ],
    }
    results_dir = os.path.join(BASE, "results")
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(results_dir, f"nemotron_fresh_proposals_{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\npropose-only OK: {len(hyps)} hypotheses, schema-validated")
    for h in hyps:
        print(f"  - {h.feature_name}: {h.text[:110]}")
        print(f"    scope={h.llm_spec['temporal_scope']} tower={h.llm_spec['tower']} "
              f"serving={h.llm_spec['serving']} sources={h.sources}")
    print(f"\nstaged: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
