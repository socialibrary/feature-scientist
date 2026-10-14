"""Scripted backend v3: fresh Wikidata-territory hypotheses.

Deterministic canned backend for the feature-scientist agentic path,
selected with FS_SCRIPTED=v3. No network. Call order is fixed and
documented here.

Staged arc (fresh territory the deterministic iteration pool never
explored with the LLM -- budget 32%, original_language 98%, awards 24%):

  H1 movie_budget_prior_llm (static, lookup, movie): log10(budget_amount)
     as a prestige/blockbuster prior for cold movies. Budgets are locked
     pre-release so the prior is PIT-safe; sparse (32%) so missing values
     are filled with the train-period median, honestly declared.
     Critic PASS -> audit PASS -> cost PASS -> ablate.

  H2 movie_nonenglish_llm (static, lookup, movie): indicator for
     non-English original_language. Time-invariant, 98% coverage,
     missing->0, documented. Critic PASS -> audit PASS -> cost PASS ->
     ablate.

  H3 movie_awards_prior_llm (all_time, lookup, movie): raw award count as
     a prestige prior -- the JUDGMENT demo on fresh territory. The
     hypothesis honestly declares temporal_scope=all_time; the critic
     passes with caution and defers to the audit, which FAILS it
     fail-closed (awards are all-time as of the crawl; wins may postdate
     prediction time). One revision round: the proposer attempts an
     "as-of" version, but the awards table has no per-award dates, so the
     code contains no prediction-time reference -- the audit kills it
     again (declaration not credible). Final verdict: rejected. The ledger
     shows the model trying to launder an all-time aggregate and the
     machinery catching it twice.

The audit kills here are real: they come from leakage.py's FAIL_SCOPES
and the B4 pit_correct credibility check, not from the script.
"""

import json

_V3_H1_CODE = '''def compute(df, ctx):
    v = df["movie_id"].map(ctx["movie_budget_log"])
    return pd.Series(v.fillna(ctx["budget_log_median"]), index=df.index)
'''

_V3_H2_CODE = '''def compute(df, ctx):
    v = df["movie_id"].map(ctx["movie_non_english"])
    return pd.Series(v.fillna(0.0), index=df.index)
'''

# H3: raw award count. The code is deliberately plain -- the trap is the
# all_time declaration, which the deterministic audit fails closed.
_V3_H3_CODE = '''def compute(df, ctx):
    v = df["movie_id"].map(ctx["movie_award_count"])
    return pd.Series(v.fillna(0.0), index=df.index)
'''

# H3 revise: "as-of awards". No per-award dates exist in the catalog, so
# this code has no prediction-time reference and no as-of logic. It
# declares pit_correct -- the audit's B4 credibility check must kill it.
_V3_H3R_CODE = '''def compute(df, ctx):
    v = df["movie_id"].map(ctx["movie_award_count"])
    return pd.Series(v.fillna(0.0), index=df.index)
'''

V3_SCRIPT = {
    "propose": [json.dumps({
        "reasoning": (
            "Worst slices: cold movies with thin rating history and the "
            "~4pp movie-recency gap. The iteration pool never explored the "
            "Wikidata columns with the LLM proposer, so the fresh "
            "information is there: (1) production budget -- a prestige / "
            "blockbuster prior locked pre-release, PIT-safe, 32% coverage; "
            "(2) original language -- non-English films may carry a "
            "different like-rate in this US-centric dataset, "
            "time-invariant, 98% coverage; (3) award counts -- prestige "
            "signal with 24% coverage, but awards are ALL-TIME as of the "
            "crawl, so this one is fail-closed territory and I declare it "
            "honestly as all_time."
        ),
        "hypotheses": [
            {"name": "movie_budget_prior_llm",
             "text": "Production budget proxies prestige and marketing "
                     "push: big-budget films get a head start with cold "
                     "audiences. Budget is locked pre-release, so log10 "
                     "budget is a PIT-safe movie-side prior.",
             "sources": ["wikidata_movies"],
             "point_in_time": "timestamp",
             "temporal_scope": "static",
             "tower": "movie",
             "serving": "lookup",
             "serving_notes": "O(1) map movie_id -> log10 budget; missing "
                              "-> train-period median (declared, 32% "
                              "coverage)",
             "code": _V3_H1_CODE},
            {"name": "movie_nonenglish_llm",
             "text": "Non-English original-language films behave "
                     "differently in this US-centric dataset -- subtitled "
                     "or dubbed audiences self-select. A time-invariant "
                     "indicator captures the segment effect.",
             "sources": ["wikidata_movies"],
             "point_in_time": "timestamp",
             "temporal_scope": "static",
             "tower": "movie",
             "serving": "lookup",
             "serving_notes": "O(1) map movie_id -> 0/1; missing -> 0 "
                              "(98% coverage, documented)",
             "code": _V3_H2_CODE},
            {"name": "movie_awards_prior_llm",
             "text": "Award counts are a direct prestige signal: winning "
                     "films get watched and liked more. Declared honestly "
                     "as all_time -- awards are all-time as of the crawl "
                     "and wins may postdate prediction time, so this is a "
                     "fail-closed trap the audit must catch.",
             "sources": ["wikidata_movies"],
             "point_in_time": "timestamp",
             "temporal_scope": "all_time",
             "tower": "movie",
             "serving": "lookup",
             "serving_notes": "O(1) map movie_id -> award count",
             "code": _V3_H3_CODE},
        ],
    })],
    # Name-keyed critiques (ScriptedBackend checks this map before the
    # order-based list).
    "critique_by_name": {
        "movie_budget_prior_llm": json.dumps({
            "verdict": "pass",
            "critique": (
                "Row-local map, 1.0 latency units. Budgets are locked "
                "pre-release: PIT-clean by construction. Sparse (32%) but "
                "the median fill is declared up front. Weak signal expected, "
                "cheap enough to test."
            ),
            "suggested_fix": "",
        }),
        "movie_nonenglish_llm": json.dumps({
            "verdict": "pass",
            "critique": (
                "Lookup, 1.0u, time-invariant original_language (98% "
                "coverage, missing->0 documented). PIT-clean by "
                "construction. A segment indicator, not a leak."
            ),
            "suggested_fix": "",
        }),
        "movie_awards_prior_llm": json.dumps({
            "verdict": "pass",
            "critique": (
                "The hypothesis honestly declares temporal_scope=all_time. "
                "Awards are all-time aggregates (wins may postdate T); I "
                "cannot approve this from the code alone -- the "
                "deterministic audit is the decider. Pass with caution."
            ),
            "suggested_fix": "",
        }),
    },
    # revise calls in order: H3 after audit-reject -> H3r (as-of attempt
    # with no as-of logic; the audit's B4 credibility check kills it).
    "revise": [json.dumps({
        "name": "movie_awards_asof_llm",
        "text": "Revised after the audit rejected raw award counts: count "
                "only awards the film had plausibly won by prediction "
                "time. But the awards table has no per-award dates, so no "
                "as-of logic is implementable -- this spec is a laundering "
                "attempt and the audit should catch it again.",
        "sources": ["wikidata_movies"],
        "point_in_time": "timestamp",
        "temporal_scope": "pit_correct",
        "tower": "movie",
        "serving": "lookup",
        "serving_notes": "O(1) map movie_id -> award count (same table)",
        "code": _V3_H3R_CODE,
    })],
}
