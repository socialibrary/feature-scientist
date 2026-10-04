"""Minimal feature registry stub (Days 1-2 of the build plan).

Every feature registers three things:
  - name: unique string id
  - compute: callable (df, ctx) -> pd.Series, aligned to df.index
  - point_in_time: name of the timestamp column that bounds what the feature
    is allowed to read. A feature is *valid* only if every source row it
    reads satisfies  source[point_in_time] <= prediction time of the scored row.

The Day 5-6 "judgment layer" (leakage detector) will audit exactly this
declaration: any feature whose declared point_in_time is violated -- e.g. a
"movie mean rating" computed over data newer than the prediction -- gets
rejected on camera, no matter how good its validation score looks.

For now the registry holds one example *correct* feature: movie/user mean
ratings computed strictly from the train period (see baseline.py). The agent
loop (Days 3-4) will register new candidate features here; the leakage
checker (Days 5-6) will read the point_in_time declarations to judge them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

# compute signature: (dataframe_to_score, context_dict) -> Series
FeatureFn = Callable[[pd.DataFrame, dict], pd.Series]


@dataclass
class Feature:
    name: str
    compute: FeatureFn
    point_in_time: str = "timestamp"
    description: str = ""
    # Provenance powers the Day 5-6 leakage audit. temporal_scope is one of:
    #   "train_only"  - reads only aggregates/tables built from train rows
    #   "pit_correct" - reads timestamped data but only at/before each row's
    #                   prediction time (as-of / point-in-time logic)
    #   "static"      - reads time-invariant metadata (no leakage possible)
    #   "all_time"    - reads aggregates that include data NEWER than the
    #                   prediction time  -> TEMPORAL LEAKAGE, must be rejected
    provenance: dict = field(default_factory=dict)


@dataclass
class FeatureRegistry:
    """Ordered registry of candidate features for the science loop."""

    _features: dict[str, Feature] = field(default_factory=dict)

    def register(self, feature: Feature) -> None:
        if feature.name in self._features:
            raise ValueError(f"feature already registered: {feature.name}")
        self._features[feature.name] = feature

    def names(self) -> list[str]:
        return list(self._features)

    def compute_all(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        """Compute every registered feature for the rows of df."""
        out = pd.DataFrame(index=df.index)
        for feature in self._features.values():
            out[feature.name] = feature.compute(df, ctx).astype("float64")
        return out


# ---------------------------------------------------------------------------
# Example CORRECT feature: movie mean rating, train period only.
# ctx must carry "movie_mean_train" (pd.Series indexed by movie_id) and
# "global_mean_train" (float) -- both built from train rows only, so scoring
# a validation row can never peek at the future.
# ---------------------------------------------------------------------------

def _movie_mean_rating_train(df: pd.DataFrame, ctx: dict) -> pd.Series:
    return (
        df["movie_id"]
        .map(ctx["movie_mean_train"])
        .fillna(ctx["global_mean_train"])
    )


registry = FeatureRegistry()
registry.register(
    Feature(
        name="movie_mean_rating_train",
        compute=_movie_mean_rating_train,
        point_in_time="timestamp",
        description=(
            "Mean rating of the movie, computed ONLY from ratings in the "
            "train period (timestamp <= split cutoff). Safe for validation "
            "rows because the lookup table is frozen before the split."
        ),
    )
)

# EXTENSION POINTS (used by later build stages):
#   - Days 3-4 (agent loop): call registry.register(Feature(...)) for each
#     candidate the proposer agent invents; pass ctx with whatever lookup
#     tables the feature needs.
#   - Days 5-6 (judgment layer): iterate registry.names(), read each
#     feature.point_in_time, and verify the compute() implementation only
#     reads source rows at or before that bound. Violations -> rejection.
