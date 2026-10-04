# Calibration screening: integration guide

The calibration screen (`src/calibration.py`) is a **screening stage** that
sits between error analysis and ablation in the science loop. It answers
"which of these 50 candidate features is worth spending retraining budget
on?" deterministically and cheaply: no retraining, inference only.

## Where it plugs in

New stage in the per-round pipeline (`src/run_scientist.py` — to be wired
by the track that owns it):

```
error analysis (slices) → CALIBRATION SCREEN (new) → ablation (top-k) → ledger
```

Concretely, after the engine proposes candidate features and their values
are computed for the validation frame, insert:

```python
from calibration import screen_candidates, screen_sparse_id, format_table

# candidates: {feature_name: pd.Series aligned with the validation frame}
results = screen_candidates(y_valid, baseline_proba, candidates)
print(format_table(results))
top = [r for r in results if r["verdict"] == "signal"][:5]
# -> ablate only `top`; the rest never cost a training run
```

For high-cardinality ID features (movie/user/director), never screen the
raw ID. Use the sparse screen instead:

```python
sparse = screen_sparse_id(
    y_valid, baseline_proba,
    valid_ids=valid["movie_id"],
    train_id_counts=train.groupby("movie_id").size(),   # train period only
    train_id_last_ts=train.groupby("movie_id")["timestamp"].max(),
    valid_ts=valid["timestamp"],
    id_name="movie_id",
)
# sparse["popularity"]["summary"] ->
#   "movies with <5 train ratings: under-predicted by X.Xpp (significant)"
```

## Function signatures

```python
calibration_screen(y_true, y_pred, feature_values, feature_name,
                   n_buckets=10, min_bucket_n=30, key_bucket=None) -> dict
# -> {"feature", "verdict", "max_significant_gap_pp", "n_buckets",
#     "buckets": [{label, n, mean_pred, actual_rate, gap_pp, se_pp,
#                   significant, significant_adj, powered}],
#     "blind_spots", "summary"}

screen_candidates(y_true, y_pred, candidates: dict[str, Series],
                  n_buckets=10, min_bucket_n=30,
                  key_buckets: dict | None = None) -> list[dict]
# ranked signal-first, then by max significant gap

screen_sparse_id(y_true, y_pred, valid_ids, train_id_counts, id_name,
                 train_id_last_ts=None, valid_ts=None,
                 min_bucket_n=30) -> dict
# -> {"popularity": <screen>, "recency": <screen> | None, "notes": [...]}

format_table(results) -> str   # ranked plain-text table for logs / prompts
```

## Feeding the LLM prompt

Each result carries `summary`: one plain-English line, e.g.

> movie_age_days: newest decile under-predicted by 2.1pp (significant),
> oldest decile over-predicted by 1.4pp (significant)

These summaries are the legible signal for hypothesis generation. They
live at the scale verbal reasoning *can* work with (percentage-point
calibration gaps per segment), unlike a +0.0005 AUC delta, which no LLM
can reason about. Suggested prompt block:

```
Calibration screen (baseline miscalibration by candidate feature;
positive gap = model under-predicts the segment):
- <summary line per screened feature>
Propose features that explain the largest significant gaps. Prefer
segments the model systematically under-predicts.
```

## What the screen can and cannot say about sparse features

**High-cardinality categoricals (movie/user/director IDs).** Bucketing by
raw ID value is meaningless — ten thousand singleton buckets prove
nothing. `screen_sparse_id` auto-derives dense proxies instead:

- **popularity**: train-period frequency bucket
  (`unseen (0)` / `rare (1-4)` / `5-19` / `20-99` / `100+`). A gap on the
  rare segment is the cold-start diagnostic: "movies with <5 train
  ratings: under-predicted by Xpp" is directly actionable (add a
  popularity-aware prior / back-off).
- **recency**: days since the ID was last seen in train, quantile-bucketed
  (`unseen` gets its own bucket). Needs `train_id_last_ts` + `valid_ts`;
  skipped with a note when unavailable.

What it *can* say: which sparse *segments* the model mishandles, and in
which direction. What it *cannot* say: anything about an individual ID —
per-ID gaps at n=1 are noise, and the screen refuses to bucket that way.

**Sparse binary / rare flags.** Screened as two buckets `{0, 1}` with the
binomial significance test. When the positive bucket is underpowered
(`n < min_bucket_n`, default 30), the verdict is **insufficient_evidence**,
never "no signal" — the output schema and the plain-English summary both
preserve the absence-of-evidence vs evidence-of-absence distinction:

> rare_flag: informative bucket '1' too small (n=12) — insufficient
> evidence, not no signal

Pass `key_bucket="1"` (the informative bucket's label) to get this
behavior; it also applies to any named bucket of interest.

## Statistical notes (for the demo and the writeup)

- **Two-tier significance.** Each bucket carries a raw 2-SE flag
  (`|gap| > 2·SE`, SE = binomial SE of the actual rate). The feature-level
  `signal` verdict additionally requires clearing a Bonferroni-adjusted
  threshold over buckets: with B buckets, ~B·0.05 would trip the raw flag
  by chance. The screen *prioritizes*; the multi-seed ablation *proves*.
- **Power gate.** Buckets with `n < min_bucket_n` are marked
  `powered=false` and can never be significant; they appear in
  `blind_spots` instead.
- **NaNs** get their own `missing` bucket — a gap there means the model
  mishandles unknown values, which is itself actionable.
- The screen is **deterministic given (y_true, y_pred)**: no retraining,
  no seed noise. That is what makes it trustworthy at 0.1%-gain scale,
  where single-run ablation deltas are mostly noise.
- A `signal` verdict is a *screening* claim ("worth ablating"), not an
  acceptance. Acceptance still requires the multi-seed ablation with the
  leakage audit and serving-cost gate.

## Demo

```bash
cd src && ../.venv/bin/python demo_calibration.py
```

Uses the existing LightGBM baseline's validation predictions (cached at
`results/lgbm_valid_proba.npz` after a one-time deterministic
generation — no retraining inside the demo; the two-tower model is
mid-migration in a parallel track, and the screen is model-agnostic
anyway). Verifies: `movie_age_days`
shows a significant gap (same signal its +0.0005 AUC ablation found),
random noise shows none, rare movies are diagnosed via the popularity
proxy, and an underpowered rare flag reports insufficient evidence.
