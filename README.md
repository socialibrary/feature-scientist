# Feature Scientist — ML Harness (Days 1–2)

An autonomous "AI Feature Scientist" agent for the Meta global AI hackathon:
given a model and an objective, it figures out **what information the model is
missing**, invents new features, proves they help, and rejects the ones that
leak or cost too much to serve.

This repo currently holds the **Days 1–2 harness**: data loading, a strict
temporal split, a baseline model, and a feature-registry stub. The agent loop
(Days 3–4), the judgment layer / leakage detector (Days 5–6), the arena UI
(Days 7–8), and demo rehearsal (Days 9–10) build on top of it.

Dataset: **MovieLens-1M** (public, https://grouplens.org/datasets/movielens/1m/)
— 1,000,209 ratings, 6,040 users, 3,706 movies. No Meta-internal anything.

## Setup

```bash
cd ~/workspace/feature-scientist
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(A `.venv` with everything installed already exists in this directory.)

## Run

```bash
.venv/bin/python src/data.py      # load + join the three .dat files
.venv/bin/python src/split.py     # strict temporal train/valid split
.venv/bin/python src/baseline.py  # train baseline, report metrics
```

End-to-end runtime: **under 1 minute** on CPU.

## Baseline results

| metric | validation |
|---|---|
| AUC | **0.7439** |
| log-loss | **0.5882** |

Train: 800,168 rows (2000-04-25 → 2000-12-02) · Valid: 200,041 rows
(2000-12-02 → 2003-02-28). Label: `liked = (rating >= 4)`, positive rate ≈ 0.57.
25 features: train-period-only user/movie aggregates, genre one-hots, user
demographics. LightGBM, ~40s training.

This is the number the agent has to beat.

## Layout

```
data/ml-1m/        # MovieLens-1M .dat files (ratings/users/movies)
src/data.py        # loader: parses '::'-separated latin-1 files, joins, labels
src/split.py       # STRICT temporal split (guarantee: max(train.ts) < min(valid.ts))
src/baseline.py    # baseline model; aggregates computed on train period ONLY
src/features.py    # FeatureRegistry stub: name + compute() + point_in_time declaration
results/           # baseline_metrics.json
```

## Key design decisions (for the demo)

- **Temporal discipline is load-bearing.** The split guarantees no validation
  row predates any train row. The Day 5–6 leakage detector relies on this: any
  feature that peeks past a row's prediction time gets rejected on camera.
- **Train-period-only aggregates.** `user_mean_rating`, `movie_mean_rating`,
  etc. are computed from train rows alone and mapped onto validation rows
  (unseen ids → global fallback). This is the pattern every agent-invented
  feature must follow.
- **`point_in_time` declarations.** Every registered feature declares which
  timestamp column bounds the data it may read (`src/features.py`). The
  leakage checker will audit these declarations.
- **Planted traps (coming Day 5–6):** a leaky feature (movie mean rating over
  *all* data including the future — looks great, must be rejected) and an
  expensive-to-serve feature, so the "rejection" demo moments are guaranteed.
