# Feature Scientist — Harness + Science Loop + Judgment Layer + Arena UI (Days 1–8)

An autonomous "AI Feature Scientist" agent for the Meta global AI hackathon:
given a model and an objective, it figures out **what information the model is
missing**, invents new features, proves they help, and rejects the ones that
leak or cost too much to serve.

This repo holds the **Days 1–2 harness** (data, temporal split, baseline) plus
the **Days 3–4 science loop**: error-slice analysis, hypothesis engines, the
leakage audit, ablations, and an append-only experiment ledger. The judgment
layer hardening (Days 5–6), arena UI (Days 7–8), and demo rehearsal (Days 9–10)
build on top.

Dataset: **MovieLens-1M** (public, https://grouplens.org/datasets/movielens/1m/)
— 1,000,209 ratings, 6,040 users, 3,706 movies — **plus IMDb enrichment**
(`data/imdb/movie_enrichment.csv`): 86% title join, directors, runtime, and
IMDb ratings/votes. Note: `imdb_averageRating`/`numVotes` are **all-time**
aggregates — raw use is temporal leakage, and the audit catches it. No
Meta-internal anything.

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
.venv/bin/python src/baseline.py  # LightGBM baseline (fallback path)
.venv/bin/python src/two_tower.py # PyTorch two-tower baseline (primary)
.venv/bin/python src/error_analysis.py            # error slices (two-tower)
.venv/bin/python src/error_analysis.py --model lgbm  # error slices (LightGBM)
.venv/bin/python src/run_scientist.py --rounds 3               # science loop (two-tower)
.venv/bin/python src/run_scientist.py --rounds 3 --model lgbm  # science loop (LightGBM)
```

End-to-end harness runtime: **under 1 minute** on CPU. A full 3-round science
loop takes ~6 minutes (one ~40s ablation per surviving hypothesis).

## Baseline results

Primary model: **PyTorch two-tower** (`src/two_tower.py`) — user tower
(5 user-side features → MLP → 48-dim embedding), movie tower (20 movie-side
features → MLP → 48-dim embedding), dot product + bias, `BCEWithLogitsLoss`.
Trains on 800k rows in ~80s on CPU (4 epochs, batch 65536).

| metric | validation |
|---|---|
| AUC | **0.7410** |
| log-loss | **0.5919** |

LightGBM (`src/baseline.py`) remains as a `--model lgbm` fallback
(AUC 0.7439). The science loop defaults to the two-tower; candidate
features are routed to the appropriate tower's input
(`provenance["tower"]`: `user` | `movie` | `wide`) and ablated against
the two-tower baseline.

Train: 800,168 rows (2000-04-25 → 2000-12-02) · Valid: 200,041 rows
(2000-12-02 → 2003-02-28). Label: `liked = (rating >= 4)`, positive rate ≈ 0.57.
25 features: train-period-only user/movie aggregates, genre one-hots, user
demographics. LightGBM, ~40s training.

This is the number the agent has to beat.

## Layout

```
data/ml-1m/        # MovieLens-1M .dat files (ratings/users/movies)
data/imdb/         # movie_enrichment.csv (86% IMDb title join; raw dumps git-ignored)
src/data.py        # loader: parses '::'-separated latin-1 files, joins, labels
src/split.py       # STRICT temporal split (guarantee: max(train.ts) < min(valid.ts))
src/baseline.py    # baseline model; aggregates computed on train period ONLY
src/features.py    # FeatureRegistry: name + compute() + point_in_time + provenance
src/error_analysis.py  # validation error slices (activity/popularity/genre/time)
src/scientist.py   # DATA_CATALOG, Hypothesis, RuleBasedEngine, LLMEngine stub,
                   #   executable feature builders, build_ctx()
src/ablation.py    # baseline+feature training, AUC/log-loss deltas
src/leakage.py     # first-pass audit: provenance vs temporal discipline
src/ledger.py      # append-only experiment ledger (results/experiments.jsonl)
src/run_scientist.py  # CLI orchestrator: --rounds N [--engine rule|llm]
src/imdb_join.py   # IMDb bulk-download + title/year join (one-off scaffold)
results/           # baseline_metrics.json, error_slices.json, experiments.jsonl
docs/imdb_join.md  # IMDb join method, hit rate, temporal caveat
```

## The science loop (Days 3–4)

```
error slices -> propose -> build -> leakage audit -> serving-cost check
    -> ablate -> ledger
```

1. **Error slices** (`error_analysis.py`): where does the baseline fail?
   Cold movies, light users, specific genres/months — written to
   `results/error_slices.json`.
2. **Propose** (`scientist.py`): `RuleBasedEngine` (default, no API keys)
   turns the worst slices into falsifiable hypotheses, reasoning over the
   v1 **data catalog** (ml-1m + IMDb). `LLMEngine` is a stub marking the
   Meta Model API seam (`META_MODEL_API_URL` / `META_MODEL_API_KEY`).
3. **Build**: each hypothesis becomes an executable `Feature`,
   auto-registered with a `point_in_time` declaration, a `provenance`
   record (`temporal_scope`: `train_only` | `pit_correct` | `static` |
   `all_time`), and a `serving` pattern (`row_local` | `lookup` |
   `history_scan` | `external`).
4. **Leakage audit** (`leakage.py`, hardened Day 5–6): declaration checks
   PLUS static inspection of the feature's actual code (AST + closure
   variables). Catches features that *lie* in their provenance — e.g. a
   "point-in-time" claim whose code closes over a future-built table.
5. **Serving-cost check** (`serving_cost.py`, new Day 5–6): every feature
   gets a latency estimate in ms-equivalents from its serving pattern;
   per-feature budget 8.0 units, cumulative budget 15.0. Expensive
   features are rejected before any training time is spent.
6. **Ablate** (`ablation.py`): survivors are routed to the right tower
   (`user`/`movie`/`wide`) and the two-tower retrains on the temporal split
   (`--model lgbm` keeps the original flat-matrix path); accept iff AUC
   lift >= 0.0005.
7. **Ledger** (`ledger.py`): every hypothesis lands in
   `results/experiments.jsonl` with its code, leakage audit, serving-cost
   verdict, metrics, and final verdict.

## The judgment layer (Days 5–6)

The loop doesn't just measure lift — it judges *how* a feature earns it.

- **Verify, don't trust** (`leakage.py`): the audit parses each feature's
  `compute()` source (AST) and inspects its closure variables against a
  registry of known-dangerous sources (`movie_mean_all`,
  `imdb_averageRating`/`imdb_numVotes`, …). A feature is rejected if its
  code touches a dangerous source or reads rows newer than prediction
  time — regardless of what its declaration claims. Demo trap:
  `sneaky_movie_mean_pit` in `src/traps.py` declares `pit_correct` while
  closing over the future-built table; the audit catches the lie twice
  (closure identity + missing temporal guard).
- **Cost verdict** (`serving_cost.py`): declared serving patterns map to
  latency units; the verdict pipeline is leakage → cost → ablation. Demo
  trap: a full-user-history scan with genuine signal but 25 units of cost
  dies at the budget gate (`src/traps.py`).

## Key design decisions (for the demo)

- **Temporal discipline is load-bearing.** The split guarantees no validation
  row predates any train row. The leakage detector relies on this: any
  feature that peeks past a row's prediction time gets rejected on camera —
  and since Day 5–6 the audit verifies the feature's *code*, not just its
  declaration.
- **Train-period-only aggregates.** `user_mean_rating`, `movie_mean_rating`,
  etc. are computed from train rows alone and mapped onto validation rows
  (unseen ids → global fallback). This is the pattern every agent-invented
  feature must follow.
- **`point_in_time` declarations.** Every registered feature declares which
  timestamp column bounds the data it may read (`src/features.py`). The
  leakage checker audits these declarations *and* inspects the
  implementation for violations.
- **Serving budget.** Every feature declares how it would be served
  (`row_local`/`lookup`/`history_scan`/`external`); the pipeline rejects
  unaffordable features before training (`src/serving_cost.py`).
- **Planted traps (demo fixtures in `src/traps.py`):** a *lying* feature
  (`sneaky_movie_mean_pit`: claims point-in-time-correct, code reads the
  future-built table) and a *budget-buster* (full-user-history scan with
  real signal but 25 units of serving cost), so the "rejection" demo
  moments are guaranteed.

## Arena UI (Days 7–8)

A demo-ready live dashboard — **stdlib only, zero new dependencies**
(`src/arena.py` + `src/arena_ui.html`).

```bash
python3 src/arena.py            # stdlib http.server, default http://localhost:8765
```

- `/` — dark-themed dashboard: header (baseline AUC, best ΔAUC, live status),
  accepted-features leaderboard, and a round-by-round experiment timeline.
  Rejections get visual emphasis — the catches (liar, budget-buster, raw IMDb
  rating) pop with 🚨/💸 chips and an expandable audit trail.
- `/api/ledger` — ledger entries with a computed `category`
  (`accepted` | `rejected:leakage` | `rejected:cost` | `rejected:no-lift`).
- `/api/summary` — baseline AUC, best delta, verdict counts, rounds.
- The page polls every 3s and re-renders on change; a missing/empty ledger
  shows a "waiting for first run" state instead of breaking.

**Two-window demo recipe:**

```bash
# terminal 1 — the dashboard
python3 src/arena.py            # open http://localhost:8765
# terminal 2 — the agent (needs the venv for torch/sklearn)
.venv/bin/python src/run_scientist.py --rounds 4
```

Watch hypotheses appear live: acceptances climb the leaderboard while the
leakage liar and the budget-buster get caught on camera.

## Agentic loop (LLM proposer + critic + retry-on-rejection)

The rule engine is scripted; the **agentic** engine puts a real LLM in the
`propose` seat (`--engine llm`). The LLM's output is free-form reasoning
plus a structured spec (name, text, sources, PIT declaration, tower,
serving pattern, and `code` = a `compute(df, ctx)` body). The reasoning is
recorded verbatim in the ledger; the machinery judges the code.

**Two roles, one loop** (`src/llm_backend.py`, `src/critic.py`,
`src/codegen.py`, `src/agentic.py`):

1. **Proposer** — reads the worst error slices + data catalog (+ API
   reference: ctx tables, PIT rules, serving budgets, tower routing) and
   returns reasoning + hypotheses.
2. **Critic** (skeptic) — tries to kill each hypothesis *before* it costs
   anything: PIT violations, serving-cost overruns, circular signals. On
   reject → **one revision round**, then re-critic once.
3. **Materialize** — the code is AST safety-checked (no imports, no I/O, no
   `eval`/`exec`/`open`), written to a real file under
   `results/gen_features/` so the hardened leakage audit can
   `inspect.getsource()` it, then `exec`'d in a restricted namespace
   (pd, np, safe builtins only). Fail-closed at every step.
4. **Deterministic gates** — leakage audit → serving-cost check → ablation,
   in that order. A rejection at the audit *or* cost gate feeds the exact
   rejection reason back to the LLM for **one revision attempt**
   (retry-on-rejection); the revision is re-audited before proceeding.
5. **Ledger** — every entry carries an `agentic` block: free-form reasoning,
   each critique, each revision (stage, feedback, from→to, code), and every
   attempt's verdict. That's the reasoning trail.

**Backends** (`--backend`, default `http`):

```bash
# Offline test double: deterministic canned arcs, no network, no keys.
.venv/bin/python src/run_scientist.py --rounds 1 --engine llm --backend scripted

# Real model: OpenAI-compatible chat completions (stdlib urllib, no new deps).
export FS_LLM_BASE_URL="https://api.openai.com/v1"   # or the Meta Model API endpoint
export FS_LLM_API_KEY="..."
export FS_LLM_MODEL="gpt-4o"                         # or llama-3.3-70b, ...
.venv/bin/python src/run_scientist.py --rounds 2 --engine llm --backend http
```

No key → clear error + automatic fallback to the rule engine for that
round. The scripted backend stages two arcs end-to-end: (a) propose →
critic-reject (serving cost) → revise → accept; (b) critic-pass →
audit-reject (all-time IMDb aggregate = temporal leakage) → revise with a
PIT-clean feature → accept. Both land in the ledger with the full trail.

Design note: the critic is *advisory* — the deterministic audit is the hard
gate. A good critic catch saves an audit cycle; a missed catch is still
caught downstream. Defense in depth, and the ledger shows which layer
caught what.

## Data catalog: offline crawl + searchable registry

The agent reasons over a **big array of candidate features**, not a hand-coded
handful. Discovery works in two layers:

**1. Offline crawl (outside the agent).** `src/crawl_wikidata.py` queries the
Wikidata Query Service (IMDb id = P345) in polite batches for directors,
cast, awards (P166), box office (P2142), budget (P2130), release date and
original language — entity refs kept as QIDs, labels resolved for
directors/awards/languages. Raw responses are cached per chunk
(`data/wikidata/raw/`), the script is resume-friendly, and coverage is
reported honestly per field.

```bash
python3 src/crawl_wikidata.py --chunk 100 --delay 65   # ~40 min for 3.9k movies
```

**2. Searchable registry.** `src/catalog/` holds `registry.json` — every table
(ml1m_ratings, ml1m_users, ml1m_movies, imdb_enrichment, wikidata_movies) with
*per-column* metadata: plain-English description, dtype, join key,
`temporal_scope` (`time_invariant` | `point_in_time_safe` | `all_time`),
serving class, and honestly-computed coverage %. A SQLite FTS5 index
(`catalog.db`) powers the discovery API:

```python
from catalog import search_datasets
search_datasets("director awards")
# -> wikidata_movies.awards  scope=all_time (fail-closed: wins may postdate
#    prediction time) ...
```

`DATA_CATALOG` in `src/scientist.py` now loads from the registry (same dict
shape as before, plus `column_meta`), and engines can call `search_datasets`
to discover data instead of reading a hardcoded list. Rebuild anytime with:

```bash
cd src && ../.venv/bin/python -m catalog.build_catalog
```

**Temporal honesty is load-bearing metadata.** Columns like
`imdb_averageRating` and `wikidata_movies.awards` are marked `all_time`:
fail-closed, raw use is temporal leakage. The LLM reasons over these
annotations when it proposes features — the same annotations the leakage
audit enforces.

## Calibration screening (screen 50, ablate 5)

`src/calibration.py` adds a screening stage between error analysis and
ablation: bucket validation rows by a *candidate* feature (not in the
model), compare the baseline's mean predicted P(liked) against the actual
like rate per bucket, and flag significant gaps. Deterministic and
inference-only — no retraining, so no seed noise, which is what makes it
trustworthy at 0.1%-gain scale. Each feature gets a one-line plain-English
summary for the LLM prompt, e.g. "newest decile under-predicted by 2.1pp".

Sparse features are handled explicitly (`screen_sparse_id`): high-cardinality
IDs are never bucketed raw — the screen derives popularity (train-period
frequency, the cold-start diagnostic) and recency proxies instead. Rare
binary flags with underpowered positive buckets report
`insufficient_evidence`, never "no signal".

```bash
cd src && ../.venv/bin/python demo_calibration.py
# movie_age_days -> signal (same finding as its +0.0005 AUC ablation, zero retraining)
# random_noise   -> no_signal
# rare_flag      -> insufficient_evidence (positive bucket n=21)
```

Integration point (new stage: error analysis → calibration screen →
ablation): `screen_candidates(y_valid, proba, {name: series})`,
`format_table(results)`; details in `docs/calibration_integration.md`.
