# Demo script — Feature Scientist live run

Rehearsed command (deterministic, scripted backend — no keys, no network):

```
python3 src/brief.py briefs/movielens.yaml --backend scripted
```

Companion visual: `python3 src/arena.py` → http://localhost:8765
(auto-refreshes every 3s; watch the leaderboard and round timeline live).

**Total wall-clock: ~8.5 minutes** (measured 8m31s, two identical runs).
Time is dominated by model training: baseline 123s + 2 ablations ~2.5–3 min
each. Everything "agentic" (propose/critic/audit/screen) runs in seconds.
For a tighter slot: `--model lgbm` trains in seconds, or `--max-rounds 1`.

Run with `python3 -u` for true per-line timestamps (stdout is block-buffered
otherwise, which collapses the timeline).

---

## The story in one line

You hand the agent a YAML brief — a problem in plain English plus schema
text — and it profiles the real tables, finds where the model is wrong,
proposes features, gets rejected twice (once for cost, once for *leakage*),
revises, proves two lifts, then stops itself when there's nothing left to try.

## Beats

### Beat 1 — The brief is the front door (0:00)
**Show:** the `briefs/movielens.yaml` file, then the run header.
**Narrate:** "The input to the agent is a prompt: an ML problem in plain
language, plus datasets as free-form schema text — paste a Hive dump here.
The load-bearing line is the temporal contract: *features may only use data
with timestamp ≤ T*. The agent profiles the actual tables — text describes,
data verifies — 4 tables, row counts, null rates, join coverage: ratings ⨝
users 100%, ratings ⨝ movies 100%, IMDb 95.4%."

### Beat 2 — The calibration battery finds signals for free (~2:45)
**Show:** the 28 `battery/...` lines scrolling past.
**Narrate:** "Before training anything else, the agent screens cheap
candidate signals for miscalibration — predicted vs. actual like-rate per
bucket. 28 significant gaps, zero retraining. The biggest: heavy users
over-predicted by 7.4 points, stale movies over-predicted by 4.7, rare users
under-predicted by 7.2. This is the agent figuring out *what information the
model is missing* — at a scale it can actually reason about, in percentage
points, not AUC dust."
**Baseline to beat:** AUC 0.7403 (two-tower, trained 0:24–2:31).

### Beat 3 — The critic kills the expensive feature; the agent revises (~2:50)
**Show:** terminal lines:
```
critic: REJECT — Serving cost: history_scan = 25 latency units vs the 8.0 per-feature budget...
revised after critic -> 'movie_age_days_llm'
re-critic: PASS
leakage audit: [PASS] ... temporal claims check out
serving cost: [PASS] ... 0.5 units
calibration screen: ... signal
ablation: AUC=0.7421 (delta +0.0018)
verdict: ACCEPTED
```
**Narrate:** "First hypothesis: user rating entropy. The skeptic kills it —
25 latency units against an 8-unit budget; it can never ship. The agent
revises to movie age — row-local, half a unit — passes the leakage audit,
shows a real calibration signal, and the ablation proves +0.0018. Propose,
criticize, revise, prove."

### Beat 4 — The leakage trap (the money moment) (~5:23)
**Show:** terminal lines:
```
leakage audit: [FAIL] imdb_quality_prior_llm: code reads ctx['imdb_map_avg']:
IMDb all-time averageRating (votes through Oct 2026, incl. post-prediction)
-- TEMPORAL LEAKAGE (code inspection); claims temporal_scope='pit_correct'
but the code never references prediction time ('timestamp') or as-of logic
-- declaration not credible (code inspection)
revised after audit -> 'movie_release_year_llm'
re-audit: [PASS]
...
verdict: ACCEPTED (AUC lift +0.0021 >= 0.0005)
```
**Narrate:** "Second hypothesis looks great — IMDb's average rating as a
quality prior for cold movies. The critic passes it with caution. Then the
deterministic audit reads the actual code: it's the *all-time* IMDb
aggregate, votes through 2026, predicting 2001 — temporal leakage, and the
feature's own PIT declaration is not credible because the code never touches
prediction time. The agent revises to the PIT-clean release year and proves
+0.0021. This is the beat that matters: the agent caught its own best idea
cheating, and fixed it."
**Arena:** the leaderboard now shows two accepted features; the ledger entry
carries the full reasoning trail, critiques, and per-attempt verdicts.

### Beat 5 — The agent stops itself (~8:31)
**Show:** terminal lines:
```
ROUND 2
   ...
   verdict: DUPLICATE (already accepted in an earlier round; not re-ablated)
   verdict: DUPLICATE (already accepted in an earlier round; not re-ablated)

STOPPING after round 2: all proposed features already accepted in earlier rounds.

round feature                verdict   delta_auc reason
1     movie_age_days_llm     accepted  +0.0018   AUC lift +0.0018 >= 0.0005
1     movie_release_year_llm accepted  +0.0021   AUC lift +0.0021 >= 0.0005
2     movie_release_year_llm duplicate n/a       already accepted in an earlier round; not re-ablated
2     movie_release_year_llm duplicate n/a       already accepted in an earlier round; not re-ablated

accepted 2/4
```
**Narrate:** "Round 2: both proposals converge onto already-accepted
features. No re-ablation, no wasted GPUs — the stopping rule fires. The
agent knows when it's done. Final: 2 accepted, 2 duplicates, 0 leakage
escapes, total serving cost 1.5 of 15 units."

---

## Determinism

Two full runs produced byte-identical stories: same features, same deltas
(+0.0018, +0.0021), same rejections, same stopping point. Same seed → same
demo. Safe to record.

## Timing breakdown (for video planning)

| stage | wall-clock |
|---|---|
| brief header + table profiling | ~5 s |
| data load + temporal split + featurize | ~20 s |
| baseline training (4 epochs, CPU) | ~2 min |
| error slices + ctx + tower cache | ~15 s |
| calibration battery (28 screens) | ~5 s |
| round 1 (2 × propose→critic→audit→cost→screen→ablate) | ~5.5 min |
| round 2 (duplicates, no training) + summary | seconds |
| **total** | **~8.5 min** |

Cuts, if the slot is tight: `--model lgbm` (baseline trains in seconds,
ablations ~10 s each → run ≈ 1 min); `--max-rounds 1` (skips the
stopping-rule beat — not recommended, it's the closer).
