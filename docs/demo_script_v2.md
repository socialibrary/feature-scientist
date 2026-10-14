# Demo script v2 — Feature Scientist (3-minute hackathon video)

## The story in one line

You hand the agent a YAML brief — a Kaggle problem in plain English — and it
profiles the data, proposes features, gets some rejected, proves 3 real lifts,
and stops. LLM suggests, deterministic implements, gates decide.

## Setup (before recording)

```bash
cd ~/workspace/feature-scientist
# Data already downloaded: data/kaggle/spaceship-titanic/train.csv
```

## Beats

### Beat 1 — The brief (0:00-0:20)
**Show:** `briefs/spaceship_titanic.yaml`
**Narrate:** "The input is a YAML brief: a problem in plain English, schema
as text, and the metric. The agent profiles the actual tables — 8,693 rows,
13 features. Text describes, data verifies."

### Beat 2 — The run (0:20-1:30)
**Show:** terminal running:
```bash
python3 src/brief.py briefs/spaceship_titanic.yaml --backend st-domain --max-rounds 3
```
**Narrate:** "The agent proposes 6 features. Watch the verdicts: 3 accepted,
3 rejected. The rejections matter — total_spend and group_size looked good
but didn't beat the baseline. The agent killed them with receipts."

**Key terminal output to highlight:**
```
verdict: ACCEPTED (AUC lift +0.0098 >= 0.0005)  # cabin_deck
verdict: ACCEPTED (AUC lift +0.0068 >= 0.0005)  # cabin_num  
verdict: ACCEPTED (AUC lift +0.0034 >= 0.0005)  # cabin_side
verdict: REJECTED (no lift (delta -0.0023 < 0.0005))  # total_spend
```

### Beat 3 — The architecture (1:30-2:30)
**Narrate:** "The architecture: Nemotron suggests ideas — it proposed 'total
spending' and 'HomePlanet equals Destination'. The deterministic layer writes
correct code. The gates — critic, leakage audit, ablation — decide what ships.
LLM for creativity, deterministic for reliability."

**Show:** The 3 accepted features with their AUC lifts.

### Beat 4 — The close (2:30-3:00)
**Narrate:** "Baseline 0.8644. Three features accepted. Zero leakage escapes.
The agent knows when to stop. That's Feature Scientist: it finds real signal
and refuses to fool itself."

## Timing

Total run: ~11 seconds. Perfect for live demo.

## Determinism

The st-domain backend is deterministic — same features, same lifts, every run.
Safe to record.
