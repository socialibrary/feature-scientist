"""Calibration screening stage for the Feature Scientist.

Idea: before spending ablation budget (retraining per candidate), screen
candidate features with a calibration study. Take the ALREADY-TRAINED
baseline's validation predictions, bucket validation rows by the candidate
feature (which is NOT in the model), and compare mean predicted P(liked)
against the actual like rate per bucket. A significant gap means the
feature carries signal the model lacks -> worth ablating.

Why this works at 0.1%-gain scale: it is deterministic (no retraining, so
no seed noise) and cheap (inference-only). It also produces LLM-legible
output: "newest decile under-predicted by 2.1pp" is a directional signal
verbal reasoning CAN work with, unlike a +0.0005 AUC delta.

Two-tier significance: each bucket carries a 2-SE flag (gap > 2*SE), but
the feature-level "signal" verdict requires clearing a Bonferroni-adjusted
threshold over buckets -- with B buckets, ~B*0.05 would trip the raw 2-SE
flag by pure chance. The screen prioritizes candidates; the ablation is
what proves them.

Sparse features need special handling (see screen_sparse_id): never bucket
by raw ID value (meaningless for high-cardinality IDs). Instead the screen
auto-derives dense proxies -- train-period frequency (popularity/cold-start
diagnostic) and recency -- and screens those.

Pure functions: no training happens in this module. It consumes
(y_true, y_pred) from the existing baseline.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

# Buckets smaller than this cannot support a significance claim.
DEFAULT_MIN_BUCKET_N = 30
# Per-bucket flag: |gap| exceeds this many binomial standard errors.
Z_SIGMA = 2.0
# Feature-level verdict uses a Bonferroni correction over buckets: with B
# buckets, ~B*0.05 would trip the 2-SE flag by chance, so "signal" requires
# clearing the family-wise threshold. The screen prioritizes; the ablation
# proves.
FAMILYWISE_ALPHA = 0.05


def _z_adj(n_buckets: int) -> float:
    return float(norm.ppf(1.0 - FAMILYWISE_ALPHA / (2.0 * max(n_buckets, 1))))


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def _ordinal_name(i: int, n: int) -> str:
    if n == 10:
        names = ["lowest decile", "2nd decile", "3rd decile", "4th decile",
                 "5th decile", "6th decile", "7th decile", "8th decile",
                 "9th decile", "highest decile"]
        return names[i]
    if n == 4:
        return ["lowest quartile", "2nd quartile", "3rd quartile",
                "highest quartile"][i]
    return f"bucket {i + 1}/{n}"


def _bucketize(values: pd.Series, n_buckets: int = 10) -> pd.Series:
    """Assign each row to a human-labeled bucket.

    Numeric -> quantile buckets; boolean/two-valued -> as-is; categorical ->
    as-is (top n_buckets-1 by count + "other"); NaN -> its own "missing"
    bucket. Returns a Series of string labels aligned to values.index.
    """
    v = values.reset_index(drop=True)
    is_missing = v.isna()
    labels = pd.Series(index=v.index, dtype=object)

    labels[is_missing] = "missing"
    vv = v[~is_missing]
    if len(vv) == 0:
        return labels

    nunique = vv.nunique(dropna=True)
    if pd.api.types.is_bool_dtype(vv.dtype) or nunique <= 2:
        # Binary / two-valued: screen {0,1} as-is. The positive bucket's
        # power is checked downstream (underpowered -> insufficient_evidence,
        # never "no signal").
        labels[~is_missing] = vv.astype(str).values
        return labels

    if pd.api.types.is_numeric_dtype(vv.dtype) and nunique > n_buckets:
        cats = pd.qcut(vv, q=n_buckets, duplicates="drop")
        nq = len(cats.cat.categories)
        bounds = [(iv.left, iv.right) for iv in cats.cat.categories]
        codes = cats.cat.codes.to_numpy()
        for i, (l, r) in enumerate(bounds):
            labels[(~is_missing) & (codes == i)] = \
                f"[{l:.3g}, {r:.3g}) ({_ordinal_name(i, nq)})"
        return labels

    # Categorical (or low-cardinality numeric): as-is, collapsing the tail.
    counts = vv.value_counts()
    if len(counts) > n_buckets:
        keep = set(counts.index[: n_buckets - 1])
        lab = vv.apply(lambda x: str(x) if x in keep else "other")
    else:
        lab = vv.astype(str)
    labels[~is_missing] = lab.values
    return labels


# ---------------------------------------------------------------------------
# Core screen
# ---------------------------------------------------------------------------

def calibration_screen(
    y_true,
    y_pred,
    feature_values,
    feature_name: str,
    n_buckets: int = 10,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
    key_bucket: str | None = None,
) -> dict:
    """Calibration study of one candidate feature against baseline predictions.

    y_true / y_pred / feature_values must be aligned (same order).
    key_bucket: for sparse binary flags, pass the label of the informative
        bucket (e.g. "1"); if it is underpowered and nothing is significant,
        the verdict is "insufficient_evidence" rather than "no_signal".

    Returns a dict with verdict in {"signal", "no_signal",
    "insufficient_evidence"}, per-bucket stats, and a plain-English summary.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    feat = pd.Series(np.asarray(feature_values)).reset_index(drop=True)
    assert len(y_true) == len(y_pred) == len(feat), "misaligned inputs"

    labels = _bucketize(feat, n_buckets)
    order = labels.unique().tolist()
    # Put "missing"/"other" last for readability.
    order.sort(key=lambda l: (l in ("missing", "other"), l))
    zcrit = _z_adj(len(order))

    buckets = []
    for lab in order:
        m = (labels == lab).to_numpy()
        n = int(m.sum())
        if n == 0:
            continue
        mp = float(y_pred[m].mean())
        ar = float(y_true[m].mean())
        gap_pp = (ar - mp) * 100.0
        se = math.sqrt(ar * (1.0 - ar) / n) if n > 0 else float("nan")
        se_pp = se * 100.0
        powered = n >= min_bucket_n
        significant = powered and abs(gap_pp) > Z_SIGMA * se_pp
        significant_adj = powered and abs(gap_pp) > zcrit * se_pp
        buckets.append({
            "label": lab, "n": n,
            "mean_pred": round(mp, 4), "actual_rate": round(ar, 4),
            "gap_pp": round(gap_pp, 2), "se_pp": round(se_pp, 2),
            "significant": bool(significant),
            "significant_adj": bool(significant_adj),
            "powered": bool(powered),
        })

    sig = [b for b in buckets if b["significant_adj"]]
    max_sig_gap = max((abs(b["gap_pp"]) for b in sig), default=None)
    blind_spots = [b["label"] for b in buckets if not b["powered"]]

    if sig:
        verdict = "signal"
    elif key_bucket is not None and any(
            b["label"] == key_bucket and not b["powered"] for b in buckets):
        verdict = "insufficient_evidence"
    elif all(not b["powered"] for b in buckets):
        verdict = "insufficient_evidence"
    else:
        verdict = "no_signal"

    summary = _summarize(feature_name, buckets, sig, blind_spots, verdict,
                         key_bucket)
    return {
        "feature": feature_name,
        "verdict": verdict,
        "max_significant_gap_pp": max_sig_gap,
        "n_buckets": len(buckets),
        "buckets": buckets,
        "blind_spots": blind_spots,
        "summary": summary,
    }


def _summarize(feature_name, buckets, sig, blind_spots, verdict,
               key_bucket) -> str:
    if verdict == "signal":
        parts = []
        for b in sorted(sig, key=lambda b: -abs(b["gap_pp"]))[:3]:
            direction = "under-predicted" if b["gap_pp"] > 0 \
                else "over-predicted"
            parts.append(f"{b['label']} {direction} by "
                         f"{abs(b['gap_pp']):.1f}pp (significant, "
                         f"n={b['n']})")
        return f"{feature_name}: " + "; ".join(parts)
    if verdict == "insufficient_evidence":
        if key_bucket is not None:
            kb = next((b for b in buckets if b["label"] == key_bucket), None)
            n = kb["n"] if kb else 0
            return (f"{feature_name}: informative bucket '{key_bucket}' too "
                    f"small (n={n}) — insufficient evidence, not no signal")
        return (f"{feature_name}: no bucket large enough to judge "
                f"(all n < powered threshold) — insufficient evidence")
    s = (f"{feature_name}: no significant miscalibration across "
         f"{len(buckets)} buckets")
    if blind_spots:
        shown = ", ".join(f"{l}" for l in blind_spots[:3])
        s += f"; blind spots (underpowered): {shown}"
    return s


# ---------------------------------------------------------------------------
# Multi-candidate screening: "screen 50, ablate 5"
# ---------------------------------------------------------------------------

_VERDICT_RANK = {"signal": 0, "insufficient_evidence": 1, "no_signal": 2}


def screen_candidates(y_true, y_pred, candidates: dict,
                      n_buckets: int = 10,
                      min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
                      key_buckets: dict | None = None) -> list[dict]:
    """Run calibration_screen over many candidate features; rank by evidence.

    candidates: {feature_name: array-like of values}, aligned with y_*.
    key_buckets: optional {feature_name: key_bucket_label} for sparse flags.
    Returns results sorted signal-first, then by max significant gap.
    """
    key_buckets = key_buckets or {}
    results = [
        calibration_screen(y_true, y_pred, vals, name, n_buckets,
                           min_bucket_n, key_buckets.get(name))
        for name, vals in candidates.items()
    ]
    results.sort(key=lambda r: (_VERDICT_RANK[r["verdict"]],
                                -(r["max_significant_gap_pp"] or 0.0)))
    return results


def format_table(results: list[dict]) -> str:
    """Render screening results as a plain-text ranked table."""
    lines = [f"{'feature':<28}{'verdict':<22}{'max_sig_gap_pp':>14}   summary",
             "-" * 110]
    for r in results:
        gap = (f"{r['max_significant_gap_pp']:.2f}"
               if r["max_significant_gap_pp"] is not None else "-")
        lines.append(f"{r['feature']:<28}{r['verdict']:<22}{gap:>14}   "
                     f"{r['summary']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sparse features: never bucket by raw ID value.
# ---------------------------------------------------------------------------

POPULARITY_BINS = [0, 1, 5, 20, 100, float("inf")]
POPULARITY_LABELS = ["unseen (0)", "rare (1-4)", "5-19", "20-99", "100+"]


def screen_sparse_id(
    y_true,
    y_pred,
    valid_ids,
    train_id_counts: pd.Series,
    id_name: str,
    train_id_last_ts: pd.Series | None = None,
    valid_ts=None,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
) -> dict:
    """Calibration screen for high-cardinality IDs (movie/user/director).

    Bucketing by raw ID value is meaningless, so the screen auto-derives
    dense proxies and screens those instead:
      * popularity: train-period frequency bucket per ID
        ("unseen (0)" / "rare (1-4)" / ...). This IS the cold-start
        diagnostic: a gap on the rare segment means the model mishandles
        sparse entities.
      * recency: days since the ID was last seen in train (needs
        train_id_last_ts + valid_ts); skipped with a note if unavailable.

    Returns {"popularity": <screen dict>, "recency": <screen dict> | None,
             "notes": [...]}.
    """
    valid_ids = pd.Series(np.asarray(valid_ids)).reset_index(drop=True)
    counts = valid_ids.map(train_id_counts).fillna(0).astype(int)
    pop_labels = pd.cut(counts, bins=POPULARITY_BINS,
                        labels=POPULARITY_LABELS, right=False).astype(str)
    popularity = calibration_screen(
        y_true, y_pred, pop_labels, f"{id_name}_train_popularity",
        n_buckets=len(POPULARITY_LABELS), min_bucket_n=min_bucket_n)

    recency, notes = None, []
    if train_id_last_ts is not None and valid_ts is not None:
        last = valid_ids.map(train_id_last_ts)
        vts = pd.Series(np.asarray(valid_ts)).reset_index(drop=True)
        days = (vts - last) / 86400.0
        rec_labels = pd.Series(index=days.index, dtype=object)
        rec_labels[last.isna()] = "unseen"
        seen = ~last.isna()
        if seen.sum() > 0:
            q = pd.qcut(days[seen], q=4, duplicates="drop")
            rec_labels[seen] = ("seen: " + q.astype(str)).values
        recency = calibration_screen(
            y_true, y_pred, rec_labels, f"{id_name}_recency_days",
            n_buckets=5, min_bucket_n=min_bucket_n)
    else:
        notes.append("recency proxy skipped: train_id_last_ts/valid_ts not "
                     "provided")
    return {"popularity": popularity, "recency": recency, "notes": notes}
