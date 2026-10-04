"""Demo traps for the Day 5-6 judgment layer.

!!! DEMO FIXTURES -- these are deliberately planted, not real discoveries. !!!

TRAP 1 -- the liar (build_sneaky_movie_mean):
    Declares temporal_scope='pit_correct' and hides its ctx table
    (ctx_keys=[]), but the code closes over ctx['movie_mean_all'], a table
    built from train+validation rows. The v1 declaration-only audit would
    PASS it; the hardened audit must catch the lie via closure-object
    identity (B2) and the missing temporal guard (B4).

TRAP 2 -- the budget-buster (build_user_history_entropy):
    A plausible signal (Shannon entropy of the user's full train-period
    rating distribution) with honest PIT-correct provenance -- but declared
    serving='history_scan' (25 latency units). It passes the leakage audit
    and dies at the serving-cost gate: 25u > 8u per-feature budget. The
    pipeline rejects it before spending any training time -- which is the
    point: unaffordable features never reach ablation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import Feature

_LN5 = float(np.log(5))


def build_sneaky_movie_mean(ctx) -> Feature:
    """TRAP 1 (demo): lies in its declaration.

    Claims pit_correct with no ctx_keys, but `s` IS ctx['movie_mean_all']
    (train+validation means). Hardened audit must catch it.
    """
    s = ctx["movie_mean_all"]  # <-- the laundered future-built table
    gm = ctx["global_mean"]

    def compute(df, c):
        return pd.Series(df["movie_id"].map(s).fillna(gm), index=df.index)

    return Feature(
        name="sneaky_movie_mean_pit",
        compute=compute,
        point_in_time="timestamp",
        description="TRAP (demo): CLAIMS as-of movie mean rating, but the "
                    "code maps a lookup table built over train+validation. "
                    "The declaration is a lie; the audit must catch it.",
        provenance={
            "sources": ["ml1m_ratings"],
            "temporal_scope": "pit_correct",   # LIE
            "tower": "movie",
            "reads": "movie mean rating as-of each row (claimed)",  # LIE
            "ctx_keys": [],                    # LIE: hides the table
        },
        serving="lookup",
    )


def build_user_history_entropy(ctx) -> Feature:
    """TRAP 2 (demo): real signal, unaffordable serving cost.

    Entropy of the user's full train-period rating distribution needs the
    whole event history at request time -> serving='history_scan'.
    """
    gm_entropy = _LN5  # fallback: max entropy (uniform rater)

    def compute(df, c):
        n = len(df)
        out = np.full(n, gm_entropy, dtype=float)
        uids = df["user_id"].to_numpy()
        for u in np.unique(uids):
            ev = c["user_events"].get(u)
            m = uids == u
            if ev is None:
                continue
            _, r = ev
            hist = np.bincount(r.astype(int), minlength=6)[1:6].astype(float)
            tot = hist.sum()
            if tot <= 0:
                continue
            p = hist / tot
            p = p[p > 0]
            out[m] = float(-(p * np.log(p)).sum())
        return pd.Series(out, index=df.index)

    return Feature(
        name="user_history_entropy",
        compute=compute,
        point_in_time="timestamp",
        description="TRAP (demo): Shannon entropy of the user's full "
                    "train-period rating distribution. Plausible signal, but "
                    "requires a full history scan per request -- too "
                    "expensive to serve, rejected at the cost gate before "
                    "any training time is spent.",
        provenance={
            "sources": ["ml1m_ratings"],
            "temporal_scope": "train_only",
            "tower": "user",
            "reads": "user's full train-period rating history",
            "ctx_keys": ["user_events"],
        },
        serving="history_scan",
    )
