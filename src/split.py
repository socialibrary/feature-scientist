"""Strict TEMPORAL train/validation split.

This is load-bearing for the whole project: the Day 5-6 leakage detector
relies on the guarantee that *no validation row is older than any train row*.
We split on the timestamp itself (not on random rows):

  1. sort every rating by timestamp (ascending)
  2. cutoff = timestamp at the 80th percentile of rows
  3. train = all rows with timestamp <= cutoff
     valid = all rows with timestamp  > cutoff

Step 3 (rather than a positional iloc split) keeps the guarantee strict even
when many rows share the exact cutoff timestamp: max(train.ts) < min(valid.ts)
always holds. The validation set therefore simulates "the future", which is
exactly what the leakage traps in the demo will try (and fail) to peek at.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from data import load_all

TRAIN_FRACTION = 0.8


def temporal_split(
    df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split a timestamped ratings dataframe into train/valid.

    Returns (train, valid, cutoff_timestamp). Guarantees
    max(train.timestamp) < min(valid.timestamp).
    """
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut_pos = int(len(df) * train_fraction)
    cutoff_ts = int(df.loc[cut_pos, "timestamp"])

    train = df[df["timestamp"] <= cutoff_ts].reset_index(drop=True)
    valid = df[df["timestamp"] > cutoff_ts].reset_index(drop=True)

    assert len(train) > 0 and len(valid) > 0, "empty split -- check timestamps"
    assert train["timestamp"].max() < valid["timestamp"].min(), \
        "temporal discipline violated: train/valid timestamps overlap"
    return train, valid, cutoff_ts


if __name__ == "__main__":
    df = load_all()
    train, valid, cutoff_ts = temporal_split(df)
    cutoff_dt = pd.to_datetime(cutoff_ts, unit="s", utc=True)
    print(f"cutoff: {cutoff_dt} (ts={cutoff_ts})")
    print(f"train: {len(train):,} rows  "
          f"[{train['datetime'].min()} -> {train['datetime'].max()}]  "
          f"pos_rate={train['liked'].mean():.3f}")
    print(f"valid: {len(valid):,} rows  "
          f"[{valid['datetime'].min()} -> {valid['datetime'].max()}]  "
          f"pos_rate={valid['liked'].mean():.3f}")
    print("temporal guarantee OK: max(train.ts) < min(valid.ts)")
