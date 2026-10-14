"""Generic data loading for briefs.

Loads tabular data from the brief's table definitions, creating
train/valid splits. Replaces the MovieLens-specific load_all() for
generic tasks.
"""

import os
import re
import pandas as pd
import numpy as np


def parse_target(label_text: str) -> str:
    """Extract target column name from brief label text.

    Examples:
      "SalePrice, continuous target in train.csv" -> "SalePrice"
      "Transported BOOL, in train.csv" -> "Transported"
    """
    # Take the first token (column name) before comma or space.
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)", label_text or "")
    return m.group(1) if m else ""


def load_brief_data(brief: dict, test_size: float = 0.2,
                    seed: int = 42) -> tuple:
    """Load train/valid DataFrames from a brief.

    Returns (train_df, valid_df, target_col).
    Uses the *_train table; splits randomly (no temporal dimension for
    static benchmarks).
    """
    from brief import _table_defs
    defs = _table_defs()

    # Find the train table (ends with _train).
    train_name = None
    for t in brief.get("tables") or []:
        if t["name"].endswith("_train"):
            train_name = t["name"]
            break
    if not train_name or train_name not in defs:
        raise ValueError(f"no train table found in brief: {train_name}")

    _, loader = defs[train_name]
    df = loader()
    target = parse_target(brief.get("label", ""))
    if not target or target not in df.columns:
        raise ValueError(f"target '{target}' not in columns: "
                         f"{list(df.columns)[:5]}...")

    # Random split (static benchmark; no temporal leakage dimension).
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_valid = int(len(df) * test_size)
    valid = df.iloc[idx[:n_valid]].reset_index(drop=True)
    train = df.iloc[idx[n_valid:]].reset_index(drop=True)
    return train, valid, target
