#!/usr/bin/env python3
"""Problem Finder: identify where the model is failing, ranked by impact.

Takes a trained baseline and validation data, slices by each feature,
and ranks problems by: (slice_error - overall_error) * slice_size.

Output: ranked list of problems with hypotheses about WHY.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def find_problems(valid_df: pd.DataFrame, valid_proba: np.ndarray,
                  target_col: str, max_slices: int = 10,
                  is_regression: bool | None = None) -> list[dict]:
    """Find worst-performing slices, ranked by impact.

    Impact = (slice_error_rate - overall_error_rate) * slice_fraction
    This prioritizes slices that are both bad AND large enough to matter.

    Handles regression (continuous target, value predictions) via residual
    errors, binary classification (proba of class 1), and multiclass
    classification (proba matrix -> argmax). is_regression=None keeps the
    old auto-detect (target with > 2 distinct values = regression).
    """
    y_true_raw = valid_df[target_col].values
    _proba = np.asarray(valid_proba)
    # Regression if the target is not binary (more than 2 distinct values),
    # unless the caller says otherwise.
    if is_regression is None:
        _is_regression = pd.Series(y_true_raw).nunique(dropna=True) > 2
    else:
        _is_regression = is_regression
    if _is_regression:
        y_true = y_true_raw.astype(float)
        y_pred = _proba.ravel().astype(float)
        errors = np.abs(y_true - y_pred)
    else:
        # Label-safe encoding: sorted unique, aligned with encode_labels().
        classes = np.sort(pd.unique(pd.Series(y_true_raw).dropna()))
        y_true = pd.Categorical(y_true_raw, categories=classes).codes
        if _proba.ndim == 2 and _proba.shape[1] > 1:
            y_pred = _proba.argmax(axis=1)
        else:
            y_pred = (_proba.ravel() > 0.5).astype(int)
        errors = (y_true != y_pred).astype(int)
    overall_error = errors.mean()
    
    problems = []
    
    for col in valid_df.columns:
        if col == target_col:
            continue
        
        s = valid_df[col]
        
        # Categorical: slice by value
        if s.dtype == object or s.dtype.name == 'category' or s.nunique() < 20:
            for val in s.dropna().unique():
                mask = (s == val).values
                if mask.sum() < 50:  # too small to matter
                    continue
                slice_error = errors[mask].mean()
                lift = slice_error - overall_error
                # Regression: relative lift (10% worse than overall);
                # classification: absolute 2pp.
                _thresh = 0.10 * overall_error if _is_regression else 0.02
                if lift > _thresh:
                    problems.append({
                        "slice": f"{col}={val}",
                        "column": col,
                        "value": str(val),
                        "n": int(mask.sum()),
                        "fraction": float(mask.mean()),
                        "error_rate": float(slice_error),
                        "lift_pp": float(lift * 100),
                        "impact": float(lift * mask.mean()),
                    })
        # Numeric: slice by quintile
        elif pd.api.types.is_numeric_dtype(s.dtype):
            try:
                bins = pd.qcut(s, 5, duplicates='drop')
                for interval in bins.cat.categories:
                    mask = (bins == interval).values
                    if mask.sum() < 50:
                        continue
                    slice_error = errors[mask].mean()
                    lift = slice_error - overall_error
                    _thresh = 0.10 * overall_error if _is_regression else 0.02
                    if lift > _thresh:
                        problems.append({
                            "slice": f"{col} in {interval}",
                            "column": col,
                            "value": str(interval),
                            "n": int(mask.sum()),
                            "fraction": float(mask.mean()),
                            "error_rate": float(slice_error),
                            "lift_pp": float(lift * 100),
                            "impact": float(lift * mask.mean()),
                        })
            except Exception:
                continue
    
    # Rank by impact (descending)
    problems.sort(key=lambda p: p["impact"], reverse=True)
    return problems[:max_slices]


def format_problems(problems: list[dict], overall_error: float) -> str:
    """Format problems for display."""
    lines = [
        f"Overall error rate: {overall_error:.1%}",
        f"Found {len(problems)} problem slices:",
        "",
    ]
    for i, p in enumerate(problems, 1):
        lines.append(
            f"{i}. {p['slice']}\n"
            f"   Error: {p['error_rate']:.1%} (+{p['lift_pp']:.1f}pp vs overall), "
            f"n={p['n']} ({p['fraction']:.1%} of data)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("Use via: from problem_finder import find_problems")
