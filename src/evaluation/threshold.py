"""Threshold tuning and operating-point selection for fraud detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, FP, FN, and expected cost across many thresholds.

    Returns a DataFrame — one row per threshold — ready for plotting
    or picking an operating point.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    total_pos = int(y_true.sum())

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / total_pos if total_pos > 0 else 0.0

        rows.append({
            "threshold": t,
            "precision": precision,
            "recall":    recall,
            "tp":        tp,
            "fp":        fp,
            "fn":        fn,
            "tn":        tn,
        })

    return pd.DataFrame(rows)


def pick_threshold_by_cost(
    sweep: pd.DataFrame,
    cost_fn: float,
    cost_fp: float,
) -> dict:
    """
    Choose the threshold that minimizes expected cost on the sweep.

    Expected cost at threshold t:
        E[cost] = cost_fn * FN(t) + cost_fp * FP(t)

    Returns the row with lowest cost, plus the full cost column
    attached to the sweep (as 'expected_cost').
    """
    sweep = sweep.copy()
    sweep["expected_cost"] = cost_fn * sweep["fn"] + cost_fp * sweep["fp"]
    best = sweep.loc[sweep["expected_cost"].idxmin()].to_dict()
    return {"best": best, "sweep_with_cost": sweep}