"""Evaluation metrics for the fraud detection model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def compute_core_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the headline evaluation metrics for a fraud model.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted fraud probabilities in [0, 1].
    threshold : float
        Cutoff for converting probabilities to hard predictions.

    Returns
    -------
    dict with:
        roc_auc, pr_auc              - threshold-independent
        precision, recall, f1        - at the given threshold
        true_pos, false_pos, true_neg, false_neg
        threshold                    - echoed back for the record
    """
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold":  threshold,
        "roc_auc":    roc_auc,
        "pr_auc":     pr_auc,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "true_pos":   int(tp),
        "false_pos":  int(fp),
        "true_neg":   int(tn),
        "false_neg":  int(fn),
    }


def precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
) -> dict:
    """
    Precision and recall if we review the top-k highest-risk transactions.

    Mirrors how a fraud team operates: given a daily review budget of k
    alerts, how many actual frauds get caught, and how clean is the queue?
    """
    # Rank all transactions by probability descending, take top k
    order = np.argsort(-y_prob)
    top_k_idx = order[:k]

    y_top = y_true[top_k_idx]
    n_fraud_caught = int(y_top.sum())
    total_fraud = int(y_true.sum())

    precision_at_k = n_fraud_caught / k if k > 0 else 0.0
    recall_at_k    = n_fraud_caught / total_fraud if total_fraud > 0 else 0.0

    return {
        "k":                k,
        "precision_at_k":   precision_at_k,
        "recall_at_k":      recall_at_k,
        "fraud_caught":     n_fraud_caught,
        "fraud_total":      total_fraud,
    }