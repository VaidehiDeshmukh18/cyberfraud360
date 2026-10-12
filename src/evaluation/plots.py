"""Evaluation plots for the fraud detection model."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)


def plot_roc_and_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figsize: tuple = (12, 5),
):
    """
    Plot ROC curve and Precision-Recall curve side by side.

    Both curves summarize threshold-free model quality:
      - ROC: TPR vs FPR as the threshold varies.
      - PR:  Precision vs Recall as the threshold varies.
    PR is the one to trust under class imbalance.
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})",
                 color="steelblue", linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray",
                 label="Random (AUC = 0.5)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # --- PR ---
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(recall, precision, label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
                 color="crimson", linewidth=2)
    axes[1].axhline(baseline, linestyle="--", color="gray",
                    label=f"Random baseline ({baseline:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()