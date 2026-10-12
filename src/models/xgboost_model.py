"""XGBoost classifier for fraud detection."""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute scale_pos_weight = n_negative / n_positive.

    XGBoost's imbalance-handling parameter. Multiplies the gradient of
    positive-class samples so the model treats fraud as weighted up to
    match the majority class. Equivalent in spirit to sklearn's
    class_weight='balanced'.
    """
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0:
        raise ValueError("No positive samples in y — cannot compute scale_pos_weight.")
    return n_neg / n_pos


def build_xgboost_pipeline(
    scale_pos_weight: float,
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Build an XGBoost pipeline for fraud detection.

    Hyperparameters chosen as reasonable defaults, NOT tuned:
      - n_estimators=400: more boosting rounds; early stopping could
        trim this later.
      - max_depth=6: XGBoost's canonical default. Shallower than RF's
        unlimited depth; each tree learns small corrections.
      - learning_rate=0.1: standard. Lower = more trees needed but
        often better accuracy.
      - scale_pos_weight: computed from training data (not hard-coded).
      - eval_metric='aucpr': PR-AUC is our target; XGBoost reports it
        during training if we add an eval set.
      - tree_method='hist': faster histogram-based algorithm, default
        on recent versions.

    No StandardScaler needed (tree-based).
    """
    return Pipeline([
        ("clf", XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])