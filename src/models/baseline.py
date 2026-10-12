"""Baseline Logistic Regression for fraud detection."""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Columns to feed into the model.
# We drop the raw 'Time' column — its signal is captured by the
# engineered hour features, and raw seconds-since-start would
# just correlate with the time-based split.
FEATURE_COLUMNS = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount", "is_zero_amount", "amount_log",
       "hour", "is_unusual_hour", "hour_sin", "hour_cos"]
)
TARGET_COLUMN = "Class"


def build_baseline_pipeline(random_state: int = 42) -> Pipeline:
    """
    Build the baseline modeling pipeline: StandardScaler + LogisticRegression.

    - StandardScaler zero-centers each feature and scales to unit variance.
    - LogisticRegression with class_weight='balanced' handles the 1:518 imbalance.
    - solver='liblinear' is reliable for medium-sized binary problems with L2.
    - max_iter=1000 gives convergence headroom; the default 100 often warns.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )),
    ])


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target y from a feature-engineered DataFrame."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y