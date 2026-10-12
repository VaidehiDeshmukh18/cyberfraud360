"""Random Forest classifier for fraud detection."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_random_forest_pipeline(
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = None,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Build a Random Forest pipeline for fraud detection.

    - No StandardScaler needed: tree splits are scale-invariant.
    - class_weight='balanced' handles the 1:518 imbalance by reweighting
      the Gini impurity calculation.
    - n_estimators=200 is a reasonable default: more trees = more stable
      predictions, with diminishing returns past ~300.
    - max_depth=None lets trees grow until pure; could overfit, but with
      200 trees averaging, the noise cancels. We'll inspect this.
    - n_jobs=-1 parallelizes across all CPU cores.
    """
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])