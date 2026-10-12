"""SHAP-based explainability for the Random Forest fraud model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap


@dataclass
class LocalExplanation:
    """Per-transaction explanation returned by explain_one()."""
    base_value: float          # Model's average prediction (no features)
    prediction: float          # Model's prediction for this row
    top_contributors: pd.DataFrame   # Top-k features pushing toward/against fraud


class ShapExplainer:
    """
    Wrapper around shap.TreeExplainer for the fraud detection pipeline.

    Handles the sklearn Pipeline → model-only unwrapping, caches a
    reusable explainer, and returns structured outputs instead of
    SHAP's raw numpy arrays.
    """

    def __init__(self, trained_pipeline, feature_names: list[str]):
        """
        Parameters
        ----------
        trained_pipeline : sklearn Pipeline
            The fitted pipeline — e.g. loaded from models/rf_pipeline.joblib.
            We extract the final estimator (the actual classifier) for SHAP.
        feature_names : list[str]
            Column names of the feature matrix, in order.
        """
        self.pipeline = trained_pipeline
        self.model = trained_pipeline.named_steps["clf"]
        self.feature_names = list(feature_names)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_many(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for a batch of rows.

        Returns
        -------
        np.ndarray of shape (n_rows, n_features)
            SHAP values for the positive class (fraud).
        """
        # TreeExplainer on binary RF returns a list [class_0_shap, class_1_shap]
        # (older API) or a 3D array (newer). Handle both.
        raw = self.explainer.shap_values(X)
        if isinstance(raw, list):
            return raw[1]                 # fraud-class contributions
        if raw.ndim == 3:
            return raw[:, :, 1]           # (rows, features, class 1)
        return raw                        # already 2D

    def explain_one(self, x_row: pd.Series, top_k: int = 5) -> LocalExplanation:
        """
        Explain a single transaction. Returns the top_k features that
        most pushed the prediction toward fraud or legitimate.
        """
        X = x_row.to_frame().T[self.feature_names]
        shap_row = self.explain_many(X)[0]              # (n_features,)
        prediction = float(self.pipeline.predict_proba(X)[0, 1])

        # Base value — the model's "average fraud probability" with no features.
        expected = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            base = float(np.asarray(expected).ravel()[-1])
        else:
            base = float(expected)

        contrib = pd.DataFrame({
            "feature": self.feature_names,
            "value":   x_row[self.feature_names].values,
            "shap":    shap_row,
        })
        contrib["abs_shap"] = contrib["shap"].abs()
        top = contrib.sort_values("abs_shap", ascending=False).head(top_k).reset_index(drop=True)

        return LocalExplanation(
            base_value=base,
            prediction=prediction,
            top_contributors=top[["feature", "value", "shap"]],
        )

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Global feature importance: mean(|SHAP value|) across many rows.

        Returns a DataFrame sorted by importance (highest first).
        """
        shap_vals = self.explain_many(X)
        importance = np.abs(shap_vals).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": importance,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)