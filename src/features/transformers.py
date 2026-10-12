"""Feature engineering transformers (sklearn-compatible)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AmountFeatures(BaseEstimator, TransformerMixin):
    """
    Adds amount-derived features:
      - is_zero_amount : binary flag for Amount == 0
      - amount_log     : log1p(Amount), compresses the heavy tail

    Stateless: no fitting needed.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["is_zero_amount"] = (X["Amount"] == 0).astype("int8")
        X["amount_log"] = np.log1p(X["Amount"]).astype("float32")
        return X


class TimeFeatures(BaseEstimator, TransformerMixin):
    """
    Adds time-derived features from the `Time` column (seconds since
    dataset start):
      - hour            : hour of day (0-23)
      - is_unusual_hour : binary flag for the 2-5 AM window
      - hour_sin        : sine of hour on the unit circle
      - hour_cos        : cosine of hour on the unit circle

    Stateless: no fitting needed.
    """

    UNUSUAL_HOURS = [2, 3, 4, 5]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["hour"] = ((X["Time"] // 3600) % 24).astype("int8")
        X["is_unusual_hour"] = X["hour"].isin(self.UNUSUAL_HOURS).astype("int8")
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24).astype("float32")
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24).astype("float32")
        return X