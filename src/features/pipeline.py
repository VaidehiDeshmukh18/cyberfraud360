"""Feature engineering pipeline — the single source of truth for
how raw transaction data is transformed into model-ready features.
Imported by training scripts, the API, and tests."""

from sklearn.pipeline import Pipeline

from src.features.transformers import AmountFeatures, TimeFeatures


def build_feature_pipeline() -> Pipeline:
    """
    Build the feature engineering pipeline.

    Current steps (stateless, Phase 3):
      1. AmountFeatures — is_zero_amount, amount_log
      2. TimeFeatures   — hour, is_unusual_hour, hour_sin, hour_cos

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline. Call .fit_transform(train) first,
        then .transform(val) and .transform(test).
    """
    return Pipeline([
        ("amount", AmountFeatures()),
        ("time",   TimeFeatures()),
    ])