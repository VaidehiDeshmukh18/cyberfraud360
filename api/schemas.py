"""Pydantic schemas for the fraud detection API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """
    A single transaction's features — the input to POST /predict.

    Field constraints are deliberately loose: the model will handle
    out-of-range values, and we don't want to reject edge cases that
    might be real.
    """
    Time: float = Field(..., description="Seconds since dataset start")
    Amount: float = Field(..., ge=0, description="Transaction amount in EUR")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class FeatureContribution(BaseModel):
    """One row of the SHAP explanation."""
    feature: str
    value: float
    shap: float


class PredictionResponse(BaseModel):
    """Response from POST /predict."""
    probability: float = Field(..., ge=0, le=1)
    risk_score: float = Field(..., ge=0, le=100)
    risk_category: str = Field(..., description="Low, Medium, or High")
    top_contributors: list[FeatureContribution]


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: str
    model_loaded: bool
    production_threshold: float