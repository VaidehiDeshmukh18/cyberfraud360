"""FastAPI service for CyberFraud360 fraud detection."""
from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path

from api.schemas import (
    TransactionRequest,
    PredictionResponse,
    FeatureContribution,
    HealthResponse,
)
from src.features.pipeline import build_feature_pipeline
from src.models.baseline import FEATURE_COLUMNS
from src.inference.risk_score import RiskConfig, probability_to_score, probability_to_category
from src.explainability.shap_wrapper import ShapExplainer


# --- Configuration ---------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "rf_pipeline.joblib"
PRODUCTION_THRESHOLD = 0.0358     # From Phase 10 (Conservative cost scenario)
HIGH_CONFIDENCE_THRESHOLD = 0.0755


# --- App + state -----------------------------------------------------
app = FastAPI(
    title="CyberFraud360 — Fraud Detection API",
    description="Real-time fraud scoring with SHAP explanations.",
    version="1.0.0",
)


# Load model + build explainer ONCE at startup, not per-request.
# Loading a 200-tree RF for every request would add ~500ms.
_model = joblib.load(MODEL_PATH)
_feature_pipeline = build_feature_pipeline()
_explainer = ShapExplainer(_model, FEATURE_COLUMNS)
_risk_config = RiskConfig(
    low_cutoff=PRODUCTION_THRESHOLD,
    high_cutoff=HIGH_CONFIDENCE_THRESHOLD,
)


# --- Endpoints -------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Returns 200 if the service is up and the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        production_threshold=PRODUCTION_THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest) -> PredictionResponse:
    """
    Score a single transaction.

    Returns the fraud probability, a 0-100 risk score, a risk category
    (Low/Medium/High), and the top 5 SHAP feature contributions.
    """
    try:
        # 1. Build a single-row DataFrame from the request
        raw_df = pd.DataFrame([transaction.model_dump()])

        # 2. Apply the feature pipeline (adds engineered features)
        features_df = _feature_pipeline.fit_transform(raw_df)
        X = features_df[FEATURE_COLUMNS]

        # 3. Score with the model
        probability = float(_model.predict_proba(X)[0, 1])

        # 4. Build the risk score + category
        risk_score = float(probability_to_score(probability))
        risk_category = str(probability_to_category(probability, _risk_config))

        # 5. Explain with SHAP
        explanation = _explainer.explain_one(X.iloc[0], top_k=5)
        contributors = [
            FeatureContribution(
                feature=str(row["feature"]),
                value=float(row["value"]),
                shap=float(row["shap"]),
            )
            for _, row in explanation.top_contributors.iterrows()
        ]

        return PredictionResponse(
            probability=probability,
            risk_score=risk_score,
            risk_category=risk_category,
            top_contributors=contributors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")