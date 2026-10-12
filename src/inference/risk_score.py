"""Convert fraud probabilities to a 0-100 risk score with Low/Medium/High tiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """
    Thresholds for mapping fraud probability to risk category.

    - low_cutoff: below this probability, risk is Low (not flagged).
    - high_cutoff: above this probability, risk is High (urgent).
    - Between the two: Medium.
    """
    low_cutoff: float      # production threshold from Phase 10
    high_cutoff: float     # higher-confidence threshold


def probability_to_score(prob: float | np.ndarray) -> float | np.ndarray:
    """Map probability in [0, 1] to a 0-100 risk score."""
    return np.round(np.asarray(prob) * 100, 2)


def probability_to_category(prob: float | np.ndarray, config: RiskConfig) -> np.ndarray:
    """Map probability to 'Low' / 'Medium' / 'High' category."""
    prob = np.asarray(prob)
    categories = np.where(
        prob < config.low_cutoff, "Low",
        np.where(prob < config.high_cutoff, "Medium", "High"),
    )
    return categories


def score_transactions(
    prob: np.ndarray,
    config: RiskConfig,
) -> pd.DataFrame:
    """
    Convert an array of probabilities to a DataFrame with score + category.

    Returns a DataFrame with columns: probability, risk_score, risk_category.
    """
    return pd.DataFrame({
        "probability":   prob,
        "risk_score":    probability_to_score(prob),
        "risk_category": probability_to_category(prob, config),
    })