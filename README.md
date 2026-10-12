# CyberFraud360 — Fraud Detection & Risk Scoring Platform

Machine learning system that detects fraudulent credit card transactions, assigns each one a 0–100 risk score, and explains predictions through SHAP feature attributions. The project runs end-to-end: raw CSV through dtype-optimized Parquet, through a sklearn feature pipeline and three trained models, through cost-based threshold tuning and tiered risk scoring, and finally through a FastAPI service that returns explanations with every prediction.

---

## Headline Results (Validation Set)

| Metric | Value | Context |
|---|---|---|
| **PR-AUC** | 0.871 | ~650× random baseline (0.0013) |
| **ROC-AUC** | 0.9799 | High, but inflated by data separability |
| **Precision @ production threshold** | 51.5% | 1 in 2 alerts is real fraud |
| **Recall @ production threshold** | 89.3% | 50 of 56 frauds caught |
| **Precision @ High-risk tier** | 80.7% | Top-priority reviewer queue |
| **API latency** | <100 ms per prediction | Model + SHAP loaded at startup |

Production model: **Random Forest** (200 trees, `class_weight='balanced'`), selected over XGBoost after a three-way comparison in which RF matches or beats XGBoost on PR-AUC and precision@k with default hyperparameters.

---

## The Business Problem

The system detects fraudulent transactions in real time, under three operational constraints:

1. **The alert queue stays manageable.** False positives cost reviewer time; a model that flags 5% of all transactions is unusable.
2. **The model catches as much fraud as possible.** Missed fraud is directly expensive.
3. **Every flag is explainable.** A reviewer sees *why* a transaction is flagged, not just the score. For regulated industries, this is not optional — GDPR Article 22 gives customers the right to an explanation for automated decisions.

The system produces a prioritized queue:

- **High tier** — top ~0.13% of transactions by score, 80.7% precision. Reviewer starts here.
- **Medium tier** — ~0.09% of transactions, 10% precision. Lower priority.
- **Low tier** — everything else. Not reviewed.

Every flagged transaction comes with its top 5 SHAP feature contributions.

---

## Dataset

**ULB Credit Card Fraud Detection** (Pozzolo et al., Université Libre de Bruxelles).

- 284,807 transactions over 48 hours of European credit card activity
- 492 fraud cases → **0.173% base rate** (1 fraud per 578 transactions)
- Features: `Time` (seconds since start), `Amount`, `V1`–`V28` (PCA-anonymized), `Class` (0/1 target)
- Direct download, no auth required — see `scripts/download_data.py`

### Dataset Limitations

The V-features are PCA-transformed for privacy, which rules out direct semantic interpretation (we do not know what `V14` represents). The dataset also lacks customer, device, and merchant identifiers, which removes some features common in production fraud systems — velocity (transactions per hour per customer), new-device flags, IP-based signals. These are the first additions on a richer dataset like IEEE-CIS, and appear under Future Improvements.

---

## Architecture

```
Raw CSV
   │
   ▼
Data Layer        (src/data/)           Loading, dtype optimization, time-based split
   │
   ▼
Feature Pipeline  (src/features/)       sklearn Pipeline: AmountFeatures, TimeFeatures
   │
   ▼
Models            (src/models/)         LogReg → RandomForest → XGBoost
   │
   ▼
Evaluation        (src/evaluation/)     Metrics, PR curves, cost-based threshold selection
   │
   ▼
Risk Score        (src/inference/)      Probability → 0-100 score → Low/Medium/High
   │
   ▼
Explainer         (src/explainability/) SHAP TreeExplainer
   │
   ▼
API               (api/)                FastAPI service: POST /predict, GET /health
```

Three design principles hold throughout:

- **Logic lives in `src/`, exploration lives in `notebooks/`.** Notebooks import reusable functions; they never define them.
- **The feature pipeline has one source of truth.** Training, the API, and tests all call `build_feature_pipeline()` — training/serving skew is impossible by construction.
- **Every artifact is reproducible from `transactions.parquet`.** The processed splits regenerate deterministically.

---

## Repository Structure

```
cyberfraud360/
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   └── schemas.py                 # Pydantic request/response models
├── data/
│   ├── raw/                       # Original CSV (gitignored)
│   ├── processed/                 # Parquet splits (gitignored)
│   └── external/
├── models/                        # Trained .joblib artifacts (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_explainability.ipynb
├── scripts/
│   ├── download_data.py
│   └── test_api.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py              # load_raw, save_parquet, load_parquet
│   │   └── splitter.py            # time_based_split
│   ├── features/
│   │   ├── transformers.py        # AmountFeatures, TimeFeatures
│   │   └── pipeline.py            # build_feature_pipeline
│   ├── models/
│   │   ├── baseline.py            # Logistic Regression + feature list
│   │   ├── random_forest.py       # Production model
│   │   └── xgboost_model.py       # Benchmark model
│   ├── evaluation/
│   │   ├── metrics.py             # Core metrics + precision@k
│   │   ├── plots.py               # ROC + PR curves
│   │   └── threshold.py           # Sweep + cost-based picker
│   ├── explainability/
│   │   └── shap_wrapper.py        # TreeExplainer wrapper
│   └── inference/
│       └── risk_score.py          # Probability → 0-100 + tiers
├── configs/
├── reports/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Create virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset (~143 MB, direct, no auth required)
python scripts/download_data.py
```

### Reproduce the Pipeline

The notebooks run in order:

1. `notebooks/01_data_exploration.ipynb` — initial inspection, loader validation, time-based split
2. `notebooks/02_eda.ipynb` — fraud signals, feature engineering plan, pipeline promotion to `src/features/`
3. `notebooks/03_baseline_model.ipynb` — three-model comparison, threshold tuning, risk tiers
4. `notebooks/04_explainability.ipynb` — SHAP analysis (local + global)

The notebooks produce `models/rf_pipeline.joblib`, which the API loads at startup.

### Run the API

```bash
uvicorn api.main:app --port 8000
```

The interactive Swagger UI runs at `http://127.0.0.1:8000/docs`. A smoke-test client is provided:

```bash
python scripts/test_api.py
```

---

## Methodology

### 1. Data Preparation

The loader downcasts the 143 MB CSV to a 33 MB Parquet file by using `float32` where `float64` is unnecessary and `int8` for binary columns. The OpenML version of the dataset wraps the `Class` column in single quotes (`"'0'"` instead of `0`); the loader strips this before casting. All downstream code reads Parquet, which loads 10–15× faster than CSV.

### 2. Train / Validation / Test Split

The split is **time-based** (70/15/15), not random. Fraud evolves over time, and a random split leaks future information into training. Fraud rate shifts across splits (0.184% / 0.131% / 0.122%) — this is a real property of the data, not a split artifact.

### 3. Exploratory Analysis

Three numeric findings drive feature engineering:

- **Zero-amount transactions are ~7× more common in fraud** (4.69% vs 0.68%), matching the known card-testing attack pattern. → `is_zero_amount`.
- **Hour 2 of the day shows ~9× elevated fraud rate** (1.71% vs 0.19% baseline). → `is_unusual_hour` for the 2–5 AM window.
- **V-features already strongly separate the classes** (Cohen's d > 5 for several), reflecting that PCA on labeled data tends to encode the rare class by construction. This makes PR-AUC the right primary metric: ROC-AUC above 0.95 is cheap on this dataset.

The data-quality pass also finds 716 exact duplicates in training (removed) and 2,860 near-duplicates (kept — they likely include card-testing probes).

### 4. Feature Engineering

Six features derive from `Amount` and `Time`, all implemented as sklearn-compatible transformers:

| Feature | Rationale |
|---|---|
| `is_zero_amount` | Direct 7× fraud signal from EDA |
| `amount_log` | `log1p(Amount)` compresses the heavy tail |
| `hour` | Hour of day (0–23) for cyclic patterns |
| `is_unusual_hour` | Direct 9× fraud signal from EDA (2–5 AM) |
| `hour_sin`, `hour_cos` | Cyclic encoding for linear models |

Transformers chain into a `Pipeline` via `build_feature_pipeline()`, which training, the API, and tests all import.

### 5. Model Comparison

Three models train with identical features and `class_weight='balanced'` (XGBoost uses the equivalent `scale_pos_weight`):

| Model | ROC-AUC | PR-AUC | Precision @ k=100 | Recall @ k=100 | Train time |
|---|---|---|---|---|---|
| Logistic Regression | 0.981 | 0.845 | 49.0% | 87.5% | 1.3 s |
| **Random Forest** | **0.980** | **0.871** | **50.0%** | **89.3%** | 26 s |
| XGBoost (defaults) | 0.988 | 0.848 | 47.0% | 83.9% | 2.6 s |

**Random Forest is chosen** on the principle of "the simpler model when performance ties." XGBoost under default hyperparameters ties LR on PR-AUC; proper tuning (grid search over `max_depth`, `learning_rate`, `n_estimators`) appears under Future Improvements.

### 6. Threshold Selection

Expected-cost minimization under three business scenarios:

| Scenario | C_FN : C_FP | Optimal threshold | Precision | Recall |
|---|---|---|---|---|
| Conservative | 100 : 1 | **0.036** | 51.5% | 89.3% |
| Balanced | 667 : 1 | 0.006 | 8.2% | 94.6% |
| Aggressive | 2500 : 1 | 0.001 | 2.0% | 96.4% |

**Conservative (0.036) is the production threshold.** The two higher-cost scenarios slide the optimizer toward near-boundary values, which indicates those ratios are unrealistic for this dataset. Cost parameters live in configuration — a product manager can tune them without a code change.

### 7. Risk Scoring

Probabilities map to a 0–100 score (`prob × 100`) and three tiers:

| Tier | Score range | Count on val | Precision | Catch rate |
|---|---|---|---|---|
| High | ≥ 7.55 | 57 (0.13%) | 80.7% | 82% of all fraud |
| Medium | 3.58 – 7.55 | 40 (0.09%) | 10.0% | 7% of all fraud |
| Low | < 3.58 | 42,624 (99.77%) | — | — |

### 8. Explainability

Every API prediction carries the top 5 SHAP contributions. The TreeExplainer walks the trained Random Forest's paths to compute exact Shapley values — the top contributors plus the base rate sum to the model's output by construction.

Global SHAP importance across 5,000 validation rows ranks **V12, V14, V4, V11, V10, V17** as dominant drivers, consistent with the Cohen's d analysis in EDA. Engineered features do not appear in the top 15, confirming the V-features carry most of the signal on this dataset. Engineered features stay in anyway as insurance against distribution shift and for interpretability on anonymized data.

### 9. API

The FastAPI service exposes two endpoints:

- `GET /health` — liveness check, returns 200 and the loaded production threshold
- `POST /predict` — scores a transaction, returns probability, risk score, category, and top 5 SHAP contributors

The model and SHAP explainer load once at startup, which keeps per-request latency under 100 ms. Pydantic enforces input and output schemas; the Swagger UI at `/docs` is auto-generated from the type hints.

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"Time": 10000, "V1": -1.2, ..., "V28": 0.1, "Amount": 149.62}'
```

**Example response:**

```json
{
  "probability": 0.945,
  "risk_score": 94.5,
  "risk_category": "High",
  "top_contributors": [
    {"feature": "V14", "value": -8.89, "shap": 0.114},
    {"feature": "V17", "value": -4.37, "shap": 0.075},
    {"feature": "V10", "value": -4.40, "shap": 0.073},
    {"feature": "V12", "value": -5.74, "shap": 0.068},
    {"feature": "V4",  "value":  7.23, "shap": 0.045}
  ]
}
```

---

## Tech Stack

Python 3.10 · pandas · numpy · scikit-learn · xgboost · shap · matplotlib · seaborn · FastAPI · uvicorn · joblib · pytest

---

## About

Dataset credit: Pozzolo et al., Université Libre de Bruxelles.