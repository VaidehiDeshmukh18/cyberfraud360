"""Test the /predict endpoint with a known fraud transaction from validation."""
import json
import sys
sys.path.append(".")

import requests
from src.data.loader import load_parquet

# Load one known fraud row
val_df = load_parquet("data/processed/val.parquet")
fraud_rows = val_df[val_df["Class"] == 1]
sample = fraud_rows.iloc[0].drop("Class").to_dict()

# Convert numpy types to native Python for JSON serialization
sample = {k: float(v) for k, v in sample.items()}

print("Sending transaction to /predict...")
response = requests.post("http://127.0.0.1:8000/predict", json=sample)
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))