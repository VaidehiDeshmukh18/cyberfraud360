"""Downloads the Credit Card Fraud dataset into data/raw/."""
import os
import urllib.request

RAW_DIR = "data/raw"
URL = "https://www.openml.org/data/get_csv/1673544/phpKo8OWT"
DEST = os.path.join(RAW_DIR, "creditcard.csv")

os.makedirs(RAW_DIR, exist_ok=True)

if os.path.exists(DEST):
    print(f"Already exists: {DEST}")
else:
    print(f"Downloading from {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    size_mb = os.path.getsize(DEST) / (1024 * 1024)
    print(f"Downloaded {DEST} ({size_mb:.1f} MB)")