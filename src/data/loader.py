"""Memory-optimized loader for the Credit Card Fraud dataset."""
import pandas as pd


def load_raw(csv_path: str) -> pd.DataFrame:
    """
    Load the raw credit card fraud CSV with memory-optimized dtypes.

    The OpenML CSV wraps the Class column in single quotes
    (e.g. "'0'" instead of 0), so we read it as a string and
    convert explicitly.

    Parameters
    ----------
    csv_path : str
        Path to creditcard.csv

    Returns
    -------
    pd.DataFrame
        Transaction data with downcast numeric dtypes.
    """
    # Read with safe defaults for the problematic column.
    df = pd.read_csv(csv_path)

    # Clean the Class column: strip quotes, convert to int.
    df["Class"] = df["Class"].astype(str).str.strip("'").astype("int8")

    # Downcast the rest.
    float_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df[float_cols] = df[float_cols].astype("float32")
    df["Time"] = df["Time"].astype("int32")

    return df

def save_parquet(df: pd.DataFrame, parquet_path: str) -> None:
    """
    Save a DataFrame to Parquet with Snappy compression.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    parquet_path : str
        Destination path, e.g. 'data/processed/transactions.parquet'
    """
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)


def load_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from Parquet.

    Parameters
    ----------
    parquet_path : str
        Path to the .parquet file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_parquet(parquet_path, engine="pyarrow")

