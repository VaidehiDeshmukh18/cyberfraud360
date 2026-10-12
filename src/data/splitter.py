"""Train/validation/test split for temporal transaction data."""
import pandas as pd
from typing import Tuple


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "Time",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transaction data by time into train / val / test.

    Earliest rows go to train, latest to test. Mirrors how the model
    will be used in production: train on the past, score the future.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    time_col : str
        Column to sort by (seconds since start).
    train_frac : float
        Fraction of rows for training.
    val_frac : float
        Fraction of rows for validation.
        Test fraction is implied: 1 - train_frac - val_frac.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac}"
        )

    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    return train_df, val_df, test_df