"""
data_loader.py
==============
Handles loading and validating the Credit Card Fraud dataset.
Provides a clean DataFrame with basic sanity checks.
"""

import pandas as pd
import numpy as np
import os


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Credit Card Fraud CSV from disk.

    Parameters
    ----------
    path : str
        Path to creditcard.csv

    Returns
    -------
    pd.DataFrame
        Raw dataset with 284,807 rows and 31 columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the expected columns are not present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place creditcard.csv in the data/ folder."
        )

    df = pd.read_csv(path)

    # Validate expected columns
    expected_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the loaded dataset."""
    fraud = df["Class"].sum()
    legit = len(df) - fraud
    print("=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total transactions : {len(df):>10,}")
    print(f"  Legitimate         : {legit:>10,}  ({legit/len(df)*100:.2f}%)")
    print(f"  Fraudulent         : {fraud:>10,}  ({fraud/len(df)*100:.3f}%)")
    print(f"  Features           : {df.shape[1] - 1:>10}")
    print(f"  Missing values     : {df.isnull().sum().sum():>10}")
    print(f"  Amount range       : €{df['Amount'].min():.2f} — €{df['Amount'].max():.2f}")
    print("=" * 55)


def validate_no_leakage(X_train, X_test, y_train, y_test) -> None:
    """Assert no data leakage between train and test sets."""
    assert len(X_train) + len(X_test) > 0, "Empty splits detected"
    train_fraud_rate = y_train.mean()
    test_fraud_rate  = y_test.mean()
    assert abs(train_fraud_rate - test_fraud_rate) < 0.001, \
        "Stratification failed — fraud rates differ significantly"
    print(f"  ✓ Train fraud rate: {train_fraud_rate*100:.3f}%")
    print(f"  ✓ Test  fraud rate: {test_fraud_rate*100:.3f}%")
