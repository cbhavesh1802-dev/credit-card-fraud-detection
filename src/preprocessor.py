"""
preprocessor.py
===============
Handles all data preprocessing steps:
  - RobustScaler for Amount and Time (outlier-resistant)
  - Stratified train/test split (preserves fraud ratio)

WHY RobustScaler?
  Financial transaction data contains extreme outliers — some transactions
  are 100x the typical amount. StandardScaler uses mean/std, which are
  heavily distorted by outliers. RobustScaler uses median/IQR instead,
  making it much more stable for financial data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class FraudPreprocessor:
    """
    Preprocessing pipeline for the Credit Card Fraud dataset.

    Attributes
    ----------
    scaler : RobustScaler
        Fitted scaler for Amount and Time columns.
    scale_cols : list
        Columns to apply scaling to.
    """

    def __init__(self, scale_cols=None, random_state=42, test_size=0.2):
        self.scale_cols   = scale_cols or ["Amount", "Time"]
        self.random_state = random_state
        self.test_size    = test_size
        self.scaler       = RobustScaler()

    def fit_transform(self, df: pd.DataFrame):
        """
        Fit scaler on full dataset, then split into train/test.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset including target 'Class' column.

        Returns
        -------
        X_train, X_test, y_train, y_test : arrays
        """
        df = df.copy()

        # Scale Amount and Time
        df[self.scale_cols] = self.scaler.fit_transform(df[self.scale_cols])

        X = df.drop("Class", axis=1)
        y = df["Class"]

        # Stratified split — preserves 0.172% fraud ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y       # CRITICAL — without this, test set may have no fraud
        )

        print(f"  Train: {len(X_train):,} rows | Fraud: {y_train.sum()}")
        print(f"  Test:  {len(X_test):,}  rows | Fraud: {y_test.sum()}")
        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to new data (inference)."""
        df = df.copy()
        df[self.scale_cols] = self.scaler.transform(df[self.scale_cols])
        return df

    def save(self, path="models/preprocessor.pkl"):
        """Save fitted scaler to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"  Scaler saved → {path}")

    def load(self, path="models/preprocessor.pkl"):
        """Load fitted scaler from disk."""
        self.scaler = joblib.load(path)
        return self
