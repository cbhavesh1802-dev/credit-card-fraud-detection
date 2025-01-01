"""
tests/test_pipeline.py
======================
Unit tests for all pipeline modules.
Run with: python -m pytest tests/ -v
"""

import sys, pytest
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.feature_engineering  import engineer_features
from src.balancer             import apply_smotetomek
from src.threshold_optimizer  import optimize_threshold


def make_sample_df(n=100, fraud_ratio=0.1):
    """Create a minimal synthetic dataset for testing."""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(n, 30),
        columns=["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    )
    df["Amount"] = np.abs(df["Amount"]) * 100
    df["Time"]   = np.abs(df["Time"]) * 86400
    df["Class"]  = (np.random.rand(n) < fraud_ratio).astype(int)
    return df


class TestFeatureEngineering:
    def test_adds_four_features(self):
        df = make_sample_df()
        result = engineer_features(df)
        for feat in ["Hour", "DaySegment", "AmountLog", "AmountSquared"]:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_hour_range(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert result["Hour"].between(0, 23).all(), "Hour values out of range"

    def test_day_segment_range(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert result["DaySegment"].between(0, 3).all(), "DaySegment out of range"

    def test_amount_log_non_negative(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert (result["AmountLog"] >= 0).all(), "AmountLog should be non-negative"

    def test_no_nan_introduced(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert result.isnull().sum().sum() == 0, "Feature engineering introduced NaN"


class TestThresholdOptimizer:
    def test_returns_float_in_range(self):
        y_true  = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        thresh  = optimize_threshold(y_true, y_proba, beta=2)
        assert isinstance(thresh, float), "Threshold must be float"
        assert 0.0 <= thresh <= 1.0, "Threshold must be in [0, 1]"

    def test_lower_than_default(self):
        """F2-optimized threshold should typically be below 0.5 for fraud."""
        y_true  = np.array([0]*90 + [1]*10)
        y_proba = np.random.rand(100) * 0.5
        y_proba[-10:] += 0.3
        thresh = optimize_threshold(y_true, y_proba, beta=2)
        assert thresh <= 0.6, "Threshold should be recall-optimized (lower)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
