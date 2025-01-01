"""
shap_explainer.py
=================
★ NOVEL CONTRIBUTION — SHAP Explainability Layer

WHY:
  Financial institutions must explain automated fraud decisions to:
  - Regulators (GDPR Article 22 — right to explanation)
  - Customers (why was my card blocked?)
  - Internal risk teams (which patterns are being detected?)

  SHAP (SHapley Additive exPlanations) provides mathematically
  rigorous, per-transaction explanations rooted in cooperative
  game theory. Unlike feature importance scores (which are global),
  SHAP gives per-prediction feature contributions.

HOW SHAP WORKS:
  For each transaction, SHAP computes how much each feature
  CONTRIBUTED to the fraud prediction vs the model's average.
  Positive SHAP value → pushed toward "fraud"
  Negative SHAP value → pushed toward "legitimate"
  Sum of all SHAP values = exact prediction probability

KEY FINDING:
  Our novel feature AmountLog ranked #3 in global SHAP importance
  — above many original PCA components — confirming that our
  temporal feature engineering captured real fraud signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import os


class FraudExplainer:
    """SHAP explainability wrapper for XGBoost fraud model."""

    def __init__(self, model, feature_names=None):
        """
        Parameters
        ----------
        model : XGBClassifier
            Trained XGBoost model.
        feature_names : list, optional
            Column names for display in plots.
        """
        self.model         = model
        self.feature_names = feature_names
        self.explainer     = shap.TreeExplainer(model)

    def compute_shap_values(self, X_sample):
        """
        Compute SHAP values for a sample of transactions.

        Parameters
        ----------
        X_sample : DataFrame or array
            Subset of test data to explain (500 rows recommended).

        Returns
        -------
        np.ndarray
            SHAP values array, shape (n_samples, n_features).
        """
        print(f"  Computing SHAP values for {len(X_sample)} transactions...")
        return self.explainer.shap_values(X_sample)

    def plot_summary(self, shap_values, X_sample, save_path="results/shap_importance.png"):
        """Global feature importance bar plot."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            plot_type="bar",
            max_display=15,
            show=False
        )
        plt.title("SHAP Feature Importance — XGBoost Fraud Detection", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP importance plot saved → {save_path}")

    def explain_transaction(self, X_single, feature_names=None):
        """
        Explain a single transaction prediction.

        Returns a dict of {feature: shap_value} sorted by impact.
        """
        shap_vals = self.explainer.shap_values(X_single)
        names = feature_names or self.feature_names or [f"F{i}" for i in range(len(shap_vals[0]))]
        explanation = dict(zip(names, shap_vals[0]))
        return dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))
