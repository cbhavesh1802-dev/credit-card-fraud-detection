"""
evaluate.py
===========
Evaluation metrics, ROC curves, confusion matrix, and results table.
All plots saved to results/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, matthews_corrcoef,
    precision_score, recall_score
)


def compute_metrics(y_true, y_pred, y_proba):
    """Compute all evaluation metrics for a single model."""
    return {
        "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc"   : round(roc_auc_score(y_true, y_proba), 4),
        "mcc"       : round(matthews_corrcoef(y_true, y_pred), 4),
    }


def print_comparison_table(all_results):
    """Print formatted comparison table of all models."""
    print("\n" + "=" * 72)
    print(f"  {'Model':<28} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'MCC':>6}")
    print("  " + "-" * 68)
    for name, r in all_results.items():
        print(f"  {name:<28} {r['precision']:>6.4f} {r['recall']:>6.4f} "
              f"{r['f1']:>6.4f} {r['roc_auc']:>6.4f} {r['mcc']:>6.4f}")
    print("=" * 72)


def plot_roc_curves(all_results, y_test, cascade_proba=None, save_path="results/roc_curves.png"):
    """Plot ROC curves for all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Credit Card Fraud Detection — Model Evaluation", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for (name, r), color in zip(all_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[0].plot(fpr, tpr, label=f"{name} ({r['roc_auc']:.3f})", color=color)

    if cascade_proba is not None:
        from sklearn.metrics import roc_auc_score
        fpr_c, tpr_c, _ = roc_curve(y_test, cascade_proba)
        auc_c = roc_auc_score(y_test, cascade_proba)
        axes[0].plot(fpr_c, tpr_c, label=f"★ Two-Stage Cascade ({auc_c:.3f})",
                     color="black", linewidth=2.5, linestyle="--")

    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Confusion matrix for cascade
    if cascade_proba is not None:
        from src.threshold_optimizer import optimize_threshold
        thresh = optimize_threshold(y_test, cascade_proba)
        cascade_pred = (cascade_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, cascade_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=["Legitimate", "Fraud"],
                    yticklabels=["Legitimate", "Fraud"])
        axes[1].set_title("Confusion Matrix — ★ Two-Stage Cascade")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Evaluation plots saved → {save_path}")


def save_results(all_results, path="results/metrics.json"):
    """Save all metrics to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {
        name: {k: v for k, v in r.items() if isinstance(v, (int, float, str))}
        for name, r in all_results.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved → {path}")
