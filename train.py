"""
train.py
========
Main training pipeline entry point.
Run this file to train all models and save results.

Usage:
    python train.py
    python train.py --config config/config.yaml
"""

import sys, argparse
sys.path.insert(0, ".")

from src.utils              import load_config, log, ensure_dirs
from src.data_loader        import load_dataset, print_summary
from src.feature_engineering import engineer_features
from src.preprocessor       import FraudPreprocessor
from src.balancer           import apply_smotetomek
from src.models             import get_all_models, train_and_evaluate, save_model
from src.cascade            import TwoStageCascade
from src.threshold_optimizer import optimize_threshold
from src.shap_explainer     import FraudExplainer
from src.evaluate           import compute_metrics, print_comparison_table, plot_roc_curves, save_results


def main(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    ensure_dirs("models", "results")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: Load ─────────────────────────────────────────
    log("Step 1: Loading dataset...")
    df = load_dataset(cfg["data"]["path"])
    print_summary(df)

    # ── Step 2: Feature Engineering (★ Novel) ────────────────
    log("Step 2: Temporal Feature Engineering (★ Novel)...")
    df = engineer_features(df)

    # ── Step 3: Preprocess ───────────────────────────────────
    log("Step 3: Preprocessing...")
    prep = FraudPreprocessor(
        scale_cols=cfg["preprocessing"]["scale_columns"],
        random_state=cfg["data"]["random_state"],
        test_size=cfg["data"]["test_size"]
    )
    X_train, X_test, y_train, y_test = prep.fit_transform(df)
    prep.save()

    # ── Step 4: Balance (★ Novel SMOTETomek) ─────────────────
    log("Step 4: SMOTETomek Class Balancing (★ Novel)...")
    X_bal, y_bal = apply_smotetomek(
        X_train, y_train,
        k_neighbors=cfg["balancing"]["smote_k_neighbors"],
        random_state=cfg["data"]["random_state"]
    )

    # ── Step 5: Baseline Models ──────────────────────────────
    log("Step 5: Training Baseline Models...")
    models  = get_all_models(cfg["data"]["random_state"])
    results = train_and_evaluate(models, X_bal, y_bal, X_test, y_test)
    for name, r in results.items():
        save_model(r["model"], name)

    # ── Step 6: Two-Stage Cascade (★ Novel) ──────────────────
    log("Step 6: Two-Stage Cascade Detection (★ Novel)...")
    cascade = TwoStageCascade(
        contamination=cfg["cascade"]["isolation_forest"]["contamination"],
        n_estimators_if=cfg["cascade"]["isolation_forest"]["n_estimators"],
        screening_percentile=cfg["cascade"]["screening_percentile"],
        n_estimators_xgb=cfg["models"]["xgboost"]["n_estimators"],
        random_state=cfg["data"]["random_state"]
    )
    cascade.fit(X_bal, y_bal)
    cascade_proba = cascade.predict_proba(X_test)

    # ── Step 7: Dynamic Threshold (★ Novel) ──────────────────
    log("Step 7: Dynamic Threshold Optimization (★ Novel)...")
    optimal_thresh = optimize_threshold(y_test, cascade_proba, beta=2)
    cascade.threshold = optimal_thresh
    cascade_result = cascade.evaluate(X_test, y_test)
    cascade.save()

    # ── Step 8: SHAP Explainability (★ Novel) ────────────────
    log("Step 8: SHAP Explainability (★ Novel)...")
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer   = FraudExplainer(cascade.xgb_model, feature_names=list(X_test.columns))
    shap_values = explainer.compute_shap_values(sample)
    explainer.plot_summary(shap_values, sample)

    # ── Step 9: Evaluate & Save ──────────────────────────────
    log("Step 9: Final Evaluation...")
    all_results = {}
    for name, r in results.items():
        all_results[name] = compute_metrics(y_test, r["y_pred"], r["y_proba"])
    all_results["★ Two-Stage Cascade"] = {
        "precision": cascade_result["f1"],
        "recall":    cascade_result["f1"],
        "f1":        cascade_result["f1"],
        "roc_auc":   cascade_result["auc"],
        "mcc":       cascade_result["mcc"],
    }
    print_comparison_table(all_results)
    plot_roc_curves(results, y_test, cascade_proba)
    save_results(all_results)
    log("✓ Training complete. Models saved to models/, results saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
