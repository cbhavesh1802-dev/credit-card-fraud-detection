"""
predict.py
==========
Run inference on new transactions using the saved cascade model.

Usage:
    python predict.py --input data/new_transactions.csv
    python predict.py --input data/new_transactions.csv --explain
"""

import argparse, sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.cascade            import TwoStageCascade
from src.preprocessor       import FraudPreprocessor
from src.feature_engineering import engineer_features
from src.shap_explainer     import FraudExplainer


def predict(input_path, explain=False):
    print("\n  Loading saved models...")
    prep    = FraudPreprocessor().load()
    cascade = TwoStageCascade().load()

    df = pd.read_csv(input_path)
    df = engineer_features(df)
    df = prep.transform(df)

    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df

    proba = cascade.predict_proba(X)
    pred  = (proba >= cascade.threshold).astype(int)

    results = pd.DataFrame({
        "transaction_id": range(len(pred)),
        "prediction":     ["FRAUD" if p == 1 else "LEGITIMATE" for p in pred],
        "fraud_probability": proba.round(4)
    })

    print(results.to_string(index=False))
    results.to_csv("results/predictions.csv", index=False)
    print("\n  Predictions saved → results/predictions.csv")

    if explain:
        explainer = FraudExplainer(cascade.xgb_model, feature_names=list(X.columns))
        for i in range(min(3, len(X))):
            row = X.iloc[[i]]
            exp = explainer.explain_transaction(row)
            print(f"\n  Transaction {i} — {results['prediction'].iloc[i]}")
            for feat, val in list(exp.items())[:5]:
                direction = "→ FRAUD" if val > 0 else "→ LEGIT"
                print(f"    {feat:<20} {val:+.4f}  {direction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="CSV file with transactions")
    parser.add_argument("--explain", action="store_true", help="Show SHAP explanations")
    args = parser.parse_args()
    predict(args.input, args.explain)
