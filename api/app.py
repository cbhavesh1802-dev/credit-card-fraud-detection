"""
api/app.py
==========
Flask REST API for real-time fraud detection inference.

Endpoints:
  POST /predict        — predict single or batch transactions
  POST /explain        — predict + SHAP explanation
  GET  /health         — API health check
  GET  /model-info     — model metadata

Usage:
  python api/app.py
  # API runs at http://localhost:5000

Example request:
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [406.0, 0.0, -1.3, 0.9, ...]}'
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from src.cascade            import TwoStageCascade
from src.preprocessor       import FraudPreprocessor
from src.feature_engineering import engineer_features
from src.shap_explainer     import FraudExplainer

app = Flask(__name__)
CORS(app)

# ── Load models at startup ───────────────────────────────────
print("Loading models...")
cascade = TwoStageCascade().load()
prep    = FraudPreprocessor().load()
explainer = FraudExplainer(cascade.xgb_model)
print("Models loaded. API ready.")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "Two-Stage Cascade"})


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    return jsonify({
        "model": "Two-Stage Cascade (Isolation Forest + XGBoost)",
        "novel_contributions": [
            "Two-Stage Cascade Detection",
            "SMOTETomek Hybrid Balancing",
            "Temporal Feature Engineering",
            "Dynamic Threshold Optimization (F2)",
            "SHAP Explainability"
        ],
        "metrics": {
            "f1_score": 0.934,
            "roc_auc":  0.991,
            "mcc":      0.931
        },
        "threshold": cascade.threshold
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud probability for one or more transactions.

    Request body (JSON):
      Single:  {"features": [val1, val2, ..., val31]}
      Batch:   {"transactions": [[val1,...], [val1,...]]}

    Response:
      {"prediction": "FRAUD", "confidence": 0.94, "anomaly_flagged": true}
    """
    data = request.get_json()
    try:
        if "features" in data:
            features = [data["features"]]
        elif "transactions" in data:
            features = data["transactions"]
        else:
            return jsonify({"error": "Provide 'features' or 'transactions'"}), 400

        # Build DataFrame with correct column names
        cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df   = pd.DataFrame(features, columns=cols[:len(features[0])])
        df   = engineer_features(df)
        df   = prep.transform(df)

        proba = cascade.predict_proba(df)
        preds = (proba >= cascade.threshold).astype(int)

        results = []
        for i, (p, prob) in enumerate(zip(preds, proba)):
            results.append({
                "transaction_id":  i,
                "prediction":      "FRAUD" if p == 1 else "LEGITIMATE",
                "fraud_probability": round(float(prob), 4),
                "risk_level":      "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.4 else "LOW"
            })

        return jsonify({"results": results, "count": len(results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """Predict + return SHAP explanation for the top 5 features."""
    data = request.get_json()
    try:
        features = [data["features"]]
        cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df   = pd.DataFrame(features, columns=cols[:len(features[0])])
        df   = engineer_features(df)
        df   = prep.transform(df)

        proba = cascade.predict_proba(df)
        pred  = "FRAUD" if proba[0] >= cascade.threshold else "LEGITIMATE"

        exp = explainer.explain_transaction(df.iloc[[0]], feature_names=list(df.columns))
        top5 = {k: round(float(v), 4) for k, v in list(exp.items())[:5]}

        return jsonify({
            "prediction":       pred,
            "fraud_probability": round(float(proba[0]), 4),
            "shap_explanation": top5,
            "interpretation": "Positive values pushed toward FRAUD, negative toward LEGITIMATE"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
