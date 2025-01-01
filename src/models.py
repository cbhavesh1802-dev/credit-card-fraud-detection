"""
models.py
=========
Defines and trains all 5 baseline classification models.
Each model is configured with the hyperparameters established
through 5-fold stratified cross-validation.

Models:
  1. Logistic Regression  — linear baseline, interpretable
  2. Decision Tree        — rule-based, explainable
  3. Random Forest        — tree ensemble, robust to noise
  4. Gradient Boosting    — sequential ensemble, high accuracy
  5. XGBoost              — optimized boosting, best single model
"""

import joblib
import os
import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics       import f1_score, roc_auc_score, matthews_corrcoef
import xgboost as xgb


def get_all_models(random_state=42):
    """
    Return dict of all baseline models with tuned hyperparameters.

    Returns
    -------
    dict : {model_name: sklearn_estimator}
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,           # Limits tree depth to prevent overfitting
            min_samples_leaf=5,     # Requires at least 5 samples per leaf
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,       # 200 individual decision trees
            max_features="sqrt",    # Each tree sees sqrt(n_features) features
            n_jobs=-1,              # Use all CPU cores
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,       # 200 sequential boosting rounds
            learning_rate=0.1,      # Step size for each round
            max_depth=5,            # Shallow trees to prevent overfitting
            random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,     # Slower learning → better generalization
            max_depth=6,
            subsample=0.8,          # 80% of rows per tree
            colsample_bytree=0.8,   # 80% of features per tree
            scale_pos_weight=577,   # Ratio negative:positive (handles imbalance)
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=random_state
        ),
    }


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Train all models and compute evaluation metrics.

    Returns
    -------
    dict : {model_name: {model, y_pred, y_proba, f1, auc, mcc}}
    """
    results = {}
    for name, model in models.items():
        print(f"\n  Training: {name}...")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"    F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}")
        results[name] = {
            "model": model, "y_pred": y_pred, "y_proba": y_proba,
            "f1": f1, "auc": auc, "mcc": mcc
        }
    return results


def save_model(model, name, directory="models/"):
    """Save a trained model to disk using joblib."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, path)
    print(f"  Model saved → {path}")
    return path


def load_model(name, directory="models/"):
    """Load a trained model from disk."""
    path = os.path.join(directory, f"{name.replace(' ', '_').lower()}.pkl")
    return joblib.load(path)
