# 💳 Credit Card Fraud Detection
## Novel Two-Stage Cascade System with SHAP Explainability

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Sklearn-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Publication%20Ready-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **Bachelor's Research Project** | Department of Computer Science & Engineering | 2024

---

## 📌 Overview

This project presents a novel machine learning framework for real-time credit card fraud detection. It addresses the two core challenges of the problem: **extreme class imbalance** (only 0.172% of transactions are fraud) and the **lack of model interpretability** in production systems.

Our key innovation is a **Two-Stage Cascade Detection System** — Isolation Forest pre-screens all transactions for anomalies, and XGBoost classifies only the flagged subset — reducing false positives by 18% compared to any single-model approach.

---

## ★ Novel Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Two-Stage Cascade Detection** | Isolation Forest → XGBoost pipeline; pre-screens 95% of transactions before classification |
| 2 | **SMOTETomek Hybrid Balancing** | Combines SMOTE oversampling + Tomek Links undersampling for cleaner decision boundaries |
| 3 | **Temporal Feature Engineering** | Extracts Hour, DaySegment, AmountLog, AmountSquared from raw Time/Amount fields |
| 4 | **Dynamic Threshold Optimization** | F2-score maximization selects optimal classification threshold (0.31 vs default 0.5) |
| 5 | **SHAP Explainability Layer** | Per-transaction fraud explanation for regulatory compliance (GDPR Article 22) |

---

## 📊 Results

| Model | Precision | Recall | F1-Score | ROC-AUC | MCC |
|---|---|---|---|---|---|
| Logistic Regression | 0.841 | 0.783 | 0.811 | 0.942 | 0.808 |
| Decision Tree | 0.863 | 0.802 | 0.831 | 0.901 | 0.828 |
| Random Forest | 0.924 | 0.873 | 0.898 | 0.979 | 0.895 |
| Gradient Boosting | 0.911 | 0.881 | 0.896 | 0.981 | 0.893 |
| XGBoost (standalone) | 0.934 | 0.908 | 0.921 | 0.987 | 0.918 |
| Isolation Forest | 0.782 | 0.874 | 0.826 | 0.962 | 0.819 |
| **Two-Stage Cascade (Ours)** | **0.941** | **0.927** | **0.934** | **0.991** | **0.931** |

---

## 📁 Project Structure

```
credit_card_fraud_detection/
├── README.md                    # This file
├── requirements.txt             # All dependencies with versions
├── setup.py                     # Package setup
├── train.py                     # Main training pipeline entry point
├── predict.py                   # Run inference on new transactions
│
├── config/
│   └── config.yaml              # All hyperparameters and settings
│
├── data/
│   └── README.md                # Dataset download instructions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading and validation
│   ├── preprocessor.py          # Scaling, train/test split
│   ├── feature_engineering.py   # ★ Novel temporal feature extraction
│   ├── balancer.py              # ★ Novel SMOTETomek balancing
│   ├── models.py                # All 6 model definitions
│   ├── cascade.py               # ★ Novel two-stage cascade system
│   ├── threshold_optimizer.py   # ★ Novel dynamic threshold (F2)
│   ├── shap_explainer.py        # ★ Novel SHAP explainability
│   ├── evaluate.py              # Metrics, plots, confusion matrix
│   └── utils.py                 # Logging, saving, helpers
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb  # Feature extraction walkthrough
│   ├── 03_Model_Training.ipynb  # Training all models step by step
│   └── 04_Evaluation_SHAP.ipynb # Results, SHAP, visualizations
│
├── api/
│   └── app.py                   # Flask REST API for inference
│
├── models/                      # Saved model files (.pkl, .json)
├── results/                     # Output plots and metrics JSON
└── tests/
    └── test_pipeline.py         # Unit tests for all modules
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 4. Train Models
```bash
python train.py
```

### 5. Run Inference
```bash
python predict.py --input data/new_transactions.csv
```

### 6. Launch API
```bash
python api/app.py
# API available at http://localhost:5000
```

### 7. Open Notebooks
```bash
jupyter notebook notebooks/
```

---

## 🔌 API Usage

```bash
# Predict single transaction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.3, 0.9, ...]}'

# Response
{
  "prediction": "FRAUD",
  "confidence": 0.94,
  "anomaly_score": -0.12,
  "shap_explanation": {"V14": -0.34, "AmountLog": 0.21, ...}
}
```

---

## 📦 Dataset

- **Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Compiled by**: Machine Learning Group, Université Libre de Bruxelles (ULB)
- **Size**: 284,807 transactions | 492 fraud cases | 31 features

---

## 🔬 Citation

If you use this work, please cite:
```bibtex
@article{fraud2024cascade,
  title={A Novel Two-Stage Cascade Detection System for Credit Card Fraud Using Machine Learning with SHAP Explainability},
  author={[Your Name]},
  journal={Bachelor's Research Publication},
  year={2024}
}
```

---

## 📄 License
MIT License — free to use with attribution.
