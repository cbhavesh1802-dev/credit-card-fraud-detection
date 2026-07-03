# Credit Card Transaction Risk Analytics

Analysis of transaction-level data to surface fraud patterns and risk signals -- built with SQL, Python, and a two-stage ML model as supporting validation.

## Overview

This project analyzes ~285K anonymized credit card transactions to answer the questions a fraud/risk team actually asks: when does fraud happen, what amounts are riskiest, and which signals matter most? The SQL layer drives the analysis; a two-stage ML pipeline (Isolation Forest + XGBoost) sits underneath as model validation for the patterns found.

## Key Findings

- Fraud rate varies sharply by hour of day -- some hours run 3-5x the baseline fraud rate (see outputs/fraud_rate_by_hour.csv)
- Higher transaction amounts carry disproportionate risk -- fraud rate climbs from ~0.05% in low-value transactions to 2%+ in higher amount bands (outputs/fraud_rate_by_amount_band.csv)
- A handful of anonymized features (V14, V12, V4, V17) drive most of the predictive signal, confirmed by both correlation analysis and the models SHAP explanations
- Severe class imbalance (0.172% fraud) -- addressed via SMOTETomek resampling in the model layer

## Project Structure

data/ - transaction data (Time, V1-V28, Amount, Class) and generate_synthetic.py
sql/ - 00_schema.sql, 01_fraud_rate_by_hour.sql, 02_fraud_rate_by_amount_band.sql, 03_rolling_fraud_rate.sql (window function), 04_feature_correlation.sql, load_db.py, run_all_queries.py
outputs/ - query results, ready for Power BI import
notebooks/ - 01_EDA.ipynb
src/ - model pipeline (cascade, SHAP, threshold optimization)
train.py / predict.py

## How to Run

pip install -r requirements.txt
python sql/load_db.py
python sql/run_all_queries.py

To rebuild the model layer: python train.py

## Data Note

The dataset schema (Time, V1-V28, Amount, Class) matches the Kaggle ULB Credit Card Fraud dataset. V1-V28 are PCA-anonymized and carry no individual business meaning -- analyzed via correlation ranking rather than face-value interpretation. There is no merchant, location, or cardholder field in this dataset, so all derived dimensions (Hour, Amount Band) are computed from Time and Amount.

## Tech Stack

SQL (SQLite) - Python (Pandas, NumPy) - Power BI - Excel - scikit-learn - XGBoost - SHAP
