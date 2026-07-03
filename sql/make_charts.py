"""
Builds chart images from the query outputs in outputs/*.csv.
Saves PNGs to outputs/charts/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# 1. Fraud rate by hour
df = pd.read_csv(os.path.join(OUT_DIR, "fraud_rate_by_hour.csv"))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df["hour_of_day"], df["fraud_rate_pct"], color="#c0392b")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Fraud Rate (%)")
ax.set_title("Fraud Rate by Hour of Day")
ax.set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "fraud_rate_by_hour.png"), dpi=150)
plt.close()

# 2. Fraud rate by amount band
df = pd.read_csv(os.path.join(OUT_DIR, "fraud_rate_by_amount_band.csv"))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(df["amount_band"], df["fraud_rate_pct"], color="#e67e22")
ax.set_xlabel("Transaction Amount Band")
ax.set_ylabel("Fraud Rate (%)")
ax.set_title("Fraud Rate by Transaction Amount Band")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "fraud_rate_by_amount_band.png"), dpi=150)
plt.close()

# 3. Rolling fraud rate trend
df = pd.read_csv(os.path.join(OUT_DIR, "rolling_fraud_rate.csv"))
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df["hour_bucket"], df["fraud_rate_pct"], alpha=0.35, label="Hourly fraud rate", color="#2980b9")
ax.plot(df["hour_bucket"], df["rolling_6h_avg_fraud_rate_pct"], linewidth=2.5, label="6-hour rolling average", color="#c0392b")
ax.set_xlabel("Hour Bucket (since dataset start)")
ax.set_ylabel("Fraud Rate (%)")
ax.set_title("Rolling Fraud Rate Trend")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "rolling_fraud_rate.png"), dpi=150)
plt.close()

# 4. Feature correlation ranking
df = pd.read_csv(os.path.join(OUT_DIR, "feature_correlation_ranked.csv")).head(15)
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#c0392b" if v < 0 else "#27ae60" for v in df["pearson_r"]]
ax.barh(df["feature"], df["pearson_r"], color=colors)
ax.set_xlabel("Pearson Correlation with Fraud (Class)")
ax.set_title("Top 15 Features Correlated with Fraud")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "feature_correlation.png"), dpi=150)
plt.close()

# 5. Class imbalance
df = pd.read_csv(os.path.join(OUT_DIR, "class_imbalance_summary.csv"))
fig, ax = plt.subplots(figsize=(6, 6))
labels = ["Legitimate", "Fraud"]
ax.pie(df["transaction_count"], labels=labels, autopct="%1.3f%%", colors=["#2980b9", "#c0392b"], explode=(0, 0.15))
ax.set_title("Transaction Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "class_imbalance.png"), dpi=150)
plt.close()

print(f"5 charts saved to {CHART_DIR}")
