"""
Generates a sample transaction dataset matching the schema of the
Kaggle ULB Credit Card Fraud dataset (Time, V1-V28, Amount, Class).

Used for local development and testing of the SQL/analysis pipeline.
The real creditcard.csv (from Kaggle) can be used in its place -- same
column names, so no downstream code changes are needed.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_TOTAL = 284_807
FRAUD_RATE = 0.00172
N_FRAUD = int(N_TOTAL * FRAUD_RATE)
N_LEGIT = N_TOTAL - N_FRAUD

def make_time(n):
    hours = np.random.choice(np.arange(24), size=n, p=_hourly_weights())
    day = np.random.choice([0, 1], size=n)
    minute_sec = np.random.randint(0, 3600, size=n)
    return day * 86400 + hours * 3600 + minute_sec

def _hourly_weights():
    base = np.array([1,1,1,1,1,2,3,5,7,8,9,9,10,10,9,9,9,8,8,7,6,4,3,2], dtype=float)
    return base / base.sum()

legit_time = make_time(N_LEGIT)
legit_amount = np.round(np.random.exponential(scale=45, size=N_LEGIT) + 1, 2)
legit_amount = np.clip(legit_amount, 0.5, 3000)
legit_V = np.random.normal(loc=0, scale=1, size=(N_LEGIT, 28))

fraud_hours = np.random.choice(
    np.arange(24), size=N_FRAUD,
    p=_hourly_weights()[::-1] / _hourly_weights()[::-1].sum()
)
fraud_day = np.random.choice([0, 1], size=N_FRAUD)
fraud_time = fraud_day * 86400 + fraud_hours * 3600 + np.random.randint(0, 3600, size=N_FRAUD)
fraud_amount = np.round(np.random.exponential(scale=120, size=N_FRAUD) + 1, 2)
fraud_amount = np.clip(fraud_amount, 0.5, 5000)
fraud_V = np.random.normal(loc=0, scale=1, size=(N_FRAUD, 28))
fraud_V[:, 13] -= 4.0
fraud_V[:, 3]  += 3.0
fraud_V[:, 11] -= 3.5
fraud_V[:, 16] -= 2.5

data = {
    "Time": np.concatenate([legit_time, fraud_time]),
    "Amount": np.concatenate([legit_amount, fraud_amount]),
    "Class": np.concatenate([np.zeros(N_LEGIT, dtype=int), np.ones(N_FRAUD, dtype=int)]),
}
V_all = np.vstack([legit_V, fraud_V])
for i in range(28):
    data[f"V{i+1}"] = V_all[:, i]

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.sort_values("Time").reset_index(drop=True)

col_order = ["Time"] + [f"V{i+1}" for i in range(28)] + ["Amount", "Class"]
df = df[col_order]
df["Amount"] = df["Amount"].round(2)
for i in range(28):
    df[f"V{i+1}"] = df[f"V{i+1}"].round(6)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(SCRIPT_DIR, "creditcard.csv")
df.to_csv(out_path, index=False)

print(f"Wrote {len(df):,} rows ({df[\'Class\'].sum()} fraud, {df[\'Class\'].mean()*100:.3f}% fraud rate) to {out_path}")
