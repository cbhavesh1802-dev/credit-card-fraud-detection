"""
Generates a synthetic dataset matching the schema and statistical shape of the
Kaggle ULB Credit Card Fraud dataset (Time, V1-V28, Amount, Class).

This is a STAND-IN for the real creditcard.csv so the SQL/EDA/dashboard pipeline
can be built and tested now. Swap in the real file later -- same column names,
so no downstream code changes needed.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_TOTAL = 100_000          # smaller than the real 284,807 for fast local iteration
FRAUD_RATE = 0.00172       # matches real dataset's ~0.172%
N_FRAUD = int(N_TOTAL * FRAUD_RATE)
N_LEGIT = N_TOTAL - N_FRAUD

SECONDS_IN_TWO_DAYS = 2 * 24 * 60 * 60

def make_time(n):
    # Transactions cluster around daytime hours, thin out at night -- matches
    # real-world transaction volume patterns (and the real dataset's shape).
    hours = np.random.choice(
        np.arange(24),
        size=n,
        p=_hourly_weights(),
    )
    day = np.random.choice([0, 1], size=n)
    minute_sec = np.random.randint(0, 3600, size=n)
    return day * 86400 + hours * 3600 + minute_sec

def _hourly_weights():
    # Rough daily activity curve: low overnight, peaks midday/evening
    base = np.array([
        1,1,1,1,1,2,3,5,7,8,9,9,10,10,9,9,9,8,8,7,6,4,3,2
    ], dtype=float)
    return base / base.sum()

# --- Legit transactions ---
legit_time = make_time(N_LEGIT)
legit_amount = np.round(np.random.exponential(scale=45, size=N_LEGIT) + 1, 2)
legit_amount = np.clip(legit_amount, 0.5, 3000)
legit_V = np.random.normal(loc=0, scale=1, size=(N_LEGIT, 28))

# --- Fraudulent transactions ---
# Fraud skews toward late-night hours and has a different amount distribution
fraud_hours = np.random.choice(
    np.arange(24), size=N_FRAUD,
    p=_hourly_weights()[::-1] / _hourly_weights()[::-1].sum()  # inverted curve -> more nighttime
)
fraud_day = np.random.choice([0, 1], size=N_FRAUD)
fraud_time = fraud_day * 86400 + fraud_hours * 3600 + np.random.randint(0, 3600, size=N_FRAUD)
fraud_amount = np.round(np.random.exponential(scale=120, size=N_FRAUD) + 1, 2)
fraud_amount = np.clip(fraud_amount, 0.5, 5000)
# Shift a few V-features to create separable signal (mimics real dataset's V14, V4, V12 etc.)
fraud_V = np.random.normal(loc=0, scale=1, size=(N_FRAUD, 28))
fraud_V[:, 13] -= 4.0   # V14-like strong negative signal
fraud_V[:, 3]  += 3.0   # V4-like positive signal
fraud_V[:, 11] -= 3.5   # V12-like negative signal
fraud_V[:, 16] -= 2.5   # V17-like negative signal

# --- Combine ---
data = {
    "Time": np.concatenate([legit_time, fraud_time]),
    "Amount": np.concatenate([legit_amount, fraud_amount]),
    "Class": np.concatenate([np.zeros(N_LEGIT, dtype=int), np.ones(N_FRAUD, dtype=int)]),
}
V_all = np.vstack([legit_V, fraud_V])
for i in range(28):
    data[f"V{i+1}"] = V_all[:, i]

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df = df.sort_values("Time").reset_index(drop=True)

col_order = ["Time"] + [f"V{i+1}" for i in range(28)] + ["Amount", "Class"]
df = df[col_order]

out_path = "/home/claude/project/data/creditcard.csv"
df.to_csv(out_path, index=False)

print(f"Wrote {len(df):,} rows ({df['Class'].sum()} fraud, "
      f"{df['Class'].mean()*100:.3f}% fraud rate) to {out_path}")
