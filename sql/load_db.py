"""
Loads data/creditcard.csv into a local SQLite database (fraud_analytics.db).
Re-run this any time the CSV is replaced (e.g. with the real Kaggle file).
"""

import sqlite3
import pandas as pd
import os

DATA_PATH = "/home/claude/project/data/creditcard.csv"
DB_PATH = "/home/claude/project/data/fraud_analytics.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

df = pd.read_csv(DATA_PATH)

conn = sqlite3.connect(DB_PATH)
df.to_sql("transactions", conn, index=False, if_exists="replace")

# Indexes to keep the analyst queries fast
conn.execute("CREATE INDEX idx_class ON transactions(Class);")
conn.execute("CREATE INDEX idx_time ON transactions(Time);")
conn.execute("CREATE INDEX idx_amount ON transactions(Amount);")
conn.commit()

count = conn.execute("SELECT COUNT(*) FROM transactions;").fetchone()[0]
print(f"Loaded {count:,} rows into {DB_PATH} (table: transactions)")

conn.close()
