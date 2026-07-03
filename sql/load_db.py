import sqlite3
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "fraud_analytics.db")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

df = pd.read_csv(DATA_PATH)

conn = sqlite3.connect(DB_PATH)
df.to_sql("transactions", conn, index=False, if_exists="replace")

conn.execute("CREATE INDEX idx_class ON transactions(Class);")
conn.execute("CREATE INDEX idx_time ON transactions(Time);")
conn.execute("CREATE INDEX idx_amount ON transactions(Amount);")
conn.commit()

count = conn.execute("SELECT COUNT(*) FROM transactions;").fetchone()[0]
print(f"Loaded {count:,} rows into {DB_PATH} (table: transactions)")

conn.close()
