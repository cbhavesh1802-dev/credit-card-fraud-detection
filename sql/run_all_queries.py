import sqlite3
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "fraud_analytics.db")
SQL_DIR = SCRIPT_DIR
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)


def run_sql_file(path, out_name):
    with open(path) as f:
        query = f.read()
    df = pd.read_sql_query(query, conn)
    out_path = os.path.join(OUT_DIR, out_name)
    df.to_csv(out_path, index=False)
    print(f"{out_name}: {len(df)} rows -> {out_path}")
    return df


run_sql_file(os.path.join(SQL_DIR, "01_fraud_rate_by_hour.sql"), "fraud_rate_by_hour.csv")
run_sql_file(os.path.join(SQL_DIR, "02_fraud_rate_by_amount_band.sql"), "fraud_rate_by_amount_band.csv")
run_sql_file(os.path.join(SQL_DIR, "03_rolling_fraud_rate.sql"), "rolling_fraud_rate.csv")

rows = []
for i in range(1, 29):
    col = f"V{i}"
    q = f"""
    WITH stats AS (
        SELECT
            COUNT(*)                     AS n,
            SUM("{col}")                 AS sum_x,
            SUM("Class")                 AS sum_y,
            SUM("{col}" * "Class")       AS sum_xy,
            SUM("{col}" * "{col}")       AS sum_x2,
            SUM("Class" * "Class")       AS sum_y2
        FROM transactions
    )
    SELECT
        '{col}' AS feature,
        (n * sum_xy - sum_x * sum_y) /
        (SQRT(n * sum_x2 - sum_x * sum_x) * SQRT(n * sum_y2 - sum_y * sum_y)) AS pearson_r
    FROM stats;
    """
    r = conn.execute(q).fetchone()
    rows.append({"feature": r[0], "pearson_r": r[1]})

corr_df = pd.DataFrame(rows)
corr_df["abs_r"] = corr_df["pearson_r"].abs()
corr_df = corr_df.sort_values("abs_r", ascending=False).drop(columns="abs_r")
corr_out = os.path.join(OUT_DIR, "feature_correlation_ranked.csv")
corr_df.to_csv(corr_out, index=False)
print(f"feature_correlation_ranked.csv: {len(corr_df)} rows -> {corr_out}")

imbalance_df = pd.read_sql_query(
    """
    SELECT
        "Class",
        COUNT(*) AS transaction_count,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 4) AS pct_of_total
    FROM transactions
    GROUP BY "Class";
    """,
    conn,
)
imbalance_out = os.path.join(OUT_DIR, "class_imbalance_summary.csv")
imbalance_df.to_csv(imbalance_out, index=False)
print(f"class_imbalance_summary.csv: {len(imbalance_df)} rows -> {imbalance_out}")

conn.close()
print("\nAll query outputs written to:", OUT_DIR)
