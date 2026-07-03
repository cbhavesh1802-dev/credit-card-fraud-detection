-- Rolling fraud rate trend over time (window function)
-- Buckets transactions into 1-hour windows across the full time span, then
-- computes a 6-bucket rolling average fraud rate -- smooths noise, surfaces trend.

WITH hourly_buckets AS (
    SELECT
        CAST("Time" AS INTEGER) / 3600                AS hour_bucket,
        COUNT(*)                                        AS total_transactions,
        SUM("Class")                                    AS fraud_count,
        ROUND(100.0 * SUM("Class") / COUNT(*), 4)       AS fraud_rate_pct
    FROM transactions
    GROUP BY hour_bucket
)
SELECT
    hour_bucket,
    total_transactions,
    fraud_count,
    fraud_rate_pct,
    ROUND(
        AVG(fraud_rate_pct) OVER (
            ORDER BY hour_bucket
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ), 4
    ) AS rolling_6h_avg_fraud_rate_pct
FROM hourly_buckets
ORDER BY hour_bucket;
