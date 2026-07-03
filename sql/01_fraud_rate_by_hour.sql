-- Fraud rate by hour of day
-- Derives Hour from Time (seconds since first transaction, wraps at 24h)
-- Business question: does fraud cluster at certain hours?

SELECT
    CAST((CAST("Time" AS INTEGER) / 3600) % 24 AS INTEGER) AS hour_of_day,
    COUNT(*)                                                AS total_transactions,
    SUM("Class")                                            AS fraud_count,
    ROUND(100.0 * SUM("Class") / COUNT(*), 4)                AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;
