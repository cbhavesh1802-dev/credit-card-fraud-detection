-- Fraud rate by transaction amount band
-- Business question: are certain amount ranges disproportionately fraudulent?

SELECT
    CASE
        WHEN "Amount" < 10   THEN '01: 0-10'
        WHEN "Amount" < 50   THEN '02: 10-50'
        WHEN "Amount" < 200  THEN '03: 50-200'
        WHEN "Amount" < 500  THEN '04: 200-500'
        WHEN "Amount" < 1000 THEN '05: 500-1000'
        ELSE                      '06: 1000+'
    END                                          AS amount_band,
    COUNT(*)                                     AS total_transactions,
    SUM("Class")                                 AS fraud_count,
    ROUND(100.0 * SUM("Class") / COUNT(*), 4)     AS fraud_rate_pct,
    ROUND(AVG("Amount"), 2)                       AS avg_amount
FROM transactions
GROUP BY amount_band
ORDER BY amount_band;
