-- V-feature correlation with Class (fraud vs. not)
-- SQLite has no native CORR() aggregate, so Pearson's r is computed manually
-- per column using the standard formula:
--   r = (n*SUM(xy) - SUM(x)*SUM(y)) / SQRT((n*SUM(x^2)-SUM(x)^2)*(n*SUM(y^2)-SUM(y)^2))
-- Example for V14 (repeat pattern for V1..V28, or use run_correlation.py which
-- loops all 28 automatically and exports a ranked CSV):

WITH stats AS (
    SELECT
        COUNT(*)                              AS n,
        SUM("V14")                            AS sum_x,
        SUM("Class")                          AS sum_y,
        SUM("V14" * "Class")                  AS sum_xy,
        SUM("V14" * "V14")                    AS sum_x2,
        SUM("Class" * "Class")                AS sum_y2
    FROM transactions
)
SELECT
    'V14' AS feature,
    (n * sum_xy - sum_x * sum_y) /
    (SQRT(n * sum_x2 - sum_x * sum_x) * SQRT(n * sum_y2 - sum_y * sum_y)) AS pearson_r
FROM stats;
