# 01 — SQL

Exhaustive learning path for SQL in data science, analytics engineering, and ML pipelines.

---

## 01 — SELECT, Filtering & Sorting
SELECT columns, WHERE clauses, LIKE, IN, BETWEEN, IS NULL, ORDER BY, LIMIT/OFFSET.
- https://mode.com/sql-tutorial/sql-select-statement/
- https://www.sqlzoo.net/wiki/SELECT_basics

## 02 — Aggregations & GROUP BY
COUNT, SUM, AVG, MIN, MAX, COUNT(DISTINCT); GROUP BY with multiple columns; HAVING vs WHERE.
- https://mode.com/sql-tutorial/sql-aggregate-functions/

## 03 — JOINs
INNER, LEFT, RIGHT, FULL OUTER, CROSS, SELF joins; choosing join type; join on multiple columns; non-equi joins.
- https://joins.spathon.com/
- https://mode.com/sql-tutorial/sql-joins/

## 04 — Subqueries
Scalar subqueries in SELECT; correlated vs non-correlated subqueries; EXISTS / NOT EXISTS; subquery in FROM (derived table).
- https://mode.com/sql-tutorial/sql-subqueries/

## 05 — CTEs (Common Table Expressions)
WITH clause; chaining multiple CTEs; readability over nested subqueries; materialized vs inline.
- https://www.postgresql.org/docs/current/queries-with.html
- https://mode.com/sql-tutorial/sql-with-statement/

## 06 — Recursive CTEs
WITH RECURSIVE for hierarchies (org charts, trees); anchor + recursive member; cycle detection.
- https://www.postgresql.org/docs/current/queries-with.html#QUERIES-WITH-RECURSIVE

## 07 — Window Functions: Ranking
ROW_NUMBER, RANK, DENSE_RANK, NTILE; OVER(PARTITION BY … ORDER BY …); top-N per group pattern.
- https://mode.com/sql-tutorial/sql-window-functions/
- https://www.postgresql.org/docs/current/functions-window.html

## 08 — Window Functions: Running Totals & Moving Averages
SUM/AVG with frames (ROWS BETWEEN, RANGE BETWEEN); rolling 7-day averages; cumulative sums.
- https://www.postgresql.org/docs/current/sql-expressions.html#SYNTAX-WINDOW-FUNCTIONS

## 09 — Window Functions: LAG, LEAD, FIRST_VALUE, LAST_VALUE
Access adjacent rows; detect state changes; period-over-period comparisons; sessionization with LAG.
- https://www.postgresql.org/docs/current/functions-window.html

## 10 — Set Operations
UNION, UNION ALL, INTERSECT, EXCEPT; deduplication semantics; performance differences.
- https://mode.com/sql-tutorial/sql-set-operations/

## 11 — NULL Handling
COALESCE, NULLIF, IS NULL, IS NOT NULL; NULL propagation in expressions; NULLs in aggregations and JOINs.
- https://www.postgresql.org/docs/current/functions-conditional.html

## 12 — CASE Expressions
Simple and searched CASE; conditional aggregation (SUM(CASE WHEN …)); pivot patterns.
- https://mode.com/sql-tutorial/sql-case/

## 13 — Date & Time Operations
DATE_TRUNC, EXTRACT, DATE_ADD, DATEDIFF, NOW(), CURRENT_DATE; time zone casting; epoch conversion.
- https://mode.com/sql-tutorial/sql-datetime-functions/
- https://www.postgresql.org/docs/current/functions-datetime.html

## 14 — String Functions
CONCAT, SUBSTRING, TRIM, UPPER, LOWER, REPLACE, SPLIT_PART, REGEXP_REPLACE, ILIKE.
- https://www.postgresql.org/docs/current/functions-string.html

## 15 — JSON & JSONB Operations (PostgreSQL)
->, ->>, #> operators; json_each, json_array_elements; JSONB indexing; jsonb_path_query.
- https://www.postgresql.org/docs/current/functions-json.html

## 16 — Array Operations
array_agg, unnest, array_length; contains (@>); overlap (&&); array in WHERE clause.
- https://www.postgresql.org/docs/current/functions-array.html

## 17 — Indexes & Query Planning
B-tree, hash, GIN, BRIN; EXPLAIN / EXPLAIN ANALYZE; seq scan vs index scan; covering indexes; partial indexes.
- https://use-the-index-luke.com/
- https://www.postgresql.org/docs/current/using-explain.html

## 18 — Query Optimization
Predicate pushdown; avoid SELECT *; avoid functions on indexed columns in WHERE; join order; statistics.
- https://www.postgresql.org/docs/current/performance-tips.html
- https://sqlperformance.com/

## 19 — Transactions & ACID
BEGIN/COMMIT/ROLLBACK; isolation levels (READ COMMITTED, REPEATABLE READ, SERIALIZABLE); deadlocks; SAVEPOINT.
- https://www.postgresql.org/docs/current/transaction-iso.html

## 20 — Views & Materialized Views
CREATE VIEW for query abstraction; CREATE MATERIALIZED VIEW with REFRESH; when to materialize vs inline.
- https://www.postgresql.org/docs/current/sql-createview.html

## 21 — Analytical Pattern: Cohort Analysis
Assign users to cohorts by signup period; track retention week-over-week with DATE_TRUNC + window functions.
- https://towardsdatascience.com/a-step-by-step-guide-to-cohort-analysis-in-sql

## 22 — Analytical Pattern: Funnel Analysis
Multi-step conversion; COUNT(DISTINCT) per step; self-JOIN or conditional aggregation; drop-off rates.
- https://towardsdatascience.com/sql-tricks-for-data-scientists-53b4b4bd2e7e

## 23 — Analytical Pattern: Sessionization
Session IDs via LAG + CASE on timestamp gaps; session duration; event count per session.
- https://medium.com/@dpraveen/sessionization-in-sql-69ed62ceb0d9

## 24 — Pivoting & Unpivoting
CASE-based pivot; CROSSTAB() in PostgreSQL; PIVOT/UNPIVOT in BigQuery / Snowflake.
- https://www.postgresql.org/docs/current/tablefunc.html
- https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#pivot_operator

## 25 — BigQuery / Snowflake Dialect
QUALIFY clause (BigQuery); ARRAY_AGG with ORDER BY; FLATTEN; LATERAL JOIN; partitioned table scanning.
- https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax
- https://docs.snowflake.com/en/sql-reference.html

## 26 — DuckDB: In-Process SQL Analytics
Run SQL on Parquet / CSV directly in Python; in-memory analytics without a server; ASOF join.
- https://duckdb.org/docs/api/python/overview.html

## 27 — SQL for ML Workflows
Feature engineering in SQL; train/test splits with MOD(); label generation; connecting to pandas/SQLAlchemy.
- https://docs.sqlalchemy.org/en/20/core/connections.html
- https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html
