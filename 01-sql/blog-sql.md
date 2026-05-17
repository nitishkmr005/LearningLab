# SQL for Data Science: From Intermediate Patterns to Production-Grade Snowflake

*Everything a data scientist needs to write fast, correct, interview-ready SQL — covering window functions, CTEs, analytical patterns, Snowflake-specific features, and query optimization.*

---

## Table of Contents

1. [Why SQL Still Dominates Data Science](#1-why-sql-still-dominates-data-science)
2. [How SQL Actually Executes](#2-how-sql-actually-executes)
3. [JOINs: Every Type, Every Gotcha](#3-joins-every-type-every-gotcha)
4. [Subqueries, CTEs & Recursive CTEs](#4-subqueries-ctes--recursive-ctes)
5. [Window Functions: The Data Scientist's Superpower](#5-window-functions-the-data-scientists-superpower)
6. [Set Operations](#6-set-operations)
7. [NULL Handling & CASE Expressions](#7-null-handling--case-expressions)
8. [Date/Time & String Functions](#8-datetime--string-functions)
9. [Semi-structured Data: JSON, Arrays & Snowflake VARIANT](#9-semi-structured-data-json-arrays--snowflake-variant)
10. [Indexes & Query Planning](#10-indexes--query-planning)
11. [Query Optimization](#11-query-optimization)
12. [Views & Materialized Views](#12-views--materialized-views)
13. [Analytical Patterns](#13-analytical-patterns)
14. [Snowflake Power Features](#14-snowflake-power-features)
15. [DuckDB: In-Process Analytics](#15-duckdb-in-process-analytics)
16. [SQL for ML Workflows](#16-sql-for-ml-workflows)
17. [Comparison Tables](#17-comparison-tables)
18. [Interview Prep: 10 Worked Problems](#18-interview-prep-10-worked-problems)
19. [References](#19-references)

---

## 1. Why SQL Still Dominates Data Science

Fifteen years after "SQL is dead" headlines, SQL is more central to data science than ever. Python consumes data that SQL has already cleaned, aggregated, and shaped. Feature stores, ML pipelines, and experiment tracking tables all sit in SQL engines. The reason is architectural: the data is in the database, and moving it out to Python just to do a GROUP BY is expensive and fragile.

For data scientists specifically, SQL proficiency separates people who prototype from people who ship. A model trained on features built in Python notebooks degrades silently when the production feature pipeline — written in SQL — computes something slightly different. Knowing SQL at a deep level means you can own both sides of that boundary. It also means you pass the 45-minute SQL screen at every FAANG company, where window functions and CTEs appear in 40% of hard-level questions ([MindfulTechie, 2024](https://medium.com/@aicoders/master-sql-window-functions-and-ctes-12-real-data-engineering-interview-questions-with-code-9b42f37c1db1)).

This blog skips SELECT, WHERE, and GROUP BY — those are assumed. It starts where interviews get hard: JOINs with traps, CTEs, window functions, Snowflake-specific syntax, and the analytical patterns (cohort, funnel, sessionization) that show up in every data science take-home.

> 📚 **Go deeper**: [Mode SQL Tutorial](https://mode.com/sql-tutorial/) — the best free interactive SQL course covering every concept in this blog with real datasets.

---

## 2. How SQL Actually Executes

Understanding the logical execution order of SQL prevents a whole class of bugs and explains why certain optimizations work. SQL is *declarative* — you say what you want, not how to get it — but the engine processes clauses in a strict order that has nothing to do with how you write them.

### 2.1 Logical Clause Order

```
1. FROM + JOINs       — identify which rows exist
2. WHERE              — filter raw rows (before any aggregation)
3. GROUP BY           — form groups
4. HAVING             — filter groups (after aggregation)
5. WINDOW             — compute window function results
6. QUALIFY            — filter window function results (Snowflake only)
7. SELECT             — project columns, apply aliases
8. DISTINCT           — deduplicate
9. ORDER BY           — sort (aliases from SELECT are now visible)
10. LIMIT / OFFSET    — paginate
```

The most common bugs come from violating this order: using a SELECT alias in a WHERE clause (illegal — SELECT hasn't run yet), or filtering on a window function result in WHERE instead of a subquery.

### 2.2 Snowflake Execution Engine

Snowflake splits query processing across two layers ([Pradhan, 2024](https://medium.com/snowflake/anatomy-of-a-snowflake-query-a-deep-dive-into-the-execution-engine-ca9061022c47)):

```
SQL Text
  → Cloud Services Layer (no credits consumed)
      ├── Parse & Validate (syntax, schema, permissions)
      ├── Cost-Based Optimizer (join reordering, predicate pushdown)
      └── Micro-Partition Pruning (skip files based on min/max metadata)
  → Virtual Warehouse Layer (credits consumed)
      └── Physical Execution DAG (columnar, vectorized, parallel)
```

**Micro-partition pruning** is the single most important performance concept in Snowflake. Every table is automatically divided into immutable micro-partitions of ~16 MB compressed. Snowflake stores min/max and null counts for every column in every micro-partition. A filter like `WHERE sale_date >= '2024-01-01'` allows the optimizer to skip entire files without reading them — dramatically reducing I/O before a single row is processed.

> 🎯 **Interview prep**: "What is the logical order of SQL clause execution?" is asked at Google, Meta, and Snowflake. The trap: aliases from SELECT are not visible in WHERE. They become visible only in ORDER BY (because ORDER BY runs after SELECT).

> 🏭 **Production note**: In Snowflake, your queries run against the Cloud Services layer for free during planning. You're only billed when the Virtual Warehouse starts scanning data. If the query profile shows 99% of partitions pruned, you have a well-written query.

**Resources**
- [Anatomy of a Snowflake Query](https://medium.com/snowflake/anatomy-of-a-snowflake-query-a-deep-dive-into-the-execution-engine-ca9061022c47) — deep dive into parsing, optimization, and vectorized execution
- [Snowflake Query Profile docs](https://docs.snowflake.com/en/user-guide/ui-query-profile) — how to read execution plans in Snowsight

---

## 3. JOINs: Every Type, Every Gotcha

JOINs are where data scientists lose hours to subtle bugs. The syntax is easy; the semantics are not. The most common mistake is treating a LEFT JOIN like an INNER JOIN — adding a WHERE filter on the right-table column silently converts it back to an inner join.

### 3.1 JOIN Types

```sql
-- INNER JOIN: only rows where condition matches in BOTH tables
SELECT o.order_id, c.name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id;

-- LEFT JOIN: all rows from left, NULLs for non-matching right rows
SELECT c.name, COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.name;                  -- includes customers with 0 orders

-- FULL OUTER JOIN: all rows from both, NULLs where no match
SELECT a.id, b.id
FROM table_a a
FULL OUTER JOIN table_b b ON a.id = b.id;

-- CROSS JOIN: cartesian product — every row paired with every row
-- Use carefully: N×M rows. Useful for generating date × product combinations.
SELECT d.date, p.product_id
FROM date_spine d
CROSS JOIN products p;

-- SELF JOIN: join a table to itself — org charts, consecutive events
SELECT e.name, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

### 3.2 Non-Equi Joins

Non-equi joins use comparison operators instead of equality. They generate large intermediate results and are expensive — but some problems require them.

```sql
-- Find all salary ranges an employee falls into
SELECT e.name, e.salary, r.grade
FROM employees e
JOIN salary_ranges r ON e.salary BETWEEN r.min_sal AND r.max_sal;
```

**Snowflake ASOF JOIN**: Snowflake has a dedicated ASOF join for time-series lookups that is far more efficient than a range join or a correlated subquery ([Greybeam, 2024](https://greybeam.medium.com/snowflake-query-optimization-7-tips-for-faster-queries-4701337e595b)):

```sql
-- Get the most recent exchange rate at or before each transaction
SELECT t.transaction_id, t.amount, r.rate
FROM transactions t
ASOF JOIN fx_rates r
  MATCH_CONDITION (t.txn_time >= r.rate_time)
  ON t.currency = r.currency;
```

> 🎯 **Interview prep**: "You have a LEFT JOIN and your result has fewer rows than the left table. What happened?" — Answer: a WHERE clause on the right table's column converted it to an INNER JOIN. Move that filter into the ON clause or use `WHERE right_col IS NULL` for anti-join patterns.

> 🏭 **Production note**: Disjunctive join conditions (using OR) force a Cartesian product in Snowflake. One team reduced query time from 4m 36s to 6.7s (40× speedup) by rewriting `ON a.id = b.id OR a.alt_id = b.id` as two UNION ALL'd equi-joins.

**Resources**
- [Visual JOIN explanations](https://joins.spathon.com/) — interactive Venn diagram for every join type
- [Snowflake ASOF JOIN docs](https://docs.snowflake.com/en/sql-reference/constructs/asof-join) — time-series merge joins

---

## 4. Subqueries, CTEs & Recursive CTEs

Every complex SQL query can be written three ways: nested subqueries, CTEs, or a single flat query. The choice affects readability, debuggability, and sometimes performance.

### 4.1 Subqueries

Subqueries come in three flavors. A **scalar subquery** returns a single value and can appear anywhere an expression is valid:

```sql
SELECT name, salary,
       salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees;
```

A **correlated subquery** references the outer query — it runs once per outer row, making it O(N) expensive:

```sql
-- For each employee, count how many others earn more (expensive for large tables)
SELECT name, salary,
       (SELECT COUNT(*) FROM employees e2
        WHERE e2.salary > e1.salary) AS rank
FROM employees e1;
```

An **uncorrelated subquery in FROM** (derived table) runs once:

```sql
SELECT dept, avg_salary
FROM (
  SELECT department AS dept, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY department
) dept_stats
WHERE avg_salary > 80000;
```

### 4.2 CTEs — Common Table Expressions

CTEs replace nested subqueries with named, reusable building blocks. They don't change performance in most databases (a CTE is inlined by the optimizer), but they dramatically improve readability — and readability is a real production concern when six engineers will debug this query at 2am.

```sql
WITH
  -- Step 1: get monthly revenue per user
  monthly_rev AS (
    SELECT user_id,
           DATE_TRUNC('month', order_date) AS month,
           SUM(amount)                      AS revenue
    FROM orders
    GROUP BY 1, 2
  ),
  -- Step 2: compute month-over-month growth
  mom_growth AS (
    SELECT user_id, month, revenue,
           LAG(revenue) OVER (PARTITION BY user_id ORDER BY month) AS prev_revenue
    FROM monthly_rev
  )
SELECT user_id, month,
       ROUND(100.0 * (revenue - prev_revenue) / NULLIF(prev_revenue, 0), 2) AS pct_growth
FROM mom_growth
WHERE prev_revenue IS NOT NULL;
```

> 🎯 **Interview prep**: Interviewers ask "when would you use a CTE over a subquery?" The answer: always, for readability. The follow-up is "does a CTE always improve performance?" — No. In PostgreSQL, CTEs are optimization fences by default (pre-v12), meaning the optimizer can't push predicates through them. In Snowflake and BigQuery, CTEs are inlined and optimized normally.

### 4.3 Recursive CTEs

Recursive CTEs solve hierarchical problems — org charts, bill-of-materials, graph traversal — that cannot be expressed in flat SQL. The structure is always the same: an **anchor member** (base case) UNION ALL'd with a **recursive member** that references the CTE itself.

```sql
-- Org chart: find all reports under a given manager
WITH RECURSIVE org_tree AS (
  -- Anchor: start with the manager
  SELECT id, name, manager_id, 0 AS depth
  FROM employees
  WHERE id = 42                    -- root node

  UNION ALL

  -- Recursive: find direct reports of current level
  SELECT e.id, e.name, e.manager_id, t.depth + 1
  FROM employees e
  INNER JOIN org_tree t ON e.manager_id = t.id
  WHERE t.depth < 10               -- cycle guard: stop after 10 levels
)
SELECT id, name, depth
FROM org_tree
ORDER BY depth, name;
```

> 🏭 **Production note**: Always include a depth limit or a visited-node check in recursive CTEs. A bad foreign key or a circular reference in the data will cause infinite recursion and crash the query. Snowflake has a `MAX_RECURSION` parameter (default 100) as a safety net, but the depth guard in SQL is the right defense.

**Resources**
- [PostgreSQL WITH clause docs](https://www.postgresql.org/docs/current/queries-with.html) — full recursive CTE specification
- [Advanced SQL for Data Engineering](https://mayursurani.medium.com/advanced-sql-for-data-engineering-2025-master-window-functions-ctes-explain-plans-materialized-f729a29cb120) — practical CTEs with EXPLAIN integration

---

## 5. Window Functions: The Data Scientist's Superpower

Window functions are the single biggest skill gap between intermediate and advanced SQL users. They solve problems that would otherwise require self-joins, correlated subqueries, or round-trips to Python — and they do it without collapsing rows the way GROUP BY does. Understanding them deeply will get you through the hardest interview questions and unlock analytical patterns that are otherwise painful to write.

### 5.1 The OVER Clause Anatomy

Every window function follows the same structure:

```
function_name([expression]) OVER (
  [PARTITION BY partition_columns]   -- divide rows into independent windows
  [ORDER BY sort_columns]            -- define row order within each window
  [frame_clause]                     -- define which rows are "in scope"
)
```

PARTITION BY and ORDER BY are both optional. Without PARTITION BY, the entire result set is one window. Without ORDER BY, the frame defaults to all rows in the partition.

### 5.2 Ranking Functions

Ranking functions assign a position to each row within its partition. The three main ones differ only in how they handle ties:

```sql
SELECT name, department, salary,
  ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num,
  RANK()       OVER (PARTITION BY department ORDER BY salary DESC) AS rnk,
  DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rnk,
  NTILE(4)     OVER (PARTITION BY department ORDER BY salary DESC) AS quartile
FROM employees;
```

Given salaries [100, 90, 90, 80] in a department:

| salary | ROW_NUMBER | RANK | DENSE_RANK |
|--------|-----------|------|------------|
| 100    | 1         | 1    | 1          |
| 90     | 2         | 2    | 2          |
| 90     | 3         | 2    | 2          |
| 80     | 4         | 4    | 3          |

**The classic interview pattern** — top-N per group:

```sql
-- Top 3 earners per department (handles ties with DENSE_RANK)
WITH ranked AS (
  SELECT name, department, salary,
         DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dr
  FROM employees
)
SELECT name, department, salary
FROM ranked
WHERE dr <= 3;
```

In Snowflake, QUALIFY eliminates the subquery entirely (see Section 14.1).

### 5.3 Frame Clauses: Running Totals & Moving Averages

The frame clause defines which rows the aggregate function sees. This is where most people get confused.

```
ROWS BETWEEN <start> AND <end>

start/end options:
  UNBOUNDED PRECEDING   — from the first row in the partition
  N PRECEDING           — N rows before the current row
  CURRENT ROW           — the current row
  N FOLLOWING           — N rows after the current row
  UNBOUNDED FOLLOWING   — to the last row in the partition
```

```sql
SELECT
  order_date,
  revenue,

  -- Cumulative sum (all rows from start through current)
  SUM(revenue) OVER (
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS cumulative_revenue,

  -- 7-day moving average (current row + 6 preceding)
  AVG(revenue) OVER (
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d_avg,

  -- Running total that resets per month
  SUM(revenue) OVER (
    PARTITION BY DATE_TRUNC('month', order_date)
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS monthly_cumulative
FROM daily_revenue;
```

**ROWS vs RANGE**: `ROWS BETWEEN` counts physical rows. `RANGE BETWEEN` groups rows with identical ORDER BY values into the same logical frame. For time-series work, `ROWS BETWEEN` almost always behaves more predictably.

> 🏭 **Production note**: Window functions with large frames over unbounded partitions can spill to disk in Snowflake, just like sorts. If you're computing cumulative sums over 100M rows, materialize an intermediate table rather than stacking five window functions in a single SELECT.

### 5.4 LAG, LEAD, FIRST_VALUE, LAST_VALUE

These functions give you access to other rows in the window without a self-join:

```sql
SELECT
  user_id,
  event_date,
  event_type,

  -- Previous event for this user
  LAG(event_type, 1)  OVER (PARTITION BY user_id ORDER BY event_date) AS prev_event,

  -- Next event for this user
  LEAD(event_type, 1) OVER (PARTITION BY user_id ORDER BY event_date) AS next_event,

  -- First event the user ever performed
  FIRST_VALUE(event_type) OVER (
    PARTITION BY user_id ORDER BY event_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS first_ever_event,

  -- Days since last purchase
  DATEDIFF('day',
    LAG(event_date) OVER (PARTITION BY user_id ORDER BY event_date),
    event_date
  ) AS days_since_last_event

FROM user_events;
```

**Period-over-period comparison** (the most common use of LAG):

```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month, SUM(revenue) AS rev
  FROM orders GROUP BY 1
)
SELECT month, rev,
       LAG(rev) OVER (ORDER BY month)                          AS prev_month_rev,
       ROUND(100.0 * (rev - LAG(rev) OVER (ORDER BY month))
             / NULLIF(LAG(rev) OVER (ORDER BY month), 0), 1)  AS pct_change
FROM monthly;
```

> 🎯 **Interview prep**: "Why does LAG return NULL for the first row?" — Because there is no previous row. Always handle with `COALESCE(LAG(col) OVER (...), 0)` or filter out NULLs downstream. Forgetting this causes off-by-one errors in growth rate calculations.

**Resources**
- [12 Real Window Function Interview Questions](https://medium.com/@aicoders/master-sql-window-functions-and-ctes-12-real-data-engineering-interview-questions-with-code-9b42f37c1db1) — FAANG-sourced problems with full solutions
- [Snowflake Analytic Functions](https://docs.snowflake.com/en/sql-reference/functions-analytic) — complete reference including approximate analytics (HyperLogLog, t-Digest)
- [pipeline2insights: Advanced SQL for Interviews](https://pipeline2insights.substack.com/p/week-332-advanced-sql-concepts-for) — structured three-step problem-solving approach

---

## 6. Set Operations

Set operations combine results from multiple SELECT statements. They're simple in concept but have performance traps.

```sql
-- UNION: combines and removes duplicates (expensive — requires sort/hash)
SELECT user_id FROM web_users
UNION
SELECT user_id FROM mobile_users;

-- UNION ALL: combines without deduplication (always prefer unless you need dedup)
SELECT user_id, 'web' AS source FROM web_users
UNION ALL
SELECT user_id, 'mobile' AS source FROM mobile_users;

-- INTERSECT: rows present in BOTH (removes duplicates)
SELECT user_id FROM premium_users
INTERSECT
SELECT user_id FROM active_last_30d;

-- EXCEPT: rows in first but NOT in second (anti-join alternative)
SELECT user_id FROM all_users
EXCEPT
SELECT user_id FROM churned_users;
```

**Performance rule**: `UNION ALL` is always faster than `UNION` because it skips sorting. Use `UNION` only when duplicate elimination is semantically required.

> 🎯 **Interview prep**: "What's the difference between UNION and UNION ALL?" is a basic question. The follow-up that trips people: "When would EXCEPT be worse than a LEFT JOIN anti-pattern?" Answer: EXCEPT materializes both sides; a `LEFT JOIN ... WHERE right_id IS NULL` can be more selective when the right side is large.

**Resources**
- [Mode SQL Set Operations Tutorial](https://mode.com/sql-tutorial/sql-set-operations/) — visual explanation with examples

---

## 7. NULL Handling & CASE Expressions

NULLs are the most dangerous values in SQL because they propagate silently. `NULL + 1 = NULL`. `NULL = NULL` is FALSE (not TRUE). `COUNT(col)` skips NULLs but `COUNT(*)` doesn't. Understanding this prevents hours of debugging.

### 7.1 NULL Functions

```sql
-- COALESCE: returns first non-NULL value in the list
SELECT COALESCE(phone, email, 'no_contact') AS contact FROM users;

-- NULLIF: returns NULL if two values are equal (prevents division by zero)
SELECT revenue / NULLIF(costs, 0) AS margin FROM financials;

-- IS NULL / IS NOT NULL: use these, never = NULL
SELECT * FROM orders WHERE shipped_date IS NULL;

-- NULLs in aggregations: COUNT(col) ignores NULLs
SELECT AVG(score)         AS avg_score_excl_nulls,   -- ignores NULL scores
       AVG(COALESCE(score, 0)) AS avg_score_incl_nulls   -- treats NULL as 0
FROM survey_responses;
```

### 7.2 CASE Expressions

CASE is the SQL equivalent of if/elif/else. Its most powerful form is **conditional aggregation** — pivoting or computing multiple metrics in a single pass:

```sql
-- Conditional aggregation: compute multiple metrics in one scan
SELECT
  DATE_TRUNC('month', order_date) AS month,
  COUNT(*)                                              AS total_orders,
  COUNT(CASE WHEN channel = 'web'    THEN 1 END)       AS web_orders,
  COUNT(CASE WHEN channel = 'mobile' THEN 1 END)       AS mobile_orders,
  SUM(CASE WHEN channel = 'web'    THEN revenue ELSE 0 END) AS web_revenue,
  SUM(CASE WHEN status = 'refunded' THEN amount ELSE 0 END) AS refund_amount
FROM orders
GROUP BY 1
ORDER BY 1;
```

> 🏭 **Production note**: Conditional aggregation replaces multiple self-joins that each do a full table scan. Replacing three self-joins with one CASE-in-aggregate query is one of the most common query rewrites in data engineering.

**Resources**
- [PostgreSQL Conditional Expressions](https://www.postgresql.org/docs/current/functions-conditional.html) — CASE, COALESCE, NULLIF, GREATEST, LEAST

---

## 8. Date/Time & String Functions

These functions are database-specific enough that knowing the Snowflake variants is essential for practical work.

### 8.1 Date/Time (Snowflake dialect)

```sql
-- Truncate to period boundary
DATE_TRUNC('month', '2024-03-15'::date)       -- → 2024-03-01
DATE_TRUNC('week', event_ts)                  -- → Monday of that week

-- Extract components
EXTRACT(year  FROM order_date)                -- → 2024
EXTRACT(dow   FROM order_date)                -- → 0=Sun ... 6=Sat
EXTRACT(epoch FROM current_timestamp())       -- → Unix timestamp (seconds)

-- Arithmetic
DATEADD(day, 7, order_date)                   -- add 7 days
DATEDIFF(day, start_date, end_date)           -- → integer days between
DATEDIFF(month, '2023-01-01', '2024-06-01')  -- → 17

-- Current values
CURRENT_DATE()      -- today's date (no time)
CURRENT_TIMESTAMP() -- full timestamp with timezone
CONVERT_TIMEZONE('America/New_York', event_ts)  -- timezone conversion

-- Time zone casting (critical for global products)
event_ts::TIMESTAMP_NTZ      -- strip timezone
event_ts::TIMESTAMP_TZ       -- keep timezone info
event_ts AT TIME ZONE 'UTC'  -- convert to UTC
```

### 8.2 String Functions

```sql
-- Cleaning and extraction
TRIM(BOTH ' ' FROM name)                   -- remove leading/trailing spaces
UPPER(email), LOWER(email)                 -- normalize case
SUBSTRING(text, 1, 100)                   -- first 100 chars
SPLIT_PART(email, '@', 2)                 -- extract domain
REGEXP_REPLACE(phone, '[^0-9]', '')       -- keep only digits
REGEXP_SUBSTR(url, 'utm_source=([^&]+)')  -- extract query param

-- Snowflake ILIKE (case-insensitive LIKE)
SELECT * FROM events WHERE source ILIKE '%google%';

-- Concatenation
CONCAT(first_name, ' ', last_name)
first_name || ' ' || last_name            -- ANSI syntax
```

> 🏭 **Production note**: In Snowflake, REGEXP functions use PCRE syntax, not POSIX. Patterns that work in PostgreSQL may need rewriting. Always test regex patterns against a sample before deploying to production transformations.

**Resources**
- [Snowflake Date/Time Functions](https://docs.snowflake.com/en/sql-reference/functions-date-time) — complete reference including DATEADD, DATEDIFF, DATE_TRUNC
- [Snowflake String Functions](https://docs.snowflake.com/en/sql-reference/functions-string) — full regex support and string manipulation

---

## 9. Semi-structured Data: JSON, Arrays & Snowflake VARIANT

Modern data pipelines ingest raw JSON from APIs, webhooks, and event streams. Both PostgreSQL and Snowflake have first-class support, but their approaches differ significantly.

### 9.1 PostgreSQL JSON/JSONB

```sql
-- -> returns JSON, ->> returns text
SELECT
  data -> 'user' ->> 'email'           AS email,    -- nested access
  data -> 'items' -> 0 ->> 'product'   AS first_product,  -- array index
  jsonb_array_length(data -> 'items')  AS item_count
FROM events
WHERE data @> '{"event_type": "purchase"}';  -- JSONB containment

-- Expand JSON array to rows
SELECT id, item
FROM events,
     jsonb_array_elements(data -> 'items') AS item;
```

### 9.2 Snowflake VARIANT & FLATTEN

Snowflake stores JSON natively in a **VARIANT** column using colon notation for traversal ([Snowflake docs](https://docs.snowflake.com/en/user-guide/querying-semistructured)):

```sql
-- Colon syntax for field access; cast with ::
SELECT
  src:user_id::STRING                AS user_id,
  src:purchase.total_amount::NUMBER  AS amount,
  src:purchase.items[0].sku::STRING  AS first_sku
FROM raw_events;

-- FLATTEN: explode an array into rows
-- Output columns: SEQ, KEY, PATH, INDEX, VALUE, THIS
SELECT
  e.src:user_id::STRING          AS user_id,
  f.index                        AS item_position,
  f.value:sku::STRING            AS sku,
  f.value:quantity::INTEGER      AS quantity,
  f.value:price::FLOAT           AS price
FROM raw_events e,
     LATERAL FLATTEN(INPUT => e.src:purchase.items) f;
```

**Chaining FLATTEN for deeply nested structures**:

```sql
-- Two levels: contacts → each contact → business addresses
SELECT
  p.id,
  f.value:type::STRING    AS contact_type,
  f1.value::STRING        AS business_addr
FROM persons p,
  LATERAL FLATTEN(INPUT => p.profile, PATH => 'contacts') f,
  LATERAL FLATTEN(INPUT => f.value:business) f1;
```

The FLATTEN output columns you need to know:
- `VALUE` — the element being exploded
- `INDEX` — position in the array (NULL for objects)
- `KEY` — the object key (NULL for arrays)
- `PATH` — full dot-path to the element

> 🎯 **Interview prep**: "How would you count the number of items in a nested JSON array in Snowflake?" — Use `ARRAY_SIZE(src:items)` for a scalar result, or `LATERAL FLATTEN` to aggregate per-element. Knowing both options shows depth.

**Resources**
- [Snowflake Semi-structured Data docs](https://docs.snowflake.com/en/user-guide/querying-semistructured) — VARIANT, FLATTEN, PARSE_JSON, GET_PATH
- [PostgreSQL JSON Functions](https://www.postgresql.org/docs/current/functions-json.html) — full JSONB operator reference

---

## 10. Indexes & Query Planning

Indexes are the primary performance lever in row-oriented databases (PostgreSQL, MySQL). In columnar systems like Snowflake, micro-partition pruning replaces traditional indexing.

### 10.1 PostgreSQL Index Types

| Index Type | Best For | Not Good For |
|---|---|---|
| B-tree (default) | =, <, >, BETWEEN, ORDER BY, LIKE 'prefix%' | Full-text search, array operations |
| Hash | Equality (=) only | Range queries |
| GIN | JSONB, arrays, full-text (`@@`) | Simple equality |
| BRIN | Very large tables with natural ordering (timestamps, sequential IDs) | Random-access patterns |

```sql
-- Standard B-tree
CREATE INDEX idx_orders_date ON orders(order_date);

-- Partial index: only index rows you actually query
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- Covering index: include all columns needed so the query never touches the table
CREATE INDEX idx_orders_covering ON orders(customer_id, order_date)
  INCLUDE (amount, status);
```

### 10.2 EXPLAIN ANALYZE

EXPLAIN shows what the optimizer *plans*. EXPLAIN ANALYZE *executes* and shows what actually happened:

```sql
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 42 AND order_date > '2024-01-01';
```

Key nodes to recognize:
- **Seq Scan** — full table scan (bad if table is large and filtered)
- **Index Scan** — using an index (good)
- **Index Only Scan** — all columns from index, never touches table heap (best)
- **Hash Join** vs **Nested Loop** — hash joins are fast for large tables; nested loops work for small inner relations

> 🎯 **Interview prep**: "Your query is slow. How do you diagnose it?" — Answer: EXPLAIN ANALYZE to find the bottleneck node. Look for Seq Scan on large tables, high estimated vs actual row counts (stale statistics), and sort nodes that could be eliminated with an index.

**Resources**
- [Use The Index, Luke](https://use-the-index-luke.com/) — the definitive book on SQL indexing, free online
- [PostgreSQL EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html) — official guide to reading query plans

---

## 11. Query Optimization

Good SQL is not just correct — it's fast. The optimizations below cover both PostgreSQL and Snowflake patterns.

### 11.1 Universal Rules

```sql
-- ❌ Avoid: functions on indexed columns in WHERE (prevents index use)
WHERE EXTRACT(year FROM order_date) = 2024

-- ✅ Use: range predicates on the raw column
WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01'

-- ❌ Avoid: SELECT * in production queries
SELECT * FROM large_table;

-- ✅ Use: explicit columns (enables column pruning in Snowflake)
SELECT id, name, created_at FROM large_table;

-- ❌ Avoid: UNION when you need UNION ALL
SELECT id FROM a UNION SELECT id FROM b;  -- sorts + deduplicates

-- ✅ Use: UNION ALL unless you specifically need dedup
SELECT id FROM a UNION ALL SELECT id FROM b;
```

### 11.2 Snowflake-Specific Optimization ([Greybeam, 2024](https://greybeam.medium.com/snowflake-query-optimization-7-tips-for-faster-queries-4701337e595b))

**Micro-partition pruning** — the most impactful lever:

```sql
-- ✅ Filter on partition-aligned columns for maximum pruning
WHERE event_date BETWEEN '2024-01-01' AND '2024-03-31'
  AND region = 'us-east'

-- ❌ This cannot be pruned (function hides the value from min/max metadata)
WHERE TO_CHAR(event_date, 'YYYY-MM') = '2024-01'
```

**Warehouse cache** — keep related workloads on the same warehouse:

Snowflake caches query results for 24 hours. Identical queries (same SQL + same underlying data) return instantly from cache at zero credit cost. If you run multiple dashboards on the same warehouse, warm cache serves subsequent queries for free.

**Clustering keys** — for very large tables (TB+) with selective filter patterns:

```sql
-- Add a clustering key on the most-filtered column
ALTER TABLE events CLUSTER BY (event_date, user_region);

-- Check how well-clustered your table is
SELECT SYSTEM$CLUSTERING_INFORMATION('events', '(event_date, user_region)');
```

> 🏭 **Production note**: Clustering consumes credits to maintain and increases storage costs due to Time Travel retention delays. The break-even point is typically tables >1TB that are queried frequently with the same filter pattern. Don't cluster small or infrequently-queried tables.

**Resources**
- [Snowflake Query Optimization: 7 Tips](https://greybeam.medium.com/snowflake-query-optimization-7-tips-for-faster-queries-4701337e595b) — production-tested optimizations with timing benchmarks
- [Snowflake Clustering Keys docs](https://docs.snowflake.com/en/user-guide/tables-clustering-keys) — when and how to cluster

---

## 12. Views & Materialized Views

Views are SQL's abstraction mechanism. They trade maintenance overhead for query simplicity and encapsulation.

```sql
-- Regular view: reruns the query every time it's accessed
CREATE VIEW active_user_stats AS
SELECT user_id, COUNT(*) AS session_count, MAX(event_ts) AS last_seen
FROM sessions
WHERE event_ts >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY user_id;

-- Materialized view (PostgreSQL): stores the result physically
CREATE MATERIALIZED VIEW daily_revenue_summary AS
SELECT DATE_TRUNC('day', order_date) AS day, SUM(amount) AS revenue
FROM orders GROUP BY 1;

-- Refresh manually
REFRESH MATERIALIZED VIEW daily_revenue_summary;

-- Concurrent refresh (doesn't lock reads)
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_revenue_summary;
```

**Snowflake Dynamic Tables** — Snowflake's answer to materialized views, with automatic incremental refresh:

```sql
CREATE OR REPLACE DYNAMIC TABLE user_features
  TARGET_LAG = '1 hour'
  WAREHOUSE = compute_wh
AS
  SELECT user_id, COUNT(*) AS event_count, MAX(event_ts) AS last_seen
  FROM events
  WHERE event_date >= CURRENT_DATE() - 90
  GROUP BY user_id;
```

> 🎯 **Interview prep**: "When would you use a materialized view vs a regular view?" — Regular views add no storage cost but can be slow if the underlying query is expensive. Materialized views trade storage + staleness for query speed. Use materialized when the view is queried far more often than the underlying data changes.

**Resources**
- [Snowflake Dynamic Tables](https://docs.snowflake.com/en/user-guide/dynamic-tables-intro) — incremental materialization
- [PostgreSQL Materialized Views](https://www.postgresql.org/docs/current/sql-creatematerializedview.html) — full syntax and refresh options

---

## 13. Analytical Patterns

These are the four patterns that appear in every data science take-home and analytics engineering role. Each one has a canonical SQL structure worth memorizing.

### 13.1 Cohort Analysis

Cohort analysis measures retention: what fraction of users who started in period X are still active in period X+N? The SQL pattern has two steps: assign cohort, then count active users per cohort × period pair.

```sql
WITH
  -- Step 1: assign each user to their signup cohort
  cohorts AS (
    SELECT user_id,
           DATE_TRUNC('month', signup_date) AS cohort_month
    FROM users
  ),
  -- Step 2: get all activity with cohort label
  activity AS (
    SELECT e.user_id,
           c.cohort_month,
           DATE_TRUNC('month', e.event_date) AS activity_month,
           DATEDIFF('month', c.cohort_month,
                    DATE_TRUNC('month', e.event_date)) AS months_since_signup
    FROM events e
    JOIN cohorts c ON e.user_id = c.user_id
  )
-- Step 3: count distinct users per cohort × period, normalize to %
SELECT
  cohort_month,
  months_since_signup,
  COUNT(DISTINCT user_id)                    AS active_users,
  FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
    PARTITION BY cohort_month ORDER BY months_since_signup
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  )                                          AS cohort_size,
  ROUND(100.0 * COUNT(DISTINCT user_id) /
    FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
      PARTITION BY cohort_month ORDER BY months_since_signup
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ), 1)                                    AS retention_pct
FROM activity
GROUP BY cohort_month, months_since_signup
ORDER BY cohort_month, months_since_signup;
```

> 🏭 **Production note**: Cohort tables grow as O(cohorts × max_age). For a product with 24 months of history and monthly cohorts, you get 24×24=576 cells. This is small. But if you're doing daily cohorts over 2 years, you're at 730×730=533K cells. Pre-aggregate to monthly cohorts for dashboards.

**Resources**
- [Cohort Retention SQL Templates for Snowflake & BigQuery](https://stellans.io/cohort-retention-sql-templates-snowflake-bigquery/) — ready-to-use production templates

### 13.2 Funnel Analysis

A funnel measures how many users complete each step in a sequence: viewed product → added to cart → purchased. The SQL challenge is counting users who completed each step in order, not just each step independently.

**Standard approach — conditional aggregation** (works everywhere):

```sql
SELECT
  COUNT(DISTINCT CASE WHEN step >= 1 THEN user_id END) AS step1_view,
  COUNT(DISTINCT CASE WHEN step >= 2 THEN user_id END) AS step2_add_to_cart,
  COUNT(DISTINCT CASE WHEN step >= 3 THEN user_id END) AS step3_purchase
FROM (
  SELECT user_id,
    MAX(CASE WHEN event = 'product_view'    THEN 1 ELSE 0 END) +
    MAX(CASE WHEN event = 'add_to_cart'     THEN 1 ELSE 0 END) +
    MAX(CASE WHEN event = 'purchase'        THEN 1 ELSE 0 END) AS step
  FROM funnel_events
  WHERE event_date >= '2024-01-01'
  GROUP BY user_id
) funnel_steps;
```

**Snowflake approach — MATCH_RECOGNIZE** ([Hoffa, 2024](https://medium.com/data-science/funnel-analytics-with-sql-match-recognize-on-snowflake-8bd576d9b7b1)):

`MATCH_RECOGNIZE` identifies sequential event patterns declaratively, like regex for rows:

```sql
SELECT *
FROM funnel_events
MATCH_RECOGNIZE (
  PARTITION BY user_id
  ORDER BY event_ts
  MEASURES
    CLASSIFIER() AS matched_step,
    COUNT(*) AS steps_to_complete
  ONE ROW PER MATCH
  PATTERN (viewed (added | anything)* purchased)
  DEFINE
    viewed    AS event = 'product_view',
    purchased AS event = 'purchase',
    added     AS event = 'add_to_cart',
    anything  AS TRUE
);
```

This finds users who viewed then eventually purchased, counting any events in between.

> 🏭 **Production note**: `MATCH_RECOGNIZE` is Snowflake-specific (also Oracle). The conditional aggregation pattern is portable across BigQuery, Redshift, and PostgreSQL. When building multi-cloud data products, stick with conditional aggregation.

**Resources**
- [Funnel Analytics with MATCH_RECOGNIZE](https://medium.com/data-science/funnel-analytics-with-sql-match-recognize-on-snowflake-8bd576d9b7b1) — worked e-commerce example with Google Analytics data
- [Fivetran Funnel Analysis Guide](https://www.fivetran.com/blog/funnel-analysis) — conversion metric patterns in SQL

### 13.3 Sessionization

Sessionization groups a user's events into discrete sessions separated by inactivity gaps. It's the core of clickstream analysis and user journey modeling.

**Standard pattern — LAG + CASE**:

```sql
WITH with_gaps AS (
  SELECT
    user_id,
    event_ts,
    event_type,
    DATEDIFF('minute',
      LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts),
      event_ts
    ) AS minutes_since_last_event
  FROM clickstream
),
session_starts AS (
  SELECT *,
    CASE WHEN minutes_since_last_event > 30
              OR minutes_since_last_event IS NULL  -- first event
         THEN 1 ELSE 0
    END AS is_new_session
  FROM with_gaps
)
SELECT
  user_id, event_ts, event_type,
  SUM(is_new_session) OVER (
    PARTITION BY user_id ORDER BY event_ts
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS session_id
FROM session_starts;
```

**Snowflake shortcut — CONDITIONAL_TRUE_EVENT** ([QOSF, 2024](https://qosf.com/sessionization.html)):

```sql
SELECT
  user_id,
  event_ts,
  DATEDIFF('minute',
    LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts),
    event_ts
  ) AS minutes_since_last,
  CONDITIONAL_TRUE_EVENT(
    DATEDIFF('minute',
      LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts),
      event_ts
    ) > 30
  ) OVER (PARTITION BY user_id ORDER BY event_ts) AS session_id
FROM clickstream;
```

`CONDITIONAL_TRUE_EVENT` increments a counter each time the condition is true — a Snowflake-native way to assign monotonically increasing session IDs in a single expression.

> 🎯 **Interview prep**: The gaps-and-islands pattern is the #1 most-asked advanced SQL problem. The canonical solution is: (1) flag rows where the gap exceeds the threshold, (2) take a cumulative sum of those flags. That cumulative sum is the session ID.

### 13.4 Pivoting & Unpivoting

**CASE-based pivot** (works everywhere):

```sql
SELECT
  user_id,
  SUM(CASE WHEN channel = 'email'  THEN revenue END) AS email_revenue,
  SUM(CASE WHEN channel = 'sms'    THEN revenue END) AS sms_revenue,
  SUM(CASE WHEN channel = 'push'   THEN revenue END) AS push_revenue
FROM channel_revenue
GROUP BY user_id;
```

**Snowflake PIVOT / UNPIVOT**:

```sql
-- PIVOT: rows to columns
SELECT * FROM channel_revenue
PIVOT (SUM(revenue) FOR channel IN ('email', 'sms', 'push'))
AS p (user_id, email_revenue, sms_revenue, push_revenue);

-- UNPIVOT: columns to rows (normalize wide tables)
SELECT user_id, channel, revenue
FROM wide_revenue_table
UNPIVOT (revenue FOR channel IN (email_revenue, sms_revenue, push_revenue));
```

**Resources**
- [Snowflake PIVOT docs](https://docs.snowflake.com/en/sql-reference/constructs/pivot) — including dynamic PIVOT with ANY VALUE

---

## 14. Snowflake Power Features

Snowflake extends standard SQL with several constructs that are genuinely useful for data science work. These are the features worth learning specifically.

### 14.1 QUALIFY — Filter Window Function Results Without a Subquery

Standard SQL requires wrapping a window function in a subquery to filter on its result. Snowflake's `QUALIFY` clause eliminates this ([Snowflake docs](https://docs.snowflake.com/en/sql-reference/constructs/qualify)):

```sql
-- ❌ Standard SQL: requires a subquery
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn
  FROM employees
) WHERE rn = 1;

-- ✅ Snowflake QUALIFY: cleaner and faster
SELECT *
FROM employees
QUALIFY ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) = 1;

-- Works with any window function
SELECT user_id, event_ts, event_type
FROM events
QUALIFY DENSE_RANK() OVER (PARTITION BY user_id ORDER BY event_ts DESC) <= 3;
```

QUALIFY runs after WINDOW in the logical order, before DISTINCT. You can use window functions defined in SELECT or define them directly in QUALIFY.

### 14.2 VARIANT, FLATTEN & Semi-Structured Data

Covered in Section 9.2. Key addition: Snowflake's `OBJECT_CONSTRUCT` and `ARRAY_CONSTRUCT` let you *build* semi-structured data in SQL, useful for creating API payloads:

```sql
SELECT OBJECT_CONSTRUCT(
  'user_id',   user_id,
  'features',  ARRAY_CONSTRUCT(age, tenure_days, avg_purchase),
  'as_of',     CURRENT_DATE()
) AS feature_payload
FROM user_feature_table;
```

### 14.3 Time Travel — Query Historical Data

Snowflake Time Travel lets you query any table as it existed at a past point in time ([Snowflake docs](https://docs.snowflake.com/en/user-guide/data-time-travel)):

```sql
-- Query table as it was 1 hour ago
SELECT * FROM orders AT (OFFSET => -3600);

-- Query as of a specific timestamp
SELECT * FROM orders AT (TIMESTAMP => '2024-03-01 09:00:00'::TIMESTAMP_TZ);

-- Query before a specific statement ran (use query ID from query history)
SELECT * FROM orders BEFORE (STATEMENT => '8e5d0ca9-005e-44e6-b858-a8f5b37c5726');

-- Restore accidentally deleted data
INSERT INTO orders SELECT * FROM orders BEFORE (STATEMENT => '<delete_query_id>');

-- Clone table at historical point for safe experimentation
CREATE TABLE orders_snapshot CLONE orders AT (TIMESTAMP => '2024-01-01'::TIMESTAMP);

-- UNDROP: restore a dropped table (within retention period)
DROP TABLE orders;
UNDROP TABLE orders;
```

Retention defaults to 1 day (Standard Edition) and can be set up to 90 days on Enterprise:

```sql
ALTER TABLE critical_table SET DATA_RETENTION_TIME_IN_DAYS = 14;
```

> 🏭 **Production note**: Time Travel storage is billed at the same rate as active storage. A table with 30-day retention that receives heavy DML can accrue 30× its logical size in historical versions. Monitor with `SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS`. The sweet spot for most analytics tables is 7–14 days.

### 14.4 Clustering Keys & Micro-Partition Pruning

Discussed in Section 11.2. The diagnostic command:

```sql
-- Returns clustering depth, average overlap, and a quality score
SELECT SYSTEM$CLUSTERING_INFORMATION('orders', '(order_date, region)');
```

Output includes a `clustering_ratio` — values near 1.0 mean excellent clustering; near 0 means heavily mixed partitions that need reclustering.

**Resources**
- [Snowflake QUALIFY docs](https://docs.snowflake.com/en/sql-reference/constructs/qualify) — complete syntax and examples
- [Snowflake Time Travel guide](https://docs.snowflake.com/en/user-guide/data-time-travel) — retention, AT/BEFORE, UNDROP
- [Snowflake Clustering Keys](https://docs.snowflake.com/en/user-guide/tables-clustering-keys) — when to cluster, choosing columns, cost

---

## 15. DuckDB: In-Process Analytics

DuckDB runs SQL directly inside your Python process on Parquet, CSV, or Pandas DataFrames — no server, no credentials, no cluster. For local analytics on datasets up to a few hundred GB, it's faster than Snowflake because there's no network round-trip.

```python
import duckdb
import pandas as pd

con = duckdb.connect()

# Query a Parquet file directly — no loading step
result = con.execute("""
  SELECT user_id, COUNT(*) AS events, SUM(revenue) AS total
  FROM read_parquet('s3://my-bucket/events/*.parquet')
  WHERE event_date >= '2024-01-01'
  GROUP BY user_id
  ORDER BY total DESC
  LIMIT 100
""").df()

# Query a Pandas DataFrame as a table
df = pd.read_csv('orders.csv')
top_customers = con.execute("""
  SELECT customer_id, SUM(amount) AS ltv
  FROM df
  GROUP BY customer_id
  HAVING SUM(amount) > 1000
""").df()

# ASOF join for time-series — native in DuckDB
con.execute("""
  SELECT t.user_id, t.event_ts, p.price
  FROM events t
  ASOF JOIN prices p ON t.product_id = p.product_id
  AND t.event_ts >= p.valid_from
""")
```

DuckDB also supports `PIVOT`, window functions, recursive CTEs, and JSON operations with the same syntax as most cloud warehouses.

> 🏭 **Production note**: DuckDB is the right tool for notebooks and local feature engineering. Use Snowflake for shared, governed production data. A common pattern: prototype features in DuckDB on a sampled Parquet export, then port the SQL to Snowflake for production.

**Resources**
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview.html) — full documentation for in-process analytics
- [DuckDB vs Pandas vs Polars benchmark](https://duckdb.org/2021/05/14/sql-on-pandas.html) — performance on in-memory operations

---

## 16. SQL for ML Workflows

SQL is where ML feature engineering happens at scale. The patterns below bridge SQL and Python ML pipelines.

### 16.1 Feature Engineering in SQL

```sql
-- Time-based features: recency, frequency, monetary (RFM)
WITH rfm AS (
  SELECT
    customer_id,
    DATEDIFF('day', MAX(order_date), CURRENT_DATE())  AS recency_days,
    COUNT(DISTINCT order_id)                           AS frequency,
    SUM(amount)                                        AS monetary_value,
    AVG(amount)                                        AS avg_order_value,
    STDDEV(amount)                                     AS order_value_stddev,
    -- Lag features
    SUM(amount) FILTER (WHERE order_date >= DATEADD(day, -30, CURRENT_DATE()))
                                                       AS revenue_last_30d,
    SUM(amount) FILTER (WHERE order_date >= DATEADD(day, -90, CURRENT_DATE()))
                                                       AS revenue_last_90d
  FROM orders
  GROUP BY customer_id
)
SELECT *, ROUND(revenue_last_30d / NULLIF(revenue_last_90d, 0), 3) AS rev_30_90_ratio
FROM rfm;
```

### 16.2 Train/Test Split in SQL

```sql
-- Deterministic random split using MOD on a hash
SELECT *,
  CASE WHEN MOD(HASH(user_id), 10) < 8 THEN 'train'
       WHEN MOD(HASH(user_id), 10) < 9 THEN 'val'
       ELSE 'test'
  END AS split
FROM features;
```

The key: `HASH(user_id)` is deterministic — the same user always gets the same split, even if you rerun the query.

### 16.3 Label Generation

```sql
-- Binary label: churned in next 30 days
WITH user_last_activity AS (
  SELECT user_id, MAX(event_date) AS last_event
  FROM events GROUP BY user_id
),
labels AS (
  SELECT u.user_id,
    CASE WHEN last_event < DATEADD(day, -30, CURRENT_DATE())
         THEN 1 ELSE 0
    END AS churned
  FROM users u
  JOIN user_last_activity a ON u.user_id = a.user_id
)
SELECT f.*, l.churned
FROM features f
JOIN labels l ON f.user_id = l.user_id;
```

### 16.4 Connecting SQL to Python

```python
import pandas as pd
import sqlalchemy

# SQLAlchemy connection to Snowflake
engine = sqlalchemy.create_engine(
    "snowflake://user:password@account/db/schema?warehouse=wh"
)

# Pull features for training
df = pd.read_sql("""
  SELECT * FROM user_features
  WHERE split = 'train' AND feature_date = '2024-01-01'
""", engine)

# Write predictions back to Snowflake
predictions_df.to_sql('model_predictions', engine, if_exists='append', index=False)
```

> 🎯 **Interview prep**: "How would you prevent data leakage when building a feature in SQL?" — The core answer: use only data available *before* the label timestamp. The implementation: always join features to a `feature_as_of_date` and filter events to `event_date < feature_as_of_date`.

**Resources**
- [SQLAlchemy docs](https://docs.sqlalchemy.org/en/20/core/connections.html) — production-grade SQL connection pooling from Python
- [Pandas SQL comparison](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html) — maps every pandas operation to SQL

---

## 17. Comparison Tables

### 17.1 JOIN Types

*Use this table to quickly pick the right join type for your analytical scenario.*

| JOIN Type | Returns | When to use | When NOT to use | Watch out for |
|---|---|---|---|---|
| INNER JOIN | Rows matching in both tables | Combining tables with guaranteed FK relationships | When you need to preserve all rows from one side | Silently drops unmatched rows |
| LEFT JOIN | All left rows + matching right | When left table defines the population (users, sessions) | When you want to filter to matches only | WHERE on right-side column converts to INNER JOIN |
| RIGHT JOIN | All right rows + matching left | Rarely — prefer LEFT JOIN by flipping table order | Most cases (just swap table order instead) | Confusing for readers |
| FULL OUTER | All rows from both | Data reconciliation, finding gaps between two sources | When tables are large — can produce huge results | Explosion risk with duplicate keys |
| CROSS JOIN | Cartesian product (N×M) | Date spine × product combos, generating all pairs | Large tables — N=10K, M=10K → 100M rows | Easy to accidentally omit a join condition |
| SELF JOIN | Table joined to itself | Org charts, consecutive event comparison | When window functions (LAG/LEAD) are available | Usually replaceable with window functions |

### 17.2 Window Function vs GROUP BY

*The key question: do you want row-level detail preserved, or collapsed aggregates?*

| Approach | How it works | Returns | When to use | When NOT to use | Example |
|---|---|---|---|---|---|
| GROUP BY + aggregate | Collapses rows into groups | One row per group | Summary statistics, reporting totals | When you need per-row detail alongside the aggregate | `SUM(revenue) GROUP BY month` |
| Window function | Computes aggregate over a window without collapsing | One row per input row | Per-row context (rank, running total, vs. group avg) | When you just need a summary — GROUP BY is simpler | `SUM(rev) OVER (PARTITION BY month)` |
| Window + GROUP BY | Filter with HAVING, then rank results | Grouped rows with window metadata | Ranked groups, percentiles within groups | Complex nested logic — use a CTE instead | `RANK() OVER ... QUALIFY rank <= 3` |

### 17.3 Snowflake vs Standard SQL Dialect

*Key syntax differences when porting queries between systems.*

| Feature | Standard SQL (PostgreSQL) | Snowflake | BigQuery |
|---|---|---|---|
| Filter window results | Subquery required | `QUALIFY` clause | Subquery required |
| Semi-structured access | `data->>'field'` (JSONB) | `col:field::TYPE` | `JSON_VALUE(col, '$.field')` |
| Array explode | `jsonb_array_elements()` | `LATERAL FLATTEN(INPUT => col)` | `UNNEST()` or `JSON_ARRAY_ELEMENTS` |
| Case-insensitive LIKE | `ILIKE` (PG extension) | `ILIKE` (native) | Not available — use `LOWER(col) LIKE` |
| Time zone conversion | `AT TIME ZONE 'zone'` | `CONVERT_TIMEZONE('zone', col)` | `DATETIME(col, 'timezone')` |
| Regex functions | `regexp_matches()`, `~` | `REGEXP_SUBSTR()`, `RLIKE` | `REGEXP_EXTRACT()` |
| Historical queries | N/A | `AT (TIMESTAMP => ...)` | N/A |
| Sessionization | LAG + cumulative SUM | `CONDITIONAL_TRUE_EVENT()` | LAG + cumulative SUM |
| Table sampling | `TABLESAMPLE SYSTEM(n)` | `SAMPLE (n ROWS)` or `SAMPLE (n PERCENT)` | `TABLESAMPLE SYSTEM(n PERCENT)` |
| Inline pivot | `CROSSTAB()` (via tablefunc) | Native `PIVOT` / `UNPIVOT` | Native `PIVOT` |

---

## 18. Interview Prep: 10 Worked Problems

These are the patterns that appear most frequently at companies like Meta, Google, Uber, and Airbnb ([MindfulTechie, 2024](https://medium.com/@aicoders/master-sql-window-functions-and-ctes-12-real-data-engineering-interview-questions-with-code-9b42f37c1db1); [pipeline2insights, 2024](https://pipeline2insights.substack.com/p/week-332-advanced-sql-concepts-for)).

---

**Q1 — Second highest salary (without using LIMIT/OFFSET)**

```sql
-- Using DENSE_RANK avoids gaps when multiple people share the top salary
SELECT salary
FROM (
  SELECT salary,
         DENSE_RANK() OVER (ORDER BY salary DESC) AS dr
  FROM employees
) t
WHERE dr = 2;
```
*Why DENSE_RANK over RANK: if three people earn the max salary, RANK gives them all rank 1 and skips to rank 4. DENSE_RANK gives rank 1 and moves to rank 2 correctly.*

---

**Q2 — Top 3 earners per department**

```sql
SELECT name, department, salary
FROM employees
QUALIFY DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) <= 3;
-- Replace QUALIFY with a subquery WHERE clause for non-Snowflake systems
```

---

**Q3 — Month-over-month revenue growth**

```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month, SUM(revenue) AS rev
  FROM orders GROUP BY 1
)
SELECT month, rev,
       LAG(rev) OVER (ORDER BY month)        AS prev_rev,
       ROUND(100.0 *
         (rev - LAG(rev) OVER (ORDER BY month)) /
         NULLIF(LAG(rev) OVER (ORDER BY month), 0), 1) AS pct_growth
FROM monthly
ORDER BY month;
```
*Edge case: NULLIF prevents division by zero when prev_rev = 0. The first row returns NULL — filter it out or COALESCE to 0 depending on the use case.*

---

**Q4 — Consecutive login streaks (3+ days)**

The "gaps and islands" trick: subtract a sequential row number from the date. For consecutive dates, the result is constant (the same "island ID"):

```sql
WITH daily_logins AS (
  SELECT user_id, DATE(login_ts) AS login_date
  FROM logins GROUP BY 1, 2  -- deduplicate multiple logins per day
),
with_groups AS (
  SELECT user_id, login_date,
    login_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date)::int
      AS grp  -- same value for consecutive dates
  FROM daily_logins
),
streaks AS (
  SELECT user_id, grp,
         MIN(login_date) AS streak_start,
         MAX(login_date) AS streak_end,
         COUNT(*)         AS streak_length
  FROM with_groups
  GROUP BY user_id, grp
)
SELECT * FROM streaks WHERE streak_length >= 3;
```

---

**Q5 — Identify users who logged in on 3+ consecutive days in the last 30 days**

```sql
WITH recent AS (
  SELECT DISTINCT user_id, DATE(login_ts) AS d
  FROM logins WHERE login_ts >= CURRENT_DATE() - 30
),
grouped AS (
  SELECT user_id, d,
         d - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY d)::int AS grp
  FROM recent
)
SELECT DISTINCT user_id
FROM grouped GROUP BY user_id, grp HAVING COUNT(*) >= 3;
```

---

**Q6 — Running 7-day average revenue per region**

```sql
SELECT region, order_date, daily_revenue,
  AVG(daily_revenue) OVER (
    PARTITION BY region
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d_avg
FROM (
  SELECT region, DATE(order_date) AS order_date, SUM(amount) AS daily_revenue
  FROM orders GROUP BY 1, 2
) daily;
```

---

**Q7 — Find the first time a user made a purchase after a free trial**

```sql
WITH trial_ends AS (
  SELECT user_id, MAX(event_ts) AS trial_end
  FROM events WHERE event_type = 'trial_end'
  GROUP BY user_id
),
first_purchase AS (
  SELECT e.user_id, MIN(e.event_ts) AS first_purchase_ts
  FROM events e
  JOIN trial_ends t ON e.user_id = t.user_id
  WHERE e.event_type = 'purchase' AND e.event_ts > t.trial_end
  GROUP BY e.user_id
)
SELECT t.user_id,
       t.trial_end,
       p.first_purchase_ts,
       DATEDIFF('day', t.trial_end, p.first_purchase_ts) AS days_to_convert
FROM trial_ends t
LEFT JOIN first_purchase p ON t.user_id = p.user_id;
```

---

**Q8 — Employee hierarchy with depth (recursive CTE)**

```sql
WITH RECURSIVE hierarchy AS (
  SELECT id, name, manager_id, 0 AS depth, name AS path
  FROM employees WHERE manager_id IS NULL  -- root nodes

  UNION ALL

  SELECT e.id, e.name, e.manager_id, h.depth + 1,
         h.path || ' > ' || e.name
  FROM employees e
  JOIN hierarchy h ON e.manager_id = h.id
  WHERE h.depth < 20
)
SELECT id, name, depth, path FROM hierarchy ORDER BY path;
```

---

**Q9 — Median salary per department (without MEDIAN function)**

```sql
-- Portable version using percentile window functions
SELECT DISTINCT department,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary)
    OVER (PARTITION BY department) AS median_salary
FROM employees;
```

*PERCENTILE_CONT interpolates between values for even-sized groups. PERCENTILE_DISC returns the nearest actual value. Both are available in Snowflake and PostgreSQL.*

---

**Q10 — Attribution: last-touch channel before purchase**

```sql
WITH purchase_events AS (
  SELECT user_id, event_ts AS purchase_ts
  FROM events WHERE event_type = 'purchase'
),
last_touch AS (
  SELECT e.user_id, e.event_ts, e.channel,
    ROW_NUMBER() OVER (
      PARTITION BY e.user_id, p.purchase_ts
      ORDER BY e.event_ts DESC
    ) AS rn
  FROM events e
  JOIN purchase_events p ON e.user_id = p.user_id
  WHERE e.event_type = 'channel_touch'
    AND e.event_ts < p.purchase_ts
)
SELECT user_id, channel, COUNT(*) AS attributed_purchases
FROM last_touch WHERE rn = 1
GROUP BY user_id, channel;
```

> 🎯 **Interview prep**: For all of these, the interviewer also wants to hear you handle edge cases: NULLs, ties, users with no data on one side of a join, empty result sets. Stating these aloud before coding shows production experience.

---

## 19. References

### Official Documentation
- Snowflake (2024). *Analytic Window Functions.* https://docs.snowflake.com/en/sql-reference/functions-analytic
- Snowflake (2024). *QUALIFY Clause.* https://docs.snowflake.com/en/sql-reference/constructs/qualify
- Snowflake (2024). *Querying Semi-structured Data.* https://docs.snowflake.com/en/user-guide/querying-semistructured
- Snowflake (2024). *FLATTEN Function.* https://docs.snowflake.com/en/sql-reference/functions/flatten
- Snowflake (2024). *Understanding and Using Time Travel.* https://docs.snowflake.com/en/user-guide/data-time-travel
- Snowflake (2024). *Clustering Keys & Clustered Tables.* https://docs.snowflake.com/en/user-guide/tables-clustering-keys
- PostgreSQL (2024). *Window Functions.* https://www.postgresql.org/docs/current/functions-window.html
- PostgreSQL (2024). *WITH Queries (Common Table Expressions).* https://www.postgresql.org/docs/current/queries-with.html

### Medium Articles
- Pradhan, A. (2024). *Anatomy of a Snowflake Query: A Deep Dive into the Execution Engine.* https://medium.com/snowflake/anatomy-of-a-snowflake-query-a-deep-dive-into-the-execution-engine-ca9061022c47
- Hoffa, F. (2024). *Funnel analytics with SQL: MATCH_RECOGNIZE() on Snowflake.* https://medium.com/data-science/funnel-analytics-with-sql-match-recognize-on-snowflake-8bd576d9b7b1
- Greybeam. (2024). *Snowflake Query Optimization: 7 Tips for Faster Queries.* https://greybeam.medium.com/snowflake-query-optimization-7-tips-for-faster-queries-4701337e595b
- MindfulTechie. (2024). *Master SQL Window Functions and CTEs: 12 Real Data Engineering Interview Questions.* https://medium.com/@aicoders/master-sql-window-functions-and-ctes-12-real-data-engineering-interview-questions-with-code-9b42f37c1db1
- Surani, M. (2025). *Advanced SQL for Data Engineering 2025.* https://mayursurani.medium.com/advanced-sql-for-data-engineering-2025-master-window-functions-ctes-explain-plans-materialized-f729a29cb120

### Substack
- pipeline2insights. (2024). *Week 3/31: Advanced SQL Concepts for Data Engineering Interviews.* https://pipeline2insights.substack.com/p/week-332-advanced-sql-concepts-for

### Technical Guides
- QOSF. (2024). *Sessionization using CONDITIONAL_TRUE_EVENT in Snowflake.* https://qosf.com/sessionization.html
- Stellans.io. (2024). *Cohort Retention SQL Templates: Snowflake & BigQuery.* https://stellans.io/cohort-retention-sql-templates-snowflake-bigquery/
- Winand, M. *Use The Index, Luke: A Guide to Database Performance for Developers.* https://use-the-index-luke.com/

### Practice Platforms
- Mode Analytics SQL Tutorial — https://mode.com/sql-tutorial/
- DataLemur SQL Questions — https://datalemur.com/questions
- StrataScratch SQL Problems — https://platform.stratascratch.com/coding
- DuckDB Python API — https://duckdb.org/docs/api/python/overview.html
