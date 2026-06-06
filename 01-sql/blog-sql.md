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

## Sample Tables Used in This Blog

Every query in this blog runs against one of these tables. Refer back here whenever you want to trace through a result yourself.

**`employees`** — used for ranking, aggregation, hierarchy problems
| id | name  | department  | salary  | manager_id |
|----|-------|-------------|---------|------------|
| 1  | Alice | Engineering | 100000  | NULL       |
| 2  | Bob   | Engineering | 90000   | 1          |
| 3  | Carol | Engineering | 90000   | 1          |
| 4  | Dave  | Marketing   | 80000   | NULL       |
| 5  | Eve   | Marketing   | 70000   | 4          |
| 6  | Frank | Marketing   | 70000   | 4          |
| 7  | Grace | Engineering | 75000   | 2          |

**`customers`** — used for JOIN examples
| id  | name  | email            | status   |
|-----|-------|------------------|----------|
| 101 | Alice | alice@co.com     | active   |
| 102 | Bob   | bob@co.com       | active   |
| 103 | Carol | carol@co.com     | inactive |
| 104 | Dave  | dave@co.com      | active   |
| 105 | Eve   | eve@co.com       | active   |

**`orders`** — used for aggregation, JOIN, analytical patterns, ML features
| order_id | customer_id | order_date  | amount | channel | status    | region   |
|----------|-------------|-------------|--------|---------|-----------|----------|
| 1        | 101         | 2024-01-05  | 250    | web     | completed | us-east  |
| 2        | 102         | 2024-01-10  | 150    | mobile  | completed | us-west  |
| 3        | 101         | 2024-02-03  | 300    | web     | completed | us-east  |
| 4        | 103         | 2024-02-15  | 500    | web     | refunded  | us-east  |
| 5        | 102         | 2024-02-20  | 200    | email   | completed | us-west  |
| 6        | 104         | 2024-03-01  | 175    | mobile  | completed | us-east  |
| 7        | 101         | 2024-03-08  | 400    | web     | completed | us-east  |
| 8        | 103         | 2024-03-12  | 320    | mobile  | completed | us-west  |

**`daily_revenue`** — used for window function frame examples
| order_date  | revenue |
|-------------|---------|
| 2024-01-01  | 200     |
| 2024-01-02  | 150     |
| 2024-01-03  | 300     |
| 2024-01-04  | 100     |
| 2024-01-05  | 250     |
| 2024-01-06  | 180     |
| 2024-01-07  | 320     |

**`user_events`** — used for funnel, sessionization, LAG/LEAD
| user_id | event_type   | event_ts                |
|---------|--------------|-------------------------|
| 101     | login        | 2024-01-01 09:00:00     |
| 101     | product_view | 2024-01-01 09:05:00     |
| 101     | add_to_cart  | 2024-01-01 09:10:00     |
| 101     | purchase     | 2024-01-01 09:15:00     |
| 101     | login        | 2024-01-01 14:00:00     |
| 101     | product_view | 2024-01-01 14:05:00     |
| 102     | login        | 2024-01-01 10:00:00     |
| 102     | product_view | 2024-01-01 10:05:00     |
| 102     | add_to_cart  | 2024-01-01 10:12:00     |
| 102     | purchase     | 2024-01-01 10:20:00     |
| 103     | login        | 2024-01-02 09:00:00     |
| 103     | product_view | 2024-01-02 09:10:00     |
| 104     | login        | 2024-01-02 11:00:00     |
| 104     | product_view | 2024-01-02 11:05:00     |
| 104     | add_to_cart  | 2024-01-02 11:15:00     |

**`clickstream`** — used for sessionization
| user_id | event_ts                | event_type |
|---------|-------------------------|------------|
| 101     | 2024-01-01 09:00:00     | login      |
| 101     | 2024-01-01 09:05:00     | page_view  |
| 101     | 2024-01-01 09:10:00     | click      |
| 101     | 2024-01-01 09:55:00     | page_view  |
| 101     | 2024-01-01 10:00:00     | click      |
| 102     | 2024-01-01 10:00:00     | login      |
| 102     | 2024-01-01 10:10:00     | page_view  |
| 102     | 2024-01-01 11:30:00     | login      |

**`logins`** — used for streak problem
| user_id | login_ts                |
|---------|-------------------------|
| 101     | 2024-01-01 08:00:00     |
| 101     | 2024-01-01 20:00:00     |
| 101     | 2024-01-02 09:00:00     |
| 101     | 2024-01-03 10:00:00     |
| 101     | 2024-01-05 11:00:00     |
| 102     | 2024-01-01 07:00:00     |
| 102     | 2024-01-02 08:00:00     |
| 102     | 2024-01-04 09:00:00     |

**`users`** + **`events`** — used for cohort analysis

**`users`**

| user_id | signup_date |
|---------|-------------|
| 101     | 2024-01-05  |
| 102     | 2024-01-12  |
| 103     | 2024-01-20  |
| 104     | 2024-02-03  |
| 105     | 2024-02-10  |
| 106     | 2024-02-15  |

**`events`**

| user_id | event_date  |
|---------|-------------|
| 101     | 2024-01-08  |
| 101     | 2024-02-05  |
| 101     | 2024-03-08  |
| 102     | 2024-01-15  |
| 102     | 2024-02-20  |
| 103     | 2024-01-22  |
| 104     | 2024-02-07  |
| 104     | 2024-03-01  |
| 105     | 2024-02-12  |
| 105     | 2024-03-05  |
| 106     | 2024-02-18  |

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

**Why this matters — a concrete example.**

Using the `orders` table, suppose you want departments with total spend > 400, broken down by channel. Watch what happens at each step:

```sql
SELECT channel, SUM(amount) AS total
FROM orders
WHERE status = 'completed'          -- step 2: removes order_id=4 (refunded)
GROUP BY channel                    -- step 3: group remaining 7 rows
HAVING SUM(amount) > 400            -- step 4: filter groups by aggregate
ORDER BY total DESC;                -- step 9: sort final result
```

**Step 2 — after WHERE** (status = 'completed', removes order 4):
| order_id | channel | amount |
|----------|---------|--------|
| 1        | web     | 250    |
| 2        | mobile  | 150    |
| 3        | web     | 300    |
| 5        | email   | 200    |
| 6        | mobile  | 175    |
| 7        | web     | 400    |
| 8        | mobile  | 320    |

**Step 3 — after GROUP BY channel:**
| channel | SUM(amount) |
|---------|-------------|
| web     | 950         |
| mobile  | 645         |
| email   | 200         |

**Step 4 — after HAVING SUM(amount) > 400** (removes email):
| channel | SUM(amount) |
|---------|-------------|
| web     | 950         |
| mobile  | 645         |

**Final result (after ORDER BY):**
| channel | total |
|---------|-------|
| web     | 950   |
| mobile  | 645   |

**Common mistakes caused by violating this order:**

**❌ Mistake 1 — using a SELECT alias in WHERE** (SELECT runs at step 7, WHERE at step 2)

```sql
-- ❌ ERROR: column "total" does not exist
SELECT channel, SUM(amount) AS total
FROM orders
WHERE total > 400;         -- "total" alias doesn't exist yet at step 2

-- ✅ Fix 1: repeat the expression in WHERE
WHERE SUM(amount) > 400    -- still wrong — can't aggregate in WHERE

-- ✅ Fix 2: use HAVING (runs after GROUP BY)
SELECT channel, SUM(amount) AS total
FROM orders
GROUP BY channel
HAVING SUM(amount) > 400;

-- ✅ Fix 3: wrap in a subquery or CTE
WITH agg AS (
  SELECT channel, SUM(amount) AS total FROM orders GROUP BY channel
)
SELECT channel, total FROM agg WHERE total > 400;
```

**❌ Mistake 2 — filtering a window function result in WHERE**

```sql
-- ❌ ERROR: window functions are not allowed in WHERE
SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
FROM employees
WHERE rnk <= 2;    -- "rnk" is computed at step 5 (WINDOW), WHERE is step 2

-- ✅ Fix: wrap in a subquery, or use QUALIFY (Snowflake only)
SELECT * FROM (
  SELECT name, department, salary,
         RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
  FROM employees
) t WHERE rnk <= 2;

-- ✅ Snowflake shortcut
SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
FROM employees
QUALIFY rnk <= 2;
```

**❌ Mistake 3 — ORDER BY in a subquery doesn't guarantee outer order**

```sql
-- ❌ Silently unreliable: the outer query may reorder the rows
SELECT * FROM (
  SELECT customer_id, amount FROM orders ORDER BY amount DESC
) sub
LIMIT 3;

-- ✅ Fix: always ORDER BY in the outermost query
SELECT customer_id, amount FROM orders ORDER BY amount DESC LIMIT 3;
```

> 🎯 **Interview prep**: These three mistakes are the most frequently tested execution-order questions. The core rule: aliases defined in SELECT are only visible in ORDER BY (step 9). Aggregates belong in HAVING (step 4), not WHERE (step 2). Window function results need a subquery wrapper — or QUALIFY in Snowflake — to filter.

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

### 3.1 INNER JOIN

Returns only rows that match in **both** tables. Customer 105 (Eve) has no orders — she disappears.

**Input — `customers` (left) and `orders` (right):**

| customers.id | customers.name | orders.order_id | orders.amount |
|---|---|---|---|
| 101 | Alice | — | — |
| 102 | Bob | — | — |
| 103 | Carol | — | — |
| 104 | Dave | — | — |
| 105 | Eve | *(no match)* | *(no match)* |

```sql
SELECT c.name, COUNT(o.order_id) AS order_count, SUM(o.amount) AS total_spend
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
GROUP BY c.name
ORDER BY total_spend DESC;
```

**Result:**
| name  | order_count | total_spend |
|-------|-------------|-------------|
| Alice | 3           | 950         |
| Carol | 2           | 820         |
| Bob   | 2           | 350         |
| Dave  | 1           | 175         |

Eve is excluded because she has no matching rows in `orders`.

### 3.2 LEFT JOIN

Returns **all rows from the left table**, NULLs where no match exists on the right. Eve now appears with 0 orders.

```sql
SELECT c.name, COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.name
ORDER BY order_count DESC;
```

**Result:**
| name  | order_count |
|-------|-------------|
| Alice | 3           |
| Bob   | 2           |
| Carol | 2           |
| Dave  | 1           |
| Eve   | 0           |

`COUNT(o.order_id)` returns 0 for Eve because `o.order_id` is NULL when there is no match — `COUNT` ignores NULLs.

> 🎯 **Interview prep**: "You have a LEFT JOIN and your result has fewer rows than the left table. What happened?" — A WHERE clause on the right-table column (`WHERE o.status = 'completed'`) converted it to an INNER JOIN. Move that filter into the ON clause instead: `ON c.id = o.customer_id AND o.status = 'completed'`.

**The left-join-to-inner-join trap:**

```sql
-- ❌ This silently becomes an INNER JOIN — filters out Eve
SELECT c.name, o.order_id
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.status = 'completed';     -- NULL != 'completed', so Eve is dropped

-- ✅ Keep the filter in ON to preserve unmatched left rows
SELECT c.name, o.order_id
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id AND o.status = 'completed';
```

**Result of the ✅ version:**
| name  | order_id |
|-------|----------|
| Alice | 1        |
| Alice | 3        |
| Alice | 7        |
| Bob   | 2        |
| Bob   | 5        |
| Carol | NULL     |
| Dave  | 6        |
| Eve   | NULL     |

Carol and Eve appear with NULL order_id because their only orders have status 'refunded' or they have no orders at all.

### 3.3 FULL OUTER JOIN

Returns all rows from both tables, NULLs where no match on either side. Useful for data reconciliation.

**Input — two small tables:**

`web_users`:
| user_id |
|---------|
| 101     |
| 102     |
| 105     |

`mobile_users`:
| user_id |
|---------|
| 101     |
| 103     |
| 106     |

```sql
SELECT w.user_id AS web_id, m.user_id AS mobile_id
FROM web_users w
FULL OUTER JOIN mobile_users m ON w.user_id = m.user_id;
```

**Result:**
| web_id | mobile_id |
|--------|-----------|
| 101    | 101       |
| 102    | NULL      |
| 105    | NULL      |
| NULL   | 103       |
| NULL   | 106       |

Rows with NULL on the left = users only in mobile. Rows with NULL on the right = users only on web.

### 3.4 CROSS JOIN

Cartesian product: every row paired with every row. Used to generate all combinations (e.g., date spine × product).

**Input:**

`date_spine` (3 rows): 2024-01-01, 2024-01-02, 2024-01-03  
`products` (2 rows): phone, tablet

```sql
SELECT d.date, p.product
FROM date_spine d
CROSS JOIN products p
ORDER BY d.date, p.product;
```

**Result (3 × 2 = 6 rows):**
| date        | product |
|-------------|---------|
| 2024-01-01  | phone   |
| 2024-01-01  | tablet  |
| 2024-01-02  | phone   |
| 2024-01-02  | tablet  |
| 2024-01-03  | phone   |
| 2024-01-03  | tablet  |

### 3.5 SELF JOIN

A table joined to itself. Classic use case: org charts.

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id
ORDER BY e.id;
```

**Result:**
| employee | manager |
|----------|---------|
| Alice    | NULL    |
| Bob      | Alice   |
| Carol    | Alice   |
| Dave     | NULL    |
| Eve      | Dave    |
| Frank    | Dave    |
| Grace    | Bob     |

Alice and Dave are roots — they have no manager.

### 3.6 Non-Equi & ASOF JOIN

Non-equi joins use comparison operators. **Snowflake ASOF JOIN** handles time-series lookups efficiently:

```sql
-- Get the most recent exchange rate at or before each transaction
SELECT t.transaction_id, t.amount, r.rate
FROM transactions t
ASOF JOIN fx_rates r
  MATCH_CONDITION (t.txn_time >= r.rate_time)
  ON t.currency = r.currency;
```

> 🏭 **Production note**: Disjunctive join conditions (using OR) force a Cartesian product in Snowflake. One team reduced query time from 4m 36s to 6.7s (40× speedup) by rewriting `ON a.id = b.id OR a.alt_id = b.id` as two UNION ALL'd equi-joins.

**Resources**
- [Visual JOIN explanations](https://joins.spathon.com/) — interactive Venn diagram for every join type
- [Snowflake ASOF JOIN docs](https://docs.snowflake.com/en/sql-reference/constructs/asof-join) — time-series merge joins

---

## 4. Subqueries, CTEs & Recursive CTEs

Every complex SQL query can be written three ways: nested subqueries, CTEs, or a single flat query. The choice affects readability, debuggability, and sometimes performance.

### 4.1 Subqueries

**Scalar subquery** — returns a single value. Example: each order's deviation from the average amount.

```sql
SELECT order_id, amount,
       ROUND(amount - (SELECT AVG(amount) FROM orders), 2) AS diff_from_avg
FROM orders
ORDER BY diff_from_avg DESC;
```

**Result** (avg = 286.25):
| order_id | amount | diff_from_avg |
|----------|--------|---------------|
| 4        | 500    | 213.75        |
| 7        | 400    | 113.75        |
| 3        | 300    | 13.75         |
| 8        | 320    | 33.75         |
| 1        | 250    | -36.25        |
| 5        | 200    | -86.25        |
| 6        | 175    | -111.25       |
| 2        | 150    | -136.25       |

**Uncorrelated subquery in FROM** (derived table) — runs once, then joined:

```sql
SELECT channel, channel_total
FROM (
  SELECT channel, SUM(amount) AS channel_total
  FROM orders
  GROUP BY channel
) channel_stats
WHERE channel_total > 400;
```

**Result:**
| channel | channel_total |
|---------|---------------|
| web     | 1450          |
| mobile  | 645           |

### 4.2 CTEs — Common Table Expressions

CTEs replace nested subqueries with named building blocks. They dramatically improve readability without changing semantics (the optimizer inlines them in Snowflake and BigQuery).

**Example: month-over-month revenue using a two-step CTE.**

```sql
WITH
  monthly_rev AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(amount)                      AS revenue
    FROM orders
    GROUP BY 1
  ),
  mom_growth AS (
    SELECT month, revenue,
           LAG(revenue) OVER (ORDER BY month) AS prev_revenue
    FROM monthly_rev
  )
SELECT month,
       revenue,
       prev_revenue,
       ROUND(100.0 * (revenue - prev_revenue) / NULLIF(prev_revenue, 0), 1) AS pct_growth
FROM mom_growth
ORDER BY month;
```

**Intermediate — `monthly_rev`:**
| month       | revenue |
|-------------|---------|
| 2024-01-01  | 400     |
| 2024-02-01  | 1000    |
| 2024-03-01  | 895     |

**Final result:**
| month       | revenue | prev_revenue | pct_growth |
|-------------|---------|--------------|------------|
| 2024-01-01  | 400     | NULL         | NULL       |
| 2024-02-01  | 1000    | 400          | 150.0      |
| 2024-03-01  | 895     | 1000         | -10.5      |

Revenue jumped 150% Jan→Feb, then dropped 10.5% Feb→Mar.

> 🎯 **Interview prep**: "Does a CTE always improve performance?" — No. In PostgreSQL (pre-v12), CTEs are optimization fences — the optimizer cannot push predicates through them. In Snowflake and BigQuery, CTEs are inlined and optimized normally.

### 4.3 Recursive CTEs

Recursive CTEs solve hierarchical problems — org charts, bill-of-materials, graph traversal. Structure: **anchor member** UNION ALL'd with a **recursive member** that references the CTE itself.

```sql
WITH RECURSIVE org_tree AS (
  -- Anchor: root nodes (no manager)
  SELECT id, name, manager_id, 0 AS depth, name AS path
  FROM employees
  WHERE manager_id IS NULL

  UNION ALL

  -- Recursive: direct reports of current level
  SELECT e.id, e.name, e.manager_id, t.depth + 1,
         t.path || ' > ' || e.name
  FROM employees e
  INNER JOIN org_tree t ON e.manager_id = t.id
  WHERE t.depth < 10    -- cycle guard
)
SELECT id, name, depth, path
FROM org_tree
ORDER BY path;
```

**Result:**
| id | name  | depth | path                    |
|----|-------|-------|-------------------------|
| 1  | Alice | 0     | Alice                   |
| 2  | Bob   | 1     | Alice > Bob             |
| 7  | Grace | 2     | Alice > Bob > Grace     |
| 3  | Carol | 1     | Alice > Carol           |
| 4  | Dave  | 0     | Dave                    |
| 5  | Eve   | 1     | Dave > Eve              |
| 6  | Frank | 1     | Dave > Frank            |

> 🏭 **Production note**: Always include a depth limit (`WHERE t.depth < 10`). A circular FK in the data causes infinite recursion. Snowflake has a `MAX_RECURSION` parameter (default 100) as a safety net, but the SQL guard is the right defense.

**Resources**
- [PostgreSQL WITH clause docs](https://www.postgresql.org/docs/current/queries-with.html) — full recursive CTE specification
- [Advanced SQL for Data Engineering](https://mayursurani.medium.com/advanced-sql-for-data-engineering-2025-master-window-functions-ctes-explain-plans-materialized-f729a29cb120) — practical CTEs with EXPLAIN integration

---

## 5. Window Functions: The Data Scientist's Superpower

Window functions are the single biggest skill gap between intermediate and advanced SQL users. They solve problems that would otherwise require self-joins, correlated subqueries, or round-trips to Python — and they do it without collapsing rows the way GROUP BY does.

### 5.1 The OVER Clause Anatomy

```
function_name([expression]) OVER (
  [PARTITION BY partition_columns]   -- divide rows into independent windows
  [ORDER BY sort_columns]            -- define row order within each window
  [frame_clause]                     -- define which rows are "in scope"
)
```

### 5.2 Ranking Functions

**Input — `employees` table, showing Engineering department sorted by salary DESC:**

| id | name  | department  | salary  |
|----|-------|-------------|---------|
| 1  | Alice | Engineering | 100000  |
| 2  | Bob   | Engineering | 90000   |
| 3  | Carol | Engineering | 90000   |
| 7  | Grace | Engineering | 75000   |
| 4  | Dave  | Marketing   | 80000   |
| 5  | Eve   | Marketing   | 70000   |
| 6  | Frank | Marketing   | 70000   |

```sql
SELECT name, department, salary,
  ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num,
  RANK()       OVER (PARTITION BY department ORDER BY salary DESC) AS rnk,
  DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rnk,
  NTILE(4)     OVER (PARTITION BY department ORDER BY salary DESC) AS quartile
FROM employees
ORDER BY department, salary DESC;
```

**Result:**
| name  | department  | salary | row_num | rnk | dense_rnk | quartile |
|-------|-------------|--------|---------|-----|-----------|----------|
| Alice | Engineering | 100000 | 1       | 1   | 1         | 1        |
| Bob   | Engineering | 90000  | 2       | 2   | 2         | 2        |
| Carol | Engineering | 90000  | 3       | 2   | 2         | 3        |
| Grace | Engineering | 75000  | 4       | 4   | 3         | 4        |
| Dave  | Marketing   | 80000  | 1       | 1   | 1         | 1        |
| Eve   | Marketing   | 70000  | 2       | 2   | 2         | 2        |
| Frank | Marketing   | 70000  | 3       | 2   | 2         | 3        |

Key differences for the Bob/Carol tie (both 90000):
- `ROW_NUMBER`: arbitrary 2 and 3 — no ties allowed
- `RANK`: both get 2, then jumps to 4 (skips 3)
- `DENSE_RANK`: both get 2, next rank is 3 (no gap)

**Top-N per group — the classic interview pattern:**

```sql
SELECT name, department, salary
FROM employees
QUALIFY DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) <= 2;
```

**Result (top 2 salaries per department):**
| name  | department  | salary |
|-------|-------------|--------|
| Alice | Engineering | 100000 |
| Bob   | Engineering | 90000  |
| Carol | Engineering | 90000  |
| Dave  | Marketing   | 80000  |
| Eve   | Marketing   | 70000  |
| Frank | Marketing   | 70000  |

Both Bob and Carol are returned because they share rank 2. If you used `RANK() <= 2` or `DENSE_RANK() <= 2`, both ties appear. If you want exactly 2 rows per department regardless of ties, use `ROW_NUMBER() <= 2`.

### 5.3 Frame Clauses: Running Totals & Moving Averages

**Input — `daily_revenue`:**

| order_date  | revenue |
|-------------|---------|
| 2024-01-01  | 200     |
| 2024-01-02  | 150     |
| 2024-01-03  | 300     |
| 2024-01-04  | 100     |
| 2024-01-05  | 250     |
| 2024-01-06  | 180     |
| 2024-01-07  | 320     |

```sql
SELECT
  order_date,
  revenue,
  SUM(revenue) OVER (
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS cumulative_revenue,
  ROUND(AVG(revenue) OVER (
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ), 1) AS rolling_7d_avg
FROM daily_revenue;
```

**Result:**
| order_date  | revenue | cumulative_revenue | rolling_7d_avg |
|-------------|---------|-------------------|----------------|
| 2024-01-01  | 200     | 200               | 200.0          |
| 2024-01-02  | 150     | 350               | 175.0          |
| 2024-01-03  | 300     | 650               | 216.7          |
| 2024-01-04  | 100     | 750               | 187.5          |
| 2024-01-05  | 250     | 1000              | 200.0          |
| 2024-01-06  | 180     | 1180              | 196.7          |
| 2024-01-07  | 320     | 1500              | 214.3          |

`rolling_7d_avg` on Jan 7 = (200+150+300+100+250+180+320)/7 = 1500/7 ≈ 214.3

> 🏭 **Production note**: `ROWS BETWEEN` counts physical rows. `RANGE BETWEEN` groups rows with identical ORDER BY values into the same logical frame. For time-series work, `ROWS BETWEEN` is almost always more predictable.

**❌ Common mistake — RANGE with tied ORDER BY values gives duplicate running totals**

The default frame when `ORDER BY` is present is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. `RANGE` treats all rows with the same `ORDER BY` value as one logical group. So when Bob and Carol both earn 90000, their frame already includes *both* of them — Bob absorbs Carol's salary before she even appears in the list.

**Input — Engineering employees, sorted by salary ASC (Bob and Carol tie):**

| name  | salary |
|-------|--------|
| Grace | 75000  |
| Bob   | 90000  |
| Carol | 90000  |
| Alice | 100000 |

```sql
SELECT name, salary,
  SUM(salary) OVER (
    PARTITION BY department ORDER BY salary
    -- ❌ default = RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_range,

  SUM(salary) OVER (
    PARTITION BY department ORDER BY salary
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW   -- ✅ explicit
  ) AS running_rows
FROM employees
WHERE department = 'Engineering'
ORDER BY salary;
```

**Result:**

| name  | salary | running_range      | running_rows |
|-------|--------|--------------------|--------------|
| Grace | 75000  | 75000              | 75000        |
| Bob   | 90000  | **255000** ❌      | 165000  ✓    |
| Carol | 90000  | **255000** ❌      | 255000  ✓    |
| Alice | 100000 | 355000             | 355000       |

With `RANGE`: Bob already shows 255000 (= 75000 + 90000 + 90000) — the frame jumped ahead and included Carol. Both Bob and Carol show the same total, making it look like a duplicate. With `ROWS`: each row advances one at a time regardless of ties.

**Rule:** always write `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` explicitly whenever ties are possible in the ORDER BY column.

> 🎯 **Interview prep**: "Why are two rows showing the same cumulative total?" The answer is always `RANGE` vs `ROWS`. Interviewers put a deliberate tie in the test data to expose this.

### 5.4 LAG, LEAD, FIRST_VALUE, LAST_VALUE

**Input — `user_events`** (showing user 101's events):

| user_id | event_type   | event_ts                |
|---------|--------------|-------------------------|
| 101     | login        | 2024-01-01 09:00:00     |
| 101     | product_view | 2024-01-01 09:05:00     |
| 101     | add_to_cart  | 2024-01-01 09:10:00     |
| 101     | purchase     | 2024-01-01 09:15:00     |

```sql
SELECT
  user_id,
  event_ts,
  event_type,
  LAG(event_type, 1)  OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_event,
  LEAD(event_type, 1) OVER (PARTITION BY user_id ORDER BY event_ts) AS next_event,
  FIRST_VALUE(event_type) OVER (
    PARTITION BY user_id ORDER BY event_ts
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS first_ever_event,
  DATEDIFF('minute',
    LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts),
    event_ts
  ) AS mins_since_last
FROM user_events
WHERE user_id = 101
ORDER BY event_ts;
```

**Result:**
| user_id | event_ts     | event_type   | prev_event   | next_event   | first_ever_event | mins_since_last |
|---------|--------------|--------------|--------------|--------------|------------------|-----------------|
| 101     | 09:00:00     | login        | NULL         | product_view | login            | NULL            |
| 101     | 09:05:00     | product_view | login        | add_to_cart  | login            | 5               |
| 101     | 09:10:00     | add_to_cart  | product_view | purchase     | login            | 5               |
| 101     | 09:15:00     | purchase     | add_to_cart  | NULL         | login            | 5               |

> 🎯 **Interview prep**: "Why does LAG return NULL for the first row?" — There is no previous row. Handle with `COALESCE(LAG(col) OVER (...), 0)` or filter out NULLs downstream. Forgetting this causes off-by-one errors in growth rate calculations.

**❌ Common mistake 1 — LAG with wrong ORDER BY direction returns future values, not past**

`LAG` returns the row that comes *before the current row in window order*. If you sort descending (newest date first), LAG's "before" is a later calendar date — you accidentally fetch next month's revenue instead of last month's.

```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS rev
  FROM orders GROUP BY 1
)
SELECT month, rev,
  LAG(rev) OVER (ORDER BY month DESC) AS wrong_prev,   -- ❌ DESC: "prev" = future month
  LAG(rev) OVER (ORDER BY month ASC)  AS correct_prev  -- ✅ ASC:  "prev" = past month
FROM monthly
ORDER BY month DESC;
```

**Result:**

| month       | rev  | wrong_prev      | correct_prev |
|-------------|------|-----------------|--------------|
| 2024-03-01  | 895  | NULL *(lucky ✓)*| 1000  ✓      |
| 2024-02-01  | 1000 | 895   ❌        | 400   ✓      |
| 2024-01-01  | 400  | 1000  ❌        | NULL  ✓      |

`wrong_prev` for January returns 1000 (February's revenue — a *future* month). March accidentally gets NULL, which looks right, but only because there is no row after it in descending order.

The practical version of this bug — you sort output newest-first for readability, and accidentally carry that `DESC` into the window:

```sql
-- ❌ MoM growth computed backwards — numbers look plausible but are wrong
SELECT month, rev,
  ROUND(100.0 * (rev - LAG(rev) OVER (ORDER BY month DESC))
        / NULLIF(LAG(rev) OVER (ORDER BY month DESC), 0), 1) AS wrong_growth,

-- ✅ Window ORDER BY ASC for logic; outer ORDER BY DESC for display
  ROUND(100.0 * (rev - LAG(rev) OVER (ORDER BY month ASC))
        / NULLIF(LAG(rev) OVER (ORDER BY month ASC), 0), 1)  AS correct_growth
FROM monthly
ORDER BY month DESC;
```

**Result:**

| month       | rev  | wrong_growth | correct_growth |
|-------------|------|--------------|----------------|
| 2024-03-01  | 895  | -10.5  ❌    | -10.5  ✓       |
| 2024-02-01  | 1000 | 150.0  ❌    | 150.0  ✓       |
| 2024-01-01  | 400  | NULL         | NULL   ✓       |

The numbers happen to be identical here — which is exactly what makes this bug dangerous. They differ as soon as the time series is non-monotone or has gaps.

**Rule:** always define window `ORDER BY` to match the *logical* sequence (ASC for time series). Control display order with the outermost `ORDER BY` separately.

---

**❌ Common mistake 2 — LAST_VALUE with the default frame returns the current row, not the partition's last**

`LAST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary)` looks like it should return the highest salary in the department. It almost never does.

The reason: the default frame is `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. `LAST_VALUE` sees only rows up to the current row, so the "last" value is always the current row's own value.

**Input — Engineering employees, sorted by salary ASC:**

| name  | salary |
|-------|--------|
| Grace | 75000  |
| Bob   | 90000  |
| Carol | 90000  |
| Alice | 100000 |

```sql
SELECT name, salary,
  LAST_VALUE(salary) OVER (
    PARTITION BY department ORDER BY salary
    -- ❌ default frame: UNBOUNDED PRECEDING to CURRENT ROW
  ) AS wrong_max,

  LAST_VALUE(salary) OVER (
    PARTITION BY department ORDER BY salary
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- ✅ full partition
  ) AS correct_max,

  MAX(salary) OVER (PARTITION BY department) AS max_simpler   -- ✅ no frame issue
FROM employees
WHERE department = 'Engineering'
ORDER BY salary;
```

**Result:**

| name  | salary | wrong_max          | correct_max | max_simpler |
|-------|--------|--------------------|-------------|-------------|
| Grace | 75000  | 75000  ❌          | 100000  ✓   | 100000  ✓   |
| Bob   | 90000  | 90000  ❌          | 100000  ✓   | 100000  ✓   |
| Carol | 90000  | 90000  ❌          | 100000  ✓   | 100000  ✓   |
| Alice | 100000 | 100000 ✓ *(lucky)* | 100000  ✓   | 100000  ✓   |

`wrong_max` is correct only for Alice because her frame happens to include all rows. Everyone else just gets their own salary.

**Rule:** `FIRST_VALUE` is immune to this bug (its default frame always includes the first row). `LAST_VALUE` needs an explicit `UNBOUNDED FOLLOWING` frame — or just use `MAX()` as a window function, which has no default frame issue.

> 🎯 **Interview prep**: "Why does `LAST_VALUE` return the wrong result?" — default frame. This is a top-10 window function interview question. Examiners test it by putting the most interesting value at the *end* of the partition so the wrong answer looks wrong immediately.

**Resources**
- [12 Real Window Function Interview Questions](https://medium.com/@aicoders/master-sql-window-functions-and-ctes-12-real-data-engineering-interview-questions-with-code-9b42f37c1db1) — FAANG-sourced problems with full solutions
- [Snowflake Analytic Functions](https://docs.snowflake.com/en/sql-reference/functions-analytic) — complete reference

---

## 6. Set Operations

Set operations combine results from multiple SELECT statements.

**Input for all examples:**

`premium_users` (user_ids): 101, 102, 104  
`active_last_30d` (user_ids): 101, 103, 104, 105  
`churned_users` (user_ids): 103

```sql
-- UNION ALL: all users from both, with source label (no dedup)
SELECT user_id, 'premium' AS src FROM premium_users
UNION ALL
SELECT user_id, 'active'  AS src FROM active_last_30d;
```

**Result (7 rows — no dedup):**
| user_id | src     |
|---------|---------|
| 101     | premium |
| 102     | premium |
| 104     | premium |
| 101     | active  |
| 103     | active  |
| 104     | active  |
| 105     | active  |

```sql
-- INTERSECT: users in BOTH premium AND active
SELECT user_id FROM premium_users
INTERSECT
SELECT user_id FROM active_last_30d;
```

**Result:**
| user_id |
|---------|
| 101     |
| 104     |

```sql
-- EXCEPT: active users who are NOT churned (anti-join pattern)
SELECT user_id FROM active_last_30d
EXCEPT
SELECT user_id FROM churned_users;
```

**Result:**
| user_id |
|---------|
| 101     |
| 104     |
| 105     |

> 🎯 **Interview prep**: `UNION ALL` is always faster than `UNION` because it skips the sort+dedup step. Use `UNION` only when duplicate elimination is semantically required.

**Resources**
- [Mode SQL Set Operations Tutorial](https://mode.com/sql-tutorial/sql-set-operations/) — visual explanation with examples

---

## 7. NULL Handling & CASE Expressions

NULLs are the most dangerous values in SQL. `NULL + 1 = NULL`. `NULL = NULL` is FALSE. `COUNT(col)` skips NULLs but `COUNT(*)` doesn't.

### 7.1 NULL Functions

**Input — a small `survey_responses` table:**

| user_id | score | email        |
|---------|-------|--------------|
| 101     | 8     | a@co.com     |
| 102     | NULL  | b@co.com     |
| 103     | 5     | NULL         |
| 104     | NULL  | NULL         |

```sql
SELECT
  user_id,
  COALESCE(score, 0)                    AS score_no_null,
  COALESCE(email, 'no_email')           AS contact,
  COUNT(score)            OVER ()       AS non_null_count,   -- 2
  COUNT(*)                OVER ()       AS total_count,      -- 4
  AVG(score)              OVER ()       AS avg_excl_nulls,   -- (8+5)/2=6.5
  AVG(COALESCE(score, 0)) OVER ()       AS avg_incl_nulls    -- (8+0+5+0)/4=3.25
FROM survey_responses;
```

**Result:**
| user_id | score_no_null | contact    | non_null_count | total_count | avg_excl_nulls | avg_incl_nulls |
|---------|--------------|------------|----------------|-------------|----------------|----------------|
| 101     | 8            | a@co.com   | 2              | 4           | 6.5            | 3.25           |
| 102     | 0            | b@co.com   | 2              | 4           | 6.5            | 3.25           |
| 103     | 5            | no_email   | 2              | 4           | 6.5            | 3.25           |
| 104     | 0            | no_email   | 2              | 4           | 6.5            | 3.25           |

### 7.2 CASE Expressions — Conditional Aggregation

Conditional aggregation replaces multiple self-joins with a single scan. **Input — `orders`:**

```sql
SELECT
  DATE_TRUNC('month', order_date)                              AS month,
  COUNT(*)                                                     AS total_orders,
  COUNT(CASE WHEN channel = 'web'    THEN 1 END)               AS web_orders,
  COUNT(CASE WHEN channel = 'mobile' THEN 1 END)               AS mobile_orders,
  SUM(CASE WHEN channel = 'web'    THEN amount ELSE 0 END)     AS web_revenue,
  SUM(CASE WHEN status = 'refunded' THEN amount ELSE 0 END)    AS refund_amount
FROM orders
GROUP BY 1
ORDER BY 1;
```

**Result:**
| month       | total_orders | web_orders | mobile_orders | web_revenue | refund_amount |
|-------------|--------------|------------|---------------|-------------|---------------|
| 2024-01-01  | 2            | 1          | 1             | 250         | 0             |
| 2024-02-01  | 3            | 2          | 0             | 800         | 500           |
| 2024-03-01  | 3            | 1          | 2             | 400         | 0             |

January: 1 web order (250) + 1 mobile (150). February: order 4 was refunded (500), hence refund_amount=500.

> 🏭 **Production note**: Conditional aggregation replaces three self-joins that each do a full table scan. This is one of the most common query rewrites in data engineering.

**Resources**
- [PostgreSQL Conditional Expressions](https://www.postgresql.org/docs/current/functions-conditional.html) — CASE, COALESCE, NULLIF, GREATEST, LEAST

---

## 8. Date/Time & String Functions

These functions are database-specific enough that knowing the Snowflake variants is essential for practical work.

### 8.1 Date/Time (Snowflake dialect)

**Input — selected rows from `orders`:**

| order_id | order_date  |
|----------|-------------|
| 1        | 2024-01-05  |
| 4        | 2024-02-15  |
| 7        | 2024-03-08  |

```sql
SELECT
  order_id,
  order_date,
  DATE_TRUNC('month', order_date)              AS month_start,    -- first day of month
  EXTRACT(dow FROM order_date)                 AS day_of_week,    -- 0=Sun…6=Sat
  DATEADD(day, 30, order_date)                 AS due_date,
  DATEDIFF('day', order_date, CURRENT_DATE())  AS days_ago
FROM orders
WHERE order_id IN (1, 4, 7);
```

**Result** (assuming today = 2026-05-17):
| order_id | order_date  | month_start | day_of_week | due_date    | days_ago |
|----------|-------------|-------------|-------------|-------------|----------|
| 1        | 2024-01-05  | 2024-01-01  | 5 (Fri)     | 2024-02-04  | 863      |
| 4        | 2024-02-15  | 2024-02-01  | 4 (Thu)     | 2024-03-16  | 822      |
| 7        | 2024-03-08  | 2024-03-01  | 5 (Fri)     | 2024-04-07  | 800      |

### 8.2 String Functions

**Input — a `contacts` table:**

| id | raw_phone      | email                 |
|----|----------------|-----------------------|
| 1  | (415) 555-1234 | Alice@Example.COM     |
| 2  | 650.555.9876   | bob@example.com       |

```sql
SELECT
  id,
  REGEXP_REPLACE(raw_phone, '[^0-9]', '')  AS digits_only,
  LOWER(email)                              AS normalized_email,
  SPLIT_PART(email, '@', 2)                AS domain
FROM contacts;
```

**Result:**
| id | digits_only | normalized_email      | domain      |
|----|-------------|-----------------------|-------------|
| 1  | 4155551234  | alice@example.com     | Example.COM |
| 2  | 6505559876  | bob@example.com       | example.com |

> 🏭 **Production note**: In Snowflake, REGEXP functions use PCRE syntax, not POSIX. Patterns that work in PostgreSQL may need rewriting. Always test regex patterns against a sample before deploying to production transformations.

**Resources**
- [Snowflake Date/Time Functions](https://docs.snowflake.com/en/sql-reference/functions-date-time) — complete reference
- [Snowflake String Functions](https://docs.snowflake.com/en/sql-reference/functions-string) — full regex support

---

## 9. Semi-structured Data: JSON, Arrays & Snowflake VARIANT

Modern data pipelines ingest raw JSON from APIs, webhooks, and event streams.

### 9.1 PostgreSQL JSON/JSONB

**Input — `events` table with a JSONB `data` column:**

| id | data |
|----|------|
| 1  | `{"user_id": 101, "event_type": "purchase", "items": [{"sku": "A1", "qty": 2}, {"sku": "B3", "qty": 1}]}` |

```sql
SELECT
  id,
  data -> 'user_id'           AS user_id_json,      -- JSON type
  data ->> 'event_type'       AS event_type,         -- text type
  data -> 'items' -> 0 ->> 'sku' AS first_sku,
  jsonb_array_length(data -> 'items') AS item_count
FROM events WHERE id = 1;
```

**Result:**
| id | user_id_json | event_type | first_sku | item_count |
|----|-------------|------------|-----------|------------|
| 1  | 101          | purchase   | A1        | 2          |

### 9.2 Snowflake VARIANT & FLATTEN

Snowflake stores JSON natively in a **VARIANT** column using colon notation ([Snowflake docs](https://docs.snowflake.com/en/user-guide/querying-semistructured)):

```sql
-- Colon syntax for field access; cast with ::
SELECT
  src:user_id::STRING                AS user_id,
  src:purchase.total_amount::NUMBER  AS amount,
  src:purchase.items[0].sku::STRING  AS first_sku
FROM raw_events;
```

**FLATTEN — explode an array into rows:**

**Input** (single VARIANT row with array):
```json
{ "user_id": 101, "items": [{"sku":"A1","qty":2}, {"sku":"B3","qty":1}] }
```

```sql
SELECT
  e.src:user_id::STRING  AS user_id,
  f.index                AS item_position,
  f.value:sku::STRING    AS sku,
  f.value:qty::INTEGER   AS qty
FROM raw_events e,
LATERAL FLATTEN(INPUT => e.src:items) f;
```

**Result:**
| user_id | item_position | sku | qty |
|---------|---------------|-----|-----|
| 101     | 0             | A1  | 2   |
| 101     | 1             | B3  | 1   |

> 🎯 **Interview prep**: "How would you count the number of items in a nested JSON array in Snowflake?" — Use `ARRAY_SIZE(src:items)` for a scalar result, or `LATERAL FLATTEN` to aggregate per-element.

**Resources**
- [Snowflake Semi-structured Data docs](https://docs.snowflake.com/en/user-guide/querying-semistructured) — VARIANT, FLATTEN, PARSE_JSON

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
SELECT * FROM orders WHERE customer_id = 101 AND order_date > '2024-01-01';
```

Key nodes to recognize:
- **Seq Scan** — full table scan (bad if table is large and filtered)
- **Index Scan** — using an index (good)
- **Index Only Scan** — all columns from index, never touches table heap (best)
- **Hash Join** vs **Nested Loop** — hash joins are fast for large tables; nested loops work for small inner relations

> 🎯 **Interview prep**: "Your query is slow. How do you diagnose it?" — Answer: EXPLAIN ANALYZE to find the bottleneck node. Look for Seq Scan on large tables, high estimated vs actual row counts (stale statistics), and sort nodes that could be eliminated with an index.

**Resources**
- [Use The Index, Luke](https://use-the-index-luke.com/) — definitive book on SQL indexing, free online
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

**Clustering keys** — for very large tables (TB+) with selective filter patterns:

```sql
ALTER TABLE events CLUSTER BY (event_date, user_region);
SELECT SYSTEM$CLUSTERING_INFORMATION('events', '(event_date, user_region)');
```

> 🏭 **Production note**: Clustering consumes credits to maintain. The break-even point is typically tables >1TB queried frequently with the same filter pattern.

**Resources**
- [Snowflake Query Optimization: 7 Tips](https://greybeam.medium.com/snowflake-query-optimization-7-tips-for-faster-queries-4701337e595b) — production-tested optimizations with timing benchmarks
- [Snowflake Clustering Keys docs](https://docs.snowflake.com/en/user-guide/tables-clustering-keys)

---

## 12. Views & Materialized Views

```sql
-- Regular view: reruns the query every time it's accessed
CREATE VIEW active_user_stats AS
SELECT user_id, COUNT(*) AS session_count, MAX(event_ts) AS last_seen
FROM sessions
WHERE event_ts >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY user_id;

-- Snowflake Dynamic Table — automatic incremental refresh
CREATE OR REPLACE DYNAMIC TABLE user_features
  TARGET_LAG = '1 hour'
  WAREHOUSE = compute_wh
AS
  SELECT user_id, COUNT(*) AS event_count, MAX(event_ts) AS last_seen
  FROM events
  WHERE event_date >= CURRENT_DATE() - 90
  GROUP BY user_id;
```

> 🎯 **Interview prep**: "When would you use a materialized view vs a regular view?" — Regular views add no storage cost but can be slow if the underlying query is expensive. Materialized views trade storage + staleness for query speed.

**Resources**
- [Snowflake Dynamic Tables](https://docs.snowflake.com/en/user-guide/dynamic-tables-intro)
- [PostgreSQL Materialized Views](https://www.postgresql.org/docs/current/sql-creatematerializedview.html)

---

## 13. Analytical Patterns

These four patterns appear in every data science take-home and analytics engineering role.

### 13.1 Cohort Analysis

Cohort analysis answers: "What fraction of users who signed up in month X are still active in month X+N?"

**Input — `users` and `events` tables** (see Sample Tables section above).

**Step 1 — assign cohort month:**

```sql
WITH cohorts AS (
  SELECT user_id,
         DATE_TRUNC('month', signup_date) AS cohort_month
  FROM users
)
SELECT * FROM cohorts ORDER BY user_id;
```

**Intermediate result:**
| user_id | cohort_month |
|---------|--------------|
| 101     | 2024-01-01   |
| 102     | 2024-01-01   |
| 103     | 2024-01-01   |
| 104     | 2024-02-01   |
| 105     | 2024-02-01   |
| 106     | 2024-02-01   |

**Step 2 — join events and compute months since signup:**

```sql
WITH cohorts AS (
  SELECT user_id, DATE_TRUNC('month', signup_date) AS cohort_month
  FROM users
),
activity AS (
  SELECT e.user_id,
         c.cohort_month,
         DATE_TRUNC('month', e.event_date)       AS activity_month,
         DATEDIFF('month', c.cohort_month,
                  DATE_TRUNC('month', e.event_date)) AS months_since_signup
  FROM events e
  JOIN cohorts c ON e.user_id = c.user_id
)
SELECT cohort_month, months_since_signup,
       COUNT(DISTINCT user_id)                                          AS active_users,
       FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
         PARTITION BY cohort_month ORDER BY months_since_signup
         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
       )                                                                AS cohort_size,
       ROUND(100.0 * COUNT(DISTINCT user_id) /
         FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
           PARTITION BY cohort_month ORDER BY months_since_signup
           ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
         ), 1)                                                           AS retention_pct
FROM activity
GROUP BY cohort_month, months_since_signup
ORDER BY cohort_month, months_since_signup;
```

**Final result:**
| cohort_month | months_since_signup | active_users | cohort_size | retention_pct |
|--------------|---------------------|--------------|-------------|---------------|
| 2024-01-01   | 0                   | 3            | 3           | 100.0         |
| 2024-01-01   | 1                   | 2            | 3           | 66.7          |
| 2024-01-01   | 2                   | 1            | 3           | 33.3          |
| 2024-02-01   | 0                   | 3            | 3           | 100.0         |
| 2024-02-01   | 1                   | 2            | 3           | 66.7          |

Reading: The January cohort (users 101, 102, 103) retained 100% at month 0, 67% at month 1 (only 101 and 102 came back in February), and 33% at month 2 (only 101 came back in March).

> 🏭 **Production note**: Cohort tables grow as O(cohorts × max_age). Daily cohorts over 2 years = 730×730 = 533K cells. Pre-aggregate to monthly cohorts for dashboards.

**Resources**
- [Cohort Retention SQL Templates for Snowflake & BigQuery](https://stellans.io/cohort-retention-sql-templates-snowflake-bigquery/)

### 13.2 Funnel Analysis

A funnel measures how many users complete each step in a sequence: product_view → add_to_cart → purchase.

**Input — `user_events`** (see Sample Tables above). Summary by user:

| user_id | has product_view? | has add_to_cart? | has purchase? | step score |
|---------|-------------------|------------------|---------------|------------|
| 101     | ✓                 | ✓                | ✓             | 3          |
| 102     | ✓                 | ✓                | ✓             | 3          |
| 103     | ✓                 | ✗                | ✗             | 1          |
| 104     | ✓                 | ✓                | ✗             | 2          |

```sql
SELECT
  COUNT(DISTINCT CASE WHEN step >= 1 THEN user_id END) AS step1_product_view,
  COUNT(DISTINCT CASE WHEN step >= 2 THEN user_id END) AS step2_add_to_cart,
  COUNT(DISTINCT CASE WHEN step >= 3 THEN user_id END) AS step3_purchase
FROM (
  SELECT user_id,
    MAX(CASE WHEN event_type = 'product_view' THEN 1 ELSE 0 END) +
    MAX(CASE WHEN event_type = 'add_to_cart'  THEN 1 ELSE 0 END) +
    MAX(CASE WHEN event_type = 'purchase'     THEN 1 ELSE 0 END) AS step
  FROM user_events
  GROUP BY user_id
) funnel;
```

**Result:**
| step1_product_view | step2_add_to_cart | step3_purchase |
|--------------------|-------------------|----------------|
| 4                  | 3                 | 2              |

50% of users who viewed a product completed a purchase. 1 user (103) dropped off before adding to cart.

**Conversion rates:**
- View → Add to cart: 3/4 = 75%
- Add to cart → Purchase: 2/3 = 67%
- Overall: 2/4 = 50%

**Snowflake approach — MATCH_RECOGNIZE** ([Hoffa, 2024](https://medium.com/data-science/funnel-analytics-with-sql-match-recognize-on-snowflake-8bd576d9b7b1)):

```sql
SELECT user_id, COUNT(*) AS funnel_completions
FROM user_events
MATCH_RECOGNIZE (
  PARTITION BY user_id
  ORDER BY event_ts
  MEASURES COUNT(*) AS steps_to_complete
  ONE ROW PER MATCH
  PATTERN (viewed (anything)* carted (anything)* purchased)
  DEFINE
    viewed    AS event_type = 'product_view',
    purchased AS event_type = 'purchase',
    carted    AS event_type = 'add_to_cart',
    anything  AS TRUE
)
GROUP BY user_id;
```

**Result:**
| user_id | funnel_completions |
|---------|--------------------|
| 101     | 1                  |
| 102     | 1                  |

> 🏭 **Production note**: `MATCH_RECOGNIZE` is Snowflake/Oracle-specific. The conditional aggregation pattern is portable across BigQuery, Redshift, and PostgreSQL.

**Resources**
- [Funnel Analytics with MATCH_RECOGNIZE](https://medium.com/data-science/funnel-analytics-with-sql-match-recognize-on-snowflake-8bd576d9b7b1)

### 13.3 Sessionization

Sessionization groups a user's events into discrete sessions separated by inactivity gaps (commonly 30 minutes).

**Input — `clickstream`:**

| user_id | event_ts                | event_type |
|---------|-------------------------|------------|
| 101     | 2024-01-01 09:00:00     | login      |
| 101     | 2024-01-01 09:05:00     | page_view  |
| 101     | 2024-01-01 09:10:00     | click      |
| 101     | **2024-01-01 09:55:00** | page_view  |  ← 45 min gap → new session
| 101     | 2024-01-01 10:00:00     | click      |
| 102     | 2024-01-01 10:00:00     | login      |
| 102     | 2024-01-01 10:10:00     | page_view  |
| 102     | **2024-01-01 11:30:00** | login      |  ← 80 min gap → new session

**Step 1 — compute gap and flag session starts:**

```sql
WITH with_gaps AS (
  SELECT
    user_id, event_ts, event_type,
    DATEDIFF('minute',
      LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts),
      event_ts
    ) AS mins_since_last
  FROM clickstream
)
SELECT *,
  CASE WHEN mins_since_last > 30
            OR mins_since_last IS NULL   -- first event
       THEN 1 ELSE 0
  END AS is_new_session
FROM with_gaps
ORDER BY user_id, event_ts;
```

**Intermediate result:**
| user_id | event_ts     | event_type | mins_since_last | is_new_session |
|---------|--------------|------------|-----------------|----------------|
| 101     | 09:00:00     | login      | NULL            | 1              |
| 101     | 09:05:00     | page_view  | 5               | 0              |
| 101     | 09:10:00     | click      | 5               | 0              |
| 101     | 09:55:00     | page_view  | 45              | 1              |
| 101     | 10:00:00     | click      | 5               | 0              |
| 102     | 10:00:00     | login      | NULL            | 1              |
| 102     | 10:10:00     | page_view  | 10              | 0              |
| 102     | 11:30:00     | login      | 80              | 1              |

**Step 2 — cumulative sum of `is_new_session` = session ID:**

```sql
WITH with_gaps AS (...),  -- same as above
session_starts AS (
  SELECT *, CASE WHEN mins_since_last > 30 OR mins_since_last IS NULL THEN 1 ELSE 0 END AS is_new_session
  FROM with_gaps
)
SELECT
  user_id, event_ts, event_type,
  SUM(is_new_session) OVER (
    PARTITION BY user_id ORDER BY event_ts
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS session_id
FROM session_starts
ORDER BY user_id, event_ts;
```

**Final result:**
| user_id | event_ts     | event_type | session_id |
|---------|--------------|------------|------------|
| 101     | 09:00:00     | login      | 1          |
| 101     | 09:05:00     | page_view  | 1          |
| 101     | 09:10:00     | click      | 1          |
| 101     | 09:55:00     | page_view  | 2          |
| 101     | 10:00:00     | click      | 2          |
| 102     | 10:00:00     | login      | 1          |
| 102     | 10:10:00     | page_view  | 1          |
| 102     | 11:30:00     | login      | 2          |

User 101 has 2 sessions (session 1: 09:00–09:10, session 2: 09:55–10:00). User 102 has 2 sessions (session 1: 10:00–10:10, session 2: 11:30).

> 🎯 **Interview prep**: The gaps-and-islands pattern is the #1 most-asked advanced SQL problem. The canonical solution: (1) flag rows where the gap exceeds threshold, (2) take cumulative sum of those flags. That sum is the session/island ID.

**Snowflake shortcut — CONDITIONAL_TRUE_EVENT:**

```sql
SELECT
  user_id, event_ts,
  CONDITIONAL_TRUE_EVENT(
    DATEDIFF('minute',
      LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts),
      event_ts
    ) > 30
  ) OVER (PARTITION BY user_id ORDER BY event_ts) AS session_id
FROM clickstream;
```

`CONDITIONAL_TRUE_EVENT` increments a counter each time the condition is true — same result as above in a single expression.

### 13.4 Pivoting & Unpivoting

**Input — `orders` by channel per month:**

```sql
SELECT
  DATE_TRUNC('month', order_date)                              AS month,
  SUM(CASE WHEN channel = 'web'    THEN amount END)            AS web_revenue,
  SUM(CASE WHEN channel = 'mobile' THEN amount END)            AS mobile_revenue,
  SUM(CASE WHEN channel = 'email'  THEN amount END)            AS email_revenue
FROM orders
GROUP BY 1
ORDER BY 1;
```

**Result:**
| month       | web_revenue | mobile_revenue | email_revenue |
|-------------|-------------|----------------|---------------|
| 2024-01-01  | 250         | 150            | NULL          |
| 2024-02-01  | 800         | NULL           | 200           |
| 2024-03-01  | 400         | 495            | NULL          |

NULL = no orders in that channel for that month. Use `COALESCE(..., 0)` if you need 0 instead.

**Resources**
- [Snowflake PIVOT docs](https://docs.snowflake.com/en/sql-reference/constructs/pivot)

---

## 14. Snowflake Power Features

### 14.1 QUALIFY — Filter Window Function Results Without a Subquery

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
```

**Result (top earner per department):**
| id | name  | department  | salary |
|----|-------|-------------|--------|
| 1  | Alice | Engineering | 100000 |
| 4  | Dave  | Marketing   | 80000  |

### 14.2 VARIANT & FLATTEN

Covered in Section 9.2. Additional: `OBJECT_CONSTRUCT` and `ARRAY_CONSTRUCT` build semi-structured data in SQL:

```sql
SELECT OBJECT_CONSTRUCT(
  'user_id',   user_id,
  'features',  ARRAY_CONSTRUCT(amount, customer_id),
  'as_of',     CURRENT_DATE()
) AS feature_payload
FROM orders WHERE order_id = 1;
```

**Result:**
| feature_payload |
|---|
| `{"user_id": 1, "features": [250, 101], "as_of": "2026-05-17"}` |

### 14.3 Time Travel — Query Historical Data

```sql
-- Query table as it was 1 hour ago
SELECT * FROM orders AT (OFFSET => -3600);

-- Restore accidentally deleted data
INSERT INTO orders SELECT * FROM orders BEFORE (STATEMENT => '<delete_query_id>');

-- Clone table at historical point for safe experimentation
CREATE TABLE orders_snapshot CLONE orders AT (TIMESTAMP => '2024-01-01'::TIMESTAMP);
```

> 🏭 **Production note**: Time Travel storage is billed at the same rate as active storage. A table with 30-day retention that receives heavy DML can accrue 30× its logical size in historical versions. Monitor with `SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS`. The sweet spot for most analytics tables is 7–14 days.

**Resources**
- [Snowflake QUALIFY docs](https://docs.snowflake.com/en/sql-reference/constructs/qualify)
- [Snowflake Time Travel guide](https://docs.snowflake.com/en/user-guide/data-time-travel)
- [Snowflake Clustering Keys](https://docs.snowflake.com/en/user-guide/tables-clustering-keys)

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

# Query a Pandas DataFrame as a SQL table
df = pd.read_csv('orders.csv')
top_customers = con.execute("""
  SELECT customer_id, SUM(amount) AS ltv
  FROM df
  GROUP BY customer_id
  HAVING SUM(amount) > 1000
""").df()
```

> 🏭 **Production note**: DuckDB is the right tool for notebooks and local feature engineering. Use Snowflake for shared, governed production data. A common pattern: prototype features in DuckDB on a sampled Parquet export, then port the SQL to Snowflake for production.

**Resources**
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview.html)
- [DuckDB vs Pandas vs Polars benchmark](https://duckdb.org/2021/05/14/sql-on-pandas.html)

---

## 16. SQL for ML Workflows

SQL is where ML feature engineering happens at scale.

### 16.1 Feature Engineering in SQL

**Input — `orders` table.** Building RFM (Recency, Frequency, Monetary) features per customer:

```sql
WITH rfm AS (
  SELECT
    customer_id,
    DATEDIFF('day', MAX(order_date), '2024-04-01')   AS recency_days,
    COUNT(DISTINCT order_id)                           AS frequency,
    SUM(amount)                                        AS monetary_value,
    AVG(amount)                                        AS avg_order_value,
    SUM(CASE WHEN order_date >= '2024-02-01'
             THEN amount ELSE 0 END)                   AS revenue_last_60d,
    SUM(CASE WHEN order_date >= '2024-01-01'
             THEN amount ELSE 0 END)                   AS revenue_last_90d
  FROM orders
  GROUP BY customer_id
)
SELECT *, ROUND(revenue_last_60d / NULLIF(revenue_last_90d, 0), 3) AS rev_60_90_ratio
FROM rfm
ORDER BY monetary_value DESC;
```

**Result:**
| customer_id | recency_days | frequency | monetary_value | avg_order_value | revenue_last_60d | revenue_last_90d | rev_60_90_ratio |
|-------------|-------------|-----------|----------------|-----------------|------------------|------------------|-----------------|
| 101         | 24          | 3         | 950            | 316.7           | 700              | 950              | 0.737           |
| 103         | 20          | 2         | 820            | 410.0           | 820              | 820              | 1.000           |
| 102         | 41          | 2         | 350            | 175.0           | 200              | 350              | 0.571           |
| 104         | 31          | 1         | 175            | 175.0           | 175              | 175              | 1.000           |

### 16.2 Train/Test Split in SQL

```sql
SELECT *,
  CASE WHEN MOD(HASH(customer_id), 10) < 8 THEN 'train'
       WHEN MOD(HASH(customer_id), 10) < 9 THEN 'val'
       ELSE 'test'
  END AS split
FROM rfm_features;
```

**Result** (HASH is deterministic — same customer always gets same split):
| customer_id | monetary_value | split |
|-------------|----------------|-------|
| 101         | 950            | train |
| 102         | 350            | train |
| 103         | 820            | val   |
| 104         | 175            | train |

### 16.3 Label Generation

```sql
-- Binary churn label: no activity in last 30 days
WITH last_activity AS (
  SELECT customer_id, MAX(order_date) AS last_order
  FROM orders GROUP BY customer_id
)
SELECT f.*, 
  CASE WHEN l.last_order < '2024-03-01' THEN 1 ELSE 0 END AS churned
FROM rfm_features f
JOIN last_activity l ON f.customer_id = l.customer_id;
```

**Result:**
| customer_id | monetary_value | churned |
|-------------|----------------|---------|
| 101         | 950            | 0       |
| 102         | 350            | 0       |
| 103         | 820            | 0       |
| 104         | 175            | 0       |

(All customers have recent activity in our small sample. In production, some would be churned = 1.)

> 🎯 **Interview prep**: "How would you prevent data leakage when building a feature in SQL?" — Use only data available *before* the label timestamp. Implementation: join features to a `feature_as_of_date` and filter events to `event_date < feature_as_of_date`.

**Resources**
- [SQLAlchemy docs](https://docs.sqlalchemy.org/en/20/core/connections.html) — production-grade SQL connection pooling from Python
- [Pandas SQL comparison](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html)

---

## 17. Comparison Tables

### 17.1 JOIN Types

| JOIN Type | Returns | When to use | When NOT to use | Watch out for |
|---|---|---|---|---|
| INNER JOIN | Rows matching in both tables | Combining tables with guaranteed FK relationships | When you need to preserve all rows from one side | Silently drops unmatched rows |
| LEFT JOIN | All left rows + matching right | When left table defines the population (users, sessions) | When you want to filter to matches only | WHERE on right-side column converts to INNER JOIN |
| RIGHT JOIN | All right rows + matching left | Rarely — prefer LEFT JOIN by flipping table order | Most cases (just swap table order instead) | Confusing for readers |
| FULL OUTER | All rows from both | Data reconciliation, finding gaps between two sources | When tables are large — can produce huge results | Explosion risk with duplicate keys |
| CROSS JOIN | Cartesian product (N×M) | Date spine × product combos, generating all pairs | Large tables — N=10K, M=10K → 100M rows | Easy to accidentally omit a join condition |
| SELF JOIN | Table joined to itself | Org charts, consecutive event comparison | When window functions (LAG/LEAD) are available | Usually replaceable with window functions |

### 17.2 Window Function vs GROUP BY

| Approach | How it works | Returns | When to use | Example |
|---|---|---|---|---|
| GROUP BY + aggregate | Collapses rows into groups | One row per group | Summary statistics, reporting totals | `SUM(revenue) GROUP BY month` |
| Window function | Computes aggregate over a window without collapsing | One row per input row | Per-row context (rank, running total, vs. group avg) | `SUM(rev) OVER (PARTITION BY month)` |
| Window + QUALIFY | Filter window results | Subset of rows | Top-N per group | `QUALIFY RANK() OVER (...) <= 3` |

### 17.3 Snowflake vs Standard SQL Dialect

| Feature | PostgreSQL | Snowflake | BigQuery |
|---|---|---|---|
| Filter window results | Subquery required | `QUALIFY` clause | Subquery required |
| Semi-structured access | `data->>'field'` (JSONB) | `col:field::TYPE` | `JSON_VALUE(col, '$.field')` |
| Array explode | `jsonb_array_elements()` | `LATERAL FLATTEN(INPUT => col)` | `UNNEST()` |
| Case-insensitive LIKE | `ILIKE` | `ILIKE` (native) | `LOWER(col) LIKE` |
| Time zone conversion | `AT TIME ZONE 'zone'` | `CONVERT_TIMEZONE('zone', col)` | `DATETIME(col, 'timezone')` |
| Historical queries | N/A | `AT (TIMESTAMP => ...)` | N/A |
| Sessionization | LAG + cumulative SUM | `CONDITIONAL_TRUE_EVENT()` | LAG + cumulative SUM |
| Inline pivot | `CROSSTAB()` (tablefunc) | Native `PIVOT` / `UNPIVOT` | Native `PIVOT` |

---

## 18. Interview Prep: 10 Worked Problems

Each problem shows the input, query, and exact output. These are the patterns that appear most frequently at Meta, Google, Uber, and Airbnb ([MindfulTechie, 2024](https://medium.com/@aicoders/master-sql-window-functions-and-ctes-12-real-data-engineering-interview-questions-with-code-9b42f37c1db1); [pipeline2insights, 2024](https://pipeline2insights.substack.com/p/week-332-advanced-sql-concepts-for)).

---

### Q1 — Second Highest Salary (without LIMIT/OFFSET)

**Input — `employees`:**

| id | name  | salary |
|----|-------|--------|
| 1  | Alice | 100000 |
| 2  | Bob   | 90000  |
| 3  | Carol | 90000  |
| 4  | Dave  | 80000  |
| 5  | Eve   | 70000  |

```sql
SELECT salary
FROM (
  SELECT salary,
         DENSE_RANK() OVER (ORDER BY salary DESC) AS dr
  FROM employees
) t
WHERE dr = 2;
```

**Result:**
| salary |
|--------|
| 90000  |

*Why DENSE_RANK over RANK: if three people share the max salary, RANK gives them all rank 1 and skips to rank 4. DENSE_RANK gives rank 1 and moves to rank 2 correctly. The second highest distinct salary is 90000 (Bob and Carol).*

---

### Q2 — Top 2 Earners per Department

**Input — `employees`** (all 7 rows).

```sql
SELECT name, department, salary
FROM employees
QUALIFY DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) <= 2
ORDER BY department, salary DESC;
```

**Result:**
| name  | department  | salary |
|-------|-------------|--------|
| Alice | Engineering | 100000 |
| Bob   | Engineering | 90000  |
| Carol | Engineering | 90000  |
| Dave  | Marketing   | 80000  |
| Eve   | Marketing   | 70000  |
| Frank | Marketing   | 70000  |

Both Bob and Carol are returned because they share rank 2. Grace (75000) is excluded — she is rank 3 in Engineering.

---

### Q3 — Month-over-Month Revenue Growth

**Input — `orders`** → aggregated by month.

```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month,
         SUM(amount) AS rev
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

**Result:**
| month       | rev  | prev_rev | pct_growth |
|-------------|------|----------|------------|
| 2024-01-01  | 400  | NULL     | NULL       |
| 2024-02-01  | 1000 | 400      | 150.0      |
| 2024-03-01  | 895  | 1000     | -10.5      |

January has no previous month (NULL). February grew 150% (400→1000). March declined 10.5% (1000→895).

*Edge case: `NULLIF(prev_rev, 0)` prevents division by zero. The first row returns NULL — filter it or `COALESCE` to 0 depending on the use case.*

---

### Q4 — Consecutive Login Streaks (3+ Days)

**Input — `logins`:**

| user_id | login_ts                |
|---------|-------------------------|
| 101     | 2024-01-01 08:00:00     |
| 101     | 2024-01-01 20:00:00     |
| 101     | 2024-01-02 09:00:00     |
| 101     | 2024-01-03 10:00:00     |
| 101     | 2024-01-05 11:00:00     |
| 102     | 2024-01-01 07:00:00     |
| 102     | 2024-01-02 08:00:00     |
| 102     | 2024-01-04 09:00:00     |

**Step 1 — deduplicate to one row per (user, date):**

| user_id | login_date  |
|---------|-------------|
| 101     | 2024-01-01  |
| 101     | 2024-01-02  |
| 101     | 2024-01-03  |
| 101     | 2024-01-05  |
| 102     | 2024-01-01  |
| 102     | 2024-01-02  |
| 102     | 2024-01-04  |

**The gaps-and-islands trick** — subtracting the row number from the date gives the same constant for consecutive dates:

| user_id | login_date  | row_num | date - row_num (grp) |
|---------|-------------|---------|----------------------|
| 101     | 2024-01-01  | 1       | 2023-12-31           |
| 101     | 2024-01-02  | 2       | 2023-12-31           |
| 101     | 2024-01-03  | 3       | 2023-12-31           |
| 101     | 2024-01-05  | 4       | 2024-01-01  ← gap!   |
| 102     | 2024-01-01  | 1       | 2023-12-31           |
| 102     | 2024-01-02  | 2       | 2023-12-31           |
| 102     | 2024-01-04  | 3       | 2024-01-01  ← gap!   |

```sql
WITH daily_logins AS (
  SELECT user_id, DATE(login_ts) AS login_date
  FROM logins GROUP BY 1, 2
),
with_groups AS (
  SELECT user_id, login_date,
    login_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date)::int AS grp
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

**Result:**
| user_id | grp         | streak_start | streak_end  | streak_length |
|---------|-------------|--------------|-------------|---------------|
| 101     | 2023-12-31  | 2024-01-01   | 2024-01-03  | 3             |

Only user 101 has a streak ≥ 3 consecutive days (Jan 1–3). User 102 only has 2 consecutive days (Jan 1–2) before the gap on Jan 3.

---

### Q5 — Users with 3+ Consecutive Logins in the Last 30 Days

```sql
WITH recent AS (
  SELECT DISTINCT user_id, DATE(login_ts) AS d
  FROM logins WHERE login_ts >= '2024-01-01'
),
grouped AS (
  SELECT user_id, d,
         d - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY d)::int AS grp
  FROM recent
)
SELECT DISTINCT user_id
FROM grouped
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

**Result:**
| user_id |
|---------|
| 101     |

---

### Q6 — Running 7-Day Average Revenue per Region

**Input — `orders`** grouped to daily regional revenue:

| region   | order_date  | daily_revenue |
|----------|-------------|---------------|
| us-east  | 2024-01-05  | 250           |
| us-west  | 2024-01-10  | 150           |
| us-east  | 2024-02-03  | 300           |
| us-east  | 2024-02-15  | 500           |
| us-west  | 2024-02-20  | 200           |
| us-east  | 2024-03-01  | 175           |
| us-east  | 2024-03-08  | 400           |
| us-west  | 2024-03-12  | 320           |

```sql
SELECT region, order_date, daily_revenue,
  ROUND(AVG(daily_revenue) OVER (
    PARTITION BY region
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ), 1) AS rolling_7d_avg
FROM (
  SELECT region, DATE(order_date) AS order_date, SUM(amount) AS daily_revenue
  FROM orders GROUP BY 1, 2
) daily
ORDER BY region, order_date;
```

**Result:**
| region   | order_date  | daily_revenue | rolling_7d_avg |
|----------|-------------|---------------|----------------|
| us-east  | 2024-01-05  | 250           | 250.0          |
| us-east  | 2024-02-03  | 300           | 275.0          |
| us-east  | 2024-02-15  | 500           | 350.0          |
| us-east  | 2024-03-01  | 175           | 306.3          |
| us-east  | 2024-03-08  | 400           | 325.0          |
| us-west  | 2024-01-10  | 150           | 150.0          |
| us-west  | 2024-02-20  | 200           | 175.0          |
| us-west  | 2024-03-12  | 320           | 223.3          |

---

### Q7 — First Purchase After Free Trial

**Input — `user_events`** (using event_type to track trial_end and purchase events):

For this problem, assume:
- event_type `trial_end` marks when the trial ended
- event_type `purchase` marks paid purchases

| user_id | event_type  | event_ts                |
|---------|-------------|-------------------------|
| 101     | trial_end   | 2024-01-01 08:00:00     |
| 101     | purchase    | 2024-01-01 09:15:00     |
| 102     | trial_end   | 2024-01-01 09:30:00     |
| 102     | purchase    | 2024-01-01 10:20:00     |
| 103     | trial_end   | 2024-01-02 08:00:00     |

```sql
WITH trial_ends AS (
  SELECT user_id, MAX(event_ts) AS trial_end
  FROM user_events WHERE event_type = 'trial_end'
  GROUP BY user_id
),
first_purchase AS (
  SELECT e.user_id, MIN(e.event_ts) AS first_purchase_ts
  FROM user_events e
  JOIN trial_ends t ON e.user_id = t.user_id
  WHERE e.event_type = 'purchase' AND e.event_ts > t.trial_end
  GROUP BY e.user_id
)
SELECT t.user_id,
       t.trial_end,
       p.first_purchase_ts,
       DATEDIFF('minute', t.trial_end, p.first_purchase_ts) AS mins_to_convert
FROM trial_ends t
LEFT JOIN first_purchase p ON t.user_id = p.user_id;
```

**Result:**
| user_id | trial_end            | first_purchase_ts    | mins_to_convert |
|---------|----------------------|----------------------|-----------------|
| 101     | 2024-01-01 08:00:00  | 2024-01-01 09:15:00  | 75              |
| 102     | 2024-01-01 09:30:00  | 2024-01-01 10:20:00  | 50              |
| 103     | 2024-01-02 08:00:00  | NULL                 | NULL            |

User 103 ended their trial but never purchased — LEFT JOIN preserves them with NULL.

---

### Q8 — Employee Hierarchy with Depth (Recursive CTE)

**Input — `employees`** (all 7 rows including manager_id).

```sql
WITH RECURSIVE hierarchy AS (
  SELECT id, name, manager_id, 0 AS depth, name AS path
  FROM employees WHERE manager_id IS NULL

  UNION ALL

  SELECT e.id, e.name, e.manager_id, h.depth + 1,
         h.path || ' > ' || e.name
  FROM employees e
  JOIN hierarchy h ON e.manager_id = h.id
  WHERE h.depth < 20
)
SELECT id, name, depth, path FROM hierarchy ORDER BY path;
```

**Result:**
| id | name  | depth | path                    |
|----|-------|-------|-------------------------|
| 1  | Alice | 0     | Alice                   |
| 2  | Bob   | 1     | Alice > Bob             |
| 7  | Grace | 2     | Alice > Bob > Grace     |
| 3  | Carol | 1     | Alice > Carol           |
| 4  | Dave  | 0     | Dave                    |
| 5  | Eve   | 1     | Dave > Eve              |
| 6  | Frank | 1     | Dave > Frank            |

Alice and Dave are root managers (depth=0). Grace is 2 levels deep — she reports to Bob who reports to Alice.

---

### Q9 — Median Salary per Department

**Input — `employees`**.

```sql
SELECT DISTINCT department,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary)
    OVER (PARTITION BY department) AS median_salary
FROM employees
ORDER BY department;
```

**Result:**
| department  | median_salary |
|-------------|---------------|
| Engineering | 90000         |
| Marketing   | 70000         |

Engineering salaries: [75000, 90000, 90000, 100000] → median = (90000+90000)/2 = 90000.  
Marketing salaries: [70000, 70000, 80000] → median = 70000.

*`PERCENTILE_CONT` interpolates between values for even-sized groups. `PERCENTILE_DISC` returns the nearest actual value. Both are available in Snowflake and PostgreSQL.*

---

### Q10 — Attribution: Last-Touch Channel Before Purchase

**Input — `user_events`** (using channel as event_type for simplicity):

| user_id | event_type    | event_ts                |
|---------|---------------|-------------------------|
| 101     | email_click   | 2024-01-01 08:30:00     |
| 101     | paid_search   | 2024-01-01 08:55:00     |
| 101     | purchase      | 2024-01-01 09:15:00     |
| 102     | organic       | 2024-01-01 09:45:00     |
| 102     | paid_search   | 2024-01-01 10:05:00     |
| 102     | purchase      | 2024-01-01 10:20:00     |

```sql
WITH purchase_events AS (
  SELECT user_id, event_ts AS purchase_ts
  FROM user_events WHERE event_type = 'purchase'
),
last_touch AS (
  SELECT e.user_id, e.event_ts, e.event_type AS channel,
    ROW_NUMBER() OVER (
      PARTITION BY e.user_id, p.purchase_ts
      ORDER BY e.event_ts DESC
    ) AS rn
  FROM user_events e
  JOIN purchase_events p ON e.user_id = p.user_id
  WHERE e.event_type NOT IN ('purchase')
    AND e.event_ts < p.purchase_ts
)
SELECT user_id, channel, COUNT(*) AS attributed_purchases
FROM last_touch WHERE rn = 1
GROUP BY user_id, channel
ORDER BY user_id;
```

**Result:**
| user_id | channel     | attributed_purchases |
|---------|-------------|----------------------|
| 101     | paid_search | 1                    |
| 102     | paid_search | 1                    |

Both users' last touchpoint before purchasing was `paid_search`. Email (user 101) and organic (user 102) touched first but don't get last-touch credit.

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
- Pradhan, A. (2024). *Anatomy of a Snowflake Query.* https://medium.com/snowflake/anatomy-of-a-snowflake-query-a-deep-dive-into-the-execution-engine-ca9061022c47
- Hoffa, F. (2024). *Funnel analytics with SQL: MATCH_RECOGNIZE() on Snowflake.* https://medium.com/data-science/funnel-analytics-with-sql-match-recognize-on-snowflake-8bd576d9b7b1
- Greybeam. (2024). *Snowflake Query Optimization: 7 Tips.* https://greybeam.medium.com/snowflake-query-optimization-7-tips-for-faster-queries-4701337e595b
- MindfulTechie. (2024). *Master SQL Window Functions and CTEs.* https://medium.com/@aicoders/master-sql-window-functions-and-ctes-12-real-data-engineering-interview-questions-with-code-9b42f37c1db1
- Surani, M. (2025). *Advanced SQL for Data Engineering 2025.* https://mayursurani.medium.com/advanced-sql-for-data-engineering-2025-master-window-functions-ctes-explain-plans-materialized-f729a29cb120

### Substack
- pipeline2insights. (2024). *Week 3/31: Advanced SQL Concepts for Data Engineering Interviews.* https://pipeline2insights.substack.com/p/week-332-advanced-sql-concepts-for

### Technical Guides
- QOSF. (2024). *Sessionization using CONDITIONAL_TRUE_EVENT in Snowflake.* https://qosf.com/sessionization.html
- Stellans.io. (2024). *Cohort Retention SQL Templates.* https://stellans.io/cohort-retention-sql-templates-snowflake-bigquery/
- Winand, M. *Use The Index, Luke.* https://use-the-index-luke.com/

### Practice Platforms
- Mode Analytics SQL Tutorial — https://mode.com/sql-tutorial/
- DataLemur SQL Questions — https://datalemur.com/questions
- StrataScratch SQL Problems — https://platform.stratascratch.com/coding
- DuckDB Python API — https://duckdb.org/docs/api/python/overview.html
