-- ============================================================
-- WINDOW FUNCTIONS — The most-tested topic in DS/ML interviews
-- Syntax: func() OVER (PARTITION BY ... ORDER BY ... frame)
-- ============================================================

-- ── RANKING FUNCTIONS ────────────────────────────────────────

-- 1. ROW_NUMBER — unique sequential number, no ties
--    Classic use: pick the latest order per customer
SELECT
  name, department, salary,
  ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num
FROM employees;

-- 2. RANK vs DENSE_RANK — both handle ties differently
--    RANK skips numbers after a tie; DENSE_RANK does not
SELECT
  name, department, salary,
  RANK()        OVER (PARTITION BY department ORDER BY salary DESC) AS rnk,
  DENSE_RANK()  OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rnk
FROM employees;
-- Bob and Carol both earn 90000 → both get RANK=2 but RANK skips 3; DENSE_RANK gives 2 then 3

-- 3. Get top-1 earner per department (filter on window result via subquery)
SELECT name, department, salary
FROM (
  SELECT name, department, salary,
         RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
  FROM employees
)
WHERE rnk = 1;

-- 4. NTILE — bucket rows into N equal groups
--    Divide employees into 3 salary tiers
SELECT
  name, salary,
  NTILE(3) OVER (ORDER BY salary DESC) AS salary_tier   -- 1=top, 3=bottom
FROM employees;

-- 5. PERCENT_RANK — percentile of each row (0 to 1)
SELECT
  name, salary,
  ROUND(PERCENT_RANK() OVER (ORDER BY salary), 3) AS percentile
FROM employees;

-- ── LAG & LEAD ───────────────────────────────────────────────

-- 6. LAG — access previous row's value within a partition
--    Per-customer: compare each order amount to the previous one
SELECT
  customer_id,
  order_date,
  amount,
  LAG(amount)  OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_amount,
  amount - LAG(amount, 1, 0) OVER (PARTITION BY customer_id ORDER BY order_date) AS change
FROM orders
ORDER BY customer_id, order_date;

-- 7. LEAD — access next row's value
--    Show the next order date for each customer
SELECT
  customer_id,
  order_date,
  amount,
  LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_date
FROM orders
ORDER BY customer_id, order_date;

-- 8. LAG with offset=2 and a default value when no prior row exists
SELECT
  customer_id,
  order_date,
  amount,
  LAG(amount, 2, 0) OVER (PARTITION BY customer_id ORDER BY order_date) AS two_orders_ago
FROM orders
ORDER BY customer_id, order_date;

-- ── FIRST_VALUE & LAST_VALUE ─────────────────────────────────

-- 9. FIRST_VALUE — first value in the partition window
--    Show each employee's salary vs the highest-paid in their department
SELECT
  name, department, salary,
  FIRST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC) AS dept_max_sal,
  FIRST_VALUE(name)   OVER (PARTITION BY department ORDER BY salary DESC) AS top_earner
FROM employees;

-- 10. LAST_VALUE — needs explicit frame or it only sees up to current row
SELECT
  name, department, salary,
  LAST_VALUE(salary) OVER (
    PARTITION BY department
    ORDER BY salary DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- full partition
  ) AS dept_min_sal
FROM employees;

-- ── AGGREGATE WINDOW FUNCTIONS ────────────────────────────────

-- 11. Running total — SUM with cumulative frame
SELECT
  order_date,
  revenue,
  SUM(revenue) OVER (ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM daily_revenue;

-- 12. 3-day moving average (1 day before + current + 1 day after)
SELECT
  order_date,
  revenue,
  ROUND(AVG(revenue) OVER (ORDER BY order_date
    ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING), 2) AS moving_avg_3day
FROM daily_revenue;

-- 13. Running count and running average
SELECT
  order_date,
  revenue,
  COUNT(*)  OVER (ORDER BY order_date ROWS UNBOUNDED PRECEDING) AS running_count,
  ROUND(AVG(revenue) OVER (ORDER BY order_date ROWS UNBOUNDED PRECEDING), 2) AS running_avg
FROM daily_revenue;

-- 14. SUM without ORDER BY = partition total (same value for every row in group)
--     Useful for computing each row's % share of its group
SELECT
  customer_id,
  order_id,
  amount,
  SUM(amount) OVER (PARTITION BY customer_id) AS customer_total,
  ROUND(amount * 100.0 / SUM(amount) OVER (PARTITION BY customer_id), 1) AS pct_of_customer
FROM orders
ORDER BY customer_id, amount DESC;

-- ── WINDOW FRAME SYNTAX REFERENCE ────────────────────────────

-- 15. Frame clause options — run each to see the difference
--     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  → cumulative
--     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW          → last 3 rows
--     ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING          → centered 3-row window
--     ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING  → suffix sum

SELECT
  order_date,
  revenue,
  SUM(revenue) OVER (ORDER BY order_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS sum_last_3
FROM daily_revenue;
