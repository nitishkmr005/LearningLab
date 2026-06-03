-- ============================================================
-- INTERVIEW PROBLEMS — 10 classic SQL interview questions
-- Each is a self-contained problem with explanation
-- ============================================================

-- ── PROBLEM 1 ─────────────────────────────────────────────────
-- "Find the second highest salary in the company"
-- Trap: LIMIT 1 OFFSET 1 breaks when there are ties

-- Approach 1: DENSE_RANK (handles ties correctly)
SELECT salary
FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
  FROM employees
)
WHERE rnk = 2;

-- Approach 2: subquery (no window functions)
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- ── PROBLEM 2 ─────────────────────────────────────────────────
-- "Find the top-earning employee in each department"
-- Trap: LIMIT 1 does not work per-group

SELECT name, department, salary
FROM (
  SELECT name, department, salary,
         RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
  FROM employees
)
WHERE rnk = 1;   -- RANK not ROW_NUMBER, so ties both appear

-- ── PROBLEM 3 ─────────────────────────────────────────────────
-- "Find customers who placed orders in every month of Q1 2024"
-- Pattern: count distinct months = 3 (all of Q1)

SELECT customer_id
FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31'
  AND status = 'completed'
GROUP BY customer_id
HAVING COUNT(DISTINCT STRFTIME('%m', order_date)) = 3;

-- ── PROBLEM 4 ─────────────────────────────────────────────────
-- "Find the running revenue total, reset at the start of each month"

SELECT
  order_date,
  STRFTIME('%Y-%m', order_date) AS month,
  amount,
  SUM(amount) OVER (
    PARTITION BY STRFTIME('%Y-%m', order_date)    -- reset per month
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS monthly_running_total
FROM orders
WHERE status = 'completed'
ORDER BY order_date;

-- ── PROBLEM 5 ─────────────────────────────────────────────────
-- "Calculate 7-day moving average of daily revenue"

SELECT
  order_date,
  revenue,
  ROUND(AVG(revenue) OVER (
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW    -- current + 6 prior days = 7
  ), 2) AS moving_avg_7day
FROM daily_revenue
ORDER BY order_date;

-- ── PROBLEM 6 ─────────────────────────────────────────────────
-- "Find employees who earn more than the average salary of their department"

SELECT name, department, salary
FROM employees e
WHERE salary > (
  SELECT AVG(salary)
  FROM employees
  WHERE department = e.department    -- correlated: references outer query
)
ORDER BY department, salary DESC;

-- ── PROBLEM 7 ─────────────────────────────────────────────────
-- "For each customer, find the date of their first and most recent order"

SELECT
  c.name,
  MIN(o.order_date) AS first_order,
  MAX(o.order_date) AS latest_order,
  COUNT(o.order_id) AS total_orders,
  ROUND(JULIANDAY(MAX(o.order_date)) - JULIANDAY(MIN(o.order_date)), 0) AS days_as_customer
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY days_as_customer DESC;

-- ── PROBLEM 8 ─────────────────────────────────────────────────
-- "Flag customers as 'new', 'returning', or 'churned'"
-- new: only 1 order, returning: 2+ orders in last 60 days, churned: no recent order

WITH customer_stats AS (
  SELECT
    customer_id,
    COUNT(*) AS order_count,
    MAX(order_date) AS last_order_date
  FROM orders WHERE status = 'completed'
  GROUP BY customer_id
)
SELECT
  c.name,
  COALESCE(cs.order_count, 0) AS orders,
  cs.last_order_date,
  CASE
    WHEN cs.customer_id IS NULL THEN 'never purchased'
    WHEN cs.order_count = 1     THEN 'new'
    WHEN JULIANDAY('2024-04-01') - JULIANDAY(cs.last_order_date) <= 60
         THEN 'returning'
    ELSE 'churned'
  END AS segment
FROM customers c
LEFT JOIN customer_stats cs ON c.id = cs.customer_id
ORDER BY segment;

-- ── PROBLEM 9 ─────────────────────────────────────────────────
-- "Pivot: show order count per customer per channel as separate columns"

SELECT
  customer_id,
  COUNT(CASE WHEN channel = 'web'    THEN 1 END) AS web_orders,
  COUNT(CASE WHEN channel = 'mobile' THEN 1 END) AS mobile_orders,
  COUNT(CASE WHEN channel = 'email'  THEN 1 END) AS email_orders,
  COUNT(*) AS total_orders
FROM orders
GROUP BY customer_id
ORDER BY customer_id;

-- ── PROBLEM 10 ────────────────────────────────────────────────
-- "Find pairs of customers who ordered on the same date"
-- Classic self-join problem

SELECT
  a.customer_id AS customer_a,
  b.customer_id AS customer_b,
  a.order_date
FROM orders a
JOIN orders b ON a.order_date = b.order_date
            AND a.customer_id < b.customer_id    -- < avoids (A,B) and (B,A) duplicates
ORDER BY a.order_date;
