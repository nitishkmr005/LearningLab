-- ============================================================
-- SUBQUERIES, CTEs & RECURSIVE CTEs
-- ============================================================

-- ── SUBQUERIES ───────────────────────────────────────────────

-- 1. Scalar subquery — returns one value, used like a column
--    Show each employee's salary vs the company average
SELECT
  name,
  department,
  salary,
  (SELECT AVG(salary) FROM employees) AS company_avg,
  salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees
ORDER BY diff_from_avg DESC;

-- 2. Row subquery in WHERE — find employees with above-average salary
SELECT name, department, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees)
ORDER BY salary DESC;

-- 3. Correlated subquery — subquery references outer table (runs once per row)
--    For each employee, show the max salary in their department
SELECT
  name,
  department,
  salary,
  (SELECT MAX(salary) FROM employees e2
   WHERE e2.department = e1.department) AS dept_max   -- references outer e1
FROM employees e1
ORDER BY department, salary DESC;

-- 4. EXISTS — check if at least one matching row exists
--    Find customers who have placed at least one web order
SELECT c.name, c.email
FROM customers c
WHERE EXISTS (
  SELECT 1 FROM orders o
  WHERE o.customer_id = c.id
    AND o.channel = 'web'
);

-- 5. IN with subquery — same as above but different approach
SELECT name, email
FROM customers
WHERE id IN (
  SELECT DISTINCT customer_id FROM orders WHERE channel = 'web'
);

-- ── CTEs (Common Table Expressions) ──────────────────────────

-- 6. Basic CTE — cleaner than a subquery, same performance
WITH completed_orders AS (
  SELECT customer_id, SUM(amount) AS total_spent, COUNT(*) AS order_count
  FROM orders
  WHERE status = 'completed'
  GROUP BY customer_id
)
SELECT c.name, co.total_spent, co.order_count
FROM customers c
JOIN completed_orders co ON c.id = co.customer_id
ORDER BY co.total_spent DESC;

-- 7. Multiple CTEs chained — each CTE can reference the ones above it
WITH dept_stats AS (
  SELECT department, AVG(salary) AS avg_sal, MAX(salary) AS max_sal
  FROM employees
  GROUP BY department
),
above_avg AS (
  SELECT e.name, e.department, e.salary, d.avg_sal
  FROM employees e
  JOIN dept_stats d ON e.department = d.department
  WHERE e.salary > d.avg_sal
)
SELECT name, department, salary, ROUND(avg_sal, 0) AS dept_avg
FROM above_avg
ORDER BY department, salary DESC;

-- 8. CTE used multiple times in the same query
--    Compute revenue quartiles and compare each order to its quartile boundary
WITH order_stats AS (
  SELECT
    AVG(amount)  AS avg_amount,
    MIN(amount)  AS min_amount,
    MAX(amount)  AS max_amount
  FROM orders WHERE status = 'completed'
)
SELECT
  o.order_id,
  o.channel,
  o.amount,
  s.avg_amount,
  CASE
    WHEN o.amount >= s.avg_amount * 1.5 THEN 'High'
    WHEN o.amount >= s.avg_amount       THEN 'Medium'
    ELSE                                     'Low'
  END AS order_tier
FROM orders o
CROSS JOIN order_stats s
WHERE o.status = 'completed'
ORDER BY o.amount DESC;

-- ── RECURSIVE CTEs ────────────────────────────────────────────

-- 9. Recursive CTE — traverse manager hierarchy top-down
--    Returns every employee with their level in the org tree
WITH RECURSIVE org_tree AS (
  -- Anchor: start with top-level managers (no manager)
  SELECT id, name, manager_id, 0 AS depth, name AS path
  FROM employees
  WHERE manager_id IS NULL

  UNION ALL

  -- Recursive step: join each employee to their manager row
  SELECT e.id, e.name, e.manager_id,
         ot.depth + 1,
         ot.path || ' → ' || e.name
  FROM employees e
  JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT depth, path, name
FROM org_tree
ORDER BY path;

-- 10. Recursive CTE — generate a number series (1 to 7)
--     Useful for generating date spines or test data
WITH RECURSIVE num_series AS (
  SELECT 1 AS n                  -- anchor: start at 1
  UNION ALL
  SELECT n + 1 FROM num_series   -- recursive step: add 1
  WHERE n < 7                    -- termination: stop at 7
)
SELECT n FROM num_series;

-- 11. Recursive CTE — date spine (every day in a range)
--     Crucial for cohort/retention queries where missing dates matter
WITH RECURSIVE date_spine AS (
  SELECT '2024-01-01' AS dt
  UNION ALL
  SELECT DATE(dt, '+1 day') FROM date_spine
  WHERE dt < '2024-01-07'
)
SELECT dt FROM date_spine;
