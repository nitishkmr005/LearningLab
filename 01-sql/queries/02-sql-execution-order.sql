-- ============================================================
-- SQL EXECUTION ORDER
-- SQL runs clauses in this order regardless of how you write it:
-- FROM → WHERE → GROUP BY → HAVING → WINDOW → SELECT → DISTINCT → ORDER BY → LIMIT
-- ============================================================

-- 1. Basic execution order demo
--    WHERE filters BEFORE aggregation; HAVING filters AFTER
SELECT channel, SUM(amount) AS total
FROM orders
WHERE status = 'completed'       -- step 2: runs BEFORE GROUP BY, removes refunded row
GROUP BY channel                 -- step 3: groups remaining rows
HAVING SUM(amount) > 400         -- step 4: filters groups by aggregate total
ORDER BY total DESC;             -- step 9: sorts final output

-- ── What each step returned ──────────────────────────────────
-- After WHERE: 7 completed orders remain
-- After GROUP BY: web=950, mobile=645, email=200
-- After HAVING: email dropped (200 < 400)
-- Final: web=950, mobile=645

-- 2. WHY you CANNOT use a SELECT alias in WHERE
--    "total" alias is created at step 7 (SELECT), WHERE is step 2
--    This would ERROR: WHERE total > 400
--    Correct fix — use HAVING:
SELECT channel, SUM(amount) AS total
FROM orders
GROUP BY channel
HAVING SUM(amount) > 400;

-- 3. WHY you CANNOT filter a window function in WHERE
--    Window functions run at step 5, WHERE is step 2
--    Fix: wrap in a subquery so the outer WHERE sees the computed column
SELECT * FROM (
  SELECT
    name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
  FROM employees
)
WHERE rnk <= 2;        -- now works because the subquery computed rnk first

-- 4. ORDER BY CAN use SELECT aliases (ORDER BY runs at step 9, after SELECT)
SELECT name, salary * 12 AS annual_salary
FROM employees
ORDER BY annual_salary DESC;   -- alias from SELECT is visible here — this is fine

-- 5. DISTINCT runs after SELECT, before ORDER BY
--    Use it to deduplicate AFTER projecting columns
SELECT DISTINCT department
FROM employees
ORDER BY department;

-- 6. LIMIT runs last — always on the already-sorted result
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 3;               -- top 3 earners
