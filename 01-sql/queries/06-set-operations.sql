-- ============================================================
-- SET OPERATIONS — UNION, INTERSECT, EXCEPT
-- All require same number of columns + compatible types
-- ============================================================

-- Setup: two sets of customer IDs to compare
-- web_buyers: customers who used web channel
-- mobile_buyers: customers who used mobile channel

-- 1. UNION ALL — combine results, keep duplicates (faster than UNION)
--    Use when you know there's no overlap, or duplicates are intentional
SELECT customer_id, 'web'    AS channel FROM orders WHERE channel = 'web'
UNION ALL
SELECT customer_id, 'mobile' AS channel FROM orders WHERE channel = 'mobile'
ORDER BY customer_id;

-- 2. UNION — combine and deduplicate (adds a DISTINCT step, slightly slower)
--    Customers who bought via web OR mobile (no duplicates)
SELECT customer_id FROM orders WHERE channel = 'web'
UNION
SELECT customer_id FROM orders WHERE channel = 'mobile'
ORDER BY customer_id;

-- 3. INTERSECT — rows that appear in BOTH queries
--    Customers who bought via BOTH web AND mobile
SELECT customer_id FROM orders WHERE channel = 'web'
INTERSECT
SELECT customer_id FROM orders WHERE channel = 'mobile'
ORDER BY customer_id;

-- 4. EXCEPT — rows in first query but NOT in second
--    Customers who used web but NEVER used mobile
SELECT customer_id FROM orders WHERE channel = 'web'
EXCEPT
SELECT customer_id FROM orders WHERE channel = 'mobile'
ORDER BY customer_id;

-- 5. UNION to combine different tables for a unified report
--    Stack a summary row below detail rows
SELECT name AS entity, salary AS value, 'employee' AS type
FROM employees
UNION ALL
SELECT department, AVG(salary), 'dept_avg'
FROM employees
GROUP BY department
ORDER BY type, value DESC;

-- 6. Real use case: full-period revenue report across two years
--    Simulated with two filtered queries stacked via UNION ALL
SELECT '2024-Q1' AS period, SUM(amount) AS revenue
FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31'
UNION ALL
SELECT '2024-Q1-web', SUM(amount)
FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31'
  AND channel = 'web'
UNION ALL
SELECT '2024-Q1-mobile', SUM(amount)
FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31'
  AND channel = 'mobile';
