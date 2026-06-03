-- ============================================================
-- JOINS — Every type with the gotchas interviewers probe
-- ============================================================

-- 1. INNER JOIN — only rows that match in BOTH tables
--    Customer 105 (Eve) has no orders → excluded
SELECT c.name, o.order_id, o.amount, o.order_date
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
ORDER BY c.name;

-- 2. LEFT JOIN — all customers, NULLs for those with no orders
--    Customer 105 (Eve) appears with NULL order columns
SELECT c.name, c.status, o.order_id, o.amount
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
ORDER BY c.name;

-- 3. Find customers with NO orders (anti-join using LEFT JOIN)
--    Classic interview question: "find users who never purchased"
SELECT c.name, c.email
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.order_id IS NULL;       -- NULL in right table = no match found

-- 4. Anti-join using NOT EXISTS (same result, often faster)
SELECT c.name, c.email
FROM customers c
WHERE NOT EXISTS (
  SELECT 1 FROM orders o WHERE o.customer_id = c.id
);

-- 5. Anti-join using NOT IN (careful: fails silently if subquery returns a NULL)
SELECT name, email
FROM customers
WHERE id NOT IN (
  SELECT DISTINCT customer_id FROM orders WHERE customer_id IS NOT NULL
);

-- 6. SELF JOIN — join a table to itself
--    Find each employee's manager name (manager_id references id in same table)
SELECT
  e.name        AS employee,
  e.salary,
  m.name        AS manager,
  m.salary      AS manager_salary
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id   -- same table aliased twice
ORDER BY m.name NULLS LAST;

-- 7. Find employees earning MORE than their manager (classic problem)
SELECT
  e.name    AS employee,
  e.salary  AS emp_salary,
  m.name    AS manager,
  m.salary  AS mgr_salary
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;

-- 8. CROSS JOIN — every combination of rows (cartesian product)
--    Useful for generating combinations; use with caution on large tables
SELECT
  c.name AS customer,
  p.name AS product
FROM customers c
CROSS JOIN products p
WHERE c.status = 'active'
  AND p.category = 'Electronics'
ORDER BY c.name, p.name;

-- 9. Joining on multiple conditions
--    Match orders to customers AND restrict to completed orders in join itself
SELECT c.name, o.order_id, o.amount, o.status
FROM customers c
JOIN orders o ON c.id = o.customer_id
           AND o.status = 'completed'   -- filter pushed into JOIN condition
ORDER BY c.name;

-- 10. Joining aggregated results (inline subquery as a derived table)
--     Find each customer's total spend and compare to average
SELECT
  c.name,
  agg.total_spent,
  ROUND(avg_spent.avg_across_all, 2) AS avg_customer_spend,
  CASE WHEN agg.total_spent > avg_spent.avg_across_all THEN 'above average'
       ELSE 'below average' END AS segment
FROM customers c
JOIN (
  SELECT customer_id, SUM(amount) AS total_spent
  FROM orders
  GROUP BY customer_id
) agg ON c.id = agg.customer_id
CROSS JOIN (
  SELECT AVG(total) AS avg_across_all
  FROM (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id)
) avg_spent
ORDER BY agg.total_spent DESC;

-- 11. FULL OUTER JOIN — all rows from both sides, NULLs where no match
--     SQLite supports this via UNION of LEFT JOIN + anti-join
SELECT c.name AS customer, o.order_id, o.amount
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
UNION ALL
SELECT NULL, o.order_id, o.amount
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.id
WHERE c.id IS NULL;
