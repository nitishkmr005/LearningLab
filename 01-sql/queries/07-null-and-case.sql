-- ============================================================
-- NULL HANDLING & CASE EXPRESSIONS
-- NULLs are the most common source of silent bugs in SQL
-- ============================================================

-- ── NULL BASICS ───────────────────────────────────────────────

-- 1. NULL comparison ALWAYS returns UNKNOWN (not TRUE or FALSE)
--    You CANNOT use = or != to find NULLs
SELECT name, stock
FROM products
WHERE stock = NULL;       -- returns 0 rows — NULL = NULL is UNKNOWN, not TRUE

-- Correct: use IS NULL
SELECT name, stock
FROM products
WHERE stock IS NULL;      -- returns Chair and Pen

-- 2. IS NOT NULL
SELECT name, stock
FROM products
WHERE stock IS NOT NULL;

-- 3. NULL in arithmetic — any expression with NULL produces NULL
SELECT name, price, stock,
       price * stock AS total_value    -- NULL where stock is NULL
FROM products;

-- ── COALESCE ─────────────────────────────────────────────────

-- 4. COALESCE — returns the FIRST non-NULL value in its list
--    Replace NULL stock with 0 for arithmetic
SELECT name, price,
       COALESCE(stock, 0) AS stock_safe,
       price * COALESCE(stock, 0) AS total_value
FROM products;

-- 5. COALESCE for fallback display values
SELECT name,
       COALESCE(CAST(stock AS TEXT), 'out of stock') AS stock_display
FROM products;

-- ── NULLIF ───────────────────────────────────────────────────

-- 6. NULLIF(a, b) — returns NULL if a = b, otherwise returns a
--    Use to avoid divide-by-zero errors
SELECT
  channel,
  SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) AS completed_revenue,
  COUNT(*) AS total_orders,
  SUM(amount) / NULLIF(COUNT(*), 0) AS avg_per_order   -- NULLIF prevents /0
FROM orders
GROUP BY channel;

-- ── CASE EXPRESSIONS ─────────────────────────────────────────

-- 7. Simple CASE — match a column to specific values
SELECT name, status,
  CASE status
    WHEN 'active'   THEN 'Currently active'
    WHEN 'inactive' THEN 'Churned'
    ELSE                 'Unknown'
  END AS status_label
FROM customers;

-- 8. Searched CASE — evaluate conditions (more flexible)
SELECT name, salary,
  CASE
    WHEN salary >= 100000 THEN 'L5 - Senior'
    WHEN salary >=  85000 THEN 'L4 - Mid'
    WHEN salary >=  70000 THEN 'L3 - Junior'
    ELSE                       'L2 - Entry'
  END AS level
FROM employees
ORDER BY salary DESC;

-- 9. CASE inside aggregate — conditional count (pivot-style)
--    Count orders per status per channel in one query
SELECT
  channel,
  COUNT(*)                                          AS total,
  COUNT(CASE WHEN status = 'completed' THEN 1 END) AS completed,
  COUNT(CASE WHEN status = 'refunded'  THEN 1 END) AS refunded,
  SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) AS completed_revenue
FROM orders
GROUP BY channel
ORDER BY total DESC;

-- 10. CASE to bucket continuous values (histogram bins)
SELECT
  CASE
    WHEN amount <  200 THEN '0-199'
    WHEN amount <  400 THEN '200-399'
    WHEN amount <  600 THEN '400-599'
    ELSE                    '600+'
  END AS amount_bucket,
  COUNT(*) AS order_count,
  ROUND(AVG(amount), 2) AS avg_in_bucket
FROM orders
GROUP BY amount_bucket
ORDER BY MIN(amount);

-- 11. NULL in aggregates — SUM/COUNT/AVG ignore NULLs automatically
--     But COUNT(*) vs COUNT(col) differ!
SELECT
  COUNT(*)     AS total_products,         -- counts ALL rows including NULLs
  COUNT(stock) AS products_with_stock,    -- ignores NULLs in stock column
  AVG(stock)   AS avg_stock_nonull,       -- divides by non-NULL rows only
  SUM(stock)   AS total_stock             -- ignores NULLs in sum
FROM products;

-- 12. NULL-safe comparison — treat NULL as equal (not standard SQL but useful pattern)
--    Find rows where manager_id matches a value OR both are NULL
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id
WHERE e.manager_id IS NULL OR e.manager_id = 1;
