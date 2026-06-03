-- ============================================================
-- VIEWS, INDEXES & QUERY OPTIMIZATION
-- ============================================================

-- ── VIEWS ─────────────────────────────────────────────────────

-- 1. Create a view — stored query, queried like a table
--    Encapsulates the join logic so callers don't need to know it
DROP VIEW IF EXISTS customer_order_summary;
CREATE VIEW customer_order_summary AS
SELECT
  c.id          AS customer_id,
  c.name,
  c.status,
  COUNT(o.order_id)  AS total_orders,
  SUM(o.amount)      AS lifetime_value,
  MAX(o.order_date)  AS last_order_date
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.status;

-- 2. Query the view exactly like a table
SELECT * FROM customer_order_summary ORDER BY lifetime_value DESC;

-- 3. Filter on a view
SELECT name, lifetime_value
FROM customer_order_summary
WHERE status = 'active' AND total_orders >= 2
ORDER BY lifetime_value DESC;

-- 4. Create a view for completed orders (hides complexity from downstream)
DROP VIEW IF EXISTS completed_orders;
CREATE VIEW completed_orders AS
SELECT * FROM orders WHERE status = 'completed';

SELECT channel, SUM(amount) AS revenue
FROM completed_orders
GROUP BY channel
ORDER BY revenue DESC;

-- ── INDEXES ───────────────────────────────────────────────────

-- 5. Create an index on a frequently filtered column
--    Without index: full table scan for every WHERE order_date = ...
--    With index: seeks directly to matching rows
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date        ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_status      ON orders(status);

-- Composite index — optimal when you filter on both columns together
CREATE INDEX IF NOT EXISTS idx_orders_status_date ON orders(status, order_date);

-- 6. EXPLAIN QUERY PLAN — see how SQLite executes a query
EXPLAIN QUERY PLAN
SELECT * FROM orders WHERE customer_id = 101 ORDER BY order_date;
-- Should show: SEARCH orders USING INDEX idx_orders_customer_id

-- 7. Without index vs with index comparison
EXPLAIN QUERY PLAN
SELECT * FROM employees WHERE department = 'Engineering';
-- SCAN employees (no index) — reads all rows

CREATE INDEX IF NOT EXISTS idx_emp_dept ON employees(department);

EXPLAIN QUERY PLAN
SELECT * FROM employees WHERE department = 'Engineering';
-- Now: SEARCH employees USING INDEX idx_emp_dept

-- ── QUERY OPTIMIZATION PATTERNS ──────────────────────────────

-- 8. Push filters into CTEs/subqueries early — reduce rows before joining
--    BAD: join everything then filter (joins a large cross-product first)
--    GOOD: filter first, then join smaller result sets

-- Filter early pattern
WITH recent_completed AS (
  SELECT customer_id, amount
  FROM orders
  WHERE status = 'completed'             -- filter inside CTE
    AND order_date >= '2024-02-01'
)
SELECT c.name, SUM(rc.amount) AS recent_spend
FROM customers c
JOIN recent_completed rc ON c.id = rc.customer_id
GROUP BY c.name;

-- 9. Avoid SELECT * in production — only fetch columns you need
--    This query fetches all columns unnecessarily
-- BAD:  SELECT * FROM orders WHERE status = 'completed';
-- GOOD:
SELECT order_id, customer_id, amount FROM orders WHERE status = 'completed';

-- 10. Use EXISTS instead of COUNT(*) > 0 for existence checks
--     EXISTS short-circuits on first match; COUNT scans all matching rows
-- BAD:
SELECT name FROM customers
WHERE (SELECT COUNT(*) FROM orders o WHERE o.customer_id = customers.id) > 0;

-- GOOD:
SELECT name FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);

-- 11. Avoid functions on indexed columns in WHERE — they disable the index
-- BAD (function wraps the column — index not used):
SELECT * FROM orders WHERE STRFTIME('%Y', order_date) = '2024';

-- GOOD (range filter — index is used):
SELECT * FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';

-- 12. LIMIT with ORDER BY — ensure order is deterministic (tie-breaking)
--     Without a unique tie-breaker, results can vary per run
SELECT customer_id, amount FROM orders
ORDER BY amount DESC, order_id ASC    -- order_id as tie-breaker
LIMIT 3;
