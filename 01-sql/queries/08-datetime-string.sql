-- ============================================================
-- DATE/TIME & STRING FUNCTIONS
-- SQLite uses TEXT dates (ISO format: YYYY-MM-DD)
-- ============================================================

-- ── DATE FUNCTIONS ────────────────────────────────────────────

-- 1. Extract parts of a date
SELECT
  order_date,
  STRFTIME('%Y', order_date)       AS year,
  STRFTIME('%m', order_date)       AS month,
  STRFTIME('%d', order_date)       AS day,
  STRFTIME('%Y-%m', order_date)    AS year_month   -- group by month
FROM orders
ORDER BY order_date;

-- 2. Date arithmetic — add/subtract days
SELECT
  order_date,
  DATE(order_date, '+7 days')   AS one_week_later,
  DATE(order_date, '-30 days')  AS thirty_days_before
FROM orders
ORDER BY order_date;

-- 3. Days between two dates
SELECT
  order_date,
  JULIANDAY('2024-04-01') - JULIANDAY(order_date) AS days_since_order
FROM orders
ORDER BY order_date;

-- 4. Group revenue by month
SELECT
  STRFTIME('%Y-%m', order_date) AS month,
  COUNT(*)                       AS num_orders,
  SUM(amount)                    AS monthly_revenue
FROM orders
WHERE status = 'completed'
GROUP BY month
ORDER BY month;

-- 5. Filter to last N days (relative to a reference date)
SELECT *
FROM orders
WHERE JULIANDAY('2024-03-31') - JULIANDAY(order_date) <= 30   -- last 30 days
ORDER BY order_date DESC;

-- 6. Day of week analysis (0=Sunday, 6=Saturday in SQLite)
SELECT
  STRFTIME('%w', order_date)  AS day_of_week,
  CASE STRFTIME('%w', order_date)
    WHEN '0' THEN 'Sunday'   WHEN '1' THEN 'Monday'
    WHEN '2' THEN 'Tuesday'  WHEN '3' THEN 'Wednesday'
    WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday'
    ELSE 'Saturday'
  END                         AS day_name,
  COUNT(*)                    AS orders,
  SUM(amount)                 AS revenue
FROM orders
GROUP BY day_of_week
ORDER BY day_of_week;

-- 7. Timestamp arithmetic — seconds between two events
SELECT
  user_id,
  MIN(event_ts)                                               AS session_start,
  MAX(event_ts)                                               AS session_end,
  (JULIANDAY(MAX(event_ts)) - JULIANDAY(MIN(event_ts))) * 86400 AS session_duration_sec
FROM user_events
GROUP BY user_id, DATE(event_ts)
ORDER BY user_id;

-- ── STRING FUNCTIONS ──────────────────────────────────────────

-- 8. UPPER, LOWER, LENGTH
SELECT
  name,
  UPPER(name)   AS name_upper,
  LOWER(email)  AS email_lower,
  LENGTH(name)  AS name_length
FROM customers;

-- 9. SUBSTR — extract part of a string
SELECT
  email,
  SUBSTR(email, INSTR(email, '@') + 1)        AS domain,      -- after @
  SUBSTR(email, 1, INSTR(email, '@') - 1)     AS username     -- before @
FROM customers;

-- 10. INSTR — find position of a character
SELECT
  email,
  INSTR(email, '@') AS at_position
FROM customers;

-- 11. REPLACE — substitute text
SELECT
  email,
  REPLACE(email, '@co.com', '@company.com') AS updated_email
FROM customers;

-- 12. TRIM — remove leading/trailing characters
SELECT
  '  Alice  ' AS raw,
  TRIM('  Alice  ')       AS trimmed,
  LTRIM('  Alice  ')      AS left_trimmed,
  RTRIM('  Alice  ')      AS right_trimmed;

-- 13. LIKE — pattern matching (case-insensitive in SQLite by default)
--     % = any sequence, _ = single character
SELECT name, email
FROM customers
WHERE email LIKE '%@co.com';    -- ends with @co.com

-- 14. String concatenation with ||
SELECT
  name || ' <' || email || '>' AS display_format
FROM customers;

-- 15. PRINTF / FORMAT for zero-padded numbers (SQLite)
SELECT
  order_id,
  PRINTF('ORD-%05d', order_id) AS formatted_id    -- ORD-00001, ORD-00002 ...
FROM orders
ORDER BY order_id;
