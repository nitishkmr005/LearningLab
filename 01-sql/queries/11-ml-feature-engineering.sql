-- ============================================================
-- SQL FOR ML — Feature engineering patterns for model training
-- These are the SQL patterns that feed features into ML models
-- ============================================================

-- ── ENTITY FEATURES (point-in-time safe) ─────────────────────

-- 1. Customer-level aggregate features (RFM: Recency, Frequency, Monetary)
--    RFM is one of the most common feature sets for customer ML models
SELECT
  c.id AS customer_id,
  -- Recency: how many days since last order
  ROUND(JULIANDAY('2024-04-01') - JULIANDAY(MAX(o.order_date)), 0) AS days_since_last_order,
  -- Frequency: number of orders
  COUNT(o.order_id) AS order_count,
  -- Monetary: total spend
  SUM(o.amount) AS lifetime_value,
  -- Average order value
  ROUND(AVG(o.amount), 2) AS avg_order_value,
  -- Max single order
  MAX(o.amount) AS max_order_value,
  -- Spend std dev (spread of order values)
  ROUND(SUM((o.amount - avg_amt.avg) * (o.amount - avg_amt.avg)) / NULLIF(COUNT(*) - 1, 0), 2) AS spend_variance
FROM customers c
JOIN orders o ON c.id = o.customer_id AND o.status = 'completed'
JOIN (SELECT customer_id, AVG(amount) AS avg FROM orders GROUP BY customer_id) avg_amt
  ON avg_amt.customer_id = c.id
GROUP BY c.id;

-- 2. Channel preference features — one-hot encoded via CASE
SELECT
  customer_id,
  SUM(CASE WHEN channel = 'web'    THEN 1 ELSE 0 END) AS orders_web,
  SUM(CASE WHEN channel = 'mobile' THEN 1 ELSE 0 END) AS orders_mobile,
  SUM(CASE WHEN channel = 'email'  THEN 1 ELSE 0 END) AS orders_email,
  -- Preferred channel (argmax pattern)
  CASE
    WHEN SUM(CASE WHEN channel = 'web'    THEN 1 ELSE 0 END) >=
         SUM(CASE WHEN channel = 'mobile' THEN 1 ELSE 0 END)
         AND
         SUM(CASE WHEN channel = 'web'    THEN 1 ELSE 0 END) >=
         SUM(CASE WHEN channel = 'email'  THEN 1 ELSE 0 END)
    THEN 'web'
    WHEN SUM(CASE WHEN channel = 'mobile' THEN 1 ELSE 0 END) >=
         SUM(CASE WHEN channel = 'email'  THEN 1 ELSE 0 END)
    THEN 'mobile'
    ELSE 'email'
  END AS preferred_channel
FROM orders
GROUP BY customer_id;

-- ── LAG-BASED SEQUENCE FEATURES ──────────────────────────────

-- 3. Per-customer order sequence features (for next-purchase prediction)
SELECT
  customer_id,
  order_id,
  order_date,
  amount,
  -- Time since previous order (days)
  ROUND(JULIANDAY(order_date) - JULIANDAY(
    LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date)
  ), 0) AS days_since_prev_order,
  -- Spend trend: current - previous
  amount - LAG(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS spend_delta,
  -- Cumulative spend at time of order
  SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_spend,
  -- Order number (1st purchase, 2nd, etc.)
  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS order_sequence
FROM orders
WHERE status = 'completed'
ORDER BY customer_id, order_date;

-- ── LABEL GENERATION ─────────────────────────────────────────

-- 4. Binary churn label — did the customer order in the next 60 days?
--    Pattern: for each customer, check if any future order exists within window
WITH last_order AS (
  SELECT customer_id, MAX(order_date) AS last_order_date
  FROM orders WHERE status = 'completed'
  GROUP BY customer_id
),
future_orders AS (
  SELECT DISTINCT customer_id
  FROM orders
  WHERE order_date > '2024-02-01'
    AND order_date <= DATE('2024-02-01', '+60 days')
    AND status = 'completed'
)
SELECT
  lo.customer_id,
  lo.last_order_date,
  CASE WHEN fo.customer_id IS NOT NULL THEN 0 ELSE 1 END AS churn_label
FROM last_order lo
LEFT JOIN future_orders fo ON lo.customer_id = fo.customer_id;

-- 5. Regression label — predicted next order value
WITH ordered AS (
  SELECT
    customer_id,
    order_date,
    amount,
    LEAD(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_amount
  FROM orders WHERE status = 'completed'
)
SELECT customer_id, order_date, amount, next_order_amount
FROM ordered
WHERE next_order_amount IS NOT NULL   -- only rows where label exists
ORDER BY customer_id, order_date;

-- ── NORMALIZATION & BINNING ────────────────────────────────────

-- 6. Min-max normalization (scale values to 0–1)
SELECT
  customer_id,
  lifetime_value,
  ROUND((lifetime_value - MIN(lifetime_value) OVER ()) /
        NULLIF(MAX(lifetime_value) OVER () - MIN(lifetime_value) OVER (), 0), 4) AS ltv_normalized
FROM (
  SELECT customer_id, SUM(amount) AS lifetime_value
  FROM orders GROUP BY customer_id
);

-- 7. Quantile binning — assign each customer to a spend decile
SELECT
  customer_id,
  lifetime_value,
  NTILE(4) OVER (ORDER BY lifetime_value) AS spend_quartile   -- 1=lowest, 4=highest
FROM (
  SELECT customer_id, SUM(amount) AS lifetime_value
  FROM orders GROUP BY customer_id
);
