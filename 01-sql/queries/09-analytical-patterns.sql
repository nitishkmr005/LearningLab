-- ============================================================
-- ANALYTICAL PATTERNS — The patterns that appear in every DS take-home
-- Cohort analysis, retention, funnel, sessionization, streak
-- ============================================================

-- ── COHORT ANALYSIS ───────────────────────────────────────────

-- 1. Assign each user to their signup month (cohort)
WITH cohorts AS (
  SELECT
    user_id,
    STRFTIME('%Y-%m', signup_date) AS cohort_month   -- group users by signup month
  FROM users
)
SELECT cohort_month, COUNT(*) AS cohort_size
FROM cohorts
GROUP BY cohort_month
ORDER BY cohort_month;

-- 2. Monthly retention — for each cohort, count active users by month offset
--    month_offset = 0 means the signup month itself
WITH cohorts AS (
  SELECT user_id, STRFTIME('%Y-%m', signup_date) AS cohort_month
  FROM users
),
activity AS (
  SELECT
    e.user_id,
    c.cohort_month,
    STRFTIME('%Y-%m', e.event_date) AS activity_month,
    -- months since signup (approximation using year+month difference)
    (STRFTIME('%Y', e.event_date) - STRFTIME('%Y', c.cohort_month || '-01')) * 12
    + (STRFTIME('%m', e.event_date) - STRFTIME('%m', c.cohort_month || '-01'))
    AS month_offset
  FROM events e
  JOIN cohorts c ON e.user_id = c.user_id
)
SELECT
  cohort_month,
  month_offset,
  COUNT(DISTINCT user_id) AS active_users
FROM activity
GROUP BY cohort_month, month_offset
ORDER BY cohort_month, month_offset;

-- ── FUNNEL ANALYSIS ───────────────────────────────────────────

-- 3. Count users at each funnel step (login → view → cart → purchase)
--    Shows where users drop off
SELECT
  COUNT(DISTINCT CASE WHEN event_type = 'login'        THEN user_id END) AS step1_login,
  COUNT(DISTINCT CASE WHEN event_type = 'product_view' THEN user_id END) AS step2_view,
  COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart'  THEN user_id END) AS step3_cart,
  COUNT(DISTINCT CASE WHEN event_type = 'purchase'     THEN user_id END) AS step4_purchase
FROM user_events;

-- 4. Funnel with conversion rates between steps
WITH funnel AS (
  SELECT
    COUNT(DISTINCT CASE WHEN event_type = 'login'        THEN user_id END) AS login_users,
    COUNT(DISTINCT CASE WHEN event_type = 'product_view' THEN user_id END) AS view_users,
    COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart'  THEN user_id END) AS cart_users,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase'     THEN user_id END) AS purchase_users
  FROM user_events
)
SELECT
  login_users,
  view_users,
  cart_users,
  purchase_users,
  ROUND(view_users    * 100.0 / login_users,    1) AS login_to_view_pct,
  ROUND(cart_users    * 100.0 / view_users,     1) AS view_to_cart_pct,
  ROUND(purchase_users* 100.0 / cart_users,     1) AS cart_to_purchase_pct,
  ROUND(purchase_users* 100.0 / login_users,    1) AS overall_conversion_pct
FROM funnel;

-- ── SESSIONIZATION ────────────────────────────────────────────

-- 5. Define sessions: new session if > 30 minutes gap from last event
--    Step 1: flag each event as session start (1) or continuation (0)
WITH session_starts AS (
  SELECT
    user_id,
    event_ts,
    event_type,
    LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts,
    CASE
      WHEN (JULIANDAY(event_ts) - JULIANDAY(
        LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts)
      )) * 1440 > 30
      OR LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) IS NULL
      THEN 1 ELSE 0
    END AS is_new_session   -- 1 = this event starts a new session
  FROM user_events
),
-- Step 2: assign session ID = cumulative sum of session start flags
sessions AS (
  SELECT *,
    SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY event_ts) AS session_id
  FROM session_starts
)
SELECT user_id, session_id, event_type, event_ts
FROM sessions
ORDER BY user_id, event_ts;

-- 6. Session summary — duration and event count per session
WITH session_starts AS (
  SELECT user_id, event_ts,
    CASE WHEN (JULIANDAY(event_ts) - JULIANDAY(LAG(event_ts)
      OVER (PARTITION BY user_id ORDER BY event_ts))) * 1440 > 30
      OR LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) IS NULL
      THEN 1 ELSE 0 END AS is_new_session
  FROM user_events
),
sessions AS (
  SELECT *, SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY event_ts) AS session_id
  FROM session_starts
)
SELECT
  user_id,
  session_id,
  COUNT(*) AS events_in_session,
  MIN(event_ts) AS session_start,
  MAX(event_ts) AS session_end,
  ROUND((JULIANDAY(MAX(event_ts)) - JULIANDAY(MIN(event_ts))) * 1440, 1) AS duration_minutes
FROM sessions
GROUP BY user_id, session_id
ORDER BY user_id, session_id;

-- ── LOGIN STREAK ──────────────────────────────────────────────

-- 7. Find the longest consecutive login streak per user
--    Key insight: subtract ROW_NUMBER from date → same value for consecutive days
WITH daily_logins AS (
  SELECT DISTINCT user_id, DATE(login_ts) AS login_date
  FROM logins
),
streak_groups AS (
  SELECT
    user_id,
    login_date,
    DATE(login_date, '-' || ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) || ' days')
    AS streak_group    -- consecutive dates map to the same anchor date
  FROM daily_logins
)
SELECT
  user_id,
  streak_group,
  COUNT(*) AS streak_length,
  MIN(login_date) AS streak_start,
  MAX(login_date) AS streak_end
FROM streak_groups
GROUP BY user_id, streak_group
ORDER BY user_id, streak_length DESC;

-- ── PERIOD-OVER-PERIOD COMPARISON ─────────────────────────────

-- 8. Month-over-month revenue change using LAG
WITH monthly AS (
  SELECT
    STRFTIME('%Y-%m', order_date) AS month,
    SUM(amount) AS revenue
  FROM orders WHERE status = 'completed'
  GROUP BY month
)
SELECT
  month,
  revenue,
  LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
  ROUND((revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0
        / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;
