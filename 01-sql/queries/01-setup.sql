-- ============================================================
-- SETUP — Run this FIRST before any other query file
-- Creates all tables used across every query file
-- Cmd+Enter on each block, or select all → Run
-- ============================================================

-- ── EMPLOYEES (ranking, aggregation, hierarchy) ──────────────
DROP TABLE IF EXISTS employees;
CREATE TABLE employees (
  id          INTEGER PRIMARY KEY,
  name        TEXT,
  department  TEXT,
  salary      INTEGER,
  manager_id  INTEGER
);
INSERT INTO employees VALUES
  (1, 'Alice', 'Engineering', 100000, NULL),
  (2, 'Bob',   'Engineering',  90000, 1),
  (3, 'Carol', 'Engineering',  90000, 1),
  (4, 'Dave',  'Marketing',    80000, NULL),
  (5, 'Eve',   'Marketing',    70000, 4),
  (6, 'Frank', 'Marketing',    70000, 4),
  (7, 'Grace', 'Engineering',  75000, 2);

-- ── CUSTOMERS (JOINs, aggregation) ───────────────────────────
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
  id      INTEGER PRIMARY KEY,
  name    TEXT,
  email   TEXT,
  status  TEXT
);
INSERT INTO customers VALUES
  (101, 'Alice', 'alice@co.com', 'active'),
  (102, 'Bob',   'bob@co.com',   'active'),
  (103, 'Carol', 'carol@co.com', 'inactive'),
  (104, 'Dave',  'dave@co.com',  'active'),
  (105, 'Eve',   'eve@co.com',   'active');

-- ── ORDERS (aggregation, JOINs, analytical patterns) ─────────
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
  order_id    INTEGER PRIMARY KEY,
  customer_id INTEGER,
  order_date  TEXT,
  amount      REAL,
  channel     TEXT,
  status      TEXT,
  region      TEXT
);
INSERT INTO orders VALUES
  (1, 101, '2024-01-05', 250, 'web',    'completed', 'us-east'),
  (2, 102, '2024-01-10', 150, 'mobile', 'completed', 'us-west'),
  (3, 101, '2024-02-03', 300, 'web',    'completed', 'us-east'),
  (4, 103, '2024-02-15', 500, 'web',    'refunded',  'us-east'),
  (5, 102, '2024-02-20', 200, 'email',  'completed', 'us-west'),
  (6, 104, '2024-03-01', 175, 'mobile', 'completed', 'us-east'),
  (7, 101, '2024-03-08', 400, 'web',    'completed', 'us-east'),
  (8, 103, '2024-03-12', 320, 'mobile', 'completed', 'us-west');

-- ── DAILY_REVENUE (window frame examples) ────────────────────
DROP TABLE IF EXISTS daily_revenue;
CREATE TABLE daily_revenue (
  order_date  TEXT PRIMARY KEY,
  revenue     REAL
);
INSERT INTO daily_revenue VALUES
  ('2024-01-01', 200),
  ('2024-01-02', 150),
  ('2024-01-03', 300),
  ('2024-01-04', 100),
  ('2024-01-05', 250),
  ('2024-01-06', 180),
  ('2024-01-07', 320);

-- ── USER_EVENTS (funnel, sessionization, LAG/LEAD) ───────────
DROP TABLE IF EXISTS user_events;
CREATE TABLE user_events (
  user_id     INTEGER,
  event_type  TEXT,
  event_ts    TEXT
);
INSERT INTO user_events VALUES
  (101, 'login',        '2024-01-01 09:00:00'),
  (101, 'product_view', '2024-01-01 09:05:00'),
  (101, 'add_to_cart',  '2024-01-01 09:10:00'),
  (101, 'purchase',     '2024-01-01 09:15:00'),
  (101, 'login',        '2024-01-01 14:00:00'),
  (101, 'product_view', '2024-01-01 14:05:00'),
  (102, 'login',        '2024-01-01 10:00:00'),
  (102, 'product_view', '2024-01-01 10:05:00'),
  (102, 'add_to_cart',  '2024-01-01 10:12:00'),
  (102, 'purchase',     '2024-01-01 10:20:00'),
  (103, 'login',        '2024-01-02 09:00:00'),
  (103, 'product_view', '2024-01-02 09:10:00'),
  (104, 'login',        '2024-01-02 11:00:00'),
  (104, 'product_view', '2024-01-02 11:05:00'),
  (104, 'add_to_cart',  '2024-01-02 11:15:00');

-- ── LOGINS (streak problem) ───────────────────────────────────
DROP TABLE IF EXISTS logins;
CREATE TABLE logins (
  user_id   INTEGER,
  login_ts  TEXT
);
INSERT INTO logins VALUES
  (101, '2024-01-01 08:00:00'),
  (101, '2024-01-01 20:00:00'),
  (101, '2024-01-02 09:00:00'),
  (101, '2024-01-03 10:00:00'),
  (101, '2024-01-05 11:00:00'),
  (102, '2024-01-01 07:00:00'),
  (102, '2024-01-02 08:00:00'),
  (102, '2024-01-04 09:00:00');

-- ── USERS + EVENTS (cohort analysis) ─────────────────────────
DROP TABLE IF EXISTS users;
CREATE TABLE users (
  user_id     INTEGER PRIMARY KEY,
  signup_date TEXT
);
INSERT INTO users VALUES
  (101, '2024-01-05'),
  (102, '2024-01-12'),
  (103, '2024-01-20'),
  (104, '2024-02-03'),
  (105, '2024-02-10'),
  (106, '2024-02-15');

DROP TABLE IF EXISTS events;
CREATE TABLE events (
  user_id     INTEGER,
  event_date  TEXT
);
INSERT INTO events VALUES
  (101, '2024-01-08'), (101, '2024-02-05'), (101, '2024-03-08'),
  (102, '2024-01-15'), (102, '2024-02-20'),
  (103, '2024-01-22'),
  (104, '2024-02-07'), (104, '2024-03-01'),
  (105, '2024-02-12'), (105, '2024-03-05'),
  (106, '2024-02-18');

-- ── PRODUCTS (for set operations + NULLs) ────────────────────
DROP TABLE IF EXISTS products;
CREATE TABLE products (
  product_id  INTEGER PRIMARY KEY,
  name        TEXT,
  category    TEXT,
  price       REAL,
  stock       INTEGER
);
INSERT INTO products VALUES
  (1,  'Laptop',     'Electronics', 1200.00, 50),
  (2,  'Mouse',      'Electronics',   45.00, 200),
  (3,  'Keyboard',   'Electronics',  110.00, 150),
  (4,  'Monitor',    'Electronics',  450.00, 80),
  (5,  'Headphones', 'Electronics',  200.00, 120),
  (6,  'Desk',       'Furniture',    350.00, 30),
  (7,  'Chair',      'Furniture',    280.00, NULL),
  (8,  'Lamp',       'Furniture',     60.00, 0),
  (9,  'Notebook',   'Stationery',    10.00, 500),
  (10, 'Pen',        'Stationery',     2.00, NULL);

-- ── VERIFY all tables ─────────────────────────────────────────
SELECT 'employees'    AS tbl, COUNT(*) AS rows FROM employees    UNION ALL
SELECT 'customers',          COUNT(*)          FROM customers    UNION ALL
SELECT 'orders',             COUNT(*)          FROM orders       UNION ALL
SELECT 'daily_revenue',      COUNT(*)          FROM daily_revenue UNION ALL
SELECT 'user_events',        COUNT(*)          FROM user_events  UNION ALL
SELECT 'logins',             COUNT(*)          FROM logins       UNION ALL
SELECT 'users',              COUNT(*)          FROM users        UNION ALL
SELECT 'events',             COUNT(*)          FROM events       UNION ALL
SELECT 'products',           COUNT(*)          FROM products;
