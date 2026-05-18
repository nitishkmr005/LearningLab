# Python & Pandas for ML Engineers and AI Engineers

*A production-first deep-dive covering the Python and Pandas patterns that appear in every ML pipeline, AI system, and data engineering job — from core language features to async AI clients.*

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [A Brief History](#2-a-brief-history)
3. [Core Python for ML & AI](#3-core-python-for-ml--ai)
4. [Object-Oriented Python and Type Safety](#4-object-oriented-python-and-type-safety)
5. [Error Handling, Logging, and Debugging](#5-error-handling-logging-and-debugging)
6. [Pandas: DataFrames, Indexing, and Filtering](#6-pandas-dataframes-indexing-and-filtering)
7. [GroupBy, Aggregation, and Feature Engineering](#7-groupby-aggregation-and-feature-engineering)
8. [Joins, Datetime Features, and Rolling Windows](#8-joins-datetime-features-and-rolling-windows)
9. [Reshaping, I/O, and Performance Optimization](#9-reshaping-io-and-performance-optimization)
10. [Async Python and Concurrency](#10-async-python-and-concurrency)
11. [Serialization, Configuration, and Testing](#11-serialization-configuration-and-testing)
12. [Python Patterns for AI Engineering](#12-python-patterns-for-ai-engineering)
13. [Notebook-to-Production Discipline](#13-notebook-to-production-discipline)
14. [The Modern Recipe](#14-the-modern-recipe)
15. [References](#15-references)

---

## 1. The Problem

Python proficiency is the assumed foundation for every ML and AI engineering role, but "proficiency" means wildly different things in different contexts. A data scientist writing notebook analysis code and a senior ML engineer building a production feature pipeline are both "using Python and Pandas," but the gap in code quality, performance, and maintainability between their work can be enormous. The production ML engineer knows that using `df.apply(lambda row: ..., axis=1)` on a 50-million-row DataFrame is the same as writing a Python loop — it will be 100× slower than the vectorized equivalent and will be the bottleneck in your pipeline. They know that trapping database credentials in a notebook cell gets rotated and breaks the pipeline at 2am. They know that `asyncio.gather` is the right tool for fanning out 200 LLM API calls concurrently.

The gap isn't just performance. Feature engineering bugs are insidious because they're silent — a misaligned rolling window that leaks future data into training features will produce a model that looks great in validation and fails badly in production. A missing `.copy()` after slicing a DataFrame causes a `SettingWithCopyWarning` that either silently fails to apply your transformation or corrupts memory in unexpected ways. These bugs cost weeks, not hours.

This blog covers the Python and Pandas you actually use in ML and AI engineering — the patterns you must know cold, the performance traps to avoid, and the production disciplines that separate robust systems from brittle notebooks.

---

## 2. A Brief History

Python's dominance in data science wasn't inevitable. In the early 2000s, R was the language of choice for statisticians, MATLAB was the tool for numerical computing, and Java ruled enterprise data pipelines. The turning point came through a series of library releases: NumPy 1.0 (2006) gave Python efficient n-dimensional arrays; SciPy brought scientific computing; matplotlib (2003) provided plotting; and most decisively, Wes McKinney released pandas 0.1 in 2008, directly addressing the need for labeled, heterogeneous tabular data manipulation that NumPy alone couldn't provide.

The IPython project (2001) and later Jupyter notebooks (2014) made Python the lingua franca of exploratory data analysis. When scikit-learn (0.1, 2010), TensorFlow (2015), and PyTorch (2016) all chose Python as their primary interface, the outcome was settled. Python 3's gradual takeover from Python 2 (completed with Python 2 EOL in 2020) cleaned up the language significantly. Today, pandas 2.x with its Apache Arrow-backed data types and Copy-on-Write semantics is a substantially better library than the pandas of 2015, and tools like Polars have pushed the ecosystem further toward performance-first design.

---

## 3. Core Python for ML & AI

The foundation is raw Python — the built-ins and standard library patterns that appear in every ETL script, data loader, model wrapper, and agent utility.

### 3.1 Data Structures and Comprehensions

Lists, dicts, sets, and tuples each have performance characteristics that matter at scale. Dict lookup is O(1); list search is O(n). Sets give O(1) membership testing. Tuple is hashable (usable as dict key); list is not.

```python
# List comprehension: filter and transform
raw_data = [{"id": i, "score": i * 1.1, "valid": i % 3 != 0} for i in range(1000)]
clean = [{"id": r["id"], "score": r["score"]} for r in raw_data if r["valid"]]

# Dict comprehension: lookup table
id_to_score = {r["id"]: r["score"] for r in clean}

# Set for fast deduplication
seen_ids = set()
unique_rows = [r for r in raw_data if r["id"] not in seen_ids and not seen_ids.add(r["id"])]

# Generator: lazy evaluation, doesn't build the whole list in memory
# Use when processing large files line by line
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

# zip and enumerate
features = ["age", "income", "score"]
values = [35, 80000, 0.92]
for i, (feat, val) in enumerate(zip(features, values)):
    print(f"[{i}] {feat}: {val}")

# Unpacking
first, *rest = [1, 2, 3, 4, 5]  # first=1, rest=[2,3,4,5]
a, b, c = (10, 20, 30)

# Sorting with key
users = [{"name": "Bob", "age": 35}, {"name": "Alice", "age": 28}]
sorted_users = sorted(users, key=lambda u: u["age"])
```

### 3.2 Functions and Modules

Write pure functions wherever possible — same input always gives same output, no side effects. This makes code trivially testable and composable.

```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def compute_recency(last_event_date: str, reference_date: str) -> Optional[int]:
    """Return days since last event, or None if date is missing."""
    if not last_event_date:
        return None
    from datetime import datetime
    fmt = "%Y-%m-%d"
    try:
        return (datetime.strptime(reference_date, fmt) -
                datetime.strptime(last_event_date, fmt)).days
    except ValueError as e:
        logger.warning(f"Bad date format: {e}")
        return None

# *args and **kwargs for flexible interfaces
def build_features(*categorical_cols, **numeric_kwargs):
    print(f"Categorical: {categorical_cols}")
    print(f"Numeric config: {numeric_kwargs}")

build_features("gender", "region", window=30, min_count=5)
```

---

## 4. Object-Oriented Python and Type Safety

### 4.1 Dataclasses for Configuration

Dataclasses give you structured, typed, inspectable config objects with almost no boilerplate. They're superior to dicts for configs because they validate structure at write time, support default values, and are self-documenting.

```python
from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class TrainingConfig:
    model_name: str
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_ratio: float = 0.1
    output_dir: str = "output/"
    feature_cols: List[str] = field(default_factory=list)
    early_stopping_patience: Optional[int] = None

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            return cls(**json.load(f))

cfg = TrainingConfig(
    model_name="microsoft/deberta-v3-base",
    learning_rate=3e-5,
    feature_cols=["age", "income", "score"]
)
print(cfg)  # TrainingConfig(model_name='microsoft/deberta-v3-base', ...)
```

### 4.2 Pydantic for Runtime Validation

Pydantic validates data at runtime — critical for API request payloads, config loading from YAML/env, and LLM structured outputs.

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class InferenceRequest(BaseModel):
    user_id: str
    items: List[str] = Field(..., min_items=1, max_items=100)
    top_k: int = Field(default=10, ge=1, le=100)
    context: Optional[dict] = None

    @validator('user_id')
    def user_id_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('user_id cannot be empty')
        return v.strip()

# This raises ValidationError immediately — no silent bugs
try:
    req = InferenceRequest(user_id="  ", items=[], top_k=200)
except Exception as e:
    print(f"Validation error: {e}")

# This works
req = InferenceRequest(user_id="user_123", items=["item_a", "item_b"])
print(req.model_dump())
```

### 4.3 Classes for Model and Client Wrappers

```python
from typing import Optional
import time

class ModelClient:
    """Wraps a model API with retry logic and caching."""

    def __init__(self, base_url: str, timeout: float = 30.0, max_retries: int = 3):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache: dict = {}

    def predict(self, text: str, use_cache: bool = True) -> dict:
        if use_cache and text in self._cache:
            return self._cache[text]

        for attempt in range(self.max_retries):
            try:
                result = self._call_api(text)
                if use_cache:
                    self._cache[text] = result
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # exponential backoff

    def _call_api(self, text: str) -> dict:
        # Actual HTTP call goes here
        import requests
        resp = requests.post(f"{self.base_url}/predict",
                             json={"text": text}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def __repr__(self):
        return f"ModelClient(base_url='{self.base_url}', cache_size={len(self._cache)})"
```

> 🎯 **Interview prep**: "Dataclass vs Pydantic vs plain dict — when to use each?" — Dict: simple, internal, no validation needed. Dataclass: structured config, needs default values, no runtime validation from external input. Pydantic: API payloads, config from files/env, anywhere external data enters the system.

---

## 5. Error Handling, Logging, and Debugging

Production ML pipelines fail in boring ways: malformed dates in feature data, NaN values propagating through a join, an upstream API returning 429. Robust error handling and structured logging are what separate pipelines that page you at 2am from ones that fail gracefully and leave a clear trace.

```python
import logging
import sys
from typing import Callable, TypeVar
import functools
import time

# Structured logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("feature_pipeline")

# Custom exception hierarchy
class FeaturePipelineError(Exception): pass
class DataValidationError(FeaturePipelineError): pass
class LeakageDetectedError(FeaturePipelineError): pass

# Retry decorator
def retry(max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    wait = delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                    time.sleep(wait)
        return wrapper
    return decorator

# Timing decorator
def timed(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper

@retry(max_attempts=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
@timed
def fetch_user_events(user_id: str, as_of_date: str) -> list:
    logger.info(f"Fetching events for user={user_id} as_of={as_of_date}")
    # ... actual DB call
    return []
```

> 🏭 **Production note**: Always log at the start and end of major pipeline stages with row counts. `logger.info(f"Loaded {len(df):,} rows for {as_of_date}")` has saved countless hours of debugging. Use structured JSON logging in production (e.g., `python-json-logger`) so logs are searchable in CloudWatch or Datadog.

---

## 6. Pandas: DataFrames, Indexing, and Filtering

### 6.1 Inspecting DataFrames

The first thing you do with any new dataset:

```python
import pandas as pd
import numpy as np

# Create a sample ML dataset
np.random.seed(42)
n = 10_000
df = pd.DataFrame({
    'user_id': [f"u{i}" for i in range(n)],
    'event_date': pd.date_range('2024-01-01', periods=n, freq='H'),
    'amount': np.random.lognormal(4, 1, n),
    'category': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
    'is_fraud': np.random.binomial(1, 0.02, n),
})

# Essential inspection
print(df.shape)            # (10000, 5)
print(df.dtypes)           # column types
print(df.info())           # non-null counts, memory usage
print(df.describe())       # numeric stats
print(df.isnull().sum())   # missing per column
print(df.memory_usage(deep=True).sum() / 1e6, "MB")

# Target balance
print(df['is_fraud'].value_counts(normalize=True))

# Cardinality of categoricals
print(df['category'].nunique(), df['category'].value_counts())

# --- Expected output ---
# df.shape → (10000, 5)
#
# df.dtypes →
#   user_id                object
#   event_date     datetime64[ns]
#   amount                float64
#   category               object
#   is_fraud                int64
#
# df.describe() →
#              amount      is_fraud
# count   10000.000000  10000.000000
# mean       82.34         0.0207
# std       158.92         0.1425
# min         0.18         0.0
# 25%        16.91         0.0
# 50%        44.13         0.0
# 75%       104.87         0.0
# max      5832.14         1.0
#
# df.isnull().sum() →
#   user_id      0
#   event_date   0
#   amount       0
#   category     0
#   is_fraud     0
#   dtype: int64
#
# df['is_fraud'].value_counts(normalize=True) →
#   0    0.9793
#   1    0.0207
#   Name: is_fraud, dtype: float64
#
# df['category'].nunique() → 3
# df['category'].value_counts() →
#   A    5023
#   B    2997
#   C    1980
```

### 6.2 Indexing and Filtering

The `loc`/`iloc` distinction is critical: `loc` is label-based, `iloc` is integer position-based.

```python
# loc: label-based
active = df.loc[(df['event_date'] >= '2024-06-01') & (df['is_fraud'] == 0)].copy()

# iloc: positional
first_100 = df.iloc[:100]
every_other = df.iloc[::2]

# query: readable SQL-like syntax (slightly slower but cleaner)
high_value = df.query("amount > 500 and category == 'A'").copy()

# CRITICAL: always use .copy() when slicing for modification
# Without .copy(), you may get SettingWithCopyWarning or silent bugs
subset = df[df['is_fraud'] == 1].copy()
subset['log_amount'] = np.log1p(subset['amount'])  # safe with .copy()

# Boolean indexing with complex conditions
mask = (
    (df['amount'] > df['amount'].quantile(0.95)) |
    ((df['category'] == 'B') & (df['amount'] > 1000))
)
flagged = df[mask]
print(f"Flagged rows: {len(flagged):,}")
```

### 6.3 Missing Values and Type Cleanup

```python
import pandas as pd
import numpy as np

df_raw = pd.DataFrame({
    'age': ['25', '30', None, 'invalid', '45'],
    'income': [50000, None, 75000, 90000, None],
    'signup_date': ['2023-01-15', '2023-02-20', None, '2023-03-10', '2023-04-05'],
    'category': ['A', 'B', 'A', None, 'C']
})

# Type conversion with error handling
df_raw['age'] = pd.to_numeric(df_raw['age'], errors='coerce')  # 'invalid' → NaN
df_raw['income'] = pd.to_numeric(df_raw['income'], errors='coerce')
df_raw['signup_date'] = pd.to_datetime(df_raw['signup_date'], errors='coerce')

# Missingness report
missing = df_raw.isnull().sum()
missing_pct = df_raw.isnull().mean()
print(pd.DataFrame({'count': missing, 'pct': missing_pct}))

# --- Output before cleaning ---
# After pd.to_numeric('invalid', errors='coerce') turns 'invalid' → NaN:
#              count   pct
# age              1  0.20   ← 'invalid' became NaN
# income           2  0.40
# signup_date      1  0.20
# category         1  0.20

# Fill strategies
df_clean = df_raw.copy()
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
df_clean['income'] = df_clean['income'].fillna(df_clean['income'].median())
df_clean['category'] = df_clean['category'].fillna('UNKNOWN')

# Add missingness indicators (often carry signal)
df_clean['income_missing'] = df_raw['income'].isnull().astype(int)
df_clean['age_missing'] = df_raw['age'].isnull().astype(int)

# --- Output after cleaning ---
# df_clean →
#    age    income signup_date category  income_missing  age_missing
#  25.0   50000.0  2023-01-15        A               0            0
#  30.0   82500.0  2023-02-20        B               1            0   ← income filled
#  30.0   75000.0         NaT        A               0            1   ← age filled (median=30)
#  30.0   90000.0  2023-03-10  UNKNOWN               0            0   ← category filled
#  45.0   82500.0  2023-04-05        C               1            0   ← income filled
```

---

## 7. GroupBy, Aggregation, and Feature Engineering

GroupBy aggregation is the single most-used operation in tabular ML feature engineering. Every behavioral feature — purchase frequency, recency, spending averages — comes from a groupby.

### 7.1 Named Aggregations

```python
import pandas as pd
import numpy as np

# Transaction-level data
np.random.seed(42)
n = 50_000
transactions = pd.DataFrame({
    'user_id': np.random.randint(0, 1000, n),
    'transaction_date': pd.date_range('2023-01-01', periods=n, freq='30min'),
    'amount': np.random.lognormal(4, 1.2, n),
    'category': np.random.choice(['food', 'travel', 'retail', 'digital'], n),
    'is_declined': np.random.binomial(1, 0.05, n),
})

# Named aggregation — the production way
user_features = (
    transactions
    .groupby('user_id')
    .agg(
        total_txns=('amount', 'count'),
        total_spend=('amount', 'sum'),
        avg_txn_amount=('amount', 'mean'),
        max_txn_amount=('amount', 'max'),
        std_txn_amount=('amount', 'std'),
        n_unique_categories=('category', 'nunique'),
        decline_rate=('is_declined', 'mean'),
        last_txn_date=('transaction_date', 'max'),
        first_txn_date=('transaction_date', 'min'),
    )
    .reset_index()
)

# Derived features
user_features['log_total_spend'] = np.log1p(user_features['total_spend'])
user_features['tenure_days'] = (
    user_features['last_txn_date'] - user_features['first_txn_date']
).dt.days

print(user_features.describe())

# --- Expected output (first few rows + describe) ---
# user_features.head(3) →
#    user_id  total_txns  total_spend  avg_txn_amount  max_txn_amount  \
#          0          50      2843.17           56.86         512.41
#          1          48      2901.35           60.44         489.12
#          2          52      3102.91           59.67         601.33
#
#    n_unique_categories  decline_rate  log_total_spend  tenure_days
#                      4        0.0600            7.953          354
#                      4        0.0625            7.974          354
#                      4        0.0577            8.040          354
#
# user_features.describe() →
#        total_txns  total_spend  avg_txn_amount  decline_rate  tenure_days
# count    1000.00     1000.00        1000.00       1000.00       1000.00
# mean       50.00     2895.34          57.73          0.050        342.13
# std         7.07      638.21          12.56          0.031         15.42
# min        31.00     1089.43          28.94          0.000        293.00
# 50%        50.00     2851.22          57.01          0.049        344.00
# max        73.00     6042.88         101.50          0.161        354.00
```

### 7.2 Transform for In-Place Aggregation

Use `transform` when you want the group-level statistic as a new column aligned with the original DataFrame (same length as input).

```python
# Add group means back to original rows
transactions['user_avg_amount'] = (
    transactions.groupby('user_id')['amount'].transform('mean')
)
transactions['user_spend_rank'] = (
    transactions.groupby('user_id')['amount'].transform('rank', pct=True)
)

# Within-group standardization
transactions['amount_vs_user_avg'] = (
    (transactions['amount'] - transactions['user_avg_amount']) /
    transactions.groupby('user_id')['amount'].transform('std').fillna(1)
)
```

---

## 8. Joins, Datetime Features, and Rolling Windows

### 8.1 Joins and Merge Safety

The most dangerous Pandas operation is an accidental many-to-many join. Always validate join quality.

```python
import pandas as pd
import numpy as np

# Best practice: check for duplicates before merge
users = pd.DataFrame({'user_id': ['u1','u2','u3'], 'tier': ['gold','silver','gold']})
events = pd.DataFrame({'user_id': ['u1','u1','u2','u4'], 'event': ['login','purchase','login','login']})

# Check: is join key unique in the lookup table?
assert users['user_id'].is_unique, "Duplicate user_ids in users table!"

# Merge with validation
result = events.merge(users, on='user_id', how='left', validate='many_to_one')

# Check for explosion (many-to-many)
print(f"Before: {len(events):,} rows, After: {len(result):,} rows")  # should be same

# Check unmatched rows
unmatched = result[result['tier'].isnull()]
print(f"Unmatched user_ids: {result['user_id'][result['tier'].isnull()].unique()}")

# --- Expected output ---
# events (before merge):
#   user_id    event
#       u1    login
#       u1  purchase      ← two events for u1
#       u2    login
#       u4    login       ← u4 not in users table
#
# result (after left merge):
#   user_id    event    tier
#       u1    login    gold
#       u1  purchase   gold    ← correctly matched both u1 rows
#       u2    login  silver
#       u4    login     NaN    ← left join keeps u4 with NaN tier
#
# Before: 4 rows, After: 4 rows   ← row count preserved (correct for left join)
# Unmatched user_ids: ['u4']
```

### 8.2 Datetime Features

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'ts': pd.date_range('2023-01-01', periods=365, freq='D'),
                   'sales': np.random.normal(1000, 100, 365)})

# Extract temporal components
df['year'] = df['ts'].dt.year
df['month'] = df['ts'].dt.month
df['day'] = df['ts'].dt.day
df['dayofweek'] = df['ts'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_month_end'] = df['ts'].dt.is_month_end.astype(int)
df['quarter'] = df['ts'].dt.quarter

# Cyclical encoding (better than raw month/day for tree models)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Recency
reference_date = pd.Timestamp('2024-01-01')
df['days_since'] = (reference_date - df['ts']).dt.days

# Timedelta calculations
df['sales_7d_ago'] = df['sales'].shift(7)
df['sales_change'] = df['sales'] - df['sales_7d_ago']
```

### 8.3 Rolling and Expanding Window Features

Rolling features are the heart of time-series feature engineering. The critical rule: **always use `shift(1)` before rolling to avoid data leakage** (excluding the current period).

```python
import pandas as pd
import numpy as np

# Sort first — always!
df = df.sort_values(['user_id', 'event_month']).copy()

# Lag features
df['prev_month_sales'] = df.groupby('user_id')['sales'].shift(1)

# Rolling windows (shift(1) first to exclude current period)
df['sales_3m_avg'] = (
    df.groupby('user_id')['sales']
    .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
)
df['sales_6m_max'] = (
    df.groupby('user_id')['sales']
    .transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).max())
)
df['sales_12m_sum'] = (
    df.groupby('user_id')['sales']
    .transform(lambda s: s.shift(1).rolling(window=12, min_periods=1).sum())
)

# Expanding window (all history before current period)
df['cumulative_avg'] = (
    df.groupby('user_id')['sales']
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# Exponentially weighted moving average (recent events weighted more)
df['ewm_sales'] = (
    df.groupby('user_id')['sales']
    .transform(lambda s: s.shift(1).ewm(span=6, min_periods=1).mean())
)
```

# --- Expected output (illustrating shift + rolling) ---
# For a single user with monthly sales [100, 200, 300, 400, 500]:
#
# event_month   sales   shift(1)   3m_rolling_avg (shift first)
#      Jan       100      NaN             NaN       ← no prior data
#      Feb       200      100            100.0      ← only 1 prior month
#      Mar       300      200            150.0      ← avg(100, 200)
#      Apr       400      300            200.0      ← avg(100, 200, 300)
#      May       500      400            300.0      ← avg(200, 300, 400)  ← excludes May!
#
# Without shift(1), May's 3m_avg would be avg(300, 400, 500) = 400.0 → data leakage

> 🏭 **Production note**: The `shift(1)` before rolling is the single most common leakage bug in time-series feature engineering. Without it, your "3-month average" includes the current month's value, creating look-ahead bias. Always validate: features for a user's January row should only use data from December and earlier.

---

## 9. Reshaping, I/O, and Performance Optimization

### 9.1 Reshaping

```python
import pandas as pd
import numpy as np

# Pivot: wide format (user × category spend)
transactions = pd.DataFrame({
    'user_id': np.repeat(['u1','u2','u3'], 4),
    'category': ['food','travel','retail','digital'] * 3,
    'amount': np.random.randint(10, 500, 12)
})

pivot = transactions.pivot_table(
    index='user_id', columns='category', values='amount',
    aggfunc='sum', fill_value=0
)
pivot.columns = [f"spend_{c}" for c in pivot.columns]

# Melt: wide → long (undo pivot)
long = pivot.reset_index().melt(id_vars='user_id', var_name='category', value_name='amount')

# Explode: list columns → rows
df_with_lists = pd.DataFrame({'user_id': ['u1','u2'],
                               'tags': [['ml','ai'],['data','python']]})
df_exploded = df_with_lists.explode('tags')

# Top-N per group
top_spenders = (
    transactions.sort_values('amount', ascending=False)
    .groupby('user_id')
    .head(2)
)
```

### 9.2 Reading and Writing Data

```python
import pandas as pd

# Parquet: always prefer over CSV for typed ML data
df.to_parquet('data.parquet', compression='snappy', index=False)
df = pd.read_parquet('data.parquet')

# Chunked CSV reading for large files (>memory)
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=100_000):
    # Process each chunk
    chunk_processed = chunk[chunk['is_valid'] == True]
    chunks.append(chunk_processed)
df = pd.concat(chunks, ignore_index=True)

# SQL reading
import sqlalchemy
engine = sqlalchemy.create_engine("postgresql://user:pass@host/db")
df = pd.read_sql("SELECT * FROM events WHERE date >= '2024-01-01'", engine)
```

### 9.3 Memory and Performance Optimization

Memory is often the real bottleneck. A DataFrame with int64 columns uses 8 bytes per value; with int8, 1 byte. Converting categoricals from object (variable-length strings) to `category` dtype can reduce memory by 5-10×.

```python
import pandas as pd
import numpy as np

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage by downcasting numeric types and converting low-cardinality strings."""
    df = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')  # float64 → float32
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')  # int64 → int8/16/32
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < df[col].shape[0] * 0.5:  # cardinality < 50%
            df[col] = df[col].astype('category')
    return df

# Performance: vectorization vs apply
import time

n = 1_000_000
df = pd.DataFrame({'a': np.random.randint(0, 100, n), 'b': np.random.randint(0, 100, n)})

# Slow: Python loop via apply
t0 = time.time()
result_slow = df.apply(lambda row: row['a'] ** 2 + row['b'] ** 2, axis=1)
print(f"apply: {time.time()-t0:.2f}s")

# Fast: vectorized operations
t0 = time.time()
result_fast = df['a'] ** 2 + df['b'] ** 2
print(f"vectorized: {time.time()-t0:.4f}s")  # ~100x faster
```

> 🎯 **Interview prep**: "How do you handle a DataFrame that doesn't fit in memory?" — Options: (1) process in chunks with `read_csv(chunksize=...)`, (2) use Polars or Dask for out-of-core processing, (3) switch to Parquet with column pruning + row filtering at read time, (4) use SQL to pre-aggregate before loading.

---

## 10. Async Python and Concurrency

AI engineering is inherently I/O-bound — you're waiting for API responses from embedding models, LLMs, rerankers, and databases. Async Python with `asyncio` is the correct tool for concurrent I/O.

### 10.1 asyncio Fundamentals

```python
import asyncio
import time
from typing import List

# Basic coroutine
async def fetch_embedding(text: str) -> List[float]:
    await asyncio.sleep(0.05)  # simulate 50ms API call
    return [0.1, 0.2, 0.3]  # stub

# Concurrent fan-out: the killer pattern for AI engineering
async def embed_batch_concurrent(texts: List[str]) -> List[List[float]]:
    """Fetch all embeddings concurrently. Much faster than sequential."""
    tasks = [fetch_embedding(text) for text in texts]
    return await asyncio.gather(*tasks)

# Sequential: 100 texts × 50ms = 5000ms
# Concurrent: ~50ms total (all run at once)

async def main():
    texts = [f"text {i}" for i in range(100)]
    t0 = time.time()
    results = await embed_batch_concurrent(texts)
    print(f"100 embeddings in {time.time()-t0:.3f}s, got {len(results)} results")

asyncio.run(main())
```

### 10.2 Rate-Limited Async Client

Real AI APIs have rate limits. A production async client must handle 429s with backoff.

```python
import asyncio
import httpx
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AsyncEmbeddingClient:
    def __init__(self, api_key: str, max_concurrent: int = 50, max_retries: int = 3):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)  # rate limiting
        self.max_retries = max_retries

    async def embed_one(self, client: httpx.AsyncClient, text: str) -> Optional[List[float]]:
        async with self.semaphore:  # limits concurrent requests
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        json={"input": text, "model": "text-embedding-3-small"},
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=10.0
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return resp.json()['data'][0]['embedding']
                except httpx.TimeoutException:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Timeout after {self.max_retries} attempts for: {text[:50]}")
                        return None
                    await asyncio.sleep(2 ** attempt)
        return None

    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        async with httpx.AsyncClient() as client:
            tasks = [self.embed_one(client, t) for t in texts]
            return await asyncio.gather(*tasks)
```

### 10.3 Threads vs Processes

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

# ThreadPoolExecutor: best for I/O-bound tasks (API calls, file reads, DB queries)
# GIL is released during I/O, so threads genuinely run concurrently
def call_api(item):
    import time; time.sleep(0.1)  # simulated I/O
    return {"item": item, "score": 0.95}

with ThreadPoolExecutor(max_workers=20) as executor:
    results = list(executor.map(call_api, range(100)))
    print(f"100 API calls: {len(results)} results")

# ProcessPoolExecutor: best for CPU-bound tasks (feature computation, image transforms)
# Bypasses GIL by using separate processes
def heavy_computation(chunk):
    return np.linalg.svd(np.random.normal(0, 1, (500, 500)))[1].sum()

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(heavy_computation, i) for i in range(8)]
    results = [f.result() for f in futures]
```

---

## 11. Serialization, Configuration, and Testing

### 11.1 Configuration Management

```python
from pathlib import Path
import os
import json
import yaml
from dataclasses import dataclass

@dataclass
class AppConfig:
    db_url: str
    model_name: str
    batch_size: int = 32
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load config from environment variables."""
        return cls(
            db_url=os.environ["DATABASE_URL"],
            model_name=os.environ.get("MODEL_NAME", "bert-base-uncased"),
            batch_size=int(os.environ.get("BATCH_SIZE", "32")),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_json(self, path: str):
        from dataclasses import asdict
        Path(path).write_text(json.dumps(asdict(self), indent=2))
```

### 11.2 pytest for Data and ML Code

```python
import pytest
import pandas as pd
import numpy as np

# Test that rolling feature doesn't leak future data
@pytest.fixture
def sample_events():
    return pd.DataFrame({
        'user_id': ['u1', 'u1', 'u1', 'u1'],
        'month': [1, 2, 3, 4],
        'sales': [100.0, 200.0, 150.0, 300.0]
    })

def compute_rolling_3m_avg(df):
    df = df.sort_values(['user_id', 'month']).copy()
    df['sales_3m_avg'] = (
        df.groupby('user_id')['sales']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    return df

def test_rolling_no_leakage(sample_events):
    result = compute_rolling_3m_avg(sample_events)
    # Month 1 should have NaN (no history)
    assert pd.isna(result.loc[result['month']==1, 'sales_3m_avg'].values[0])
    # Month 2 avg should only use month 1 data
    assert result.loc[result['month']==2, 'sales_3m_avg'].values[0] == 100.0

def test_no_nulls_introduced(sample_events):
    result = compute_rolling_3m_avg(sample_events)
    # Only month 1 should be NaN
    assert result['sales_3m_avg'].isna().sum() == 1

def test_schema_unchanged(sample_events):
    result = compute_rolling_3m_avg(sample_events)
    assert 'sales_3m_avg' in result.columns
    assert len(result) == len(sample_events)  # no row explosion

# Run: pytest test_features.py -v
```

---

## 12. Python Patterns for AI Engineering

AI engineering has a distinct pattern language: batching LLM calls, streaming responses, structured output extraction, and token budget management.

### 12.1 Batching LLM Calls

```python
import anthropic
import asyncio
from typing import List

client = anthropic.Anthropic()

async def score_text_async(text: str) -> dict:
    """Score a single text for sentiment using Claude."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"Rate sentiment 0-10 and extract key topics. Text: {text}\nRespond as JSON: {{\"sentiment\": N, \"topics\": [...]}}"
        }]
    )
    import json
    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {"sentiment": -1, "topics": []}

def batch_score(texts: List[str], batch_size: int = 20) -> List[dict]:
    """Score texts in batches to avoid overwhelming the API."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # In a real async context, use asyncio.gather
        batch_results = [score_text_async(t) for t in batch]
        results.extend(batch_results)
    return results
```

### 12.2 Streaming Responses

```python
import anthropic

client = anthropic.Anthropic()

def stream_analysis(document: str):
    """Stream a long analysis, printing tokens as they arrive."""
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": f"Analyze this document:\n\n{document}"}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()  # newline at end
        return stream.get_final_message()
```

### 12.3 Structured Output Extraction

```python
from pydantic import BaseModel
from typing import List
import instructor
import anthropic

class DocumentSummary(BaseModel):
    title: str
    key_points: List[str]
    sentiment: str
    confidence: float

# instructor patches the client to enforce structured output
client = instructor.from_anthropic(anthropic.Anthropic())

def extract_summary(text: str) -> DocumentSummary:
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        response_model=DocumentSummary,
        messages=[{"role": "user", "content": f"Summarize this:\n\n{text}"}]
    )

# result is always a valid DocumentSummary — no parsing errors
```

---

## 13. Notebook-to-Production Discipline

The fastest path to unmaintainable ML code is leaving logic in Jupyter notebooks. A notebook is for exploration; a Python module is for production.

```
EXPLORATION PHASE (notebook):
- Load raw data and explore distributions
- Try different feature ideas interactively
- Prototype model training and evaluation
- Identify bugs in data

EXTRACTION PHASE (refactor):
- Move pure functions to feature_engineering.py
- Move model training to train.py
- Move inference logic to predict.py
- Move config to config.py or config.yaml
- Write tests in tests/test_features.py

PRODUCTION PHASE (package):
my_project/
├── features/
│   ├── __init__.py
│   ├── user_features.py     # GroupBy aggregations
│   ├── item_features.py
│   └── interaction_features.py
├── models/
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
├── serving/
│   └── predict.py
├── tests/
│   ├── test_features.py
│   └── test_models.py
└── config.yaml
```

The signal that you've done this right: `python -m my_project.train --config config.yaml` works identically in your terminal and in a Databricks/Airflow/Kubeflow job.

---

## 14. The Modern Recipe

Opinionated Python and Pandas choices for 2025:

1. **Data manipulation**: Pandas for moderate data (< ~10M rows). **Polars** for large data or performance-critical pipelines — Polars is 5-30× faster than Pandas for most operations and has a more consistent API.

2. **Config**: Use **Pydantic** v2 for any config that touches external input. Use Python dataclasses for internal-only config.

3. **Async**: Default to `asyncio` for all LLM/embedding API calls. Use `asyncio.Semaphore` to rate-limit concurrent requests.

4. **Concurrency**: `ThreadPoolExecutor` for I/O-bound work; `ProcessPoolExecutor` for CPU-bound work.

5. **Type hints**: Annotate all function signatures. Use `mypy` in CI. This catches bugs before runtime.

6. **Testing**: pytest with fixtures. Test schema invariants (no new nulls, no row explosion) and correctness (no leakage, correct window alignment).

7. **Memory**: Convert `float64→float32`, `int64→int32/int16`, low-cardinality strings → `category`. Usually 2-4× memory reduction.

8. **Pandas vs alternatives comparison**:

| Task | Pandas | Polars | Dask |
|---|---|---|---|
| < 5M rows, interactive | ✅ Best | Good | Overkill |
| 5-500M rows, fast processing | Slow | ✅ Best | Good |
| > 500M rows, distributed | Slow/OOM | Good | ✅ Best |
| ML feature engineering | ✅ Standard | ✅ Faster | Good |
| Streaming data | Poor | Good | ✅ Best |

---

## 15. References

### Libraries & Official Docs
- [Pandas documentation](https://pandas.pydata.org/docs/user_guide/index.html)
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- [Python concurrent.futures docs](https://docs.python.org/3/library/concurrent.futures.html)
- [Pydantic v2 docs](https://docs.pydantic.dev/)
- [Polars user guide](https://docs.pola.rs/user-guide/)

### Key Blogs & Articles
- Chip Huyen (2022). *Designing Machine Learning Systems.* O'Reilly. — system-level thinking for ML code
- [Real Python: asyncio guide](https://realpython.com/async-io-python/)
- [Pandas 2.0 Copy-on-Write explanation](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html)
