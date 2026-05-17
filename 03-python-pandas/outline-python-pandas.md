# 03 — Python & Pandas

Production-first study outline for the Python that ML engineers and AI engineers use every day.

---

## Format Used In This Outline
- `Concept`: what to learn.
- `Why it matters`: where it shows up in ML or AI engineering.
- `Typical code`: the kind of code you should be able to write from memory.

## 01 — Core Python You Actually Use
- `Concept`: lists, dicts, sets, tuples, comprehensions, generators, unpacking, sorting with `key=`, `enumerate`, `zip`.
- `Why it matters`: almost every ETL, feature build, batching utility, and config transform uses these.
- `Typical code`: filter invalid rows, build lookup maps, flatten nested outputs, group small objects before turning them into DataFrames.

## 02 — Functions, Modules, and Project Structure
- `Concept`: pure functions, argument defaults, `*args` and `**kwargs`, imports, reusable modules.
- `Why it matters`: clean feature code and model code are easier to test and reuse in notebooks and jobs.
- `Typical code`: one file for feature builders, one file for model training, one for inference helpers.

## 03 — Object-Oriented Python For AI Systems
- `Concept`: classes, dataclasses, inheritance, composition, `__repr__`, `@property`, `@classmethod`, `@staticmethod`.
- `Why it matters`: useful for config objects, dataset adapters, model wrappers, retrievers, evaluators, API clients.
- `Typical code`: `TrainingConfig`, `FeatureStoreClient`, `ModelRunner`, `PromptBuilder`.

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
```

## 04 — Type Hints and Validation
- `Concept`: type hints, `TypedDict`, dataclasses, Pydantic models.
- `Why it matters`: AI systems pass around nested configs, payloads, prompts, tool calls, and model outputs that break easily without validation.
- `Typical code`: validate inference request payloads and config files before training starts.

## 05 — Error Handling, Logging, and Debugging
- `Concept`: `try/except`, custom exceptions, `logging`, `pdb`, stack traces, retries.
- `Why it matters`: batch pipelines, model-serving APIs, and data ingestion jobs fail in boring ways, not glamorous ones.
- `Typical code`: log batch ids, data ranges, model versions, and row counts at each stage.

## 06 — Pandas DataFrame Basics
- `Concept`: create DataFrames, inspect `shape`, `info`, `describe`, `dtypes`, `memory_usage`.
- `Why it matters`: this is the fastest path from raw tabular data to an ML-ready dataset.
- `Typical code`: load Parquet, inspect schema drift, check null counts, profile target balance.

## 07 — Indexing and Filtering
- `Concept`: `loc`, `iloc`, boolean masks, `query`, column selection, assignment.
- `Why it matters`: nearly every feature engineering or error-analysis notebook starts here.
- `Typical code`: filter training data to an observation window and assign derived labels.

```python
active = df.loc[(df["event_date"] >= "2025-01-01") & (df["status"] == "active")].copy()
```

## 08 — Missing Values and Type Cleanup
- `Concept`: `isna`, `fillna`, `dropna`, `astype`, `to_numeric`, `to_datetime`.
- `Why it matters`: models fail quietly when string dates, mixed numeric types, or null-heavy features slip through.
- `Typical code`: cast columns consistently before feature pipelines and train/test splits.

## 09 — GroupBy, Aggregation, and Named Agg
- `Concept`: `groupby`, `agg`, `transform`, `nunique`, `value_counts`, `size`.
- `Why it matters`: most classical ML feature engineering is grouped aggregation.
- `Typical code`: user-level, account-level, merchant-level, and item-level summaries.

```python
features = (
    events.groupby("user_id")
    .agg(
        total_orders=("order_id", "nunique"),
        avg_amount=("amount", "mean"),
        last_event=("event_time", "max"),
    )
    .reset_index()
)
```

## 10 — Joins and Table Stitching
- `Concept`: `merge`, `join`, `concat`, join keys, duplicate explosion checks.
- `Why it matters`: feature tables usually come from many systems, and silent many-to-many joins are a common failure mode.
- `Typical code`: join base population, labels, static attributes, and historical aggregates.

## 11 — Datetime Features
- `Concept`: `pd.to_datetime`, `.dt.year`, `.dt.month`, `.dt.dayofweek`, `.dt.is_month_end`, timedeltas.
- `Why it matters`: churn, demand, fraud, and campaign response models all depend heavily on temporal behavior.
- `Typical code`: recency, tenure, seasonality, weekday/weekend, month-end payroll effects.

## 12 — Rolling, Expanding, and Lag Features
- `Concept`: `shift`, `rolling`, `expanding`, `ewm`.
- `Why it matters`: this is the heart of time-based feature engineering.
- `Typical code`: 3-month average, 6-month max, 30-day count, days since last activity.

```python
df = df.sort_values(["user_id", "event_month"])
df["attended_3m_max"] = (
    df.groupby("user_id")["seminars_attended"]
      .transform(lambda s: s.shift(1).rolling(3, min_periods=1).max())
)
```

## 13 — Reshaping Data
- `Concept`: `pivot_table`, `melt`, `stack`, `unstack`, `explode`, `crosstab`.
- `Why it matters`: useful for feature matrices, reporting tables, lift reports, and recommender interaction matrices.
- `Typical code`: turn transaction logs into user-by-category summaries or model-monitoring dashboards.

## 14 — The Pandas Code Patterns You Use Most In Practice
- `Concept`: sort, deduplicate, rank within groups, top-N per group, map from lookup tables.
- `Why it matters`: these patterns appear constantly in error analysis and candidate generation.
- `Typical code`: latest transaction per user, top 5 recommended items per session, decile bucketing, threshold tables.

## 15 — Reading and Writing Data
- `Concept`: `read_csv`, `read_parquet`, `read_sql`, chunked reads, compression.
- `Why it matters`: I/O is often the real bottleneck.
- `Typical code`: prefer Parquet for typed analytics data; use chunking for large CSVs and backfills.

## 16 — Performance and Memory Optimization
- `Concept`: categorical dtypes, downcasting, vectorization, avoiding row-wise `apply`, chunking.
- `Why it matters`: many DS pipelines are slow because they are written like spreadsheet logic.
- `Typical code`: replace Python loops with vectorized ops, avoid exploding memory on joins.

## 17 — Async Python
- `Concept`: `async`, `await`, event loop, coroutines, `asyncio.gather`, cancellation, timeouts.
- `Why it matters`: LLM apps, agent frameworks, streaming APIs, and many inference fan-out workflows are I/O-bound.
- `Typical code`: call multiple embedding, reranking, or tool endpoints concurrently.

```python
import asyncio

async def fetch_one(client, payload):
    return await client.post("/score", json=payload)

async def fetch_all(client, payloads):
    return await asyncio.gather(*(fetch_one(client, p) for p in payloads))
```

## 18 — Multithreading vs Multiprocessing
- `Concept`: GIL, `ThreadPoolExecutor`, `ProcessPoolExecutor`, CPU-bound vs I/O-bound work.
- `Why it matters`: choose the wrong concurrency model and you either waste cores or overcomplicate simple I/O.
- `Typical code`: threads for API calls and file downloads; processes for CPU-heavy preprocessing.

## 19 — Context Managers, Decorators, and Utilities
- `Concept`: `with`, custom context managers, decorators for retry, timing, caching.
- `Why it matters`: useful for DB sessions, temporary resources, instrumentation, and robust model-serving code.
- `Typical code`: timing decorators around embedding calls or feature joins.

## 20 — Serialization and Configuration
- `Concept`: JSON, YAML, environment variables, `.env`, `pathlib`, reproducible config loading.
- `Why it matters`: training and serving need the same knobs expressed cleanly.
- `Typical code`: read training config once, store with the model artifact, reuse at inference.

## 21 — Testing for Data and ML Code
- `Concept`: `pytest`, fixture-based tests, schema tests, deterministic small-sample assertions.
- `Why it matters`: feature bugs are expensive and subtle.
- `Typical code`: test that a 3-month rolling feature excludes the current month and handles missing months correctly.

## 22 — Python Patterns For AI Engineering
- `Concept`: clients for model APIs, batching, retries, caching, streaming, rate limiting.
- `Why it matters`: AI engineers spend less time proving linear algebra and more time building reliable systems around models.
- `Typical code`: batch 100 prompts, stream tokens, parse JSON output, back off on 429s.

## 23 — Notebook-to-Production Discipline
- `Concept`: notebook exploration, then extract pure functions into modules.
- `Why it matters`: the fastest path to messy ML code is leaving core logic trapped in notebooks.
- `Typical code`: notebook for exploration; package for repeatable training and inference.

## 24 — Recommended Hands-On Builds
- `Feature engineering notebook`: build user-level rolling aggregates from an event table.
- `Async scoring client`: hit an internal inference API concurrently.
- `Training config package`: dataclass or Pydantic config with CLI entrypoint.
- `Lift report utility`: create deciles from predicted probabilities with Pandas.

## References
- Python asyncio docs: https://docs.python.org/3/library/asyncio.html
- concurrent.futures docs: https://docs.python.org/3/library/concurrent.futures.html
- Pandas user guide: https://pandas.pydata.org/docs/user_guide/index.html
