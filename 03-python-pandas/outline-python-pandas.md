# 03 — Python & Pandas

Exhaustive learning path for Python data manipulation: NumPy, Pandas, and data wrangling workflows for data science and ML.

---

## 01 — Python Data Structures & Comprehensions
Lists, dicts, sets, tuples; list/dict/set comprehensions; generators; zip, enumerate, map, filter; unpacking.
- https://docs.python.org/3/tutorial/datastructures.html

## 02 — NumPy Arrays
ndarray creation, dtypes, shape, reshape; broadcasting rules; vectorized operations; indexing & slicing; np.where.
- https://numpy.org/doc/stable/user/quickstart.html
- https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

## 03 — NumPy Linear Algebra & Math
dot, matmul, einsum; linalg.inv, linalg.eig, linalg.svd; random number generation (rng seeds, distributions).
- https://numpy.org/doc/stable/reference/routines.linalg.html

## 04 — Pandas Series & DataFrame
Creating from dicts/lists/CSV/Parquet; dtypes; index; head/tail/info/describe; memory_usage.
- https://pandas.pydata.org/docs/user_guide/10min.html

## 05 — Indexing & Selection
loc vs iloc; boolean indexing; .at/.iat; MultiIndex; query(); chained indexing pitfalls and SettingWithCopyWarning.
- https://pandas.pydata.org/docs/user_guide/indexing.html

## 06 — Data Cleaning
Handling missing values (isna, fillna, dropna, interpolate); duplicates; type coercion; string normalization; category dtype.
- https://pandas.pydata.org/docs/user_guide/missing_data.html

## 07 — String & Datetime Operations
str accessor (split, contains, extract, replace); pd.to_datetime; dt accessor (year, month, dayofweek, floor/ceil/round); timedelta arithmetic.
- https://pandas.pydata.org/docs/user_guide/text.html
- https://pandas.pydata.org/docs/user_guide/timeseries.html

## 08 — GroupBy & Aggregation
groupby split-apply-combine; agg with multiple functions; transform vs apply; named aggregation; groupby + shift/cumsum.
- https://pandas.pydata.org/docs/user_guide/groupby.html

## 09 — Merge, Join & Concat
merge (inner/left/right/outer, on, suffixes); join on index; concat (axis=0/1, keys); merge_asof for time-series lookups.
- https://pandas.pydata.org/docs/user_guide/merging.html

## 10 — Reshaping: pivot, melt, stack, unstack
pivot_table; melt (wide → long); stack/unstack; crosstab; explode; get_dummies for one-hot encoding.
- https://pandas.pydata.org/docs/user_guide/reshaping.html

## 11 — Window Functions
rolling, expanding, ewm; custom window aggregations; shift/diff for lag/lead features; time-aware rolling.
- https://pandas.pydata.org/docs/user_guide/window.html

## 12 — Apply, Map & Vectorization
apply vs map vs applymap; when to avoid apply (prefer vectorized ops); np.vectorize; numba @jit for bottleneck loops.
- https://pandas.pydata.org/docs/reference/frame.html#function-application-groupby-window

## 13 — IO: CSV, Parquet, JSON, SQL, Excel
read_csv (dtypes, parse_dates, chunksize); to_parquet/read_parquet (pyarrow, fastparquet); to_sql/read_sql; Excel (openpyxl).
- https://pandas.pydata.org/docs/user_guide/io.html

## 14 — Performance & Memory Optimization
Categorical dtype; downcast numerics; chunked processing; Dask for out-of-core; profiling with memory_profiler.
- https://pandas.pydata.org/docs/user_guide/scale.html
- https://docs.dask.org/en/stable/dataframe.html

## 15 — Plotting with Pandas & Matplotlib
df.plot API (line, bar, hist, scatter, box, kde); subplots; Matplotlib fig/ax; seaborn for statistical plots.
- https://pandas.pydata.org/docs/user_guide/visualization.html
- https://seaborn.pydata.org/tutorial.html
