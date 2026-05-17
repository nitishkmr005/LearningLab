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

## 16 — Polars: Fast DataFrame Library
Lazy vs eager API; scan_csv/scan_parquet; expressions and chained ops; group_by + agg; vs pandas performance benchmark; when to switch from pandas.
- https://docs.pola.rs/user-guide/getting-started/
- https://pola.rs/posts/benchmarks/

## 17 — Python OOP, Decorators & Context Managers
Classes for ML pipelines; @property, @classmethod, @staticmethod; @dataclass; __slots__ for memory; context managers (with/__enter__/__exit__); decorators for retry/logging.
- https://docs.python.org/3/tutorial/classes.html
- https://realpython.com/python-classes/

## 18 — Type Hints, Pydantic & Dataclasses
Function annotations; mypy basics; Pydantic BaseModel for config/API schemas; Field validators; model_validator; dataclasses vs NamedTuple vs TypedDict; used throughout LLM tooling (LangChain, FastAPI, Instructor).
- https://docs.pydantic.dev/latest/
- https://mypy.readthedocs.io/en/stable/

## 19 — Python Debugging & Profiling
pdb/ipdb breakpoints; cProfile + snakeviz; line_profiler (@profile); memory_profiler; py-spy for production; diagnosing slow notebooks and data pipelines.
- https://docs.python.org/3/library/profile.html
- https://github.com/benfred/py-spy

## 20 — Virtual Environments & Packaging
venv vs conda vs uv (fast resolver); pyproject.toml; pip-tools for lock files; packaging a DS utility library; reproducible environments for MLOps.
- https://docs.astral.sh/uv/
- https://packaging.python.org/en/latest/guides/

## 21 — Async Python (asyncio)
Event loop; async/await syntax; coroutines vs threads; asyncio.gather and asyncio.create_task for concurrency; aiohttp for async HTTP; async generators; StreamingResponse; essential for LLM streaming, voice agents, and real-time API servers.
- https://docs.python.org/3/library/asyncio.html
- https://realpython.com/async-io-python/

## 22 — Multithreading & Multiprocessing
threading.Thread for I/O-bound tasks; GIL and why it limits CPU parallelism; concurrent.futures (ThreadPoolExecutor, ProcessPoolExecutor); multiprocessing.Pool for CPU-bound; shared memory; Queue for inter-process communication; use in audio pipelines, data loaders, and voice agent backends.
- https://docs.python.org/3/library/concurrent.futures.html
- https://docs.python.org/3/library/multiprocessing.html

## 23 — FastAPI for ML & AI Serving
Path/query/body params; Pydantic request/response models; async endpoints; background tasks; StreamingResponse for LLM token streaming; middleware; deploy with uvicorn + gunicorn; used for LLM, ML model, and voice agent APIs.
- https://fastapi.tiangolo.com/tutorial/
- https://fastapi.tiangolo.com/advanced/custom-response/

## 24 — Streamlit for ML & LLM Apps
st.dataframe, st.plotly_chart, st.selectbox, st.chat_message; session state for multi-turn UIs; caching with @st.cache_data and @st.cache_resource; deploy with Docker or Streamlit Cloud; production uses: LLM output validation dashboards, model bias review, bandit/experiment monitoring, RAG QA apps; most popular rapid prototyping tool for ML teams; alternative: Gradio (better for demos), Panel (more complex layouts).
- https://docs.streamlit.io/
- https://docs.streamlit.io/develop/concepts/architecture/caching
