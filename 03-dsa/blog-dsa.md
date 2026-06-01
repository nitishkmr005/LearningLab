# Data Structures & Algorithms for Data Science: A Practical Python Guide

*The complete reference for DS/ML/AI engineers — covering every major data structure and algorithm through the lens of Python and real data science workloads, from feature engineering to LLM inference.*

---

## Table of Contents

1. [Why DSA Matters for Data Scientists](#1-why-dsa-matters-for-data-scientists)
2. [Complexity Cheat-Sheet](#2-complexity-cheat-sheet)
3. [Naive vs Optimised: The Cost of Wrong Choices](#3-naive-vs-optimised-the-cost-of-wrong-choices)
4. [Arrays & NumPy Arrays](#4-arrays--numpy-arrays)
5. [Hash Maps, Sets & Feature Encoding](#5-hash-maps-sets--feature-encoding)
6. [Stacks & Queues](#6-stacks--queues)
7. [Heaps & Priority Queues](#7-heaps--priority-queues)
8. [Sorting & Binary Search](#8-sorting--binary-search)
9. [Two Pointers & Sliding Window](#9-two-pointers--sliding-window)
10. [Linked Lists & LRU Cache](#10-linked-lists--lru-cache)
11. [Trees & Binary Search Trees](#11-trees--binary-search-trees)
12. [Graphs, BFS & DFS](#12-graphs-bfs--dfs)
13. [Union-Find (Disjoint Set Union)](#13-union-find-disjoint-set-union)
14. [Dynamic Programming](#14-dynamic-programming)
15. [Tries (Prefix Trees)](#15-tries-prefix-trees)
16. [String Algorithms & Edit Distance](#16-string-algorithms--edit-distance)
17. [Matrix Operations & Linear Algebra](#17-matrix-operations--linear-algebra)
18. [Reservoir Sampling](#18-reservoir-sampling)
19. [Probabilistic Data Structures](#19-probabilistic-data-structures)
20. [Approximate Nearest Neighbours](#20-approximate-nearest-neighbours)
21. [Bit Manipulation](#21-bit-manipulation)
22. [Interval Trees & Range Queries](#22-interval-trees--range-queries)
23. [Pick the Right Structure: Decision Guide](#23-pick-the-right-structure-decision-guide)
24. [Common DS/ML Interview Questions & Answers](#24-common-dsml-interview-questions--answers)
25. [Production Failure Modes](#25-production-failure-modes)
26. [References](#26-references)

---

## 1. Why DSA Matters for Data Scientists

You are three hours into a data pipeline that produces rolling 7-day feature windows across 50 million user events. The naive approach — a nested loop over every event pair — runs overnight. A colleague rewrites it with a deque-based sliding window and it finishes in four minutes. You just witnessed what a data structure choice does to a production system.

Data scientists are not expected to implement red-black trees from scratch, but they are expected to choose the right abstraction when it matters: a `Counter` instead of a nested `for` loop to count label frequencies; a `heapq` to pull the top-K recommendations from a million candidates without sorting everything; a Bloom filter to deduplicate a billion training URLs in 1 GB of RAM. The distance between a script that times out and one that ships is often one data structure change.

This blog covers every major data structure and algorithm that appears in DS, ML, and AI engineering interviews and production systems, with three things for each: the Python equivalent you reach for today, a working code snippet, and an explanation of where this concept shows up in real workloads — from scikit-learn internals to LLM inference.

> 📚 **Go deeper**: [Tech Interview Handbook — DS & Algorithm Study Cheatsheets](https://www.techinterviewhandbook.org/algorithms/study-cheatsheet/) — great pattern-based prep guide

---

## 2. Complexity Cheat-Sheet

Understanding Big-O is not about memorising formulas — it is about predicting which part of your pipeline will explode when the dataset grows from 1 million to 1 billion rows. Before picking a data structure, ask: "What is the bottleneck operation, and how often does it run in the hot path?"

### 2.1 Python Built-in Complexity Reference

([CPython Wiki, TimeComplexity](https://wiki.python.org/moin/TimeComplexity))

| Structure | Access | Search | Insert | Delete | Notes |
|---|---|---|---|---|---|
| `list` | O(1) | O(n) | O(n) amortised¹ | O(n) | ¹append is O(1) amortised; insert at position is O(n) |
| `dict` | O(1) avg | O(1) avg | O(1) avg | O(1) avg | O(n) worst (hash collision) |
| `set` | — | O(1) avg | O(1) avg | O(1) avg | No random access |
| `deque` | O(n) mid | O(n) | O(1) ends | O(1) ends | O(n) middle access — use for FIFO/LIFO only |
| `heapq` (min-heap) | O(1) min | O(n) | O(log n) | O(log n) | Via `heappush`/`heappop` |
| `sortedcontainers.SortedList` | O(log n) | O(log n) | O(log n) | O(log n) | Pure Python balanced BST alternative |

### 2.2 Algorithm Complexity Reference

| Algorithm | Time | Space | DS/ML where it appears |
|---|---|---|---|
| Binary search | O(log n) | O(1) | Threshold tuning, sorted prediction lookup |
| Timsort (`sorted`) | O(n log n) avg | O(n) | Ranking by score, sorting predictions |
| BFS / DFS | O(V + E) | O(V) | Pipeline DAG traversal, GNN neighbourhood |
| Dijkstra | O((V + E) log V) | O(V) | Knowledge graph shortest path |
| Dynamic programming | O(n²) typical | O(n) | Edit distance, Viterbi, CTC decoding |
| Union-Find (path compressed) | O(α(n)) ≈ O(1) | O(n) | Connected components, entity resolution |
| Heap top-K | O(n log k) | O(k) | Top-K recommendations, beam search |

> 🎯 **Interview prep**: The most common Big-O question in DS interviews is "what's the complexity of your feature engineering step?" The answer interviewers want includes both time **and space**, and a note on whether the operation is vectorisable (NumPy removes the Python loop overhead, but the algorithmic complexity stays the same).

**Resources**
- [CPython Time Complexity wiki](https://wiki.python.org/moin/TimeComplexity) — authoritative reference for Python built-ins
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/) — quick visual reference

---

## 3. Naive vs Optimised: The Cost of Wrong Choices

Most data science code is written to be correct first, then left as-is once it works on a sample. The problem surfaces when the same code runs on 10 million rows instead of 10,000: a script that finished in 2 seconds now takes 55 hours. The root cause is almost never a missing optimisation flag — it is a data structure mismatch. A `list` used as a lookup table, a nested loop used to count frequencies, a full sort used to find the top-10 items. Each of these swaps one line of code and buys 10–1000× speedup with zero loss in correctness.

This section catalogs the most common naive-to-optimised rewrites in real DS/ML code. Every row is a pattern you will recognise from your own pipelines.

### 3.1 Master Comparison Table

| DS/ML Task | Naive Python | Naive Big-O | Optimised Python | Optimised Big-O | Typical speedup¹ |
|---|---|---|---|---|---|
| Count label/token frequencies | `{x: data.count(x) for x in data}` | O(n²) | `Counter(data)` | O(n) | **100–500×** |
| Membership test ("have I seen X?") | `if x in my_list` | O(n) | `if x in my_set` | O(1) avg | **50–200×** at n=1M |
| Remove duplicates from a list | nested `for` loop with comparison | O(n²) | `list(dict.fromkeys(data))` | O(n) | **100×+** |
| Queue: process items FIFO | `queue.pop(0)` on a list | O(n) per pop | `deque.popleft()` | O(1) | **50–500×** at n=100K |
| Top-K from scored items | `sorted(items, reverse=True)[:k]` | O(n log n) | `heapq.nlargest(k, items)` | O(n log k) | **5–50×** at k=10, n=1M |
| Rolling window statistic | nested `for` loop, recompute each window | O(n × W) | `pd.rolling(W).mean()` or deque | O(n) | **100–1000×** |
| Lookup by string key | linear scan over list of tuples | O(n) | `dict[key]` | O(1) avg | **50–300×** |
| Find nearest embedding | brute-force `np.dot` over all vectors | O(n × d) | FAISS / hnswlib | O(log n) approx | **100–10000×** at n=1M |
| Deduplicate 1M documents | pairwise Jaccard comparison | O(n²) | MinHash LSH | O(n) | **1000×+** |
| Count distinct items in stream | `len(set(stream))` — holds all items | O(n) space | HyperLogLog | O(1) space | **1000× less RAM** |
| Interval overlap join | nested loop over all pairs | O(n × m) | `IntervalTree.query()` | O(log n + k) | **50–500×** |
| Sorted insert into running list | `list.append()` + `list.sort()` | O(n log n) per insert | `bisect.insort()` | O(log n) search + O(n) insert | **10–100×** |

*¹ Speedup measured on n = 1 million items on a single CPU core. Your numbers will vary by hardware, data shape, and k.*

---

### 3.2 Worked Benchmarks

The table above gives intuition; the benchmarks below give numbers. Every snippet is self-contained and runnable.

#### Benchmark 1 — Frequency Count: nested dict vs Counter

```python
import time
from collections import Counter

data = (["cat"] * 400_000 + ["dog"] * 300_000 +
        ["bird"] * 200_000 + ["fish"] * 100_000)

# ❌ Naive: list.count() inside a loop — O(n²)
t0 = time.perf_counter()
naive = {x: data.count(x) for x in set(data)}   # .count() scans the whole list each time
print(f"Naive:   {time.perf_counter() - t0:.3f}s  → {naive}")

# ✅ Optimised: Counter — single O(n) pass
t0 = time.perf_counter()
fast = Counter(data)
print(f"Counter: {time.perf_counter() - t0:.3f}s  → {fast}")
# Naive: ~0.12s   Counter: ~0.025s  → ~5× here, but scales to 500× at n=10M
```

#### Benchmark 2 — Membership Test: list vs set

```python
import time, random

pool = [str(i) for i in range(1_000_000)]
queries = random.choices(pool, k=10_000)

# ❌ Naive: membership check in a list — O(n) per check
t0 = time.perf_counter()
found = sum(1 for q in queries if q in pool)    # scanning 1M items per query
print(f"list 'in': {time.perf_counter() - t0:.3f}s  found={found}")

# ✅ Optimised: set lookup — O(1) per check
pool_set = set(pool)
t0 = time.perf_counter()
found = sum(1 for q in queries if q in pool_set)
print(f"set  'in': {time.perf_counter() - t0:.3f}s  found={found}")
# Typical: list ~2.5s  set ~0.001s → 2500× faster
```

> 🏭 **Production note**: This exact pattern appears in vocabulary OOV detection, known-entity lookup, and feature allowlisting. At 1M vocabulary size, the list version makes a model-serving endpoint 2000× slower than necessary for this one check.

#### Benchmark 3 — FIFO Queue: list.pop(0) vs deque.popleft()

```python
import time
from collections import deque

N = 100_000
items = list(range(N))

# ❌ Naive: pop from front of list — shifts every remaining element, O(n) per pop
lst = items.copy()
t0 = time.perf_counter()
while lst:
    _ = lst.pop(0)    # O(n) — each call shifts ~N/2 elements on average
print(f"list.pop(0):      {time.perf_counter() - t0:.3f}s")   # ~1.5s

# ✅ Optimised: deque.popleft() — O(1), doubly-linked under the hood
dq = deque(items)
t0 = time.perf_counter()
while dq:
    _ = dq.popleft()  # O(1)
print(f"deque.popleft():  {time.perf_counter() - t0:.3f}s")   # ~0.006s
# ~250× faster at N=100K. At N=1M streaming events, the list version never finishes.
```

#### Benchmark 4 — Top-K: full sort vs heap

```python
import time, random, heapq

scores = [(random.random(), f"item_{i}") for i in range(1_000_000)]
K = 10

# ❌ Naive: sort everything, slice top-K — O(n log n)
t0 = time.perf_counter()
naive_top = sorted(scores, reverse=True)[:K]
print(f"sorted()[:K]:        {time.perf_counter() - t0:.3f}s")  # ~0.45s

# ✅ Optimised: nlargest — O(n log k), heap of size K
t0 = time.perf_counter()
fast_top = heapq.nlargest(K, scores)
print(f"heapq.nlargest(K):   {time.perf_counter() - t0:.3f}s")  # ~0.09s  (~5× faster)

# ✅✅ Even faster for numeric-only scores: np.argpartition — O(n)
import numpy as np
arr = np.array([s for s, _ in scores])
t0 = time.perf_counter()
top_idx = np.argpartition(arr, -K)[-K:]     # O(n) partial sort
print(f"np.argpartition:     {time.perf_counter() - t0:.3f}s")  # ~0.008s  (~55× faster)
```

#### Benchmark 5 — Rolling Window: nested loop vs sliding deque

```python
import time
from collections import deque
import numpy as np

ts = list(range(100_000))   # time series of 100K points
W = 50

# ❌ Naive: recompute sum over window from scratch each step — O(n × W)
t0 = time.perf_counter()
naive_roll = []
for i in range(W - 1, len(ts)):
    naive_roll.append(sum(ts[i - W + 1 : i + 1]) / W)   # sums W elements every step
print(f"Naive loop:    {time.perf_counter() - t0:.3f}s")   # ~0.5s

# ✅ Optimised: running sum with deque — O(n)
t0 = time.perf_counter()
window  = deque(maxlen=W)   # auto-evicts oldest element
running = 0.0
fast_roll = []
for x in ts:
    if len(window) == W:
        running -= window[0]       # subtract the element being evicted
    window.append(x)
    running += x
    if len(window) == W:
        fast_roll.append(running / W)
print(f"Deque O(n):    {time.perf_counter() - t0:.3f}s")   # ~0.015s  (~33×)

# ✅✅ NumPy stride tricks — zero Python loops, fastest for batch processing
t0 = time.perf_counter()
arr = np.array(ts, dtype=np.float64)
np_roll = np.lib.stride_tricks.sliding_window_view(arr, W).mean(axis=1)
print(f"NumPy strides: {time.perf_counter() - t0:.3f}s")   # ~0.002s  (~250×)
```

#### Benchmark 6 — Deduplication: nested loop vs set

```python
import time, random

data = [random.randint(0, 10_000) for _ in range(500_000)]  # ~50% duplicates

# ❌ Naive: check if already in result list — O(n²)
t0 = time.perf_counter()
seen, unique_naive = [], []
for x in data:
    if x not in seen:       # O(n) scan every time
        seen.append(x)
        unique_naive.append(x)
print(f"Naive list dedup:  {time.perf_counter() - t0:.2f}s  → {len(unique_naive)} unique")

# ✅ Optimised: set for O(1) seen-check
t0 = time.perf_counter()
seen_set = set()
unique_fast = []
for x in data:
    if x not in seen_set:   # O(1) average
        seen_set.add(x)
        unique_fast.append(x)
print(f"Set dedup:         {time.perf_counter() - t0:.4f}s  → {len(unique_fast)} unique")
# Naive: ~25s   Set: ~0.06s  → ~400× faster at n=500K
```

---

### 3.3 When the Speedup Is Non-Negotiable in Production

The table below maps each rewrite to the production consequence of getting it wrong.

| Naive pattern | Consequence at production scale | Fix |
|---|---|---|
| `list.pop(0)` in a streaming pipeline | Mini-batch queue becomes the bottleneck; GPU starves waiting for CPU | `deque.popleft()` |
| `if x in vocab_list` in tokeniser | 10ms latency per token → model serving too slow for SLA | `if x in vocab_set` |
| `data.count(label)` for class distribution | Feature pipeline hangs for hours on 10M rows | `Counter(data)` |
| `sorted(candidates)[:k]` in re-ranking | Re-ranker is 10× slower than the retriever it wraps | `heapq.nlargest(k)` |
| Nested loop rolling mean in feature store | Daily feature refresh takes 8 hours instead of 15 minutes | `pd.rolling()` or deque |
| Brute-force cosine search over 10M embeddings | RAG retrieval latency 10s → user-facing timeout | FAISS IVF+PQ |
| Pairwise Jaccard for LLM data dedup | O(n²) comparison on 1B docs is computationally infeasible | MinHash LSH |
| `len(set(events))` for DAU in streaming | Holds all user IDs in RAM — GBs for large systems | HyperLogLog |

> 🎯 **Interview prep**: When an interviewer says "how would you optimise this?" they are almost always asking you to identify a naive O(n²) pattern and replace it with the correct data structure. The mental checklist: Is there a lookup in a loop? → use `set`/`dict`. Is there a queue implemented with a list? → use `deque`. Is there a full sort to find K items? → use `heapq`. Is there a nested loop producing pairs? → ask if you need all pairs or just the best ones.

> 🏭 **Production note**: The single highest-leverage rule in data science code is: **never call any method inside a `for` loop that has to scan the full collection**. That includes `list.count()`, `list.index()`, `if x in my_list`, and `list.remove()`. Each one turns O(n) iterations into O(n²) work.

**Resources**
- [CPython Time Complexity wiki](https://wiki.python.org/moin/TimeComplexity) — the definitive source for Python data structure costs
- [Python performance tips (Real Python)](https://realpython.com/python-performance-tips/) — practical rewrites for common bottlenecks

---

## 4. Arrays & NumPy Arrays

Python's `list` is a dynamic array — each element is a pointer to a Python object, which means iterating 10 million floats in a list touches 10 million object headers. NumPy's `ndarray` stores raw numbers contiguously in memory, which is why `np.sum(arr)` on a million elements runs 100× faster than `sum(arr)`. Every tensor in PyTorch and every feature matrix in scikit-learn is ultimately a contiguous block of typed memory — understanding this is the foundation for everything else.

### 3.1 Python list vs NumPy ndarray

```python
import numpy as np
import time

n = 1_000_000

# Python list: 10M object pointers
py_list = list(range(n))
t0 = time.perf_counter()
result = sum(x**2 for x in py_list)
print(f"Python list: {time.perf_counter() - t0:.3f}s")  # ~0.15s

# NumPy array: contiguous float64 memory
np_arr = np.arange(n, dtype=np.float64)
t0 = time.perf_counter()
result = np.sum(np_arr**2)
print(f"NumPy array: {time.perf_counter() - t0:.3f}s")  # ~0.002s

# Broadcasting: compute pairwise distances without loops
X = np.random.randn(1000, 128)           # 1000 samples, 128 features
norms = np.linalg.norm(X, axis=1)        # L2 norm per sample, shape (1000,)
X_normed = X / norms[:, np.newaxis]      # broadcast: divide each row by its norm

# Efficient window slicing with stride tricks
from numpy.lib.stride_tricks import sliding_window_view
ts = np.random.randn(100)               # time series of 100 points
windows = sliding_window_view(ts, 7)    # shape (94, 7): 94 windows of length 7
rolling_mean = windows.mean(axis=1)     # rolling mean, zero Python loops
```

### 3.2 Key NumPy Patterns for DS/ML

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Vectorised conditional: replace negatives with 0
arr_clipped = np.where(arr > 3, arr, 0)    # [0, 0, 4, 0, 5, 9, 0, 6]

# argsort: get indices that would sort the array (used in ranking)
ranks = np.argsort(arr)[::-1]              # descending rank order

# Boolean masking: select high-value predictions
scores = np.array([0.9, 0.3, 0.7, 0.1, 0.8])
top = scores[scores > 0.5]                 # [0.9, 0.7, 0.8]

# Advanced indexing: build a feature matrix from integer IDs
vocab = np.random.randn(10000, 128)        # embedding table: 10K tokens × 128 dim
token_ids = np.array([42, 7, 999])         # token IDs in a sentence
embeddings = vocab[token_ids]              # shape (3, 128): gather rows by index
```

> 🎯 **Interview prep**: "Why is NumPy faster than a Python loop?" — The answer is: contiguous memory layout (cache locality), SIMD/vectorised CPU instructions via BLAS, and elimination of Python object overhead. Not just "it's optimised."

> 🏭 **Production note**: `np.argsort` on 1M scores is fine (≈5ms). If you need the top-K only, use `np.argpartition(scores, -k)[-k:]` — it's O(n) instead of O(n log n) and 10× faster for k << n.

**Resources**
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — broadcasting, stride tricks, advanced indexing
- [numpy.lib.stride_tricks.sliding_window_view](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html) — zero-copy rolling windows

---

## 5. Hash Maps, Sets & Feature Encoding

The hash map is the most important data structure in practical data science. Vocabulary lookups in tokenisers, label-to-index mappings in classifiers, frequency counts for class imbalance analysis — all of these are O(1) hash map operations. Reaching for a sorted list or a linear scan when you need a lookup is one of the most common performance mistakes in data pipelines.

### 4.1 dict, Counter, defaultdict

```python
from collections import Counter, defaultdict

# Counter: frequency map for class distribution analysis
labels = ["cat", "dog", "cat", "bird", "dog", "cat"]
freq = Counter(labels)          # Counter({'cat': 3, 'dog': 2, 'bird': 1})
print(freq.most_common(2))      # [('cat', 3), ('dog', 2)]

# defaultdict: build inverted index (token → list of doc IDs)
inverted_index = defaultdict(list)
docs = {0: "cat sat mat", 1: "dog ran fast", 2: "cat ran away"}
for doc_id, text in docs.items():
    for token in text.split():
        inverted_index[token].append(doc_id)
print(dict(inverted_index))     # {'cat': [0, 2], 'sat': [0], ...}

# Vocabulary encoding: string → integer ID
vocab = {word: idx for idx, word in enumerate(inverted_index)}
print(vocab)                    # {'cat': 0, 'sat': 1, ...}

# Two-pass deduplication with a set
seen = set()
unique_events = []
for event in ["click", "view", "click", "purchase", "view"]:
    if event not in seen:       # O(1) lookup
        seen.add(event)
        unique_events.append(event)
print(unique_events)            # ['click', 'view', 'purchase']
```

### 4.2 Feature Hashing (The Hashing Trick)

When your categorical feature has millions of distinct values (user IDs, product SKUs), building an explicit vocabulary dictionary runs out of memory. Feature hashing maps any string to an integer in [0, n) using a hash function, with no dictionary required. ([Weinberger et al., 2009](https://arxiv.org/abs/0902.2206))

```python
from sklearn.feature_extraction import FeatureHasher

# Hash high-cardinality features into a fixed-size vector
hasher = FeatureHasher(n_features=2**18, input_type="string")
data = [["user_id=12345", "city=bangalore", "device=mobile"],
        ["user_id=99999", "city=mumbai", "device=desktop"]]
X = hasher.transform(data)          # sparse matrix (2, 262144)
print(X.shape)                      # (2, 262144) — fixed width regardless of vocab size
```

> 🎯 **Interview prep**: "How do you handle a categorical feature with 10 million unique values?" — Feature hashing: no vocabulary dictionary, fixed memory, O(1) per feature. Trade-off: hash collisions cause minor accuracy loss (empirically < 1% with n_features=2^18).

> 🏭 **Production note**: Use `mmh3` (MurmurHash3) directly for custom hashing pipelines — it's 5× faster than Python's built-in `hash()` and deterministic across processes (Python's `hash()` is randomised for security).

**Resources**
- [Feature Hashing for Large Scale Multitask Learning (Weinberger et al., 2009)](https://arxiv.org/abs/0902.2206) — the original hashing trick paper
- [Python collections module](https://docs.python.org/3/library/collections.html) — `Counter`, `defaultdict`, `OrderedDict`

---

## 6. Stacks & Queues

Stacks (LIFO) and queues (FIFO) are the backbone of graph traversal and streaming data processing. In data science, the most frequent production use is the mini-batch queue: a background thread fills a `deque` while the GPU trains, so GPU utilisation stays high. The less obvious use is pipeline dependency resolution: an ML workflow scheduler uses a queue-based topological sort (BFS) to decide which tasks to launch first.

### 5.1 deque: The Right Tool for Both

```python
from collections import deque

# Stack (LIFO): use for DFS, expression parsing, undo history
stack = deque()
stack.append("layer_1")        # push
stack.append("layer_2")
top = stack.pop()              # pop → "layer_2" (LIFO)

# Queue (FIFO): use for BFS, streaming mini-batch buffer
queue = deque(maxlen=1000)     # bounded buffer: auto-evicts oldest when full
queue.append("batch_1")        # enqueue
queue.append("batch_2")
first = queue.popleft()        # dequeue → "batch_1" (FIFO)

# Topological sort via BFS (Kahn's algorithm) for DAG pipelines
from collections import defaultdict

def topo_sort(nodes, edges):
    """Return execution order for a DAG of pipeline steps."""
    in_degree = defaultdict(int)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    queue = deque(n for n in nodes if in_degree[n] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in adj[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)
    return order

nodes = ["ingest", "clean", "features", "train", "evaluate"]
edges = [("ingest","clean"), ("clean","features"), ("features","train"), ("train","evaluate")]
print(topo_sort(nodes, edges))   # ['ingest', 'clean', 'features', 'train', 'evaluate']
```

> 🎯 **Interview prep**: "How would you detect a cycle in an ML pipeline DAG?" — Run Kahn's algorithm (BFS topo sort). If the output order has fewer nodes than the graph has, a cycle exists. O(V + E).

> 🏭 **Production note**: Never use a Python `list` as a queue — `list.pop(0)` is O(n) because it shifts every element. Always use `deque.popleft()` which is O(1).

**Resources**
- [Python collections.deque](https://docs.python.org/3/library/collections.html#collections.deque) — official docs with complexity notes

---

## 7. Heaps & Priority Queues

A heap maintains the smallest (or largest) element at the top in O(log n) time for insertions and O(log n) for removal — without ever fully sorting the collection. The key insight for data science: whenever you need the top-K of anything (top-K recommendations, K nearest neighbours, K most common labels), a heap of size K processes n items in O(n log k) time rather than O(n log n) for a full sort. For K=10 and n=1M, that's roughly 20× faster.

### 6.1 heapq: Python's Min-Heap

```python
import heapq

# Top-K recommendations from a scored item list
scores = [(0.9, "item_A"), (0.3, "item_B"), (0.7, "item_C"),
          (0.85, "item_D"), (0.1, "item_E")]

# nlargest: internally maintains a min-heap of size K → O(n log k)
top3 = heapq.nlargest(3, scores)   # [(0.9, 'item_A'), (0.85, 'item_D'), (0.7, 'item_C')]
print(top3)

# Streaming top-K: process items one at a time (e.g., from a data stream)
def streaming_topk(stream, k):
    """Keep the k highest-scored items seen so far, O(n log k) total."""
    heap = []
    for score, item in stream:
        heapq.heappush(heap, (score, item))    # push
        if len(heap) > k:
            heapq.heappop(heap)                # pop smallest (min-heap), keeping top-k
    return sorted(heap, reverse=True)

result = streaming_topk(scores, k=2)
print(result)   # [(0.9, 'item_A'), (0.85, 'item_D')]

# Merge K sorted prediction lists (e.g., from K retrieval shards)
lists = [[0.9, 0.7, 0.3], [0.85, 0.5, 0.1], [0.95, 0.6]]
merged = list(heapq.merge(*[sorted(l, reverse=True) for l in lists], reverse=True))
print(merged[:5])   # [0.95, 0.9, 0.85, 0.7, 0.6]
```

### 6.2 Streaming Median (Two-Heaps Pattern)

```python
import heapq

class StreamingMedian:
    """Maintain running median over a stream using two heaps."""
    def __init__(self):
        self.lo = []   # max-heap (store negated for Python's min-heap)
        self.hi = []   # min-heap

    def add(self, num):
        heapq.heappush(self.lo, -num)              # push to max-heap
        heapq.heappush(self.hi, -heapq.heappop(self.lo))  # balance
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2.0

sm = StreamingMedian()
for x in [5, 15, 1, 3, 2, 8]:
    sm.add(x)
print(sm.median())   # 4.0
```

> 🎯 **Interview prep**: "How would you find the top-100 products by sales from a 10 billion row log?" — Maintain a min-heap of size 100. For each row, if the sale count exceeds the heap minimum, push and pop. O(n log 100) ≈ O(7n). Memory: O(100). No sorting required.

> 🏭 **Production note**: `heapq.nlargest(k, iterable)` is faster than `sorted(iterable, reverse=True)[:k]` when k << n. For k > n/2, just sort. CPython's implementation switches strategy automatically.

**Resources**
- [heapq — Python docs](https://docs.python.org/3/library/heapq.html) — `nlargest`, `nsmallest`, `merge`
- [Beam Search in sequence generation (Hugging Face)](https://huggingface.co/blog/how-to-generate) — heap-based beam search in practice

---

## 8. Sorting & Binary Search

Every ranking metric in ML — NDCG, MAP, precision@k — requires sorting predictions by score. Python's Timsort (`sorted`, `list.sort`) is O(n log n) and adaptive: it runs in O(n) on already-sorted data, which matters when you're re-ranking a prediction list that was previously sorted. Binary search (`bisect`) gives O(log n) lookup in any sorted sequence, and its less obvious use case is threshold calibration: binary-searching over the score axis to find the threshold that achieves a target recall.

### 7.1 Sorting for Ranking Metrics

```python
import numpy as np
from bisect import bisect_left, bisect_right, insort

# Sort predictions for NDCG computation
y_scores = [0.9, 0.3, 0.7, 0.1, 0.85]
y_true   = [1,   0,   1,   0,   1  ]

# Sort by score descending → get relevance in rank order
ranked = sorted(zip(y_scores, y_true), reverse=True)
relevances = [r for _, r in ranked]   # [1, 1, 1, 0, 0]

# NDCG@k
def dcg(relevances, k):
    """Discounted Cumulative Gain at k."""
    return sum(rel / np.log2(i + 2)          # log base-2, 1-indexed
               for i, rel in enumerate(relevances[:k]))

ideal = dcg(sorted(relevances, reverse=True), k=3)
ndcg = dcg(relevances, k=3) / ideal if ideal > 0 else 0
print(f"NDCG@3 = {ndcg:.4f}")

# Binary search: find threshold for target recall
sorted_scores = sorted(y_scores)              # sort once
threshold_idx = bisect_left(sorted_scores, 0.6)  # O(log n): first score >= 0.6
print(f"Scores >= 0.6: {sorted_scores[threshold_idx:]}")  # [0.7, 0.85, 0.9]

# insort: maintain a sorted list of running scores (O(log n) search + O(n) insert)
running = []
for score in y_scores:
    insort(running, score)
print(running)   # [0.1, 0.3, 0.7, 0.85, 0.9]
```

> 🎯 **Interview prep**: "What is Timsort and why does Python use it?" — Timsort is a hybrid merge-sort / insertion-sort that exploits natural runs in real data. It's O(n) on sorted input and O(n log n) worst case. Perfect for re-sorting nearly-sorted prediction lists.

> 🏭 **Production note**: For very large arrays, `numpy.argsort` with `kind='stable'` is the right tool. For top-K only, `numpy.argpartition` is O(n) and 5-10× faster than full sort when k << n.

**Resources**
- [Python Sorting HOWTO](https://docs.python.org/3/howto/sorting.html) — key functions, stability, performance
- [bisect — array bisection algorithm](https://docs.python.org/3/library/bisect.html) — `bisect_left`, `bisect_right`, `insort`

---

## 9. Two Pointers & Sliding Window

Time-series feature engineering often reduces to one question: "what statistic applies to the last W timestamps?" The naive answer is a nested loop — O(n×W). The right answer is a sliding window — O(n). The two-pointer pattern is the same idea applied to sorted arrays: instead of checking all pairs O(n²), move two pointers inward and examine O(n) pairs total.

### 8.1 Sliding Window for Time-Series Features

```python
import numpy as np
import pandas as pd
from collections import deque

# NumPy: zero-copy rolling window (no Python loop)
ts = np.random.randn(1000)
windows = np.lib.stride_tricks.sliding_window_view(ts, window_shape=7)
rolling_mean  = windows.mean(axis=1)    # shape (994,)
rolling_std   = windows.std(axis=1)
rolling_max   = windows.max(axis=1)

# Pandas: rolling statistics for feature engineering
df = pd.DataFrame({"value": ts})
df["roll_mean_7"] = df["value"].rolling(7).mean()
df["roll_std_7"]  = df["value"].rolling(7).std()
df["roll_zscore"] = (df["value"] - df["roll_mean_7"]) / df["roll_std_7"]

# Deque-based sliding window max (O(n), not O(n×W))
def sliding_max(arr, w):
    """Compute sliding maximum in O(n) using a monotonic deque."""
    dq = deque()   # stores indices; front = index of current max
    result = []
    for i, val in enumerate(arr):
        while dq and arr[dq[-1]] <= val:   # remove smaller elements from back
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:                 # remove elements out of window
            dq.popleft()
        if i >= w - 1:
            result.append(arr[dq[0]])      # front is the max
    return result

data = [2, 3, 1, 1, 5, 2, 3, 6]
print(sliding_max(data, w=3))   # [3, 3, 5, 5, 5, 6]
```

### 8.2 Two Pointers: Deduplication and Merging

```python
# Two-pointer deduplication of a sorted array (in-place, O(n))
def deduplicate_sorted(arr):
    if not arr:
        return []
    write = 1
    for read in range(1, len(arr)):
        if arr[read] != arr[read - 1]:   # new value found
            arr[write] = arr[read]
            write += 1
    return arr[:write]

print(deduplicate_sorted([1, 1, 2, 3, 3, 3, 4]))   # [1, 2, 3, 4]

# Merge two sorted score lists (merge step of merge-sort, O(n + m))
def merge_sorted(a, b):
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] >= b[j]:      # descending: larger score first
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    return result + a[i:] + b[j:]

print(merge_sorted([0.9, 0.7, 0.3], [0.85, 0.5, 0.1]))
# [0.9, 0.85, 0.7, 0.5, 0.3, 0.1]
```

> 🎯 **Interview prep**: "Design a function that computes rolling 7-day revenue with O(n) complexity." — Use a deque-based sliding window: add today's revenue, subtract the value that just left the window, keep a running sum. No inner loop needed.

> 🏭 **Production note**: Pandas `rolling()` uses C extensions and is fast for batch processing. If you need streaming (one event at a time), use a `deque(maxlen=W)` — it auto-evicts old values and is O(1) per new event.

**Resources**
- [pandas.DataFrame.rolling](https://pandas.pydata.org/docs/user_guide/window.html) — all window statistics in pandas
- [numpy.lib.stride_tricks](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html) — zero-copy rolling windows

---

## 10. Linked Lists & LRU Cache

You will likely never implement a linked list in production Python — the standard library handles it. But the *concept* appears in one of the most important DS/ML systems: the KV cache in LLM inference. vLLM's PagedAttention manages GPU memory with an LRU (Least Recently Used) eviction policy: when GPU memory fills up, the least-recently-used sequence's KV cache is evicted. The canonical LRU cache implementation is a doubly linked list + hash map, and Python's `OrderedDict` implements exactly this.

### 9.1 LRU Cache in Python

```python
from collections import OrderedDict
import functools

# functools.lru_cache: memoisation for expensive pure functions
@functools.lru_cache(maxsize=512)
def compute_feature(user_id: int, feature_name: str) -> float:
    """Expensive feature lookup — cached automatically."""
    return hash((user_id, feature_name)) % 100 / 100.0  # simulate expensive computation

print(compute_feature(42, "tenure"))     # computed
print(compute_feature(42, "tenure"))     # cache hit, O(1)
print(compute_feature.cache_info())      # CacheInfo(hits=1, misses=1, maxsize=512, currsize=1)

# Manual LRU cache using OrderedDict (interview pattern)
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()           # maintains insertion/access order

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)          # mark as most recently used
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)   # evict least recently used (front)

lru = LRUCache(3)
lru.put("embed_user_1", [0.1, 0.2, 0.3])
lru.put("embed_user_2", [0.4, 0.5, 0.6])
lru.put("embed_user_3", [0.7, 0.8, 0.9])
lru.get("embed_user_1")                      # access user_1 → moves to back (MRU)
lru.put("embed_user_4", [1.0, 1.1, 1.2])    # evicts user_2 (LRU)
print(list(lru.cache.keys()))                # ['embed_user_3', 'embed_user_1', 'embed_user_4']
```

> 🎯 **Interview prep**: "Design a feature store cache for serving real-time ML features." — LRU cache with `OrderedDict`: O(1) get and put. Capacity = number of hot users. Add a TTL by storing `(value, timestamp)` tuples and checking expiry on `get`. This is exactly how Redis and feature stores implement caching.

> 🏭 **Production note**: `functools.lru_cache` is not thread-safe for concurrent writes. For multi-threaded serving, use `cachetools.LRUCache` with a threading lock, or offload caching to Redis.

**Resources**
- [functools.lru_cache docs](https://docs.python.org/3/library/functools.html#functools.lru_cache) — built-in memoisation
- [vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) — KV cache management with paged eviction

---

## 11. Trees & Binary Search Trees

Decision trees are the most visible use of tree data structures in data science — every `sklearn.tree.DecisionTreeClassifier` is a binary tree where each internal node stores a feature index and threshold, and each leaf stores a class distribution. But trees appear in less obvious places too: segment trees answer range-max queries over time-series features in O(log n) instead of O(n); interval trees detect overlapping date ranges in event logs; expression trees represent hyperparameter search spaces in AutoML.

### 10.1 Binary Tree Node (Interview Pattern)

```python
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class TreeNode:
    val: Any
    left: Optional["TreeNode"] = field(default=None)
    right: Optional["TreeNode"] = field(default=None)

# Build a simple binary tree
root = TreeNode(val=5)
root.left  = TreeNode(val=3)
root.right = TreeNode(val=8)
root.left.left  = TreeNode(val=1)
root.left.right = TreeNode(val=4)

# Inorder traversal (left-root-right) → sorted order for BST
def inorder(node):
    if node is None:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

print(inorder(root))   # [1, 3, 4, 5, 8]
```

### 10.2 SortedList as Production BST

```python
from sortedcontainers import SortedList

# Maintain a sorted running list of prediction scores
running_scores = SortedList()
for score in [0.9, 0.3, 0.7, 0.1, 0.85, 0.5]:
    running_scores.add(score)           # O(log n) insert

# O(log n) rank query: how many scores are below 0.6?
rank = running_scores.bisect_left(0.6)
print(f"Scores below 0.6: {rank}")      # 3

# O(log n) range query: scores between 0.5 and 0.8
lo = running_scores.bisect_left(0.5)
hi = running_scores.bisect_right(0.8)
print(list(running_scores[lo:hi]))      # [0.5, 0.7]
```

### 10.3 Decision Tree Internals

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([[2.5, 1.0], [1.0, 3.0], [3.0, 2.0], [0.5, 0.5]])
y = np.array([1, 0, 1, 0])
clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

# Inspect the internal tree structure
tree = clf.tree_
print("Feature indices for splits:", tree.feature[:tree.node_count])
print("Thresholds:", tree.threshold[:tree.node_count])
print("Leaf node values:", tree.value[tree.children_left == -1])
```

> 🎯 **Interview prep**: "How does a decision tree decide where to split?" — It tries every feature and threshold, computes the Gini impurity (or information gain) of the resulting split, and picks the one that maximises the reduction in impurity. Complexity: O(n × d × log n) per node, where d = number of features.

**Resources**
- [sortedcontainers docs](http://www.grantjenks.com/docs/sortedcontainers/) — pure-Python O(log n) sorted structures
- [sklearn Decision Tree](https://scikit-learn.org/stable/modules/tree.html) — tree internals and feature importance

---

## 12. Graphs, BFS & DFS

Graphs are everywhere in modern ML infrastructure. An Airflow DAG is a directed acyclic graph where BFS determines task execution order. A knowledge graph is a heterogeneous attributed graph where BFS extracts entity neighbourhoods. A PyTorch computational graph is a DAG where reverse-mode autograd runs DFS from loss node to parameters. Graph Neural Networks extend all of this to learned representations over graph-structured data.

### 11.1 BFS and DFS in Python

```python
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.adj = defaultdict(list)       # adjacency list

    def add_edge(self, u, v):
        self.adj[u].append(v)

    def bfs(self, start):
        """BFS: level-by-level traversal. Use for shortest path, pipeline ordering."""
        visited = {start}
        queue = deque([start])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in self.adj[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return order

    def dfs(self, start, visited=None):
        """DFS: deep exploration. Use for cycle detection, connected components."""
        if visited is None:
            visited = set()
        visited.add(start)
        result = [start]
        for neighbour in self.adj[start]:
            if neighbour not in visited:
                result.extend(self.dfs(neighbour, visited))
        return result

# Example: dependency graph for a feature pipeline
g = Graph()
for u, v in [("raw","clean"), ("clean","features"), ("features","model"),
             ("clean","validate"), ("validate","model")]:
    g.add_edge(u, v)

print("BFS order:", g.bfs("raw"))    # level-by-level execution order
print("DFS order:", g.dfs("raw"))    # deep-first path exploration
```

### 11.2 Cycle Detection (Critical for Pipeline Validation)

```python
def has_cycle(graph):
    """Detect cycle in directed graph using DFS with 3-colour marking."""
    WHITE, GREY, BLACK = 0, 1, 2   # unvisited, in-progress, done
    colour = {node: WHITE for node in graph.adj}
    colour.update({v: WHITE for u in graph.adj for v in graph.adj[u]})

    def dfs(node):
        colour[node] = GREY
        for neighbour in graph.adj.get(node, []):
            if colour.get(neighbour, WHITE) == GREY:
                return True          # back-edge → cycle
            if colour.get(neighbour, WHITE) == WHITE:
                if dfs(neighbour):
                    return True
        colour[node] = BLACK
        return False

    return any(dfs(n) for n in list(colour) if colour[n] == WHITE)

# Add a cycle and detect it
g.add_edge("model", "raw")           # creates cycle
print("Has cycle:", has_cycle(g))    # True
```

### 11.3 networkx for Production Graphs

```python
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([("ingest","clean"), ("clean","features"), ("features","train")])
print("Topological order:", list(nx.topological_sort(G)))
print("Has cycle:", not nx.is_directed_acyclic_graph(G))
print("Predecessors of 'train':", list(G.predecessors("train")))
```

> 🎯 **Interview prep**: "How does PyTorch autograd work?" — PyTorch builds a DAG where each operation is a node and each tensor is an edge. Backpropagation is a DFS traversal from the loss node, accumulating gradients via the chain rule. Calling `.backward()` triggers this traversal.

> 🏭 **Production note**: For graphs with millions of nodes (social networks, knowledge graphs), `networkx` is too slow — it stores Python objects. Use `scipy.sparse` adjacency matrices for bulk operations, or PyTorch Geometric (`torch_geometric`) for GNNs.

**Resources**
- [NetworkX docs](https://networkx.org/documentation/stable/) — graph algorithms, layout, I/O
- [A Comprehensive Survey on Graph Neural Networks (Wu et al., 2020)](https://arxiv.org/abs/1901.00596) — GCN, GraphSAGE, GAT, spatial-temporal GNNs
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — GNNs in production

---

## 13. Union-Find (Disjoint Set Union)

Union-Find answers one question in near-constant time: "are these two items in the same group?" It maintains a forest of disjoint sets, and two optimisations — path compression and union by rank — make both `find` and `union` run in O(α(n)) ≈ O(1) amortised time, where α is the inverse Ackermann function. In data science, this appears in entity resolution (are these two customer records the same person?), connected-component labelling in image segmentation, and hierarchical clustering.

### 12.1 Union-Find Implementation

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))     # each node is its own root initially
        self.rank   = [0] * n            # tree depth estimate for union by rank

    def find(self, x):
        """Path compression: flatten the tree on every find."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # recursive path compression
        return self.parent[x]

    def union(self, x, y):
        """Union by rank: attach smaller tree under larger tree."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False                 # already connected
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx             # ry's root points to rx's root
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Entity resolution: merge customer records that share the same email
records = list(range(6))               # 6 customer records
uf = UnionFind(6)
same_customer = [(0, 2), (1, 3), (2, 4)]   # records known to be the same person
for a, b in same_customer:
    uf.union(a, b)

print(uf.connected(0, 4))   # True: 0→2→4 same customer
print(uf.connected(0, 1))   # False: different customers
```

### 12.2 scipy for Large-Scale Connected Components

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

# Build adjacency matrix for a co-purchase graph (sparse format)
rows = np.array([0, 0, 1, 2, 3])
cols = np.array([1, 2, 2, 3, 4])
data = np.ones(5)
adj = sp.csr_matrix((data, (rows, cols)), shape=(5, 5))
adj = adj + adj.T                        # make symmetric (undirected)

n_components, labels = csgraph.connected_components(adj, directed=False)
print(f"Components: {n_components}, Labels: {labels}")  # 2 components: [0,1,2,3,4] split
```

> 🎯 **Interview prep**: "How would you deduplicate a dataset where duplicate pairs are detected by a similarity model?" — Build a graph where duplicate pairs are edges, then find connected components with Union-Find. All records in the same component are duplicates; keep one representative per component.

**Resources**
- [scipy.sparse.csgraph.connected_components](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html)
- [Union-Find explainer (Sedgewick, Algorithms 4e)](https://algs4.cs.princeton.edu/15uf/)

---

## 14. Dynamic Programming

Dynamic programming solves problems by breaking them into overlapping subproblems and storing the result of each subproblem to avoid recomputation. In data science, it powers: edit distance (data cleaning, record deduplication), the Viterbi algorithm (sequence labelling with HMMs/CRFs), CTC loss (Connectionist Temporal Classification — the training objective in Whisper and other ASR models), and the sequence alignment algorithms underlying BLEU score computation.

### 13.1 Edit Distance (Levenshtein)

```python
import functools

# Top-down DP with memoisation
@functools.lru_cache(maxsize=None)
def edit_distance(s1, s2):
    """Minimum edit operations (insert, delete, substitute) to convert s1 to s2."""
    if not s1: return len(s2)    # insert all of s2
    if not s2: return len(s1)    # delete all of s1
    if s1[0] == s2[0]:
        return edit_distance(s1[1:], s2[1:])    # characters match, no cost
    return 1 + min(
        edit_distance(s1[1:], s2),              # delete s1[0]
        edit_distance(s1, s2[1:]),              # insert s2[0]
        edit_distance(s1[1:], s2[1:])           # substitute
    )

print(edit_distance("kitten", "sitting"))   # 3

# Bottom-up DP (avoids recursion limit, O(n×m) time and space)
def edit_distance_dp(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i   # delete i chars from s1
    for j in range(m + 1): dp[0][j] = j   # insert j chars from s2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # delete
                           dp[i][j-1] + 1,       # insert
                           dp[i-1][j-1] + cost)  # substitute
    return dp[n][m]

print(edit_distance_dp("kitten", "sitting"))   # 3
```

### 13.2 Longest Common Subsequence (BLEU/ROUGE Internals)

```python
def lcs_length(s1, s2):
    """Length of longest common subsequence — used in ROUGE-L metric."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

ref   = "the cat sat on the mat".split()
hyp   = "the cat is on the mat".split()
lcs   = lcs_length(ref, hyp)
rouge_l = (2 * lcs) / (len(ref) + len(hyp))   # F1-style ROUGE-L
print(f"ROUGE-L = {rouge_l:.4f}")   # 0.9167
```

> 🎯 **Interview prep**: "What DP algorithm underlies CTC loss (used in Whisper)?" — CTC uses forward-backward DP similar to the Baum-Welch algorithm. The forward pass computes the probability of all alignments ending at each timestep; the backward pass accumulates gradients. Complexity: O(T × U) where T = time steps, U = output symbols.

> 🏭 **Production note**: For production edit-distance on large datasets, use `rapidfuzz` (C++ backend, 50–100× faster than pure Python). For computing BLEU/ROUGE at scale, use `sacrebleu` or `evaluate` from Hugging Face.

**Resources**
- [CTC loss (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — original CTC paper, used in all modern ASR
- [rapidfuzz docs](https://rapidfuzz.github.io/RapidFuzz/) — fast fuzzy matching in production

---

## 15. Tries (Prefix Trees)

A trie stores strings character-by-character in a tree, where each path from root to leaf represents one string. Looking up whether a prefix exists takes O(m) time where m is the string length — independent of the vocabulary size. This is exactly what Hugging Face's `tokenizers` library uses internally for BPE merge lookup: rather than scanning the entire vocabulary, it walks the trie to find the longest matching token prefix. ([Sennrich et al., 2016](https://arxiv.org/abs/1508.07909))

### 14.1 Trie Implementation

```python
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)  # char → TrieNode
        self.is_end = False                    # marks end of a valid word

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]         # create node if missing
        node.is_end = True

    def search(self, word):
        """Exact match: O(m) where m = len(word)."""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        """Prefix existence check: used in tokeniser lookup."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def autocomplete(self, prefix):
        """Return all words with given prefix: used in search autocomplete."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        results = []
        def dfs(n, path):
            if n.is_end:
                results.append(prefix + path)
            for c, child in n.children.items():
                dfs(child, path + c)
        dfs(node, "")
        return results

# Build a simple NLP vocabulary trie
vocab = ["token", "tokenize", "tokenizer", "model", "modeling", "modelling"]
trie = Trie()
for word in vocab:
    trie.insert(word)

print(trie.search("tokenizer"))        # True
print(trie.starts_with("token"))       # True
print(trie.autocomplete("model"))      # ['model', 'modeling', 'modelling']
```

> 🎯 **Interview prep**: "How does BPE tokenisation work at inference time?" — Build a trie of all vocabulary tokens. For each position in the input, walk the trie to find the longest matching token (longest prefix match). This runs in O(m) per position rather than O(V×m) for a naive linear scan over the vocabulary.

> 🏭 **Production note**: In practice, use `pygtrie` for production trie workloads — it handles Unicode, prefix compression, and is significantly faster than a pure-Python dict-based trie. The Hugging Face `tokenizers` library (Rust-backed) uses similar structures for sub-millisecond tokenisation.

**Resources**
- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) — BPE tokenisation
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) — production tokenisation with trie-backed BPE

---

## 16. String Algorithms & Edit Distance

String manipulation is unavoidable in NLP data pipelines: normalising entity names, deduplicating training documents, computing BLEU/ROUGE/ChrF for evaluation, and finding near-duplicate records in a labelled dataset. The canonical production library is `rapidfuzz` — a C++-backed Python library that computes Levenshtein, Jaro-Winkler, and partial-ratio distances 50–100× faster than pure Python `difflib`.

### 15.1 Fuzzy Matching with rapidfuzz

```python
# pip install rapidfuzz
from rapidfuzz import distance, fuzz, process

# Edit distance: data cleaning — are "Nitish Harsoor" and "Nitish Harsur" the same person?
d = distance.Levenshtein.distance("Nitish Harsoor", "Nitish Harsur")
print(f"Edit distance: {d}")   # 1

# Fuzzy ratio: 0-100 score (100 = identical)
print(fuzz.ratio("Flipkart India", "Flipkart Pvt Ltd"))    # ~65
print(fuzz.token_sort_ratio("LG Electronics", "Electronics LG"))  # 100 (order-invariant)

# Bulk matching: find best match for each record from a reference list
# (e.g., map raw company names to canonical names)
raw_names    = ["Googl", "Microsoft Corp", "Amazn"]
canonical    = ["Google", "Microsoft", "Amazon", "Apple", "Meta"]
for name in raw_names:
    match, score, idx = process.extractOne(name, canonical)
    print(f"{name!r:20s} → {match!r} (score={score})")
```

### 15.2 n-gram Jaccard for Document Deduplication

```python
def shingle(text, k=3):
    """Create a set of k-character shingles for Jaccard similarity."""
    text = text.lower().replace(" ", "")
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def jaccard(a, b):
    sa, sb = shingle(a), shingle(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

s1 = "the quick brown fox"
s2 = "the fast brown fox"
print(f"Jaccard similarity: {jaccard(s1, s2):.3f}")   # ~0.6

# For large-scale deduplication, use MinHash (see Section 18)
```

> 🎯 **Interview prep**: "How would you deduplicate a dataset of 10 million documents?" — Exact dedup: hash each document (MD5/SHA256), O(1) lookup. Near-dedup: MinHash LSH (Section 18) — estimate Jaccard similarity in O(1) per pair using compact signature vectors. Used to deduplicate LLM training corpora like C4 and The Pile.

**Resources**
- [rapidfuzz docs](https://rapidfuzz.github.io/RapidFuzz/) — production fuzzy matching
- [ROUGE metric (Lin, 2004)](https://aclanthology.org/W04-1013/) — n-gram recall for summarisation evaluation

---

## 17. Matrix Operations & Linear Algebra

Every ML model is ultimately matrix multiplication. PCA is eigendecomposition of the covariance matrix. Collaborative filtering is low-rank matrix factorisation. The attention mechanism is three matrix multiplies (Q, K, V projections) followed by a softmax and another multiply. Understanding which matrix operations are cheap, which are expensive, and which have hardware-optimised implementations is what separates an engineer who writes slow model code from one who writes fast model code.

### 16.1 Core NumPy Linear Algebra

```python
import numpy as np

X = np.random.randn(1000, 50)    # 1000 samples, 50 features

# Covariance matrix: O(n × d²) — expensive for wide datasets
cov = np.cov(X.T)                 # (50, 50) covariance matrix

# PCA via SVD: more numerically stable than eigendecomposition
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
#  U: (1000, 50), S: (50,), Vt: (50, 50)
X_pca = U[:, :2] * S[:2]          # project onto top 2 PCs, shape (1000, 2)

# Explained variance ratio
explained_var = S**2 / (S**2).sum()
print(f"Top 2 PCs explain {explained_var[:2].sum():.1%} of variance")

# Matrix solve: ridge regression, O(d³)
# min ||Xw - y||² + λ||w||² → w = (XᵀX + λI)⁻¹Xᵀy
y = np.random.randn(1000)
lam = 0.1
A = X.T @ X + lam * np.eye(50)   # (50, 50) system
b = X.T @ y                       # (50,) RHS
w = np.linalg.solve(A, b)         # O(d³) — use solve, never np.linalg.inv()
print(f"Ridge weights shape: {w.shape}")   # (50,)

# Cosine similarity between embedding vectors
a = np.random.randn(128)
b_vec = np.random.randn(128)
cos_sim = np.dot(a, b_vec) / (np.linalg.norm(a) * np.linalg.norm(b_vec))
print(f"Cosine similarity: {cos_sim:.4f}")

# Batch cosine similarity (query vs all document embeddings)
query = np.random.randn(1, 128)
docs  = np.random.randn(10000, 128)
query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
docs_norm  = docs  / np.linalg.norm(docs,  axis=1, keepdims=True)
scores = (docs_norm @ query_norm.T).squeeze()  # (10000,)
top5 = np.argpartition(scores, -5)[-5:]        # O(n) top-5, not O(n log n)
```

> 🎯 **Interview prep**: "Why is `np.linalg.solve(A, b)` better than `np.linalg.inv(A) @ b`?" — `solve` uses LU decomposition, O(d³), and is numerically stable. `inv` also costs O(d³) but accumulates more floating-point error. Never invert a matrix when you can solve a linear system.

> 🏭 **Production note**: FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) rewrites the O(n²) attention matrix multiply to be IO-aware — it tiles the computation to keep intermediate matrices in fast SRAM rather than slow HBM, achieving 3–5× speedup without changing the mathematical result.

**Resources**
- [NumPy linalg docs](https://numpy.org/doc/stable/reference/routines.linalg.html) — SVD, eigendecomposition, solve
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

---

## 18. Reservoir Sampling

When a dataset is too large to fit in memory, or arrives as a stream of unknown length, you cannot use `random.sample` — it requires the full population. Reservoir sampling solves this: it produces a uniformly random sample of size k from a stream of n items (where n may be unknown) in a single pass, using only O(k) memory. ([Vitter, 1985](https://www.cs.umd.edu/~samir/498/vitter.pdf)) This is how distributed training frameworks sample mini-batches from datasets that don't fit in RAM, and how you build a representative evaluation set from a live production log.

### 17.1 Algorithm R

```python
import random

def reservoir_sample(stream, k):
    """
    Uniformly sample k items from an iterable stream of unknown length.
    Each item has equal probability k/n of being in the final sample.
    """
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)              # fill reservoir with first k items
        else:
            j = random.randint(0, i)            # random index in [0, i]
            if j < k:
                reservoir[j] = item             # replace with probability k/(i+1)
    return reservoir

# Sample 5 from a streaming log
log = (f"event_{i}" for i in range(1_000_000))     # generator: doesn't materialise
sample = reservoir_sample(log, k=5)
print(sample)   # 5 random events, each with probability 5/1M

# Weighted reservoir sampling (A-Res algorithm): sample proportional to weights
def weighted_reservoir_sample(stream_with_weights, k):
    """Stream of (item, weight) tuples. Heavier items sampled more often."""
    import heapq
    heap = []   # min-heap of (key, item)
    for item, weight in stream_with_weights:
        key = random.random() ** (1.0 / weight)       # Efraimidis & Spirakis key
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))      # replace min-key item
    return [item for _, item in heap]

# Sample 3 items proportional to weight (e.g., class-balanced sampling)
items = [(f"class_{c}_sample_{i}", w)
         for c, w in [("rare", 0.1), ("common", 0.9)]
         for i in range(100)]
print(weighted_reservoir_sample(items, k=3))
```

> 🎯 **Interview prep**: "How do you sample 1000 records uniformly from a 100GB log file in a single pass?" — Reservoir sampling (Algorithm R). Time: O(n), Space: O(k). Each item has equal probability k/n. For weighted sampling (e.g., class-balanced batches from imbalanced data), use the A-Res weighted variant.

> 🏭 **Production note**: PyTorch's `WeightedRandomSampler` uses a similar principle for class-imbalanced training — it doesn't load all weights into memory at once. For streaming datasets, implement reservoir sampling in the data loader worker.

**Resources**
- [Random Sampling with a Reservoir (Vitter, 1985)](https://www.cs.umd.edu/~samir/498/vitter.pdf)
- [Reservoir sampling — Wikipedia](https://en.wikipedia.org/wiki/Reservoir_sampling) — weighted variants explained

---

## 19. Probabilistic Data Structures

Some problems are too large for exact answers but perfect for approximate ones. A Bloom filter answers "have I seen this URL?" with zero false negatives and tunable false positive rate — in kilobytes, not gigabytes. HyperLogLog counts distinct items in a stream using only 1.5 KB of memory regardless of stream size. Count-Min Sketch estimates item frequencies without storing counts for every item. MinHash estimates Jaccard similarity between documents without comparing all pairs. All four are standard tools in large-scale ML pipelines.

### 18.1 Comparison Table

| Structure | Question answered | Error type | Memory | Python library |
|---|---|---|---|---|
| **Bloom filter** | "Is item X in set S?" | False positives only | O(m) bits | `pybloom_live` |
| **HyperLogLog** | "How many distinct items?" | ±2% relative error | ~1.5 KB | `hyperloglog` |
| **Count-Min Sketch** | "How frequent is item X?" | Overestimate only | O(w×d) counters | `datasketch.cms` |
| **MinHash LSH** | "Are docs A and B near-duplicates?" | Jaccard estimate ε | O(k) hashes | `datasketch` |

### 18.2 Bloom Filter: URL Deduplication at Web Scale

```python
# pip install pybloom-live
from pybloom_live import BloomFilter

# 1 million items, 0.1% false positive rate → ~2.4 MB
bf = BloomFilter(capacity=1_000_000, error_rate=0.001)

training_urls = [f"https://example.com/doc_{i}" for i in range(100_000)]
for url in training_urls:
    bf.add(url)

# O(k) lookup (k = number of hash functions, typically 10-20)
print("doc_5000 seen:", "https://example.com/doc_5000" in bf)   # True
print("doc_999999 seen:", "https://example.com/doc_999999" in bf)  # False (never added)
# A small fraction of "False" will incorrectly return True (false positives)
```

([Bloom, 1970](https://dl.acm.org/doi/10.1145/362686.362692))

### 18.3 HyperLogLog: Distinct Count in Streaming Pipelines

```python
# pip install hyperloglog
from hyperloglog import HyperLogLog

# Count distinct users in a streaming event log — uses only ~1.5 KB
hll = HyperLogLog(error_rate=0.01)   # 1% relative error
import random
user_ids = [random.randint(1, 1_000_000) for _ in range(5_000_000)]
for uid in user_ids:
    hll.add(str(uid))

print(f"Estimated distinct users: {len(hll):,}")   # close to 1,000,000
```

([Heule et al., 2013](https://dl.acm.org/doi/10.1145/2452376.2452456))

### 18.4 MinHash LSH: Near-Duplicate Document Detection

```python
# pip install datasketch
from datasketch import MinHash, MinHashLSH

# Build MinHash signatures for documents
def minhash_doc(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode("utf8"))
    return m

docs = {
    "doc1": "the quick brown fox jumps over the lazy dog",
    "doc2": "the quick brown fox leaps over the lazy dog",   # near-duplicate of doc1
    "doc3": "machine learning is a subset of artificial intelligence",
}

# LSH index: threshold=0.5 means docs with Jaccard ≥ 0.5 are candidates
lsh = MinHashLSH(threshold=0.5, num_perm=128)
signatures = {}
for name, text in docs.items():
    sig = minhash_doc(text)
    signatures[name] = sig
    lsh.insert(name, sig)

# Query: find near-duplicates of doc1
results = lsh.query(signatures["doc1"])
print(f"Near-duplicates of doc1: {results}")   # ['doc1', 'doc2']
```

> 🎯 **Interview prep**: "How do you deduplicate the 1 trillion tokens of a LLM training corpus?" — MinHash LSH. Convert each document to a MinHash signature (k random hash functions), insert into LSH index, query for near-duplicates with Jaccard ≥ threshold. This is how C4, RefinedWeb, and Dolma deduplicated their training data.

> 🏭 **Production note**: Bloom filter false positive rate degrades as occupancy exceeds capacity. Monitor `len(bf) / bf.capacity`. At >80% full, rebuild with double the capacity. Cuckoo filters are a modern alternative with lower FP rates and support for deletion.

**Resources**
- [Space/time trade-offs in hash coding with allowable errors (Bloom, 1970)](https://dl.acm.org/doi/10.1145/362686.362692)
- [HyperLogLog in practice (Heule et al., 2013)](https://dl.acm.org/doi/10.1145/2452376.2452456)
- [datasketch — MinHash, LSH, HyperLogLog](https://ekzhu.com/datasketch/documentation.html)

---

## 20. Approximate Nearest Neighbours

Exact nearest-neighbour search (brute-force cosine similarity) scales as O(n × d) per query — 10ms per query for 1 million 768-dim embeddings, which is 10,000 QPS on a single CPU. At 100 million vectors, exact search becomes prohibitive. Approximate Nearest Neighbour (ANN) search trades a small recall loss (typically 1-5%) for 100-1000× speedup, and it is the cornerstone of every production RAG system, recommendation engine, and semantic search application.

### 19.1 ANN Library Comparison

| Library | Algorithm | Speed | Recall | Memory | Best for |
|---|---|---|---|---|---|
| **FAISS** | IVF + PQ | ★★★★★ | ★★★★☆ | ★★★★★ | Large-scale (100M+), GPU support |
| **hnswlib** | HNSW | ★★★★★ | ★★★★★ | ★★★☆☆ | In-memory, highest recall |
| **Annoy** | Random forests | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | Static index, mmap for low RAM |
| **ScaNN** | Tree-AH | ★★★★★ | ★★★★★ | ★★★★☆ | Google TPUs, highest quality |

**Use FAISS** when: index > 10M vectors, or you need GPU acceleration, or memory is tight.
**Use hnswlib** when: index < 10M vectors, you need maximum recall, and all vectors fit in RAM.
**Use Annoy** when: index is static (no insertions), you want mmap for memory sharing.

### 19.2 FAISS: Billion-Scale Search

([Johnson et al., 2019](https://arxiv.org/abs/1702.08734))

```python
# pip install faiss-cpu   (or faiss-gpu for NVIDIA GPU)
import faiss
import numpy as np

d = 128          # embedding dimension
n = 100_000      # number of documents

# Generate document embeddings (in practice: from a sentence transformer)
docs  = np.random.randn(n, d).astype(np.float32)
query = np.random.randn(1, d).astype(np.float32)

# Flat (exact) index: brute-force baseline
index_flat = faiss.IndexFlatL2(d)
index_flat.add(docs)
D, I = index_flat.search(query, k=5)       # D=distances, I=indices
print("Exact top-5:", I[0])

# IVF+PQ: approximate index for 100M+ vectors
nlist   = 1000    # number of Voronoi cells (coarse quantiser)
m       = 8       # number of sub-quantisers (PQ segments)
bits    = 8       # bits per sub-quantiser code

quantiser = faiss.IndexFlatL2(d)
index_ivfpq = faiss.IndexIVFPQ(quantiser, d, nlist, m, bits)
index_ivfpq.train(docs)                    # must train before adding
index_ivfpq.add(docs)
index_ivfpq.nprobe = 32                    # search 32 of 1000 cells (recall vs speed)
D_approx, I_approx = index_ivfpq.search(query, k=5)
print("Approximate top-5:", I_approx[0])

# Save and load index (for production deployment)
faiss.write_index(index_ivfpq, "/tmp/my_index.faiss")
loaded_index = faiss.read_index("/tmp/my_index.faiss")
```

### 19.3 hnswlib: Highest Recall In-Memory Search

([Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320))

```python
# pip install hnswlib
import hnswlib
import numpy as np

d = 128
n = 100_000

docs  = np.random.randn(n, d).astype(np.float32)
query = np.random.randn(1, d).astype(np.float32)

# Build HNSW index
index = hnswlib.Index(space="cosine", dim=d)
index.init_index(max_elements=n, ef_construction=200, M=16)
index.add_items(docs, ids=np.arange(n))
index.set_ef(50)                           # query-time beam width (higher = more recall)

labels, distances = index.knn_query(query, k=5)
print("HNSW top-5:", labels[0])
```

> 🎯 **Interview prep**: "How does HNSW work?" — HNSW builds a multi-layer graph where upper layers have long-range connections (for fast global navigation) and lower layers have dense local connections (for precise local search). Query traversal starts at the top layer and greedily descends, O(log n) hops. Recall vs speed is controlled by `ef` (beam width).

> 🏭 **Production note**: FAISS `nprobe` is the main recall-latency dial: `nprobe=1` is fastest but low recall; `nprobe=nlist` is exact but defeats the purpose. A good starting point is `nprobe = sqrt(nlist)`, then tune with recall@k benchmarks on your data.

**Resources**
- [FAISS documentation](https://faiss.ai/) — indexing guide, GPU support, benchmarks
- [Billion-scale similarity search with GPUs (Johnson et al., 2019)](https://arxiv.org/abs/1702.08734)
- [Efficient and Robust ANN Search using HNSW (Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320)

---

## 21. Bit Manipulation

Bit manipulation is the hidden enabler of two critical ML systems: quantisation and sparse feature encoding. Model quantisation packs 32-bit float weights into 4-bit integers, reducing model size 8× and enabling inference that was previously memory-bound to become compute-bound. In feature engineering, binary feature vectors (0/1) can be packed into bitfields and compared with `popcount` (Hamming distance) — GPU hardware can compute this 32× faster than float comparisons.

### 20.1 Python Bit Operations

```python
# Python int supports arbitrary-precision bitwise operations
x = 0b1010_1100    # 172
print(bin(x & 0xFF))               # AND mask: 0b10101100
print(bin(x | 0b0000_0011))        # OR: set bits 0,1 → 0b10101111
print(bin(x ^ 0b1111_1111))        # XOR: flip all bits → 0b01010011
print(bin(x << 2))                 # left shift: multiply by 4
print(bin(x >> 1))                 # right shift: floor divide by 2

# Count set bits (Hamming weight): popcount
def popcount(n):
    count = 0
    while n:
        count += n & 1    # test least significant bit
        n >>= 1           # shift right
    return count
print(popcount(0b10110110))   # 5

# Python one-liner using bin
print(bin(0b10110110).count('1'))   # 5
```

### 20.2 INT8 Quantisation in NumPy

```python
import numpy as np

# Simulate quantising float32 weights to int8 (like bitsandbytes does)
weights_f32 = np.random.randn(1000).astype(np.float32)

# Symmetric per-tensor quantisation
abs_max = np.abs(weights_f32).max()
scale = abs_max / 127.0
weights_int8 = np.round(weights_f32 / scale).clip(-128, 127).astype(np.int8)

# Memory: float32 = 4 bytes/weight, int8 = 1 byte/weight (4× reduction)
print(f"float32 size: {weights_f32.nbytes} bytes")    # 4000
print(f"int8 size:    {weights_int8.nbytes} bytes")   # 1000

# Dequantise for inference (multiply by scale)
weights_dequant = weights_int8.astype(np.float32) * scale
error = np.abs(weights_f32 - weights_dequant).mean()
print(f"Mean quantisation error: {error:.6f}")   # ~0.003 for int8
```

> 🎯 **Interview prep**: "What is the trade-off in INT8 quantisation?" — 4× memory reduction and 2-4× throughput on hardware with INT8 matrix units (NVIDIA A100, H100). Accuracy loss is typically <1% with per-channel quantisation. INT4 (GPTQ/AWQ) gives 8× reduction with 1-3% accuracy loss on LLMs.

**Resources**
- [bitsandbytes library](https://huggingface.co/docs/bitsandbytes/index) — INT8/INT4 quantisation in PyTorch
- [GPTQ: Accurate Post-Training Quantization for GPUs (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)

---

## 22. Interval Trees & Range Queries

Interval trees answer one question efficiently: "which stored intervals overlap with this query interval?" In data science, this appears whenever you join events by time range: ad impressions matched to conversions within a 30-minute window, session spans overlapping with model deployment periods, or NLP annotation spans intersecting with named entity spans. A naive loop is O(n) per query; an interval tree is O(log n + k) where k is the number of matching intervals.

### 21.1 intervaltree for Event Overlap

```python
# pip install intervaltree
from intervaltree import IntervalTree, Interval
import pandas as pd

# Store user session intervals: (start_timestamp, end_timestamp, session_id)
sessions = IntervalTree()
sessions[0:30]   = "session_A"
sessions[15:45]  = "session_B"
sessions[50:90]  = "session_C"
sessions[20:25]  = "session_D"

# Query: which sessions were active at timestamp 20?
active = sessions[20]
print("Active at t=20:", {i.data for i in active})   # {'session_A', 'session_B', 'session_D'}

# Query: which sessions overlap with window [10, 35]?
overlapping = sessions[10:35]
print("Overlaps [10,35]:", {i.data for i in overlapping})   # A, B, D

# Pandas IntervalIndex: vectorised range joins for data pipelines
sessions_df = pd.DataFrame({
    "start": [0, 15, 50, 20],
    "end":   [30, 45, 90, 25],
    "label": ["A", "B", "C", "D"]
})
idx = pd.IntervalIndex.from_arrays(sessions_df.start, sessions_df.end, closed="both")
sessions_df.index = idx
# Find all sessions containing timestamp 20
containing_20 = sessions_df[idx.contains(20)]
print(containing_20)
```

> 🎯 **Interview prep**: "How would you compute attribution for ad clicks — matching each click to all impressions within a 30-minute lookback window?" — Build an interval tree over impression time ranges. For each click, query the tree in O(log n + k). Much faster than a range join in SQL for very large logs.

**Resources**
- [intervaltree Python library](https://github.com/chaimleib/intervaltree) — pure-Python, supports add/remove/query
- [pandas IntervalIndex](https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.html) — vectorised interval operations

---

## 23. Pick the Right Structure: Decision Guide

The most common interview mistake is spending time on the algorithm when the data structure choice was wrong from the start. Use this guide to select the right structure before writing code.

```
What is the primary operation in the hot path?
│
├── Lookup by key (exact match)
│   ├── Key is a string/integer → dict  (O(1) avg)
│   ├── Key is a prefix         → Trie  (O(m), m = key length)
│   └── Approximate membership  → Bloom filter
│
├── Find minimum / maximum
│   ├── Need top-K only        → heapq  (O(n log k))
│   ├── Need sorted order      → sorted() / SortedList
│   └── Streaming median       → Two-heap pattern
│
├── Ordered range query
│   ├── Sorted array + binary  → bisect  (O(log n))
│   ├── Dynamic insert + query → SortedList  (O(log n))
│   └── Interval overlap       → IntervalTree  (O(log n + k))
│
├── Graph / connectivity
│   ├── Shortest path (unweighted) → BFS with deque
│   ├── Cycle detection            → DFS with colour marking
│   ├── Connected components       → Union-Find
│   └── Topological order          → Kahn's BFS
│
├── Sequence / recurrence
│   ├── Overlapping subproblems    → DP with lru_cache
│   ├── String similarity          → Edit distance DP
│   └── Sequence labelling         → Viterbi DP
│
├── Similarity search over vectors
│   ├── < 1M vectors, max recall   → hnswlib
│   ├── > 10M vectors / GPU        → FAISS IVF+PQ
│   └── Near-duplicate documents   → MinHash LSH
│
└── Approximate counting / dedup (streaming, huge scale)
    ├── Distinct count             → HyperLogLog
    ├── Membership dedup           → Bloom filter
    └── Frequency estimation       → Count-Min Sketch
```

---

## 24. Common DS/ML Interview Questions & Answers

These are the questions that appear most frequently in data science and ML engineering coding rounds. For each, the expected answer hits both the algorithm and its DS/ML context.

**Q1. Find the top-K most frequent items in a stream.**
> Use a `Counter` + `heapq.nlargest(k, counter.items(), key=lambda x: x[1])`. Time: O(n log k). DS/ML: computing top-K label frequencies, top-K query terms, most common error types in a production log.

**Q2. Compute a rolling 7-day average over a time series with O(n) complexity.**
> Use a deque with `maxlen=7` (auto-evict) and a running sum variable. Add new value, subtract the evicted value. Time: O(n). DS/ML: rolling revenue, rolling engagement rate, any lagged feature.

**Q3. Design an LRU cache for an ML feature store.**
> `OrderedDict` with `move_to_end` on access and `popitem(last=False)` on eviction. Time: O(1) get/put. DS/ML: caching user embeddings, expensive SQL feature lookups, Bloom filters for cold-start.

**Q4. Given K sorted recommendation lists, merge them into one ranked list.**
> Use `heapq.merge()` (K-way merge) or maintain a min-heap of size K with `(score, list_idx, element_idx)` tuples. Time: O(n log K). DS/ML: merging retrieval results from multiple vector index shards.

**Q5. Find all pairs of near-duplicate documents in a corpus of 1 million docs.**
> MinHash LSH: generate k-shingle sets → MinHash signatures → LSH bucketing → within-bucket exact Jaccard check. Time: O(n × num_perms) for indexing, O(1) per pair check. DS/ML: training data deduplication for LLMs.

**Q6. Detect whether a data pipeline DAG has a circular dependency.**
> Run DFS with three-colour marking (WHITE/GREY/BLACK). A back-edge (GREY → GREY) indicates a cycle. Time: O(V + E). DS/ML: Airflow/Prefect raises this error automatically; knowing the algorithm helps you debug.

**Q7. Find the median of a stream of numbers.**
> Two-heap pattern: max-heap for lower half, min-heap for upper half. Balance after each insertion. Time: O(log n) insert, O(1) query. DS/ML: streaming monitoring of model latency percentiles.

**Q8. Count distinct users in a 10 billion row event stream using 2 KB of memory.**
> HyperLogLog with 1-2% relative error. Time: O(n), Space: ~1.5 KB. DS/ML: DAU/MAU metrics in data warehouses (BigQuery, Snowflake use HLL for `APPROX_COUNT_DISTINCT`).

**Q9. Given a vocabulary of 10 million tokens, check if a new token is out-of-vocabulary in O(1).**
> Bloom filter: insert all vocabulary tokens at build time (seconds), check membership in O(k) = O(1). False positive rate is tunable. DS/ML: OOV detection in tokenisers, URL deduplication in web crawlers.

**Q10. Compute edit distance between two strings and explain its ML applications.**
> Bottom-up DP: O(n × m) time and space. Applications: record deduplication (entity resolution), BLEU/ROUGE computation (n-gram overlap), typo correction in search queries, sequence alignment in bioinformatics.

---

## 25. Production Failure Modes

Every data structure has a pathological case that can silently kill your pipeline. These are the ones that appear most often in production and are least obvious from documentation.

### 24.1 List as Queue (O(n) pop)
**Problem**: `my_list.pop(0)` is O(n) because every element shifts left. A streaming pipeline that pops from the front of a list processes items in O(n²) total instead of O(n).

**Fix**: Always use `collections.deque` for FIFO queues — `popleft()` is O(1).

### 24.2 dict / set Rehashing Stall
**Problem**: When a Python `dict` exceeds its load factor, it rehashes — all keys are re-inserted. This is O(n) and can cause a latency spike in real-time serving if a dict is growing during inference.

**Fix**: Pre-size dictionaries at creation: `dict.fromkeys(keys)` or initialise with known keys. For read-heavy serving, freeze the dict and use it as a lookup table.

### 24.3 Recursive DP Hitting RecursionError
**Problem**: Top-down DP with `@lru_cache` on strings longer than ~900 characters hits Python's default recursion limit (`sys.getrecursionlimit() == 1000`).

**Fix**: Always use bottom-up DP (explicit 2D array) for production. If you must use recursion, call `sys.setrecursionlimit(10_000)` at the top of the script — but never in a library.

### 24.4 networkx Too Slow for Large Graphs
**Problem**: `networkx` stores each node as a Python object. On a 10M-node graph, a single BFS traversal can take minutes.

**Fix**: For bulk graph operations, use `scipy.sparse.csgraph` (CSR matrix, C-backed). For GNNs, use `torch_geometric` with batched tensor operations.

### 24.5 Bloom Filter Saturation
**Problem**: A Bloom filter becomes unreliable once actual insertion count exceeds `capacity`. False positive rate climbs from the target (e.g., 0.1%) to >10%.

**Fix**: Monitor `len(bloom_filter) / bloom_filter.capacity`. Rebuild at >80% occupancy. Use `ScalableBloomFilter` from `pybloom_live` which auto-expands.

### 24.6 FAISS nprobe Set Too Low
**Problem**: Low `nprobe` (number of cells to search) gives fast but low-recall results. Teams often benchmark on small test sets where recall looks fine, but deploy with nprobe=1 and get poor recommendation quality.

**Fix**: Benchmark recall@k on a representative query set before deploying. For RAG systems, recall@10 should be ≥ 0.95 to avoid missed context.

### 24.7 Sliding Window Pandas vs NumPy Disagreement
**Problem**: `pd.rolling(7).mean()` uses NaN for the first 6 values; `np.lib.stride_tricks.sliding_window_view` returns only (n-6) values. Mixing them in a feature pipeline produces shape mismatches.

**Fix**: Choose one approach consistently. Pad with `NaN` if needed for alignment.

### 24.8 heapq With Tuples: Comparison on Second Element
**Problem**: `heapq.heappush(heap, (score, item))` breaks if two scores are equal and `item` is not comparable (e.g., a dict or numpy array). Python will try to compare the second element and raise a `TypeError`.

**Fix**: Use a tie-breaker counter: `heapq.heappush(heap, (score, counter, item))`. This is the canonical pattern.

---

## 26. References

### Foundational Papers
- Bloom, B.H. (1970). *Space/time trade-offs in hash coding with allowable errors.* Communications of the ACM, 13(7). https://dl.acm.org/doi/10.1145/362686.362692
- Vitter, J.S. (1985). *Random Sampling with a Reservoir.* ACM TOMS. https://www.cs.umd.edu/~samir/498/vitter.pdf
- Weinberger, K. et al. (2009). *Feature Hashing for Large Scale Multitask Learning.* ICML 2009. https://arxiv.org/abs/0902.2206

### Algorithms & Data Structures
- Malkov, Y.A. & Yashunin, D.A. (2018). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI. https://arxiv.org/abs/1603.09320
- Johnson, J., Douze, M. & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE TBDS. https://arxiv.org/abs/1702.08734
- Heule, S., Nunkesser, M. & Hall, A. (2013). *HyperLogLog in practice.* EDBT 2013. https://dl.acm.org/doi/10.1145/2452376.2452456

### NLP & Tokenisation
- Sennrich, R., Haddow, B. & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. https://arxiv.org/abs/1508.07909

### ML Infrastructure & Systems
- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. https://arxiv.org/abs/2205.14135

### Graphs & GNNs
- Wu, Z. et al. (2020). *A Comprehensive Survey on Graph Neural Networks.* IEEE TNNLS. https://arxiv.org/abs/1901.00596

### Libraries & Tools
- [CPython Time Complexity](https://wiki.python.org/moin/TimeComplexity) — authoritative Big-O for Python built-ins
- [FAISS documentation](https://faiss.ai/) — Meta AI's library for large-scale similarity search
- [hnswlib](https://github.com/nmslib/hnswlib) — fast HNSW-based ANN search
- [datasketch](https://ekzhu.com/datasketch/documentation.html) — MinHash, LSH, HyperLogLog
- [sortedcontainers](http://www.grantjenks.com/docs/sortedcontainers/) — pure-Python BST alternatives
- [networkx](https://networkx.org/documentation/stable/) — graph algorithms in Python
- [rapidfuzz](https://rapidfuzz.github.io/RapidFuzz/) — fast fuzzy string matching
- [intervaltree](https://github.com/chaimleib/intervaltree) — interval overlap queries

### Interview Preparation
- [Tech Interview Handbook — DSA Cheatsheets](https://www.techinterviewhandbook.org/algorithms/study-cheatsheet/) — pattern-based interview prep
- [LeetCode Top Interview 150](https://leetcode.com/studyplan/top-interview-150/) — curated problem set
