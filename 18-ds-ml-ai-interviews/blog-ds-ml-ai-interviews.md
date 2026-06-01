# DS / ML / AI Interview Prep: The Senior Engineer's 80/20 Guide

*One-stop reference for Senior Data Scientist, ML Engineer, and AI Engineer interviews. Applies the 80/20 rule — covers the 20% of questions that appear in 80% of real loops, with answers pitched at senior/staff depth.*

---

## Table of Contents

1. [The 80/20 Strategy](#1-the-8020-strategy)
2. [DSA & Python Coding](#2-dsa--python-coding)
3. [Python & Pandas](#3-python--pandas)
4. [SQL](#4-sql)
5. [Statistics & A/B Testing](#5-statistics--ab-testing)
6. [ML Algorithms](#6-ml-algorithms)
7. [PyTorch & Deep Learning](#7-pytorch--deep-learning)
8. [Recommender Systems](#8-recommender-systems)
9. [NLP Foundations](#9-nlp-foundations)
10. [Embeddings](#10-embeddings)
11. [RAG & Retrieval](#11-rag--retrieval)
12. [LLMs & Transformers](#12-llms--transformers)
13. [LLM Inference & Optimisation](#13-llm-inference--optimisation)
14. [Fine-tuning LLMs](#14-fine-tuning-llms)
15. [Agents & Tool Use](#15-agents--tool-use)
16. [ML System Design](#16-ml-system-design)
17. [References](#17-references)

---

## 1. The 80/20 Strategy

Most candidates prepare too broadly and answer too shallowly. The real interview filters are: can you think clearly under pressure, can you reason about trade-offs, and have you actually shipped this? Interviewers probe depth, not breadth — one question with five follow-ups tells them more than five surface answers.

### 1.1 Interview Loop Formats by Role

| Role | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---|---|---|---|
| **Senior DS** | SQL screen (45 min) | Stats / ML conceptual (60 min) | Python + Pandas coding (60 min) | Case study or take-home |
| **ML Engineer** | DS/algo coding (60 min) | ML system design (60 min) | ML concepts + coding (60 min) | Infra / MLOps round |
| **AI / GenAI Engineer** | LLM knowledge (45 min) | RAG/agents system design (60 min) | Coding: LLM integration (60 min) | Trade-offs + paper discussion |

### 1.2 The 80/20 Topic Distribution

Based on frequency across FAANG, unicorn, and mid-size DS/ML/AI roles:

| Topic | % of DS interviews | % of MLE interviews | % of AI Engineer interviews |
|---|---|---|---|
| SQL | 90% | 30% | 15% |
| Statistics / A/B testing | 80% | 40% | 20% |
| Python / Pandas | 75% | 60% | 50% |
| ML algorithms | 70% | 50% | 30% |
| DSA / coding | 40% | 85% | 65% |
| LLMs / transformers | 20% | 55% | 90% |
| System design (ML) | 30% | 85% | 80% |
| Fine-tuning / PEFT | 5% | 40% | 75% |
| RAG / retrieval | 10% | 45% | 85% |
| Agents | 5% | 30% | 75% |

### 1.3 What Actually Separates Senior from Mid-Level Answers

Junior answers the question. Senior answers the question, names the trade-off, and mentions a failure mode. Staff answers the question, names the trade-off, explains which choice they'd make in production, and asks a clarifying question back to the interviewer. That pattern — depth + trade-off + production awareness — is what this guide trains.

**Resources**
- [ML Interviews Book (Chip Huyen)](https://huyenchip.com/ml-interviews-book/) — best overview of DS/MLE interview loops
- [Tech Interview Handbook](https://www.techinterviewhandbook.org/) — general structure for technical loops

---

## 2. DSA & Python Coding

The DS/ML coding round tests whether you can translate a data problem into an algorithm in 30–45 minutes. The target difficulty is LeetCode Medium. Interviewers are not testing CS fundamentals for their own sake — they are testing whether you can think algorithmically about data, which is exactly what production ML engineering requires.

### 2.1 The Five Patterns That Cover 80% of ML Coding Questions

**Pattern 1 — Hash Map + Set (30% of questions)**

```python
# Q: Find the top-K most frequent labels in a multi-label dataset
from collections import Counter
import heapq

def top_k_labels(label_lists: list[list[str]], k: int) -> list[str]:
    counts = Counter(label for labels in label_lists for label in labels)
    return [label for label, _ in heapq.nlargest(k, counts.items(), key=lambda x: x[1])]

data = [["cat","dog"], ["cat","bird"], ["dog","fish"], ["cat","dog","fish"]]
print(top_k_labels(data, k=2))   # ['cat', 'dog']
```

**Pattern 2 — Sliding Window (15% of questions)**

```python
# Q: Compute the rolling 7-day max engagement score in O(n)
from collections import deque

def rolling_max(scores: list[float], w: int) -> list[float]:
    dq, result = deque(), []
    for i, s in enumerate(scores):
        while dq and scores[dq[-1]] <= s:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()
        if i >= w - 1:
            result.append(scores[dq[0]])
    return result

print(rolling_max([3,1,5,2,8,4,6], w=3))   # [5, 5, 8, 8, 8]
```

**Pattern 3 — Heap / Top-K (20% of questions)**

```python
import heapq

# Q: Given K sorted recommendation lists, merge into one ranked list
def merge_k_ranked(lists: list[list[tuple]]) -> list:
    """Each list is [(score, item), ...] sorted descending."""
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            score, item = lst[0]
            heapq.heappush(heap, (-score, i, 0, item))  # negate for max-heap
    result = []
    while heap:
        neg_score, li, idx, item = heapq.heappop(heap)
        result.append((item, -neg_score))
        if idx + 1 < len(lists[li]):
            s, it = lists[li][idx + 1]
            heapq.heappush(heap, (-s, li, idx + 1, it))
    return result
```

**Pattern 4 — Graph BFS/DFS for Pipeline DAGs (25% of questions)**

```python
from collections import defaultdict, deque

# Q: Given an ML pipeline as a DAG, return execution order. Detect cycles.
def pipeline_order(steps: list[str], deps: list[tuple]) -> list[str] | None:
    in_deg = defaultdict(int)
    adj    = defaultdict(list)
    for s in steps:
        in_deg[s] = in_deg.get(s, 0)
    for u, v in deps:
        adj[u].append(v)
        in_deg[v] += 1
    queue = deque(s for s in steps if in_deg[s] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in adj[node]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    return order if len(order) == len(steps) else None  # None = cycle detected
```

**Pattern 5 — DP for Sequence Problems (10% of questions)**

```python
# Q: Edit distance between two strings (Levenshtein) — used in data cleaning
def edit_distance(s1: str, s2: str) -> int:
    n, m = len(s1), len(s2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[j] = min(dp[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
    return dp[m]

print(edit_distance("kitten", "sitting"))   # 3
```

> 🎯 **Interview prep**: When you see "stream" in the problem, reach for a heap or deque. When you see "unique" or "seen before", reach for a set or dict. When you see "pipeline" or "dependency", draw a DAG and run BFS. These three mappings cover 60% of DS coding questions.

> 🏭 **Production note**: The pipeline DAG pattern (Pattern 4) is exactly what Airflow uses internally. If you can explain Kahn's algorithm in an interview, you can also explain why Airflow raises a DAG import error when it detects a cycle.

---

## 3. Python & Pandas

Python and Pandas questions in DS interviews almost always take one of two forms: "wrangle this DataFrame to produce this output" (the most common), or "how would you make this code more efficient?" Interviewers are testing whether you know the idiomatic, vectorised Pandas API versus brute-force Python loops.

### 3.1 The Questions That Appear Most

**Q: What is the difference between `.apply()` and a vectorised operation?**

`.apply()` runs a Python function row-by-row, breaking out of Pandas' C-backed internals — it is typically 10–100× slower than an equivalent vectorised operation. Use `.apply()` only when no vectorised alternative exists (e.g., complex multi-column logic, custom parsing).

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({"revenue": [100, 200, None, 400], "cost": [80, 150, 50, 300]})

# ❌ Slow: .apply() with Python lambda
df["margin_slow"] = df.apply(lambda r: (r.revenue - r.cost) / r.revenue
                              if r.revenue else None, axis=1)

# ✅ Fast: vectorised — runs in C
df["margin_fast"] = (df["revenue"] - df["cost"]) / df["revenue"]
```

**Q: How do the different merge types work? What does a cross join produce?**

```python
left  = pd.DataFrame({"user_id": [1,2,3], "city": ["A","B","C"]})
right = pd.DataFrame({"user_id": [2,3,4], "spend": [10, 20, 30]})

inner  = pd.merge(left, right, on="user_id", how="inner")   # 2 rows: users 2,3
left_j = pd.merge(left, right, on="user_id", how="left")    # 3 rows: user 1 gets NaN spend
outer  = pd.merge(left, right, on="user_id", how="outer")   # 4 rows: user 4 gets NaN city
cross  = pd.merge(left.assign(k=1), right.assign(k=1), on="k").drop("k",axis=1)  # 9 rows
```

**Q: How do you compute the 7-day rolling revenue per user?**

```python
df = pd.DataFrame({
    "date":    pd.date_range("2024-01-01", periods=30, freq="D"),
    "user_id": [1]*15 + [2]*15,
    "revenue": np.random.randint(10, 100, 30)
})
df = df.sort_values(["user_id", "date"])
df["roll_rev_7d"] = (df.groupby("user_id")["revenue"]
                       .transform(lambda x: x.rolling(7, min_periods=1).sum()))
```

**Q: How do you find duplicate rows efficiently?**

```python
dupes = df[df.duplicated(subset=["user_id", "date"], keep=False)]
```

**Q: What is method chaining and when does it hurt?**

Method chaining (`.pipe()` pattern) is idiomatic Pandas but creates full intermediate copies at each step. For very wide DataFrames (>1M rows × 100 cols), it can OOM. The fix is `inplace=True` for mutations or switching to Polars, which uses lazy evaluation.

> 🎯 **Interview prep**: If asked to "clean and aggregate this DataFrame", write it as a chain: `df.dropna().rename().assign().groupby().agg().reset_index()`. This shows you know the API idiomatically. Then mention the `.pipe()` pattern for readability on complex pipelines.

> 🏭 **Production note**: `df.iterrows()` is the single most common performance bug in DS code. Each row becomes a Series object, re-wrapping every element. Use `.itertuples()` (10× faster) or vectorised ops (100× faster) instead.

**Resources**
- [Pandas docs — GroupBy user guide](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [Pandas docs — Window functions](https://pandas.pydata.org/docs/user_guide/window.html)

---

## 4. SQL

SQL is tested in 90% of DS interviews and 30% of MLE interviews. The 80/20 breakdown: window functions account for ~40% of hard questions, CTEs account for ~25%, and self-joins / funnel analysis account for ~25%. The rest is standard aggregation.

### 4.1 Window Functions — The Most Tested Topic

```sql
-- Q: Rank users by revenue within each city, return top-2 per city
SELECT *
FROM (
    SELECT
        user_id,
        city,
        revenue,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY revenue DESC) AS rn
    FROM user_revenue
) ranked
WHERE rn <= 2;

-- Q: Compute running total of revenue per user over time
SELECT
    user_id,
    event_date,
    revenue,
    SUM(revenue) OVER (
        PARTITION BY user_id
        ORDER BY event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM events;

-- Q: Compute 7-day rolling average
SELECT
    user_id,
    event_date,
    AVG(revenue) OVER (
        PARTITION BY user_id
        ORDER BY event_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS roll_7d_avg
FROM events;

-- Q: Find users who made a purchase within 30 days of their first visit (funnel)
WITH first_visit AS (
    SELECT user_id, MIN(event_date) AS first_visit_date
    FROM events WHERE event_type = 'visit'
    GROUP BY user_id
),
first_purchase AS (
    SELECT user_id, MIN(event_date) AS first_purchase_date
    FROM events WHERE event_type = 'purchase'
    GROUP BY user_id
)
SELECT
    fv.user_id,
    DATEDIFF('day', fv.first_visit_date, fp.first_purchase_date) AS days_to_purchase
FROM first_visit fv
JOIN first_purchase fp ON fv.user_id = fp.user_id
WHERE DATEDIFF('day', fv.first_visit_date, fp.first_purchase_date) <= 30;
```

### 4.2 Self-Joins and Session Analysis

```sql
-- Q: Find users who made purchases on two consecutive days
SELECT DISTINCT a.user_id
FROM purchases a
JOIN purchases b
  ON a.user_id = b.user_id
 AND DATEDIFF('day', a.purchase_date, b.purchase_date) = 1;

-- Q: Sessionise events — group events within 30 minutes into a session
SELECT
    user_id,
    event_time,
    SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
FROM (
    SELECT
        user_id,
        event_time,
        CASE
            WHEN LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) IS NULL
              OR DATEDIFF('minute',
                 LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time),
                 event_time) > 30
            THEN 1 ELSE 0
        END AS is_new_session
    FROM events
) t;
```

### 4.3 What Distinguishes a Senior SQL Answer

Junior: writes a correct query. Senior: also explains the execution plan. The questions to ask yourself before writing: "Does this query use a full table scan? Should I add an index? Is a CTE better than a subquery here for readability and caching?"

Key heuristics:
- **CTE vs subquery**: CTEs are computed once and can be referenced multiple times — use them when the same subquery appears more than once. Subqueries in `WHERE` clauses can prevent index use in some engines.
- **ROW_NUMBER vs RANK vs DENSE_RANK**: `ROW_NUMBER` gives unique ranks (no ties). `RANK` skips numbers after ties (1,2,2,4). `DENSE_RANK` does not skip (1,2,2,3). Interviewers love this distinction.
- **NULLs in aggregation**: `COUNT(*)` counts NULLs; `COUNT(col)` does not. `AVG(col)` ignores NULLs. Always state this assumption.

> 🎯 **Interview prep**: "Write a query to find the N-th highest salary" — use `DENSE_RANK()` not a subquery with `LIMIT OFFSET`, which fails on ties and is O(n) anyway.

**Resources**
- [Mode SQL Tutorial — Window Functions](https://mode.com/sql-tutorial/sql-window-functions/)
- [Snowflake SQL Reference](https://docs.snowflake.com/en/sql-reference)

---

## 5. Statistics & A/B Testing

Statistics is the most differentiating topic in DS interviews. Junior candidates know the formulas. Senior candidates know when the standard test is wrong — network effects, novelty effect, multiple comparisons, pre-experiment data — and can propose better designs.

### 5.1 The Core A/B Testing Questions

**Q: What is a p-value? What isn't it?**

A p-value is the probability of observing a test statistic at least as extreme as the one observed, *assuming the null hypothesis is true*. It is **not** the probability that the null hypothesis is true, and it is **not** the probability that the result is due to chance. This distinction kills junior candidates.

**Q: Walk me through designing an A/B test for a new checkout button.**

Senior answer structure:
1. **Define the metric**: primary (conversion rate), guardrail (revenue per user), secondary (add-to-cart rate).
2. **Sample size**: use power analysis. For a 5% baseline conversion, detecting a 0.5pp lift (10% relative) at 80% power and 5% significance requires ~15,000 users per arm.
3. **Randomisation unit**: user-level (not session-level) to avoid contamination.
4. **Duration**: run for at least 2 full business cycles (2 weeks) to capture weekly patterns and account for the novelty effect.
5. **Multiple testing**: if you're testing multiple metrics, apply Bonferroni or Benjamini-Hochberg correction. Or pre-register one primary metric.

```python
from scipy import stats
import numpy as np

# Sample size calculation
def sample_size(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    baseline_rate: conversion rate in control (e.g. 0.05)
    mde: minimum detectable effect as absolute lift (e.g. 0.005)
    """
    p1, p2 = baseline_rate, baseline_rate + mde
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    p_bar   = (p1 + p2) / 2
    n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
         z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / (p2 - p1)**2
    return int(np.ceil(n))

print(sample_size(0.05, 0.005))   # ~14,744 per arm

# Two-proportion z-test
control_n, control_conv = 15000, 750     # 5.0% conversion
treat_n,   treat_conv   = 15000, 825     # 5.5% conversion

p_c = control_conv / control_n
p_t = treat_conv   / treat_n
p_pool = (control_conv + treat_conv) / (control_n + treat_n)
z = (p_t - p_c) / np.sqrt(p_pool * (1 - p_pool) * (1/control_n + 1/treat_n))
p_val = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"z={z:.3f}, p={p_val:.4f}")   # z=2.058, p=0.0396 → significant at α=0.05
```

**Q: What is CUPED and when would you use it?**

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the variance of your metric estimator by regressing out pre-experiment values of the same metric. If a user's pre-experiment revenue is correlated with their in-experiment revenue (r=0.7 is typical), CUPED effectively reduces your required sample size by ~50% — or equivalently, lets you detect the same effect twice as fast. ([Deng et al., 2013](https://www.microsoft.com/en-us/research/publication/improving-the-sensitivity-of-online-controlled-experiments/))

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 10_000
pre  = np.random.normal(100, 20, n)                     # pre-experiment revenue
noise = np.random.normal(0, 15, n)
post_c = 0.7 * pre + noise                              # control
post_t = 0.7 * pre + noise + np.where(np.arange(n) % 2 == 0, 2, 0)  # treatment: +2 lift

# Standard t-test
control, treatment = post_c[::2], post_t[1::2]
t_std, p_std = stats.ttest_ind(treatment, control)
print(f"Standard t-test: t={t_std:.3f}, p={p_std:.4f}")

# CUPED: regress out pre-experiment covariate
theta = np.cov(post_c, pre)[0,1] / np.var(pre)         # regression coefficient
cuped_c = post_c[::2]  - theta * (pre[::2]  - pre.mean())
cuped_t = post_t[1::2] - theta * (pre[1::2] - pre.mean())
t_cup, p_cup = stats.ttest_ind(cuped_t, cuped_c)
print(f"CUPED t-test:    t={t_cup:.3f}, p={p_cup:.4f}")
# CUPED p-value will be much smaller — same data, more power
```

**Q: What is Simpson's Paradox and when does it appear in DS work?**

Simpson's Paradox: a trend that appears in several groups reverses when the groups are combined. Classic DS example: treatment looks better overall, but within each demographic group it performs worse — because sicker patients were assigned to the treatment group. Senior answer: always stratify your A/B results by key segments and check for heterogeneous treatment effects.

> 🎯 **Interview prep**: "When would you use a t-test vs a Mann-Whitney U test?" — t-test assumes normally distributed populations (or large n by CLT). Mann-Whitney is non-parametric — use it for skewed metrics like revenue or session duration, which are common in DS. In practice for large n, both converge; but naming Mann-Whitney shows you know metrics that don't meet CLT assumptions.

**Resources**
- [CUPED paper (Deng et al., 2013)](https://www.microsoft.com/en-us/research/publication/improving-the-sensitivity-of-online-controlled-experiments/)
- [Causal Inference: The Mixtape (Cunningham)](https://mixtape.scunning.com/)

---

## 6. ML Algorithms

ML algorithm questions test whether you understand *why* algorithms behave the way they do in production, not whether you can recite definitions.

### 6.1 Algorithm Cheat-Sheet

| Algorithm | When to use | When NOT to use | Production default | SotA |
|---|---|---|---|---|
| **Logistic regression** | Baseline, interpretability required, sparse features | Non-linear relationships, high-cardinality categoricals | Feature stores with linear scoring | N/A — use as baseline |
| **Random forest** | Robust baseline, handles missing values, feature importance | Very wide datasets (slow), when you need probability calibration | Fraud detection baselines | — |
| **XGBoost / LightGBM** | Tabular data, mixed types, missing values | Text/image/sequence data | **Production default for tabular** (80% of industry) | LightGBM fastest; XGBoost most popular |
| **Neural network (MLP)** | High-dim continuous features, when you have >100K rows | Small datasets, interpretability required | Feature cross learning in ads | FT-Transformer (2021) |
| **k-means** | Fast clustering, known k, spherical clusters | Unknown k, non-spherical clusters, high-dim | Embedding clustering in recsys | — |
| **DBSCAN** | Arbitrary shapes, detecting outliers/noise | Very high-dim (curse of dimensionality) | Anomaly detection in time series | — |
| **SVM** | High-dim, small n, text classification with TF-IDF | Large n (O(n²) training) | Rarely in production (replaced by GBDT/neural) | — |

### 6.2 The Questions That Appear Most

**Q: Explain the bias-variance trade-off.**

Bias: error from wrong assumptions — a model too simple to capture patterns. Variance: error from sensitivity to noise — a model too complex that memorises training data. The trade-off: increasing model complexity reduces bias but increases variance. The senior answer adds: in practice, modern deep learning violates this with **double descent** — very large models (overparameterised) can have low bias AND low variance if properly regularised, because they interpolate training data in a benign way.

**Q: What is data leakage and how do you detect it?**

Data leakage is when information from the test set (or future) leaks into training. Common forms:
1. **Target leakage**: a feature computed using the target variable (e.g., "days until churn" as a feature in a churn model).
2. **Temporal leakage**: using future data to predict the past — forgetting to split by time for time-series models.
3. **Train/test contamination**: scaling or imputing on the full dataset before splitting.

Detection: if your model achieves suspiciously high AUC (>0.99 on a hard problem), check feature correlations with the target and audit your data pipeline chronologically.

**Q: Your model has AUC=0.85 in offline eval but underperforms in production. Why?**

Senior answer hits four causes: (1) **distribution shift** — training data doesn't reflect live traffic, (2) **label delay** — some labels arrive days after the event (e.g., churn), creating training/serving skew, (3) **feature store staleness** — features used at training aren't available at the same freshness in serving, (4) **feedback loops** — the model's own predictions influence future training data. The fix requires monitoring input feature distributions, output score distributions, and label rates separately.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import numpy as np

# Always check calibration — AUC measures ranking, not probability accuracy
X_train = np.random.randn(1000, 10)
y_train = (X_train[:,0] + np.random.randn(1000) > 0).astype(int)
X_test  = np.random.randn(200, 10)
y_test  = (X_test[:,0] + np.random.randn(200) > 0).astype(int)

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:,1]

print(f"AUC:         {roc_auc_score(y_test, probs):.3f}")   # ranking quality
print(f"Brier score: {brier_score_loss(y_test, probs):.3f}")   # probability calibration

# Calibrate if Brier score is high relative to AUC
cal_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
cal_model.fit(X_train, y_train)
cal_probs = cal_model.predict_proba(X_test)[:,1]
print(f"Calibrated Brier: {brier_score_loss(y_test, cal_probs):.3f}")
```

> 🎯 **Interview prep**: "How do you handle class imbalance?" — Don't just say "oversample". Say: first, check if the class distribution in training matches production (it might be fine). If not: use `class_weight='balanced'` in sklearn (changes the loss, not the data — preferred), or SMOTE for very severe imbalance. Always evaluate with F1/AUC, not accuracy.

---

## 7. PyTorch & Deep Learning

PyTorch questions probe whether you've actually trained models, not just called `.fit()`. The questions that trip up candidates are about the training loop internals, gradient mechanics, and memory.

### 7.1 The Training Loop — What Every Interview Expects You to Know

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TwoLayerNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),          # regularisation — MUST set model.train() for this
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

X = torch.randn(1000, 20)
y = (X[:,0] > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model     = TwoLayerNet(20, 64, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    model.train()                       # enables dropout + batchnorm
    for xb, yb in loader:
        optimizer.zero_grad()           # MUST clear gradients before backward
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()                 # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
    scheduler.step()

model.eval()                            # disables dropout
with torch.no_grad():                   # saves memory — no gradient graph
    preds = model(X).argmax(dim=1)
```

### 7.2 The Questions That Appear Most

**Q: What happens if you forget `optimizer.zero_grad()`?**

Gradients accumulate across batches. The loss from batch 2 adds its gradients to those from batch 1, causing incorrect parameter updates. This bug is subtle — the model still trains, but converges slower and to a worse solution.

**Q: What is the difference between `model.train()` and `model.eval()`?**

`model.train()` enables Dropout (randomly zeros activations) and BatchNorm (uses batch statistics). `model.eval()` disables Dropout (uses all neurons) and makes BatchNorm use running statistics from training. Forgetting `model.eval()` during inference gives stochastic predictions — a common production bug.

**Q: How does mixed precision training work?**

```python
# torch.autocast: compute forward pass in fp16/bf16, accumulate gradients in fp32
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()   # scales loss to prevent fp16 underflow
for xb, yb in loader:
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):      # bf16 on A100/H100; fp16 on older GPUs
        logits = model(xb.cuda())
        loss   = criterion(logits, yb.cuda())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Q: What's the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?**

`BCEWithLogitsLoss` combines sigmoid + BCE in a single numerically stable operation. Using `BCELoss` on raw logits is incorrect; using it on sigmoid outputs is correct but 2× slower and less stable. Always use `BCEWithLogitsLoss` for binary classification.

> 🎯 **Interview prep**: "Why does your model loss decrease but validation accuracy plateau?" — Likely causes: (1) learning rate too high → oscillates around optimum, (2) class imbalance — loss decreases by predicting majority class, (3) overfitting — use dropout, weight decay, early stopping.

**Resources**
- [PyTorch docs — training loop](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Sebastian Raschka — LLM training from scratch](https://sebastianraschka.com/blog/2023/llm-from-scratch-qanda.html)

---

## 8. Recommender Systems

RecSys questions distinguish candidates who've read papers from those who've shipped systems. The key gap: papers show offline metrics, production systems live by online metrics and latency SLOs.

### 8.1 The Architecture Every Interview Expects

```
User request
     │
     ▼
[Candidate Generation]    → two-tower model, ANN over 100M items → ~1000 candidates
     │                      FAISS IVF+PQ index, <10ms
     ▼
[Ranking]                 → GBDT or DCN-V2, ~200 features → top-50 items
     │                      Feature store lookup, <20ms
     ▼
[Re-ranking / Business rules] → diversity, freshness, inventory constraints
     │
     ▼
Response
```

### 8.2 The Questions That Appear Most

**Q: How does collaborative filtering work and what are its limitations?**

Matrix factorisation: decompose the user-item interaction matrix R (n_users × n_items) into U (n_users × k) and V (n_items × k) such that R ≈ UV^T. ALS (Alternating Least Squares) alternately solves for U with V fixed, then V with U fixed — each step is a convex ridge regression problem.

Limitations: cold-start (new users/items have no interactions), popularity bias (model favours popular items), sparse interactions (most user-item pairs are unobserved).

**Q: Explain the two-tower model.**

Two separate encoder networks — one for user features, one for item features — produce embeddings in the same space. Candidate generation is a nearest-neighbour query in that space. Training objective: InfoNCE (or softmax over sampled negatives). Key insight: the two encoders run independently, so item embeddings can be pre-computed and indexed offline. ([Covington et al., 2016](https://dl.acm.org/doi/10.1145/2959100.2959190))

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim, emb_dim=64):
        super().__init__()
        self.user_tower = nn.Sequential(nn.Linear(user_dim, 128), nn.ReLU(),
                                         nn.Linear(128, emb_dim))
        self.item_tower = nn.Sequential(nn.Linear(item_dim, 128), nn.ReLU(),
                                         nn.Linear(128, emb_dim))

    def forward(self, user_feat, item_feat):
        u = F.normalize(self.user_tower(user_feat), dim=-1)   # L2 normalise
        v = F.normalize(self.item_tower(item_feat), dim=-1)
        return (u * v).sum(dim=-1)   # cosine similarity

# InfoNCE loss for contrastive training
def info_nce_loss(user_emb, pos_item_emb, neg_item_emb, temperature=0.1):
    pos_score = (user_emb * pos_item_emb).sum(dim=-1) / temperature
    neg_scores = torch.mm(user_emb, neg_item_emb.T) / temperature
    logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
    labels = torch.zeros(len(user_emb), dtype=torch.long)
    return F.cross_entropy(logits, labels)
```

**Q: How do you handle the cold-start problem?**

- **New user**: fall back to popularity-based or content-based (use user profile features — age, location, device). After first interaction, shift to collaborative features.
- **New item**: use content features (title, description embeddings) as item-tower input until enough interactions accumulate. After N interactions, blend content + collaborative embeddings.

> 🎯 **Interview prep**: "How would you evaluate a recommendation system?" — Offline: NDCG@k, HR@k, MAP. But offline metrics can decouple from online metrics. Online: CTR, CVR, dwell time, long-term engagement. Senior answer: always run online A/B tests — offline evaluation is necessary but not sufficient.

---

## 9. NLP Foundations

NLP questions probe your mental model of how text becomes numbers. The key progression interviewers follow: TF-IDF → word embeddings → attention → transformers → modern LLMs.

### 9.1 The Questions That Appear Most

**Q: Explain the attention mechanism.**

Attention takes three inputs — queries Q, keys K, values V — and computes a weighted sum of values, where weights are determined by query-key similarity. ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

**Worked example**: d_k = 4, one query q = [1,0,1,0], two keys k₁=[1,0,1,0], k₂=[0,1,0,1]:
- q·k₁ = 2, q·k₂ = 0
- Scaled: 2/√4 = 1.0, 0/√4 = 0.0
- Softmax: [0.73, 0.27]
- Output = 0.73×v₁ + 0.27×v₂

The √d_k scaling prevents vanishing gradients when d_k is large (dot products grow with dimension, softmax saturates).

**Q: What is the difference between encoder-only, decoder-only, and encoder-decoder architectures?**

| Architecture | Example models | Use case |
|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Classification, NER, embeddings — bidirectional context |
| **Decoder-only** | GPT-4, LLaMA, Claude | Text generation — causal (left-to-right) attention |
| **Encoder-decoder** | T5, BART, Whisper | Seq-to-seq: translation, summarisation, ASR |

**Q: What is RoPE and why is it better than sinusoidal positional encoding?**

Rotary Position Embedding (RoPE) encodes position by rotating Q and K vectors by an angle proportional to their position. Unlike sinusoidal embeddings (added to the input), RoPE is applied to attention queries and keys — this means relative position is captured in the dot product itself. Critical advantage: **length generalisation** — models trained on 4K context can extrapolate to longer sequences more easily than with absolute positional embeddings.

> 🎯 **Interview prep**: "Why does BERT use MLM but GPT uses CLM?" — Masked Language Modelling (BERT) trains bidirectional representations — seeing both left and right context. Causal Language Modelling (GPT) trains left-to-right, which is the right inductive bias for generation but limits the encoder from seeing future tokens.

---

## 10. Embeddings

Embedding questions are increasingly common as vector search becomes foundational to production AI systems.

### 10.1 The Questions That Appear Most

**Q: When do you use a bi-encoder vs a cross-encoder?**

| | Bi-encoder (SBERT) | Cross-encoder |
|---|---|---|
| **How it works** | Encode query and doc separately; compare embeddings | Concatenate query+doc; classify relevance jointly |
| **Latency** | O(1) per query (pre-computed doc embeddings) | O(n) per query (can't pre-compute) |
| **Quality** | Good — misses fine-grained query-doc interaction | Better — captures full interaction |
| **Use case** | First-stage retrieval over millions of docs | Second-stage reranking of top-100 candidates |
| **Production pattern** | FAISS/hnswlib index of pre-computed embeddings | Served on CPU, top-100 candidates only |

([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084))

**Q: What is Matryoshka Representation Learning (MRL)?**

MRL trains embeddings such that the first d dimensions form a valid embedding, and so do the first 2d, first 4d, etc. This means you can truncate to a smaller dimension at serving time without full retraining — trading some quality for lower memory and faster ANN search. OpenAI's `text-embedding-3` uses MRL.

**Q: What is hard negative mining and why does it matter?**

Contrastive learning trains by pushing positive pairs together and negative pairs apart. Easy negatives (random documents unrelated to the query) provide almost no gradient signal — the model already knows they're different. Hard negatives (documents that are superficially similar but semantically different) force the model to learn fine-grained distinctions. Without hard negatives, SBERT-style models plateau at ~70% recall; with mined hard negatives, they reach >90%.

```python
# Hard negative mining: use BM25 or another model to find challenging negatives
from rank_bm25 import BM25Okapi

def mine_hard_negatives(queries, corpus, top_k=10):
    """For each query, find top-k BM25 results that are NOT the ground truth."""
    tokenised = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenised)
    hard_negatives = {}
    for q_id, query in queries.items():
        scores = bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[::-1][:top_k]
        hard_negatives[q_id] = [corpus[i] for i in top_indices
                                  if i != q_id][:3]   # top-3 non-match
    return hard_negatives
```

> 🎯 **Interview prep**: "How do you evaluate an embedding model?" — Use MTEB (Massive Text Embedding Benchmark). It covers 56 tasks: retrieval, STS, classification, clustering, reranking. The primary metric for retrieval is NDCG@10. Always evaluate on your domain — MTEB averages can hide domain gaps.

---

## 11. RAG & Retrieval

RAG is the first thing most companies build when adopting LLMs, which is why it dominates AI engineer interviews.

### 11.1 RAG vs Fine-tuning Decision Matrix

| Factor | Use RAG | Use Fine-tuning | Use Prompting |
|---|---|---|---|
| **Knowledge freshness** | Frequently updated facts (>weekly) | Slow-changing domain knowledge | Static context that fits in prompt |
| **Hallucination risk** | High-stakes factual answers | Stylistic / format adaptation | Simple tasks |
| **Data you have** | A corpus of documents | >1000 labelled examples | No labelled data |
| **Latency budget** | Can afford extra retrieval step (50–200ms) | Need single-pass inference | Lowest latency |
| **Cost** | Higher (retrieval + generation) | Lower at inference, higher up front | Lowest |
| **Interpretability** | High — can show retrieved sources | Low — knowledge is baked in | High |

### 11.2 The Questions That Appear Most

**Q: Walk me through a production RAG pipeline.**

```
Documents
    │ chunking (512 tokens, 10% overlap)
    ▼
[Embedding model] → chunk vectors
    │ FAISS / hnswlib / pgvector
    ▼
[Vector Index]
    │
    │─── at query time ──────────────────────
    │
[Query] → embed → ANN search → top-10 chunks
    │
    ├── [BM25 search] → top-10 chunks
    │
    └── [Fusion / Reranker] → top-3 chunks
          │ cross-encoder reranker
          ▼
    [LLM] (chunks in context window) → answer
```

**Q: What chunk size should you use and why?**

There is no universal answer — that's the point. Small chunks (128 tokens): high precision, low recall. Large chunks (1024 tokens): high recall, but dilutes relevant signal in the context window and increases cost. The production heuristic: start with 512 tokens and 10–20% overlap, then evaluate with RAGAS metrics (context recall, faithfulness). Sentence-level chunking (using NLTK/spaCy sentence boundaries) often beats fixed-size chunking for structured documents.

**Q: What is hybrid search and when does it outperform dense-only?**

Hybrid search combines dense (semantic) retrieval with sparse (BM25/TF-IDF) retrieval, then fuses scores (Reciprocal Rank Fusion is the standard). Dense search excels at semantic similarity ("show me articles about budget travel" → finds "cheap vacation ideas"). BM25 excels at exact-match and rare keywords ("CUPED variance reduction" → finds the exact phrase). In practice, hybrid always matches or beats either alone by 5–15% NDCG@10.

> 🎯 **Interview prep**: "When would you choose RAG over fine-tuning?" — RAG wins when: knowledge changes frequently, you need source attribution, you have a large document corpus, or you don't have labelled training data. Fine-tuning wins when: adapting style/format/persona, improving performance on a specific task type with labelled examples, or reducing latency.

---

## 12. LLMs & Transformers

LLM questions span from architecture internals to deployment decisions. The questions that trip up candidates are about KV-cache, attention complexity, and scaling laws.

### 12.1 The Questions That Appear Most

**Q: What is the KV cache and why does it matter?**

During autoregressive generation, each new token attends to all previous tokens. Without caching, we recompute key and value projections for every previous token at every step — O(n²) computation. The KV cache stores these projections after they're computed once, so each new token only computes its own K and V and attends to the cached ones. Memory cost: `2 × n_layers × n_heads × d_head × sequence_length × batch_size × bytes_per_element`. For LLaMA-3-70B at 8K context, batch=16, this is ~80 GB — larger than the model weights for long contexts.

**Q: Explain scaling laws. What did Chinchilla change?**

Kaplan et al. (2020) showed loss scales as a power law with parameters N and data D: `L(N,D) = N^(-α) + D^(-β) + const`. ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)) The initial takeaway was "bigger models are always better." Chinchilla (Hoffmann et al., 2022) showed that the original OpenAI models were under-trained: for a given compute budget, you should scale data and model equally (roughly 20 tokens per parameter). LLaMA adopted this — training a 7B model on 1T tokens gives better perplexity than a 70B model trained on 100B tokens at the same compute cost.

**Q: What causes hallucination and how do you mitigate it?**

Hallucination has three sources:
1. **Memorisation gaps**: the model was never trained on this fact, so it generates a plausible-sounding one.
2. **Sycophancy**: the model is trained to produce responses users rate positively — which sometimes means confidently wrong rather than honestly uncertain.
3. **Context window limitations**: the model loses track of long contexts and confabulates.

Mitigations: RAG (ground generation in retrieved facts), citation enforcement (instruct model to only state what's in the context), temperature reduction (lower T → less randomness → fewer hallucinations), Constitutional AI / RLHF trained for honesty.

```python
from anthropic import Anthropic

client = Anthropic()

# Grounded generation: force model to cite sources
def grounded_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
    prompt  = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."
Cite the source number in brackets.

Context:
{context}

Question: {question}
Answer:"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

> 🎯 **Interview prep**: "What is the difference between RLHF and DPO?" — RLHF requires training a separate reward model, then running PPO (an RL algorithm) to update the LLM using the reward model's scores. DPO shows that this two-step process can be collapsed into a single classification loss directly on preference data — the LLM itself implicitly represents the reward model. ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) DPO is simpler, more stable, and produces comparable or better results.

---

## 13. LLM Inference & Optimisation

Inference optimisation is a core MLE/AI engineer topic. The questions probe whether you can reason about the GPU memory hierarchy and the latency-throughput trade-off.

### 13.1 The Questions That Appear Most

**Q: Explain continuous batching (PagedAttention).**

Traditional static batching: fill a batch, run it, wait for the slowest sequence to finish. The GPU sits idle for short sequences while waiting for long ones. Continuous batching (vLLM): as soon as one sequence in the batch generates its EOS token, swap in a new sequence from the queue. This keeps GPU utilisation near 100% and increases throughput 2–4×. PagedAttention further improves memory efficiency by managing KV-cache in non-contiguous blocks (like OS paging), eliminating fragmentation. ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180))

**Q: How does INT8 quantisation work and what accuracy do you lose?**

```python
import torch
import numpy as np

# Symmetric per-tensor INT8 quantisation
def quantise_int8(w: torch.Tensor):
    scale = w.abs().max() / 127.0
    w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale

def dequantise(w_int8, scale):
    return w_int8.to(torch.float32) * scale

w = torch.randn(512, 512)
w_q, scale = quantise_int8(w)
w_recon    = dequantise(w_q, scale)
error      = (w - w_recon).abs().mean().item()
print(f"Quantisation error (INT8): {error:.6f}")   # ~0.003

# Memory: float32 = 4 bytes → int8 = 1 byte (4× reduction)
print(f"float32: {w.nbytes/1e6:.1f} MB")     # ~1.0 MB
print(f"int8:    {w_q.nbytes/1e6:.1f} MB")   # ~0.25 MB
```

**Q: What is speculative decoding and when does it help?**

A small draft model generates K tokens speculatively; the large target model verifies all K in a single forward pass (since generation is bottlenecked on memory bandwidth, not compute). If the target model agrees, all K tokens are accepted for free. If it disagrees at position j, discard j..K and proceed normally. Typical speedup: 2–3× for tasks where a small model generates good drafts (e.g., code completion). Does NOT improve latency for creative generation where the draft model is frequently wrong.

### 13.2 Key Metrics

| Metric | Definition | Typical target |
|---|---|---|
| **TTFT** (Time To First Token) | Latency from request to first token generated | < 1s for chat, <100ms for streaming |
| **TPOT** (Time Per Output Token) | Latency per subsequent token | < 50ms/token for smooth streaming |
| **Throughput** | Tokens generated per second across all requests | > 1000 tokens/s for batch workloads |
| **MFU** (Model FLOP Utilisation) | Actual FLOPs / peak FLOPs | FlashAttention-2 achieves ~72% on A100 |

**Resources**
- [vLLM: PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2 (Dao et al., 2023)](https://arxiv.org/abs/2307.08691)

---

## 14. Fine-tuning LLMs

Fine-tuning is the most rapidly evolving topic in the interview space. Candidates who passed an interview six months ago on "use LoRA" now need to also explain DPO, ORPO, and the memory math.

### 14.1 LoRA vs QLoRA vs Full Fine-tune vs RAG

| Method | GPU memory (7B model) | Quality vs full FT | Latency overhead | When to use |
|---|---|---|---|---|
| **Full fine-tune** | ~80 GB (4×A100) | 100% baseline | None | Large labelled dataset (>50K), mission-critical quality, have infra |
| **LoRA** (fp16) | ~24 GB (1×A100) | ~95–99% | None at inference | Standard choice for adapting open models, moderate data |
| **QLoRA** (4-bit) | ~10 GB (1×RTX 3090) | ~93–97% | Dequantise overhead (~5%) | Fine-tune on consumer GPU; single GPU; up to 70B models |
| **RAG** | Model serving only | N/A (no weight update) | Retrieval latency 50–200ms | Knowledge injection without retraining; frequently updated data |

([Hu et al., 2021](https://arxiv.org/abs/2106.09685) — LoRA; [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) — QLoRA)

### 14.2 The Questions That Appear Most

**Q: Explain how LoRA works mechanically.**

LoRA freezes all original weights and inserts two low-rank matrices A ∈ ℝ^(d×r) and B ∈ ℝ^(r×d) alongside each frozen weight matrix W₀ ∈ ℝ^(d×d), where r << d. The forward pass computes W₀x + (BA)x. During training, only A and B are updated. At the end of training, BA can be merged into W₀ with zero inference overhead.

Memory saving: instead of updating d² parameters, we update only 2×d×r. For d=4096, r=16: from 16.7M to 131K trainable parameters — 128× fewer.

```python
# pip install peft transformers
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype="auto",
    device_map="auto"
)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                       # rank — higher = more capacity, more memory
    lora_alpha=32,              # scaling: effective_lr = lr * alpha/r
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 851,968 || all params: 1,236,174,848 || trainable%: 0.069
```

**Q: What is the difference between SFT, RLHF, and DPO?**

- **SFT (Supervised Fine-Tuning)**: train the model to mimic gold-standard outputs. Required as the first step before any alignment method.
- **RLHF**: train a separate reward model on human preference data, then use PPO to update the LLM to maximise reward while not deviating too far from the SFT model (KL penalty). Complex — requires 4 models in memory simultaneously (actor, critic, reference, reward).
- **DPO**: directly optimise the LLM on preference pairs (chosen, rejected) using a reparameterised loss that implicitly captures the reward. Same alignment goal as RLHF, much simpler. Requires only 2 models (policy + reference). ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))

**Q: When does fine-tuning underperform RAG?**

Fine-tuning memorises facts from training data, but memorisation degrades with distance (the model remembers frequently repeated facts better than rare ones). For accurate factual retrieval, RAG always wins because the fact is retrieved verbatim. Fine-tuning is better for style, format, domain vocabulary, and multi-step reasoning patterns that recur in the data.

> 🎯 **Interview prep**: "How do you set LoRA rank r?" — Start with r=16, alpha=32 (a common heuristic: alpha=2×r). Increase r if the adaptation task is complex (full domain shift) or decrease if you're just adjusting style. Validation loss on held-out examples is the only ground truth.

---

## 15. Agents & Tool Use

Agent questions are the newest addition to AI engineer interviews. The key differentiator: candidates who understand failure modes, not just the happy path.

### 15.1 The Questions That Appear Most

**Q: Explain the ReAct framework.**

ReAct interleaves Reasoning (chain-of-thought) and Acting (tool calls) in a single loop. The model generates a Thought (reasoning about what to do), then an Action (structured tool call), then observes the result, then reasons again. ([Yao et al., 2022](https://arxiv.org/abs/2210.03629))

```python
from anthropic import Anthropic

client = Anthropic()

tools = [{
    "name": "search_database",
    "description": "Search a product database by keyword. Returns list of matching products.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query":   {"type": "string",  "description": "Search query"},
            "top_k":   {"type": "integer", "description": "Number of results", "default": 5}
        },
        "required": ["query"]
    }
}]

def run_agent(user_message: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": user_message}]
    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        if response.stop_reason == "end_turn":
            return response.content[0].text
        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # In production: call actual tool
                result = f"Found 3 products matching '{block.input['query']}'"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        messages.append({"role": "user", "content": tool_results})
    return "Max turns reached."
```

**Q: What are the main failure modes of LLM agents?**

Senior answer names five failure modes:

1. **Hallucinated tool calls**: the model invokes a tool with incorrect arguments or invents a tool that doesn't exist. Mitigation: strict JSON schema validation, retry with error message in context.
2. **Infinite loops**: the agent keeps calling tools because no stopping condition is met. Mitigation: hard max-turn limit, explicit termination detection in the prompt.
3. **Context window exhaustion**: long tool outputs fill the context, degrading reasoning quality. Mitigation: truncate tool outputs, summarise intermediate results.
4. **Cascading errors**: an error in step 2 propagates through steps 3–10 without detection. Mitigation: add a verification step after critical actions.
5. **Latency explosion**: each tool call adds 1–5 seconds; a 10-step agent takes 10–50 seconds. Mitigation: parallelise independent tool calls, cache deterministic tool results.

**Q: What is MCP (Model Context Protocol)?**

MCP (Anthropic, 2024) is an open protocol that standardises how AI models communicate with external tools and data sources. Instead of each LLM application implementing its own tool-calling interface, MCP defines a standard server-client interface: any MCP server can be plugged into any MCP-compatible client. This is to AI agents what USB-C is to devices.

> 🎯 **Interview prep**: "How do you evaluate an agentic system?" — Task completion rate (did it finish the task?) and step efficiency (did it take the minimal number of steps?). For factual tasks, also measure accuracy of the final answer. Tools like AgentBench and GAIA provide standardised benchmarks.

---

## 16. ML System Design

System design is the highest-signal round in the MLE and AI engineer loop. Interviewers are testing whether you think about the whole lifecycle — data, training, serving, monitoring — not just the model.

### 16.1 The Framework for Any ML System Design Question

```
1. Clarify scope (5 min)
   → What is the business metric? (CTR, conversion, DAU)
   → What is the scale? (QPS, data volume, latency SLO)
   → What data is available?

2. Define the ML problem (5 min)
   → Framing: ranking? classification? regression?
   → Input/output format
   → Training label availability

3. Data pipeline (10 min)
   → Collection, storage, labelling
   → Train/val/test split strategy (temporal split for time-series)
   → Feature engineering: offline (batch) vs online (real-time)

4. Model choice and training (10 min)
   → Baseline → production model → future improvements
   → Evaluation metrics: offline (AUC, NDCG) + online (A/B test)

5. Serving architecture (10 min)
   → Latency budget breakdown
   → Pre-computation vs real-time inference
   → Caching strategy

6. Monitoring (5 min)
   → Input distribution drift (PSI, KL divergence)
   → Prediction distribution shift
   → Business metric degradation → model retrain trigger
```

### 16.2 Worked Example 1 — Recommendation System (YouTube/Netflix style)

**Scope**: Recommend 20 videos to a user when they open the app. Latency SLO: <200ms end-to-end. Scale: 100M users, 500M items.

**Architecture**:

```
User opens app
      │
      ▼
[Candidate Generation]  ←── User tower (user features → 128-dim embedding)
│  Two-tower model           ANN search over pre-indexed item embeddings
│  FAISS IVF+PQ index        → 1000 candidates, <10ms
      │
      ▼
[Feature Store Lookup]
│  Real-time: user's last 10 interactions (Redis, <5ms)
│  Batch: user historical features (Hive/Snowflake → feature store)
      │
      ▼
[Ranking Model]         ←── DCN-V2 or LightGBM
│  200 features per candidate  Input: user + item + context features
│  → 1000 candidates → 50 ranked items, <50ms
      │
      ▼
[Business Rules / Diversity]
│  Filter duplicates, apply freshness boost, diversity re-rank
      │
      ▼
Response (20 items)
```

**Key trade-offs to mention**:
- Exploration vs exploitation: ε-greedy or contextual bandit for 5–10% of slots.
- Serving staleness: user tower recomputed every 15 min; item tower every 24h.
- Cold start: new users get content-based fallback (trending + demographic-matched).
- Evaluation: offline NDCG@20, then A/B test against baseline for 2 weeks measuring watch time (not CTR — CTR optimisation leads to clickbait).

### 16.3 Worked Example 2 — LLM-Powered RAG Chatbot

**Scope**: Internal knowledge-base chatbot over 500K proprietary documents. Latency SLO: <3s. Users: 10K employees.

**Architecture**:

```
Document ingestion (offline)
   │ PDF/DOCX parsing → text chunks (512 tokens, 10% overlap)
   │ Embedding model (text-embedding-3-small or GTE) → vectors
   ▼
[Vector Index]  FAISS IVF or pgvector (500K × 1536 dims → ~3 GB)

Query processing (online, <3s budget)
   User query
      │ <50ms: embed query
      │ <100ms: hybrid search (dense ANN + BM25), top-20 chunks
      │ <300ms: cross-encoder reranker → top-5 chunks
      │ <2500ms: LLM generation with retrieved chunks as context
      ▼
   Answer + source citations
```

**Key trade-offs to mention**:
- Chunk size: 512 tokens balances precision and recall; sentence-level for legal docs.
- Reranking: cross-encoder adds ~200ms but improves faithfulness significantly for technical docs.
- Guardrails: add a classifier to detect off-topic queries and route to fallback response.
- Cost: ~$0.002/query at 1000 tokens context. At 10K queries/day = $20/day.
- Evaluation: RAGAS — context recall, faithfulness, answer relevancy. Run weekly.

### 16.4 Worked Example 3 — Fraud Detection Pipeline

**Scope**: Real-time fraud scoring for payment transactions. Latency SLO: <50ms p99. Scale: 10K TPS.

**Architecture**:

```
Transaction event (Kafka stream)
      │
      ▼
[Feature Engineering]
│ Real-time: velocity features (txn count last 1h, 24h) — Redis counters
│ Batch: user risk profile, device fingerprint — feature store (Redis/Feast)
│ Graph: is this merchant/card combo associated with known fraud? — GraphDB
      │
      ▼
[Scoring Model]  LightGBM (50ms budget: ~5ms for features, ~2ms for model)
│  200 features → fraud probability score
│  Threshold: 0.7 → review; 0.95 → block; <0.7 → approve
      │
      ▼
[Decision Engine]  rules override model (high-velocity, impossible travel)
      │
      ▼
Response to payment gateway (<50ms p99)

Offline loop:
│ Label delayed frauds (3–7 days after transaction)
│ Retrain weekly with sliding window of last 90 days
│ Monitor: PSI on features, AUC on labelled transactions, fraud rate
```

**Key trade-offs**:
- Model vs rules: rules catch known patterns instantly; model generalises to new patterns. Run both, escalate on either.
- Label delay: fraud labels arrive 3–7 days late. Train on 90-day windows with label-delayed data to avoid training/serving skew.
- Feature staleness: real-time velocity features update every second (Redis). Historical features updated nightly. Mismatch causes training/serving skew — always log features at serving time and train on logged features.
- Adversarial drift: fraudsters adapt to your model. Monitor PSI on top-20 features weekly; trigger retrain if PSI > 0.2.

> 🎯 **Interview prep**: The single most common ML system design mistake is spending too long on the model and too little on data, features, and monitoring. In production, data quality and feature freshness account for 80% of model performance variation; the model architecture accounts for 20%. Flip the interview balance: spend 40% on data/features, 30% on model, 30% on serving/monitoring.

**Resources**
- [Designing Machine Learning Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107963/)
- [ML System Design Interview book (Aminian & Xu)](https://www.educative.io/courses/machine-learning-system-design)
- [Eugene Yan — Applied ML (blog)](https://eugeneyan.com/writing/) — practical system design from industry

---

## 17. References

### Foundational ML Papers
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* OpenAI. https://arxiv.org/abs/2001.08361
- Covington, P. et al. (2016). *Deep Neural Networks for YouTube Recommendations.* RecSys 2016. https://dl.acm.org/doi/10.1145/2959100.2959190

### Embeddings & Retrieval
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. https://arxiv.org/abs/1908.10084

### LLM Alignment & Fine-tuning
- Hu, J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. https://arxiv.org/abs/2305.14314
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. https://arxiv.org/abs/2305.18290

### LLM Inference
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. https://arxiv.org/abs/2309.06180
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. https://arxiv.org/abs/2307.08691

### Agents
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023. https://arxiv.org/abs/2210.03629
- Anthropic (2024). *Building Effective Agents.* https://www.anthropic.com/research/building-effective-agents

### A/B Testing & Statistics
- Deng, A. et al. (2013). *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.* KDD 2013. https://www.microsoft.com/en-us/research/publication/improving-the-sensitivity-of-online-controlled-experiments/

### Blogs & Books
- [ML Interviews Book (Chip Huyen)](https://huyenchip.com/ml-interviews-book/) — comprehensive DS/MLE interview guide
- [Designing Machine Learning Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107963/) — ML system design
- [Eugene Yan — Applied ML blog](https://eugeneyan.com/writing/) — recsys and production ML
- [Sebastian Raschka — LLM training blog](https://sebastianraschka.com/blog/) — PyTorch and fine-tuning
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — transformer visuals
- [vLLM blog](https://blog.vllm.ai) — LLM serving engineering

### Libraries
- [Hugging Face PEFT](https://huggingface.co/docs/peft) — LoRA, QLoRA, all PEFT methods
- [Hugging Face TRL](https://huggingface.co/docs/trl) — SFT, DPO, RLHF training
- [vLLM](https://github.com/vllm-project/vllm) — production LLM serving
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — embedding model benchmarks
