# Recommender Systems: From Collaborative Filtering to Production Two-Tower Architectures

> A production-first deep dive for Data Scientists, ML Engineers, and AI Engineers preparing for recommendation-system interviews and real-world deployments.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Core Metrics That Drive Decisions](#2-core-metrics-that-drive-decisions)
3. [Probability and Exposure Bias](#3-probability-and-exposure-bias)
4. [Collaborative Filtering](#4-collaborative-filtering)
5. [Implicit Feedback Statistics](#5-implicit-feedback-statistics)
6. [Two-Stage Architecture](#6-two-stage-architecture)
7. [Feature Engineering for Ranking Models](#7-feature-engineering-for-ranking-models)
8. [Popularity Bias and Exposure Bias](#8-popularity-bias-and-exposure-bias)
9. [Propensity Scoring and Debiasing](#9-propensity-scoring-and-debiasing)
10. [A/B Testing for Recommenders](#10-ab-testing-for-recommenders)
11. [Offline vs Online Metrics](#11-offline-vs-online-metrics)
12. [Calibration in Recommenders](#12-calibration-in-recommenders)
13. [Diversity, Novelty, and Serendipity](#13-diversity-novelty-and-serendipity)
14. [Fairness in Recommenders](#14-fairness-in-recommenders)
15. [Sequential Recommendation](#15-sequential-recommendation)
16. [Bandits and Exploration](#16-bandits-and-exploration)
17. [Launch Checklist](#17-launch-checklist)
18. [Practical Build Order](#18-practical-build-order)
19. [References](#19-references)

---

## 1. Problem Framing

Before writing a single line of model code, three questions determine the entire architecture: what are you retrieving, what are you optimizing, and whose long-term interest are you serving?

**Retrieval vs Ranking vs Re-ranking.** A recommender is rarely a single model. It is a pipeline. Retrieval casts a wide net — pulling a few hundred candidates from millions of items in milliseconds. Ranking scores each candidate more carefully using richer features. Re-ranking applies business logic: deduplication, freshness boosts, diversity constraints, policy rules. Conflating these stages causes problems: retrieval models trained as classifiers miss the recall property they need; ranking models applied to the full catalog are too slow.

**Explicit vs Implicit Feedback.** Explicit feedback (star ratings, thumbs up/down) is clean but rare. Netflix famously found that predicted ratings did not correlate well with what users actually watched ([Gomez-Uribe & Hunt, 2015](https://dl.acm.org/doi/10.1145/2827872)). Implicit feedback (clicks, plays, scrolls, add-to-cart, skips) is abundant but noisy. A click means interest; a long watch means satisfaction; a skip signals active rejection. The signal type determines the loss function.

**Short-term vs Long-term Objective.** Optimizing next-click CTR is easy to measure and easy to game. A model that maximizes CTR can learn to recommend clickbait — high short-term engagement, low long-term retention. YouTube's shift to watch time over clicks ([Covington et al., 2016](https://dl.acm.org/doi/10.1145/2959100.2959190)) was an attempt to align the proxy metric closer to the real objective (user satisfaction). Getting the objective right matters more than getting the model right.

🎯 **Interview prep:** An interviewer asking "design a recommendation system for X" expects you to open by clarifying: explicit or implicit feedback? Short-term or long-term metric? Who are we optimizing for — the user, the content creator, or the platform?

---

## 2. Core Metrics That Drive Decisions

Every metric in recommenders is a proxy. Understanding what each proxy actually measures — and where it breaks — is a core interviewing and production skill.

**Click-through Rate (CTR)** = clicks / impressions. High position items get more impressions by default, so raw CTR is position-biased. CTR also rewards thumbnails and titles over actual content quality.

**Conversion Rate (CVR)** = purchases (or enrollments, completions) / clicks. CVR introduces a selection problem: only clicked items can convert, so the denominator is already filtered by CTR quality.

**Dwell time / Watch time** — more honest than clicks because the user has to actually consume the content. But they can still be gamed (autoplay, captive audience).

**NDCG (Normalized Discounted Cumulative Gain)** is the dominant offline ranking metric. It discounts relevance by rank position and normalizes by the ideal ordering:

```
DCG@K = Σ (rel_i / log2(i+1)) for i=1..K
NDCG@K = DCG@K / IDCG@K
```

A relevance label of 1 at rank 5 contributes less than a relevance label of 1 at rank 1. NDCG rewards putting good things at the top.

**Recall@K** measures whether relevant items appear in the top K at all, regardless of their exact rank. Useful for retrieval-stage evaluation.

**Coverage** = fraction of catalog recommended at least once. A model with 95% NDCG that only recommends 5% of the catalog is failing at discovery.

**Novelty** — average inverse popularity of recommended items. High novelty means the recommender surfaces items the user hasn't seen before.

```python
import numpy as np

def ndcg_at_k(relevances: list[int], k: int) -> float:
    """Compute NDCG@K given a ranked list of relevance labels."""
    gains = np.array(relevances[:k], dtype=float)
    discounts = np.log2(np.arange(2, k + 2))          # log2(2), log2(3), ...
    dcg = np.sum(gains / discounts)                    # discounted cumulative gain
    ideal = np.sort(gains)[::-1]                       # best possible ordering
    idcg = np.sum(ideal / discounts)                   # ideal DCG
    return dcg / idcg if idcg > 0 else 0.0

# A result with the most relevant item at rank 1
print(ndcg_at_k([3, 2, 1, 0, 0], k=5))   # close to 1.0
# A result with the most relevant item buried at rank 4
print(ndcg_at_k([0, 0, 0, 3, 2], k=5))   # much lower
```

🏭 **Production note:** Teams that only optimize NDCG end up with a "filter bubble" — high relevance, low diversity, declining long-term engagement. Track NDCG alongside coverage, novelty, and online retention.

---

## 3. Probability and Exposure Bias

The fundamental probability the model is trying to estimate is `P(click | user, item, context)`. But there is a trap: the model only sees (user, item, context) combinations that were actually exposed. Items that were never shown have zero click data — not because they are bad, but because the system never gave them a chance.

This is **exposure bias** ([Chen et al., 2023](https://arxiv.org/abs/2202.06084)). A naive model trained on historical logs will learn:

- Observed (user, item) pairs: positive labels
- Unobserved (user, item) pairs: treated as negatives

But unobserved ≠ negative. An item might have a high true probability of being liked but was never recommended. Training on biased logs reinforces the existing recommendation policy.

Formally, if `o_ui = 1` when item `i` was observed by user `u`, the naive model minimizes:

```
L_naive = Σ_{u,i: o_ui=1} loss(y_ui, ŷ_ui)
```

But this is not an unbiased estimate of `E[loss(y_ui, ŷ_ui)]` because `o_ui` correlates with `y_ui`.

🎯 **Interview prep:** "How does your training data affect what your recommender learns?" — exposure bias is the core answer. Follow up with propensity scoring (Section 9) and exploration policies (Section 16).

---

## 4. Collaborative Filtering

Collaborative filtering (CF) is the idea that users who agreed in the past will agree in the future — and items liked by similar users are likely to be liked by the target user.

**User-user CF:** find users similar to the target user, aggregate their ratings. Expensive at scale because user-user distances must be recomputed as users accumulate history.

**Item-item CF:** for each item, precompute similar items. At serving time, look up items similar to what the user has interacted with. Item similarities are more stable and cheaper to update than user similarities. Amazon's original patent ([Linden et al., 2003](https://ieeexplore.ieee.org/document/1167344)) was item-item CF.

**Matrix Factorization (MF)** decomposes the user-item interaction matrix R into user embeddings P and item embeddings Q such that R ≈ P · Q^T. Singular Value Decomposition (SVD) gives the optimal low-rank approximation but requires dense data. Alternating Least Squares (ALS) handles sparse implicit data well ([Hu et al., 2008](https://ieeexplore.ieee.org/document/4781121)).

```python
import numpy as np

def als_step(R: np.ndarray, U: np.ndarray, V: np.ndarray,
             reg: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """One ALS step: fix V, solve for U; fix U, solve for V."""
    n_users, n_items = R.shape
    k = U.shape[1]                                             # latent dim
    # Solve for each user embedding with V fixed
    for u in range(n_users):
        mask = R[u] > 0                                        # observed items
        V_obs = V[mask]                                        # item embeddings for observed items
        R_obs = R[u, mask]                                     # ratings for observed items
        A = V_obs.T @ V_obs + reg * np.eye(k)                 # normal equations LHS
        b = V_obs.T @ R_obs                                    # normal equations RHS
        U[u] = np.linalg.solve(A, b)                          # closed-form solution
    # Solve for each item embedding with U fixed
    for i in range(n_items):
        mask = R[:, i] > 0
        U_obs = U[mask]
        R_obs = R[mask, i]
        A = U_obs.T @ U_obs + reg * np.eye(k)
        b = U_obs.T @ R_obs
        V[i] = np.linalg.solve(A, b)
    return U, V

# Tiny interaction matrix (5 users, 4 items)
R = np.array([[5,3,0,1],[4,0,4,1],[1,1,0,5],[1,0,0,4],[0,1,5,4]], dtype=float)
U = np.random.randn(5, 2)  # 2-dimensional user embeddings
V = np.random.randn(4, 2)  # 2-dimensional item embeddings
for _ in range(20):
    U, V = als_step(R, U, V)
print("Reconstructed R:\n", np.round(U @ V.T, 1))
```

🏭 **Production note:** Pure MF misses context (time of day, session length, device). Most production systems use MF embeddings as features inside a larger ranking model rather than as the final scorer.

---

## 5. Implicit Feedback Statistics

Almost no production recommender has clean explicit ratings. What it has is a log of user actions, each carrying a different signal strength.

| Action | Signal Strength | Caveat |
|---|---|---|
| Impression | Very weak | Just means it was shown |
| Click | Weak-medium | Curiosity, not satisfaction |
| 5-second view | Weak | Could be accidental |
| 30-minute watch | Strong | Engaged consumption |
| Add to cart | Strong | Purchase intent |
| Purchase | Very strong | Ultimate conversion |
| Skip / scroll past | Negative | Active rejection |

The canonical approach from [Hu et al., 2008](https://ieeexplore.ieee.org/document/4781121) introduces a **confidence weighting** scheme for implicit data. All unobserved (user, item) pairs are treated as "not preferred" with low confidence. Observed interactions get higher confidence proportional to their count:

```
confidence(u, i) = 1 + α · count(u, i)
```

where `α` is a hyperparameter (often 40 in their paper). The preference label `p_ui = 1` if any interaction was observed, 0 otherwise. The model then minimizes a weighted squared error:

```
L = Σ_{u,i} c_ui (p_ui - û_ui)² + regularization
```

🎯 **Interview prep:** "How do you handle the fact that not seeing an item isn't the same as disliking it?" — confidence weighting and propensity scoring are the two main answers.

---

## 6. Two-Stage Architecture

The two-stage (retrieval + ranking) architecture dominates production recommenders at scale ([Covington et al., 2016](https://dl.acm.org/doi/10.1145/2959100.2959190); [Zhao et al., 2019](https://arxiv.org/abs/1902.08588)).

**Stage 1 — Candidate Retrieval.** Given millions of items, find a few hundred candidates fast enough to serve in real time. Methods:
- **Approximate Nearest Neighbor (ANN)** search over user and item embeddings (FAISS, ScaNN)
- **Inverted indexes** over user history (item-item CF)
- **Keyword/tag matching** for fresh content

The retrieval stage optimizes for **recall** — it's fine to over-retrieve. The ranking stage will filter.

**Stage 2 — Ranking.** Score each of the ~500 candidates with a richer model. Can use user history, item metadata, context features, and cross features that would be too expensive to compute over the full catalog. LambdaMART, gradient-boosted trees, or a two-tower neural network with a dot-product layer.

**Stage 3 — Re-ranking.** Apply business rules after the model score: deduplicate by creator, enforce freshness constraints, insert sponsored items, enforce diversity targets. This layer is often rule-based, not learned.

```
Full Catalog (10M items)
        │
        ▼  Retrieval (ANN, ~10ms)
  Candidates (~500)
        │
        ▼  Ranking (gradient boosted tree or DNN, ~50ms)
    Top 50
        │
        ▼  Re-ranking (rules + diversity)
     Final 10-20 shown to user
```

🏭 **Production note:** The retrieval and ranking stages are often trained with different objectives (recall-focused vs precision-focused) and must be evaluated separately. A retrieval model that misses good candidates creates a ceiling for the ranking model — no matter how good ranking is, it can't rank items it never sees.

---

## 7. Feature Engineering for Ranking Models

Even deep learning ranking models are heavily feature-driven. The features that matter most in production:

**User features:**
- Long-term profile: historic category mix over 90 days
- Short-term session: last 3–5 items in current session
- Demographics and lifecycle: new user vs veteran, subscription status
- Device and context: mobile vs desktop, time of day

**Item features:**
- CTR in the last 7/30 days (recency-weighted)
- Average rating or completion rate
- Age (freshness)
- Category, topic, length
- Creator reputation signals

**User-item interaction features (cross features):**
- Category affinity: how often has the user engaged with this item's category?
- Creator affinity: has the user watched this creator before?
- Novelty signal: how different is this item from the user's recent history?

**Context features:**
- Hour of day, day of week
- Surface (home page, search result, email)
- Position in list (if used for position-bias correction)

```python
import pandas as pd
import numpy as np

def build_ranking_features(user_history: pd.DataFrame,
                           candidates: pd.DataFrame) -> pd.DataFrame:
    """Build features for ranking stage given user history and candidate items."""
    # User-level aggregates from history
    user_stats = user_history.groupby("user_id").agg(
        user_clicks_7d=("clicked", "sum"),                    # total clicks last 7 days
        user_watch_time_7d=("watch_seconds", "sum"),          # total watch time
        user_top_category=("category", lambda x: x.mode()[0])# most common category
    ).reset_index()

    # Item-level aggregates
    item_stats = user_history.groupby("item_id").agg(
        item_ctr_7d=("clicked", "mean"),                      # item CTR last 7 days
        item_impressions_7d=("item_id", "count")              # exposure count
    ).reset_index()

    # Merge onto candidates
    features = candidates.merge(user_stats, on="user_id", how="left")
    features = features.merge(item_stats, on="item_id", how="left")

    # Cross feature: does the candidate's category match the user's top category?
    features["category_affinity"] = (
        features["category"] == features["user_top_category"]
    ).astype(int)

    # Freshness: days since item was published (lower is fresher)
    features["item_age_days"] = (
        pd.Timestamp.now() - pd.to_datetime(features["published_at"])
    ).dt.days

    return features.fillna(0)                                 # fill missing with 0 as conservative default
```

---

## 8. Popularity Bias and Exposure Bias

These two biases compound each other in a feedback loop that is one of the hardest structural problems in recommenders.

**Popularity bias:** the model sees many training examples for popular items, learns their patterns well, and recommends them more. This gives popular items more exposure, more clicks, more training data — and the cycle continues. Long-tail items never accumulate enough signal to be recommended even when they are highly relevant.

**Exposure bias:** items that are never recommended never accumulate clicks. The model interprets the absence of clicks as negative signal and further avoids recommending these items.

The resulting system "rich gets richer" dynamics are documented extensively ([Abdollahpouri et al., 2017](https://dl.acm.org/doi/10.1145/3109859.3109912)). Measuring it is straightforward:

```python
from collections import Counter
import numpy as np

def gini_coefficient(recommendations: list[list[str]]) -> float:
    """Gini coefficient over item frequency distribution in recommendations."""
    counts = Counter(item for recs in recommendations for item in recs)  # item frequencies
    freqs = sorted(counts.values())                                       # sort ascending
    n = len(freqs)
    cumsum = np.cumsum(freqs)                                             # cumulative sum
    gini = (2 * np.sum(cumsum) - (n + 1) * sum(freqs)) / (n * sum(freqs))
    return gini  # 0 = perfect equality, 1 = one item gets all recommendations

def catalog_coverage(recommendations: list[list[str]], catalog_size: int) -> float:
    """Fraction of catalog recommended at least once."""
    unique_recommended = len(set(item for recs in recommendations for item in recs))
    return unique_recommended / catalog_size

# Simulate heavily popularity-biased recommendations
biased = [["item_1", "item_1", "item_2"] for _ in range(100)]
print(f"Gini (biased): {gini_coefficient(biased):.3f}")      # high
print(f"Coverage (biased): {catalog_coverage(biased, 1000):.3f}")  # very low
```

🎯 **Interview prep:** "How do you know if your recommender has popularity bias?" — check the Gini coefficient of recommended item frequency and compare it to the natural item popularity Gini. If they're equal or the recommender is more concentrated, you're amplifying bias.

---

## 9. Propensity Scoring and Debiasing

The standard fix for exposure bias is **Inverse Propensity Weighting (IPW)** ([Schnabel et al., 2016](https://dl.acm.org/doi/10.1145/2939672.2939745)). The propensity `p_ui` is the probability that user `u` would have been exposed to item `i` under the current logging policy. Weighting each observed interaction by `1/p_ui` creates an unbiased estimator:

```
L_IPW = Σ_{u,i: o_ui=1} (1/p_ui) · loss(y_ui, ŷ_ui)
```

Items that are rarely shown (low propensity) get upweighted. Items that are shown constantly (high propensity) get downweighted. This removes the bias but increases variance.

**Position bias** is a concrete instance of exposure bias. Items shown at rank 1 get clicked more than identical items at rank 5, purely because of their position. The propensity model for position:

```python
import numpy as np

# Empirical position propensity: P(click | position) assuming no item effect
# Estimated from randomization experiments or regression discontinuity
position_propensity = np.array([1.0, 0.62, 0.45, 0.34, 0.26])  # positions 1-5

def ipw_loss(y_true: np.ndarray, y_pred: np.ndarray,
             positions: np.ndarray) -> float:
    """IPW-corrected cross-entropy loss for position bias."""
    propensities = position_propensity[positions - 1]              # look up propensity by position
    weights = 1.0 / propensities                                   # inverse propensity weights
    weights = weights / weights.sum()                               # normalize weights
    log_loss = -y_true * np.log(y_pred + 1e-7) - (1 - y_true) * np.log(1 - y_pred + 1e-7)
    return np.sum(weights * log_loss)                              # weighted average loss
```

**Doubly Robust (DR) estimation** combines IPW with a direct model of the outcome, achieving lower variance when the outcome model is approximately correct ([Wang et al., 2019](https://arxiv.org/abs/1812.04889)).

🏭 **Production note:** Propensity estimation requires a propensity model — which is itself trained on logged data and can be biased. At minimum, run randomization experiments (epsilon-greedy or bolted-on explore) to collect unbiased data for propensity calibration.

---

## 10. A/B Testing for Recommenders

Online A/B testing is the final arbiter of recommender quality. But recommender A/B tests have specific failure modes that standard A/B testing doctrine doesn't cover.

**Novelty effect:** a new recommender often shows a short-term CTR bump simply because the content is fresh. Users click on different items not because they're better, but because they're new. This novelty effect decays after 1–2 weeks.

**Long-term effects:** a recommender that maximizes short-term engagement can harm long-term retention if it collapses diversity or accelerates content fatigue. Run A/B tests for at least 2 weeks; check 30-day retention as a guardrail.

**Interference / network effects:** if item A is recommended more in the treatment group, and item A shares impressions with item B (e.g., same creator, same playlist), the control group's metrics for item B are affected. This violates the SUTVA assumption. Solutions: geo-based splitting, user-cluster randomization.

**Primary metrics and guardrails:**

| Metric | Role |
|---|---|
| CTR or CVR on target action | Primary — what the model optimizes |
| Diversity (intra-list) | Guardrail — should not drop |
| Catalog coverage | Guardrail — should not collapse |
| 14-day retention | Guardrail — should not drop |
| Creator/item exposure fairness | Guardrail — check for winners/losers |

🎯 **Interview prep:** "How long do you run a recommender A/B test?" — long enough to wash out novelty effects (usually 2 weeks minimum) and long enough to measure at least one retention cohort.

---

## 11. Offline vs Online Metrics

The correlation between offline and online metrics is real but imperfect. Understanding where it breaks is critical.

**Why offline metrics mislead:**
1. The test set is drawn from the same biased logging distribution as training — it doesn't reflect what users would do given new recommendations.
2. NDCG@K assumes you know which items are relevant. In practice, relevance labels come from past behavior — biased again.
3. Offline metrics don't capture presentation effects (layout, thumbnails, copywriting).

**Why a model with worse offline NDCG can win online:**
- Better freshness (offline test set is historical, doesn't reward new items)
- Better diversity (offline set doesn't penalize filter bubbles)
- Better cold-start handling (offline set has few cold items by definition)

**Rule of thumb:** if an offline improvement is smaller than 1–2%, treat it as noise until confirmed online. If offline improvement is large (> 5%), it usually (but not always) translates online.

```python
from scipy import stats

def offline_online_correlation(offline_delta: list[float],
                               online_delta: list[float]) -> float:
    """Pearson correlation between offline and online metric deltas across experiments."""
    r, p = stats.pearsonr(offline_delta, online_delta)
    print(f"Correlation: {r:.2f}, p-value: {p:.4f}")
    return r

# Example: past 12 A/B tests, offline NDCG delta vs online CTR delta
offline_d = [0.005, 0.012, -0.002, 0.008, 0.020, 0.001, 0.015, -0.003, 0.010, 0.003, 0.018, 0.006]
online_d  = [0.008, 0.015, -0.001, 0.005, 0.022, -0.002, 0.011, -0.004, 0.013, 0.000, 0.014, 0.004]
offline_online_correlation(offline_d, online_d)
```

🏭 **Production note:** Build a "prediction accuracy" tracking system for your team — log every A/B test's offline prediction vs online outcome. Over time, this tells you whether your offline evaluation is predictive or misleading.

---

## 12. Calibration in Recommenders

A ranking model trained on clicks predicts `P(click)`. But for downstream use — budget allocation, triggering thresholds, exploration policies — you need those probabilities to be calibrated: if the model says 0.7, roughly 70% of those cases should actually click.

Most ranking models are **not** calibrated out of the box. Common miscalibration patterns:
- Models trained with cross-entropy on highly imbalanced data (1% CTR) tend to produce compressed probability scores clustered near the mean.
- Models trained with ranking losses (LambdaMART) produce scores that are good for ordering but meaningless as probabilities.

**Calibration curve** check:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Plot reliability diagram for a recommender score."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")   # calibration curve
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")# diagonal
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_curve.png")
    print("Saved calibration_curve.png")

# Expected Calibration Error (ECE)
def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error — weighted mean absolute calibration gap."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])       # items in this score bin
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()                            # actual positive rate in bin
        bin_conf = y_prob[mask].mean()                           # mean predicted probability
        ece_val += mask.sum() * abs(bin_acc - bin_conf)          # weighted absolute gap
    return ece_val / len(y_true)
```

**Calibration fixes:** Platt scaling (logistic regression on model output), isotonic regression (non-parametric, more flexible but needs more data). Apply calibration on a held-out validation set, not the training set.

---

## 13. Diversity, Novelty, and Serendipity

Accuracy metrics alone produce filter bubbles. Three beyond-accuracy properties are regularly tracked in production systems ([Silveira et al., 2019](https://dl.acm.org/doi/10.1145/3331184.3331369)):

**Diversity** measures how different items in the same recommendation list are from each other. Intra-list diversity (ILD):

```
ILD = (2 / (K*(K-1))) · Σ_{i≠j} distance(item_i, item_j)
```

where distance can be cosine distance in embedding space or Jaccard distance over category tags.

**Novelty** measures how unexpected items are given user history. A simple proxy: average inverse log-popularity of recommended items. Rare items have high novelty.

**Serendipity** combines relevance and surprise — an item the user wouldn't have found on their own but ends up loving. Hard to measure offline; requires long-term user surveys or proxy metrics.

```python
import numpy as np
from itertools import combinations

def intra_list_diversity(item_embeddings: np.ndarray) -> float:
    """Average pairwise cosine distance within a recommendation list."""
    n = len(item_embeddings)
    if n < 2:
        return 0.0
    total_dist = 0.0
    count = 0
    for i, j in combinations(range(n), 2):
        a, b = item_embeddings[i], item_embeddings[j]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        total_dist += 1 - cos_sim                               # cosine distance
        count += 1
    return total_dist / count

def novelty(recommended_items: list[str],
            item_popularity: dict[str, int],
            total_interactions: int) -> float:
    """Average self-information (surprise) of recommended items."""
    scores = []
    for item in recommended_items:
        p = item_popularity.get(item, 1) / total_interactions   # popularity probability
        scores.append(-np.log2(p))                              # self-information
    return np.mean(scores)

# Example
embs = np.random.randn(5, 16)                   # 5 items, 16-dimensional embeddings
print(f"ILD: {intra_list_diversity(embs):.3f}")

pop = {"item_a": 10000, "item_b": 50, "item_c": 200}
print(f"Novelty: {novelty(['item_a', 'item_b'], pop, 100000):.2f}")
```

**Re-ranking for diversity:** Maximum Marginal Relevance (MMR) greedily selects items that balance relevance to the user and dissimilarity from already-selected items:

```
MMR_score(item) = λ · relevance(item) - (1-λ) · max_similarity(item, selected_items)
```

---

## 14. Fairness in Recommenders

Recommendation systems can be unfair in multiple directions simultaneously ([Burke et al., 2017](https://dl.acm.org/doi/10.1145/3109859.3109900)):

**User fairness:** do different user groups receive equally relevant recommendations? A new user gets poor recommendations (cold start) — that's a fairness gap relative to established users.

**Provider/item fairness:** do all content creators or sellers receive proportional exposure? A marketplace recommender that systematically underexposes small sellers harms them economically.

**Exposure fairness:** should exposure be proportional to relevance (relevance-proportional exposure) or to some equity criterion (uniform exposure)? There is no universally correct answer — it depends on the platform's values.

**Measuring provider fairness:**

```python
import pandas as pd
import numpy as np

def provider_exposure_gini(recommendations: pd.DataFrame) -> float:
    """Gini coefficient over creator/provider exposure in recommendations."""
    exposures = recommendations.groupby("provider_id")["item_id"].count()
    exposures = exposures.sort_values().values.astype(float)
    n = len(exposures)
    cumsum = np.cumsum(exposures)
    gini = (2 * np.sum(cumsum) - (n + 1) * exposures.sum()) / (n * exposures.sum())
    return gini                                                 # 0 = uniform, 1 = one provider gets all

def relevant_exposure_ratio(recommendations: pd.DataFrame,
                            relevance: pd.DataFrame) -> pd.DataFrame:
    """Compare actual exposure share to relevance share by provider."""
    exposure_share = (recommendations.groupby("provider_id")["item_id"].count()
                      / len(recommendations))                  # fraction of total exposures
    relevance_share = (relevance.groupby("provider_id")["relevance"].mean()
                       / relevance["relevance"].mean())        # normalized relevance
    result = pd.DataFrame({
        "exposure_share": exposure_share,
        "relevance_share": relevance_share
    }).dropna()
    result["over_under_exposed"] = result["exposure_share"] / result["relevance_share"]
    return result
```

🎯 **Interview prep:** "Is your recommender fair to content creators?" — measure provider exposure Gini, compare actual exposure to relevance-proportional exposure, and identify systematically underexposed providers.

---

## 15. Sequential Recommendation

A user's immediate next action depends more on their recent session context than on their full historical profile. Sequential models capture this by treating recommendation as sequence modeling.

**SASRec (Self-Attentive Sequential Recommendation)** ([Kang & McAuley, 2018](https://arxiv.org/abs/1808.09781)) applies a Transformer encoder to the user's item interaction sequence. The model attends to relevant past interactions to predict the next item.

**BERT4Rec** ([Sun et al., 2019](https://arxiv.org/abs/1904.06690)) uses bidirectional attention with masked item prediction (like BERT's masked language modeling) to learn richer item representations from sequences.

The key design question: **how long is the relevant context window?** A 5-item session window is usually sufficient for next-item prediction. A 30-item window captures short-term topic drift. Full history captures stable preferences.

```python
import torch
import torch.nn as nn

class SessionEncoder(nn.Module):
    """Simple GRU-based session encoder for sequential recommendation."""
    def __init__(self, n_items: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0) # +1 for padding
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)          # sequence encoder
        self.output = nn.Linear(hidden_dim, n_items)                         # score over items

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(item_seq)                          # (batch, seq_len, embed_dim)
        _, hidden = self.gru(emb)                              # (1, batch, hidden_dim)
        logits = self.output(hidden.squeeze(0))                # (batch, n_items)
        return logits

model = SessionEncoder(n_items=1000)
seq = torch.randint(1, 1001, (32, 10))                        # batch of 32, sessions of length 10
scores = model(seq)
print(f"Output shape: {scores.shape}")                        # (32, 1000) scores over all items
```

🏭 **Production note:** Session context staleness is a real problem. If the user's session started 2 hours ago, the last 3 actions from that session are more predictive than 200 historic actions from 3 months ago. Many production systems maintain a real-time session embedding updated per click.

---

## 16. Bandits and Exploration

Every recommender faces the **exploration-exploitation tradeoff**. Exploit: recommend items you know the user likes. Explore: recommend items you're uncertain about to gather information.

Pure exploitation leads to stagnation — cold-start items and cold-start users never get enough signal to be recommended. Three classic bandit strategies:

**Epsilon-greedy:** with probability ε, recommend a random item; otherwise recommend the best known item. Simple but inefficient — random exploration wastes opportunities.

**UCB (Upper Confidence Bound):** recommend the item with the highest upper confidence bound on its expected reward. UCB1:

```
UCB(i) = μ_i + √(2 ln(t) / n_i)
```

where `μ_i` is the estimated reward, `t` is total pulls so far, and `n_i` is times item `i` was shown. This gives items a bonus for being underexplored.

**Thompson Sampling:** maintain a Beta distribution `Beta(α_i, β_i)` for each item's click probability. At each step, sample a probability from each item's distribution and recommend the item with the highest sample. More Bayesian and often more efficient than UCB in practice.

```python
import numpy as np

class ThompsonSamplingBandit:
    """Multi-armed bandit with Thompson sampling for item exploration."""
    def __init__(self, n_items: int):
        self.alpha = np.ones(n_items)                          # successes + 1 (Beta prior)
        self.beta = np.ones(n_items)                           # failures + 1 (Beta prior)

    def select_item(self) -> int:
        """Sample from each item's Beta distribution, pick the argmax."""
        samples = np.random.beta(self.alpha, self.beta)        # sample from each Beta posterior
        return int(np.argmax(samples))                         # pick item with highest sample

    def update(self, item_idx: int, reward: int):
        """Update posterior after observing a reward (1=click, 0=no click)."""
        self.alpha[item_idx] += reward                         # increment successes
        self.beta[item_idx] += 1 - reward                      # increment failures

bandit = ThompsonSamplingBandit(n_items=100)
# Simulate 1000 impressions with true CTRs
true_ctrs = np.random.uniform(0.01, 0.20, 100)               # random CTRs for 100 items
rewards = []
for _ in range(1000):
    item = bandit.select_item()
    reward = int(np.random.random() < true_ctrs[item])        # simulate click
    bandit.update(item, reward)
    rewards.append(reward)
print(f"Average reward: {np.mean(rewards[-200:]):.3f}")       # reward in last 200 steps
print(f"Best item CTR: {true_ctrs.max():.3f}")                # upper bound
```

**Contextual bandits** extend this to use features (user context, item features) to predict the reward distribution, enabling personalized exploration. LinUCB ([Li et al., 2010](https://dl.acm.org/doi/10.1145/1772690.1772758)) and Neural Bandits are the production-ready versions.

---

## 17. Launch Checklist

Before shipping a new recommender or ranking model, check all six dimensions:

**Check 1 — Overall CTR and CVR.** Primary metrics should be at or above baseline in A/B test. Check absolute values, not just relative lifts.

**Check 2 — By-position metrics.** Does CTR at rank 1 match historical rank-1 CTR? A model that degrades rank-1 quality while improving average NDCG is often a bad trade.

**Check 3 — By-user-segment performance.** Break primary metric by: new vs veteran users, mobile vs desktop, user activity quartile. A win on average can mask a loss for a critical segment (e.g., new users who determine retention).

**Check 4 — Popularity concentration and catalog coverage.** Check Gini coefficient of recommended item distribution. Coverage (fraction of catalog in recommendations) should not decrease significantly.

**Check 5 — Calibration by score bucket.** If model scores drive business logic (triggers, budgets, bids), verify ECE < 5% across score deciles.

**Check 6 — Long-tail item exposure.** What fraction of recommended items are in the bottom 20% of item popularity? A diversity-healthy recommender sends some traffic to the long tail.

```python
def launch_checklist(experiment_results: dict) -> list[str]:
    """Return list of failed checks for a recommender launch."""
    failures = []
    if experiment_results["ctr_lift"] < -0.005:
        failures.append("FAIL: CTR dropped > 0.5%")
    if experiment_results["coverage_new"] < 0.8 * experiment_results["coverage_control"]:
        failures.append("FAIL: catalog coverage dropped > 20%")
    if experiment_results["gini_new"] > experiment_results["gini_control"] + 0.05:
        failures.append("FAIL: item concentration (Gini) increased > 5 points")
    if experiment_results["ece"] > 0.05:
        failures.append("FAIL: ECE > 5% — model is miscalibrated")
    if experiment_results["new_user_ctr_lift"] < -0.01:
        failures.append("FAIL: new user CTR dropped > 1%")
    if experiment_results["longtail_exposure_ratio"] < 0.8:
        failures.append("FAIL: long-tail exposure dropped > 20%")
    return failures if failures else ["All checks passed — safe to launch"]
```

---

## 18. Practical Build Order

1. **Define business objective and counter-metric.** Choose one primary metric (e.g., 30-day retention) and one counter-metric that you will not sacrifice (e.g., catalog coverage ≥ 40%).

2. **Choose base candidate generation.** For cold start: popularity + content-based retrieval. For warm users: item-item CF or two-tower ANN. Measure recall@500 on a time-aware test split.

3. **Build ranking features and time-aware splits.** Feature engineering as in Section 7. Always split on time — train on interactions before date D, validate on D to D+N, test on D+N onward.

4. **Evaluate offline ranking metrics and calibration.** NDCG@10 and Recall@50 as primary offline metrics. ECE for calibration. Run SHAP to understand which features drive rankings.

5. **Check popularity bias, diversity, and fairness slices.** Gini coefficient, catalog coverage, provider exposure report (Section 14), diversity ILD (Section 13).

6. **Run online A/B test with guardrails.** Two weeks minimum. Primary metric + all six launch-checklist dimensions. Check novelty effect by looking at week-1 vs week-2 metric trend.

---

## 19. References

### Foundational Papers

- [Linden et al. (2003). Amazon.com Recommendations: Item-to-Item Collaborative Filtering. *IEEE Internet Computing*.](https://ieeexplore.ieee.org/document/1167344)
- [Hu et al. (2008). Collaborative Filtering for Implicit Feedback Datasets. *ICDM*.](https://ieeexplore.ieee.org/document/4781121)
- [Covington et al. (2016). Deep Neural Networks for YouTube Recommendations. *RecSys*.](https://dl.acm.org/doi/10.1145/2959100.2959190)
- [Gomez-Uribe & Hunt (2015). The Netflix Recommender System. *ACM TIST*.](https://dl.acm.org/doi/10.1145/2827872)

### Bias and Fairness

- [Schnabel et al. (2016). Recommendations as Treatments: Debiasing Learning and Evaluation. *ICML*.](https://dl.acm.org/doi/10.1145/2939672.2939745)
- [Abdollahpouri et al. (2017). Controlling Popularity Bias in Learning to Rank for Recommendation. *RecSys*.](https://dl.acm.org/doi/10.1145/3109859.3109912)
- [Burke et al. (2017). Multisided Fairness for Recommendation. *FAT ML Workshop*.](https://dl.acm.org/doi/10.1145/3109859.3109900)
- [Chen et al. (2023). Bias and Debias in Recommender System: A Survey and Future Directions.](https://arxiv.org/abs/2202.06084)

### Sequential Recommendation

- [Kang & McAuley (2018). Self-Attentive Sequential Recommendation. *ICDM*.](https://arxiv.org/abs/1808.09781)
- [Sun et al. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder. *CIKM*.](https://arxiv.org/abs/1904.06690)

### Bandits

- [Li et al. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. *WWW*.](https://dl.acm.org/doi/10.1145/1772690.1772758)
- [Wang et al. (2019). Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random.](https://arxiv.org/abs/1812.04889)

### Beyond Accuracy

- [Silveira et al. (2019). How Good Your Recommender System Is? A Survey on Evaluations in Recommendation. *IJIS*.](https://dl.acm.org/doi/10.1145/3331184.3331369)
- [Zhao et al. (2019). Recommending What Video to Watch Next: A Multitask Ranking System. *RecSys*.](https://arxiv.org/abs/1902.08588)

### Production Guides

- [Google Machine Learning Guides — Recommendation Systems](https://developers.google.com/machine-learning/recommendation)
- [Google Responsible AI — Fairness](https://developers.google.com/machine-learning/guides/intro-responsible-ai/fairness)
