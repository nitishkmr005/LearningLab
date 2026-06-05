# Metrics Guide

Reference for the `brainstorm-ds-project` skill. Covers business metrics, model metrics, metric connection patterns, and the standard reporting format.

---

## The Three Metric Layers

Every DS/ML project must define metrics at three layers and connect them explicitly:

```
Layer 1 — BUSINESS METRICS
(what leadership tracks; can be affected by many things besides the model)
        ↓ the model influences this via...
Layer 2 — PROXY / OPERATIONAL METRICS
(what the model directly controls; measurable in the product)
        ↓ which we measure offline using...
Layer 3 — MODEL / TRAINING METRICS
(what we optimize during development; lives in the experiment tracker)
```

If you can't draw this chain for your project, you don't yet know what you're optimizing.

---

## Layer 1 — Business Metrics

These are owned by product, finance, or operations. The model team influences but does not fully control them.

### Common business metrics by domain

**Retention / Churn:**
- Monthly Churn Rate: `churned_users / active_users_start_of_month`
- Annual Retention Rate: `1 - annual_churn_rate`
- Revenue at Risk: `churned_users × avg_revenue_per_user`
- Customer Lifetime Value (CLV/LTV): `avg_revenue_per_user × (1 / churn_rate)`

**Revenue / Conversion:**
- Conversion Rate: `conversions / sessions`
- Revenue per User (RPU): `total_revenue / active_users`
- Incremental Revenue: `(treated_revenue - control_revenue) / control_revenue`

**Engagement:**
- DAU/MAU ratio (stickiness): `daily_active_users / monthly_active_users`
- Session length: `avg minutes per session`
- Feature adoption rate: `users_using_feature / total_users`

**Operations / Cost:**
- False positive cost: `false_positives × cost_per_false_positive_action`
- Automation rate: `decisions_automated / total_decisions`
- Human review rate: `cases_sent_to_human / total_cases`

---

## Layer 2 — Proxy / Operational Metrics

These are directly measurable in the production system and connect model behavior to business outcomes.

| Business metric | Proxy metric | Connection |
|---|---|---|
| Churn rate ↓ | Precision on high-value churners ↑ | High-precision model → targeted interventions → less wasted outreach → more retention |
| Conversion rate ↑ | Click-through rate ↑ on top-K recommendations | Better recommendations → more clicks → more purchases |
| Support cost ↓ | Auto-resolution rate ↑ | More tickets auto-resolved → fewer agents needed |
| Fraud loss ↓ | Recall on fraudulent transactions ↑ | Catching more fraud → less financial loss |

---

## Layer 3 — Model / Training Metrics

These are what you report in experiments. Choose based on the task type.

### Classification metrics

| Metric | Formula | When to use | When NOT to use |
|---|---|---|---|
| **AUC-ROC** | Area under ROC curve | Ranking tasks; class-imbalanced datasets; comparing models | When you need a specific operating threshold |
| **AUC-PR (Precision-Recall)** | Area under PR curve | Highly imbalanced classes (fraud, rare events) | When classes are balanced |
| **F1 Score** | `2 × (P × R) / (P + R)` | When false positives and false negatives are equally costly | When one type of error matters more |
| **Precision** | `TP / (TP + FP)` | When false positives are costly (spam filter, alert systems) | When recall is the priority |
| **Recall** | `TP / (TP + FN)` | When false negatives are costly (medical, fraud detection) | When precision is the priority |
| **Precision@K** | Precision in top-K predictions | When only the top predictions are acted on | When all predictions are used |
| **Log Loss** | Cross-entropy loss | When calibrated probabilities matter | When ranking order is all that matters |

### Regression metrics

| Metric | Formula | When to use |
|---|---|---|
| **RMSE** | `√(mean(y_pred - y_true)²)` | When large errors are disproportionately costly |
| **MAE** | `mean(|y_pred - y_true|)` | When all errors are equally costly; more interpretable |
| **MAPE** | `mean(|y_pred - y_true| / y_true)` | When relative error matters more than absolute |
| **R²** | Explained variance ratio | For communicating fit quality to non-technical stakeholders |

### Ranking / Recommendation metrics

| Metric | When to use |
|---|---|
| **NDCG@K** | When position in ranking matters (higher rank = more value) |
| **MRR (Mean Reciprocal Rank)** | When finding the first relevant item quickly matters |
| **Hit Rate@K** | When any relevant item in top-K counts as success |
| **Coverage** | When diversity of recommendations matters |
| **Novelty / Serendipity** | When over-recommending popular items is a risk |

### NLP metrics

| Metric | Task |
|---|---|
| **BLEU** | Machine translation, text generation (n-gram overlap) |
| **ROUGE-L** | Summarization (longest common subsequence) |
| **BERTScore** | Any generation task (semantic similarity via embeddings) |
| **Perplexity** | Language model quality |
| **Exact Match / F1** | QA tasks |

### Time-series metrics

| Metric | When to use |
|---|---|
| **MASE (Mean Absolute Scaled Error)** | Scale-independent; handles seasonality |
| **sMAPE** | Symmetric MAPE; handles near-zero actuals |
| **Winkler score** | When prediction intervals matter |

---

## Metrics Reporting Format

Use this format consistently in all sections of the project doc and in presentations:

### Single metric report
```
[Metric name]: [value]
  Delta vs. baseline: ↑/↓ [X%] ([absolute change])
  Confidence interval: [[lower bound], [upper bound]]
  Sample size: n = [N]
  Measurement window: [start date] — [end date]
  Source: [dashboard name / experiment ID / notebook path]
```

### Experiment comparison table
```markdown
| Model | [Metric 1] | [Metric 2] | [Guardrail metric] | Training time | Notes |
|---|---|---|---|---|---|
| Baseline (current) | [value] | [value] | [value] | — | Current production |
| Experiment A | [value ± CI] | [value ± CI] | [value] | [Xmin] | [key difference] |
| Experiment B | [value ± CI] | [value ± CI] | [value] | [Xmin] | [key difference] |
| **Chosen model** | **[value]** | **[value]** | **[value]** | **[Xmin]** | **Winner** |
```

### A/B test result format
```
Test: [Name]
Period: [Start] — [End]
Treatment: [N users] | Control: [N users]

Primary metric: [Metric name]
  Treatment: [value]
  Control: [value]
  Lift: [+X%]
  p-value: [X] (significant: yes/no at α=0.05)
  95% CI on lift: [[lower], [upper]]

Guardrail metrics (must not degrade):
  [Metric]: Treatment [value] vs. Control [value] — ✅ OK / ❌ DEGRADED

Decision: Ship / Do not ship / Needs more data
```

---

## Metric Anti-Patterns

| Anti-pattern | Problem | Fix |
|---|---|---|
| Reporting only accuracy on imbalanced dataset | Inflated metric — model predicts majority class | Use AUC-PR or F1 |
| No confidence intervals | Overstates certainty | Always report CI on key metrics |
| Single-metric optimization | Other metrics degrade silently | Define guardrail metrics |
| Offline metric doesn't predict online performance | Model looks great offline, fails in production | Add online A/B metric as the final gate |
| Changing the metric mid-experiment | P-hacking / results inflation | Pre-register the metric before the experiment starts |
| Metric cannibalism | Optimizing metric A degrades metric B | Define tradeoff explicitly (e.g., "precision ≥ 0.7 AND recall ≥ 0.5") |

---

## Business Metric → Model Metric Connection Examples

### Churn prediction
```
Business: Monthly churn rate ↓ (owned by Growth team)
  └── Operational: % of high-risk users receiving timely intervention ↑
        └── Model: Recall@P≥0.7 (catch churners without too many false alarms)
              └── Training: Binary cross-entropy loss, class weights adjusted for imbalance
```

### Recommendation system
```
Business: Revenue per session ↑ (owned by Monetization team)
  └── Operational: CTR on recommended items ↑, Add-to-cart rate ↑
        └── Model: NDCG@10 ↑, Hit Rate@5 ↑
              └── Training: BPR loss (Bayesian Personalized Ranking) or cross-entropy on implicit feedback
```

### Document classification
```
Business: Support cost per ticket ↓ (owned by Operations team)
  └── Operational: Auto-resolution rate ↑ (tickets closed without human)
        └── Model: Precision ≥ 0.9 on top-1 classification (high precision gate to auto-close)
              └── Training: Cross-entropy loss, label smoothing for noisy labels
```

### Fraud detection
```
Business: Fraud loss rate ↓, False decline rate ↓ (owned by Risk team)
  └── Operational: Fraud catch rate ↑ at specific false positive rate threshold
        └── Model: Recall at FPR = 1% (operating point on ROC curve)
              └── Training: Focal loss (handles extreme imbalance) or cost-sensitive learning
```
