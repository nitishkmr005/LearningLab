# [Project Name] — Project Brainstorm Document

**Date:** [YYYY-MM-DD]
**Status:** Draft
**Owner:** [USER TO FILL]
**Domain:** [Tabular ML / NLP / CV / Recommendation / Time-series / RL / Multimodal]
**Reviewed by:** [USER TO FILL]

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current & Previous Approach](#3-current--previous-approach)
4. [Market Landscape](#4-market-landscape)
5. [Literature Survey](#5-literature-survey)
6. [Opportunities & Benefits](#6-opportunities--benefits)
7. [Competitive Moats](#7-competitive-moats)
8. [Metrics Framework](#8-metrics-framework)
9. [EDA Plan](#9-eda-plan)
10. [Technical Approach](#10-technical-approach)
11. [Production Design](#11-production-design)
12. [Project Scope](#12-project-scope)
13. [AI Recommendations](#13-ai-recommendations)
14. [Agent Task Breakdown](#14-agent-task-breakdown)
15. [MVP Roadmap](#15-mvp-roadmap)

---

## 1. Executive Summary

> Write this section last. 5–7 sentences summarizing: what problem, why now, proposed approach, expected impact, and first milestone.

[WRITE LAST]

---

## 2. Problem Statement

> Use the SMART format: Specific, Measurable, Achievable, Relevant, Time-bound.

**In one sentence:** [What specific decision or outcome are you trying to improve?]

**Current pain / gap:**
- [Pain point 1 — with a number if possible]
- [Pain point 2]
- [Pain point 3]

**Who is affected:** [USER TO FILL — teams, customers, processes]

**Why now:** [What changed recently that makes this the right time?]

**Hypothesis:** If we [build X], we expect [metric Y] to improve by [Z%] within [timeframe].

**Out of scope for this problem:** [What adjacent problems are explicitly NOT being solved here]

---

## 3. Current & Previous Approach

### Current Approach
**What is being done today:**
[Describe the current system, process, or manual workflow]

**Current system details:**
- Method: [rule-based / model / manual / hybrid]
- Built by: [USER TO FILL — team/vendor]
- Age: [USER TO FILL]
- Known issues: [list the failure modes or limitations that motivated this project]

**Current performance:**
| Metric | Current Value | Source |
|---|---|---|
| [Business metric 1] | [USER TO FILL] | [Internal dashboard] |
| [Model/proxy metric] | [USER TO FILL] | [Internal dashboard] |

### Previous Approaches
**What was tried before (if anything):**

| Approach | When | Why it failed / was abandoned |
|---|---|---|
| [Approach 1] | [Year] | [Reason] |
| [Approach 2] | [Year] | [Reason] |

**Lessons learned from previous attempts:**
- [Lesson 1]
- [Lesson 2]

---

## 4. Market Landscape

> How is this problem being solved elsewhere? Know the competitive and open-source landscape before designing your approach.

### Commercial Solutions
| Solution | Vendor | Approach | Strengths | Weaknesses | Relevant? |
|---|---|---|---|---|---|
| [Product name] | [Company] | [Brief description] | [+] | [-] | Yes/No |
| [Product name] | [Company] | [Brief description] | [+] | [-] | Yes/No |

### Open Source Approaches
| Library / Framework | Approach | Stars / Adoption | Relevant? |
|---|---|---|---|
| [Name] | [Description] | [GitHub stars or known adoption] | Yes/No |

### Academic / Research Approaches
> See Section 5 (Literature Survey) for detailed citations.

Summary of dominant research direction: [1-2 sentences]

**Why we are not simply using an off-the-shelf solution:**
- [Reason 1 — data privacy, customization, cost, scale, etc.]
- [Reason 2]

---

## 5. Literature Survey

> Papers, articles, and production systems most relevant to this problem. Sourced via web search. All citations required.

### Foundational Papers
| Paper | Authors | Year | Key contribution | Link |
|---|---|---|---|---|
| [Title] | [Authors] | [Year] | [What it contributes to this problem] | [URL] |

### Recent SotA (last 2–3 years)
| Paper / System | Authors | Year | Key contribution | Link |
|---|---|---|---|---|
| [Title] | [Authors] | [Year] | [What it contributes to this problem] | [URL] |

### Production Systems (industry engineering blogs)
| System | Company | Year | Key learning | Link |
|---|---|---|---|---|
| [System name] | [Company] | [Year] | [What to borrow from their design] | [URL] |

### Key Insight from Literature
[2–3 sentences: what does the research consensus say about the best approach, and how does our proposed approach compare or diverge?]

---

## 6. Opportunities & Benefits

> Now that we know the problem, current state, market, and literature — what are the real opportunities?

### Business Opportunities
- [Opportunity 1 — tie to a metric or revenue/cost impact]
- [Opportunity 2]
- [Opportunity 3]

### Technical Opportunities
- [New data signal available that wasn't before]
- [New algorithm / framework that enables this now]
- [Infrastructure improvement that removes previous bottleneck]

### Expected Benefits
| Benefit | Estimated Impact | Confidence | Notes |
|---|---|---|---|
| [Reduce X cost] | [USER TO FILL] | Low / Med / High | [Assumption] |
| [Improve Y metric] | [USER TO FILL] | Low / Med / High | [Assumption] |
| [Save Z hours/week] | [USER TO FILL] | Low / Med / High | [Assumption] |

**Biggest risk to realizing these benefits:** [What could make the benefit not materialize?]

---

## 7. Competitive Moats

> What makes this approach hard to replicate once built? Pick the ones that apply.

| Moat Type | Does it apply? | How we build it | Strength (1–5) |
|---|---|---|---|
| **Data moat** — proprietary or hard-to-replicate data | Yes / No | [Description] | [1–5] |
| **Feedback loop** — model use generates training signal | Yes / No | [Description] | [1–5] |
| **Integration depth** — deeply embedded in workflow | Yes / No | [Description] | [1–5] |
| **Algorithmic edge** — novel approach vs. standard baselines | Yes / No | [Description] | [1–5] |
| **Network effect** — more users → better model for all users | Yes / No | [Description] | [1–5] |
| **Speed moat** — operational cost or latency advantage | Yes / No | [Description] | [1–5] |

**Primary moat for this project:** [Which moat is strongest and why]

**Moat risk:** [What could erode this moat? Competitor data access? Open-source release?]

---

## 8. Metrics Framework

> See `references/metrics-guide.md` for format guidance and metric selection by domain.

### Business Metrics (what the business cares about)
| Metric | Definition | Baseline | Target | Owner | Cadence |
|---|---|---|---|---|---|
| [e.g., Churn rate] | [Definition] | [USER TO FILL] | [USER TO FILL] | [Team] | Weekly |
| [e.g., Revenue at risk] | [Definition] | [USER TO FILL] | [USER TO FILL] | [Team] | Monthly |

### Model / Technical Metrics (what we optimize during training)
| Metric | Definition | Why chosen over alternatives | Baseline | Target |
|---|---|---|---|---|
| [e.g., AUC-ROC] | [Definition] | [Why this metric fits this problem] | [Current] | [Target] |
| [e.g., Precision@K] | [Definition] | [Why this metric fits this problem] | [Current] | [Target] |

### Metric Connection Map
```
Business metric: [Churn rate ↓]
    └── Model metric: [Recall ↑ on high-value churners]
            └── Proxy metric: [AUC on holdout set ≥ 0.85]
                    └── Training signal: [Binary cross-entropy loss]
```

### Guardrail Metrics (things we must NOT break)
| Metric | Constraint | Why it matters |
|---|---|---|
| [e.g., Latency P95] | [≤ 200ms] | [SLA commitment to users] |
| [e.g., False positive rate] | [≤ 5%] | [Prevents alert fatigue] |

### Metrics Reporting Format
```
[Metric name]: [value] ([delta vs. baseline]: ↑/↓ [X%])
Confidence interval: [[lower], [upper]]  n=[sample size]
Measurement window: [date range]
Source: [dashboard / experiment ID]
```

---

## 9. EDA Plan

> What analysis must be done on the data before any modeling decision is made?

**Data sources identified:**
| Source | Type | Owner | Access method | Refresh cadence | Size estimate |
|---|---|---|---|---|---|
| [Source 1] | [Tabular / text / image / event logs] | [USER TO FILL] | [DB query / API / flat file] | [Daily/weekly/static] | [Rows / GB] |

### EDA Checklist

**Data quality checks:**
- [ ] Missing value rate per feature — flag any column >20% missing
- [ ] Duplicate row check — identify and document deduplication strategy
- [ ] Schema validation — confirm data types match expectations
- [ ] Date range coverage — confirm data covers required training window
- [ ] Data freshness — confirm no stale/delayed data

**Distribution analysis:**
- [ ] Target variable distribution — class imbalance ratio if classification
- [ ] Key feature distributions — histograms, outlier identification
- [ ] Temporal drift check — does distribution shift over time?
- [ ] Segment-level analysis — does the distribution differ by key segments?

**Correlation & signal analysis:**
- [ ] Correlation matrix — identify multicollinear features
- [ ] Feature–target correlation — ranked list of predictive features
- [ ] Mutual information scores — for non-linear relationships
- [ ] Interaction effects — are there feature pairs that matter jointly?

**Data leakage risk check:**
- [ ] Features available at prediction time vs. training time — document any gap
- [ ] Target leakage — are any features derived from the target?
- [ ] Temporal leakage — is future data being used to predict the past?

**Specific EDA questions for this project:**
- [EDA question 1 specific to this domain — e.g., "What is the churn rate by cohort?"]
- [EDA question 2]
- [EDA question 3]

**EDA output artifacts:**
- [ ] Data quality report (one-pager)
- [ ] Feature importance pre-screen (correlation / mutual info table)
- [ ] Recommended feature engineering candidates
- [ ] Train/val/test split strategy

---

## 10. Technical Approach

> Approach is now informed by Sections 4 (Market Landscape) and 5 (Literature Survey).

### Algorithm Approach
**Proposed algorithm:** [Algorithm name and family]

**Why this algorithm over alternatives:**
| Algorithm | Pros | Cons | Verdict |
|---|---|---|---|
| [Proposed] | [+] | [-] | **Chosen** |
| [Alternative 1] | [+] | [-] | Rejected / Fallback |
| [Alternative 2] | [+] | [-] | Rejected / Fallback |

**What the literature informed about this choice:** [1–2 sentences connecting this decision to Section 5]

**Algorithm assumptions:** [What data properties does this algorithm assume? Are they met?]

### Tech Stack
| Component | Tool / Framework | Version | Why chosen |
|---|---|---|---|
| Language | [Python / Scala / etc.] | [Version] | [Reason] |
| ML framework | [sklearn / PyTorch / TF / XGBoost / etc.] | [Version] | [Reason] |
| Feature store | [Feast / Tecton / Redis / None] | [Version] | [Reason] |
| Experiment tracking | [MLflow / W&B / Neptune / None] | [Version] | [Reason] |
| Data pipeline | [Spark / dbt / Airflow / etc.] | [Version] | [Reason] |
| Model serving | [FastAPI / TorchServe / Sagemaker / etc.] | [Version] | [Reason] |
| Monitoring | [Evidently / Arize / Whylogs / custom] | [Version] | [Reason] |
| Infra | [AWS / GCP / Azure / on-prem] | — | [Reason] |

### Limitations of This Approach
| Limitation | Severity (High/Med/Low) | Mitigation plan |
|---|---|---|
| [e.g., Cold-start problem for new users] | High | [Use content-based fallback] |
| [e.g., Model interpretability limited] | Med | [Add SHAP explanations] |
| [e.g., Retraining cost high] | Low | [Schedule weekly batch retraining] |

---

## 11. Production Design

### End Users
| User type | How they interact | What they need from the model output |
|---|---|---|
| [e.g., Customer Success Manager] | [Via CRM dashboard] | [Churn risk score + top 3 risk factors] |
| [e.g., Automated email system] | [Via API] | [Binary flag + probability score] |

### Serving Mechanism
**Primary interface:** [REST API / gRPC / Tableau dashboard / Streamlit app / Kafka stream / Scheduled report / Other]

**Secondary interface (if any):** [description]

| Dimension | Design |
|---|---|
| Latency requirement | [Sync ≤ Xms / Async / Batch] |
| Throughput | [Requests/sec or records/hour] |
| Serving pattern | [Real-time online / Near-real-time streaming / Batch offline] |
| Scale | [Expected peak load — users, RPS, records/day] |
| Availability SLA | [99.X% uptime / best-effort] |

### Real-time vs. Batch Decision
- **Real-time serving:** [When and why — latency-sensitive use cases]
- **Batch inferencing:** [When and why — nightly scores, bulk processing]
- **Hybrid:** [If both patterns needed, describe how they split]

### Model Serving Architecture
```
[Data source] → [Feature pipeline] → [Feature store]
                                          ↓
                               [Model serving layer]
                                          ↓
                    [API / Dashboard / Downstream system]
                                          ↓
                              [Monitoring + alerting]
```

### Production Scenarios & Edge Cases
| Scenario | Expected behavior | Fallback |
|---|---|---|
| Model API timeout | [Return default / cached score] | [Rule-based fallback] |
| Feature pipeline delay | [Use stale features up to Xh] | [Flag as low-confidence] |
| Distribution shift detected | [Alert oncall, freeze model update] | [Revert to previous model] |
| Cold start (new entity) | [No historical features] | [Global average / content-based] |

### Future Enhancements
- [Enhancement 1 — out of current scope, planned for later]
- [Enhancement 2]

---

## 12. Project Scope

### In Scope
- [ ] [Deliverable 1]
- [ ] [Deliverable 2]
- [ ] [Deliverable 3]

### Out of Scope (explicit)
- [ ] [What will NOT be built in this project — be specific]
- [ ] [Adjacent problem not being solved]
- [ ] [Feature deferred to future phase]

### Dependencies
| Dependency | Owner | Risk | Mitigation |
|---|---|---|---|
| [Data access from X team] | [USER TO FILL] | High / Med / Low | [Escalation path] |
| [Infra provisioning] | [USER TO FILL] | High / Med / Low | [Timeline] |

### Assumptions
- [Assumption 1 — if this turns out to be false, the approach changes]
- [Assumption 2]

---

## 13. AI Recommendations

> How could this project be made significantly better? Concrete, named techniques only. Informed by Sections 5 and 10.

### Recommendation 1: [Short title]
**What:** [Specific technique, framework, or architectural change]
**Why better:** [What metric or problem does it address, and by how much — cite if possible]
**Effort:** Low / Medium / High
**When to consider:** MVP1 / MVP2 / MVP3 / Future

### Recommendation 2: [Short title]
**What:** [description]
**Why better:** [description]
**Effort:** Low / Medium / High
**When to consider:** MVP1 / MVP2 / MVP3 / Future

### Recommendation 3: [Short title]
**What:** [description]
**Why better:** [description]
**Effort:** Low / Medium / High
**When to consider:** MVP1 / MVP2 / MVP3 / Future

### How to Track These
Add these to the backlog. Revisit after MVP1 results are in.

---

## 14. Agent Task Breakdown

> Tasks decomposed so each can be executed independently by a human or AI agent.
> Format: specific inputs → specific outputs. No vague tasks.

| # | Task name | Description | Inputs | Outputs | Agent type | Effort |
|---|---|---|---|---|---|---|
| 1 | [e.g., Data quality audit] | [Run EDA checks on raw dataset] | [Raw CSV + schema doc] | [Quality report + flagged columns] | AI agent / Data Scientist | [Xh] |
| 2 | [e.g., Feature engineering] | [Build feature set per EDA plan] | [Clean dataset + EDA report] | [Feature matrix + feature doc] | AI agent / MLE | [Xh] |
| 3 | [e.g., Baseline model training] | [Train logistic regression baseline] | [Feature matrix + label column] | [Trained model + eval metrics] | AI agent | [Xh] |
| 4 | [e.g., Model evaluation report] | [Run holdout eval + generate report] | [Trained model + test set] | [Metrics table + confusion matrix] | AI agent | [Xh] |
| 5 | [e.g., API scaffolding] | [Build FastAPI endpoint for model serving] | [Trained model + schema] | [Working API + OpenAPI spec] | AI agent | [Xh] |
| 6 | [e.g., Monitoring setup] | [Configure drift detection + alerting] | [Prod data schema + model] | [Monitoring dashboard + alert config] | AI agent / MLE | [Xh] |

**Which tasks are parallelizable:** [e.g., Tasks 2 and 5 can run in parallel once Task 1 is done]

**Which tasks require human review before proceeding:**
- After Task 1: Review flagged data quality issues before proceeding to feature engineering
- After Task 4: Human sign-off on metrics before production deployment

---

## 15. MVP Roadmap

> Max 4 MVP levels. Each must have a clear deliverable, success criterion, and time estimate.

### MVP0 — [Hypothesis Validation]
**Goal:** Prove the core hypothesis is achievable before investing in infrastructure.
**What we build:**
- [Minimum deliverable — e.g., offline model on sample data, evaluated on holdout set]
- [No production serving required]

**Success criterion:** [Specific, measurable — e.g., "AUC ≥ 0.75 on holdout set"]
**What we do NOT build:** [Be explicit — no API, no monitoring, no scale]
**Estimated effort:** [X person-weeks]
**Key risk:** [What could make MVP0 fail?]

---

### MVP1 — [Working System in Production]
**Goal:** First real deployment usable by internal users or in a limited pilot.
**What we build:**
- [Serving layer — API or batch job]
- [Basic monitoring]
- [Integration with end-user system]

**Success criterion:** [e.g., "System deployed, used by X users, business metric improves Y%"]
**What we do NOT build:** [Advanced features, full scale, external users]
**Estimated effort:** [X person-weeks]
**Dependencies on MVP0:** [What MVP0 must prove before MVP1 begins]

---

### MVP2 — [Production-Ready]
**Goal:** Scale, reliability, and observability at production standard.
**What we build:**
- [Improved model — more features, better algorithm if MVP1 showed headroom]
- [Full monitoring + alerting]
- [Performance at scale]
- [Feedback loop or active learning if applicable]

**Success criterion:** [e.g., "P95 latency < 200ms at 1000 RPS, AUC maintained ≥ 0.82"]
**Estimated effort:** [X person-weeks]

---

### MVP3 — [Advanced / Strategic]
**Goal:** Realize the full vision or competitive moat identified in Section 7.
**What we build:**
- [Advanced algorithmic improvements from Section 13 AI Recommendations]
- [Personalization, multimodal, or RL enhancements if applicable]
- [Self-improving feedback loop]

**Success criterion:** [Business metric target from Section 8]
**Estimated effort:** [X person-weeks]
**Note:** MVP3 scope should be revisited after MVP2 results — don't over-design now.

---

*Document generated by `/brainstorm-ds-project` skill — LearningLab*
