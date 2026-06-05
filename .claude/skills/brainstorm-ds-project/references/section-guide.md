# Section Writing Guide

Detailed guidance for each section in `assets/project-doc-template.md`. Read the relevant part before writing that section.

---

## Section 2 — Problem Statement

**Goal:** Make the problem unambiguous. A vague problem leads to a misaligned model.

Use the SMART framework:
- **Specific:** Name the exact decision, outcome, or process being improved
- **Measurable:** Anchor to a metric (churn rate, revenue, latency, accuracy)
- **Achievable:** Is ML actually the right tool? Could a rule-based system solve this cheaper?
- **Relevant:** Connect to a business goal (revenue, cost, risk, user experience)
- **Time-bound:** When should the improvement be visible?

**Hypothesis template:**
```
If we [build a model that predicts X],
we expect [metric Y] to improve by [Z%]
within [timeframe],
assuming [key assumption].
```

**Red flags that signal a vague problem:**
- "We want a better model" → better than what, measured how?
- "We want to improve customer experience" → which interaction, which signal?
- "We want to use AI/ML" → is there a specific problem being solved?

**If the problem is vague:** Ask the user — "What is one concrete decision that a human makes today that this model will automate or assist?"

---

## Section 3 — Current & Previous Approach

**Goal:** Document what exists so we don't reinvent it, and learn from what failed.

**For the current approach, capture:**
- Is it manual, rule-based, or model-based?
- What data does it use?
- What are its known failure modes? (This motivates the new project)
- How old is it? (Older systems often have data drift, stale features)

**For previous approaches, ask:**
- Why did the previous approach fail? (overfitting, data unavailable, business change, infra issues)
- What signals did it use that we should reconsider?
- What time was wasted that we should avoid?

**Tip:** The "why it failed" column is the most important. Repeating a past failure because it wasn't documented is one of the most common DS project mistakes.

---

## Section 4 — Opportunities & Benefits

**Goal:** Make the business case. Separate what we know (current system's pain) from what we're estimating (impact of the new system).

**Confidence levels:**
- High: based on current measured baseline + conservative model improvement estimate
- Medium: requires assumption about model accuracy or adoption rate
- Low: rough order of magnitude, needs validation

**Common mistake:** Inflating benefits to get project approved → metrics aren't met → trust destroyed. Be conservative and flag assumptions explicitly.

**Opportunity framing (borrowed from DSSG scoping guide):**
1. What decisions will be better?
2. Who makes those decisions and how often?
3. What is the cost of a wrong decision today?
4. What is the cost of a delayed decision?

---

## Section 5 — Market Landscape

**Goal:** Know the competitive and open-source landscape so we don't reinvent a solved problem.

**For commercial solutions:** Focus on adoption, not marketing. If a vendor has many large customers solving the same problem, that's evidence the problem is real and the solution is viable.

**For open-source:** Check GitHub stars, last commit date, and whether production deployments are documented.

**"Why not off-the-shelf" is mandatory.** If you can't answer this, reconsider building custom. Valid reasons include:
- Data privacy (can't send data to a vendor)
- Customization needed beyond what the tool offers
- Cost at scale prohibitive
- Need to own the model for compliance/auditability

---

## Section 6 — Metrics Framework

**See `references/metrics-guide.md` for full format guidance.**

**Critical distinction:**
- **Business metrics** — what the business measures (churn rate, NPS, revenue)
- **Model metrics** — what we optimize during training (AUC, F1, RMSE)
- **These must be connected.** If you can't draw a direct line from model metric to business metric, you're optimizing the wrong thing.

**Common mistakes:**
- Optimizing accuracy on a class-imbalanced dataset → use precision/recall instead
- Optimizing AUC when the business cares about precision at a specific threshold → use Precision@K
- Not defining guardrail metrics → model is deployed and breaks a downstream system

**The Metric Connection Map** is the most important part of this section. Write it first, then fill in the table.

---

## Section 7 — Competitive Moats

**Goal:** Identify what makes this system defensible and hard to replicate.

**Moat types (borrow from ML competitive strategy research):**
- **Data moat:** You have training data competitors don't. Example: historical user behavior, proprietary labels, rare domain data.
- **Feedback loop:** Every prediction creates a new training signal. Example: recommendation systems where clicks train the next model.
- **Integration depth:** Model is woven into the product workflow — removing it is disruptive. Example: an LLM that drafts emails inside a CRM.
- **Algorithmic edge:** You've developed a novel approach. Short-lived moat — gets commoditized quickly.
- **Network effect:** More users → more data → better model for all users. Example: Waze, Duolingo.
- **Speed moat:** Significantly cheaper or faster than alternatives. Cost advantage is a moat.

**Key question:** If a well-funded competitor copied your dataset today, how long would it take them to match your model? If the answer is "3 months", your moat is weak.

---

## Section 8 — EDA Plan

**Goal:** Define the analysis that must happen before any modeling decision.

**Why EDA first, model second:**
Modeling on bad or misunderstood data is the #1 cause of DS project failure. EDA answers:
1. Does the data actually contain the signal needed to solve the problem?
2. What preprocessing is required before training?
3. What are the risks of data leakage?

**Domain-specific EDA questions to always ask:**
- **Classification:** What is the class imbalance ratio? How does it vary by segment?
- **Regression:** Is the target normally distributed? Any outliers that need capping?
- **Recommendation:** How sparse is the user-item matrix? What's the cold-start fraction?
- **Time-series:** Is there seasonality? Trend? Autocorrelation?
- **NLP:** What is the vocabulary size? Average document length? Language distribution?
- **CV:** What is the image resolution distribution? Class distribution? Are labels noisy?

**Data leakage is the most dangerous risk:**
A leaky feature that's available at training time but not at prediction time causes a model that looks great offline and fails in production. Always document the exact timestamp when each feature is generated vs. when the prediction is made.

---

## Section 9 — Technical Approach

**Algorithm selection guidance by domain:**

| Domain | Production default | Research SotA | When to use SotA |
|---|---|---|---|
| Tabular classification | XGBoost / LightGBM | FT-Transformer, TabNet | When linear relationships are weak and you have >100K rows |
| Tabular regression | XGBoost / LightGBM | Same | Same |
| NLP classification | Fine-tuned BERT family | GPT-4 / Claude few-shot | When labelled data is scarce |
| NLP generation | Fine-tuned LLM | Latest frontier model | When quality > cost |
| CV classification | ResNet / EfficientNet | ViT, DINO | When you have large labelled datasets |
| Recommendation | Matrix factorization / LightFM | Two-tower neural networks | When user/item features are rich |
| Time-series | Prophet / ARIMA | TFT, PatchTST | When complex seasonality and external regressors are needed |

**Tech stack selection principles:**
- Prefer boring technology for infrastructure (PostgreSQL over NoSQL if relational fits)
- Match the framework to team expertise, not hype
- Feature store is worth it when you have >3 models sharing features or real-time serving
- MLflow is the default experiment tracker unless the team is already on W&B

**Limitations section is mandatory.** Every approach has failure modes. Documenting them upfront enables building mitigations into MVP0/MVP1.

---

## Section 10 — Production Design

**Goal:** Design how the model lives in production before writing a line of code.

**Real-time vs. batch decision framework:**
| Factor | Lean real-time | Lean batch |
|---|---|---|
| User sees prediction instantly | ✅ Real-time | |
| Prediction informs background process | | ✅ Batch |
| Latency requirement < 500ms | ✅ Real-time | |
| Score refreshed daily is sufficient | | ✅ Batch |
| Complex feature computation needed | | ✅ Batch (pre-compute) |
| Cold-start users are common | ✅ Real-time (with fallback) | |

**Serving interface selection:**
- **REST API:** Default for most ML use cases. Use FastAPI.
- **gRPC:** When latency is critical and client controls both ends.
- **Tableau / BI dashboard:** When end users are analysts who can't use an API.
- **Kafka / streaming:** When predictions need to react to event streams in real time.
- **Scheduled report / email:** When output is periodic and consumed by non-technical users.

**Production scenarios to always design for:**
1. Model API is unavailable → fallback strategy
2. Feature pipeline is delayed → stale feature handling
3. Input data is malformed → validation + graceful degradation
4. Distribution shift detected → alert + model freeze protocol
5. Cold start (new entity, no history) → fallback rule or global average

---

## Section 11 — Project Scope

**Being explicit about what is OUT of scope is more important than what is in scope.**

Common scope creep patterns:
- "While we're at it, can we also..." → add to backlog, not current scope
- "This should be easy to add..." → often isn't; document it as a future enhancement
- "The stakeholder mentioned X in passing" → get written confirmation before adding

**Scope = what we commit to by MVP1.** Everything else is a future enhancement or a separate project.

---

## Section 12 — Literature Survey

**Run these searches before writing this section:**
1. `"[problem type] machine learning survey [year]"` — find a recent survey paper
2. `"[problem type] production system [company name]"` — find engineering blog posts
3. `"[problem type] benchmark dataset"` — find canonical evaluation datasets

**Good sources for production systems:**
- Netflix Tech Blog, Uber Engineering, Airbnb Engineering, LinkedIn Engineering
- Google Research Blog, Meta AI Blog, Amazon Science
- arXiv (cs.LG, stat.ML, cs.IR for RecSys)

**What to extract from each paper:**
1. What problem did they solve?
2. What dataset did they use?
3. What was the key contribution?
4. What metric did they report and what was the result?
5. What can we borrow for our approach?

---

## Section 13 — AI Recommendations

**Goal:** Don't just describe what you'll build — recommend how it could be significantly better.

**Sources for recommendations:**
- What does the Literature Survey say is the gap between current practice and SotA?
- What limitations in Section 9 have known mitigations?
- What data collected in MVP0/MVP1 enables more sophisticated approaches later?

**Good AI recommendation structure:**
- Specific: name the technique, not "use deep learning"
- Grounded: reference a paper or production system that proves it works
- Effort-honest: don't recommend something that would take 6 months when the team has 2 weeks
- Timed: tie it to a specific MVP level

**Examples of specific vs. vague recommendations:**
- Vague: "Use more features" → Specific: "Add user-level session embeddings from the event stream, following the approach in [Covington et al., 2016 YouTube DNN](URL)"
- Vague: "Try a neural network" → Specific: "Replace LightGBM with a two-tower neural network for the retrieval stage, following the architecture from [Yi et al., 2019 Sampling-Bias-Corrected](URL)"

---

## Section 14 — Agent Task Breakdown

**Goal:** Decompose the project into tasks small enough that each could be executed by a single AI agent or developer with minimal supervision.

**What makes a good agent task:**
- One clear deliverable (not "do the modeling" — that's a project, not a task)
- Specific inputs listed (exact files, schemas, or data)
- Specific outputs listed (what artifact is produced)
- Can be verified (the output can be checked without running the whole system)

**Agent types to assign:**
- **AI agent:** Fully automatable by Claude Code or similar — data cleaning, boilerplate code, report generation, test writing
- **Human (DS):** Requires judgment — feature selection, metric definition, result interpretation
- **Human (MLE):** Infra/deployment decisions, production SLA, scaling design
- **Human (PM/stakeholder):** Business decisions, scope calls, success criterion sign-off

**Parallelization note:** Identify which tasks can run simultaneously. This directly reduces clock time, even if total effort is the same.

---

## Section 15 — MVP Roadmap

**MVP0 is the most important level. Get it right.**

**MVP0 principles (from Dataiku's MVP framework):**
- Should prove the core hypothesis, not the full product
- Offline evaluation only — no production serving required
- Can be done in 1–4 weeks
- Uses the simplest algorithm that could plausibly work
- Success criterion is binary: "does the signal exist in the data?"

**Common MVP0 mistakes:**
- Including production serving (that's MVP1)
- Using the "right" algorithm instead of the fastest-to-validate algorithm
- Adding monitoring before you know the model works
- Building a UI before the model is validated

**MVP levels purpose:**
- MVP0: Prove the problem is solvable with this data
- MVP1: Prove it works in a real environment with real users
- MVP2: Prove it works at the required scale and reliability
- MVP3: Realize the strategic/competitive vision

**Max 4 MVPs.** If you need more, you're over-planning. Anything beyond MVP3 belongs in "Future Enhancements" in Section 10.
