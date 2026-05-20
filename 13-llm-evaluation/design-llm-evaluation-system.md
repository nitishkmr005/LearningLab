# Task-Agnostic Rubric-Driven LLM Evaluation System

*A production architecture for evaluating LLM outputs using LLM-as-judge — task-agnostic, provider-agnostic, and designed to survive model migrations.*

> **Companion document** to the [LLM Evaluation literature survey](./blog-llm-evaluation.md). Where the survey covers open-source frameworks (DeepEval, Promptfoo, LangSmith, EvidentlyAI) and academic methods (G-Eval, RAGAS, FActScoring), this document describes the architecture of a purpose-built evaluation system. The core bet: convert any LLM output into atomic, independently verifiable claims, then evaluate each claim separately with chain-of-thought reasoning and rubric-driven criteria.

---

## Table of Contents

**Background**

- [Introduction: Why Build Custom](#introduction-why-build-custom)
  - [Framework-by-Framework Gap Map](#framework-by-framework-gap-map)
  - [Universal Blind Spots Across All Frameworks](#universal-blind-spots-across-all-frameworks)
  - [What This Design Addresses](#what-this-design-addresses)
- [Scope: MVP Task Coverage](#scope-mvp-task-coverage)
  - [In Scope for MVP](#in-scope-for-mvp)
  - [Phase 2 (Post-MVP)](#phase-2-post-mvp)
  - [Out of Scope](#out-of-scope-current-architecture)
  - [Template Summarization: Decomposition Strategy](#template-based-summarization-decomposition-strategy)

**Core Data Model**

- [Terminology: Rubric, Acceptance Criteria, Evaluation Criteria, and Evaluation Steps](#terminology-rubric-acceptance-criteria-evaluation-criteria-and-evaluation-steps)

1. [Design Principles](#1-design-principles)
2. [The Validation Object](#2-the-validation-object-core-data-model)
   - 2.1 [Metric-Agnostic Design: Pluggable Custom Metrics](#21-metric-agnostic-design-pluggable-custom-metrics)
   - 2.2 [Metrics Catalog: All Supported Metrics by Task](#22-metrics-catalog-all-supported-metrics-by-task)

**System Architecture**

3. [Full System Architecture](#3-full-system-architecture)
4. [LLM-Agnostic Adapter Layer](#4-llm-agnostic-adapter-layer)
5. [YAML-Based Judge Configuration](#5-yaml-based-judge-configuration)

**Evaluation Pipeline**

6. [Evaluation Mode Router and Pipeline](#6-evaluation-mode-router-and-pipeline)
   - 6.0 [Evaluation Mode Router](#60-evaluation-mode-router)
   - 6.1 [Claim Decomposition Engine](#61-claim-decomposition-engine)
   - 6.2 [Full Pipeline: Step by Step](#62-full-pipeline-step-by-step)
   - 6.3 [Batching Strategy](#63-batching-strategy)
   - 6.4 [Holistic Path: Section-Level Evaluation](#64-holistic-path-section-level-evaluation)
7. [The Judge Prompt](#7-the-judge-prompt)
   - 7.1 [Modular Judge Prompt Design](#71-modular-judge-prompt-design)
   - 7.2 [Cascading Judge: Cost-Efficient Multi-Tier Evaluation](#72-cascading-judge-cost-efficient-multi-tier-evaluation)
8. [Field Type Strategy and Complexity Segmentation](#8-field-type-strategy-and-complexity-segmentation)

**Operations**

9.  [Evaluation Scenarios and Modes](#9-evaluation-scenarios-and-modes)
    - 9.1 [Validation Tool: Ground Truth Creation](#91-validation-tool-ground-truth-creation)
10. [Prompt Caching Strategy](#10-prompt-caching-strategy)
11. [Error Theme Extraction, Pattern Finder, and Human-in-the-Loop](#11-error-theme-extraction-pattern-finder-and-human-in-the-loop)
    - 11.1 [Error Theme → Prompt Improvement Pipeline](#111-error-theme--prompt-improvement-pipeline)
    - 11.2 [Prompt Advisor Prompt](#112-prompt-advisor-prompt)
    - 11.3 [Human-in-the-Loop: The Review Record](#113-human-in-the-loop-the-review-record)
    - 11.4 [Theme Assignment and Clustering](#114-theme-assignment-and-clustering-bridge-to-the-metrics-layer)
12. [Report Rollup and Benchmarking Dashboard](#12-report-rollup-and-benchmarking-dashboard)
    - 12.1 [Report Rollup Pipeline](#121-report-rollup-pipeline)
    - 12.2 [Dashboard: Five-Tab Structure](#122-dashboard-five-tab-structure)
13. [Observability](#13-observability)

**Assessment**

14. [Design Critique and Known Risks](#14-design-critique-and-known-risks)
15. [Known Limitations](#15-known-limitations)
16. [Production Deployment Checklist](#16-production-deployment-checklist)

---

## Introduction: Why Build Custom

Every major open-source framework and algorithmic approach solves one part of the evaluation problem. None solves all of it. This section maps the specific gaps each one leaves — drawn from GitHub issues, practitioner reports, and independent benchmarks (2024–2025) — to the design decisions in this document.

### Framework-by-Framework Gap Map

| Framework / Approach | Core Strength | Critical Limitation | Design Decision That Fills It |
|---|---|---|---|
| **DeepEval** | 50+ RAG metrics, pytest CI integration | OSS tier: no production monitoring, no shared dashboards, JSON output fragility (GitHub #1610, #1149), context scoring mean 0.46 vs 0.82–0.91 for peers | Production Sampler in Submission Layer; post-validation with `json_repair`; `ValidationObject` per claim replaces holistic scores |
| **Promptfoo** | Multi-model comparison, red-teaming, YAML CI | SQLite backend (explicitly "not production-ready"); ~11 tests/min throughput ceiling; no claim-level granularity; YAML complexity wall at scale | Async evaluation queue (Redis/SQS) designed for production volume; `ValidationObject.field_path` attributes every failure to a specific claim |
| **LangSmith** | Full LangChain trace capture, annotation queues, production observability | LangChain-only depth; per-trace cost escalation (thousands/month at scale); no custom rubrics in OSS; cloud-only (GDPR friction); no prompt improvement loop | LLM-agnostic adapters; YAML rubrics configurable by any team member; fully self-hosted; Error Theme → Prompt Advisor → HITL closes the loop |
| **EvidentlyAI** | Classical ML + LLM monitoring, HTML reports, descriptor pre-checks | No built-in faithfulness/context-precision/recall; no multi-model prompt comparison (unshipped as of 2025); observational only — no prompt improvement mechanism | Per-claim faithfulness via verbatim evidence post-validation; Dual Territory (cross-model) dashboard view; Prompt Advisor generates improvement suggestions |
| **G-Eval** | CoT + log-prob weighting, customizable criteria, reference-free | Position bias (2.5%–82.5% win-rate shift from ordering alone); verbosity bias; self-enhancement bias (87.76% self-preference); score clustering; prompt-sensitive results | Dual-ordering pairwise evaluation; length rubric + `output_tokens` tracking; cross-family judge config; binary verdict for extractive fields instead of clustered rubric scores |
| **RAGAS** | 8 RAG metrics, reference-free, lightweight, 5M+ evaluations/month | Faithfulness uses logical entailment (not factual correctness); opaque scores — no claim breakdown; RAG-only scope; no CI/CD or production path; cannot validate source quality | Per-claim binary verdict with verbatim evidence (not entailment); full claim breakdown in Data Export tab; task-agnostic pipeline handles non-RAG tasks; Production Sampler for live monitoring |
| **Prometheus** | Open-source 13B judge, GPT-4 parity on rubrics, 10× cheaper | Hard reference + rubric dependency; 0.39 faithfulness vs GPT-4's 0.93 on RAG (LlamaIndex benchmark); hallucinates feedback text; overly punitive scoring | Reference-free evaluation is the default; judge is any provider via adapter (not a fixed model); evidence post-validation catches hallucinated quotes |

### Universal Blind Spots Across All Frameworks

These gaps are not framework-specific — they are structural to the current generation of tools:

1. **No claim-level failure attribution**: All frameworks evaluate at the response level. None attributes a failure to a specific field, sentence, or claim within the output. This is why prompt debugging is slow — you know the score fell, but not why.

2. **No closed-loop prompt improvement**: Evaluation is observational. Failures appear on dashboards or in test reports, but no framework provides a mechanism that takes those failures, identifies the root cause in the prompt, generates a specific fix, and routes it through a governed review before deployment.

3. **No cross-provider judge with bias mitigations built in**: Every framework either ties the judge to a specific provider or treats provider-switching as a manual configuration task. Position bias, verbosity bias, and self-enhancement bias are documented but not architecturally mitigated.

4. **Production monitoring requires a commercial tier**: In every major OSS framework, production traffic evaluation — sampling, drift detection, alerting — requires either a paid cloud platform or substantial custom infrastructure. The OSS layer is development-only.

5. **Rubric changes silently invalidate historical scores**: No framework versions rubrics and stores the rubric version on every evaluation record. As a result, a rubric edit makes historical score comparisons meaningless without re-running everything.

### What This Design Addresses

This architecture was built specifically to close all five blind spots:

- **Claim-level attribution**: `ValidationObject` per `field_path` is the atomic unit. Every verdict, error theme, and evidence quote is at the claim level.
- **Closed-loop improvement**: Error Theme Aggregator → Pattern Finder → Prompt Advisor → Human-in-the-Loop governance → versioned prompt update. The loop is designed into the architecture, not bolted on.
- **Judge bias mitigations**: Dual-ordering pairwise evaluation, cross-family judge configuration, explicit length rubric, `output_tokens` tracking in the 9-metric leaderboard.
- **Production-first**: Production Sampler is a first-class submission mode. The async evaluation queue (Redis/SQS) and PostgreSQL result store are designed for production volume, not augmented from a dev tool.
- **Rubric versioning**: `rubric_version` on every `ValidationObject`. Rollup Tester validates taxonomy integrity before writing metrics. Dual Territory dashboard compares two rubric versions on identical inputs.

---

## Scope: MVP Task Coverage

This evaluation system is built for **LLM outputs that process input source documents** — extracting, summarizing, or classifying information from a document or conversation. The architecture is task-agnostic within this scope; agents, multi-step workflows, and chatbots are explicitly deferred to a future phase.

### In Scope for MVP

| Task Type | Description | Claim Decomposition Strategy | Decomposition Method |
|---|---|---|---|
| **JSON Extraction** | LLM extracts structured fields from a document (e.g., call transcript → contract fields) | One `ValidationObject` per JSON field path | Deterministic — parse JSON keys |
| **Template-Based Summarization** | LLM fills a defined template with sections (e.g., executive summary, key decisions, next steps) | Convert template to JSON; one `ValidationObject` per section field | Deterministic — map section names to JSON keys |
| **Classification** | LLM assigns a label or category to input (e.g., sentiment, intent, risk flag) | One `ValidationObject` per classification field; `allowed_labels` validated per YAML config | Deterministic — parse label from output |

### Phase 2 (Post-MVP)

The following task types require the **LLM-based claim decomposition engine** (§6 Claim Decomposer). They are architecturally supported but deferred until the pipeline is calibrated on MVP tasks:

| Task Type | Why Deferred | Phase 2 Approach |
|---|---|---|
| **Free-Form Summarization** | Claim count varies; compound claims produce ambiguous binary verdicts | Dedicated cheap model (Haiku) splits into ≤10 atomic claims; 3-rubric per claim (accuracy + completeness + hallucination) |
| **Free-Form Text Generation** | Same decomposition fragility as free-form summarization | Same as above; rubric focuses on instruction-following + source grounding |

### Out of Scope (Current Architecture)

- **Agentic workflows** — multi-step tool use, planning loops, ReAct-style agents
- **LLM-to-LLM pipelines** — chained calls where the output of one LLM is the input to another
- **Chatbot / dialogue evaluation** — multi-turn conversation quality
- **RAG pipeline evaluation** — retrieval quality, context relevance, faithfulness at retrieval layer

These require extending the Claim Decomposition Engine with new strategies; the core judge pipeline does not change.

### Template-Based Summarization: Decomposition Strategy

Template-based summarization converts directly to JSON extraction — no LLM decomposition needed:

```
Template-Based Summarization (MVP)
────────────────────────────────────
LLM fills a template:
  Section 1: Executive Summary
  Section 2: Key Decisions
  Section 3: Action Items

→ Decompose deterministically as JSON:
  { "executive_summary": "...", "key_decisions": "...", "action_items": "..." }

→ Evaluate each section as its own free_form ValidationObject.
  Rubric per section defined in YAML: completeness, accuracy, source grounding.
  This reduces template summarization to the same evaluation path as JSON extraction.
```

Free-form summarization (no template) is deferred to Phase 2. It requires the LLM-based claim decomposer (§6) which is calibrated after the MVP pipeline is stable on structured tasks.

---

## 1. Design Principles

Before diving into components, the non-negotiable constraints that drive every decision:

1. **Task-agnostic**: The same pipeline must handle JSON extraction, template summarization, classification, and constrained text generation — all tasks where an LLM processes a source document to produce a deterministic output. The architecture is extensible to RAG and agentic workflows in future phases without changes to the core judge pipeline.
2. **LLM-agnostic**: The judge and the system under test can be any provider — OpenAI, Anthropic, Enterprise Gateway (Azure AI, AWS Bedrock), OLLAMA, llama-cpp. No vendor lock-in at any layer.
3. **Claim-level granularity**: Evaluation happens at the level of atomic claims, not at the level of the full response. A "score of 3.4 overall" is not actionable. "Claim 7 is incorrect because the context states Q3 revenue was $4.2M but the extraction says $4.8M" is.
4. **Rubric-driven, not metric-driven**: Users define rubrics in YAML (per field, per section, or globally). The system generates evaluation steps from those rubrics. New criteria (harmfulness, instruction-following, tone) are plugins, not code changes.
5. **Async by default**: Never in the request path. Evaluation is a background concern.
6. **Prompt caching everywhere**: The most expensive repeated computation in this system is the judge LLM call. Evaluation steps generated from rubrics, system prompts, and static context should be cached aggressively.
7. **Schema evolution support**: When prompt definitions or output schema structures change, the judge evaluates whether the new schema improves extraction quality before rollout — blocking schema regressions the same way a CI gate blocks code regressions.
8. **Human-in-the-loop governance for prompt changes**: Prompt Advisor suggestions are never deployed automatically. Every suggested change passes through an Accept / Edit / Reject review with a rationale summary and supporting evidence records, so prompt evolution is auditable and deliberate.

---

## Terminology: Rubric, Acceptance Criteria, Evaluation Criteria, and Evaluation Steps

These four terms appear throughout this document and are often conflated. The distinction matters because each lives in a different layer of the system and produces a different output.

| Term | What it is | Where it lives | Produces |
|---|---|---|---|
| **Rubric** | The overall scoring framework for a field — what dimensions matter and what the field is trying to measure | YAML config, per field/section | Nothing directly — it's the spec |
| **Acceptance Criteria** | The specific binary pass/fail conditions — what the extracted value must satisfy to be `is_correct=True` | Inside the rubric: `criteria` + `failure_condition` + `example_pass` + `example_fail` | `is_correct` verdict |
| **Evaluation Criteria** | The graded quality dimensions the judge scores during reasoning — coherence, faithfulness, readability, etc. | Inside the rubric: `rubric_scales` dict (named dimensions with 0–1 anchors) | `rubric_scores: dict[str, float]` in ValidationObject |
| **Evaluation Steps** | The operationalized, numbered reasoning sequence the judge executes to apply both acceptance and evaluation criteria | Generated by a cached LLM call from the full rubric spec; injected into every judge prompt | `evaluation_steps: list[str]` in ValidationObject |

**In plain terms:**
- The **rubric** is the rulebook for a field.
- **Acceptance criteria** are the specific rules that determine whether the answer is binary correct or incorrect.
- **Evaluation criteria** are the graded quality dimensions — not binary, but scored 0–1 on named axes.
- **Evaluation steps** are the concrete reasoning actions the judge takes to check both — generated once, cached, then reused on every judge call for that field.

---

#### What Evaluation Steps Are Generated From

Evaluation steps are **not** generated from acceptance criteria alone, nor from evaluation criteria alone. They are generated from the **full rubric specification** for a field — the LLM step-generator reads everything and produces an ordered reasoning sequence that covers both binary verdict logic and any graded scoring:

```
Input to STEP_GENERATION_PROMPT:
  - field_path        (which field)
  - field_type        (extractive / inferential / classification / free_form)
  - criteria          (acceptance criteria — the pass/fail rule)
  - failure_condition (specific edge case that triggers is_correct=False)
  - rubric            (global rubric for the task — context and intent)
  - rubric_scales     (evaluation criteria — graded dimensions if any)
  - error_taxonomy    (standard error themes for classification)

Output: numbered evaluation steps, e.g.:
  1. Locate the contract value in the source context.
  2. Identify whether the stated period is annual or monthly.
  3. Apply the failure condition: do not multiply if monthly — extract as-is.
  4. Check the extracted value against the stated amount.
  5. If rubric_scales are defined, score completeness and unit_clarity on 0–1 scale.
  6. Assign is_correct and, if wrong, wrap the correct value in <corrected> tags.
```

These steps are **cached per `(criteria_hash, rubric_version)`** — so a rubric change invalidates the cache and regenerates the steps, but the same rubric reuses the cached steps across thousands of documents. The judge receives these steps verbatim and must execute them in order before producing a verdict. This is the mechanism that forces chain-of-thought reasoning and makes the judge's reasoning auditable.

---

#### Concrete Example — field `contract.annual_value`

```
Rubric (the rulebook — global context):
  "Contract fields require exact figures. Unit normalization is an error."

Acceptance Criteria (binary pass/fail):
  criteria:          "Total annual contract value in USD."
  failure_condition: "If the stated period is monthly, do not multiply."
  example_pass:      "4000  (when transcript states $4,000/month — extracted as-is)"
  example_fail:      "48000 (monthly $4,000 multiplied by 12 — forbidden)"
  → produces: is_correct = false

Evaluation Criteria (graded dimensions — optional for this field):
  rubric_scales:
    unit_clarity:
      0.0: "Amount stated with no period context"
      0.5: "Period mentioned but ambiguous"
      1.0: "Period and amount stated unambiguously"
  → produces: rubric_scores = {"unit_clarity": 0.5}

Evaluation Steps (generated from everything above, cached):
  Step 1: Find contract value, annual fee, or pricing in the transcript.
  Step 2: Identify the stated period (annual / monthly / other).
  Step 3: Apply failure condition — if monthly, do not multiply; extract as-is.
  Step 4: Compare extracted value to actual_value.
  Step 5: Score unit_clarity on the 0–1 rubric scale.
  Step 6: Set is_correct; if false, provide corrected_value in <corrected> tags.
  → produces: evaluation_steps[] in the ValidationObject
```

---

#### Where Quality Metrics Like Faithfulness, Coherence, Toxicity, Readability Go

These standard NLP quality metrics map directly onto the terminology:

| Metric | Binary or Graded | Maps to | Field types where it applies |
|---|---|---|---|
| **Faithfulness** | Both | Binary: `is_correct` (any unfaithful claim → fail). Graded: `rubric_scores["faithfulness"]` | Inferential, free_form, template sections |
| **Coherence** | Graded | `rubric_scales: coherence` → `rubric_scores["coherence"]` | Free_form, template sections |
| **Toxicity** | Binary | Acceptance criteria `failure_condition: "output contains harmful content"` → `is_correct=False` | Any field type, any task |
| **Readability** | Graded | `rubric_scales: readability` → `rubric_scores["readability"]` | Free_form, template sections |
| **Completeness** | Graded | `rubric_scales: completeness` → `rubric_scores["completeness"]` | Free_form, template, extractive |
| **Relevance** | Graded | `rubric_scales: relevance` → `rubric_scores["relevance"]` | Free_form, inferential |

The key distinction: **binary quality signals** (does the output contain toxicity? is the claim faithful at all?) belong in **acceptance criteria** and produce `is_correct`. **Graded quality dimensions** (how coherent? how readable? how faithful on a scale?) belong in **evaluation criteria** as `rubric_scales` and produce `rubric_scores` entries.

**In our design, faithfulness is handled as both:**
- **Binary**: Any extractive or inferential claim that cannot be grounded in the source context is `is_correct=False`, with `reason_for_incorrect="source_document_does_not_contain_this_information"` — the evidence list is empty, which is the signal.
- **Graded**: For free_form and template sections, `rubric_scales` can include a `faithfulness` dimension scored 0–1. This appears in `rubric_scores["faithfulness"]` and feeds the leaderboard faithfulness metric.

Toxicity is a special case — it is defined as a failure condition in the rubric, but the field type is still whatever the field normally is. The judge checks it as part of the evaluation steps and sets `is_correct=False` with `error_theme="toxicity"` when triggered.

---

#### Scoring Granularity: Why Binary for Verdicts and 0–1 Float (Not 1–5) for Scores

The scoring granularity choice is constrained by the fact that this system evaluates **atomic claims**, not holistic responses.

**`is_correct` and `is_missing` are boolean, not float — because atomic claims have no partial state.**

An atomic claim is a single, independently verifiable assertion: "The renewal date is March 15th." That date is either in the source document or it isn't. There is no 0.4 version of a date. Converting `is_correct` to a float (0.0/0.4/1.0) would imply a meaningful partial state that doesn't exist at the atomic level — and would break precision/recall arithmetic, which requires binary verdicts.

**`rubric_scores` uses 0–1 float with three anchors — not a 1–5 Likert scale.**

1–5 scales are designed for **holistic human evaluation of full responses**. G-Eval uses them, and the research on that approach shows a consistent problem: score clustering. Judges gravitate to 3 and 5; values of 1, 2, and 4 are rarely used. In practice, a 1–5 scale collapses to 3 effective levels.

For atomic claim evaluation, those 3 levels map naturally to `0.0 / 0.5 / 1.0`:

```
0.0  →  absent / completely wrong / not captured
0.5  →  partial / present but incomplete / ambiguous
1.0  →  complete / fully and accurately captured
```

Three anchors is the right granularity for an atomic claim. More distinctions don't add information — they add noise and inter-rater disagreement. A claim either captures the key fact (1.0), partially captures it (0.5), or misses it (0.0). There is no meaningful difference between "3 out of 5" and "4 out of 5" at this level.

**0–1 float also aggregates cleanly.** You average directly across claims and documents to get a field-level or run-level score. A 1–5 average requires normalization before it's interpretable. Thresholds are intuitive: `> 0.8` means good, `< 0.5` means failing — no conversion needed.

**Holistic metrics belong at the document level, not per-claim.**

Metrics like overall readability or document-level coherence are not meaningful on an atomic claim — they measure properties of the whole output. These belong as **document-level metrics**, computed once after all claims are evaluated and stored separately from the per-claim `ValidationObject`. Putting them inside `rubric_scores` on each claim would be the wrong granularity — you'd be scoring readability on "The renewal date is March 15th" individually, which doesn't make sense.

The boundary is:
```
Per-claim (ValidationObject.rubric_scores):
  completeness, faithfulness, unit_clarity — claim-level dimensions

Document-level (EvalResult aggregate):
  readability, coherence, overall_quality — whole-output dimensions
```

**Summary table:**

| What | Scale | Why |
|---|---|---|
| `is_correct` | `bool` (binary) | Atomic claim: correct or not — no partial state |
| `is_missing` | `bool` (binary) | Present or absent — no partial state |
| `rubric_scores` | `float`, 0–1, anchors at 0.0/0.5/1.0 | Three meaningful states per claim; cleaner aggregation than 1–5 |
| Document-level holistic metrics | Separate aggregate, not in ValidationObject | Not meaningful per-claim; measure whole-output properties |

---

**Why `is_correct` and `is_missing` stay in the ValidationObject, not the rubric:**

The rubric contains the *definition* of what correct means (acceptance criteria). The ValidationObject stores the *verdict* — the result of applying that definition to a specific extraction. Moving verdicts into the rubric would conflate the scoring key with the score sheet. The analogy: a grading rubric defines what an A means; it doesn't record which student got an A. The grade sheet does that.

`missing_rule` in the YAML is the one exception — it is a rubric-level rule (`"if no account number stated, is_missing=true, severity=completely"`). The rule lives in the rubric; the verdict (`is_missing=true`) lives in the ValidationObject.

---

## 2. The Validation Object: Core Data Model

The fundamental unit of this system is the **Validation Object** — one per field in the JSON being evaluated, or one per atomic claim extracted from free-form text. Every downstream feature (error dashboards, prompt advisor, model comparisons) is built on aggregations of these objects.

Two design decisions distinguish this model from most open-source frameworks:

1. **Two separate boolean flags** instead of a combined status enum. `is_correct` and `is_missing` are independent: a field can be present-but-wrong (`is_correct=False, is_missing=False`), missing entirely (`is_correct=False, is_missing=True`), or correct (`is_correct=True, is_missing=False`). This separation lets the dashboard filter and aggregate along two orthogonal failure axes.

2. **`corrected_value` with `<corrected>` tags** in the raw judge output. When `is_correct=False`, the judge is asked to provide the correct value wrapped in `<corrected>` XML tags. This is post-processed into a `corrected_value` field — so the Data Export tab shows both what the LLM said and what it should have said, side by side.

```python
from typing import Optional
from pydantic import BaseModel

class ValidationObject(BaseModel):
    # --- Identity ---
    field_path: str            # e.g. "customer.account_number" or "summary.claim[3]"
    field_type: str            # "extractive" | "inferential" | "free_form" | "classification"

    # --- Values ---
    actual_value: Optional[str]    # What the LLM extracted/generated (always the raw LLM output)
    corrected_value: Optional[str] # Correct value per judge; populated only when is_correct=False.
                                   # Parsed from <corrected>...</corrected> in judge's raw output.

    # --- Evaluation reasoning ---
    evaluation_steps: list[str]    # Chain-of-thought steps executed by judge
    evidence: list[str]            # Verbatim quotes from source context.
                                   # Plain strings only — no location or polarity metadata.
                                   # Post-validation: substring-check every quote against context.

    # --- Verdict: two independent flags ---
    is_correct: bool               # False if value is wrong or hallucinated
    is_missing: bool               # True if the value is absent (partially or completely)
    missing_severity: Optional[str]  # "partially" | "completely" — only set when is_missing=True

    # --- Error classification ---
    reason_for_incorrect: Optional[str]  # Cites which rubric definition failed, OR "hallucination",
                                          # OR "source_document_does_not_contain_this_information"
    error_theme: Optional[str]           # From user-supplied taxonomy; null when is_correct=True

    # --- Custom metrics (metric-agnostic extension point) ---
    rubric_scores: dict[str, float]  # Named scores for custom metrics, e.g.:
                                     # {"toxicity": 0.05, "coherence": 0.88}
                                     # Empty dict {} for binary extractive/inferential fields.
                                     # Metric names declared in YAML rubric_scale blocks.
                                     # See §2.1 for pluggable metric design.

    # --- Metadata ---
    rubric_version: str
    judge_model: str
    latency_ms: int
    cached: bool                   # Was this result served from cache?
    eval_version: str              # Composite attribution key: "rubric={rubric_version},prompt={prompt_version}"
                                   # Never change rubric and prompt in the same evaluation run —
                                   # use this key to filter dashboard comparisons to a fixed evaluation context.
```

**Example 1: Correct extraction**

```json
{
  "field_path": "contract.renewal_date",
  "field_type": "extractive",
  "actual_value": "2025-03-15",
  "corrected_value": null,

  "evaluation_steps": [
    "1. Search the transcript for any mention of contract renewal or renewal date.",
    "2. Identify the exact date stated by agent or customer.",
    "3. Verify the extracted value matches that date (format-normalized).",
    "4. If no date mentioned, set is_missing=true, missing_severity='completely'.",
    "5. If a range stated but single date extracted, set is_missing=true, missing_severity='partially'."
  ],
  "evidence": [
    "Agent: 'Your renewal comes up on March 15th of next year.'"
  ],

  "is_correct": true,
  "is_missing": false,
  "missing_severity": null,

  "reason_for_incorrect": null,
  "error_theme": null,
  "rubric_scores": {},

  "rubric_version": "contracts-v2.1",
  "judge_model": "claude-sonnet-4-6",
  "latency_ms": 820,
  "cached": false,
  "eval_version": "rubric=contracts-v2.1,prompt=v3.4"
}
```

**Example 2: Wrong value (unit confusion)**

```json
{
  "field_path": "contract.annual_value",
  "field_type": "extractive",
  "actual_value": "48000",
  "corrected_value": "4000",

  "evaluation_steps": [
    "1. Search transcript for contract value, annual fee, or pricing.",
    "2. Identify stated amount and currency.",
    "3. Check rubric: 'Total annual contract value in USD. If stated as monthly, do not multiply.'",
    "4. The extraction of $48,000 fails rubric: context states $4,000/month, not $48,000/year."
  ],
  "evidence": [
    "Customer: 'We're paying four thousand a month under the current deal.'"
  ],

  "is_correct": false,
  "is_missing": false,
  "missing_severity": null,

  "reason_for_incorrect": "Rubric 'contract.annual_value': 'If stated as monthly, do not multiply — mark as unit_ambiguous.' The model multiplied the monthly figure ($4,000) by 12 to produce $48,000.",
  "error_theme": "unit_confusion",
  "rubric_scores": {},

  "rubric_version": "contracts-v2.1",
  "judge_model": "claude-sonnet-4-6",
  "latency_ms": 1140,
  "cached": false,
  "eval_version": "rubric=contracts-v2.1,prompt=v3.4"
}
```

**Example 3: Missing value**

```json
{
  "field_path": "customer.account_number",
  "field_type": "extractive",
  "actual_value": null,
  "corrected_value": null,

  "evaluation_steps": [
    "1. Search transcript for account number stated by agent or customer.",
    "2. No account number found in transcript.",
    "3. Rubric: 'If no account number stated, set is_missing=true, missing_severity=completely.'"
  ],
  "evidence": [],   // empty — source document contains no account number

  "is_correct": false,
  "is_missing": true,
  "missing_severity": "completely",

  "reason_for_incorrect": "source_document_does_not_contain_this_information",
  "error_theme": "missing_from_source",
  "rubric_scores": {},

  "rubric_version": "contracts-v2.1",
  "judge_model": "claude-sonnet-4-6",
  "latency_ms": 610,
  "cached": false,
  "eval_version": "rubric=contracts-v2.1,prompt=v3.4"
}
```

> The `reason_for_incorrect` field always cites one of three root causes: (1) a specific rubric definition that failed (quoted verbatim), (2) the string `"hallucination"` if the value is not found anywhere in the context, or (3) `"source_document_does_not_contain_this_information"` if the field was expected but absent. This makes the error reason machine-parseable, not just human-readable.

---

### 2.1 Metric-Agnostic Design: Pluggable Custom Metrics

The default metrics — `is_correct` and `is_missing` — cover extraction quality. But real deployments also need metrics like toxicity, coherence, verbosity, or domain-specific criteria (e.g., "does the summary mention a resolution?").

The `rubric_scores: dict[str, float]` field is the extension point. It is an empty dict `{}` for binary extractive and inferential fields. For custom metrics, the YAML config declares `judge_strategy: rubric_scoring` with a `rubric_scale` per named metric, and the judge outputs a named dict instead of a single float. This means any number of custom metrics can coexist on the same field — no extra ValidationObject records, no separate data model.

```yaml
# In judge-configs/call-summary-v1.yaml
#
# rubric_scales uses a 0–1 float with three anchors: 0.0 / 0.5 / 1.0.
# This is intentional — not a 1–5 Likert scale.
# Atomic claims have three meaningful states: absent, partial, complete.
# 1–5 scales produce score clustering at 3 and 5 (G-Eval research); the extra
# distinctions don't add information at the per-claim level and complicate aggregation.

sections:
  summary:
    fields:
      resolution_mention:
        field_type: free_form
        judge_strategy: rubric_scoring
        criteria: "Does the summary clearly state whether the customer's issue was resolved?"
        rubric_scales:
          resolution:        # metric name → appears as rubric_scores["resolution"]
            0.0: "No mention of resolution or outcome"
            0.5: "Resolution implied but not stated explicitly"
            1.0: "Resolution clearly stated with outcome"
          completeness:      # second metric on the same field
            0.0: "Key facts missing from summary"
            0.5: "Most facts present, one or two gaps"
            1.0: "All key facts accurately captured"

      toxicity_check:
        field_type: free_form
        judge_strategy: rubric_scoring
        # Toxicity is binary — no partial state, so no 0.5 anchor.
        # But it lives in rubric_scores (not is_correct) because it's an additive
        # quality dimension, not a verdict on whether the field was extracted correctly.
        criteria: "Flag if the generated text contains harmful, biased, or inappropriate language."
        rubric_scales:
          toxicity:
            0.0: "No harmful or biased content detected"
            1.0: "Contains harmful, biased, or inappropriate content"
```

Judge output for a field with custom metrics:
```json
{
  "field_path": "summary.resolution_mention",
  "is_correct": true,
  "is_missing": false,
  "rubric_scores": {
    "resolution": 0.9,
    "completeness": 0.75
  }
}
```

The Metrics Calculator aggregates each key in `rubric_scores` independently — `avg(rubric_scores["resolution"])` across all runs, `avg(rubric_scores["completeness"])`, etc. These appear as separate rows in the dashboard. You can track resolution coverage, coherence score, or toxicity rate as first-class metrics across model versions — no pipeline code changes, only YAML additions.

---

### 2.2 Metrics Catalog: All Supported Metrics by Task

Every metric from the major evaluation frameworks maps onto one of four evaluation approaches in this system:

| Approach | Column label | How it works |
|---|---|---|
| `claim_level › is_correct` | Binary verdict | Acceptance criteria → `is_correct` bool per claim |
| `claim_level › rubric_scores` | Scored per claim | `rubric_scales` → `rubric_scores[name]` float (0.0/0.5/1.0) |
| `holistic › rubric_scores` | Scored per section | `HolisticJudge` → `HolisticResult.rubric_scores[name]` |
| `aggregate` | Computed post-evaluation | Derived from ValidationObject stats; no judge call |
| `instrumental` | Measured outside judge | Token counts, latency, cost — system metadata |

The tables below list every metric this system can produce, grouped by task. The "Source" column names the framework(s) that defined or popularised the metric. "Needs reference" means a ground-truth answer is required.

---

#### Universal — All Tasks

| Metric | Source | Description | Approach | Needs reference |
|---|---|---|---|---|
| **Toxicity** | DeepEval, EvidentlyAI, Promptfoo | Presence of harmful, offensive, or biased content in the output | `claim_level › is_correct` (acceptance criteria `failure_condition`; `error_theme="toxicity"`) or `holistic › rubric_scores["toxicity"]` | No |
| **Bias** | DeepEval, Prometheus | Unfair stereotyping or differential treatment of demographic groups | `holistic › rubric_scores["bias"]` (0.0=overt bias, 0.5=subtle, 1.0=none) | No |
| **Instruction Following** | Prometheus, G-Eval | Degree to which output satisfies each stated instruction | `claim_level › is_correct` — one claim per instruction; binary pass/fail per constraint | No |
| **Fluency** | G-Eval, SummEval | Grammar, naturalness, and sentence-level quality | `holistic › rubric_scores["fluency"]` | No |
| **Readability** | EvidentlyAI, Prometheus | Ease of reading; clarity and accessibility of phrasing | `holistic › rubric_scores["readability"]` | No |
| **Verbosity** | EvidentlyAI, Custom | Whether output length is appropriate — not too long, not too short | `holistic › rubric_scores["verbosity"]` | No |
| **PII Leakage** | Custom, LangSmith | Output contains personally identifiable information beyond what is in the input | `claim_level › is_correct` — failure_condition on PII patterns; `error_theme="pii_leakage"` | No |
| **Format Compliance** | DeepEval (JSON Correctness) | Output matches required schema, format, or structural constraints | `claim_level › is_correct` — deterministic schema validation before judge call | No |
| **Hallucination Rate** | DeepEval, RAGAS, FActScoring | Fraction of claims that introduce information absent from the source | `aggregate` — `is_correct=False` with `reason_for_incorrect="source_document_does_not_contain_this_information"` rate | No |
| **Latency (p50/p95)** | Promptfoo, DeepEval | Time-to-first-token and total generation time | `instrumental` — `ValidationObject.latency_ms`; p50/p95 computed in dashboard | No |
| **Token Count** | Promptfoo, DeepEval | Input + output token usage per call | `instrumental` — provider metadata stored in `RunMetadata` | No |
| **Cost per 1k evaluations** | Promptfoo, LangSmith | Dollar cost normalized to 1,000 evaluation calls | `instrumental` — token count × provider price; computed in report rollup | No |
| **Judge Agreement Rate** | Custom | Rate at which Tier 1 and Tier 2 judges agree (escalation signal quality) | `aggregate` — compare `escalated=True` verdicts vs. Tier 1 verdicts on same claims | No |
| **Cache Hit Rate** | Custom | Fraction of judge calls served from Redis or provider prefix cache | `instrumental` — `ValidationObject.cached` flag rate | No |

---

#### JSON Extraction / Structured Output

| Metric | Source | Description | Approach | Needs reference |
|---|---|---|---|---|
| **Field Precision** | RAGAS (adapted), Custom | Fraction of extracted fields that are correct | `aggregate` — `is_correct=True` count / total fields across ValidationObjects | Optional |
| **Field Recall** | RAGAS (adapted), Custom | Fraction of expected fields that were extracted (not missing) | `aggregate` — `(1 - is_missing_rate)` across fields | Yes — expected field list |
| **Field F1** | Standard | Harmonic mean of field precision and recall | `aggregate` — computed from precision and recall | Yes |
| **Field Correctness** | Custom | Binary: is the extracted value correct per rubric criteria | `claim_level › is_correct` — one ValidationObject per field | No |
| **Missing Field Rate** | Custom | Fraction of expected fields entirely absent from output | `aggregate` — `is_missing=True` rate; `missing_severity="completely"` | No |
| **Partial Extraction Rate** | Custom | Fraction of fields present but incomplete | `aggregate` — `is_missing=True, missing_severity="partially"` rate | No |
| **Format Compliance** | DeepEval | Extracted values match expected format (e.g., ISO date, 10-digit string) | `claim_level › is_correct` — `expected_format` in rubric; failure_condition on format | No |
| **Unit Normalization Error** | Custom | Incorrect unit conversion (e.g., monthly figure annualized) | `claim_level › is_correct` — explicit `failure_condition`; `error_theme="unit_ambiguous"` | No |
| **Hallucination** | DeepEval, FActScoring | Extracted value not present in source document | `claim_level › is_correct` — `evidence=[]` + `is_correct=False` + `reason_for_incorrect="source_document_does_not_contain_this_information"` | No |
| **Entity Coverage** | FActScoring (adapted) | Fraction of named entities from source captured correctly | `claim_level › is_correct` — per-entity field; `is_missing` for absent entities | Yes |
| **Completeness** | Custom | How much of the required structured output was populated | `claim_level › rubric_scores["completeness"]` or `aggregate` from is_missing counts | No |

---

#### Summarization — Template (section-by-section)

| Metric | Source | Description | Approach | Needs reference |
|---|---|---|---|---|
| **Faithfulness** | RAGAS, DeepEval, G-Eval (Consistency) | Every claim in the summary is supported by the source document | `claim_level › is_correct` — verbatim evidence post-validates each claim | No |
| **Coverage / Completeness** | SummEval, DeepEval | Key facts from the source are captured in the summary | `claim_level › rubric_scores["completeness"]` per section | Yes |
| **Factual Consistency** | G-Eval, RAGAS | No claim in the summary contradicts the source | `claim_level › is_correct` — `is_correct=False` when claim contradicts source evidence | No |
| **Hallucination** | DeepEval, FActScoring | Claims present in summary with no source grounding | `claim_level › is_correct` — empty `evidence[]` + failed verdict | No |
| **Coherence** | G-Eval, SummEval | Logical flow and internal consistency across the whole section | `holistic › rubric_scores["coherence"]` | No |
| **Relevance** | G-Eval, RAGAS | Important content selected; irrelevant details excluded | `holistic › rubric_scores["relevance"]` | No |
| **Conciseness / Verbosity** | EvidentlyAI, SummEval | Summary length is appropriate — no padding, no omissions | `holistic › rubric_scores["verbosity"]` | No |
| **Redundancy** | SummEval | Absence of repeated information within the summary | `holistic › rubric_scores["redundancy"]` (0.0=highly redundant, 1.0=none) | No |
| **Resolution Mention** | Custom (call centre) | Summary explicitly states whether the customer issue was resolved | `claim_level › rubric_scores["resolution"]` | No |
| **Precision (reference-based)** | ROUGE / BLEU (adapted) | N-gram or semantic overlap of generated summary with reference | `claim_level › is_correct` with reference provided | Yes |
| **Recall (reference-based)** | ROUGE / BLEU (adapted) | Fraction of reference content present in generated summary | `aggregate` with reference; `is_missing` per expected claim | Yes |

---

#### Summarization — Free-form

Inherits all template summarization metrics above, plus:

| Metric | Source | Description | Approach | Needs reference |
|---|---|---|---|---|
| **Abstractiveness** | SummEval | Degree of paraphrasing vs. verbatim copy from source | `holistic › rubric_scores["abstractiveness"]` (0.0=fully extractive, 1.0=fully abstract) | No |
| **Density** | SummEval | Ratio of extractive fragments to total text length | `holistic › rubric_scores["density"]` | No |
| **Fluency** | G-Eval, SummEval | Grammar and naturalness at sentence level | `holistic › rubric_scores["fluency"]` | No |
| **Claim Decomposition Coverage** | FActScoring | Fraction of atomic facts in source covered by generated summary | `claim_level › is_missing` — Phase 2 LLM decomposer on free-form output | Yes |

---

#### Classification

| Metric | Source | Description | Approach | Needs reference |
|---|---|---|---|---|
| **Label Accuracy** | Standard, DeepEval | Assigned label matches the correct label | `claim_level › is_correct` — one ValidationObject per classification field | Yes (ground truth label) |
| **Label Precision (per label)** | Standard | True positives / (TP + FP) for each allowed label | `aggregate` — computed from `is_correct` across ValidationObjects filtered by label | Yes |
| **Label Recall (per label)** | Standard | True positives / (TP + FN) for each allowed label | `aggregate` — computed from `is_correct` + `is_missing` per label | Yes |
| **Label F1 (per label)** | Standard | Harmonic mean of per-label precision and recall | `aggregate` — derived from precision + recall | Yes |
| **Out-of-Label Detection** | Custom | Model assigned a label not in `allowed_labels` | `claim_level › is_correct` — deterministic: `actual_value ∉ allowed_labels` → `is_correct=False` | No |
| **Missing Label Rate** | Custom | Label was not assigned when one was required | `aggregate` — `is_missing=True` rate across classification fields | No |
| **Label Confidence Calibration** | Custom | Judge's stated confidence correlates with observed accuracy | `aggregate` — compare `confidence` field to `is_correct` rate per confidence bucket | No |
| **Multi-label Subset Accuracy** | Custom | All correct labels assigned, no incorrect labels (for `multi_label: true` fields) | `claim_level › is_correct` — binary: exact set match | Yes |

---

#### RAG / Question Answering

| Metric | Source | Description | Approach | Needs reference |
|---|---|---|---|---|
| **Faithfulness** | RAGAS | Fraction of answer claims supported by retrieved context (not hallucinated) | `claim_level › is_correct` — evidence from retrieved chunks; empty evidence = hallucination | No |
| **Answer Relevance** | RAGAS, DeepEval | How directly the answer addresses the question asked | `holistic › rubric_scores["answer_relevance"]` | No |
| **Context Precision** | RAGAS | Fraction of retrieved chunks that are relevant to the query | `claim_level › is_correct` — one claim per chunk; is the chunk relevant? | Yes |
| **Context Recall** | RAGAS | Fraction of ground-truth information present across retrieved chunks | `claim_level › is_missing` — one claim per expected piece of information | Yes |
| **Context Relevance** | DeepEval | Relevance of each retrieved chunk to the user query | `claim_level › rubric_scores["relevance"]` per chunk | No |
| **Answer Correctness** | RAGAS | Factual match between generated answer and ground-truth answer | `claim_level › is_correct` with reference provided | Yes |
| **Grounding / Citation Accuracy** | Custom, LangSmith | Citations point to real, supporting passages in the source | `claim_level › is_correct` — post-validate citation text as substring of cited source | No |
| **Response Completeness** | DeepEval | All aspects of a multi-part question are addressed | `claim_level › is_missing` — one claim per question aspect | No |
| **Noise Sensitivity** | RAGAS | Accuracy degradation when irrelevant chunks are added to context | `aggregate` — compare `is_correct` rates across clean vs. noisy context eval runs | No |
| **Answer Similarity** | RAGAS | Semantic similarity between generated and reference answer | `claim_level › is_correct` with reference + semantic overlap rubric | Yes |
| **Source Attribution** | Custom | Each factual claim is attributed to a specific source document | `claim_level › is_correct` — evidence validates attribution; `is_correct=False` if claim lacks source | No |

---

#### Agent / Tool Use

| Metric | Source | Description | Approach | Needs reference |
|---|---|---|---|---|
| **Task Completion** | DeepEval | Whether the agent completed the assigned end-to-end task | `claim_level › is_correct` — one claim per task requirement | Yes |
| **Tool Call Accuracy** | DeepEval | Correct tools called with correct parameters in correct order | `claim_level › is_correct` — one claim per tool call step | Yes |
| **Trajectory Quality** | LangSmith, Custom | Quality of the reasoning path — steps are logical and non-redundant | `holistic › rubric_scores["trajectory_quality"]` on full agent trace | No |
| **Plan Quality** | Custom | Agent's plan is appropriate and sufficient for the task | `holistic › rubric_scores["plan_quality"]` | No |
| **Efficiency** | Custom | Minimal steps taken; no unnecessary tool calls or loops | `holistic › rubric_scores["efficiency"]` (0.0=many unnecessary steps, 1.0=optimal) | No |
| **Role Adherence** | DeepEval | Agent stays within its defined persona and constraints | `claim_level › is_correct` — one claim per constraint; failure_condition on violation | No |
| **Memory Faithfulness** | Custom | Agent correctly uses information from prior conversation turns | `claim_level › is_correct` — evidence from conversation history | No |
| **Handoff Accuracy** | Custom (multi-agent) | Correct sub-agent or tool selected for delegation | `claim_level › is_correct` — one claim per handoff decision | Yes |

---

#### Reference-Based NLG (for comparison / regression baselines)

These classical metrics cannot be computed by an LLM judge — they are computed deterministically. They appear as instrumental metrics in `RunMetadata`, not in `ValidationObject`.

| Metric | Source | Description | Approach |
|---|---|---|---|
| **BLEU** | Papineni et al. 2002 | N-gram precision overlap with reference; standard MT benchmark | `instrumental` — computed via `sacrebleu` library |
| **ROUGE-1/2/L** | Lin 2004 | Recall-oriented n-gram overlap; standard summarization benchmark | `instrumental` — computed via `rouge-score` library |
| **METEOR** | Banerjee & Lavie 2005 | BLEU variant with stemming, synonyms, and word order | `instrumental` — computed via `nltk.translate.meteor_score` |
| **BERTScore** | Zhang et al. 2019 | Token-level semantic similarity using BERT embeddings | `instrumental` — computed via `bert-score` library |
| **MoverScore** | Zhao et al. 2019 | Earth mover distance between sentence embeddings | `instrumental` — computed via `moverscore` library |
| **Exact Match** | SQuAD benchmark | Binary: generated answer exactly matches reference string | `instrumental` — string equality check |
| **Semantic Similarity** | LangSmith, EvidentlyAI | Cosine similarity between output and reference embeddings | `instrumental` — embedding model (e.g. `text-embedding-3-small`) |

---

**Reading the table:** The "Approach" column tells you exactly where each metric lives in the system. `claim_level › is_correct` means it is an acceptance criterion evaluated per atomic claim — it goes in the YAML `criteria` / `failure_condition` fields. `holistic › rubric_scores["name"]` means it goes in a `holistic` section's `rubric_scales`. `aggregate` means no YAML change needed — it is computed by the Metrics Calculator from existing `ValidationObject` fields. `instrumental` means it is measured outside the LLM judge entirely.

---

## 3. Full System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              SUBMISSION LAYER                                     │
│                                                                                   │
│   Dev CI/CD Gate          Production Sampler          Validation Tool             │
│   (on every PR)           (5% of live traffic)        (golden set labeling, §9.1) │
└───────────────────────────────────┬──────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION REQUEST                                      │
│                                                                                   │
│  {                                                                                │
│    task_type:    "json_extraction" | "summary_template" | "classification"        │
│                  (MVP) + "summary_freeform" | "freeform_text" (Phase 2)           │
│    llm_output:   <raw LLM response>                                               │
│    context:      <source document / transcript / retrieved chunks>                │
│    reference:    <optional golden answer>                                         │
│    judge_config: "contracts-v2.1.yaml"    ← points to rubric YAML                │
│    scenario:     "dev" | "prod" | "migration" | "pairwise"                       │
│    model_meta:   { provider, model_id, prompt_version, temperature }             │
│  }                                                                                │
└───────────────────────────────────┬──────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │      PRE-VALIDATION CHECKS      │
                    │  (deterministic, synchronous)   │
                    │                                 │
                    │  ✓ Schema validation            │
                    │  ✓ Output length bounds         │
                    │  ✓ Required fields present      │
                    │  ✓ Encoding / format check      │
                    │  ✓ Context length within limits │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │    CLAIM DECOMPOSITION ENGINE   │
                    │                                 │
                    │  JSON extraction   → per-field (deterministic)     │
                    │  Template summary  → JSON → per-field (det.)       │
                    │  Classification    → per-label (deterministic)     │
                    │  [Phase 2] Free-form → Haiku call → ≤10 claims     │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │    FIELD TYPE CLASSIFIER        │
                    │                                 │
                    │  extractive  → exact match      │
                    │              + evidence lookup  │
                    │  inferential → CoT reasoning    │
                    │  free_form   → rubric scoring   │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │   EVAL STEP GENERATOR           │
                    │   (cached per criteria+version) │
                    │                                 │
                    │  criteria → LLM call →          │
                    │  numbered evaluation steps      │
                    │  (G-Eval Improvement 1)         │
                    └───────────────┬────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                    │
   ┌───────────▼──────┐  ┌─────────▼──────┐  ┌─────────▼──────────┐
   │  JUDGE POOL      │  │  JUDGE POOL     │  │  JUDGE POOL        │
   │  Primary Judge   │  │  Cross-check    │  │  Specialized       │
   │                  │  │  (pairwise /    │  │  (harmfulness,     │
   │  - OpenAI GPT    │  │   ensemble)     │  │   code validity,   │
   │  - Anthropic     │  │                 │  │   factcheck)       │
   │  - Enterprise GW │  │  Anti-bias:     │  │                    │
   │  - OLLAMA        │  │  swap order,    │  │  Pluggable via     │
   │  - llama-cpp     │  │  cross-family   │  │  criteria YAML     │
   └───────────┬──────┘  └─────────┬──────┘  └─────────┬──────────┘
               │                    │                    │
               └────────────────────┴────────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │     POST-VALIDATION CHECKS      │
                    │                                 │
                    │  ✓ Judge output schema valid    │
                    │  ✓ Score within expected range  │
                    │  ✓ Evidence quotes exist in ctx │
                    │  ✓ Reasoning non-empty          │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │   VALIDATION OBJECT ASSEMBLER   │
                    │                                 │
                    │  Merges judge output into       │
                    │  ValidationObject per field     │
                    │  Attaches error_theme from      │
                    │  user taxonomy                  │
                    └───────────────┬────────────────┘
                                    │
          ┌─────────────────────────┼──────────────────────────┐
          │                         │                          │
┌─────────▼──────┐      ┌──────────▼──────────┐    ┌─────────▼──────────┐
│  RESULT STORE  │      │  ASYNC QUEUE         │    │  PROMPT CACHE      │
│  (PostgreSQL)  │      │  (Redis / SQS)       │    │  (Redis)           │
│                │      │                      │    │                    │
│  ValidationObj │      │  Error theme batch   │    │  eval_steps cache  │
│  + run metadata│      │  Pattern scoring     │    │  (criteria_hash →  │
│  + model_meta  │      │  Prompt advisor      │    │   steps[])         │
│  Versioned     │      │  Alert engine        │    │  judge_result cache│
└────────────────┘      └──────────────────────┘    └────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          ANALYTICS LAYER                                 │
│                                                                          │
│  Benchmarking Dashboard    Error Theme Dashboard    Prompt Advisor       │
│  (per model/provider)      (batch pattern mining)   (fix suggestions)    │
│                                                                          │
│  Observability: OpenTelemetry traces + spans for every judge call        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. LLM-Agnostic Adapter Layer

The system under evaluation and the judge model may come from completely different providers. The adapter layer wraps every provider behind a single interface, so the evaluation pipeline never imports `openai` or `anthropic` directly.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    text: str
    logprobs: dict[str, float] | None  # token → probability; None if provider doesn't support
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_id: str
    provider: str

class LLMAdapter(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_logprobs: int = 5,        # For G-Eval probability weighting
        cache_control: bool = True,   # Enable prompt caching where supported
    ) -> LLMResponse:
        ...

# ── OpenAI ────────────────────────────────────────────────────────────────
class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model

    async def complete(self, messages, temperature=0.0, max_tokens=2048,
                       top_logprobs=5, cache_control=True) -> LLMResponse:
        import time; t0 = time.monotonic()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        logprobs = {
            tok.token: tok.logprob
            for tok in (resp.choices[0].logprobs.content[0].top_logprobs or [])
        }
        return LLMResponse(
            text=resp.choices[0].message.content,
            logprobs=logprobs,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=int((time.monotonic() - t0) * 1000),
            model_id=self.model,
            provider="openai",
        )

# ── Anthropic ─────────────────────────────────────────────────────────────
class AnthropicAdapter(LLMAdapter):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self.model = model

    async def complete(self, messages, temperature=0.0, max_tokens=2048,
                       top_logprobs=5, cache_control=True) -> LLMResponse:
        import anthropic, time; t0 = time.monotonic()
        # Inject prompt cache control on large static blocks
        if cache_control and messages and len(messages[0].get("content", "")) > 1024:
            messages[0]["content"] = [
                {"type": "text", "text": messages[0]["content"],
                 "cache_control": {"type": "ephemeral"}}
            ]
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        # Anthropic doesn't expose logprobs — fall back to sampling for G-Eval
        return LLMResponse(
            text=resp.content[0].text,
            logprobs=None,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=int((time.monotonic() - t0) * 1000),
            model_id=self.model,
            provider="anthropic",
        )

# ── OLLAMA (local) ─────────────────────────────────────────────────────────
class OLLAMAAdapter(LLMAdapter):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        import httpx
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120)
        self.model = model

    async def complete(self, messages, temperature=0.0, max_tokens=2048,
                       top_logprobs=5, cache_control=True) -> LLMResponse:
        import time; t0 = time.monotonic()
        resp = await self.client.post("/api/chat", json={
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        })
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            logprobs=None,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=int((time.monotonic() - t0) * 1000),
            model_id=self.model,
            provider="ollama",
        )

# ── Factory ────────────────────────────────────────────────────────────────
def get_adapter(provider: str, model: str, **kwargs) -> LLMAdapter:
    return {
        "openai":      lambda: OpenAIAdapter(model),
        "anthropic":   lambda: AnthropicAdapter(model),
        "ollama":      lambda: OLLAMAAdapter(model, **kwargs),
        "llama_cpp":   lambda: LlamaCppAdapter(model, **kwargs),   # similar pattern
        "enterprise":  lambda: EnterpriseGatewayAdapter(model, **kwargs),
    }[provider]()
```

---

## 5. YAML-Based Judge Configuration

Every evaluation task is defined in a YAML file. The pipeline reads this YAML, generates evaluation steps per criterion (cached), and routes each field to the appropriate judge strategy. This keeps the code path identical across tasks — only the YAML changes.

```yaml
# judge-configs/call-contract-extraction-v2.yaml

metadata:
  name: "Call Contract Extraction Evaluator"
  version: "2.1"
  task_type: json_extraction
  error_taxonomy: taxonomies/finance-errors.yaml   # user-supplied

global_rubrics:
  - id: hallucination_guard
    criteria: "No extracted value should introduce information absent from the transcript."
    applies_to: all
  - id: harmful_content
    criteria: "No extracted field should contain or expose PII beyond what is in the transcript."
    applies_to: all
    plugin: harmfulness_detector   # loads specialized judge plugin

sections:
  customer:
    # eval_mode defaults to claim_level when not specified — each field becomes one Claim
    rubric: "Customer fields must match verbatim statements in the transcript. Do not infer."
    fields:
      account_number:
        field_type: extractive
        criteria: "Must be a 10-digit numeric string matching the account number stated by agent or customer."
        failure_condition: "If the extracted value has fewer or more than 10 digits, or contains non-numeric characters, mark as incorrect."
        expected_format: "10-digit numeric string, e.g. '4829301020'"
        example_pass: "4829301020"
        example_fail: "482930102 (9 digits — digit dropped)"
        missing_rule: "If no account number is stated, set is_missing=true, missing_severity='completely'."
      customer_name:
        field_type: extractive
        criteria: "Full name as stated; do not expand abbreviations or correct spelling."
        failure_condition: "If the name is expanded (e.g., 'Bob' → 'Robert') or spelling is changed, mark as incorrect."
        expected_format: "Name exactly as spoken, no normalization"
        example_pass: "J. Smith"
        example_fail: "John Smith (expanded from 'J. Smith')"

  contract:
    rubric: "Contract fields require exact figures. Unit normalization (monthly→annual) is an error."
    fields:
      renewal_date:
        field_type: extractive
        criteria: "ISO 8601 date (YYYY-MM-DD). If transcript states month and year only, set is_missing=true, missing_severity='partially'."
        failure_condition: "If the date format is not ISO 8601, or if the extracted date does not match what was stated, mark as incorrect."
        expected_format: "YYYY-MM-DD"
        example_pass: "2025-03-15"
        example_fail: "March 15, 2025 (not ISO 8601)"
      annual_value:
        field_type: extractive
        criteria: "Total annual contract value in USD."
        failure_condition: "If the stated period is monthly, do not multiply to annual — extract the monthly figure and set error_theme to unit_ambiguous. Any multiplication or conversion is an error."
        expected_format: "Integer or decimal, no currency symbol"
        example_pass: "4000  (when transcript states $4,000/month — extracted as-is)"
        example_fail: "48000 (multiplied monthly $4,000 × 12 — forbidden by rubric)"

  risk_flags:
    rubric: "Risk flags require explicit evidence. Absence of mention means not_flagged, not missing."
    fields:
      churn_risk:
        field_type: inferential
        criteria: "True only if customer explicitly threatened to cancel or expressed strong dissatisfaction."
        failure_condition: "Mild frustration or general complaints do not qualify. Customer must use explicit cancellation language or direct threat."
        expected_format: "Boolean: true or false"
        example_pass: "true — Customer said 'I'm considering switching providers'"
        example_fail: "true — Customer said 'this is taking a while' (insufficient signal)"
        cot_required: true
        missing_rule: "If no churn signal present, set is_missing=false and is_correct=true with value=false."

  sentiment:
    rubric: "Sentiment classification must be grounded in explicit customer statements."
    fields:
      call_sentiment:
        field_type: classification
        criteria: "Classify the overall customer sentiment based on explicit statements in the transcript."
        allowed_labels:
          - label: "positive"
            criteria: "Customer expresses satisfaction, gratitude, or explicit approval of the outcome."
            example: "Customer said 'That's exactly what I needed, thank you.'"
          - label: "negative"
            criteria: "Customer expresses frustration, dissatisfaction, or complaint about service or outcome."
            example: "Customer said 'This is unacceptable. I've been waiting three weeks.'"
          - label: "neutral"
            criteria: "Customer is transactional — neither positive nor negative. No emotional language."
            example: "Customer said 'OK, noted. Is that all?'"
        multi_label: false
        missing_rule: "If no sentiment signal present, assign 'neutral'. Do not leave blank."

  call_summary:
    # eval_mode: holistic — skip claim decomposition entirely.
    # The whole section text is sent to the judge in one call.
    # Use for metrics that measure whole-output properties, not individual claims.
    # Coherence, readability, and verbosity have no meaning at the atomic-claim level.
    eval_mode: holistic
    rubric: "Evaluate the overall quality of the generated call summary as a whole."
    rubric_scales:
      coherence:
        0.0: "Disjointed — ideas don't connect or contradict each other"
        0.5: "Mostly coherent with minor structural gaps"
        1.0: "Well-organized, ideas flow logically"
      readability:
        0.0: "Dense, jargon-heavy, or hard to follow"
        0.5: "Readable with some awkward phrasing"
        1.0: "Clear and concise — easily understood on first read"
      verbosity:
        0.0: "Far too long (padding) or too short (key facts missing)"
        0.5: "Acceptable length with minor bloat"
        1.0: "Appropriately concise for the content"

  summary_faithfulness:
    # eval_mode: both — run claim decomposition (for faithfulness precision/recall)
    # AND a holistic pass (for section-level coherence).
    # ValidationObjects carry is_correct per claim; HolisticResult carries rubric_scores.
    eval_mode: both
    rubric: "Summary must be faithful to the transcript and internally coherent."
    fields:
      key_issue:
        field_type: free_form
        criteria: "The stated key issue must be explicitly mentioned by the customer."
        failure_condition: "If the issue is inferred rather than stated, mark as incorrect."
      resolution:
        field_type: free_form
        criteria: "Resolution must match what the agent confirmed at the end of the call."
    rubric_scales:
      coherence:
        0.0: "Summary reads as disconnected fragments"
        0.5: "Mostly coherent"
        1.0: "Flows as a unified narrative"

judge_settings:
  primary_judge:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.0
    prompt_cache: true
  cross_check_judge:             # Used for pairwise and high-stakes fields
    provider: openai
    model: gpt-4o-2024-11-20
    temperature: 0.0
  evaluation_steps_cache_ttl: 86400   # 24h; regenerate daily or on rubric change
  max_concurrent_field_evals: 10
```

---

## 6. Evaluation Mode Router and Pipeline

The pipeline starts with a routing decision: does this field or section need claim decomposition, or should it be evaluated as a whole? The router reads `eval_mode` from the YAML config and splits the workload before any LLM call is made. The two resulting paths — claim-level and holistic — run concurrently and produce different result types that are merged at the end.

```
EVALUATION MODE ROUTING

                    EvalRequest
                         │
              ┌──────────▼──────────┐
              │   EvalModeRouter    │
              │  reads eval_mode    │
              │  from YAML config   │
              └──┬──────────────┬───┘
                 │              │
          claim_level        holistic
          (default)          (no decomposition)
                 │              │
    ┌────────────▼──┐    ┌──────▼────────────┐
    │ ClaimDecomposer│    │  HolisticJudge    │
    │ → Claim[]      │    │  (whole section)  │
    └────────────┬──┘    └──────┬────────────┘
                 │              │
    ┌────────────▼──┐    ┌──────▼────────────┐
    │ CascadingJudge │    │  HolisticResult   │
    │ per Claim      │    │  rubric_scores{}  │
    └────────────┬──┘    └──────┬────────────┘
                 │              │
              ┌──▼──────────────▼──┐
              │     EvalResult     │
              │  validation_objects│  ← from claim-level path
              │  holistic_results  │  ← from holistic path
              └────────────────────┘

eval_mode: both → both branches run in parallel; results merged into EvalResult
```

| `eval_mode` | What runs | Output type | Metrics produced |
|---|---|---|---|
| `claim_level` (default) | ClaimDecomposer → CascadingJudge per claim | `list[ValidationObject]` | `is_correct`, `is_missing`, `rubric_scores` per claim; precision/recall/F1 in aggregate |
| `holistic` | HolisticJudge on full section text, no decomposition | `HolisticResult` | `rubric_scores` (coherence, readability, verbosity) — no `is_correct`/`is_missing` |
| `both` | Both branches run concurrently | `list[ValidationObject]` + `HolisticResult` | All of the above; claim-level faithfulness + section-level coherence together |

**When to use each mode:**

| Metric | eval_mode | Why |
|---|---|---|
| Faithfulness, precision, recall, completeness | `claim_level` | Require knowing which specific claims are correct/missing |
| Coherence, readability, verbosity, fluency | `holistic` | Measure whole-output properties — meaningless on an atomic claim |
| Toxicity | `claim_level` or `holistic` | Claim-level catches specific hallucinated harmful facts; holistic catches tone/style toxicity |
| Both faithfulness and coherence together | `both` | Run claim-level for verdict + holistic for quality scores in one eval |

```python
from dataclasses import dataclass, field

@dataclass
class RoutingDecision:
    claim_level_sections: list[SectionConfig]   # → ClaimDecomposer → CascadingJudge
    holistic_sections: list[SectionConfig]       # → HolisticJudge directly
    both_sections: list[SectionConfig]           # → both paths run concurrently

class EvalModeRouter:
    def route(self, config: JudgeConfig) -> RoutingDecision:
        claim_level, holistic, both = [], [], []
        for section in config.sections.values():
            mode = section.get("eval_mode", "claim_level")   # claim_level is default
            if mode == "claim_level":
                claim_level.append(section)
            elif mode == "holistic":
                holistic.append(section)
            elif mode == "both":
                both.append(section)
            else:
                raise ValueError(f"Unknown eval_mode: {mode!r}")
        return RoutingDecision(claim_level, holistic, both)
```

### 6.0 Evaluation Mode Router

*(See routing diagram and table above.)*

### 6.1 Claim Decomposition Engine

The decomposer converts any LLM output into a flat list of independently verifiable `Claim` objects. MVP task types use deterministic decomposition (no LLM needed). Phase 2 task types use a dedicated cheap LLM call.

```python
import json, re, asyncio
from dataclasses import dataclass

@dataclass
class Claim:
    field_path: str          # e.g. "customer.account_number" or "summary.claim[3]"
    field_type: str          # "extractive" | "inferential" | "free_form" | "classification"
    actual_value: str | None
    criteria: str            # From YAML rubric for this field
    rubric: str | None       # Section-level rubric text
    allowed_labels: list[str] | None  # For classification fields only

# ── Decomposition prompt for Phase 2 free-form tasks ──────────────────────────
DECOMPOSE_PROMPT = """Split the following text into atomic, independently verifiable claims.

Rules:
- Maximum 10 claims
- One claim per numbered line
- Each claim must be a single, complete assertion (no pronouns without antecedents)
- Do not split a claim if both parts are only meaningful together
- Do not duplicate claims that express the same fact differently

Text:
{text}

Output format (numbered list only, no preamble):
1. <claim>
2. <claim>
"""

class ClaimDecomposer:

    def __init__(self, cheap_adapter=None):
        self.cheap_adapter = cheap_adapter  # Required for Phase 2 free-form tasks

    def decompose(self, task_type: str, llm_output: str, config: "JudgeConfig") -> list[Claim]:
        if task_type == "json_extraction":
            return self._decompose_json(llm_output, config)
        elif task_type == "summary_template":
            return self._decompose_template(llm_output, config)
        elif task_type == "classification":
            return self._decompose_classification(llm_output, config)
        elif task_type in ("summary_freeform", "freeform_text"):
            # Phase 2 only — requires cheap_adapter
            raise NotImplementedError(
                f"task_type={task_type!r} requires Phase 2 LLM decomposer. "
                "Use summary_template for MVP summarization tasks."
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type!r}")

    def _decompose_json(self, output: str, config) -> list[Claim]:
        """Deterministic — parse JSON output into one Claim per field_path."""
        data = json.loads(output)
        claims = []
        for section_name, section_cfg in config.sections.items():
            for field_name, field_cfg in section_cfg["fields"].items():
                field_path = f"{section_name}.{field_name}"
                value = data.get(section_name, {}).get(field_name)
                claims.append(Claim(
                    field_path=field_path,
                    field_type=field_cfg["field_type"],
                    actual_value=str(value) if value is not None else None,
                    criteria=field_cfg["criteria"],
                    rubric=section_cfg.get("rubric"),
                    allowed_labels=None,
                ))
        return claims

    def _decompose_template(self, output: str, config) -> list[Claim]:
        """Deterministic — map template sections to JSON fields, then treat as json_extraction."""
        parsed = self._parse_template_to_dict(output, config)
        return self._decompose_json(json.dumps(parsed), config)

    def _decompose_classification(self, output: str, config) -> list[Claim]:
        """Deterministic — one Claim per classification field in the config."""
        claims = []
        for section_name, section_cfg in config.sections.items():
            for field_name, field_cfg in section_cfg["fields"].items():
                if field_cfg.get("field_type") != "classification":
                    continue
                field_path = f"{section_name}.{field_name}"
                assigned_label = output.strip()  # Classification output is the label string
                allowed = [l["label"] for l in field_cfg.get("allowed_labels", [])]
                claims.append(Claim(
                    field_path=field_path,
                    field_type="classification",
                    actual_value=assigned_label if assigned_label in allowed else None,
                    criteria=field_cfg["criteria"],
                    rubric=section_cfg.get("rubric"),
                    allowed_labels=allowed,
                ))
        return claims

    # ── Phase 2: LLM-based free-form decomposition ────────────────────────
    async def decompose_freeform(self, output: str, config) -> list[Claim]:
        """Phase 2 only. Uses cheap LLM (Haiku) to split text into atomic claims."""
        prompt = DECOMPOSE_PROMPT.format(text=output)
        response = await self.cheap_adapter.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw_claims = self._parse_numbered_list(response.text)[:10]  # Cap at 10
        return [
            Claim(
                field_path=f"summary.claim[{i}]",
                field_type="free_form",
                actual_value=claim_text,
                criteria=config.global_rubrics[0]["criteria"] if config.global_rubrics else "",
                rubric=None,
                allowed_labels=None,
            )
            for i, claim_text in enumerate(raw_claims)
        ]

    def _parse_numbered_list(self, text: str) -> list[str]:
        return [
            re.sub(r"^\d+\.\s*", "", line).strip()
            for line in text.strip().splitlines()
            if re.match(r"^\d+\.", line.strip())
        ]
```

### 6.2 Full Pipeline: Step by Step

```python
import asyncio, hashlib, json
from typing import Any

class EvaluationPipeline:

    async def evaluate(self, request: EvalRequest) -> EvalResult:

        # ── Stage 1: Pre-validation ───────────────────────────────────────
        self.pre_validate(request)   # schema, lengths, encoding — raises on failure

        # ── Stage 2: Claim decomposition ─────────────────────────────────
        claims = self.decomposer.decompose(
            task_type=request.task_type,
            llm_output=request.llm_output,
            config=JudgeConfig.load(request.judge_config),
        )
        # json_extraction  → one Claim per field_path (deterministic)
        # summary_template → template → JSON → one Claim per section (deterministic)
        # classification   → one Claim per classification field (deterministic)
        # [Phase 2] summary_freeform / freeform_text → LLM decomposer, max 10 claims

        # ── Stage 3: Load judge config (already loaded for decomposition; re-use) ──
        config = JudgeConfig.load(request.judge_config)

        # ── Stage 4: Generate evaluation steps (cached) ───────────────────
        steps_per_field = await asyncio.gather(*[
            self.get_eval_steps(claim, config)
            for claim in claims
        ])

        # ── Stage 5: Parallel judge execution ────────────────────────────
        validation_objects = await asyncio.gather(*[
            self.judge_claim(claim, steps, config, request)
            for claim, steps in zip(claims, steps_per_field)
        ])

        # ── Stage 6: Post-validation ──────────────────────────────────────
        for vo in validation_objects:
            self.post_validate(vo, request.context)  # verify evidence quotes exist in ctx

        # ── Stage 7: Assemble result ──────────────────────────────────────
        return EvalResult(
            validation_objects=validation_objects,
            aggregate_scores=self.aggregate(validation_objects),
            run_metadata=self.build_metadata(request),
        )

    async def get_eval_steps(self, claim: Claim, config: JudgeConfig) -> list[str]:
        """Generate evaluation steps from criteria. Cached per (criteria_hash, rubric_version)."""
        cache_key = hashlib.sha256(
            f"{claim.criteria}::{config.version}".encode()
        ).hexdigest()

        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        prompt = STEP_GENERATION_PROMPT.format(
            task_introduction=config.metadata["name"],
            field_path=claim.field_path,
            field_type=claim.field_type,
            criteria=claim.criteria,
            rubric=claim.rubric or config.global_rubric,
            error_taxonomy=config.error_taxonomy,
        )
        response = await self.judge_adapter.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            cache_control=True,   # Cache the system prompt in the LLM provider too
        )
        steps = self.parse_numbered_list(response.text)
        await self.cache.set(cache_key, steps, ttl=config.eval_steps_cache_ttl)
        return steps

    async def judge_claim(
        self, claim: Claim, steps: list[str], config: JudgeConfig, req: EvalRequest
    ) -> ValidationObject:
        """Run the judge on one claim with the pre-generated evaluation steps."""

        # Check result cache first (same input + same judge + same criteria version)
        cache_key = hashlib.sha256(json.dumps({
            "field_path": claim.field_path,
            "actual_value": claim.actual_value,
            "context_hash": hashlib.md5(req.context.encode()).hexdigest(),
            "criteria_version": config.version,
            "judge_model": config.judge_settings["primary_judge"]["model"],
        }, sort_keys=True).encode()).hexdigest()

        cached = await self.cache.get(f"judge:{cache_key}")
        if cached:
            return ValidationObject(**cached, cached=True)

        prompt = JUDGE_PROMPT.format(
            field_path=claim.field_path,
            field_type=claim.field_type,
            actual_value=claim.actual_value,
            context=req.context,
            reference=req.reference or "Not provided",
            evaluation_steps="\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)),
            error_taxonomy=config.error_taxonomy_text,
        )
        response = await self.judge_adapter.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        vo = self.parse_judge_output(response, claim, config)
        await self.cache.set(f"judge:{cache_key}", vo.dict(), ttl=3600)
        return vo
```

### 6.3 Batching Strategy

Sending each claim to the judge in a separate API call is correct for development but wasteful in production. The batching strategy reduces cost and latency in three ways: grouping claims from the same document, grouping the same field across multiple documents, and running both tiers concurrently.

#### Three Batching Modes

```
BATCHING MODES

Mode 1: Within-Document (Tier 1 fields)
  One document → all Tier 1 claims → one judge call
  Source context sent once; claims listed in a numbered batch
  Provider prefix cache covers the static prompt + context on all calls after the first

Mode 2: Cross-Document Field Batching (same field, many docs)
  Same field + same criteria → batch up to 15 documents
  Ideal for offline evaluation runs over a dataset
  Each document's context is a numbered block in the batch prompt

Mode 3: Async Tier Parallelism
  Tier 1 and Tier 2 claims from different documents run concurrently
  Tier 2 waits for Tier 1 escalation signals; no blocking between documents
```

#### Within-Document Batch Call

All Tier 1 fields from a single document are grouped into one prompt. The context is sent once; the judge evaluates each field in sequence and returns a JSON array.

```python
WITHIN_DOC_BATCH_PROMPT = """
You are evaluating multiple fields extracted from the same source document.
The source context appears once below. Evaluate each field against it.

SOURCE CONTEXT:
{context}

FIELDS TO EVALUATE (one per numbered block):
{field_blocks}

Return a JSON array with one object per field, in the same order.
Each object must match the ValidationObject schema.
"""

async def judge_document_batch(
    claims: list[Claim],          # all Tier 1 claims for one document
    config: JudgeConfig,
    req: EvalRequest,
    adapter: LLMAdapter,
) -> list[ValidationObject]:

    # Build numbered field blocks
    field_blocks = "\n\n".join(
        f"[Field {i+1}] field_path={c.field_path}\n"
        f"  actual_value: {c.actual_value}\n"
        f"  criteria: {c.criteria}\n"
        f"  evaluation_steps:\n" + "\n".join(f"    {j+1}. {s}" for j, s in enumerate(c.steps))
        for i, c in enumerate(claims)
    )

    prompt = WITHIN_DOC_BATCH_PROMPT.format(
        context=req.context,
        field_blocks=field_blocks,
    )

    response = await adapter.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        cache_control=True,   # static prefix (system + context) cached by provider
    )

    raw = json.loads(response.text)   # list of dicts, one per field
    return [
        ValidationObject(**item, judge_model=adapter.model_id)
        for item in raw
    ]
```

#### Cross-Document Field Batching

For offline evaluation runs over a dataset, the same field across multiple documents can be batched together. This maximises the value of prefix caching (rubric + evaluation steps are identical across all documents; only the context changes).

```python
async def judge_field_across_docs(
    field_path: str,
    doc_claims: list[tuple[Claim, EvalRequest]],  # (claim, req) per document
    config: JudgeConfig,
    adapter: LLMAdapter,
    batch_size: int = 15,
) -> list[ValidationObject]:

    results = []
    for batch in chunked(doc_claims, batch_size):
        doc_blocks = "\n\n".join(
            f"[Document {i+1}]\n"
            f"  actual_value: {claim.actual_value}\n"
            f"  context: {req.context}"
            for i, (claim, req) in enumerate(batch)
        )

        prompt = CROSS_DOC_BATCH_PROMPT.format(
            field_path=field_path,
            criteria=batch[0][0].criteria,          # same criteria for all docs
            evaluation_steps=batch[0][0].steps,
            doc_blocks=doc_blocks,
        )

        response = await adapter.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            cache_control=True,   # criteria + steps cached; only doc_blocks varies
        )

        results.extend(parse_batch_response(response.text, batch))

    return results
```

#### Full Async Dispatch with Tier Pre-routing

The pipeline segments claims by complexity tier before dispatching. Tier 1 claims are grouped into within-document batches; Tier 2 claims run in parallel as individual calls (they've already been flagged as complex — no benefit to batching them).

```python
async def evaluate_with_batching(
    request: EvalRequest,
    claims: list[Claim],
    config: JudgeConfig,
    judge: CascadingJudge,
) -> list[ValidationObject]:

    # Pre-route by complexity tier (set on each Claim by ClaimDecomposer)
    tier1_claims = [c for c in claims if c.judge_tier == "tier1"]
    tier2_claims = [c for c in claims if c.judge_tier == "tier2"]

    # Tier 1: one batched call for all easy fields in this document
    tier1_task = judge_document_batch(
        claims=tier1_claims,
        config=config,
        req=request,
        adapter=judge.cheap,
    ) if tier1_claims else asyncio.coroutine(lambda: [])()

    # Tier 2: parallel individual calls for complex/inferential fields
    tier2_tasks = [
        judge.judge(claim=c, steps=c.steps, config=config, req=request)
        for c in tier2_claims
    ]

    # Run both tiers concurrently — Tier 1 batch and all Tier 2 calls fire together
    tier1_results, *tier2_results = await asyncio.gather(
        tier1_task, *tier2_tasks
    )

    # Merge preserving original field order
    results_by_path = {vo.field_path: vo for vo in tier1_results}
    for vo in tier2_results:
        results_by_path[vo.field_path] = vo

    return [results_by_path[c.field_path] for c in claims]
```

#### Prompt Caching Integration

Batching and prompt caching work at different layers and compose naturally:

| Layer | What is cached | TTL | Provider |
|---|---|---|---|
| Redis eval-steps cache | Evaluation steps per `(criteria_hash, rubric_version)` | 24 h | Your infrastructure |
| Provider prefix cache | Static prompt prefix: system instructions + rubric + evaluation steps | 5 min (Anthropic ephemeral) | Anthropic / OpenAI |
| Redis judge-result cache | Full `ValidationObject` per `(field_path, actual_value, context_hash, criteria_version)` | 1 h | Your infrastructure |

Within-document batching maximises prefix cache hits: the system prompt, rubric, and evaluation steps are identical for all fields in the batch — they land in the provider's prefix cache after the first document, making subsequent documents in the same run near-free on the static portion.

Cross-document field batching goes further: the criteria and evaluation steps for the same field are cached across all 15 documents in a batch, so you pay for the context portion only.

```
CACHE HIT SEQUENCE (within-document batch, 10 documents, 8 Tier 1 fields each):

Doc 1, Batch call:   [MISS] system + rubric + steps → cached by provider
Doc 2, Batch call:   [HIT]  system + rubric + steps from cache → pay only for context
Doc 3–10:            [HIT]  same cache hit pattern
Result: ~70–80% token reduction on the static prefix across the run
```

### 6.4 Holistic Path: Section-Level Evaluation

The holistic path skips claim decomposition entirely. The whole section text is sent to the judge in one call with the `rubric_scales` from YAML. The judge scores each named dimension and returns a `HolisticResult` — no `is_correct`, no `is_missing`, no per-claim breakdown.

#### HolisticResult Data Model

```python
from pydantic import BaseModel
from typing import Optional

class HolisticResult(BaseModel):
    section_path: str                 # e.g. "call_summary" or "report.executive_summary"
    eval_mode: str = "holistic"
    section_text: str                 # The full section text that was evaluated
    rubric_scores: dict[str, float]   # e.g. {"coherence": 0.8, "readability": 1.0, "verbosity": 0.5}
    evaluation_steps: list[str]       # Judge's reasoning before scores
    judge_model: str
    latency_ms: int
    cached: bool = False
    eval_version: str                 # "rubric=summary-v1.2,prompt=v2.0"
```

No `is_correct` or `is_missing` — those are claim-level verdicts. A section is not "correct" or "incorrect" as a whole; it is more or less coherent, readable, verbose. The `rubric_scores` dict is the complete output.

#### HolisticJudge

```python
HOLISTIC_JUDGE_PROMPT = """You are evaluating the overall quality of a text section.
Do NOT check individual facts. Evaluate only the named quality dimensions below.

SECTION: {section_path}
TEXT TO EVALUATE:
{section_text}

QUALITY DIMENSIONS TO SCORE (0.0 / 0.5 / 1.0 only):
{rubric_scales_formatted}

INSTRUCTIONS:
1. Read the full section text.
2. For each dimension, reason briefly about what you observe.
3. Assign a score: 0.0 (absent/poor), 0.5 (partial/acceptable), 1.0 (complete/good).
4. Do not assign scores between anchors (e.g. 0.3 or 0.7) — use only 0.0, 0.5, 1.0.

OUTPUT FORMAT (JSON):
{{
  "evaluation_steps": ["<step 1>", "<step 2>", ...],
  "rubric_scores": {{
    "<dimension_name>": 0.0 | 0.5 | 1.0,
    ...
  }}
}}
"""

class HolisticJudge:

    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter

    def _format_rubric_scales(self, rubric_scales: dict) -> str:
        lines = []
        for name, anchors in rubric_scales.items():
            lines.append(f"{name}:")
            for score, description in sorted(anchors.items()):
                lines.append(f"  {score}: {description}")
        return "\n".join(lines)

    async def evaluate(
        self,
        section_path: str,
        section_text: str,
        config: SectionConfig,
        req: EvalRequest,
    ) -> HolisticResult:

        import time, hashlib
        t0 = time.monotonic()

        # Result cache: same section text + same rubric version → same result
        cache_key = hashlib.sha256(
            f"{section_path}::{section_text}::{config.version}".encode()
        ).hexdigest()
        cached = await self.cache.get(f"holistic:{cache_key}")
        if cached:
            return HolisticResult(**cached, cached=True)

        prompt = HOLISTIC_JUDGE_PROMPT.format(
            section_path=section_path,
            section_text=section_text,
            rubric_scales_formatted=self._format_rubric_scales(config.rubric_scales),
        )

        response = await self.adapter.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            cache_control=True,   # system + rubric_scales cached; only section_text varies
        )

        parsed = json.loads(response.text)
        result = HolisticResult(
            section_path=section_path,
            section_text=section_text,
            rubric_scores=parsed["rubric_scores"],
            evaluation_steps=parsed["evaluation_steps"],
            judge_model=self.adapter.model_id,
            latency_ms=int((time.monotonic() - t0) * 1000),
            eval_version=config.eval_version,
        )
        await self.cache.set(f"holistic:{cache_key}", result.dict(), ttl=3600)
        return result
```

#### Updated EvalResult — Both Paths Merged

```python
class EvalResult(BaseModel):
    # Claim-level results (eval_mode: claim_level or both)
    validation_objects: list[ValidationObject]

    # Section-level holistic results (eval_mode: holistic or both)
    holistic_results: list[HolisticResult]

    aggregate_scores: dict[str, float]   # precision, recall, F1, avg rubric_scores, etc.
    run_metadata: RunMetadata
```

#### Full Dispatch with Both Paths

```python
async def evaluate(self, request: EvalRequest) -> EvalResult:

    config = JudgeConfig.load(request.judge_config)
    routing = EvalModeRouter().route(config)

    # ── Claim-level path (claim_level + both sections) ──────────────────────
    claim_sections = routing.claim_level_sections + routing.both_sections
    claim_level_task = self._run_claim_level(claim_sections, config, request)

    # ── Holistic path (holistic + both sections) ─────────────────────────────
    holistic_sections = routing.holistic_sections + routing.both_sections
    holistic_task = self._run_holistic(holistic_sections, config, request)

    # Both paths run concurrently
    validation_objects, holistic_results = await asyncio.gather(
        claim_level_task,
        holistic_task,
    )

    return EvalResult(
        validation_objects=validation_objects,
        holistic_results=holistic_results,
        aggregate_scores=self.aggregate(validation_objects, holistic_results),
        run_metadata=self.build_metadata(request),
    )
```

---

## 7. The Judge Prompt

The judge prompt produces the `ValidationObject` content. It forces chain-of-thought reasoning before the verdict, and extracts evidence as verbatim quotes with location. The prompt is modular: static sections (system instructions, output format, error taxonomy) are cached at the provider level, while dynamic sections (rubrics, evaluation steps, context, golden dataset) vary per call.

```
[SYSTEM INSTRUCTIONS — cached at provider level]
You are an expert evaluator for an AI {task_type} system.
Your job is to evaluate whether the AI's output is correct, complete, and grounded in the source document.
Always reason step-by-step before issuing a verdict. Never hallucinate evidence.

[EVALUATION STEPS — cached per (criteria, rubric_version)]
Follow these steps precisely in order:
{evaluation_steps}

[RUBRICS — cached per judge_config version]
{rubrics}

[TEST SET UNDER EVALUATION]
Field path:   {field_path}
Field type:   {field_type}   (extractive | inferential | free_form)
Task:         {task}   (Summarization | Extraction | Classification)
Actual value extracted by the AI: "{actual_value}"

[SOURCE CONTEXT]
{context}

[GOLDEN DATASET / REFERENCE — optional]
{reference}   ← "Not provided" for reference-free evaluation

[OUTPUT FORMATTING — respond with valid JSON only]
{
  "evaluation_steps_executed": [
    "Step 1: <your reasoning for step 1>",
    "Step 2: <your reasoning for step 2>",
    ...
  ],
  "evidence": [
    "<verbatim quote from the source context>",
    "<another verbatim quote if needed>"
  ],
  "actual_value": "<same as input if correct; <corrected>correct value here</corrected> if incorrect>",
  "is_correct": true | false,
  "is_missing": true | false,
  "missing_severity": "partially" | "completely" | null,
  "reason_for_incorrect": "<quote the specific rubric definition that failed, OR 'hallucination', OR 'source_document_does_not_contain_this_information'; null if correct>",
  "rubric_scores": {"<metric_name>": <0.0–1.0>, ...},
  "error_theme": "<from taxonomy below, or null if correct>"
}
Note: rubric_scores is {} (empty dict) for binary extractive/inferential fields.
      For fields with rubric_scales defined in YAML, populate one key per metric name.

[ERROR TAXONOMY]
If the extraction is incorrect, classify using exactly one of these themes:
{error_taxonomy}
(e.g. unit_confusion, hallucinated_entity, scope_mismatch, format_error, temporal_confusion, missing_from_source)
```

> 🏭 **Production note**: Evidence is a plain `list[str]` of verbatim quotes — no location, no polarity flag. This keeps the judge output minimal and the post-validation check simple: substring-match every quote against the source context. If a quote does not appear in the context, the judge hallucinated it.

> 🏭 **Production note**: The `<corrected>` tag convention in `actual_value` makes the judge output easy to post-process: a simple regex extracts the corrected value and populates `corrected_value` in the `ValidationObject`. This pattern avoids schema ambiguity — one field, two modes, zero parsing ambiguity.

> 🏭 **Production note**: Request JSON output explicitly and validate the schema in your post-validation step. Add a `json_repair` fallback for near-JSON output (trailing commas, missing quotes). Never send malformed judge output downstream — it silently corrupts error theme statistics.

---

### 7.1 Modular Judge Prompt Design

The judge prompt is split into static and dynamic sections, mapped to two caching layers:

```
JUDGE PROMPT STRUCTURE
──────────────────────────────────────────────────────────────
STATIC — cached at provider level (Anthropic ephemeral / OpenAI prefix)
  ┌─────────────────────┐
  │ System Instructions │  Role, evaluation philosophy, output format rules
  ├─────────────────────┤
  │ Evaluation Steps    │  Generated from criteria + rubric (cached in Redis 24h)
  ├─────────────────────┤
  │ Rubrics             │  YAML-defined acceptance criteria per field/section
  └─────────────────────┘

DYNAMIC — varies per call (not cached at provider)
  ┌───────────────────────────┐
  │ TestSet Under Evaluation  │  field_path, field_type, actual_value, task type
  ├───────────────────────────┤
  │ Context                   │  Source document or transcript excerpt
  ├───────────────────────────┤
  │ Golden Dataset/Reference  │  Optional; omitted for reference-free evaluation
  └───────────────────────────┘

FOOTER (static)
  ┌─────────────────┐
  │ Output Format   │  JSON schema, field descriptions, error taxonomy
  └─────────────────┘
```

This split is what makes prompt caching economically viable. The static prefix (system instructions + eval steps + rubrics + output format) can exceed 2,000 tokens and is identical across hundreds of judge calls in a batch run. With Anthropic's `cache_control: ephemeral` or OpenAI's automatic prefix caching, each call after the first pays only for the dynamic suffix tokens.

---

### 7.2 Cascading Judge: Cost-Efficient Multi-Tier Evaluation

Running an expensive frontier judge on every claim is economically unviable at production volume. The cascading judge resolves this by routing easy claims to a cheap model and escalating only uncertain or high-stakes cases to a more capable one — the same pattern used in production content moderation and medical AI triage.

```
CASCADING JUDGE FLOW

Input: claim + context + rubric

┌─────────────────────────────────────────────────────────┐
│  TIER 1: Cheap Judge                                     │
│  claude-haiku-4-5 / gpt-4o-mini                         │
│  Cost: ~10% of frontier model                           │
│  Handles: ~75–80% of claims                             │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼──────────┐
              │  Confidence Check  │
              │                   │
              │  confidence ≥ 0.85 │──── YES ──→ Accept verdict (no escalation)
              │  AND verdict is   │
              │  unambiguous?     │
              └────────┬──────────┘
                       │ NO
                       ▼
┌─────────────────────────────────────────────────────────┐
│  TIER 2: Frontier Judge                                  │
│  claude-sonnet-4-6 / gpt-4o-2024-11-20                  │
│  Cross-family preferred (Claude judges GPT outputs)      │
│  Handles: ~20–25% of claims                             │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼──────────┐
              │  Golden / Migration│──── YES ──→ Always use Tier 2 + cross-family
              │  run?              │             (bypass Tier 1 entirely)
              └───────────────────┘
```

**Two-stage routing**: Field complexity (pre-computed) determines the starting tier. Confidence (runtime signal) determines whether to escalate from Tier 1 to Tier 2. Pre-routing eliminates the wasted Tier 1 call for fields that will always need escalation.

```python
class CascadingJudge:

    AMBIGUOUS_THEMES = {"scope_mismatch", "temporal_confusion", "inferential_leap"}

    # Default tier by field type — overridden by YAML judge_tier annotation
    DEFAULT_TIER: dict[str, str] = {
        "extractive":     "tier1",   # deterministic, high-confidence in cheap models
        "classification": "tier1",   # label-from-set is well-bounded
        "inferential":    "tier2",   # implicit reasoning → cheap models fail more
        "free_form":      "tier2",   # rubric scoring needs nuanced judgment
    }

    def __init__(
        self,
        cheap_adapter: LLMAdapter,      # e.g. AnthropicAdapter("claude-haiku-4-5-20251001")
        expensive_adapter: LLMAdapter,  # e.g. OpenAIAdapter("gpt-4o-2024-11-20") — cross-family
        confidence_threshold: float = 0.85,
        always_escalate_scenarios: list[str] = ["golden_set", "migration"],
    ):
        self.cheap = cheap_adapter
        self.expensive = expensive_adapter
        self.threshold = confidence_threshold
        self.always_escalate = always_escalate_scenarios

    def resolve_tier(self, claim: Claim) -> str:
        """Determine starting tier from YAML annotation or field_type default."""
        # Explicit YAML annotation takes precedence
        if claim.judge_tier:
            return claim.judge_tier
        # Data-driven override: field has a high historical error rate → escalate
        if claim.historical_error_rate and claim.historical_error_rate > 0.30:
            return "tier2"
        return self.DEFAULT_TIER.get(claim.field_type, "tier1")

    async def judge(
        self, claim: Claim, steps: list[str], config: JudgeConfig, req: EvalRequest
    ) -> ValidationObject:

        # Scenario-level override: golden set and migration always use Tier 2
        if req.scenario in self.always_escalate:
            vo = await judge_claim(claim, steps, config, req, adapter=self.expensive)
            vo.escalated = True
            return vo

        # Field-level pre-routing: bypass Tier 1 for complex fields
        starting_tier = self.resolve_tier(claim)
        if starting_tier == "tier2":
            vo = await judge_claim(claim, steps, config, req, adapter=self.expensive)
            vo.escalated = False   # pre-routed, not escalated
            return vo

        # Tier 1 path: run cheap judge, then check for escalation
        vo = await judge_claim(claim, steps, config, req, adapter=self.cheap)

        needs_escalation = (
            vo.confidence < self.threshold
            or vo.error_theme in self.AMBIGUOUS_THEMES
            or (not vo.is_correct and not vo.evidence)  # verdict without evidence
        )

        if needs_escalation:
            vo = await judge_claim(claim, steps, config, req, adapter=self.expensive)
            vo.escalated = True

        return vo
```

**When each tier runs:**

| Scenario | Tier 1 (cheap) | Tier 2 (frontier) |
|---|---|---|
| Production sampling (5%) | ✓ Default | Only uncertain claims (~20%) |
| Dev iteration | ✓ Default | Only uncertain claims |
| Golden set creation | — | ✓ Always (bypass Tier 1) |
| Model migration sign-off | — | ✓ Always (bypass Tier 1) |
| High-stakes field audit | — | ✓ Always |

**Anti-bias configuration**: For Tier 2, always use a cross-family judge (Claude judges GPT outputs, GPT judges Claude outputs). Self-enhancement bias is documented at 87.76% self-preference rate — cross-family escalation eliminates this as a systematic effect on the cases that matter most (uncertain or flagged claims).

```yaml
judge_settings:
  tier1:
    provider: anthropic
    model: claude-haiku-4-5-20251001
    temperature: 0.0
    prompt_cache: true
  tier2:
    provider: openai               # Cross-family: different from the model being evaluated
    model: gpt-4o-2024-11-20
    temperature: 0.0
  escalation_threshold: 0.85       # Confidence below this → escalate to tier2
  always_escalate_scenarios:
    - golden_set
    - migration
```

---

## 8. Field Type Strategy and Complexity Segmentation

The field type determines how the judge approaches the evaluation. Complexity segmentation extends this by pre-routing fields to either a cheap (Tier 1) or capable (Tier 2) judge before any evaluation runs — eliminating the wasted Tier 1 call for fields that will always need escalation. This is the most important routing decision in the pipeline — using a rubric-scoring judge on an extractive field (like an account number) wastes tokens and produces noisy results; using exact-match on a sentiment summary misses everything meaningful.

```
FIELD TYPE → JUDGE STRATEGY MAPPING

┌────────────────┬──────────────────────┬──────────────────────────────────────────────────┐
│ Field Type     │ Examples             │ Judge Strategy                                   │
├────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ Extractive     │ Account number,      │ 1. Locate value in context (evidence)            │
│                │ renewal date,        │ 2. Exact match or format-normalized match        │
│                │ dollar amount        │    against actual_value (uses structured rubric: │
│                │                      │    criteria + failure_condition + example)       │
│                │                      │ 3. Binary is_correct + is_missing               │
│                │                      │    corrected_value = correct value from context  │
├────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ Inferential    │ Churn risk flag,     │ 1. CoT: reason about implicit meaning            │
│                │ customer intent,     │ 2. Cite evidence that implies the inferred value │
│                │ risk classification  │ 3. Binary verdict with full reasoning            │
│                │                      │    Evidence quality is the primary signal        │
├────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ Classification │ Call sentiment,      │ 1. Receive allowed_labels + per-label criteria   │
│                │ intent category,     │ 2. Evaluate: does evidence support assigned label?│
│                │ document type,       │ 3. If wrong: which label is correct? Why?        │
│                │ risk tier            │    corrected_value = correct label from allowed  │
│                │                      │    is_missing=true if no label returned          │
│                │                      │    Validates label is in allowed_labels set      │
├────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ Free-form      │ Template summary     │ 1. rubric_scoring via named rubric_scales        │
│ (template      │ sections (Phase 1),  │ 2. rubric_scores dict output per metric name     │
│  sections)     │ rewrite output       │ 3. Cite what is missing or factually wrong       │
│                │                      │    Reference optional but improves signal        │
└────────────────┴──────────────────────┴──────────────────────────────────────────────────┘
```

#### Complexity Segmentation: Pre-Routing by Field Type

Field type alone gives a strong prior on which tier to start with. The routing table below encodes this prior as the default. Two overrides are layered on top: an explicit YAML annotation per field, and a data-driven escalation trigger based on the field's historical error rate in production.

**Default tier assignment:**

| Field Type | Default Tier | Rationale |
|---|---|---|
| `extractive` | Tier 1 (cheap) | Locate + exact/format-match is well-bounded; Haiku handles it reliably |
| `classification` | Tier 1 (cheap) | Label-from-allowed-set is structurally bounded; little ambiguity |
| `inferential` | Tier 2 (frontier) | Implicit reasoning over unstated information; cheap models fail above 30% |
| `free_form` | Tier 2 (frontier) | Rubric scoring requires nuanced comparative judgment |

**YAML annotation — explicit override:**

Any field can be pinned to a tier regardless of its `field_type`. Use this when a specific field is known to be harder than its type suggests:

```yaml
contract_terms:
  annual_value:
    field_type: extractive
    judge_tier: tier1              # default; no override needed
    criteria: "Total annual contract value in USD."
    failure_condition: "Do not multiply monthly figures to annual."
    expected_format: "Integer or decimal, no currency symbol"
    example_pass: "4000"
    example_fail: "48000"

  jurisdiction_clause:
    field_type: extractive
    judge_tier: tier2              # override: legal language is ambiguous even for extraction
    criteria: "The governing law jurisdiction stated in the contract."
    failure_condition: "If multiple jurisdictions mentioned, extract the primary one."

  churn_risk:
    field_type: inferential
    judge_tier: tier2              # default for inferential; annotation is optional but explicit
    criteria: "Infer whether the customer is at risk of churning."
    rubric_scales:
      churn_signal:
        0.0: "No churn indicators present"
        0.5: "Mild dissatisfaction; no explicit threat"
        1.0: "Explicit threat to cancel or competitor mention"
```

**Data-driven escalation — production override:**

After enough production runs accumulate, the pipeline reads historical error rates per field and overrides the tier assignment when a field consistently exceeds the 30% error threshold:

```python
def resolve_tier(claim: Claim, error_rate_store: ErrorRateStore) -> str:
    # Explicit YAML annotation always wins
    if claim.judge_tier:
        return claim.judge_tier

    # Data-driven override: high historical error rate → escalate regardless of field_type
    historical_rate = error_rate_store.get(claim.field_path)
    if historical_rate is not None and historical_rate > 0.30:
        return "tier2"

    # Fall back to field_type default
    return CascadingJudge.DEFAULT_TIER.get(claim.field_type, "tier1")
```

```python
class ErrorRateStore:
    """Reads aggregated ValidationObject history to produce per-field error rates."""

    def get(self, field_path: str, lookback_days: int = 30) -> Optional[float]:
        rows = self.db.query(
            "SELECT AVG(CASE WHEN is_correct = false THEN 1.0 ELSE 0.0 END) "
            "FROM validation_objects "
            "WHERE field_path = ? AND created_at > NOW() - INTERVAL ? DAY",
            field_path, lookback_days,
        )
        return rows[0][0] if rows else None
```

The resolution order is: **YAML annotation → data-driven override → field_type default**. This ensures that team knowledge (annotation) always takes precedence, production evidence (error rate) corrects wrong defaults, and the table provides a sensible starting point for new fields.

**Classification judge prompt additions:**

For `field_type: classification`, the judge prompt receives additional context from the YAML `allowed_labels` block:

```
[CLASSIFICATION TASK — additional context]
This is a classification field. The AI must assign exactly one of the following labels.
Evaluate whether the assigned label is correct given the evidence in the source document.

Allowed labels and their criteria:
  "positive":  Customer expresses satisfaction, gratitude, or explicit approval.
               Example: "Customer said 'That's exactly what I needed, thank you.'"
  "negative":  Customer expresses frustration, dissatisfaction, or complaint.
               Example: "Customer said 'This is unacceptable. I've been waiting three weeks.'"
  "neutral":   Customer is transactional — no emotional language.
               Example: "Customer said 'OK, noted. Is that all?'"

Actual label assigned: "{actual_value}"

Evaluate:
  1. Does the transcript contain evidence supporting the assigned label?
  2. If the label is wrong, which label from the allowed set is correct?
  3. Is the label absent entirely (is_missing=true)?

Output corrected_value only if the assigned label is wrong — use the correct label from the allowed set.
```

---

## 9. Evaluation Scenarios and Modes

The same pipeline handles four distinct scenarios by varying what is passed in and how results are compared.

**Scenario 1: Dev (Iterative Development)**

No golden set yet. Evaluation is reference-free. The goal is to surface failure modes and build the rubric.

```
Dev workflow:
  Run 1: reference_free=True → get ValidationObjects → inspect error_themes
  Run 2: refine rubrics in YAML → re-run same inputs → compare score distribution
  Run 3: human reviews 10 outputs → approves 10 as references → golden-v0.1.yaml
  Run 4: reference_based=True on those 10 → calibrate judge against human labels
```

**Scenario 2: Production Monitoring**

Sample 5% of live traffic. Async, reference-free. Focus on drift detection and latency.

```python
PROD_EVAL_CONFIG = {
    "sample_rate": 0.05,
    "reference_based": False,
    "metrics_to_track": [
        "is_correct_rate",       # fraction of fields correct per run
        "is_missing_rate",       # fragility signal: fields absent from source
        "error_theme_distribution",  # which errors are growing?
        "judge_latency_p95",
        "output_verbosity",      # token count trend — verbosity drift
    ],
    "alert_thresholds": {
        "is_correct_rate":     {"min": 0.82, "window": "7d"},
        "unit_confusion_rate": {"max": 0.05, "window": "1d"},
    }
}
```

**Scenario 3: Model Migration**

Evaluate both models against the same frozen golden dataset. Per-field comparison, not just aggregate.

```python
async def run_migration_comparison(
    baseline_provider: str, baseline_model: str,
    candidate_provider: str, candidate_model: str,
    golden_dataset_version: str,
    judge_config: str,
) -> MigrationReport:
    dataset = GoldenDataset.load(golden_dataset_version)

    baseline_results = await pipeline.evaluate_batch(
        dataset, provider=baseline_provider, model=baseline_model,
        judge_config=judge_config,
    )
    candidate_results = await pipeline.evaluate_batch(
        dataset, provider=candidate_provider, model=candidate_model,
        judge_config=judge_config,
        # Use cross-family judge when candidate is a different provider
        judge_override=get_cross_family_judge(candidate_provider),
    )

    return MigrationReport(
        per_field_delta=compute_field_deltas(baseline_results, candidate_results),
        error_theme_shift=compare_error_distributions(baseline_results, candidate_results),
        regression_fields=find_regressions(baseline_results, candidate_results, threshold=0.05),
        improvement_fields=find_improvements(baseline_results, candidate_results, threshold=0.05),
        bootstrap_significance=bootstrap_test(baseline_results, candidate_results, n=1000),
    )
```

> 🎯 **Interview prep**: The critical insight on model migrations is that aggregate score changes hide field-level regressions. A migration that improves summary quality by 8% while degrading account number extraction by 15% shows a net positive aggregate — but the account number regression is catastrophic for downstream data pipelines. Always break migration results down by field_path and field_type.

**Scenario 4: Prompt Changes**

Same model, same golden dataset, different prompt version. Pair each input's baseline and candidate outputs and run pairwise comparison on the delta fields.

```python
# Pairwise judge prompt addition:
PAIRWISE_SUFFIX = """
You are now comparing two extractions for the same field from the same transcript.

Extraction A (baseline prompt v{prompt_v1}): "{value_a}"
Extraction B (candidate prompt v{prompt_v2}): "{value_b}"

Using the same evaluation steps above, which extraction is more correct?
Respond with: {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}
"""
```

To mitigate position bias in pairwise evaluation, always evaluate both orderings (A vs B and B vs A) and flag disagreements:

```python
async def pairwise_eval(field, value_a, value_b, steps, config):
    result_ab = await judge(field, value_a, value_b, "A", "B", steps, config)
    result_ba = await judge(field, value_b, value_a, "B", "A", steps, config)

    if result_ab["winner"] != result_ba["winner"]:
        return {"winner": "inconclusive", "bias_detected": True}
    return result_ab
```

---

### 9.1 Validation Tool: Ground Truth Creation

During initial development, there is no golden dataset. The **Validation Tool** is a lightweight UI that bridges the gap — it lets a human reviewer create ground-truth labels from raw LLM outputs before any rubric-based evaluation runs.

```
VALIDATION TOOL WORKFLOW

┌─────────────────────────────────────────────────────────────────────┐
│                      VALIDATION TOOL UI                              │
│                                                                      │
│  Source Document    ──→  LLM Output          ──→  Reviewer Decision  │
│  (transcript, doc)        (extracted JSON            per field:      │
│                            or summary)                               │
│                                                  ✓ Correct          │
│                                                  ✗ Incorrect        │
│                                                    (enter correction)│
│                                                  ? Skip / Unsure    │
│                                                                      │
│  Reviewer provides:                                                  │
│    - Binary correct/incorrect per field                              │
│    - Corrected value (when incorrect)                                │
│    - Optional note on why it's wrong                                 │
│    - Missing severity: partial | complete (when is_missing=True)     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
                        Golden Dataset Entry
                        (stored as reference for
                         subsequent judge evaluations)
```

The Validation Tool serves two purposes:

1. **Bootstrap the golden set**: The first 50–100 manually reviewed samples become the initial `golden-v0.1.yaml`. These are the reference outputs the judge compares against in `reference_based=True` mode.

2. **Calibrate the judge**: Compare the judge's verdicts against the human labels. If the judge disagrees with humans on >15% of fields, the rubric needs revision before trusting the automated pipeline.

```python
# Validation tool output format — becomes a row in the golden dataset
{
  "input_id": "call_transcript_0047",
  "field_path": "contract.annual_value",
  "llm_actual_value": "48000",
  "human_verdict": "incorrect",
  "human_corrected_value": "4000 (monthly, not annual)",
  "human_note": "Model multiplied. Context clearly says monthly.",
  "missing_severity": null,
  "reviewed_by": "reviewer_id_42",
  "reviewed_at": "2025-05-20T14:32:11Z"
}
```

> The Validation Tool is a development-phase component. Once a stable rubric is established and judge accuracy exceeds the calibration threshold (typically ≥85% agreement with human labels), the production pipeline no longer requires manual validation — it runs autonomously with HITL reserved for prompt improvement suggestions.

---

## 10. Prompt Caching Strategy

Prompt caching is the single highest-leverage cost optimization in this system. The evaluation system's prompts are long (rubric YAML, error taxonomy, evaluation steps) and largely static across thousands of calls.

```
CACHE LAYERS AND WHAT THEY CACHE

Layer 1 — Evaluation Steps Cache (Redis, 24h TTL)
  Key:   SHA256(criteria_text + rubric_version)
  Value: list[str] of numbered evaluation steps
  Hit rate in practice: ~85% (same criteria applied to many records)
  Cost impact: eliminates 1 LLM call per field per eval run after warmup

Layer 2 — Judge Provider Prompt Cache (Anthropic/OpenAI)
  What: The static system prompt (judge role, error taxonomy, output format)
  How:  Anthropic: cache_control {"type": "ephemeral"} on the system message
        OpenAI:    prefix caching (automatic, >1024 tokens)
  TTL:  5 minutes (Anthropic ephemeral) — re-warm on cache miss
  Cost impact: 90% token cost reduction on the static prefix of every judge call

Layer 3 — Judge Result Cache (Redis, 1h TTL for prod / persistent for golden set)
  Key:   SHA256(field_path + actual_value + context_hash + criteria_version + judge_model)
  Value: full ValidationObject dict
  When:  Same input re-evaluated (dev iteration, golden set re-scoring)
  Cost impact: ~40% hit rate during active development
```

```python
# Anthropic prompt cache setup for the judge system prompt
JUDGE_SYSTEM_PROMPT_CACHED = [
    {
        "type": "text",
        "text": STATIC_JUDGE_PREAMBLE,  # role, output format, error taxonomy — never changes
        "cache_control": {"type": "ephemeral"},
    },
    {
        "type": "text",
        "text": dynamic_rubric_section,  # changes per judge_config — not cached at provider
    }
]
```

> 🏭 **Production note**: Monitor your cache hit rates per layer in your observability dashboard. A drop in Layer 1 hit rate signals rubric churn (someone is editing YAML too frequently). A drop in Layer 2 hit rate means your system prompt is varying between calls — audit the code that builds it. These two signals predict cost spikes before your invoice arrives.

---

## 11. Error Theme Extraction, Pattern Finder, and Human-in-the-Loop

After a batch run, `ValidationObject` records with `is_correct=false` are fed into an error theme analysis pipeline. The goal is to turn individual failures into auditable, human-reviewed prompt changes.

### 11.1 Error Theme → Prompt Improvement Pipeline

```
ERROR THEME → PROMPT IMPROVEMENT PIPELINE

Batch of ValidationObjects (is_correct=False)
    │
    ▼
Error Theme Aggregator
    │  Count occurrences per error_theme
    │  Compute frequency per field_path
    │  Rank by: frequency × impact_weight
    ▼
Pattern Priority Scores
    │
    │  e.g.  unit_confusion      → 34 occurrences → score 0.87 (HIGH)
    │        hallucinated_entity → 12 occurrences → score 0.61 (MED)
    │        format_error        → 3 occurrences  → score 0.21 (LOW)
    ▼
Pattern Finder                                ← distinct stage; clusters related
    │  Group errors that share a root cause      error records into a named,
    │  Identify field_paths most affected        reproducible pattern with
    │  Attach supporting ValidationObject IDs    evidence records attached
    ▼
Prompt Advisor (LLM call)
    │
    │  Inputs:  Top-N patterns from Pattern Finder
    │           Sample ValidationObject.reasoning for each pattern
    │           Current extraction prompt text
    │           Prompting Best Practices (static doc; cached)
    │  Output:  Ranked list of specific suggested edits,
    │           one per pattern, with rationale + confidence
    ▼
Prompt Improvement Suggestions
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              HUMAN IN THE LOOP                       │  ← governance gate
│                                                      │
│  For each suggestion, reviewer sees:                 │
│    Current Definition  (verbatim from IE prompt)     │
│    Issue Summary       (pattern + supporting records)│
│    Suggested Improvement (new prompt text)           │
│    Rationale            (why this fixes the pattern) │
│                                                      │
│  Decision:  ● Accept   Edit   Reject                 │
│                                                      │
│  Secondary: Reviewer note (free-text annotation)     │
└──────────────────┬──────────────────────────────────┘
                   │  Accept
                   ▼
       Applied to revisioned IE_prompt.py
       (version-tracked; rubric_version bumped)
                   │
                   ▼
          Metrics Calculations
          (new prompt version begins accumulating its own
           ValidationObject records for comparison)
```

### 11.2 Prompt Advisor Prompt

```python
PROMPT_ADVISOR_PROMPT = """
You are a prompt engineering expert.

The following extraction prompt is producing systematic errors:

[CURRENT EXTRACTION PROMPT]
{current_prompt}

[TOP ERROR PATTERNS FROM LAST BATCH]
{error_theme_summary}
# Format: error_theme | count | example_reasoning | example_field | example_actual_value

[TASK]
For each error pattern:
1. Identify the root cause in the current prompt (missing instruction, ambiguous wording, wrong example)
2. Write a specific, minimal change to the prompt that would prevent this error
3. If the error requires a new example in the prompt, provide the example

Respond with a JSON list of suggested changes:
[
  {
    "error_theme": "unit_confusion",
    "root_cause": "Prompt says 'extract the contract value' without specifying annual vs monthly",
    "suggested_change": "Add instruction: 'Extract the ANNUAL contract value in USD. If only monthly figures are stated, do not multiply — extract the monthly figure and add the suffix (monthly).'",
    "confidence": 0.92
  }
]
"""
```

### 11.3 Human-in-the-Loop: The Review Record

The Prompt Advisor's output is never deployed automatically. Each suggestion goes through a structured review UI before it touches the production prompt:

```
HUMAN IN THE LOOP — REVIEW RECORD

┌─────────────────────────────────────────────────────────────────────┐
│ [HIGH] interaction_details.mentions_of_past_historical_interaction  │
│        & any_prior_details → scope_restriction                       │
├──────────────────────────────┬──────────────────────────────────────┤
│ Current Definition           │ Suggested Improvement                │
│ (from IE prompt)             │                                      │
│                              │ "Did customer explicitly mention a   │
│ Is any detail of past        │ prior interaction/call with [Org]    │
│ interaction/calls            │ that occurred BEFORE this call?      │
│ mentioned? (Yes/No)          │ (Yes/No). EXCLUDE: current call      │
│                              │ events, account setup history,       │
│                              │ general customer practices,          │
│                              │ employment history, mentions of      │
│                              │ other financial accounts, or any     │
│                              │ historical context that is not a     │
│                              │ documented prior [Org] interaction." │
├──────────────────────────────┴──────────────────────────────────────┤
│ Issue Summary                                                        │
│                                                                      │
│ Records 106, 107, 158, 199, 204–206, 213–214, 226–227, 278–279 show │
│ the model conflating current call events (enrollment discussions,   │
│ check issues, account setup history, general practices, external    │
│ accounts) with past interactions. The definition does not           │
│ distinguish between "past historical interaction with [Org] prior   │
│ to this call" versus "current call context, account history,        │
│ employment history, or external accounts."                          │
├─────────────────────────────────────────────────────────────────────┤
│ Rationale                                                            │
│                                                                      │
│ Directly addresses recurring pattern from records 106–279. Current  │
│ prompt does not clearly distinguish between "past historical        │
│ interaction (prior to this call)" versus "earlier moments within    │
│ the current call" and does not explicitly exclude account setup     │
│ history or general customer practices. Added EXCLUDE list creates   │
│ hard boundaries for all observed conflation patterns.               │
├─────────────────────────────────────────────────────────────────────┤
│ Recommendation decision:  ● Accept   Edit   Reject                  │
│                                                                      │
│ Reviewer note: "Add explicit exclusions matching intent. Maintains  │
│ Yes/No format without introducing ambiguity to adjacent            │
│ date/summary fields."                                               │
├─────────────────────────────────────────────────────────────────────┤
│ Metadata                                                             │
│   Type: scope_restriction | Confidence: high | Section: interaction │
│   Prompt: prompt_current | Supporting records: 106, 107, 158, ...   │
└─────────────────────────────────────────────────────────────────────┘
```

Key properties of this governance step:

- **Evidence-backed decisions**: every suggestion arrives with the list of supporting `ValidationObject` record IDs so the reviewer can spot-check the actual judge reasoning, not just trust the summary.
- **Reviewer note**: the reviewer's free-text annotation is version-tracked alongside the decision — the audit trail records *why* a suggestion was accepted or rejected, not just *that* it was.
- **Edit path**: reviewers can modify the suggested prompt text before accepting; the edited version (not the raw Advisor output) is what gets versioned. This prevents blind acceptance of machine-generated text.
- **Reject path**: rejected suggestions are flagged in the error taxonomy so the same pattern does not resurface in the next Prompt Advisor run without new evidence.
- **Prompt version bumped on accept**: the IE prompt file is versioned (e.g. `IE_prompt_v2.3.py`), and all new `ValidationObject` records produced under it carry `rubric_version: v2.3` — enabling before/after comparison in the dashboard without re-running historical data.

> 🎯 **Interview prep**: "How do you prevent the Prompt Advisor from introducing regressions?" — The answer is the HITL gate plus version tracking. Human review prevents bad suggestions from shipping. Version tracking on every `ValidationObject` means you can filter the dashboard to `rubric_version = v2.3` and compare it against `v2.2` on the same golden dataset without any additional tooling.

### 11.4 Theme Assignment and Clustering: Bridge to the Metrics Layer

Raw `error_theme` labels also feed a parallel path into the metrics pipeline via the **Rollup Tester** — a validation gate that prevents stale taxonomy labels from silently corrupting dashboard scores.

```
THEME ASSIGNMENT AND CLUSTERING

Batch of ValidationObjects (is_correct=False) with error_theme labels
    │
    ▼
Theme Aggregator
    │  Group:  { error_theme → [ValidationObject, ...] }
    │  Weight: frequency × field_impact_weight
    │          (extractive fields weighted higher — precision-critical)
    ▼
Ranked Meta-Theme Clusters
    │
    │  unit_confusion       → 34 claims  (HIGH)
    │  hallucinated_entity  → 12 claims  (MED)
    │  scope_mismatch       →  6 claims  (MED)
    │  format_error         →  3 claims  (LOW)
    │
    ▼
Rollup Tester  ← validation gate; must pass before metrics are written
    │
    │  ✓ All error_theme labels map to entries in the active taxonomy
    │  ✓ correct_count + incorrect_count == total_claim_count
    │  ✓ Theme percentages are non-negative and sum to incorrect_count
    │  ✓ Section-level theme distributions show no sudden unexplained spikes
    ▼
Theme Scores  (per theme, per section, per run)
    │
    │  unit_confusion_rate:      0.11  (34 / 309 extractive claims)
    │  hallucinated_entity_rate: 0.04  (12 / 309)
    │  format_error_rate:        0.01  ( 3 / 309)
    ▼
Metrics Calculator → Overall is_correct_rate, per-section accuracy,
                     precision / recall on extractive fields, task-specific KPIs
```

The Rollup Tester exists because prompt or taxonomy changes can orphan existing `error_theme` labels — claims with stale themes are silently dropped from theme scores, making error rates appear to fall when they haven't.

---

## 12. Report Rollup and Benchmarking Dashboard

The **Report Rollup** converts claim-level `ValidationObject` records into the structured reports and dashboards that engineering and product teams consume. Every metric is derived from this aggregation hierarchy.

### 12.1 Report Rollup Pipeline

```
REPORT ROLLUP PIPELINE

┌─────────────────────────────────────────────────────────────────────┐
│                         CLAIM REVIEW                                 │
│  All ValidationObjects for the run (one per field / per claim)       │
│  Fields: field_path, is_correct, error_theme, confidence, evidence   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CLAIM SCORES                                  │
│  Per-field binary score (0/1 for extractive/inferential)             │
│  Per-field rubric_scores dict (for free_form + custom metric fields)  │
│  Missing penalty: is_missing=true, severity=partially → 0.5          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    THEME-ASSIGNED CLUSTERS                            │
│  Rollup Tester validates taxonomy integrity (see §11.4)              │
│  Theme scores computed per section (customer / contract / risk_flags)│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      METRICS CALCULATOR                               │
│                                                                       │
│  Overall           is_correct_rate, is_missing_rate, avg_confidence  │
│  Per section       section_accuracy (weighted by field count)        │
│  Per error theme   theme_rate, theme_trend_7d                        │
│  Precision/Recall  for structured JSON extraction tasks              │
│  Custom metrics    rubric_scores[metric_name] averages per run       │
│  Cost              total_judge_cost, per_call_cost, cache_savings    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       Task-Specific    Per-Model Run      Cross-Run Trend
         Metrics          Summary              Delta
```

### 12.2 Dashboard: Five-Tab Structure

```
TAB 1 — METRICS
  Single Territory (one model, one run):
    Overall is_correct_rate, missing_rate, avg_confidence
    Per-section breakdown (customer: 0.93 | contract: 0.84 | risk_flags: 0.79)
    Per-field heatmap  (which field_paths are failing?)

  Dual Territory (cross-model or cross-prompt comparison):
    Side-by-side scorecards for two runs / two models
    Per-field delta  (where did the candidate improve or regress?)
    Statistically significant regressions flagged (bootstrap p < 0.05)

  Compare Performance — 9-metric model leaderboard:
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  Model               │ Precision │ Recall │  F1  │ Completeness │ Helpfulness │ Cost/1k │ Latency p95 │ Tokens │ Judge Eval
  ────────────────────┼───────────┼────────┼──────┼──────────────┼─────────────┼─────────┼─────────────┼────────┼───────────
  gpt-4o-2024-11-20   │   0.93    │  0.89  │ 0.91 │    0.88      │    0.91     │ $18.20  │  1,840 ms   │  412   │   0.87
  claude-sonnet-4-6   │   0.91    │  0.87  │ 0.89 │    0.86      │    0.89     │ $12.40  │  1,210 ms   │  387   │   0.85
  gpt-4o-mini         │   0.85    │  0.81  │ 0.83 │    0.79      │    0.83     │  $2.10  │    680 ms   │  401   │   0.79
  llama3.1-70b-ollama │   0.80    │  0.76  │ 0.78 │    0.72      │    0.77     │  $0.00  │  2,300 ms   │  443   │   0.71
  ────────────────────────────────────────────────────────────────────────────────────────────────────────

  Metric definitions:
    Precision       — of all extracted claims, fraction that are correct
    Recall          — of all ground-truth claims, fraction the model extracted
    F1              — harmonic mean of precision and recall
    Completeness    — fraction of required fields present (not null / not missing)
    Helpfulness     — avg(rubric_scores["helpfulness"]) on template summary section fields
    Cost/1k         — judge + generation cost per 1,000 evaluation calls
    Latency p95     — 95th-percentile end-to-end latency
    Tokens          — average output token count (verbosity signal)
    Judge Eval      — judge self-consistency score; measures how reliably the
                      judge reaches the same verdict when re-run on the same input

TAB 2 — ERROR THEMES
  Single Territory:
    Ranked list of error themes by frequency and field impact
    Drill-down per theme: sample claims, evidence snippets, field_paths affected

  Dual Territory (cross-model):
    Which error themes grew / shrank between model versions?
    Theme shift table: baseline_rate → candidate_rate → delta

TAB 3 — PROMPT IMPROVEMENTS
  Input:  Top error themes from current run + Prompt Advisor output (see §11)
  Output: Ranked list of suggested prompt edits

  ─────────────────────────────────────────────────────────────────────
  Priority │ Error Theme        │ Suggested Change           │ Confidence
  ─────────┼────────────────────┼────────────────────────────┼───────────
  HIGH     │ unit_confusion     │ Specify annual vs monthly  │  0.92
  MED      │ hallucinated_entity│ Add "verbatim only" guard  │  0.81
  LOW      │ format_error       │ Normalize date format ex   │  0.74
  ─────────────────────────────────────────────────────────────────────
  Status: pending human review before deployment

TAB 4 — DATA EXPORT  (Flattened Data from Judge)
  Claim-level export: one row per ValidationObject
  ──────────────────────────────────────────────────────────────────────────────────────────
  field_path           │ actual_value │ corrected_value │ is_correct │ is_missing │ error_theme        │ reason_for_incorrect
  ─────────────────────┼─────────────┼─────────────────┼────────────┼────────────┼────────────────────┼──────────────────────
  contract.annual_value│ 48000       │ 4000            │ false      │ false      │ unit_confusion     │ Rubric: 'do not multiply monthly to annual'
  customer.account_num │ 4829301020  │ null            │ true       │ false      │ null               │ null
  customer.name        │ null        │ null            │ false      │ true       │ missing_from_source│ source_document_does_not_contain_this_information
  ──────────────────────────────────────────────────────────────────────────────────────────
  Sortable / filterable by any column
  Used for: golden-set labeling, audit trails, manual spot-checks, Validation Tool seed

TAB 5 — RUN SUMMARY
  Report context:    task_type, judge_config_version, model, prompt_version
  Per-call cost:     total_cost, input_tokens, output_tokens, cache_hit_rate
  Run metadata:      timestamp, sample_count, evaluation_mode (single/pairwise)
  Notes:             free-text field for human annotation (linked to run_id)
```

> 🎯 **Interview prep**: The "Dual Territory" comparison view is the most operationally important report during a model migration. It exposes the hidden cost of aggregate-only evaluation: a migration that improves `is_correct_rate` by 3% overall can still contain field-level regressions on extractive fields that cascade into broken downstream pipelines. Always drill into the per-field delta before signing off on a migration.

> 🏭 **Production note**: Tie the Score Trend view (is_correct_rate over time per run) to your alerting system. A 3% drop in `is_correct_rate` sustained over two consecutive daily runs should page the oncall engineer — not wait for a quarterly model review.

---

## 13. Observability

Every judge call is wrapped in an OpenTelemetry span. This gives you distributed traces across the full evaluation pipeline — from request ingestion through claim decomposition, step generation, judge execution, and result storage.

```python
from opentelemetry import trace

tracer = trace.get_tracer("llm-eval")

async def judge_claim_instrumented(claim, steps, config, req):
    with tracer.start_as_current_span("judge_claim") as span:
        span.set_attribute("field_path", claim.field_path)
        span.set_attribute("field_type", claim.field_type)
        span.set_attribute("judge.provider", config.primary_judge["provider"])
        span.set_attribute("judge.model", config.primary_judge["model"])
        span.set_attribute("cache.hit", False)  # updated below

        result = await judge_claim(claim, steps, config, req)

        span.set_attribute("cache.hit", result.cached)
        span.set_attribute("is_correct", result.is_correct)
        span.set_attribute("error_theme", result.error_theme or "none")
        span.set_attribute("latency_ms", result.latency_ms)
        span.set_attribute("confidence", result.confidence)

    return result
```

Key spans to instrument: `pre_validate`, `claim_decompose`, `eval_steps_generate`, `judge_call` (one per field), `post_validate`, `result_store`. This lets you identify which stage is the latency bottleneck per task type.

---

## 14. Design Critique and Known Risks

This section is an honest assessment of the approach — where it is strong, where it has gaps, and where teams building on this design should invest additional effort.

### Strengths

**1. Claim-level granularity closes the feedback loop.** Every open-source framework gives you a score. This design gives you a reason — `reason_for_incorrect` citing the specific rubric that failed, plus verbatim evidence. This is the single biggest gap in the current tool ecosystem (see §0 Gap Map, blind spot #1), and it makes prompt debugging hours instead of days.

**2. Metric-agnostic via `rubric_scores: dict[str, float]`.** Adding toxicity, coherence, or domain-specific metrics requires zero code changes — only YAML additions. Multiple custom metrics can coexist on the same field in a single ValidationObject. This mirrors how the best production ML systems treat feature engineering: declarative, not imperative.

**3. The HITL governance loop is a first-class component, not an afterthought.** Most teams discover the need for human review after their first bad model migration. This design builds the review queue, version tracking, and audit trail in from day one.

**4. Prompt caching is architecturally mandatory.** The evaluation steps cache + provider prompt cache + judge result cache means that at steady state, most of the evaluation cost is eliminated. This makes production-volume evaluation economically viable — something no open-source framework handles well.

### Gaps and Risks

**1. Claim decomposition for free-form text is the weakest link (deferred to Phase 2).**

Splitting a free-form summary into "atomic claims" introduces errors that compound downstream: over-splitting inflates recall, under-splitting gives compound claims a single binary verdict that hides partial correctness.

*Resolution (selected)*: Free-form summarization and free-form text generation are deferred to Phase 2. MVP uses only deterministic decomposition (JSON extraction, template summarization, classification) — the decomposer never fails or produces ambiguous claim counts. For Phase 2, a dedicated Haiku call with a fixed decomposition prompt (max 10 claims, numbered list output) isolates decomposition failures from judge failures.

**2. The `reason_for_incorrect` rubric citation requires well-written rubrics.**

Vague rubrics produce vague citations. "The value does not meet the rubric" is not useful; "Rubric: 'failure_condition: If stated as monthly, do not multiply' — extracted $48,000 is monthly $4,000 × 12" is. The structured rubric schema forces the specificity the judge needs.

*Resolution (selected)*: Structured rubric schema (§5) replaces freetext rubric lines with typed fields: `criteria`, `failure_condition`, `expected_format`, `example_pass`, `example_fail`. The judge prompt template renders all five fields, ensuring the judge always has a concrete failure example to cite. Rubric vagueness becomes a YAML authoring problem, not a runtime problem.

**3. Evaluation cost without cascading.**

A frontier model on every claim is economically unviable at production volume. The cascading judge resolves this: cheap model handles ~75–80% of easy claims, frontier model handles only uncertain or ambiguous cases (~20–25%).

*Resolution (selected)*: Cascading judge (§7.2) — Tier 1 (Haiku/GPT-4o-mini) accepts claims with `confidence ≥ 0.85`; Tier 2 (Sonnet/GPT-4o, cross-family) handles escalations. Golden set and migration runs always use Tier 2 regardless of confidence. Expected cost: ~25–30% of frontier-only evaluation at equivalent quality.

**4. Rubric and prompt version attribution.**

If rubric version and prompt version change simultaneously, score deltas cannot be attributed to either cause. The `eval_version` composite key makes this explicit on every ValidationObject.

*Resolution (selected)*: `eval_version: str = "rubric=contracts-v2.1,prompt=v3.4"` is stored on every ValidationObject. This makes simultaneous changes visible in the dashboard — any run where two `eval_version` values differ in both rubric and prompt components should be manually flagged. The Production Deployment Checklist §16 enforces this as a procedural gate.

**5. Evidence post-validation (substring match) is not exhaustive.**

Substring matching catches hallucinated verbatim quotes. It does not catch paraphrased hallucinations or correctly quoted evidence used to support the wrong verdict.

*Resolution (not implemented — deferred)*: Substring matching is sufficient for MVP. The "Judge Eval" metric in the 9-metric leaderboard (§12) tracks judge self-consistency as a proxy for evidence quality: if the judge produces different verdicts on re-run, its evidence is likely unreliable. Route `judge_inconsistency_rate > 10%` claims to HITL in a future iteration.

**6. Free-form summarization calibration.**

Free-form atomic claim evaluation has fewer benchmarks than extraction tasks. The 3-rubric structure (accuracy + completeness + hallucination per claim) is the correct approach for Phase 2 but is not MVP scope.

*Resolution (selected)*: Deferred to Phase 2 per L6 decision. MVP scope is JSON extraction, template summarization, and classification — all deterministic decomposition. This de-risks the MVP calibration entirely.

### How This Design Resolves Framework Gaps (Linked)

Connecting back to the §0 Gap Map: every design decision here traces to a specific framework limitation:

| Design Decision | Framework Gap It Closes | Evidence |
|---|---|---|
| `ValidationObject` per claim, not per response | Gap #1: No claim-level attribution (all 7 frameworks) | DeepEval context scoring: 0.46 vs 0.82–0.91 peers; Promptfoo: no field-level granularity |
| Pattern Finder → Prompt Advisor → HITL loop | Gap #2: No closed-loop prompt improvement | LangSmith: observational only; DeepEval/RAGAS: no improvement mechanism |
| Cross-family judge + dual-ordering pairwise | Gap #3: No systematic bias mitigation | G-Eval: 2.5%–82.5% win-rate shift from ordering; 87.76% self-preference rate |
| Production Sampler + async queue (Redis/SQS) | Gap #4: Production requires paid tier | Promptfoo: SQLite "not production-ready"; LangSmith: cloud-only GDPR friction |
| `rubric_version` on every ValidationObject | Gap #5: Rubric changes silently invalidate history | No framework versions rubrics alongside evaluation records |
| `reason_for_incorrect` with rubric citation | New: Root cause attribution | No framework attributes a failure to a specific criterion definition |
| `<corrected>` tag pattern | New: Correction visibility | No framework surfaces the judge's corrected value alongside the wrong extraction |

---

## 15. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Position bias** | Pairwise judge systematically prefers first response | Evaluate both orderings; flag disagreements as inconclusive |
| **Verbosity bias** | Judge rewards longer responses regardless of quality | Add explicit length rubric; track is_correct vs output_tokens correlation |
| **Self-preference** | GPT-4o gives higher scores to GPT-4o outputs | Use cross-family judge (Claude judges GPT, GPT judges Claude) |
| **Sycophancy** | Adding "expert says X" in prompt makes judge agree | Never include opinions in judge prompt; audit prompt template quarterly |
| **Evidence hallucination** | Judge fabricates quotes that don't exist in context | Post-validate: check every `evidence.text` substring exists in `context` |
| **Context window collapse** | Judge misses errors in middle of long contexts | Evaluate per-claim, not full response; chunk long contexts |
| **Rubric drift** | Rubric edits change scores without changing extraction quality | Version rubrics; store rubric_version on every ValidationObject |
| **Claim decomposition errors** | Over/under-splitting free-form text changes evaluation count | Cap claims per response; validate decomposition separately |
| **Rubric vagueness** | Vague rubrics produce vague `reason_for_incorrect` citations | Treat rubric writing as a design task; each line must be human-applicable |

> 🎯 **Interview prep**: "What's the hardest problem in building an LLM evaluator?" — Evidence hallucination in the judge is the one most teams miss. The judge can produce confident, plausible-sounding reasoning with verbatim-looking quotes that don't actually exist in the context. This makes `is_correct=false` verdicts look well-supported when they're fabricated. Post-validation substring matching on every evidence quote is mandatory, not optional.

---

## 16. Production Deployment Checklist

```
Before first run:
  [ ] Judge YAML configs reviewed and versioned (contracts-v1.0.yaml)
  [ ] Error taxonomy YAML defined with domain-specific themes
  [ ] LLM adapters tested for all providers in use (openai, anthropic, ollama)
  [ ] Pre/post validation schemas defined per task type
  [ ] Prompt cache warmup run (generate + cache eval steps for all criteria)
  [ ] Judge model pinned to specific version (not "gpt-4o", use "gpt-4o-2024-11-20")
  [ ] Async queue configured (Redis / SQS) with dead-letter queue
  [ ] Result store schema migrated (ValidationObject table + indexes)
  [ ] OTel instrumentation verified (check spans appear in your observability tool)

Before going to production monitoring:
  [ ] Sample rate configured (5% default; adjust based on volume)
  [ ] Alert thresholds set per metric per judge_config
  [ ] Pairwise bias mitigation enabled (dual-ordering) for any comparison runs
  [ ] Cross-family judge configured (don't judge with same provider as generator)
  [ ] Evidence post-validation enabled (substring check on every evidence quote)
  [ ] HITL review queue set up (prompt improvement suggestions routed to reviewers)
  [ ] Prompt version bumping automated on HITL Accept decision

Before each model migration:
  [ ] Golden dataset frozen and versioned (golden-contracts-v1.2.yaml)
  [ ] Baseline scores recorded for current model on frozen dataset
  [ ] Migration report template configured (per-field delta + error theme shift)
  [ ] Regression thresholds set per field_path (not just aggregate)
  [ ] Bootstrap significance test configured (n≥1000, α=0.05)
  [ ] Dual Territory dashboard view confirmed working for the two model versions
```
