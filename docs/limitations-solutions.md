Here's the complete review. I'll present the analysis, 3 options per limitation with my recommendation starred, then the usage guide. Once you confirm which options to implement, I'll update the document.

  ---
  Complete Design Review

  Task-Agnostic Assessment

  What works well:

  The routing through field_type: extractive | inferential | free_form plus task_type on the evaluation request is genuinely task-agnostic — the judge pipeline doesn't change, only the YAML config does. The ValidationObject, error taxonomy, caching layers, and HITL loop are all task-neutral.

  Gaps:

  The design currently treats all 5 in-scope tasks as if they decompose cleanly. They don't equally well:

  ┌───────────────────────────┬───────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │           Task            │                 Decomposition Quality                 │                                                     Gap                                                     │
  ├───────────────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ JSON Extraction           │ Strong — one VO per field_path                        │ None. Deterministic.                                                                                        │
  ├───────────────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Template Summarization    │ Strong — template → JSON → one VO per section         │ None. Converts to the extraction problem.                                                                   │
  ├───────────────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Classification            │ Partially specified — "one VO per label" is ambiguous │ Multi-label vs single-label is unspecified. actual_value for a label isn't the same as an extracted string. │
  ├───────────────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Free-Form Summarization   │ Weak — LLM decomposition is fragile                   │ Claim count varies, compound claims slip through.                                                           │
  ├───────────────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Free-Form Text Generation │ Missing — not distinct from free-form summarization   │ Bullet generation, rewrite tasks need separate rubric guidance.                                             │
  └───────────────────────────┴───────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  The design principle §1 still says "chatbot responses, RAG, agentic AI" — this contradicts the MVP scope. The principle needs updating to match scope.

  ---
  Metric-Agnostic Assessment

  What works well:

  rubric_score: Optional[float] + judge_strategy: rubric_scoring in YAML is the right extension mechanism. Adding toxicity, coherence, or domain-specific metrics needs no code changes.

  Gap — single rubric_score per ValidationObject:

  A field can only carry one custom rubric_score. If you want to score the same field on both coherence (0.0–1.0) AND toxicity (0.0–1.0), you need two separate ValidationObjects for the same field. This doubles record count and makes aggregation confusing.

  Recommended fix: Change rubric_score: Optional[float] to rubric_scores: dict[str, float] — a map from metric name to score. This is the metric-agnostic design it claims to be.

  Gap — no metric aggregation spec for custom metrics:

  The Metrics Calculator (§12) defines aggregation for is_correct_rate and is_missing_rate but not for custom rubric_score metrics. The 9-metric leaderboard has fixed columns, which contradicts metric-agnostic design.

  ---
  Judge Prompt Targeted Improvements

  Reading the prompt structure in §7 carefully, here are the specific issues:

  Problem 1 — actual_value field is used twice with different meanings.
  The prompt input says Actual value extracted by the AI: "{actual_value}" and the output format also has "actual_value": "<same if correct; <corrected>...</corrected> if incorrect>". This is the same field name for two different things. The output actual_value should be renamed corrected_value in the JSON output schema (the Python model already has corrected_value — but the judge prompt doesn't output it that way). This creates a
  parsing ambiguity.

  Problem 2 — Evaluation steps are generated separately from the judge call.
  The eval steps (generated from criteria, cached in Redis) are injected into the judge prompt. If the steps are stale (cache hit from an older rubric), the steps and rubric are out of sync. The judge runs steps that no longer match the active rubric.

  Problem 3 — Classification task has no structured handling.
  For classification, actual_value is a label string ("positive", "churn_risk"), not an extracted entity. The judge needs different instructions: "Does the assigned label match the criteria? What label should it be?" But the current prompt has no branch for this.

  Problem 4 — Long context degrades judge quality.
  The full context is passed as-is. For documents over ~4,000 tokens, the judge suffers the "lost in the middle" problem — evidence from the middle of the context is systematically missed. There's no context relevance filtering before the judge call.

  Problem 5 — The output format asks for freeform JSON without native structured output.
  The prompt says "respond with valid JSON only" — but Anthropic and OpenAI both support native structured output (tool use / response_format). Native structured output eliminates the json_repair fallback need entirely and reduces parsing errors to near zero.

  ---
  Limitations with 3 Options Each

  ---
  L1 — Claim decomposition for free-form text is fragile

  Option A: Rule-based splitter (spaCy sentence tokenizer + deduplication)
  Split on sentence boundaries using spaCy, then cosine-similarity dedup to remove paraphrased duplicates (threshold 0.88). Zero LLM cost. Fast (milliseconds). Fails on complex multi-clause sentences — "The agent offered a discount, which the customer rejected" becomes one claim that is half-correct.

  Option B★ (recommended): Dedicated decomposition call using a small, cheap model
  A separate, constrained LLM call (Claude Haiku or GPT-4o-mini) with a fixed structured prompt: "Split this text into numbered atomic claims. Each claim must be independently verifiable. Maximum 10 claims." Output is a numbered list. Cost: ~$0.001/response. This model is not the judge — it's just a splitter. Failures are detectable (empty output, malformed list) and retriable.

  Option C: NLI-based decomposition + verification (DeBERTa-v3-large)
  Fine-tuned NLI model handles both splitting and entailment checking. Open-source, no per-call cost after deployment. Requires hosting and calibration against your domain. Best at detecting subtle claim-context contradictions, but overkill for MVP and adds operational complexity (another model to monitor).

  Why B: The problem is that free-form decomposition quality directly controls everything downstream. A lightweight LLM with a fixed, tested decomposition prompt is the right tradeoff: interpretable failures, retryable, cheap, and calibratable by reading the outputs. Commit to a max-10-claims cap to bound edge cases.

  ---
  L2 — reason_for_incorrect quality depends on rubric quality

  Option A: Rubric linting at load time
  A YAML linter validates each rubric field: minimum 25 words, must contain at least one of {must, shall, never, only if, do not, exactly}, must include a failure example (failure_example: key). Rejects the config before evaluation runs. Catches vague rubrics but doesn't tell you if a well-worded rubric is semantically clear.

  Option B★ (recommended): Structured rubric schema — replace freetext with typed fields
  Replace each rubric line with a structured block:
  annual_value:
    criteria: "Total annual contract value in USD."
    failure_condition: "If the stated period is monthly, do not multiply."
    expected_format: "Integer, no currency symbol"
    example_pass: "48000"
    example_fail: "4000 (monthly stated but annualized)"
  Forces specificity at authoring time. The judge prompt template renders criteria + failure_condition + example_pass/fail — the judge always has a concrete failure example to cite in reason_for_incorrect. This makes the citation unambiguous.

  Option C: Rubric calibration gate before production
  Before a new rubric ships, run it against 20 golden samples and verify reason_for_incorrect contains the rubric text in ≥80% of incorrect cases. Automated check — if citation rate < 80%, block the rubric from production and return a calibration report. Good as a gating mechanism but doesn't prevent bad rubrics from being written — just catches them later.

  Why B: Structured templates prevent the problem at the source. Combined with Option C as a gating check, this is the complete solution. Option A (linting) catches structural issues but not semantic vagueness.

  ---
  L3 — Committee of judges doubles or triples cost

  Option A: Self-consistency via temperature sampling (same model, 3 runs)
  Run the same judge 3× at temperature=0.4. 3/3 agree → high confidence, accept. 2/3 agree → medium confidence, accept with flag. 0/3 or split → route to HITL. Cost: 3× single-judge, but same provider so no cross-family bias mitigation. Cheaper than multi-model but doesn't catch self-enhancement bias.

  Option B★ (recommended): Cascading judge (cheap-first, escalate on uncertainty)
  Run Haiku/GPT-4o-mini first. If confidence ≥ 0.85 and verdict is clear (is_correct=True or a common error theme), accept. If confidence < 0.85 or verdict is ambiguous, escalate to Sonnet/GPT-4o. For golden set creation and model migration runs, always escalate to cross-family committee. In practice, ~75-80% of claims (clear extractions, common errors) are resolved by the cheap model.

  Option C: Spot committee on stratified 1% sample
  Apply full 2-model committee to 1% of production traffic, stratified by field_type and error_theme distribution. Use the agreement rate as an estimate of single-judge reliability across the full 5% sample. This gives you a statistically valid reliability signal without paying committee cost for every claim.

  Why B: Cascading is the production-proven pattern (used in medical diagnosis AI, content moderation, and LLM routing). It makes the cheap model the workhorse and the expensive model the arbitrator. Combine with Option C as a periodic calibration check: spot-committee 1% to verify the escalation threshold is still well-calibrated.

  ---
  L4 — Rubric version and prompt version tracked separately (attribution ambiguity)

  Option A: Composite eval_version key
  Introduce eval_version = f"rubric={rubric_version},prompt={prompt_version}" stored on every ValidationObject. This makes the combination explicit but doesn't prevent simultaneous changes — it just makes them visible.

  Option B★ (recommended): Change isolation gate in the submission layer
  Before an evaluation run starts, compare current_rubric_version and current_prompt_version against the last run. If both have changed, reject the run with: "Cannot evaluate with simultaneous rubric and prompt changes. Separate into two sequential runs." This is a hard architectural constraint, not a suggestion. Implemented as a pre-flight check in the submission layer.

  Option C: Causal attribution via fixed reference set
  Maintain a fixed 50-sample reference set (from the golden dataset) that is run automatically whenever either rubric or prompt changes. By comparing scores on the same 50 inputs before/after each type of change, you can attribute score shifts to rubric vs prompt causally. Requires the fixed reference set to be maintained and adds latency to each change.

  Why B: The gate is the right call. Attribution scoring (C) sounds appealing but is fragile — you need a reference set that covers all error modes, which is hard to guarantee. The gate makes the constraint explicit and enforces it at submission time, which is the correct place.

  ---
  L5 — Evidence post-validation (substring matching) is necessary but not sufficient

  Option A: Semantic similarity threshold (embedding model)
  Compute cosine similarity between each evidence.text and every sentence in the context using bge-small-en-v1.5 (fast, cheap). Evidence is valid if max similarity > 0.90 to any context sentence. Catches paraphrased hallucinations. Adds ~10ms per evidence quote. Requires hosting an embedding model.

  Option B★ (recommended): Judge self-consistency re-run on flagged claims
  For any claim where the judge's confidence < 0.80 OR the evidence fails substring validation, re-run the judge call once (at temperature=0.0). If the verdict flips, route to HITL and flag as judge_inconsistent=True in the ValidationObject. If it agrees, accept with double_checked=True. Track the judge_inconsistency_rate in the 9-metric leaderboard — if it exceeds 10%, the judge model or rubric has a problem.

  Option C: Separate evidence re-verification call
  For each evidence quote, make a standalone cheap LLM call: "Does the following quote appear verbatim or near-verbatim in the source document? Answer YES or NO with a confidence score." More precise than substring match (handles minor whitespace/formatting differences) and cheaper than re-running the full judge. Adds one extra LLM call per evidence quote — cost depends on evidence density.

  Why B: Self-consistency re-run is the most comprehensive check because it catches all judge unreliability simultaneously — hallucinated evidence, flipped verdicts, paraphrased quotes — with a single mechanism. It also produces a calibration signal (judge_inconsistency_rate) that you can track and alert on. Option C (evidence verification) is complementary but adds more calls; consider it as a future layer after the pipeline is
  stable.

  ---
  L6 — Free-form summarization harder to calibrate than extraction
  
  Option A: Defer free-form to Phase 2
  MVP evaluates only JSON extraction, template summarization (reduces to JSON), and classification. Free-form summarization is Phase 2 after the pipeline is calibrated on the structured tasks first.

  Option B★ (recommended): 3-rubric structure for free-form claims
  Every free-form claim is evaluated on three independent rubric dimensions, each binary: (1) Factual accuracy — is the claim grounded in the source document? (2) Completeness — does the claim miss critical qualifying information? (3) Hallucination check — does the claim introduce content not in the source? Each produces its own is_correct flag. The composite score for a free-form claim is the average of the three. This gives
  structured calibration targets: if accuracy is high but hallucination is high, your LLM is adding fabricated context to real facts.

  Option C: Hybrid scoring (judge + ROUGE-L)
  Run both judge evaluation AND ROUGE-L against reference summaries. If ROUGE-L > 0.65 AND judge is_correct=True, high confidence. If they disagree (good judge score but low ROUGE-L, or vice versa), route to HITL. ROUGE-L is a weak proxy for semantic quality but is useful as a sanity check for complete hallucinations (ROUGE-L ~0 when the model writes something completely different from the reference).

  Why B: The 3-rubric structure converts free-form evaluation into three binary questions — the same model as extraction. This makes calibration tractable: you can independently measure accuracy rate, completeness rate, and hallucination rate, and track each separately in the dashboard. Option C is a reasonable addition for Phase 2, but ROUGE-L requires reference summaries which may not exist.

  ---
  L7 (new) — rubric_score: Optional[float] can only hold one custom metric per field
  
  Option A★ (recommended): Change to rubric_scores: dict[str, float]
  Replace the single float with a named dict: rubric_scores: dict[str, float] = {}. YAML declares metric names; the judge outputs {"toxicity": 0.1, "coherence": 0.87}. The Metrics Calculator aggregates each key independently. This is the correct metric-agnostic design — any number of metrics, no ValidationObject duplication.

  Option B: One ValidationObject per metric per field
  Keep rubric_score: Optional[float] but create multiple ValidationObjects for the same field_path when multiple custom metrics apply — differentiated by a metric_name field. More records, harder to join in queries.

  Option C: Separate CustomMetricObject model
  Keep ValidationObject as the binary/missing core model. Add a separate CustomMetricObject model for scored metrics, stored in a parallel table. Clean separation of concerns but adds a second data model and a second aggregation pipeline.

  Why A: Dict is the cleanest. It's still one record per field, supports arbitrary metrics, and the Metrics Calculator just iterates the dict keys. The YAML already has rubric_scale per field — the key is the metric name.

  ---
  L8 (new) — Design Principle §1 contradicts MVP scope
  
  The first design principle says "handle summary evaluation, JSON extraction, chatbot responses, RAG, and agentic AI". The MVP scope explicitly excludes chatbots, RAG, and agents.

  Fix: Update §1 Design Principle #1 to match the MVP scope, with a note that the architecture is extensible to the broader scope.

  ---
  L9 (new) — Classification task is underspecified

  There's no concrete handling for how classification maps to ValidationObject. For classification:
  - actual_value is a label ("churn_risk", "positive"), not extracted text
  - corrected_value is the correct label
  - is_missing means the model returned no label
  - The judge needs to know the allowed label set and the criteria per label

  Option A★ (recommended): Classification rubric block in YAML
  Add a classification field type in the YAML with allowed_labels and criteria_per_label. The judge receives the allowed label set and the criteria for each, and evaluates: "Is this the correct label? If not, what should it be?" Generates a corrected_value with the right label.

  Option B: Binary per-label evaluation
  For multi-label classification, one ValidationObject per label in allowed_labels. Each is binary: was this label assigned correctly? This handles multi-label naturally but produces N ValidationObjects for N labels even when most are correctly absent.

  Option C: Pass-through to free_form rubric
  Treat classification as a free_form evaluation where the rubric is "The assigned label must be X because Y." This reuses existing infrastructure but loses the structured label-set constraint.

  Why A: Native classification support with allowed_labels + criteria_per_label is the right design. It handles both single-label and multi-label naturally (multi-label gets multiple entries in the same YAML block), and gives the judge the explicit constraint it needs to produce a meaningful corrected_value.

  ---
  My Recommended Implementation Set
  
  Before I update the document, here are the options I'm recommending — confirm or change any:

  ┌────────────────────────────┬───────────────────────────────────────────────────────────────────────┬─────────────────────────────────────┐
  │         Limitation         │                                My Pick                                │       Alternative to consider       │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L1 Free-form decomposition │ B — Dedicated cheap model (Haiku) with max-10-claims cap              │ C if you want zero ongoing LLM cost │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L2 Rubric quality          │ B — Structured rubric schema (criteria + failure_condition + example) │ Add C as a CI gate on top           │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L3 Committee cost          │ B — Cascading judge (cheap-first) + C as spot calibration             │ —                                   │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L4 Version attribution     │ B — Submission layer change isolation gate                            │ —                                   │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L5 Evidence validation     │ B — Self-consistency re-run on low-confidence claims                  │ Add C later for extra coverage      │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L6 Free-form calibration   │ B — 3-rubric structure (accuracy + completeness + hallucination)      │ —                                   │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L7 Single rubric_score     │ A — Change to rubric_scores: dict[str, float]                         │ —                                   │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L8 Principle mismatch      │ Fix §1 to match MVP scope                                             │ —                                   │
  ├────────────────────────────┼───────────────────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ L9 Classification gap      │ A — Native classification YAML block with allowed_labels              │ —                                   │
  └────────────────────────────┴───────────────────────────────────────────────────────────────────────┴─────────────────────────────────────┘

  ---
  Usage Guide: Production Monitoring, Prompt Comparison, and Model Leaderboard

  Scenario 1 — Production Monitoring

  Your live system extracts JSON from call transcripts all day.
  You want to know: "Is quality degrading? Which fields are failing more often?"

  SETUP:
    1. In your service, after every LLM call, push a sampling event:
         eval_queue.push({
             "input": transcript,
             "output": extracted_json,
             "judge_config": "contracts-v2.1.yaml",
             "scenario": "prod",
             "model_meta": { "provider": "anthropic", "model": "claude-sonnet-4-6",
                             "prompt_version": "v3.4" }
         })

    2. The sampler accepts 5% of events (reservoir sampling — not every 20th call,
       which would miss burst failures).

    3. The async evaluation queue picks them up, runs the judge pipeline,
       writes ValidationObjects to PostgreSQL.

  WHAT YOU SEE:
    Dashboard Tab 1 (Metrics):
      is_correct_rate this week:   0.89  ↓ from 0.93 last week  ⚠
      is_missing_rate this week:   0.04  stable
      Worst field:  contract.annual_value  0.71 ✗

    Dashboard Tab 2 (Error Themes):
      unit_confusion: 34 occurrences (+18 vs last week) ← flagged

  ALERTS:
    Configured threshold: is_correct_rate < 0.85 for 2 consecutive daily runs → page oncall
    unit_confusion_rate > 0.05/day → Prompt Advisor triggered automatically

  ACTION:
    Pattern Finder clusters the unit_confusion errors.
    Prompt Advisor generates a specific fix.
    Reviewer accepts in the HITL UI.
    Prompt version bumped to v3.5.
    New ValidationObjects tagged prompt_version=v3.5 start accumulating.

  ---
  Scenario 2 — Comparing Two Prompt Versions

  You've rewritten the extraction prompt. You want to know:
  "Is v3.5 better than v3.4? On which specific fields?"

  SETUP:
    1. Take a frozen set of 100 inputs (ideally from your golden dataset or
       a recent production sample with human labels).
    2. Run both prompts against those 100 inputs:
         batch_a = await run_batch(inputs, prompt_version="v3.4", model="claude-sonnet-4-6")
         batch_b = await run_batch(inputs, prompt_version="v3.5", model="claude-sonnet-4-6")
    3. Submit both to the evaluation pipeline in "pairwise" scenario mode.

  HOW PAIRWISE WORKS:
    For each claim:
      run judge(claim_a, claim_b, ordering="A_first") → winner: A | B | tie
      run judge(claim_b, claim_a, ordering="B_first") → winner: B | A | tie
      if results disagree → inconclusive (position bias detected)
      if both agree     → accept winner

  WHAT YOU SEE (Dual Territory — Dashboard Tab 1):
    ─────────────────────────────────────────────────────────
    field_path              │ v3.4 is_correct │ v3.5 is_correct │ delta
    ────────────────────────┼─────────────────┼─────────────────┼──────
    contract.annual_value   │  0.71           │  0.91           │ +0.20 ✓
    contract.renewal_date   │  0.88           │  0.85           │ -0.03 ⚠ regression
    customer.account_number │  0.97           │  0.96           │ -0.01 within noise
    ─────────────────────────────────────────────────────────

    Bootstrap p-value on overall delta: 0.002 → statistically significant improvement

  DECISION RULE:
    ✓ Ship v3.5 if: overall delta > 0, no extractive field regresses > 5%
    ✗ Block if:     any precision-critical field (account_number, renewal_date)
                    regresses > 3% — even with overall improvement

  ---
  Scenario 3 — Model Leaderboard (Comparing LLM Providers)

  You're evaluating gpt-4o vs claude-sonnet vs gpt-4o-mini vs llama-3.1-70b
  for your extraction task. You want a single ranked table covering quality,
  cost, and latency.

  SETUP:
    1. Freeze your golden dataset (200 inputs with human-verified correct answers).
    2. Run each model through your extraction prompt, collect outputs.
    3. Submit all 4 to the evaluation pipeline:
         for model in ["gpt-4o", "claude-sonnet", "gpt-4o-mini", "llama-3.1-70b"]:
             await pipeline.evaluate_batch(golden_dataset, model=model,
                                           judge_config="contracts-v2.1.yaml",
                                           judge="claude-sonnet-4-6")   ← use cross-family judge

  HOW THE LEADERBOARD IS BUILT:
    For each model, aggregate across all ValidationObjects:

      Precision   = correct_claims / total_claims_extracted
      Recall      = correct_claims / total_claims_in_golden_dataset
      F1          = 2 * P * R / (P + R)
      Completeness= (1 - is_missing_rate)           ← fraction of fields present
      Helpfulness = avg(rubric_score) on free_form fields only
      Cost/1k     = (avg_input_tokens × price_in + avg_output_tokens × price_out) × 1000
      Latency p95 = 95th percentile of end-to-end generation latency
      Tokens      = avg output_tokens (verbosity signal — more tokens ≠ better)
      Judge Eval  = judge self-consistency: % of claims judge agrees with itself on re-run

  LEADERBOARD OUTPUT:
    ─────────────────────────────────────────────────────────────────────────────
    Model               │  P   │  R   │  F1  │Compl │Help │$/1k  │Lat p95 │Tok │JEval
    ────────────────────┼──────┼──────┼──────┼──────┼─────┼──────┼────────┼────┼──────
    gpt-4o-2024-11-20   │ 0.93 │ 0.89 │ 0.91 │ 0.88 │0.91 │$18.2 │1840ms  │412 │0.87
    claude-sonnet-4-6   │ 0.91 │ 0.87 │ 0.89 │ 0.86 │0.89 │$12.4 │1210ms  │387 │0.85
    gpt-4o-mini         │ 0.85 │ 0.81 │ 0.83 │ 0.79 │0.83 │ $2.1 │ 680ms  │401 │0.79
    llama-3.1-70b-ollama│ 0.80 │ 0.76 │ 0.78 │ 0.72 │0.77 │ $0.0 │2300ms  │443 │0.71
    ─────────────────────────────────────────────────────────────────────────────

  KEY INSIGHTS THE LEADERBOARD REVEALS:
    - gpt-4o-mini has 0.83 F1 at $2.10/1k vs gpt-4o's 0.91 F1 at $18.20/1k
      → 9% quality loss for 8.6× cost reduction. Is that acceptable for your use case?
    - llama-3.1-70b has highest token count (443) with lowest quality (0.78 F1)
      → verbosity without accuracy. The Tokens column exposes this.
    - claude-sonnet is 2nd in quality but 1st in latency (1210ms vs 1840ms for gpt-4o)
      → often the right production choice when latency matters

  CRITICAL: The leaderboard only catches aggregate regressions.
  Always drill into the per-field delta for precision-critical fields before migration.
  A model that scores 0.91 F1 overall may have 0.60 on account_number extraction —
  that one field can break your downstream pipeline even if everything else improved.

  ---
  Please confirm which options you want implemented (or change any picks), and I'll update the design document with all selected changes in one pass.