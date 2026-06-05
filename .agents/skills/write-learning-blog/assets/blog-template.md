# [TOPIC]: From Basics to State of the Art

*[One-sentence description: what this blog covers and why it matters.]*

---

## 1. The Problem

[2-4 paragraphs of pure narrative. No bullet lists here.

Open with a concrete failure scenario: what goes wrong when you don't have this technology?
Example: "You're building a search engine. A user types 'cheap flights to NY'. Your keyword matcher returns articles about 'inexpensive airfare to New York' as rank 10 because it shares no words with the query."

Then: what is the goal of this technology?
Then: why is it hard?
End with: a one-sentence preview of the journey this blog takes.]

---

## 2. A Brief History: From [Earliest Approach] to [Current SotA]

### 2.1 [Era Name] ([Year Range])

[The state of the field before the breakthrough. What did people do? Why was it insufficient?]

**[Paper/Technique Name]** ([Author et al., YEAR](https://arxiv.org/abs/XXXX)) introduced [core idea in one sentence]. The key insight was [the "aha" moment]. But it had a fundamental limitation: [what it couldn't do].

### 2.2 [Next Era] ([Year Range])

[Continue chronologically. Each era: what broke the previous limitation, what new limitation emerged.]

> 📚 **Go deeper**: [Best survey paper or explainer for this history](URL) — [one-line description]

**Resources**
- [Paper](URL) — [contribution]
- [Blog post](URL) — [practical insight]

---

## 3. Core Mechanism: How It Actually Works

[2-3 sentences explaining why this section exists: the reader now knows the history, this section gives them the deep understanding of how the dominant approach works.]

### 3.1 Architecture

```
[Input] → [Component A] → [Component B] → [Output]
                ↓
          [External Store / Index]
```

[Explain each component in the diagram. What does it do? What goes in and out?]

### 3.2 The Key Formula

[One paragraph explaining what the formula computes and why that's the right thing to compute.]

```
[FORMULA IN PLAIN MATH NOTATION]

Where:
  [variable] = [what it is]
  [variable] = [what it is]
```

**Worked example:**
```
Given: [concrete input values]

Step 1: [compute first term] = [value]
Step 2: [compute second term] = [value]
Step 3: [combine] = [final value]

Result: [what this means in plain English]
```

> 🎯 **Interview prep**: Interviewers often ask "[common question about this mechanism]". The key answer is [one sentence with the nuance].

**Resources**
- [Paper](URL) — [contribution]
- [Blog post with visual explanation](URL) — [what makes it good]

---

## 4. Dataset Preparation

[2-3 sentences: the data format is dictated by the loss function / algorithm you choose. This section covers every format used in this domain.]

### 4.1 [Format Name] `(field_a, field_b[, label])`

[What this format is used for. When you'd choose it.]

```python
data = [
    {"field_a": "example input A", "field_b": "example input B", "label": 1.0},
    # ... more examples
]
```

**Canonical datasets for this format:**

| Dataset | HuggingFace Link | Description | Size |
|---|---|---|---|
| [Dataset Name] | [`org/dataset`](https://huggingface.co/datasets/org/dataset) | [What it contains] | [N examples] |

### 4.2 [Next Format Name]

[Repeat for each format.]

### 4.3 Hard Negative Mining

[If applicable: what are hard negatives, why they matter, strategies to mine them.]

```python
# [Mining strategy name] example
[minimal runnable code]
```

> 🏭 **Production note**: [What changes about data prep when you're working at scale or with domain-specific data]

**Resources**
- [Dataset paper](URL) — [what it contributes]
- [HuggingFace datasets docs](URL) — [how to load efficiently]

---

## 5. Loss Functions / Training Objectives

[2-3 sentences: the loss function is the single most important factor in training quality. Walk through the evolution from simplest to SotA.]

### 5.1 Generation 1 — [Earliest Loss Name]

[What problem it was trying to solve. What data format it used.]

```
FORMULA:
[formula in plain math]
```

[Intuitive explanation of what the formula does in one paragraph.]

**Why it's limited:** [Concrete limitation that motivated the next generation.]

### 5.2 Generation 2 — [Next Loss Name]

[Continue pattern. Mark the modern standard with ⭐]

### 5.N ⭐ Generation N — [Modern Standard] (Current Best Practice)

[This is the one to use by default today. Explain why it's better than all predecessors.]

---

**Complete Loss Function Reference**

| Loss | Data Format | When to Use | When NOT to Use | Quality |
|---|---|---|---|---|
| `[LossName]` | `(a, b, label)` | [scenario] | [counter-scenario] | ★★☆☆☆ |
| `[LossName]` ⭐ | `(a, b)` | [scenario] | [counter-scenario] | ★★★★★ |

> 🎯 **Interview prep**: "What loss function would you use for [common scenario]?" — Answer: "[loss name] because [one-sentence reason tied to the data format and what it optimizes]."

**Resources**
- [Paper for modern loss](URL) — [contribution]
- [Comparative study](URL) — [what it benchmarks]

---

## 6. Training Script

[2-3 sentences: what this script demonstrates and what you'll be able to do after running it.]

```python
# [Topic]: minimal end-to-end training example
# Requires: pip install [library]

from [library] import [key classes]
from datasets import load_dataset

# --- 1. Load model ---
model = [ModelClass]("[real-model-id]")

# --- 2. Load dataset ---
dataset = load_dataset("[real-hf-dataset-id]", split="train[:10000]")

# --- 3. Define loss / objective ---
loss = [LossClass](model)

# --- 4. Training arguments ---
args = [TrainingArgsClass](
    output_dir="output/[topic]-model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
)

# --- 5. Train ---
trainer = [TrainerClass](
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss,
)
trainer.train()

# --- 6. Save ---
model.save_pretrained("output/[topic]-model/final")
```

**Key choices explained:**
- `batch_size=[N]`: [why this value matters for this specific algorithm]
- `learning_rate=2e-5`: [why this range is right for fine-tuning transformers]
- `[other key hyperparameter]`: [why]

> 🏭 **Production note**: [What changes when you train on a full dataset vs. this minimal example. Common training pitfalls.]

---

## 7. Evaluation

[2-3 sentences: what you're measuring and why the choice of metric matters.]

### 7.1 [Primary Metric Name]

**What it measures:** [Plain English.]

```
FORMULA:
[metric formula]

Where:
  [variable] = [definition]
```

**Worked example:**
```
[Concrete numbers, step by step]
```

**What "good" looks like:** [State-of-the-art score on the canonical benchmark. What a baseline scores.]

```python
from [evaluation library] import [EvaluatorClass]

evaluator = [EvaluatorClass](
    [inputs],
    name="my-eval",
)
result = evaluator(model)
print(result)
```

### 7.2 [Secondary Metric]

[Repeat pattern.]

### 7.3 Quick Sanity Check

```python
# Run in under 1 minute to verify the model is working
[minimal sanity check code]
```

**Complete Evaluator Reference**

| Evaluator | Input Format | Primary Metric | Use Case |
|---|---|---|---|
| `[EvaluatorName]` | `(input_a, input_b, label)` | [metric] | [when to use] |

> 🎯 **Interview prep**: "How would you evaluate [model type] without a labeled dataset?" — Answer: [one-sentence answer covering proxy metrics or human eval approaches]

**Resources**
- [Benchmark paper](URL) — [what it covers]
- [Evaluation framework docs](URL) — [how to use it]

---

## 8. Inference

[2-3 sentences: the right inference pattern matters. Models trained with special prefixes or formats will silently underperform if you call them wrong.]

### 8.1 Basic Inference

```python
from [library] import [ModelClass]

model = [ModelClass]("[real-model-id]")
result = model.[infer_method]("your input text")
print(result)
```

### 8.2 Batch Inference (Always Prefer This)

```python
inputs = ["input 1", "input 2", "input 3"]
results = model.[infer_method](inputs, batch_size=64, show_progress_bar=True)
```

### 8.3 [Special Pattern for This Domain]

[e.g., Instruction prefixes for E5/BGE, prompt templates for LLMs, query/document asymmetry for retrieval]

```python
# [pattern description]
[code example]
```

> 🏭 **Production note**: [Latency characteristics. What's fast vs. slow. Common mistake in production inference.]

---

## 9. [Canonical Benchmark]: What It Measures and How

[2-3 sentences: [Benchmark Name] is the gold standard for evaluating [topic]. It covers [N] tasks across [domains]. Understanding what each task tests — and which metric to use — is the key to picking the right model for your use case.]

### 9.1 [Task Type 1] — Primary metric: [metric name]

**What it tests:** [Plain English.]

**Metric formula:**
```
[FORMULA]
```

**Worked example:**
```
[Step by step with numbers]
Result: [what the score means]
```

**Range:** [min] to [max]. State-of-the-art models score ~[value].

### 9.2 [Task Type 2]

[Repeat for each task type in the benchmark.]

**Running the benchmark locally:**

```python
from [benchmark library] import [BenchmarkClass]
from [model library] import [ModelClass]

model = [ModelClass]("[real-model-id]")
evaluation = [BenchmarkClass](tasks=["[task1]", "[task2]"])
results = evaluation.run(model, output_folder="results/[model-name]")
```

> 📚 **Go deeper**: [[Benchmark Name] leaderboard](URL) — see current SotA scores and compare models

---

## 10. How to Choose the Right Approach / Model

[2-3 sentences: the leaderboard lists hundreds of options. Here's a principled framework.]

### Step 1: Match your use case to the primary metric

| Your Goal | Task Type | Primary Metric |
|---|---|---|
| [Goal A] | [Task Type] | [Metric] |
| [Goal B] | [Task Type] | [Metric] |

### Step 2: Filter by constraints

- **Low latency / edge deployment** → [small model recommendation]
- **Maximum quality** → [large model recommendation]
- **Multilingual** → [multilingual recommendation]
- **No GPU available** → [CPU-friendly recommendation]

### Step 3: Validate on your domain

Leaderboard averages can hide domain-specific failure. Always:
1. Pick top-3 leaderboard models for your task
2. Evaluate on 100-500 samples from your actual data
3. Select based on in-domain results, not leaderboard rank

**Quick Selection Guide**

| Scenario | Recommended Approach / Model |
|---|---|
| [Common scenario] | [Concrete recommendation] |
| [Another scenario] | [Concrete recommendation] |

---

## 11. State-of-the-Art Comparison

### 11.1 Open-Source Models / Approaches

*Popularity: 🔥 >1M/mo · ⭐ 100K–1M · 📈 10K–100K · 🆕 <10K (HuggingFace monthly downloads, [Month Year])*

| Model | HF Link | Size | Provider | Released | Best Use Cases | Approach | Popularity |
|---|---|---|---|---|---|---|---|
| `[model-name]` | [🤗](https://huggingface.co/org/model) | [size] | [Provider] | [Date] | [Use cases] | [Training approach] | 🔥 [N]M/mo |

### 11.2 Closed-Source / API Options

| Model | Docs | Size | Provider | Released | Best Use Cases | Approach | Pricing |
|---|---|---|---|---|---|---|---|
| `[model-name]` | [Docs](URL) | Unknown | [Provider] | [Date] | [Use cases] | Proprietary | $[N]/[unit] |

### 11.3 Quality vs. Size / Cost

```
Quality
  ▲
  │                     ● [Large, expensive model]
  │          ● [Medium model]
  │  ● [Small model]
  └──────────────────────────► Size / Cost / Latency
     Fast/Cheap           Slow/Expensive
```

### 11.4 Key Trends

1. **[Trend 1]**: [from old way] → [new way]. Example: [concrete model or paper that exemplifies this]
2. **[Trend 2]**: [description]
3. **[Trend 3]**: [description]

---

## 12. The Modern Recipe

What to do today to get SotA results on [topic]. Copy-paste ready.

1. **[Step 1]**: Use `[specific library/tool]` — [why this over alternatives]
2. **[Step 2]**: Start from `[specific pre-trained model ID]` — [why this base model]
3. **[Step 3]**: [specific data strategy] — [why]
4. **[Step 4]**: [specific hyperparameters] — `lr=2e-5, batch_size=64, epochs=3`
5. **[Step 5]**: Evaluate with `[specific evaluator]` on `[specific benchmark subset]`
6. **[Step 6]**: [production serving recommendation]

> 🎯 **Interview prep**: If asked "how would you build [X] from scratch?", this recipe is your answer. Hit steps 1-4 confidently, then mention step 5 (evaluation) to show production readiness.

---

## References

### Foundational
- [Author et al. (YEAR). *Title.* https://arxiv.org/abs/XXXX]

### Architecture / Core Techniques
- [Author et al. (YEAR). *Title.* https://arxiv.org/abs/XXXX]

### Training / Algorithms
- [Author et al. (YEAR). *Title.* https://arxiv.org/abs/XXXX]

### Evaluation / Benchmarks
- [Author et al. (YEAR). *Title.* https://arxiv.org/abs/XXXX]

### Libraries / Tools
- [[Library name] documentation](URL)
- [[GitHub repo]](URL) — [what it provides]

### Blogs / Articles
- [Author (YEAR). *Article title.* URL]
- [Org blog (YEAR). *Post title.* URL]
