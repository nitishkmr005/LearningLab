# Blog Structure Guide

Use this file in Phase 3 to choose the right sections for your topic.
Sections are a menu — not a fixed template. Pick what fits.

---

## Section Menu

### Always include these

**1. The Problem** (narrative, no bullet lists)
Why does this topic exist? What pain does it solve? Open with a concrete, vivid failure mode — the thing that happens when you don't have this technology. One or two paragraphs, prose only.

**2. History: From X to Y** (chronological evolution)
Walk through each era of the field. For each era:
- What was the approach before?
- What paper introduced the breakthrough? (inline citation with arxiv link)
- What was the core insight?
- What limitation led to the next advance?

This section should feel like a story, not a list of paper names.

**3. Core Mechanism / Architecture** (the "how does it actually work" section)
The dominant architecture or algorithm explained in depth.
- ASCII diagram of data flow (always helpful even for non-visual topics)
- Key formula with step-by-step explanation
- The "aha" insight — what makes the design clever
- One worked example end-to-end

**Working Code** (self-contained, runnable)
The canonical way to use this technology. Must use real library names and real dataset IDs.

**Evaluation and Metrics** (with formulas and worked examples)
How do you know if it's working? Every metric needs:
- What it measures
- The formula
- A worked numerical example
- What "good" looks like in practice

**The Modern Recipe** (opinionated checklist)
What to do today to get SotA results. Concrete names, numbers, commands. The "save this" section.

**References** (organized by category)
Every inline citation has an entry. Grouped into: Foundational, Architecture, Training/Algorithms, Evaluation/Benchmarks, Libraries/Tools, Blogs/Articles.

---

### Include when the topic has training / model development

**Dataset Preparation**
Every data format used to train models:
- Format name + description
- Concrete Python example (small, inline)
- Table of canonical HuggingFace datasets:
  `| Dataset | HuggingFace Link | Description | Size |`
  Link format: `` [`org/dataset`](https://huggingface.co/datasets/org/dataset) ``
- Hard negative mining (if contrastive learning applies)
- Synthetic data generation (if no labeled data scenario applies)
- Multi-dataset training (if the framework supports it)

**Loss Functions / Training Objectives** (full evolution, oldest to SotA)
For each loss function:
- Generation label (Gen 1, Gen 2, ... mark the modern standard with ⭐)
- Formula in code block (plain math, not LaTeX)
- Intuitive explanation
- The limitation that led to the next generation

End with Loss Function Reference Table:
```
| Loss | Data Format | When to Use | When NOT to Use | Quality |
```
Use ★ ratings for quality column.

**Training Script** (complete, minimal, runnable)
Full end-to-end: data loading, model definition, training loop, save.
No pseudocode. Real imports. Real HuggingFace IDs.

---

### Include when the topic involves deploying / serving

**Inference Patterns**
- Basic single input
- Batch inference (always preferred)
- Special prefixes or prompt formats required by modern models
- Quantized inference / reduced precision
- Variable dimension / truncation (if Matryoshka-style)

**Production Considerations**
The things that don't appear in papers:
- Latency breakdown (where time is actually spent)
- Memory footprint and scaling
- Common failure modes in production (what breaks first)
- Cost estimation patterns
- Monitoring signals to watch

---

### Include when the topic has a canonical benchmark

**Benchmark Deep-Dive**
The standard evaluation framework for this domain:
- What tasks/datasets it covers
- For each task type: what it tests, primary metric formula, worked example, what "good" looks like
- How to run it locally (code snippet)
- Where to find the leaderboard

---

### Include when multiple approaches/models/tools compete

**Comparison Tables** — see `references/comparison-table-guide.md` for exact formats

Types:
- Approach comparison (dense vs sparse retrieval, PPO vs DPO, etc.)
- Model comparison (open source table + closed source table)
- Tool / framework comparison (LangChain vs LlamaIndex, etc.)

Always include: Quality vs. Size / Latency ASCII scatter chart.

---

### Include for complex systems or pipelines

**Architecture Diagram**
ASCII art showing how components connect. For pipelines:
```
Input → [Component A] → [Component B] → [Component C] → Output
                              ↓
                        [External Store]
```

**Integration Patterns**
How this technology fits into a larger system:
- What it connects to upstream and downstream
- What protocol / interface it exposes
- Common architectural patterns (e.g., RAG as pre-retrieval vs post-retrieval, agent as orchestrator vs specialist)

---

## Audience Callout Formats

Use these consistently. They help readers self-select the depth they need:

```markdown
> 🎯 **Interview prep**: Interviewers commonly ask about [specific question or tradeoff]. The key answer is [one-sentence answer with the nuance they want to hear].
```

```markdown
> 🏭 **Production note**: In production, [the thing that changes from what the paper says]. Watch for [specific failure mode].
```

```markdown
> 📚 **Go deeper**: [Article/paper title](URL) — one-line description of what you'll learn.
```

Place these throughout each section. Aim for at least one per major section.

---

## Resources Box Format

End every major section with a Resources box:

```markdown
**Resources**
- [Paper title](arxiv URL) — one-line: what this paper proved or introduced
- [Blog post title](URL) — one-line: what practical insight this adds beyond the paper
- [Library docs page](URL) — one-line: what you'll find here
```

Keep to 2-4 links. Quality over quantity.

---

## Citation Format Standards

**Inline paper citation:**
`([Author et al., YEAR](https://arxiv.org/abs/XXXX))`

**Inline single-author blog:**
`([Author Name, YEAR](https://url))`

**HuggingFace dataset link:**
`` [`org/dataset-name`](https://huggingface.co/datasets/org/dataset-name) ``

**HuggingFace model link:**
`` [`org/model-name`](https://huggingface.co/org/model-name) ``

**GitHub repo link:**
`[org/repo](https://github.com/org/repo)` — always add star count if known

**References section entry:**
```
- Author et al. (YEAR). *Title.* https://arxiv.org/abs/XXXX
- Author (YEAR). *Blog post title.* https://url
```

---

## Comparison Tables — Column Standards

Read `references/comparison-table-guide.md` for full table formats.
Quick rule: every table must have a "When to use" and "When NOT to use" column (or equivalent).

---

## Tone Rules

**Be opinionated**: "Use X for Y" not "X or Y could both work depending on your needs."
**Be honest about weaknesses**: every technique has a failure mode. State it before the reader hits it.
**Progressive complexity**: start with the simplest correct explanation, then add nuance.
**No filler intros**: never write "In this section we will explore..." — start with the content.
**One non-obvious insight per section**: something a junior reading the paper wouldn't know.

---

## Word Count Targets

| Section | Target |
|---|---|
| The Problem | 200-400 words |
| History | 500-800 words |
| Core Mechanism | 600-1000 words |
| Dataset Preparation | 400-700 words |
| Loss Functions | 800-1200 words |
| Training Script | 200-400 words (mostly code) |
| Evaluation | 400-700 words |
| Inference | 300-500 words |
| Benchmark Deep-Dive | 600-1000 words |
| Model Comparison | 400-600 words (mostly table) |
| Modern Recipe | 200-400 words |
| References | 200-400 words |

Total target: 4,000 - 8,000 words depending on topic complexity.
