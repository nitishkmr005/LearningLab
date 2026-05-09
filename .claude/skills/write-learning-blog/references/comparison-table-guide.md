# Comparison Table Guide

Use this file to choose the right table format for each comparison scenario.
Every table must tell the reader: when to use it, when not to, and how popular it is.

---

## Table Type 1: Approach / Algorithm Comparison

Use when: comparing fundamentally different methods for solving the same problem.
Examples: Dense vs sparse retrieval, PPO vs DPO vs GRPO, BM25 vs neural search, batch vs stream processing.

```markdown
| Approach | How it works | Strengths | Weaknesses | When to use | When NOT to use | Real-world users |
|---|---|---|---|---|---|---|
| BM25 | TF-IDF based lexical scoring | No training needed, fast, interpretable | Misses synonyms, context-blind | Keyword-heavy queries, cold start, low budget | Semantic queries, multilingual | Elasticsearch, Solr, most search engines |
| Dense retrieval | Bi-encoder embeds query + doc | Semantic understanding, language-agnostic | Needs training data, slower indexing | Semantic search, RAG with good embeddings | When you have no labeled data | Google, Bing semantic layer, most RAG systems |
| Sparse neural (SPLADE) | Learned sparse weights over vocab | BM25 + semantic, efficient | Needs training, harder to debug | Best of both worlds when you can afford training | Simple use cases | Cohere, Pinecone hybrid mode |
```

---

## Table Type 2: Open-Source Model Comparison

Use when: comparing pre-trained models users can download and run.

Columns: Model | HF Link | Size | Provider | Released | Best Use Cases | Training Approach | Popularity
- HF Link: `[🤗](https://huggingface.co/org/model-name)`
- Size: parameter count (7B, 13B, 70B) or embedding dims
- Training Approach: brief description of the key training technique
- Popularity:
  - 🔥 >1M downloads/month
  - ⭐ 100K–1M/month
  - 📈 10K–100K/month
  - 🆕 <10K/month

```markdown
*Popularity: 🔥 >1M/mo · ⭐ 100K–1M · 📈 10K–100K · 🆕 <10K (HuggingFace monthly downloads)*

| Model | HF Link | Size | Provider | Released | Best Use Cases | Training Approach | Popularity |
|---|---|---|---|---|---|---|---|
| `Llama-3.1-8B-Instruct` | [🤗](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B | Meta | Jul 2024 | General instruction following, RAG | SFT + RLHF on diverse instruction data | 🔥 45M/mo |
```

Always add a note about how to read the table:
```markdown
> Models marked `†` support variable configuration (e.g. context length, quantization levels).
> Download numbers are approximate monthly figures from HuggingFace Hub (checked May 2025).
```

---

## Table Type 3: Closed-Source / API Model Comparison

Same columns as open-source but:
- Replace HF Link with [Provider Docs](URL)
- Replace Popularity with `💼 API only` since these aren't on HuggingFace Hub

```markdown
| Model | API Docs | Size | Provider | Released | Best Use Cases | Training Approach | Pricing |
|---|---|---|---|---|---|---|---|
| `gpt-4o` | [OpenAI](https://platform.openai.com/docs) | Unknown | OpenAI | May 2024 | Multimodal reasoning, complex tasks | RLHF + constitutional AI | $5/1M input tokens |
```

---

## Table Type 4: Tool / Framework / Library Comparison

Use when: comparing software tools or frameworks.
Examples: LangChain vs LlamaIndex, Chroma vs Qdrant vs Weaviate, PyTorch vs JAX.

```markdown
| Tool | GitHub | Stars | What it does | Strengths | Weaknesses | When to use | Production-ready? |
|---|---|---|---|---|---|---|---|
| LangChain | [link](https://github.com/langchain-ai/langchain) | 90K+ | LLM application framework | Huge ecosystem, fast prototyping | Abstractions can hide bugs, complex for simple use cases | Rapid prototyping, broad integrations needed | ✅ Yes |
| LlamaIndex | [link](https://github.com/run-llama/llama_index) | 35K+ | RAG-focused data framework | Best-in-class RAG primitives, data connectors | Narrower scope than LangChain | When RAG is the core use case | ✅ Yes |
```

---

## Table Type 5: Evolution / History Table

Use when: showing how a technique evolved over time. Useful as an alternative or supplement to the history prose section.

```markdown
| Era | Approach | Key Paper | Core Idea | Limitation |
|---|---|---|---|---|
| 2013 | Word2Vec | [Mikolov et al.](https://arxiv.org/abs/1301.3781) | Predict context words from center word | One vector per word regardless of context |
| 2014 | GloVe | [Pennington et al.](https://aclanthology.org/D14-1162/) | Factor global co-occurrence matrix | Still static, no context |
| 2018 | ELMo | [Peters et al.](https://arxiv.org/abs/1802.05365) | Bidirectional LSTM language model | Feature-based only, not fine-tunable |
| 2018 | BERT | [Devlin et al.](https://arxiv.org/abs/1810.04805) | Masked LM + deep bidirectional transformer | [CLS] token not good for sentence similarity |
| 2019 | SBERT | [Reimers & Gurevych](https://arxiv.org/abs/1908.10084) | Siamese BERT with mean pooling | Needed labeled sentence pairs |
```

---

## Table Type 6: Metric / Benchmark Comparison

Use when: showing which benchmark a model should be evaluated on, or comparing models across multiple benchmarks.

```markdown
| Model | MTEB Retrieval nDCG@10 | MTEB STS Spearman | BEIR avg nDCG@10 | Params | Dims |
|---|---|---|---|---|---|
| `bge-large-en-v1.5` | 54.3 | 87.1 | 48.2 | 335M | 1024 |
| `nomic-embed-text-v1.5` | 53.8 | 86.7 | 47.9 | 137M | 768† |
| `all-MiniLM-L6-v2` | 41.0 | 82.4 | 39.7 | 22M | 384 |
```

Always state the benchmark version and date scores were pulled:
```markdown
> Scores from MTEB leaderboard, May 2025. nDCG@10 on 15 BEIR datasets. Higher is better for all metrics.
```

---

## Quality vs. Size ASCII Chart

Always include this chart when comparing models of different sizes:

```
Quality (primary metric)
  ▲
  │                           ● Large model (slow, expensive)
  │               ● Medium model
  │     ● Small model (fast, cheap)
  │  ● Tiny model (very fast)
  └──────────────────────────────► Size / Latency
     Fast/Cheap              Slow/Expensive
```

Replace the labels with actual model names and metric scores.

---

## Real-World Users Column

When comparing approaches or tools, always try to include a "Real-world users" or "Used by" column.
Search for: `<tool/approach> used by production case study blog`

Examples:
- "Google, Facebook, most large search engines"
- "Notion, Stripe (via their AI features)"
- "Perplexity AI (dense retrieval), Elasticsearch users (BM25)"
- "Mistral, Together AI (vLLM for serving)"

This column answers: "Does anyone actually use this at scale?" — which is what practitioners and interviewers both care about.

---

## When to Use Each Table Type

| Situation | Table Type |
|---|---|
| "Should I use A or B approach?" | Approach comparison (Type 1) |
| "Which model should I pick?" | Model comparison (Type 2 or 3) |
| "Which library/framework?" | Tool comparison (Type 4) |
| "How did this field evolve?" | Evolution table (Type 5) |
| "How do models rank on benchmarks?" | Metric comparison (Type 6) |
| "What's the quality/cost tradeoff?" | ASCII scatter chart |

---

## Table Formatting Rules

1. Every table must have a "When to use" column (or equivalent)
2. Every table must have a "When NOT to use" column (or equivalent) — omitting this is the most common mistake
3. Popularity must be real data — search HuggingFace downloads or GitHub stars, don't guess
4. Keep cell text under ~80 characters — break into bullet points for longer content
5. Every table gets a one-sentence caption above it explaining what it shows and how to read it
6. Add a legend for any symbols (🔥 ⭐ 📈 🆕 💼 ✅ ❌)
