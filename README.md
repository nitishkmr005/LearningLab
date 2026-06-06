# LearningLab

An exhaustive, interview-ready learning system covering SQL through advanced LLMs, vision, speech, and RL — targeted at four roles:

| Role | Focus |
|---|---|
| **Data Scientist** | Statistics, experimentation, classical ML, SQL, feature engineering |
| **ML Engineer** | Model training, serving, MLOps, data pipelines, distributed systems |
| **AI Engineer** | LLMs, RAG, agents, prompt engineering, evaluation, fine-tuning |
| **GenAI Engineer** | Multimodal models, inference optimization, tool use, reasoning models, voice/vision agents |

Every topic is chosen because it appears in interviews, production systems, or both for at least one of these roles.

---

## Deliverables per topic

| File | Purpose |
|---|---|
| `outline-<topic>.md` | Exhaustive roadmap: subtopics, production-popular approaches, SotA models, and authoritative reference links. Delete after blog is written. |
| `blog-<topic>.md` | One-stop literature survey: history, theory, formulas, code, benchmarks, citations. |
| `scripts/NN-<concept>.py` | Minimal, self-contained learning snippets — one concept per file, every line commented, runnable output teaches the idea. |

---

## Topics

| # | Folder | Description | Roles | Sections | Status |
|---|---|---|---|---|---|
| 01 | `01-sql/` | SQL for data science: window functions, CTEs, analytical patterns, Snowflake | DS, ML | 20 | blog |
| 02 | `02-statistics/` | Probability, inference, A/B testing, causal inference, Bayesian methods | DS, ML | 31 | blog |
| 03 | `03-dsa/` | Data structures & algorithms for DS/ML: arrays, heaps, graphs, DP, probabilistic structures, ANN | DS, ML, AI | 22 | blog |
| 04 | `04-python-pandas/` | Python, NumPy, Pandas, Polars, async, FastAPI, Streamlit | DS, ML, AI | 24 | blog |
| 05 | `05-pytorch/` | Tensors, autograd, training loops, distributed training, deployment | ML, AI | 17 | blog |
| 06 | `06-git/` | Git internals, branching, rebase, CI/CD for ML, conventional commits | ML, AI | 16 | blog |
| 07 | `07-machine-learning/` | Classical ML algorithms, evaluation, AutoML, fairness, MLOps, AWS SageMaker | DS, ML | 37 | blog |
| 08 | `08-deep-learning/` | Neural nets, activations, backprop, loss functions (with formulas), optimizers, CNNs, RNNs, attention, regularisation, transfer learning, diffusion | DS, ML, AI | 12 | outline |
| 09 | `09-recommenders/` | Collaborative filtering, neural recsys, two-tower, GNNs, bandits, dataset prep, inference patterns | DS, ML, AI | 32 | blog |
| 10 | `10-nlp-foundations/` | Text preprocessing, TF-IDF, n-grams, RNNs, LSTMs, GRUs, seq2seq, attention — the pre-Transformer NLP stack | DS, ML, AI | — | blog |
| 11 | `11-embedding-models/` | Word vectors → sentence transformers → multimodal, code embeddings | ML, AI | 24 | blog |
| 12 | `12-rag/` | RAG pipelines: chunking, retrieval, reranking, evaluation, agentic RAG | AI, GenAI | 30 | blog |
| 13 | `13-agents/` | ReAct, tool use, memory, multi-agent, LangGraph, MCP, computer use | AI, GenAI | 25 | blog |
| 14 | `14-llm/` | Transformer architecture, fine-tuning, alignment, RLHF, reasoning models | ML, AI, GenAI | 30 | blog |
| 15 | `15-llm-inferencing/` | Quantization, vLLM, flash attention, speculative decoding, KV cache | ML, GenAI | 24 | blog |
| 16 | `16-llm-evaluation/` | Benchmarks, metrics, LLM-as-judge, hallucination, agentic eval | AI, GenAI | 18 | blog |
| 17 | `17-reinforcement-learning/` | RL algorithms, deep RL, PPO, SAC, RLHF, DPO, GRPO | ML, AI | 16 | blog |
| 18 | `18-speech/` | ASR, TTS, speaker recognition, Whisper, voice agent architecture | AI, GenAI | 17 | blog |
| 19 | `19-vision-ocr/` | CNNs, detection, segmentation, ViT, OCR, SAM, DINOv2, video | ML, AI, GenAI | 18 | blog |
| 20 | `20-interview/` | 80/20 interview prep: DSA, SQL, stats, ML, PyTorch, NLP, embeddings, LLMs, fine-tuning, agents, system design | DS, ML, AI, GenAI | 16 | blog |

---

## Docs

Reference materials organised by topic under `docs/`:

```
docs/
  llm/          — inference engineering, fine-tuning guide, transformer internals
  rag/          — RAG evaluation & testing in production
  statistics/   — daily dose of data science
  claude-code/  — building skills for Claude
```

---

## How to use

### Learn a subtopic (add a script)
1. Open `outline-<topic>.md` and pick a subtopic by number.
2. Ask: *"Create `scripts/NN-<subtopic-name>.py` for [topic]."*
3. Run the script, read the comments, understand the output, move to the next.

### Write a blog for a topic
Use the `/write-learning-blog` skill — it handles research, citations, structure, and saving automatically. Delete `outline-<topic>.md` after the blog is saved.

### Add a new topic
1. Create `<NN>-<topic>/outline-<topic>.md` following the existing format.
2. Update this README table.

---

## Script conventions

```
scripts/NN-<concept-name>.py   # e.g. 01-window-functions.py, 03-reading-audio.py
```

- One concept per file, runnable as-is with no extra setup.
- Every line has an inline comment explaining what it does.
- Running the script prints useful output so the learner sees the concept in action.
- No pseudocode, no placeholder comments.
