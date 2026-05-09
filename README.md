# LearningLab

Practical, code-first learning for data science and ML. Each topic folder contains:

- `outline-<topic>.md` — exhaustive learning roadmap with reference links
- `XX-<topic-name>.py` — minimal, self-contained scripts (one concept per file, runnable as-is)

## Topics

| Folder | Description | Subtopics |
|---|---|---|
| `01-sql/` | SQL for data science & analytics | 27 |
| `02-statistics/` | Probability, inference, experimentation, causal | 28 |
| `03-machine-learning/` | Classical ML algorithms & workflows | 28 |
| `04-recommenders/` | Collaborative filtering → neural recsys | 26 |
| `05-embedding-models/` | Word vectors → multimodal embeddings | 22 |
| `06-rag/` | RAG pipelines → agentic RAG | 28 |
| `07-agents/` | ReAct, tools, memory, multi-agent | 23 |
| `08-llm/` | Architecture, fine-tuning, alignment | 28 |
| `09-llm-inferencing/` | Quantization, serving, optimization | 23 |

## File naming

```
XX-<topic-name>.py   # e.g. 01-basic-rag.py, 02-chunking-strategies.py
```

Each `.py` file is intentionally minimal — just enough code to understand one concept, runnable as-is.

## Workflow

1. Open the `outline-<topic>.md` for the topic you want to learn
2. Pick a subtopic (by number)
3. Ask to create `XX-<subtopic-name>.py` — a minimal, runnable script for that concept
4. Run it, read it, understand it, then move to the next
