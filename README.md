# LearningLab

Practical, code-first learning for data science and ML. Each topic folder contains:

- `outline-<topic>.md` — exhaustive learning roadmap with reference links
- `blog-<topic>.md` — one-stop literature survey with citations
- `scripts/` — minimal, self-contained scripts (one concept per file, runnable as-is, every line commented)

## Topics

| Folder | Description | Subtopics |
|---|---|---|
| `01-sql/` | SQL for data science & analytics | 27 |
| `02-statistics/` | Probability, inference, experimentation, causal | 28 |
| `03-python-pandas/` | Python, NumPy, Pandas, data wrangling | 15 |
| `04-pytorch/` | PyTorch: tensors, autograd, training, deployment | 15 |
| `05-git/` | Git internals, workflows, CI/CD for ML | 15 |
| `06-machine-learning/` | Classical ML algorithms & workflows | 28 |
| `07-recommenders/` | Collaborative filtering → neural recsys | 26 |
| `08-embedding-models/` | Word vectors → multimodal embeddings | 22 |
| `09-rag/` | RAG pipelines → agentic RAG | 28 |
| `10-agents/` | ReAct, tools, memory, multi-agent | 23 |
| `11-llm/` | Architecture, fine-tuning, alignment | 28 |
| `12-llm-inferencing/` | Quantization, serving, optimization | 23 |
| `13-llm-evaluation/` | Benchmarks, metrics, human eval, pipelines | 15 |
| `14-reinforcement-learning/` | RL algorithms, deep RL, RLHF, DPO | 15 |
| `15-speech/` | ASR, TTS, speaker recognition, audio processing | 15 |
| `16-vision-ocr/` | CNNs, detection, segmentation, ViT, OCR | 15 |

## File naming

```
scripts/NN-<concept-name>.py   # e.g. 01-groupby.py, 03-reading-audio.py
```

Each script is intentionally minimal — one concept, runnable as-is, every line commented so output in the terminal teaches the idea.

## Workflow

1. Open the `outline-<topic>.md` for the topic you want to learn
2. Pick a subtopic (by number)
3. Ask to create `scripts/NN-<subtopic-name>.py` — a minimal, runnable script for that concept
4. Run it, read the comments, understand the output, then move to the next
