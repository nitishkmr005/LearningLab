# Web Research Sources Directory

Use this file during Phase 2 to find the right sources for any AI/ML/data topic.
Fetch the relevant URLs, read the content, and extract insights before writing.

---

## Official Organization Documentation

### HuggingFace
- Blog: `https://huggingface.co/blog` — search for topic name
- Docs: `https://huggingface.co/docs` — transformers, datasets, peft, trl, accelerate, diffusers
- Papers: `https://huggingface.co/papers` — daily arxiv papers with community discussion
- Datasets: `https://huggingface.co/datasets` — search canonical datasets for the topic
- Models: `https://huggingface.co/models` — filter by task type, sort by downloads
- Spaces/Demos: `https://huggingface.co/spaces` — live demos of approaches
- **What to extract**: practical training tips, dataset format standards, model card patterns, leaderboard positions

### NVIDIA
- Developer blog: `https://developer.nvidia.com/blog` — search for topic
- Technical blog: `https://blogs.nvidia.com/blog/category/deep-learning/`
- Documentation: `https://docs.nvidia.com/deeplearning/` — TensorRT, cuDNN, Triton
- NGC catalog: `https://catalog.ngc.nvidia.com/` — optimized model containers
- **What to extract**: performance benchmarks, hardware optimization patterns, production deployment guidance

### Anthropic / Claude
- Research: `https://www.anthropic.com/research` — papers and technical reports
- Claude docs: `https://docs.anthropic.com` — prompting guides, API patterns, tool use
- News: `https://www.anthropic.com/news` — model announcements and research updates
- **What to extract**: prompting patterns, safety considerations, tool use best practices

### OpenAI
- Blog: `https://openai.com/blog` — research announcements
- API docs: `https://platform.openai.com/docs` — practical implementation patterns
- Cookbook: `https://cookbook.openai.com` — code examples for common tasks
- **What to extract**: API patterns, prompt engineering, fine-tuning guidance

### Google / DeepMind
- Google AI blog: `https://ai.google/research/` and `https://blog.research.google`
- DeepMind: `https://deepmind.google/research/`
- Vertex AI docs: `https://cloud.google.com/vertex-ai/docs`
- Google Colab examples: `https://colab.research.google.com` — search for topic
- **What to extract**: scaling insights, research breakthroughs, production patterns on GCP

### Meta AI
- Blog: `https://ai.meta.com/blog/` — LLaMA, PyTorch, fairseq research
- PyTorch docs: `https://pytorch.org/docs/` — core framework patterns
- **What to extract**: open-source model releases, PyTorch patterns, research at scale

### AWS / Amazon
- Machine Learning blog: `https://aws.amazon.com/blogs/machine-learning/`
- SageMaker docs: `https://docs.aws.amazon.com/sagemaker/`
- **What to extract**: production deployment patterns, managed service trade-offs

### Microsoft
- Research blog: `https://www.microsoft.com/en-us/research/blog/`
- Azure AI docs: `https://learn.microsoft.com/en-us/azure/ai-services/`
- **What to extract**: enterprise patterns, DeepSpeed/Phi model details

---

## Popular Author Blogs

### General ML / Deep Learning
| Author | Blog URL | Best topics |
|---|---|---|
| Lilian Weng (OpenAI) | `https://lilianweng.github.io` | Attention, RL, agents, diffusion, alignment |
| Andrej Karpathy | `https://karpathy.github.io` and `https://karpathy.ai` | LLMs from scratch, neural nets, backprop |
| Sebastian Raschka | `https://sebastianraschka.com/blog/` | LLM training, fine-tuning, PyTorch |
| Chip Huyen | `https://huyenchip.com/blog/` | ML systems, production ML, LLMOps |
| Jay Alammar | `https://jalammar.github.io` | Transformers, BERT, GPT — illustrated visuals |
| Sebastian Ruder | `https://ruder.io` | NLP, transfer learning, multilingual models |
| Eugene Yan | `https://eugeneyan.com/writing/` | Recommenders, production ML, LLMs applied |
| Josh Tobin | `https://josh-tobin.com/blog` | ML infrastructure, tooling |
| Simon Willison | `https://simonwillison.net` | LLMs, practical AI, tool use |
| Shreya Shankar | `https://www.shreya-shankar.com` | ML pipelines, data quality, evaluation |

### Embedding Models / NLP
| Author | Key Articles |
|---|---|
| Nils Reimers (SBERT) | `https://www.sbert.net/` — SBERT docs, training guides |
| Tom Aarsen (HF) | `https://huggingface.co/blog/train-sentence-transformers` — ST v3 |
| Nils Reimers blog | `https://www.nils-reimers.de/` |

### RAG / Retrieval
| Author | Key Articles |
|---|---|
| Jerry Liu (LlamaIndex) | `https://blog.llamaindex.ai` |
| Harrison Chase (LangChain) | `https://blog.langchain.dev` |
| Connor Shorten | `https://weaviate.io/blog` — RAG patterns |

### LLM Inference / Systems
| Author | Key Articles |
|---|---|
| Woosuk Kwon (vLLM) | vLLM blog `https://blog.vllm.ai` |
| Tim Dettmers | `https://timdettmers.com` — quantization, bitsandbytes |
| Tri Dao (FlashAttention) | FA blog posts on Hugging Face |

### Agents
| Author | Key Articles |
|---|---|
| Lilian Weng | `https://lilianweng.github.io/posts/2023-06-23-agent/` — LLM powered agents |
| Anthropic team | `https://www.anthropic.com/research/building-effective-agents` |
| Andrew Ng | `https://www.deeplearning.ai/the-batch/` — agentic AI series |

---

## arXiv Search Patterns

Use these search patterns on `https://arxiv.org/search/`:

```
# Find foundational papers
<topic> original paper survey 2017 2018 2019

# Find recent SotA
<topic> state of the art 2024 2025

# Find benchmark papers
<topic> benchmark evaluation dataset 2022 2023

# Find survey papers (great for comprehensive coverage)
survey <topic> large language models 2024

# Find specific technique papers
<technique name> efficient scalable training
```

When reading arxiv papers, extract:
- Abstract: core contribution in one sentence
- Key formula or algorithm (usually Section 3-4)
- Main results table
- Limitations section

---

## GitHub Libraries by Topic

### Embedding Models
- `sentence-transformers/sentence-transformers` — training + inference
- `UKPLab/sentence-transformers` — original repo
- `cvangysel/rankeval` — evaluation

### RAG
- `langchain-ai/langchain` — pipeline framework
- `run-llama/llama_index` — RAG-focused framework
- `chroma-core/chroma` — vector store
- `qdrant/qdrant` — production vector store
- `facebookresearch/faiss` — ANN search

### LLM Fine-tuning
- `huggingface/peft` — LoRA, QLoRA, adapters
- `huggingface/trl` — SFT, DPO, RLHF
- `hiyouga/LLaMA-Factory` — unified fine-tuning
- `artidoro/qlora` — QLoRA original

### LLM Inference
- `vllm-project/vllm` — PagedAttention serving
- `ggerganov/llama.cpp` — local CPU/GPU inference
- `huggingface/text-generation-inference` — TGI
- `NVIDIA/TensorRT-LLM` — NVIDIA optimized serving

### Agents
- `microsoft/autogen` — multi-agent framework
- `crewAIInc/crewAI` — role-based agents
- `langchain-ai/langgraph` — stateful agent graphs
- `BerriAI/litellm` — unified LLM API

### Recommenders
- `microsoft/recommenders` — classical + neural
- `NVIDIA/Merlin` — large-scale GPU recsys
- `RUCAIBox/RecBole` — research framework

### Machine Learning
- `scikit-learn/scikit-learn` — classical ML
- `dmlc/xgboost` — gradient boosting
- `microsoft/LightGBM` — fast GBDT
- `optuna/optuna` — hyperparameter optimization

**For any library, always fetch:**
1. `README.md` — quick start and key examples
2. `docs/` or documentation site — full API reference
3. Recent issues/PRs — what's changing and why
4. Stars and recent commit activity — is it maintained?

---

## Benchmark and Leaderboard Sources

| Topic | Benchmark | URL |
|---|---|---|
| Embedding models | MTEB Leaderboard | `https://huggingface.co/spaces/mteb/leaderboard` |
| LLMs (general) | LMSYS Chatbot Arena | `https://chat.lmsys.org` |
| LLMs (capabilities) | HELM | `https://crfm.stanford.edu/helm/` |
| LLMs (reasoning) | BIG-bench | `https://github.com/google/BIG-bench` |
| Retrieval | BEIR | `https://github.com/beir-cellar/beir` |
| Agents | AgentBench | `https://github.com/THUDM/AgentBench` |
| Code generation | HumanEval, MBPP | `https://github.com/openai/human-eval` |
| Reasoning | GSM8K, MATH | `https://github.com/openai/grade-school-math` |
| Recommenders | RecSys challenge datasets | `https://paperswithcode.com/task/recommendation-systems` |
| Papers with Code | All topics | `https://paperswithcode.com` — find benchmarks + SOTA results |

---

## Domain-Specific Sources

### For topics involving training datasets
Always check:
- `https://huggingface.co/datasets` — search topic, filter by task
- `https://paperswithcode.com/datasets` — datasets with benchmark results
- Dataset cards on HuggingFace — shows format, size, license, usage code

### For topics involving model comparison
Always check:
- `https://huggingface.co/models` — filter by task, sort by downloads (real popularity)
- `https://paperswithcode.com` — SotA results with model comparisons
- MTEB/HELM/Arena for the relevant domain

### For topics involving production systems
Always check:
- AWS/GCP/Azure blog posts for managed service patterns
- Chip Huyen's blog for MLOps and serving patterns
- NVIDIA blog for hardware optimization
- GitHub issues on the key library for common failure modes

---

## What to Extract From Each Source

When you fetch a source, extract:
1. **Core claim** — what does this source say is the best approach?
2. **Key numbers** — benchmarks, speedups, memory savings, accuracy scores
3. **Code pattern** — the canonical way to use this approach in code
4. **Caveats** — what doesn't work well, edge cases, limitations
5. **Publication date** — how recent is this? Is it still SotA?
6. **Citation URL** — the exact URL to link in the blog
