# LearningLab Reference URLs

Curated links from bookmarks, organized by topic. Every URL here came from your saved bookmarks — no extras added.

---

## General Resources

These span multiple topics and are worth checking frequently.

- **Hugging Face — Learn**
  Free courses and hands-on notebooks covering NLP, vision, audio, RL, and agents. The most beginner-friendly entry point into each HuggingFace library.
  <https://huggingface.co/learn>

- **Hugging Face — Docs**
  Official API documentation for Transformers, Datasets, PEFT, Accelerate, Diffusers, and every other HuggingFace library.
  <https://huggingface.co/docs>

- **Hugging Face — Blog**
  Technical posts from HuggingFace researchers: new model releases, training techniques, fine-tuning walkthroughs, and benchmark analyses.
  <https://huggingface.co/blog>

- **Aman's AI Journal (aman.ai)**
  Long-form primers on transformers, attention, RLHF, RAG, agents, and more. Highly recommended as a pre-read before any deep-dive blog in this repo.
  <https://aman.ai>

- **Ahead of AI — Sebastian Raschka**
  Newsletter + blog covering LLM research, training techniques, and paper breakdowns with working code. Raschka is author of *Build a Large Language Model from Scratch*.
  <https://magazine.sebastianraschka.com>

- **Daily Dose of DS**
  Daily visual tips on statistics, ML, Python, and AI. Good for filling gaps on foundational concepts quickly.
  <https://www.dailydoseofds.com>

- **alphaXiv**
  Enhanced arXiv viewer with discussion threads and comments on ML/AI papers. Useful for seeing community reactions to a paper before reading it.
  <https://www.alphaxiv.org>

- **Unsloth Blog**
  Technical posts from the Unsloth team on efficient fine-tuning, continued pretraining, GRPO, and quantization. Very practical, with Colab notebooks.
  <https://unsloth.ai/blog>

---

## 07 — Recommenders

- **recommenders-team/recommenders** (Microsoft)
  Microsoft's best-practices repository for recommendation systems. Covers collaborative filtering, content-based, deep learning (NCF, LightGBM, SAR, ALS), and more — with production-quality implementations.
  <https://github.com/recommenders-team/recommenders>

- **LightFM**
  Hybrid recommendation library combining collaborative and content-based filtering via matrix factorisation with item/user features. Good baseline for cold-start problems.
  <https://github.com/lyst/lightfm>

---

## 08 — Embedding Models

- **FlagEmbedding (BAAI)**
  Repository for BAAI's BGE embedding model family — consistently top-ranked on MTEB leaderboard. Includes BGE-M3 (multi-lingual, multi-granularity) and reranker models.
  <https://github.com/FlagOpen/FlagEmbedding>

- **Training and Fine-tuning Sparse Embedding Models with Sentence Transformers v5** (HuggingFace Blog)
  Step-by-step guide to training sparse encoders (SPLADE-style) for hybrid search, covering data prep, loss functions, and evaluation.
  <https://huggingface.co/blog/train-sparse-encoder>

- **Contextual Document Embeddings — cde-small-v1** (HuggingFace)
  Model page for CDE (John Morris et al.), which conditions each embedding on the surrounding corpus context — state-of-the-art on MTEB when this was released.
  <https://huggingface.co/jxm/cde-small-v1>

- **GLiNER — Medium Span NER**
  HuggingFace Space demoing GLiNER, a zero-shot named entity recogniser useful for extracting structured chunks before embedding.
  <https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1>

---

## 09 — RAG

- **RAG Techniques** (NirDiamant)
  Comprehensive notebook collection from naive RAG through advanced techniques: HyDE, RAPTOR, adaptive retrieval, and self-RAG. Each technique is a standalone, runnable notebook.
  <https://github.com/NirDiamant/RAG_Techniques>

- **Anthropic Claude Cookbooks**
  Anthropic's official cookbook repo. The `capabilities/` folder includes contextual embeddings RAG (contextual retrieval), summarisation, and classification examples with production patterns.
  <https://github.com/anthropics/claude-cookbooks>

- **Building a Knowledge Graph From Scratch Using LLMs** (Towards Data Science)
  Tutorial on extracting entities and relations from unstructured text with LLMs and wiring them into a queryable knowledge graph — an alternative retrieval structure to vector search.
  <https://medium.com/towards-data-science/building-a-knowledge-graph-from-scratch-using-llms-f6f677a17f07>

- **Firecrawl Docs**
  Documentation for Firecrawl, a web-crawling API that converts any URL into clean Markdown for RAG ingestion. Covers scraping, crawling, mapping, and extraction endpoints.
  <https://docs.firecrawl.dev/introduction>

- **Docling** (IBM)
  Open-source document parser that converts PDFs, DOCX, PPTX, and images to structured Markdown or JSON. Preserves tables, figures, and reading order — purpose-built for RAG pre-processing.
  <https://github.com/docling-project/docling>

- **MarkItDown** (Microsoft)
  Lightweight Microsoft tool for converting Office files, PDFs, HTML, and audio to Markdown for LLM ingestion. Simpler alternative to Docling for quick pipelines.
  <https://github.com/microsoft/markitdown>

- **Local MCP Client with LlamaIndex** (Colab notebook)
  Jupyter notebook demonstrating how to build a local MCP client using LlamaIndex and Ollama — useful for understanding MCP-based tool calling in retrieval pipelines.
  <https://github.com/patchy631/ai-engineering-hub/blob/main/llamaindex-mcp/ollama_client.ipynb>

---

## 10 — Agents

- **LangGraph — Official Docs**
  Documentation for LangGraph, the graph-based stateful agent framework built on LangChain. Covers nodes, edges, state, human-in-the-loop, persistence, and multi-agent patterns.
  <https://langchain-ai.github.io/langgraph/>

- **LangGraph Course — freeCodeCamp Implementation**
  Full implementation repo for the popular freeCodeCamp LangGraph course. Covers building agents with tools, memory, reflection, and streaming from scratch.
  <https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp>

- **Reflection Agents** (LangChain Blog)
  Explains the reflection agent pattern: the agent generates a response, then critiques its own output and revises — a lightweight alternative to RLHF for quality improvement.
  <https://blog.langchain.com/reflection-agents/>

- **Reflection Agent Implementation** (emarco177/langgraph-course)
  Working LangGraph implementation of the reflection agent from the course by Eden Marco. Good reference for the generate → reflect → revise loop.
  <https://github.com/emarco177/langgraph-course/tree/project/reflection-agent>

- **Reflexion Agent Implementation** (emarco177/reflexion)
  Implementation of the Reflexion paper (Shinn et al., 2023) — verbal reinforcement learning where the agent reflects on past failures stored in memory.
  <https://github.com/emarco177/reflexion/tree/main>

- **CrewAI**
  Multi-agent orchestration framework that assigns roles, goals, and backstories to agents. Focuses on role-based collaboration and is widely used in production agent pipelines.
  <https://github.com/crewAIInc/crewAI>

- **CrewAI Examples**
  Official example projects for CrewAI covering research assistants, code review, trip planning, and other real-world multi-agent workflows.
  <https://github.com/crewAIInc/crewAI-examples>

- **OpenAI Swarm**
  OpenAI's lightweight experimental framework for multi-agent orchestration, focused on agent handoffs and routines. Minimal abstraction — good for learning the primitives.
  <https://github.com/openai/swarm>

- **OpenAI Codex CLI**
  OpenAI's Codex-powered CLI for AI-assisted coding with file context. Relevant for understanding how coding agents interact with file systems and terminals.
  <https://github.com/openai/codex>

- **mem0 — Memory Layer for AI Agents**
  Persistent memory infrastructure for AI agents: stores facts, preferences, and context across sessions using a hybrid vector + graph store. Used to give agents long-term memory.
  <https://github.com/mem0ai/mem0>

- **agentmemory** (rohitg00)
  Persistent memory implementations for AI coding agents, benchmarked on real-world tasks. Useful reference for comparing memory strategies.
  <https://github.com/rohitg00/agentmemory>

- **Awesome LLM Apps**
  Curated collection of LLM-powered applications (RAG, agents, voice, multimodal) with source code. Good for seeing what the community is building end-to-end.
  <https://github.com/Shubhamsaboo/awesome-llm-apps>

- **TradingAgents**
  Multi-agent system for financial market analysis using LLM agents playing analyst, researcher, and trader roles. A practical end-to-end agent system reference.
  <https://github.com/tauricresearch/tradingagents>

- **SQL Database Agent with LangChain** (Medium)
  Tutorial on building a natural language to SQL agent using LangChain's SQL toolkit, covering prompt design and safe query execution.
  <https://medium.com/@LawrencewleKnight/build-your-first-sql-database-agent-with-langchain-19af8064ae18>

---

## 11 — LLM

- **LLMs from Scratch** (Sebastian Raschka)
  Complete, chapter-by-chapter implementation of an LLM in PyTorch — tokenisation, attention, pre-training, fine-tuning, RLHF. Companion to the O'Reilly book.
  <https://github.com/rasbt/LLMs-from-scratch>

- **LLaMA-Factory**
  Unified fine-tuning framework supporting 100+ LLMs with LoRA, QLoRA, full fine-tuning, DPO, PPO, GRPO, and RLHF. The most complete open fine-tuning toolkit available.
  <https://github.com/hiyouga/LLaMA-Factory>

- **LLaMA-Factory — Examples**
  Official example configs and scripts within LLaMA-Factory covering Alpaca fine-tuning, preference alignment, and multi-task training across model families.
  <https://github.com/hiyouga/LLaMA-Factory/tree/main/examples>

- **Ultrascale Playbook — Training LLMs on GPU Clusters** (HuggingFace)
  HuggingFace's interactive guide to distributed LLM training: data parallelism, tensor parallelism, pipeline parallelism, ZeRO, and mixed precision. Interactive parameter explorer included.
  <https://huggingface.co/spaces/nanotron/ultrascale-playbook>

- **Continued LLM Pretraining with Unsloth** (Unsloth Blog)
  Practical guide to continued pretraining for domain adaptation using Unsloth — covers data formatting, training loop, and evaluation, with Colab notebook.
  <https://unsloth.ai/blog/contpretraining>

---

## 12 — LLM Inferencing

- **Inside vLLM: Anatomy of a High-Throughput LLM Inference System** (Aleksa Gordić)
  Deep technical walkthrough of vLLM's internals — PagedAttention, continuous batching, chunked prefill, speculative decoding, and the scheduler. One of the best system-level explanations available.
  <https://www.aleksagordic.com/blog/vllm>

- **KV Cache from Scratch in nanoVLM** (HuggingFace Blog)
  HuggingFace blog building KV cache from first principles inside a minimal VLM codebase. Explains why the cache exists, how it grows, and how static vs dynamic allocation differ.
  <https://huggingface.co/blog/kv-cache>

- **KV Cache Explained: From Simple Buffers to Distributed Memory** (Medium)
  Comprehensive guide tracing KV cache evolution: per-layer buffers → PagedAttention → distributed/offloaded caches in production systems.
  <https://luv-bansal.medium.com/the-evolution-of-kv-cache-from-simple-buffers-to-distributed-memory-systems-df51cb8ce26f>

- **LLM RAM Calculator**
  Interactive calculator for estimating VRAM/RAM needed to load and run an LLM at various quantisation precisions (fp32, fp16, int8, int4). Useful before any local deployment decision.
  <https://llm-calc.rayfernando.ai/?ram=48>

- **MLX — Apple's ML Framework Docs**
  Official documentation for MLX, Apple's NumPy-like ML framework optimised for Apple Silicon unified memory. Covers arrays, autograd, neural nets, and distributed training on Mac.
  <https://ml-explore.github.io/mlx/build/html/index.html>

- **mlx-examples**
  Official MLX example implementations: LLaMA, Mixtral, Whisper, Stable Diffusion, LoRA fine-tuning — all tuned for Apple Silicon. The fastest way to run LLMs on a Mac.
  <https://github.com/ml-explore/mlx-examples>

- **exo — Run LLMs Across Consumer Devices**
  Framework that clusters heterogeneous consumer devices (iPhones, MacBooks, Raspberry Pis) into a single inference backend. Implements ring attention and dynamic model partitioning.
  <https://github.com/exo-explore/exo>

- **Open WebUI**
  Self-hosted ChatGPT-like UI for running LLMs locally via Ollama, OpenAI API, or any OpenAI-compatible backend. Supports multimodal, RAG, agents, and function calling.
  <https://github.com/open-webui/open-webui>

---

## 13 — LLM Evaluation

- **DeepEval — LLM Evaluation Framework**
  Production-grade evaluation framework with 14+ metrics for RAG (faithfulness, contextual precision/recall), agents, and general LLM quality. Integrates with CI/CD pipelines.
  <https://github.com/confident-ai/deepeval>

- **Prometheus-Eval**
  Open-source LLM-based judge (trained on Feedback Collection dataset) that replicates GPT-4-level evaluation at low cost. Supports absolute and relative scoring with critique generation.
  <https://github.com/prometheus-eval/prometheus-eval>

- **LangSmith**
  LangChain's platform for tracing, evaluating, testing, and monitoring LLM applications. Captures full input/output traces, supports human annotation, and runs automated evals.
  <https://smith.langchain.com>

- **Opik by Comet ML**
  Open-source LLM evaluation and observability platform. Provides tracing, automated metrics (hallucination, relevance, toxicity), and a dashboard for comparing runs.
  <https://www.comet.com/opik>

---

## 15 — Speech

- **Aman's AI Journal — Speech Processing Primer**
  Comprehensive primer covering the full speech stack: ASR (CTC, RNN-T, Whisper), TTS (Tacotron, FastSpeech, VITS), speaker diarisation, and audio transformers.
  <https://aman.ai/primers/ai/speech-processing/>

- **Pocket TTS** (Kyutai Research)
  Kyutai's blog post on their lightweight, high-quality TTS model optimised for low-latency real-time synthesis — relevant for voice agent pipelines.
  <https://kyutai.org/blog/2026-01-13-pocket-tts>

- **NVIDIA NeMo TTS Primer** (Jupyter Notebook)
  Official NVIDIA NeMo notebook walking through TTS inference and fine-tuning with pre-trained FastPitch + HiFi-GAN models. Good hands-on starting point.
  <https://github.com/NVIDIA-NeMo/NeMo/blob/stable/tutorials/tts/NeMo_TTS_Primer.ipynb>

- **NVIDIA NeMo TTS — Official Docs**
  Documentation for NVIDIA's NeMo TTS toolkit covering model architectures (FastPitch, Mixer-TTS, HiFi-GAN), training, and inference APIs.
  <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/intro.html>

- **DeepFilterNet — Neural Noise Suppression**
  Deep learning noise suppression model (real-time capable) for cleaning speech audio. Useful as a preprocessing step before ASR or speaker diarisation.
  <https://github.com/rikorose/deepfilternet>

- **FireRedVAD — Voice Activity Detection**
  FireRed's fast, accurate VAD model for detecting speech segments in audio. Preprocessing component used before ASR transcription in production pipelines.
  <https://github.com/FireRedTeam/FireRedVAD>

---

## 16 — Vision & OCR

- **screenshot-to-code**
  Converts screenshots, Figma designs, and wireframes to HTML/Tailwind/React code using GPT-4o or Claude vision. A practical reference for vision + code generation agents.
  <https://github.com/abi/screenshot-to-code>

- **Genesis — Physics Simulation for Embodied AI**
  Generative world simulation platform for robotics and embodied AI — builds 4D dynamic scenes from text descriptions. State-of-the-art for robot learning environments.
  <https://github.com/Genesis-Embodied-AI/Genesis>

---

## Tools & Platforms (Cross-topic)

Useful tools that support multiple topics across this repo.

- **LangSmith** — Tracing and evaluation for LLM apps: <https://smith.langchain.com>
- **Opik** — Open-source LLM observability: <https://www.comet.com/opik>
- **NotebookLM** — AI-powered research assistant for reading papers: <https://notebooklm.google.com>
- **Open WebUI** — Local LLM UI with RAG and agents: <https://github.com/open-webui/open-webui>
- **Ollama** — Run open LLMs locally with one command: <https://ollama.com/search>
- **HuggingFace Models** — Search and download any open model: <https://huggingface.co/models>
- **Google AI Studio** — Gemini API playground and key management: <https://aistudio.google.com>
- **OpenRouter** — Unified API across 100+ LLM providers: <https://openrouter.ai/models>
- **GroqCloud** — Fastest inference API (Llama, Mixtral, Whisper) via custom LPU hardware: <https://console.groq.com>
- **MLflow** — Experiment tracking, model registry, and serving: <https://mlflow.org>
- **DVC** — Data and model version control for ML pipelines: <https://dvc.org>
