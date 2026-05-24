# LearningLab — Reference URLs

All URLs sourced from bookmarks. 16 core topics appear first; extra categories follow.

---

## 01 — SQL

- **SQL Database Agent with LangChain**
  Step-by-step tutorial on building a natural language → SQL agent using LangChain's SQL toolkit, covering prompt design, query execution safety, and output formatting.
  <https://medium.com/@LawrencewleKnight/build-your-first-sql-database-agent-with-langchain-19af8064ae18>

---

## 02 — Statistics

_No bookmarks added for this topic yet._

---

## 03 — Python & Pandas

- **Poetry — Dependency Management & Packaging**
  Official docs for Poetry, the modern Python packaging tool. Covers virtual environments, `pyproject.toml`, dependency resolution, and publishing to PyPI.
  <https://python-poetry.org/docs/>

---

## 04 — PyTorch

_No bookmarks added for this topic yet._

---

## 05 — Git

_No bookmarks added for this topic yet._

---

## 06 — Machine Learning

- **creme — Online / Streaming Machine Learning**
  Python library for incremental learning: models that update one sample at a time without storing data. Covers streams, concept drift, online metrics, and pipelines.
  <https://github.com/sroecker/creme>

- **NLP with Python** (susanli2016)
  Curated Jupyter notebooks covering text classification, sentiment analysis, topic modelling, and NER using scikit-learn, NLTK, and spaCy — a good bridge between classical ML and NLP.
  <https://github.com/susanli2016/NLP-with-Python>

---

## 07 — Recommenders

- **Microsoft Recommenders — Best Practices**
  Microsoft's production-quality repository covering collaborative filtering (ALS, SVD), content-based, hybrid, and deep learning recommenders (NCF, LightGBM, SAR). Includes notebooks and evaluation utilities.
  <https://github.com/recommenders-team/recommenders>

- **LightFM**
  Hybrid recommendation library that combines collaborative and content-based filtering via matrix factorisation with user/item feature embeddings. Good baseline for cold-start problems.
  <https://github.com/lyst/lightfm>

---

## 08 — Embedding Models

- **FlagEmbedding — BAAI BGE Models**
  Repository for BAAI's BGE embedding family (BGE-M3, BGE-reranker, LLM-Embedder). BGE-M3 supports 100+ languages and mixed dense/sparse/ColBERT retrieval in one model. Consistently top-ranked on MTEB.
  <https://github.com/FlagOpen/FlagEmbedding>

- **Training and Fine-tuning Sparse Embedding Models with Sentence Transformers v5** (HuggingFace Blog)
  Walks through training SPLADE-style sparse encoders for hybrid search — covers data prep, `SparseEncoder` class, loss functions (FLOPS regularisation), and MTEB evaluation.
  <https://huggingface.co/blog/train-sparse-encoder>

- **Contextual Document Embeddings — cde-small-v1** (HuggingFace)
  Model page for CDE (Morris et al., 2024), which conditions each document embedding on the surrounding corpus via a dataset embedder — top MTEB performance at small size.
  <https://huggingface.co/jxm/cde-small-v1>

- **GLiNER — Zero-Shot NER** (HuggingFace Space)
  Interactive demo for GLiNER, a span-based zero-shot named entity recogniser useful for extracting structured entities from chunks before embedding or knowledge-graph construction.
  <https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1>

---

## 09 — RAG

- **RAG Techniques** (NirDiamant)
  Largest curated notebook collection on RAG: from naive RAG through HyDE, RAPTOR, contextual compression, self-RAG, adaptive retrieval, and agentic RAG — each as a standalone runnable notebook.
  <https://github.com/NirDiamant/RAG_Techniques>

- **Anthropic Claude Cookbooks**
  Anthropic's official examples repo. The `capabilities/` folder contains the canonical Contextual Retrieval implementation, classification, and summarisation patterns built around Claude.
  <https://github.com/anthropics/claude-cookbooks>

- **Building a Knowledge Graph From Scratch Using LLMs** (Towards Data Science)
  Tutorial on extracting entities and relations from unstructured text with LLMs and building a queryable Neo4j knowledge graph — an alternative retrieval structure to vector search.
  <https://medium.com/towards-data-science/building-a-knowledge-graph-from-scratch-using-llms-f6f677a17f07>

- **Firecrawl — Docs**
  Official documentation for Firecrawl, a web-crawling API that converts any URL to clean Markdown for RAG ingestion. Covers scrape, crawl, map, and extract endpoints.
  <https://docs.firecrawl.dev/introduction>

- **Firecrawl — GitHub**
  Open-source repo for self-hosting Firecrawl. Useful for understanding how JS-rendered pages, sitemaps, and rate-limiting are handled in production web-to-RAG pipelines.
  <https://github.com/firecrawl/firecrawl>

- **Docling** (IBM)
  IBM's document parser that converts PDFs, DOCX, PPTX, and images to structured Markdown or JSON while preserving tables, figures, and reading order — purpose-built for RAG pre-processing.
  <https://github.com/docling-project/docling>

- **MarkItDown** (Microsoft)
  Lightweight Microsoft tool for converting Office files, PDFs, HTML, and audio to Markdown for LLM ingestion. Simpler than Docling for quick pipelines.
  <https://github.com/microsoft/markitdown>

- **Local MCP Client with LlamaIndex** (Colab Notebook)
  Notebook showing how to build a local MCP client using LlamaIndex and Ollama — demonstrates tool-calling patterns inside retrieval pipelines.
  <https://github.com/patchy631/ai-engineering-hub/blob/main/llamaindex-mcp/ollama_client.ipynb>

---

## 10 — Agents

- **LangGraph — Official Docs**
  Documentation for LangGraph, the graph-based stateful agent framework on top of LangChain. Covers nodes, edges, conditional routing, persistence, streaming, human-in-the-loop, and multi-agent patterns.
  <https://langchain-ai.github.io/langgraph/>

- **LangGraph Course — freeCodeCamp**
  Full implementation repo for the popular freeCodeCamp LangGraph course — builds agents with tools, memory, reflection, and streaming from scratch.
  <https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp>

- **Reflection Agents** (LangChain Blog)
  Explains the reflection pattern: agent generates → critiques its own output → revises. Covers the generate/reflect loop and compares it to RLHF-based improvement.
  <https://blog.langchain.com/reflection-agents/>

- **Reflection Agent Implementation** (emarco177)
  Working LangGraph implementation of the reflection agent from Eden Marco's course — clean reference for the generate → reflect → revise state graph.
  <https://github.com/emarco177/langgraph-course/tree/project/reflection-agent>

- **Reflexion Agent Implementation** (emarco177)
  Implementation of the Reflexion paper (Shinn et al., 2023): verbal reinforcement where the agent reflects on past failures stored in episodic memory to improve future actions.
  <https://github.com/emarco177/reflexion/tree/main>

- **CrewAI**
  Multi-agent orchestration framework that assigns roles, goals, and backstories to agents. Role-based collaboration with task routing — widely used in production agent pipelines.
  <https://github.com/crewAIInc/crewAI>

- **CrewAI Examples**
  Official example projects: research assistants, code review, trip planners, financial analysis — real end-to-end agent workflows using CrewAI.
  <https://github.com/crewAIInc/crewAI-examples>

- **OpenAI Swarm**
  OpenAI's lightweight experimental framework for multi-agent orchestration, focused on agent handoffs and routines. Minimal abstraction — good for learning the primitives.
  <https://github.com/openai/swarm>

- **OpenAI Codex CLI**
  OpenAI's Codex-powered CLI for AI-assisted coding with file context, shell access, and tool use. Reference for how coding agents interact with file systems and terminals.
  <https://github.com/openai/codex>

- **mem0 — Memory Layer for AI Agents**
  Persistent memory infrastructure for agents: stores facts, preferences, and context across sessions using a hybrid vector + graph store. Drop-in long-term memory for any agent framework.
  <https://github.com/mem0ai/mem0>

- **agentmemory** (rohitg00)
  Persistent memory implementations for AI coding agents with real-world benchmarks. Good reference for comparing memory strategies (vector, KV, structured).
  <https://github.com/rohitg00/agentmemory>

- **Awesome LLM Apps**
  Curated collection of LLM-powered apps (RAG, agents, voice, multimodal) with source code across many use cases — good for surveying what's buildable end-to-end.
  <https://github.com/Shubhamsaboo/awesome-llm-apps>

- **TradingAgents**
  Multi-agent system for financial market analysis with LLM agents playing analyst, researcher, and trader roles — a practical end-to-end production-grade agent system reference.
  <https://github.com/tauricresearch/tradingagents>

- **wshobson/agents — Multi-Agent Orchestration for Claude Code**
  Intelligent automation and multi-agent orchestration patterns specifically designed for Claude Code workflows.
  <https://github.com/wshobson/agents>

- **Deep Agents — Blog Post** (LangChain via skills.sh)
  LangChain's deep agents technical post covering architecture, planning, tool use, and agent memory design.
  <https://skills.sh/langchain-ai/deepagents/blog-post>

- **Deep Agents — LangGraph Docs** (LangChain via skills.sh)
  LangGraph documentation bundled as a skill — useful for exploring the latest graph-based agent primitives alongside the deep-agents project.
  <https://skills.sh/langchain-ai/deepagents/langgraph-docs>

---

## 11 — LLM

- **LLMs from Scratch** (Sebastian Raschka)
  Step-by-step PyTorch implementation of an LLM from scratch: BPE tokenisation, multi-head attention, pre-training loop, fine-tuning, and RLHF. Companion to the O'Reilly book.
  <https://github.com/rasbt/LLMs-from-scratch>

- **LLaMA-Factory**
  Unified fine-tuning framework for 100+ LLMs: LoRA, QLoRA, full fine-tuning, DPO, PPO, GRPO, and RLHF — the most complete open fine-tuning toolkit available.
  <https://github.com/hiyouga/LLaMA-Factory>

- **LLaMA-Factory — Example Configs**
  Official example scripts and configs within LLaMA-Factory covering Alpaca SFT, DPO preference alignment, multi-task training, and evaluation across model families.
  <https://github.com/hiyouga/LLaMA-Factory/tree/main/examples>

- **Ultrascale Playbook — Training LLMs on GPU Clusters** (HuggingFace)
  Interactive guide to distributed LLM training: data parallelism, tensor parallelism, pipeline parallelism, ZeRO, mixed precision — with an interactive parameter explorer for memory and compute trade-offs.
  <https://huggingface.co/spaces/nanotron/ultrascale-playbook>

- **Continued LLM Pretraining with Unsloth** (Unsloth Blog)
  Practical guide to continued pretraining for domain adaptation — covers data formatting, packing, training loop, and evaluation with a working Colab notebook.
  <https://unsloth.ai/blog/contpretraining>

- **Mistral v0.3 (7B) — Continued Pretraining Colab** (Unsloth)
  Unsloth's Colab notebook for continued pretraining of Mistral 7B v0.3 — shows the exact training loop, data packing, and checkpointing setup.
  <https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb>

- **Mistral (7B) — Text Completion Fine-tuning Colab** (Unsloth)
  Unsloth's Colab notebook for fine-tuning Mistral 7B for text completion — covers QLoRA setup, training loop, and inference.
  <https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb>

- **Fine-tune GPT-2 for Text Generation Using PyTorch** (Towards Data Science)
  Tutorial walking through GPT-2 fine-tuning with HuggingFace Transformers and PyTorch — covers dataset prep, `Trainer` API, and text generation sampling strategies.
  <https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7>

- **Text Generation with HuggingFace GPT-2** (Kaggle)
  Kaggle notebook demonstrating GPT-2 text generation: tokenisation, model loading, greedy/beam/sampling decoding, and temperature/top-k/top-p controls.
  <https://www.kaggle.com/tuckerarrants/text-generation-with-huggingface-gpt2>

---

## 12 — LLM Inferencing

- **Inside vLLM: Anatomy of a High-Throughput LLM Inference System** (Aleksa Gordić)
  Deep technical walkthrough of vLLM internals — PagedAttention, continuous batching, chunked prefill, speculative decoding, and the scheduler design. One of the best system-level explanations available.
  <https://www.aleksagordic.com/blog/vllm>

- **KV Cache from Scratch in nanoVLM** (HuggingFace Blog)
  Builds a KV cache from first principles inside a minimal VLM codebase — explains why the cache exists, how it grows per decode step, and static vs dynamic allocation.
  <https://huggingface.co/blog/kv-cache>

- **KV Cache Explained: From Simple Buffers to Distributed Memory** (Medium)
  Comprehensive guide tracing KV cache evolution: per-layer buffers → PagedAttention → offloaded and distributed caches in production multi-GPU systems.
  <https://luv-bansal.medium.com/the-evolution-of-kv-cache-from-simple-buffers-to-distributed-memory-systems-df51cb8ce26f>

- **LLM RAM Calculator**
  Interactive calculator for estimating VRAM/RAM needed to load and run an LLM at various quantisation precisions (fp32, fp16, int8, int4). Essential before any local deployment decision.
  <https://llm-calc.rayfernando.ai/>

- **MLX — Apple ML Framework Docs**
  Official docs for Apple's MLX framework — NumPy-like API optimised for Apple Silicon unified memory. Covers arrays, autograd, neural net layers, and distributed training on Mac.
  <https://ml-explore.github.io/mlx/build/html/index.html>

- **mlx-examples**
  Official MLX example implementations: LLaMA, Mixtral, Whisper, Stable Diffusion, LoRA fine-tuning — all tuned for Apple Silicon. Fastest path to running LLMs on a Mac.
  <https://github.com/ml-explore/mlx-examples>

- **exo — Distributed Inference Across Consumer Devices**
  Clusters heterogeneous consumer devices (iPhones, MacBooks, Raspberry Pis) into a single inference backend via ring attention and dynamic model partitioning.
  <https://github.com/exo-explore/exo>

- **Open WebUI**
  Self-hosted ChatGPT-like UI for local LLMs via Ollama or any OpenAI-compatible backend. Supports multimodal input, RAG, agents, and function calling.
  <https://github.com/open-webui/open-webui>

- **A Complete Guide to AI Accelerators for Deep Learning Inference** (Towards Data Science)
  Covers GPUs (A100, H100), AWS Inferentia, Amazon Trainium, and TPUs — when to use each, cost trade-offs, and throughput/latency benchmarks for inference workloads.
  <https://towardsdatascience.com/a-complete-guide-to-ai-accelerators-for-deep-learning-inference-gpus-aws-inferentia-and-amazon-7a5d6804ef1c>

---

## 13 — LLM Evaluation

- **DeepEval — LLM Evaluation Framework**
  Production-grade eval framework with 14+ metrics for RAG (faithfulness, contextual precision/recall), agents, hallucination, and general LLM quality. Integrates with CI/CD pipelines via pytest.
  <https://github.com/confident-ai/deepeval>

- **Prometheus-Eval**
  Open-source LLM-based judge trained to replicate GPT-4-level evaluation — supports absolute scoring, relative ranking, and critique generation at low cost.
  <https://github.com/prometheus-eval/prometheus-eval>

- **LangSmith**
  LangChain's platform for tracing, evaluating, testing, and monitoring LLM applications. Captures full input/output traces, supports human annotation, dataset management, and automated evals.
  <https://smith.langchain.com>

- **Opik by Comet ML**
  Open-source LLM evaluation and observability platform — tracing, automated metrics (hallucination, relevance, toxicity), experiment comparison, and a self-hostable backend.
  <https://www.comet.com/opik>

---

## 14 — Reinforcement Learning

_No bookmarks added for this topic yet._

---

## 15 — Speech

- **Aman's AI Journal — Speech Processing Primer**
  Comprehensive primer covering the full speech stack: ASR (CTC, RNN-T, Whisper architecture), TTS (Tacotron2, FastSpeech2, VITS), speaker diarisation, codec models, and audio transformers.
  <https://aman.ai/primers/ai/speech-processing/>

- **Pocket TTS** (Kyutai Research)
  Kyutai's post on their lightweight, high-quality TTS model optimised for real-time low-latency synthesis — directly relevant for voice agent pipelines requiring fast streaming audio.
  <https://kyutai.org/blog/2026-01-13-pocket-tts>

- **NVIDIA NeMo TTS Primer** (Jupyter Notebook)
  Official NeMo notebook walking through TTS inference and fine-tuning with FastPitch + HiFi-GAN pre-trained models — hands-on starting point for production TTS on NVIDIA hardware.
  <https://github.com/NVIDIA-NeMo/NeMo/blob/stable/tutorials/tts/NeMo_TTS_Primer.ipynb>

- **NVIDIA NeMo TTS — Official Docs**
  Documentation for NeMo's TTS toolkit covering architecture choices (FastPitch, Mixer-TTS, HiFi-GAN, EnCodec), training, and inference APIs.
  <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/intro.html>

- **DeepFilterNet — Neural Noise Suppression**
  Real-time deep learning noise suppressor for clean speech output — useful as a preprocessing step before ASR transcription or speaker diarisation in production pipelines.
  <https://github.com/rikorose/deepfilternet>

- **FireRedVAD — Voice Activity Detection**
  FireRed's fast, accurate VAD model for detecting speech segments in audio — standard preprocessing component used before ASR transcription.
  <https://github.com/FireRedTeam/FireRedVAD>

---

## 16 — Vision & OCR

- **screenshot-to-code**
  Converts screenshots, Figma designs, and wireframes into HTML/Tailwind/React code using GPT-4o or Claude vision — practical reference for vision + code generation agent pipelines.
  <https://github.com/abi/screenshot-to-code>

- **Genesis — Physics Simulation for Embodied AI** (GitHub)
  Generative world simulation platform for robotics and embodied AI — builds 4D dynamic scenes from text descriptions. State-of-the-art environment for robot learning research.
  <https://github.com/Genesis-Embodied-AI/Genesis>

- **Genesis — Project Website**
  Official website with demos, documentation links, and research papers for the Genesis simulation platform.
  <https://genesis-embodied-ai.github.io/>

- **KLING AI**
  AI-powered video generation platform from Kuaishou — generates high-quality videos from text or images. Relevant for understanding the state of commercial multimodal generation.
  <https://www.klingai.com/>

- **Runway**
  Generative AI video and image editing platform — reference for understanding commercial vision generation models and their API capabilities.
  <https://runwayml.com/>

---
---

## A — General AI Resources

Cross-topic reading, research discovery, and authoritative blogs.

- **Hugging Face — Learn**
  Free courses and hands-on notebooks for NLP, computer vision, audio, RL, and agents. The canonical starting point for any HuggingFace library.
  <https://huggingface.co/learn>

- **Hugging Face — Docs**
  Official API documentation for Transformers, Datasets, PEFT, Accelerate, Diffusers, and every other HuggingFace library.
  <https://huggingface.co/docs>

- **Hugging Face — Blog**
  Technical posts from HuggingFace researchers: new model releases, training techniques, fine-tuning walkthroughs, and benchmark analyses.
  <https://huggingface.co/blog>

- **Hugging Face — Models**
  Search and download any open-weight model — filter by task, library, language, and licence. The primary model registry for the open-source AI ecosystem.
  <https://huggingface.co/models>

- **aman.ai — AI Primers**
  Long-form, equation-heavy primers on transformers, attention, RLHF, RAG, agents, speech, and more by Aman Chadha. Recommended pre-read before any deep-dive blog in this repo.
  <https://aman.ai/>

- **Ahead of AI — Sebastian Raschka**
  Newsletter + blog covering LLM research, training techniques, and paper breakdowns with working code. Raschka is author of *Build a Large Language Model from Scratch* (O'Reilly).
  <https://magazine.sebastianraschka.com/>

- **Daily Dose of DS**
  Daily visual tips on statistics, ML, Python, and AI — good for filling conceptual gaps quickly without reading a full paper.
  <https://www.dailydoseofds.com/>

- **alphaXiv**
  Enhanced arXiv viewer with threaded community discussion on ML/AI papers. Read reactions before deciding which papers to study deeply.
  <https://www.alphaxiv.org/>

- **Unsloth Blog**
  Technical posts from the Unsloth team on efficient fine-tuning, continued pretraining, GRPO, and quantisation — very practical, always shipped with Colab notebooks.
  <https://unsloth.ai/blog>

- **Genmind.ch — Posts** (Gian Paolo Santopaolo)
  Technical blog covering LLM systems, inference optimisation, and distributed ML — referenced from the Runpod ecosystem.
  <https://genmind.ch/posts/>

- **Gemini 3 Developer Guide** (Google AI)
  Official guide for the Gemini 3 model family: capabilities, context window, multimodal inputs, and API usage patterns.
  <https://ai.google.dev/gemini-api/docs/gemini-3>

- **Gemini Image Generation API** (Google AI)
  Documentation for Gemini's image generation capabilities via Imagen 3 — prompt engineering, safety filters, and API usage.
  <https://ai.google.dev/gemini-api/docs/image-generation>

---

## B — MCP — Model Context Protocol

- **MCP — Building a Client Tutorial** (modelcontextprotocol.io)
  Official step-by-step tutorial for building an MCP client from scratch — covers the JSON-RPC protocol, tool registration, and request/response flow.
  <https://modelcontextprotocol.io/tutorials/building-a-client>

- **@modelcontextprotocol/sdk** (npm)
  Official TypeScript/JavaScript SDK for MCP. Reference for client and server implementation, schema definitions, and transport types (stdio, SSE).
  <https://www.npmjs.com/package/@modelcontextprotocol/sdk>

- **awesome-mcp-servers**
  Curated list of MCP server implementations across databases, APIs, file systems, code tools, and more — the best starting point for discovering what's available.
  <https://github.com/punkpeye/awesome-mcp-servers>

- **modelcontextprotocol/servers** (GitHub)
  Official repository of reference MCP server implementations (filesystem, GitHub, Postgres, Puppeteer, etc.) maintained by the MCP team.
  <https://github.com/modelcontextprotocol/servers>

- **fastmcp**
  High-level Python framework for building MCP servers with minimal boilerplate — decorators for tool registration, automatic schema generation, and built-in testing utilities.
  <https://github.com/jlowin/fastmcp>

- **Context7**
  MCP server that serves up-to-date library documentation to AI coding assistants — eliminates hallucinated API calls by grounding the model in real, versioned docs.
  <https://context7.com/>

- **upstash/context7** (GitHub)
  Open-source code for the Context7 MCP server — useful for understanding how documentation is indexed, chunked, and served to LLM clients.
  <https://github.com/upstash/context7>

---

## C — Claude Code & AI Coding Skills

Resources for Claude Code, skill development, and AI-assisted coding workflows.

- **Claude** — <https://claude.ai/new>
- **Claude Code** — <https://claude.ai/code>
- **Claude API Docs** — <https://platform.claude.com/docs/en/home>

- **Claude Code Handbook** (nikiforovall.blog)
  Community-maintained handbook of Claude Code rules, patterns, hooks, and best practices for getting the most out of Claude Code in daily development.
  <https://nikiforovall.blog/claude-code-rules/>

- **Claude Code Plugins & Agent Skills Directory**
  Searchable directory of Claude Code plugins and skills sorted by downloads — browse what the community has built.
  <https://claude-plugins.dev/>

- **Plugins for Claude Code** (Anthropic)
  Anthropic's official plugins page for Claude Code — curated list of available integrations and cowork tools.
  <https://claude.com/plugins>

- **awesome-claude-code**
  Community-curated list of Claude Code tips, scripts, hooks, prompts, and workflow improvements.
  <https://github.com/hesreallyhim/awesome-claude-code>

- **awesome-claude-code-subagents**
  Collection of reusable Claude Code sub-agent definitions for common engineering tasks.
  <https://github.com/VoltAgent/awesome-claude-code-subagents>

- **claude-code-templates**
  Ready-to-use Claude Code project templates and CLAUDE.md starters for different project types.
  <https://github.com/davila7/claude-code-templates>

- **claude-code-cheat-sheet**
  Quick reference sheet for Claude Code commands, shortcuts, and configuration options.
  <https://github.com/Njengah/claude-code-cheat-sheet>

- **claude-code-guide**
  Comprehensive guide to setting up and using Claude Code effectively across different project types.
  <https://github.com/zebbern/claude-code-guide>

- **claude-flow**
  Workflow orchestration framework built on Claude Code — enables complex multi-step agentic flows with state management.
  <https://github.com/ruvnet/claude-flow>

- **Claude Code Superpowers** (obra)
  Collection of Claude Code hooks, commands, and configurations that extend its default capabilities significantly.
  <https://github.com/obra/superpowers>

- **claude-mem**
  Persistent memory implementation for Claude Code — stores and retrieves structured context across sessions.
  <https://github.com/thedotmack/claude-mem>

- **claude-scientific-skills** (K-Dense-AI)
  Skills for scientific computing tasks in Claude Code — covers data analysis, plotting, LaTeX, and research workflows.
  <https://github.com/K-Dense-AI/claude-scientific-skills>

- **andrej-karpathy-skills**
  Claude Code skills inspired by Andrej Karpathy's coding style and pedagogical approach — includes ML and deep learning focused skills.
  <https://github.com/forrestchang/andrej-karpathy-skills>

- **anthropics/skills** (GitHub)
  Anthropic's official repository of Claude Code skills — the reference implementation for how skills should be structured.
  <https://github.com/anthropics/skills>

- **huggingface/skills** (GitHub)
  HuggingFace's collection of Claude Code skills for ML workflows — model loading, training, evaluation, and HuggingFace Hub interactions.
  <https://github.com/huggingface/skills>

- **mattpocock/skills**
  Skills for professional software engineers by Matt Pocock — production-quality patterns straight from his `.claude` directory.
  <https://github.com/mattpocock/skills>

- **Agent Skills Marketplace** (skillsmp.com)
  Marketplace for discovering, sharing, and installing Claude Code agent skills across domains.
  <https://skillsmp.com/>

- **The Agent Skills Directory** (skills.sh)
  Directory of Claude Code skills searchable by category, author, and use case.
  <https://skills.sh/>

- **github/spec-kit**
  GitHub's framework for writing AI-readable feature specifications — structured format that helps AI coding assistants understand requirements precisely.
  <https://github.com/github/spec-kit>

- **ccusage**
  CLI tool for tracking and visualising Claude Code token usage and costs over time.
  <https://github.com/ryoppippi/ccusage>

- **CodexBar** (steipete)
  macOS menu bar app for quick access to AI coding tools — productivity utility for AI-assisted development.
  <https://github.com/steipete/CodexBar>

- **Clawdbot — Personal AI Assistant**
  Personal AI assistant built on Claude — reference for building a persistent conversational agent with memory and tool use.
  <https://clawd.bot/>

- **Cursor Directory**
  Community-maintained collection of `.cursorrules` files for different tech stacks — transferable patterns for AI coding assistant configuration.
  <https://cursor.directory/>

---

## D — MLOps

- **MLflow**
  Open-source platform for experiment tracking, model versioning, packaging, and serving. The most widely deployed MLOps tool in production data science teams.
  <https://mlflow.org/>

- **DVC — Data Version Control**
  Git-compatible version control for data and ML models — tracks large files in remote storage (S3, GCS, Azure) with a lightweight metadata layer.
  <https://dvc.org/>

- **Cookiecutter Data Science**
  Opinionated project structure template for reproducible data science — enforces separation of raw data, features, models, and reports from the start.
  <https://drivendata.github.io/cookiecutter-data-science/>

- **mlops_main** (c17hawke)
  End-to-end MLOps demo project covering data ingestion, validation, transformation, model training, evaluation, and deployment — good reference architecture.
  <https://github.com/c17hawke/mlops_main>

- **wafer_mlops_docs** (c17hawke)
  Documentation and walkthrough for a wafer fault detection MLOps project — covers the full ML pipeline with CI/CD using GitHub Actions.
  <https://github.com/c17hawke/wafer_mlops_docs>

- **Deploy ML Pipeline on Google Kubernetes Engine** (Towards Data Science)
  Tutorial on containerising an ML model with Docker and deploying via CI/CD to GKE — covers zero-downtime deployments and autoscaling.
  <https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b>

---

## E — AI Tools & Platforms

AI inference, model hosting, experimentation, and productivity platforms.

- **Google AI Studio**
  Google's web-based playground for Gemini models — API key management, prompt experimentation, multimodal input, and usage monitoring.
  <https://aistudio.google.com>

- **OpenAI API Platform**
  OpenAI's developer platform for GPT-4o, o1, embeddings, and fine-tuning — API key management, usage dashboards, and playground.
  <https://platform.openai.com>

- **OpenRouter**
  Unified API across 100+ LLM providers (OpenAI, Anthropic, Google, Meta, Mistral, etc.) — single key, standardised endpoint, usage-based billing. Good for model comparisons.
  <https://openrouter.ai/models>

- **GroqCloud**
  Fastest publicly available LLM inference via custom LPU (Language Processing Unit) hardware — hosts Llama 3, Mixtral, and Whisper. Use for latency-sensitive applications.
  <https://console.groq.com>

- **DeepSeek**
  DeepSeek's chat interface for DeepSeek-V3 and DeepSeek-R1 — state-of-the-art open-weight models for reasoning and coding tasks.
  <https://chat.deepseek.com/>

- **Perplexity AI**
  AI-powered search engine that cites sources — useful for quick literature search and finding authoritative references during research.
  <https://www.perplexity.ai/>

- **Mistral AI Chat**
  Chat interface for Mistral's models (Mistral Large, Codestral, Pixtral) — direct access for experimentation and API prototyping.
  <https://chat.mistral.ai/>

- **NotebookLM** (Google)
  AI-powered research assistant — upload papers, docs, or URLs and ask questions with grounded citations. Excellent for synthesising multiple papers before writing a blog.
  <https://notebooklm.google.com/>

- **Google Colab**
  Free GPU/TPU Jupyter notebook environment — primary tool for running and sharing training experiments without local hardware.
  <https://colab.research.google.com>

- **Runpod**
  GPU cloud platform with per-minute billing — on-demand access to H100/A100/RTX pods for training and inference. Cost-effective for burst compute needs.
  <https://www.runpod.io/>

- **Lambda AI — The Superintelligence Cloud**
  GPU cloud optimised for AI/ML workloads — persistent instances, on-demand GPUs, and clusters with pre-installed ML frameworks.
  <https://lambda.ai/>

- **Lightning AI**
  Cloud platform by the PyTorch Lightning team — managed training, Studios (cloud dev environments), and deployment for ML models.
  <https://lightning.ai/>

- **Ollama**
  One-command local LLM runner — downloads and runs open models (Llama 3, Mistral, Phi-3, Gemma) locally with a simple REST API compatible with the OpenAI SDK.
  <https://ollama.com/search>

- **Open WebUI** (openwebui.com)
  Self-hostable ChatGPT-like interface connected to Ollama or any OpenAI-compatible endpoint — supports multimodal, RAG, tool use, and plugin extensions.
  <https://openwebui.com/>

- **Gamma.app**
  AI-powered presentation builder — converts prompts or documents into polished slide decks. Useful for quickly generating visual summaries of technical topics.
  <https://gamma.app/>

- **Replit**
  Cloud-based IDE with AI coding assistance — run Python, Node, and other runtimes instantly in-browser. Good for quick prototyping without local setup.
  <https://replit.com/>

- **CodeGPT**
  AI coding assistant platform with support for multiple LLM backends (Claude, GPT-4, local models) integrated into VS Code and other editors.
  <https://app.codegpt.co/>

- **Wispr Flow**
  AI-powered voice dictation tool for macOS — converts speech to text in any app using a local model. Useful for dictating code comments, documentation, or prompts.
  <https://wisprflow.ai/>

- **PopAI**
  AI productivity platform with document analysis, image generation, and chat — useful for quick document Q&A during research.
  <https://www.popai.pro/>

---

## F — Web & Frontend Development

UI components, design tools, no-code builders, and frontend infrastructure.

- **v0 by Vercel**
  AI-powered UI generation from text prompts — outputs React + Tailwind/shadcn components. Fast way to scaffold UI prototypes for ML demos and tools.
  <https://v0.dev/>

- **bolt.new**
  Full-stack AI coding environment in the browser — generates, runs, and deploys complete web apps from a prompt. Vite + React + Node.js by default.
  <https://bolt.new/>

- **CopyCoder**
  Converts screenshots of UIs into production-ready code — complements screenshot-to-code with a more structured output pipeline.
  <https://copycoder.ai/>

- **Framer**
  Professional no-code website builder with CMS, animations, and React component support — used for production-quality sites without a full dev stack.
  <https://www.framer.com/>

- **Aceternity UI**
  Library of animated React + Tailwind components with striking visual effects (glowing cards, 3D tilt, spotlight, etc.) — useful for AI product landing pages.
  <https://ui.aceternity.com/>

- **HeroUI (formerly NextUI)**
  Beautiful, accessible React component library with dark mode support and a modern design system built on Tailwind and Radix UI.
  <https://www.heroui.com/>

- **Supahero — Hero Section Library**
  Curated library of website hero section designs and code snippets — good reference for landing page UI patterns for AI products.
  <https://www.supahero.io/>

- **Aura — Free Web Templates**
  Browse and download free web component templates — useful for jumpstarting dashboards, landing pages, and admin panels.
  <https://www.aura.build/>

- **Bento Grids**
  Gallery of bento grid layout designs and components — the grid-card layout pattern popularised by Apple, widely used in AI product UIs.
  <https://bentogrids.com/>

- **Iconify**
  Unified icon library aggregating 200,000+ icons from 150+ icon sets — searchable, framework-agnostic, and available as SVG or component.
  <https://icon-sets.iconify.design/>

- **Google Fonts**
  Free, open-source font library hosted by Google — 1,500+ typefaces optimised for web performance with variable font support.
  <https://fonts.google.com/>

- **SwiftUI** (Apple Developer)
  Apple's declarative UI framework for iOS, macOS, watchOS, and tvOS — relevant for building native AI-powered apps on Apple platforms.
  <https://developer.apple.com/xcode/swiftui/>

- **Sandbox.dev by Vibecode**
  Browser-based coding sandbox with live preview — useful for testing UI components and prototyping without local setup.
  <https://sandbox.dev/>

- **Instantdb**
  Realtime local-first database for React apps — sync state across clients instantly with a Firebase-like API, but self-hostable.
  <https://www.instantdb.com/>

- **Convex**
  Full-stack TypeScript backend — realtime database, serverless functions, and file storage designed for React apps. Good for building AI app backends quickly.
  <https://docs.convex.dev/home>

- **Resend**
  Developer-first email API — send transactional emails from code with React Email templates. Useful for notifications in AI-powered SaaS apps.
  <https://resend.com/>

---

## G — Visualization & Diagramming

- **Excalidraw**
  Virtual whiteboard for sketching diagrams, system designs, and architecture flows with a hand-drawn aesthetic — widely used for ML system design sketches.
  <https://excalidraw.com/>

- **Manim — Quickstart** (Manim Community)
  Official quickstart tutorial for Manim, the Python animation library used by 3Blue1Brown — create mathematical animations and visualisations programmatically.
  <https://docs.manim.community/en/stable/tutorials/quickstart.html>

- **Manim — Rendering Text and Formulas**
  Guide for rendering LaTeX formulas and styled text in Manim animations — essential for creating visual explanations of ML algorithms.
  <https://docs.manim.community/en/stable/guides/using_text.html>

- **Mermaid Live Editor**
  Browser-based Mermaid diagram editor with live preview — renders flowcharts, sequence diagrams, ER diagrams, and Gantt charts from markdown-like syntax.
  <https://mermaid.live/>

- **Anime.js**
  Lightweight JavaScript animation engine — smooth SVG and CSS animations for ML dashboards, interactive tutorials, and product demos.
  <https://animejs.com/>

- **Carbon**
  Create and share beautiful code screenshots — customisable themes, fonts, and backgrounds. Standard tool for sharing code snippets in blog posts and slides.
  <https://carbon.now.sh/>

---

## H — Data & Search APIs

APIs for web search, web scraping, and data retrieval — feed RAG pipelines and agents with fresh external data.

- **SerpApi**
  Structured search results API (Google, Bing, Yahoo, etc.) — returns parsed JSON from search engine results pages. Used in agents that need web search as a tool.
  <https://serpapi.com/>

- **Brave Search API**
  Privacy-focused search API with independent indexing — good alternative for agents needing web search without Google dependency.
  <https://api-dashboard.search.brave.com/>

- **Tavily Search API**
  AI-optimised web search API that returns clean, summarised results designed for LLM consumption — native integration with LangChain and LlamaIndex.
  <https://app.tavily.com/>

- **Firecrawl — App Dashboard**
  Firecrawl's hosted dashboard for managing crawl jobs, API keys, and monitoring usage — complements the self-hosted GitHub version.
  <https://www.firecrawl.dev/>

---

## I — Algorithms & CS Concepts

Foundational CS algorithms relevant to search, string matching, and data structures used in ML systems.

- **Damn Cool Algorithms: BK-Trees** (Nick's Blog)
  Classic post explaining BK-Trees — a metric space data structure for efficient fuzzy string matching. Directly relevant to typo-tolerant search and entity resolution in NLP pipelines.
  <http://blog.notdot.net/2007/4/Damn-Cool-Algorithms-Part-1-BK-Trees>

- **Damn Cool Algorithms: Levenshtein Automata** (Nick's Blog)
  Explains Levenshtein automata for fast approximate string matching at O(n) regardless of dictionary size — used in spell checkers and fuzzy search engines.
  <http://blog.notdot.net/2010/07/Damn-Cool-Algorithms-Levenshtein-Automata>

- **Awesome Regex Resources**
  Curated list of regex tutorials, tools, cheat sheets, and libraries — useful for text preprocessing, extraction patterns, and structured data parsing in ML pipelines.
  <https://github.com/Varunram/Awesome-Regex-Resources/blob/master/README.md>

---

## J — Dev Tools

Terminal, productivity, and developer environment tools used alongside the ML/AI workflow.

- **ghostty** (ghostty-org)
  Fast, native terminal emulator with GPU rendering, split panes, and rich customisation — written in Zig, excellent macOS and Linux support.
  <https://github.com/ghostty-org/ghostty>

- **CleanShot X**
  macOS screenshot and screen recording tool with annotation, scrolling capture, and cloud hosting — standard tool for documenting ML experiments and UI changes.
  <https://cleanshot.com/>

- **Udemy Business**
  Corporate Udemy platform for structured video courses — used primarily for ML, deep learning, and data engineering courses.
  <https://www.udemy.com/>
