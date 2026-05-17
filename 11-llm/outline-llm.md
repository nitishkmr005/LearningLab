# 08 — LLM (Large Language Models)

Exhaustive learning path: architecture, pre-training, prompting, fine-tuning, alignment, and evaluation.

---

## 01 — Transformer Architecture (from scratch)
Multi-head self-attention; positional encoding; layer norm; residual connections; encoder vs decoder-only.
- https://jalammar.github.io/illustrated-transformer/
- https://nlp.seas.harvard.edu/annotated-transformer/

## 02 — Attention: Scaled Dot-Product & Multi-Head
QKV projections; scaling by √d_k; causal masking; multi-head concatenation; complexity O(n²d).
- https://arxiv.org/abs/1706.03762

## 03 — Efficient Attention Variants
Multi-Query (MQA); Grouped-Query (GQA); Multi-head Latent (MLA); Flash Attention IO-aware tiling.
- https://arxiv.org/abs/2305.13245 (GQA)
- https://arxiv.org/abs/2205.14135 (FlashAttention)

## 04 — Positional Encodings
Absolute sinusoidal; learned; RoPE (rotary); ALiBi (linear bias); YaRN for context extension.
- https://arxiv.org/abs/2104.09864 (RoPE)
- https://arxiv.org/abs/2108.12409 (ALiBi)

## 05 — Tokenization
BPE, WordPiece, SentencePiece, Unigram; vocabulary trade-offs; byte-level BPE; tiktoken; fertility.
- https://huggingface.co/docs/tokenizers/index
- https://www.youtube.com/watch?v=zduSFxRajkE (Karpathy BPE video)

## 06 — Pre-Training Objectives
Causal LM (CLM/GPT); masked LM (MLM/BERT); prefix LM; span corruption (T5); next sentence prediction.
- https://arxiv.org/abs/1810.04805 (BERT)
- https://arxiv.org/abs/2005.14165 (GPT-3)

## 07 — Scaling Laws
Chinchilla compute-optimal (tokens ∝ params); emergent abilities; inference cost vs training cost.
- https://arxiv.org/abs/2203.15556 (Chinchilla)
- https://arxiv.org/abs/2001.08361 (Kaplan)

## 08 — Pre-Training Data & Curation
Common Crawl filtering; quality signals; deduplication; data mixing; Dolma, RedPajama datasets.
- https://arxiv.org/abs/2309.17453 (Dolma)

## 09 — Loading & Running LLMs (HuggingFace)
AutoModel + AutoTokenizer; device_map="auto"; generate() params; batched inference; stopping criteria.
- https://huggingface.co/docs/transformers/llm_tutorial

## 10 — Decoding Strategies
Greedy, beam search, temperature sampling, top-k, top-p, min-p; repetition penalty; diversity beam.
- https://huggingface.co/docs/transformers/generation_strategies

## 11 — Prompt Engineering
Zero-shot, few-shot, CoT, system prompts, role prompting, delimiters, XML tags, structured output prompts.
- https://www.promptingguide.ai/
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview

## 12 — Function Calling & Tool Use
JSON schema definitions; parallel tool calls; multi-turn tool use; Anthropic and OpenAI APIs.
- https://docs.anthropic.com/en/docs/tool-use
- https://platform.openai.com/docs/guides/function-calling

## 13 — Structured Output (JSON Mode)
Instructor library; Outlines constrained decoding; Pydantic extraction; reliability across models.
- https://python.useinstructor.com/
- https://github.com/outlines-dev/outlines

## 14 — Full Fine-Tuning with HuggingFace Trainer
DataCollator; compute_metrics; gradient accumulation; mixed precision; early stopping; saving checkpoints.
- https://huggingface.co/docs/transformers/training

## 15 — LoRA: Low-Rank Adaptation
Freeze base; inject trainable A·B matrices into attention projections; rank r; merge at inference.
- https://arxiv.org/abs/2106.09685
- https://huggingface.co/docs/peft/index

## 16 — QLoRA: Quantized LoRA
4-bit NF4 quantization of frozen base + fp16 LoRA; double quantization; train 65B on 1 GPU.
- https://arxiv.org/abs/2305.14314

## 17 — Other PEFT Methods: DoRA, IA³, Prefix Tuning
DoRA decomposes magnitude + direction; IA³ rescales activations; Prefix Tuning learns soft prompt tokens.
- https://huggingface.co/docs/peft/index

## 18 — Instruction Tuning & FLAN
Fine-tune on (instruction, response) pairs; FLAN-T5; task diversity improves zero-shot generalization.
- https://arxiv.org/abs/2210.11610

## 19 — RLHF (Reinforcement Learning from Human Feedback)
Reward model from preference pairs; PPO to maximize reward while staying close to reference policy.
- https://arxiv.org/abs/2203.02155 (InstructGPT)

## 20 — DPO (Direct Preference Optimization)
Bypass explicit reward model; optimize policy directly from (chosen, rejected) pairs; simpler than PPO.
- https://arxiv.org/abs/2305.18290

## 21 — Constitutional AI & RLAIF
Self-critique with principles; AI-generated preference labels; reduce reliance on human labelers.
- https://arxiv.org/abs/2212.08073

## 22 — Mixture of Experts (MoE)
Sparse gating; top-k expert routing; auxiliary load-balancing loss; Mixtral 8x7B; DeepSeek-V3.
- https://arxiv.org/abs/2401.04088 (Mixtral)
- https://arxiv.org/abs/2412.19437 (DeepSeek-V3)

## 23 — Long-Context LLMs
Positional interpolation; LongLoRA; YaRN; sliding window + global attention; lost-in-the-middle.
- https://arxiv.org/abs/2307.03172 (LongLoRA)
- https://arxiv.org/abs/2309.00071 (YaRN)

## 24 — Multimodal LLMs (Vision-Language)
ViT patch embeddings; cross-attention or concat fusion; LLaVA; Gemini; GPT-4V.
- https://arxiv.org/abs/2304.08485 (LLaVA)

## 25 — Knowledge Distillation for LLMs
Teacher → student; soft targets; sequence-level KD; DistilBERT; TinyLlama; data-free distillation.
- https://arxiv.org/abs/1911.02727

## 26 — Catastrophic Forgetting & Continual Learning
Replay buffers; EWC (elastic weight consolidation); LoRA adapters for task-specific layers.
- https://arxiv.org/abs/1612.00796

## 27 — LLM Evaluation: Benchmarks
MMLU, HellaSwag, GSM8K, HumanEval, MATH, ARC, TruthfulQA; LM-Evaluation-Harness; LLM-as-judge.
- https://github.com/EleutherAI/lm-evaluation-harness
- https://arxiv.org/abs/2306.05685

## 28 — Training a Tiny LLM (NanoGPT)
Character-level GPT from scratch; dataset prep; training loop; generation; understand every component.
- https://github.com/karpathy/nanoGPT
- https://www.youtube.com/watch?v=kCc8FmEb1nY

## 29 — Extended Thinking & Reasoning Models
o1 / o3 architecture; DeepSeek R1; Claude 3.7 extended thinking; chain-of-thought budgets (thinking tokens); process reward models vs outcome reward models; when reasoning models outperform standard; cost vs quality.
- https://arxiv.org/abs/2501.12599
- https://www.anthropic.com/news/claude-3-7-sonnet

## 30 — Model Merging
Task vectors; SLERP (spherical linear interpolation); DARE-TIES (prune + merge); merging without retraining; MergeKit; when merging beats fine-tuning; domain-specialist + generalist fusion.
- https://arxiv.org/abs/2312.01552
- https://github.com/arcee-ai/mergekit
