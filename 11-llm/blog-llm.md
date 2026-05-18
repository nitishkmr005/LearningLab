# Large Language Models: Architecture, Training, Fine-Tuning, and Alignment

> A deep technical reference for ML Engineers and AI Engineers — covering transformer internals, scaling laws, fine-tuning methods, RLHF/DPO, and evaluation.

---

## Table of Contents

1. [Transformer Architecture](#1-transformer-architecture)
2. [Attention Mechanisms](#2-attention-mechanisms)
3. [Efficient Attention Variants](#3-efficient-attention-variants)
4. [Positional Encodings](#4-positional-encodings)
5. [Tokenization](#5-tokenization)
6. [Pre-Training Objectives](#6-pre-training-objectives)
7. [Scaling Laws](#7-scaling-laws)
8. [Pre-Training Data and Curation](#8-pre-training-data-and-curation)
9. [Loading and Running LLMs](#9-loading-and-running-llms)
10. [Decoding Strategies](#10-decoding-strategies)
11. [Prompt Engineering](#11-prompt-engineering)
12. [Fine-Tuning: Full and LoRA](#12-fine-tuning-full-and-lora)
13. [QLoRA and PEFT Methods](#13-qlora-and-peft-methods)
14. [Instruction Tuning and RLHF](#14-instruction-tuning-and-rlhf)
15. [DPO and Constitutional AI](#15-dpo-and-constitutional-ai)
16. [Mixture of Experts](#16-mixture-of-experts)
17. [Long-Context and Multimodal LLMs](#17-long-context-and-multimodal-llms)
18. [Knowledge Distillation](#18-knowledge-distillation)
19. [LLM Evaluation](#19-llm-evaluation)
20. [Extended Thinking and Reasoning Models](#20-extended-thinking-and-reasoning-models)
21. [Model Merging](#21-model-merging)
22. [References](#22-references)

---

## 1. Transformer Architecture

The Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) replaced recurrent networks with pure attention, enabling parallelization across sequence positions during training. All modern LLMs are decoder-only Transformers.

**Core components:**

```
Input tokens
     │
Embedding layer (vocab → d_model)
     │
Positional encoding
     │ ┌─────────────────────────────┐
     ├─►  Multi-head self-attention   │  × N layers
     │   Residual + LayerNorm         │
     │   Feed-forward (d_model→4d→d) │
     │   Residual + LayerNorm         │
     └─────────────────────────────┘
     │
Linear projection + softmax → vocabulary logits
```

**Encoder vs Decoder:**
- **Encoder-only (BERT):** bidirectional attention, good for classification and embedding tasks
- **Decoder-only (GPT, LLaMA, Claude):** causal (left-to-right) attention, good for generation
- **Encoder-decoder (T5, BART):** encoder processes input, decoder generates output; best for translation, summarization

All frontier LLMs (GPT-4, Claude, LLaMA, Mistral) are decoder-only.

```python
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    """One transformer decoder block."""
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(                                # feed-forward network
            nn.Linear(d_model, d_ff),
            nn.GELU(),                                          # GELU activation (modern LLMs use SwiGLU)
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)                       # pre-attention layer norm (modern: pre-norm)
        self.ln2 = nn.LayerNorm(d_model)                       # pre-FF layer norm
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm + attention + residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed,        # Q=K=V=normed (self-attention)
                                 attn_mask=causal_mask,
                                 is_causal=True)
        x = x + self.drop(attn_out)                            # residual connection
        # Pre-norm + feed-forward + residual
        x = x + self.drop(self.ff(self.ln2(x)))               # residual connection
        return x
```

---

## 2. Attention Mechanisms

The core attention equation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- Q, K, V are linear projections of the input: Q = XW_Q, K = XW_K, V = XW_V
- Dividing by √d_k prevents dot products from growing large and saturating softmax
- **Causal masking** (decoder) sets future positions to -∞ before softmax, ensuring position i can only attend to positions ≤ i

**Multi-head attention** runs h independent attention heads in parallel, each with d_k = d_model/h dimensions:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(QW_Qi, KW_Ki, VW_Vi)
```

```python
import torch
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor,
                                   V: torch.Tensor,
                                   causal: bool = True) -> torch.Tensor:
    """Scaled dot-product attention with optional causal masking."""
    d_k = Q.size(-1)                                           # key dimension
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (seq, seq)

    if causal:
        seq_len = Q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))       # mask future positions

    weights = torch.softmax(scores, dim=-1)                    # attention weights
    return torch.matmul(weights, V)                            # weighted sum of values

# Test: batch=1, heads=1, seq=4, d_k=8
Q = K = V = torch.randn(1, 4, 8)
out = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {out.shape}")                            # (1, 4, 8)
```

🎯 **Interview prep:** "Why do we divide by √d_k?" — at high d_k, dot products have large variance, pushing softmax into near-zero gradient regions. Dividing by √d_k keeps the dot product variance at ~1.

---

## 3. Efficient Attention Variants

Standard multi-head attention has O(n²) memory in the KV cache — with n heads, each storing K and V for every sequence position.

**Multi-Query Attention (MQA):** all query heads share a single K and V head. Reduces KV cache size by h×. Used in Falcon, PaLM.

**Grouped-Query Attention (GQA)** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)): h query heads share G groups of K/V heads where G < h. Balance between MQA speed and MHA quality. Used in LLaMA 2/3, Mistral, Gemma.

**Flash Attention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)): IO-aware tiling that keeps the attention computation in SRAM (fast) rather than HBM (slow). Does not approximate attention — computes the exact result but 2–4× faster and O(n) memory instead of O(n²).

```
Standard attention:  Read Q,K,V from HBM → compute S=QK^T → write S → read S → softmax → write A → read A,V → compute out
FlashAttention:      Tile Q,K,V in SRAM → compute full attention in tiles → write only output to HBM
```

**Multi-head Latent Attention (MLA)** — DeepSeek's approach: compress K and V into a low-dimensional latent vector, reducing KV cache by 5–8× while preserving expressiveness.

---

## 4. Positional Encodings

Transformers have no inherent notion of position — without encoding, "the cat sat on the mat" and "the mat sat on the cat" would be identical. Several encoding strategies:

**Absolute sinusoidal** (original Transformer): fixed functions of position i and dimension d:
```
PE(i, 2j) = sin(i / 10000^(2j/d_model))
PE(i, 2j+1) = cos(i / 10000^(2j/d_model))
```

**Learned absolute:** learned embedding per position. Limited to training context length.

**RoPE (Rotary Position Embedding)** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)): rotate Q and K vectors by angles proportional to position. The dot product QK depends only on the relative position (i-j), giving good length generalization. Used in LLaMA, Mistral, and most modern models.

**ALiBi** ([Press et al., 2021](https://arxiv.org/abs/2108.12409)): subtract a linear bias from attention scores proportional to distance. No position embedding — directly penalizes long-range attention. Extrapolates well beyond training length.

**YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)): extends RoPE to longer contexts by rescaling the rotation frequencies using NTK-aware interpolation. Enables LLaMA-2 (4K trained length) to handle 128K tokens without full re-training.

---

## 5. Tokenization

Tokenization converts raw text to integer sequences. The vocabulary-speed-quality tradeoff:

- **Large vocabulary** (100K+): fewer tokens per sentence (lower sequence length), but embedding matrix is huge
- **Small vocabulary** (<10K): shorter embedding table, but long sequences

**Byte Pair Encoding (BPE):** start with characters; iteratively merge the most frequent adjacent pair until vocabulary size is reached. Used in GPT-2, GPT-4 (tiktoken), LLaMA.

**WordPiece:** like BPE but merges pairs that maximize likelihood instead of frequency. Used in BERT.

**SentencePiece:** language-agnostic, operates on raw Unicode without pre-tokenization. Used in LLaMA (with BPE), T5.

```python
# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")                    # GPT-4 tokenizer
text = "The Transformer architecture revolutionized NLP."
tokens = enc.encode(text)                                     # tokenize
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded: {enc.decode(tokens)}")                       # round-trip

# Fertility: average tokens per word
words = text.split()
print(f"Fertility: {len(tokens)/len(words):.2f} tokens/word") # ~1.3 for English
```

🎯 **Interview prep:** "How does the tokenizer affect model performance?" — models tokenize code differently than prose. GPT-4 tokenizes Python identifiers as single tokens (good). Chinese, Korean, and Japanese characters often tokenize to many tokens (worse compression, shorter effective context for the same token budget).

---

## 6. Pre-Training Objectives

**Causal Language Modeling (CLM):** predict the next token given all previous tokens. Loss = cross-entropy over all positions. This is how GPT-series, LLaMA, Claude, and all decoder-only models are trained.

```
L_CLM = -Σ_t log P(x_t | x_1, ..., x_{t-1})
```

**Masked Language Modeling (MLM):** mask 15% of tokens randomly; predict the masked tokens from surrounding context. Bidirectional — can see both left and right context. BERT's objective.

**Prefix LM:** some tokens form a "prefix" (seen bidirectionally); the rest are predicted causally. Used in T5 variants.

**Span corruption (T5):** randomly mask contiguous spans; decoder predicts the masked spans. More efficient than MLM — the loss focuses on masked positions only.

```python
import torch
import torch.nn.functional as F

def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor,
                   ignore_index: int = -100) -> torch.Tensor:
    """Compute cross-entropy loss for causal language modeling."""
    # logits: (batch, seq_len, vocab_size)
    # targets: (batch, seq_len) — shift by 1 (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()              # remove last prediction
    shift_targets = targets[:, 1:].contiguous()                # remove first target
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),          # (batch*seq, vocab)
        shift_targets.view(-1),                                 # (batch*seq,)
        ignore_index=ignore_index                               # ignore padding tokens
    )
```

---

## 7. Scaling Laws

**Kaplan et al. (2020)** found that LLM loss follows power laws in model size (N), dataset size (D), and compute (C). Bigger model → lower loss, predictably.

**Chinchilla scaling laws** ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) refined this: given a fixed compute budget C, the optimal allocation is:

```
N_optimal ∝ C^0.5
D_optimal ∝ C^0.5
```

The compute-optimal frontier: **train for approximately 20 tokens per parameter**. GPT-3 was undertrained (300B params, 300B tokens, 1:1 ratio). Chinchilla (70B params, 1.4T tokens, 20:1 ratio) outperforms GPT-3 at less compute.

**Emergent abilities** ([Wei et al., 2022](https://arxiv.org/abs/2206.07682)): some capabilities appear abruptly at scale and are near-zero below a threshold — multi-step arithmetic, few-shot chain-of-thought, code completion. The threshold is model-dependent and unpredictable.

**Inference cost vs training cost:** a model trained once is served billions of times. A smaller model that reaches the same quality as a larger model through better training data is dramatically cheaper in inference costs over its lifetime.

---

## 8. Pre-Training Data and Curation

A 70B parameter Chinchilla-optimal model needs ~1.4T tokens. The main data source is Common Crawl — a web crawl covering petabytes of internet text — filtered for quality.

**Quality filters applied to Common Crawl:**
1. Language identification (keep English or target language)
2. Perplexity filtering: discard text where a small reference LM assigns very high perplexity (likely spam, code garbled as text)
3. Heuristic filters: remove documents with >90% non-alphabetic characters, very short documents, repetition
4. Deduplication: MinHash near-duplicate removal at document level; exact duplicate removal at n-gram level

**Key datasets:**
- **Dolma** ([Soldaini et al., 2023](https://arxiv.org/abs/2309.17453)) — 3T token open dataset from Allen AI
- **RedPajama** — 1.2T token open reproduction of LLaMA training data
- **The Pile** (EleutherAI) — 800GB curated from 22 sources including arXiv, GitHub, Wikipedia

**Data mixing:** combine web crawl with high-quality sources (Wikipedia, books, GitHub code, scientific papers) at carefully tuned ratios. Code data substantially improves reasoning even in non-code tasks.

---

## 9. Loading and Running LLMs

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with automatic device mapping across available GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,                               # use bfloat16 for memory efficiency
    device_map="auto"                                          # auto-distribute across GPUs
)

# Tokenize input
messages = [{"role": "user", "content": "Explain attention in transformers in 2 sentences."}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=200,                                    # limit generation length
        temperature=0.7,                                       # sampling temperature
        top_p=0.9,                                             # nucleus sampling
        do_sample=True,                                        # enable sampling
        pad_token_id=tokenizer.eos_token_id                   # pad with EOS
    )

response = tokenizer.decode(output[0][input_ids.shape[-1]:],  # decode only new tokens
                             skip_special_tokens=True)
print(response)
```

---

## 10. Decoding Strategies

The decoding strategy determines how the model selects the next token from the predicted probability distribution.

| Strategy | Description | When to Use |
|---|---|---|
| **Greedy** | Always pick highest-probability token | Deterministic tasks, structured output |
| **Beam search** | Keep top-K sequences simultaneously | Translation (with references) |
| **Temperature (τ)** | Divide logits by τ before softmax | Creative tasks (high τ), precision tasks (low τ) |
| **Top-k** | Sample from top-k tokens only | Reduce tail noise |
| **Top-p (nucleus)** | Sample from smallest set summing to p | Most production systems |
| **Min-p** | Remove tokens below min_p * max_prob | Better than top-k at high temperatures |

```python
def temperature_sampling(logits: torch.Tensor, temperature: float) -> int:
    """Sample next token with temperature scaling."""
    if temperature == 0:
        return int(torch.argmax(logits))                       # greedy
    scaled = logits / temperature                              # scale logits
    probs = torch.softmax(scaled, dim=-1)                     # convert to probabilities
    return int(torch.multinomial(probs, 1))                    # sample

def top_p_sampling(logits: torch.Tensor, p: float = 0.9) -> int:
    """Nucleus sampling: sample from the smallest token set summing to p."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens once cumulative probability exceeds p
    sorted_probs[cumulative - sorted_probs > p] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()           # renormalize
    idx = int(torch.multinomial(sorted_probs, 1))
    return int(sorted_indices[idx])
```

🎯 **Interview prep:** "What's the difference between temperature and top-p?" — temperature scales the entire distribution (makes it sharper or flatter); top-p truncates the distribution to a probability mass. They address different problems: temperature controls confidence, top-p controls diversity.

---

## 11. Prompt Engineering

Prompting is the fastest way to modify model behavior without any training cost.

**Zero-shot vs few-shot:** provide 3–8 examples of (input, desired output) to demonstrate the task format. Few-shot examples are disproportionately effective at controlling output format and style.

**Chain-of-Thought:** "Let's think step by step" reliably improves multi-step reasoning for models with >100B parameters (but has minimal effect on smaller models).

**Key Anthropic prompting techniques** ([Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)):
- Use XML tags to clearly delimit content sections: `<document>`, `<context>`, `<instructions>`
- Put instructions before the content, not after
- For long documents, restate key instructions near the end
- Prefill the assistant turn to control output format

```python
# Anthropic: prefill assistant turn to control output format
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[
        {"role": "user", "content": "Classify this review as positive, negative, or neutral:\n\n'The product is okay but shipping was very slow.'"},
        {"role": "assistant", "content": '{"sentiment": "'}  # prefill forces JSON output
    ]
)
print('{"sentiment": "' + response.content[0].text)         # complete the JSON
```

---

## 12. Fine-Tuning: Full and LoRA

**Full fine-tuning** updates all model parameters. For a 7B model at fp16, that's 14GB of parameters plus optimizer states (~56GB for AdamW) — requiring multiple high-end GPUs.

**LoRA (Low-Rank Adaptation)** ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) freezes the pre-trained weights and injects trainable low-rank matrices:

```
W' = W + α/r · BA
```

where W is the frozen weight (d×k), B is (d×r) and A is (r×k) are the trainable matrices, r is the rank (typically 8–64), and α is a scaling factor. The number of trainable parameters is r(d+k) instead of dk — a 100× reduction for typical dimensions.

```python
# pip install peft transformers
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                                       # rank — higher = more capacity
    lora_alpha=32,                                              # scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]   # which layers to adapt
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~4M || all params: ~8B || trainable%: ~0.05%
```

At inference time, LoRA weights can be merged into the base model (zero overhead) or kept separate (enables hot-swapping adapters without reloading the base model).

---

## 13. QLoRA and PEFT Methods

**QLoRA** ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) enables fine-tuning of 65B+ models on a single consumer GPU by:
1. Quantizing the frozen base model to 4-bit (NF4 format)
2. Training LoRA adapters in fp16
3. Using "double quantization" to quantize the quantization constants

A 65B model in 4-bit uses ~33GB — fits on a single 40GB A100.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                                         # 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,                    # compute in bfloat16
    bnb_4bit_use_double_quant=True,                            # double quantization
    bnb_4bit_quant_type="nf4"                                  # NF4 quantization type
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Other PEFT methods:**
- **DoRA** (Decomposed LoRA): decomposes weight updates into magnitude and direction components; often outperforms LoRA at the same parameter count
- **IA³** (Infused Adapter by Inhibiting and Amplifying): rescales activations with learned vectors; 100× fewer params than LoRA
- **Prefix Tuning:** learns soft prompt tokens prepended to every layer's key and value; keeps base model completely frozen

---

## 14. Instruction Tuning and RLHF

**Instruction tuning** fine-tunes a pre-trained LLM on (instruction, response) pairs to make it follow instructions rather than just predict next tokens. FLAN ([Wei et al., 2022](https://arxiv.org/abs/2210.11610)) showed that instruction tuning on hundreds of diverse tasks dramatically improves zero-shot generalization.

**RLHF (Reinforcement Learning from Human Feedback)** ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) aligns a model to human preferences in three phases:

1. **Supervised Fine-Tuning (SFT):** fine-tune on high-quality demonstrations
2. **Reward Model (RM):** train a classifier on human preference pairs (preferred vs rejected responses)
3. **RL with PPO:** optimize the SFT model to maximize RM score while staying close to SFT (KL penalty)

```
Objective = E[R_RM(y|x)] - β · KL(π_θ || π_SFT)
```

The KL penalty prevents reward hacking — without it, the model collapses to generating high-reward but nonsensical text.

---

## 15. DPO and Constitutional AI

**DPO (Direct Preference Optimization)** ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) eliminates the separate reward model training phase. It mathematically shows that the optimal policy under RLHF can be expressed directly in terms of the preference data, yielding a simple classification loss:

```
L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

where y_w is the preferred response, y_l is the rejected response, and π_ref is the SFT reference policy. DPO is simpler, more stable, and avoids the difficulty of reward model training and PPO hyperparameter tuning.

**Constitutional AI (CAI)** ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) from Anthropic uses a set of principles ("constitution") to make AI self-critique and revise its outputs. The AI generates responses, critiques them against the principles, and revises them — generating preference data (RLAIF) without human labelers. This scales alignment feedback without proportional human effort.

---

## 16. Mixture of Experts

**MoE (Mixture of Experts)** runs each token through only a subset of the model's "experts" (separate feed-forward layers), making large models cheap to run. A router network selects the top-k experts per token.

**Mixtral 8x7B** ([Jiang et al., 2024](https://arxiv.org/abs/2401.04088)): 8 expert FFN layers per block, 2 selected per token. Total params: ~47B; active params per token: ~13B. Inference cost of a 13B dense model, quality approaching a 70B model.

**DeepSeek-V3** ([DeepSeek, 2024](https://arxiv.org/abs/2412.19437)): 671B total params, 37B active per token. Achieves GPT-4 level performance. Uses an auxiliary load-balancing loss to prevent expert collapse (where all tokens route to the same 2 experts):

```
L_balance = α · Σ_i f_i · P_i
```

where f_i is the fraction of tokens routed to expert i, P_i is the average router probability for expert i, and α is a small constant (0.01). This encourages uniform routing.

---

## 17. Long-Context and Multimodal LLMs

**Long-context extension:** models trained at 4K context length struggle at 32K. Solutions:
- **Positional interpolation:** scale RoPE frequencies to fit longer contexts (interpolate rather than extrapolate)
- **YaRN:** NTK-aware RoPE rescaling that preserves short-context performance while extending context
- **LongLoRA:** efficient fine-tuning for long context using sparse attention during training, reverting to full attention at inference

**Lost-in-the-middle** ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)): LLMs attend better to information at the start and end of long contexts. Middle positions are underutilized — relevant for RAG systems stuffing many chunks.

**Vision-Language Models (VLMs):** patch images into tokens using a Vision Transformer (ViT), project the patch embeddings into the LLM's embedding space, and prepend them to the text context.

**LLaVA** ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)): CLIP visual encoder → linear projection → LLaMA decoder. Two-stage training: (1) train only the projection layer on image-text pairs; (2) fine-tune projection + LLM on instruction-following visual data.

---

## 18. Knowledge Distillation

**Distillation** trains a small "student" model to mimic a large "teacher" model. The student learns from soft targets (teacher's output probability distribution) rather than hard labels — the distribution contains richer signal about what tokens are similar.

```
L_distill = α · L_CE(student_logits, hard_labels) 
           + (1-α) · KL(softmax(teacher_logits/T), softmax(student_logits/T))
```

where T is a temperature that softens the distributions, amplifying small probability differences.

**Sequence-level KD:** distill not just token probabilities but full response sequences — have the teacher generate responses on the training set, then train the student on the teacher's outputs. This is how many open-source models are built from GPT-4 (controversially — violates API ToS).

Notable examples:
- **DistilBERT:** 40% smaller, 60% faster, retains 97% of BERT performance
- **TinyLlama** (1.1B): trained on 3T tokens, outperforms many 7B models on specific tasks

---

## 19. LLM Evaluation

Standard benchmarks for comparing LLMs:

| Benchmark | What It Tests |
|---|---|
| **MMLU** | Academic knowledge across 57 subjects |
| **HellaSwag** | Commonsense reasoning / next-sentence completion |
| **GSM8K** | Grade-school math (multi-step word problems) |
| **HumanEval** | Python code generation (function completion) |
| **MATH** | Competition mathematics |
| **TruthfulQA** | Avoidance of popular misconceptions |
| **ARC-Challenge** | Grade-school science (adversarially filtered) |

```python
# pip install lm-eval
# lm-eval handles benchmark loading, prompting, and scoring automatically

# CLI: lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu,gsm8k --device cuda

# Or in Python:
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct",
    tasks=["mmlu", "gsm8k"],
    batch_size=8,
    device="cuda"
)
print(results["results"])
```

**LLM-as-judge** ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)) uses a strong LLM (GPT-4, Claude) to evaluate open-ended responses on criteria like helpfulness, correctness, and safety. MT-Bench scores models on 80 multi-turn conversation prompts.

---

## 20. Extended Thinking and Reasoning Models

**Chain-of-thought reasoning models** like OpenAI o1/o3, DeepSeek R1, and Claude 3.7 with extended thinking generate long internal reasoning chains before producing a final answer. These "thinking tokens" are usually hidden from the user.

**Key design choices:**
- **Process Reward Models (PRMs)** reward intermediate reasoning steps — higher quality but harder to collect data for
- **Outcome Reward Models (ORMs)** reward only the final answer — simpler, scales better, but can reward correct answers via wrong reasoning
- **Thinking token budget:** reasoning models can trade more thinking tokens for higher accuracy on hard problems

**DeepSeek R1** ([DeepSeek, 2025](https://arxiv.org/abs/2501.12599)) achieves o1-level math and coding performance using GRPO (Group Relative Policy Optimization) — a variant of RLHF that does not require a separate value network, reducing training cost significantly.

```
GRPO objective:
  For a group of G outputs {y_1, ..., y_G} from the old policy:
  Advantage(y_i) = (r_i - mean(r)) / std(r)
  Maximize: E[clip(π_θ/π_old, 1-ε, 1+ε) · Advantage - β·KL]
```

**When reasoning models outperform standard models:**
- Math competition problems (IMO, AIME)
- Multi-step code debugging
- Scientific reasoning requiring hypothesis generation and testing
- Logic puzzles with many valid intermediate steps

They underperform standard models on simple tasks (latency and cost are much higher for the same quality on easy queries).

---

## 21. Model Merging

Model merging combines multiple fine-tuned models without retraining, enabling multi-capability models from specialist components.

**Task vectors** ([Ilharco et al., 2023](https://arxiv.org/abs/2312.01552)): a task vector is the difference between fine-tuned and base model weights: τ = θ_ft - θ_base. Adding task vectors combines capabilities:

```
θ_merged = θ_base + λ_1·τ_1 + λ_2·τ_2
```

**SLERP (Spherical Linear Interpolation):** interpolate weights on the hypersphere rather than linearly. Better preserves vector magnitudes:

```
SLERP(W_1, W_2, t) = sin((1-t)·θ)/sin(θ) · W_1 + sin(t·θ)/sin(θ) · W_2
```

**DARE-TIES** ([Yu et al., 2023](https://arxiv.org/abs/2311.01388)): randomly drop (DARE) and resolve sign conflicts (TIES) before merging. Handles interference between task vectors that target the same weights.

**MergeKit** ([arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)) implements all major merging strategies and supports merging models too large to fit in VRAM via CPU offloading.

**When merging beats fine-tuning:** merging is free (no GPU training); it works best when combining specialist models trained from the same base. It fails when the task vectors conflict strongly (opposite directions in weight space) or when the target capability requires information genuinely absent from all source models.

---

## 22. References

### Architecture

- [Vaswani et al. (2017). Attention Is All You Need.](https://arxiv.org/abs/1706.03762)
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models.](https://arxiv.org/abs/2305.13245)
- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention.](https://arxiv.org/abs/2205.14135)

### Positional Encoding

- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE).](https://arxiv.org/abs/2104.09864)
- [Press et al. (2021). Train Short, Test Long: Attention with Linear Biases (ALiBi).](https://arxiv.org/abs/2108.12409)
- [Peng et al. (2023). YaRN: Efficient Context Window Extension.](https://arxiv.org/abs/2309.00071)

### Pre-Training and Scaling

- [Vaswani et al. (2017). BERT.](https://arxiv.org/abs/1810.04805)
- [Brown et al. (2020). GPT-3.](https://arxiv.org/abs/2005.14165)
- [Hoffmann et al. (2022). Chinchilla Scaling Laws.](https://arxiv.org/abs/2203.15556)
- [Soldaini et al. (2023). Dolma Dataset.](https://arxiv.org/abs/2309.17453)

### Fine-Tuning and Alignment

- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.](https://arxiv.org/abs/2106.09685)
- [Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.](https://arxiv.org/abs/2305.14314)
- [Wei et al. (2022). Finetuned Language Models Are Zero-Shot Learners (FLAN).](https://arxiv.org/abs/2210.11610)
- [Ouyang et al. (2022). InstructGPT: Training Language Models to Follow Instructions.](https://arxiv.org/abs/2203.02155)
- [Rafailov et al. (2023). Direct Preference Optimization (DPO).](https://arxiv.org/abs/2305.18290)
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback.](https://arxiv.org/abs/2212.08073)

### Architecture Innovations

- [Jiang et al. (2024). Mixtral of Experts.](https://arxiv.org/abs/2401.04088)
- [DeepSeek (2024). DeepSeek-V3 Technical Report.](https://arxiv.org/abs/2412.19437)
- [Liu et al. (2023). LLaVA: Visual Instruction Tuning.](https://arxiv.org/abs/2304.08485)

### Reasoning Models

- [DeepSeek (2025). DeepSeek R1: Incentivizing Reasoning Capability in LLMs via RL.](https://arxiv.org/abs/2501.12599)
- [Anthropic (2025). Claude 3.7 Sonnet.](https://www.anthropic.com/news/claude-3-7-sonnet)

### Evaluation and Merging

- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench.](https://arxiv.org/abs/2306.05685)
- [Ilharco et al. (2023). Editing Models with Task Arithmetic.](https://arxiv.org/abs/2312.01552)
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MergeKit (Arcee AI)](https://github.com/arcee-ai/mergekit)
