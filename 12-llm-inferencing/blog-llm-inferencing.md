# LLM Inferencing: KV Cache to Production Serving at Scale

> A systems-level guide to LLM serving — quantization, batching, caching, parallelism, and hardware efficiency — for ML Engineers and AI Engineers.

---

## Table of Contents

1. [Autoregressive Decoding and the KV Cache](#1-autoregressive-decoding-and-the-kv-cache)
2. [Latency Metrics: TTFT, TPOT, Throughput](#2-latency-metrics-ttft-tpot-throughput)
3. [Mixed Precision: FP16 and BF16](#3-mixed-precision-fp16-and-bf16)
4. [Quantization: INT8, GPTQ, AWQ](#4-quantization-int8-gptq-awq)
5. [GGUF and llama.cpp](#5-gguf-and-llamacpp)
6. [Flash Attention](#6-flash-attention)
7. [vLLM and PagedAttention](#7-vllm-and-pagedattention)
8. [Continuous Batching](#8-continuous-batching)
9. [Speculative Decoding](#9-speculative-decoding)
10. [Prompt Caching](#10-prompt-caching)
11. [Tensor and Pipeline Parallelism](#11-tensor-and-pipeline-parallelism)
12. [Expert Parallelism for MoE](#12-expert-parallelism-for-moe)
13. [Constrained Decoding](#13-constrained-decoding)
14. [Disaggregated Prefill and Decode](#14-disaggregated-prefill-and-decode)
15. [TensorRT-LLM and ONNX](#15-tensorrt-llm-and-onnx)
16. [LoRA Serving at Scale](#16-lora-serving-at-scale)
17. [KV Cache Quantization and Compression](#17-kv-cache-quantization-and-compression)
18. [Benchmarking and Profiling](#18-benchmarking-and-profiling)
19. [References](#19-references)

---

## 1. Autoregressive Decoding and the KV Cache

Every LLM generates text one token at a time. At step t, the model runs a full forward pass over the entire sequence (tokens 1…t) to predict token t+1. Naively, this recomputes all attention keys and values for every previous token at every step — O(t) work per token, O(t²) total.

The **KV cache** eliminates this redundancy: after computing the key K_i and value V_i for token i, store them. At step t+1, only compute K_{t+1} and V_{t+1} for the new token, retrieve all cached K_{1..t} and V_{1..t}, and compute attention. Each decoding step is now O(1) in attention operations.

**KV cache memory footprint:**

```
KV cache size = 2 × num_layers × num_kv_heads × d_head × seq_len × bytes_per_element
```

For LLaMA-3.1-70B (80 layers, 8 KV heads, 128 head dim, fp16):
```
= 2 × 80 × 8 × 128 × seq_len × 2 bytes
= 327,680 × seq_len bytes
= ~320 KB per token
```

A batch of 32 requests, each 2K tokens long = 32 × 2000 × 320KB = ~20GB — just for the KV cache, before model weights (140GB).

```python
def kv_cache_size_gb(n_layers: int, n_kv_heads: int, d_head: int,
                     seq_len: int, batch_size: int, dtype_bytes: int = 2) -> float:
    """Estimate KV cache memory in GB."""
    total_bytes = (2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * dtype_bytes)
    return total_bytes / (1024 ** 3)

# LLaMA-3.1-70B serving 32 requests at 2K tokens each (fp16)
gb = kv_cache_size_gb(n_layers=80, n_kv_heads=8, d_head=128,
                       seq_len=2000, batch_size=32, dtype_bytes=2)
print(f"KV cache: {gb:.1f} GB")                               # ~19.5 GB
```

🎯 **Interview prep:** "Where does the memory go when serving LLMs?" — model weights + KV cache. For a 70B model: ~140GB weights + ~20GB KV cache per batch. The KV cache grows with sequence length and batch size, which is why long-context serving is expensive.

---

## 2. Latency Metrics: TTFT, TPOT, Throughput

Two fundamentally different latency metrics characterize LLM serving:

**TTFT (Time to First Token):** latency from request submission to receiving the first output token. This is dominated by the **prefill phase** — running a forward pass over the entire input prompt. Long prompts → high TTFT. Measures user-perceived responsiveness for streaming use cases.

**TPOT (Time Per Output Token):** average latency per token during the **decode phase**. This is dominated by memory bandwidth — loading model weights and KV cache from HBM. Measures streaming speed (tokens/sec visible to the user).

**Throughput:** total tokens (input + output) processed per second across all concurrent requests. Maximized with large batch sizes. Trades off against latency (larger batches = higher TPOT for individual requests).

```python
import time
from anthropic import Anthropic

def measure_llm_latency(prompt: str) -> dict:
    """Measure TTFT and overall throughput for an LLM call."""
    client = Anthropic()
    start = time.perf_counter()
    first_token_time = None
    tokens_received = 0

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()         # time to first token
            tokens_received += len(text.split())               # rough token count

    total_time = time.perf_counter() - start
    decode_time = total_time - (first_token_time - start)

    return {
        "ttft_ms": (first_token_time - start) * 1000,
        "total_latency_ms": total_time * 1000,
        "tpot_ms": decode_time * 1000 / max(tokens_received, 1),
        "throughput_tps": tokens_received / total_time
    }
```

🏭 **Production note:** Target TTFT < 300ms for interactive chat, TPOT < 50ms/token for readable streaming. For batch processing (offline inference), optimize for throughput, not latency.

---

## 3. Mixed Precision: FP16 and BF16

Running models in full fp32 (4 bytes/element) is wasteful. Half-precision inference is standard:

| Format | Bits | Range | Use Case |
|---|---|---|---|
| FP32 | 32 | ±3.4×10³⁸ | Training stability |
| FP16 | 16 | ±65,504 | Inference; careful with overflow |
| BF16 | 16 | ±3.4×10³⁸ | Preferred for LLMs (same range as FP32) |
| INT8 | 8 | -128 to 127 | Post-training quantization |
| INT4 | 4 | -8 to 7 | Aggressive quantization |

BF16 is preferred over FP16 for LLMs because it has the same exponent range as FP32 — activations with large values don't overflow. FP16's limited range requires loss scaling during training.

```python
import torch

# Check memory savings: fp32 vs bf16 for a 7B model
params_7b = 7e9
fp32_gb = params_7b * 4 / (1024**3)   # 4 bytes per parameter
bf16_gb = params_7b * 2 / (1024**3)   # 2 bytes per parameter
print(f"FP32: {fp32_gb:.1f} GB")      # 26.0 GB
print(f"BF16: {bf16_gb:.1f} GB")      # 13.0 GB

# Load model in bf16
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,        # half precision inference
    device_map="auto"
)

# Inference with autocast (ensures bf16 operations)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(input_ids=torch.randint(0, 32000, (1, 10)).cuda())
```

---

## 4. Quantization: INT8, GPTQ, AWQ

Quantization reduces weight precision to save memory and improve throughput. The key challenge: LLM activations contain large-magnitude outliers in specific channels. Naive INT8 quantization crushes the rest of the distribution to accommodate these outliers.

**LLM.int8()** ([Dettmers et al., 2022](https://arxiv.org/abs/2208.07339)): mixed-precision quantization that identifies outlier channels and keeps them in fp16, quantizing everything else to INT8:

```python
# pip install bitsandbytes
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)  # LLM.int8()
)
```

**GPTQ** ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)): layer-wise quantization using the inverse Hessian to compensate for quantization error. Achieves near-lossless INT4 compression. Offline calibration step required.

**AWQ (Activation-Aware Weight Quantization)** ([Lin et al., 2023](https://arxiv.org/abs/2306.00978)): identifies salient weight channels via activation statistics and applies per-channel scaling before INT4 quantization. Often outperforms GPTQ at the same bit width because it protects the channels that matter most.

| Method | Bits | Overhead | Quality |
|---|---|---|---|
| FP16 | 16 | Baseline | Baseline |
| LLM.int8() | 8 | Minimal | ~99% FP16 |
| GPTQ | 4 | Calibration required | ~97% FP16 |
| AWQ | 4 | Calibration required | ~98% FP16 |
| GGUF Q4_K_M | 4 | None (GGUF format) | ~97% FP16 |

---

## 5. GGUF and llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp) enables LLM inference on CPUs, Apple Silicon (Metal), and CUDA GPUs without Python or GPU dependencies. The GGUF format packages quantized weights with model metadata.

**Quantization levels in GGUF:**

| Level | Bits | Size (7B) | Quality |
|---|---|---|---|
| Q4_K_M | ~4.5 | 4.2 GB | Good — recommended |
| Q5_K_M | ~5.5 | 5.1 GB | Better |
| Q8_0 | 8 | 7.2 GB | Near-lossless |
| Q2_K | ~2.5 | 2.4 GB | Low — for edge |

```bash
# Convert HuggingFace model to GGUF and quantize
# python llama.cpp/convert_hf_to_gguf.py meta-llama/Llama-3.1-8B --outfile llama-3.1-8b.gguf
# ./llama.cpp/quantize llama-3.1-8b.gguf llama-3.1-8b-Q4_K_M.gguf Q4_K_M

# Run inference (server mode)
# ./llama.cpp/llama-server -m llama-3.1-8b-Q4_K_M.gguf --host 0.0.0.0 --port 8080
```

```python
# Python via llama-cpp-python
from llama_cpp import Llama

llm = Llama(
    model_path="llama-3.1-8b-Q4_K_M.gguf",
    n_ctx=4096,                            # context window
    n_gpu_layers=-1                         # -1 = offload all layers to GPU
)
response = llm("What is attention in transformers?", max_tokens=200, temperature=0.7)
print(response["choices"][0]["text"])
```

🏭 **Production note:** llama.cpp is the standard for local/edge inference. Q4_K_M is the sweet spot — 4x smaller than FP16 with minimal quality loss. For CPU-only servers, Q4_K_M at 8B params can serve ~15–20 tokens/sec on a modern server CPU.

---

## 6. Flash Attention

The standard attention implementation materializes the full (seq × seq) attention score matrix S = QK^T in HBM (GPU main memory). For a 4K sequence, this is a 4K×4K float matrix — 64M elements, 128MB in fp16 per layer per batch. This is the main memory bottleneck for long sequences.

**Flash Attention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) tiles the computation in SRAM (fast on-chip memory). Instead of writing the full attention matrix to HBM and reading it back, it:

1. Loads a tile of Q, K, V into SRAM
2. Computes the attention for that tile using the online softmax trick
3. Accumulates the result into the output
4. Never writes the full attention matrix to HBM

The result is the **exact same output** as standard attention but 2–4× faster and O(n) memory instead of O(n²). The online softmax trick maintains running max and sum to compute the exact softmax without seeing the full row at once.

```python
# Flash Attention 2 is built into PyTorch >= 2.0
import torch
import torch.nn.functional as F

# Standard attention (materializes full attention matrix)
def standard_attention(Q, K, V, causal=True):
    scale = Q.size(-1) ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale    # O(n^2) memory
    if causal:
        mask = torch.triu(torch.ones(Q.size(-2), K.size(-2), device=Q.device), 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), V)

# Flash Attention via PyTorch SDPA (uses Flash Attention kernel automatically)
def flash_attention(Q, K, V, causal=True):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)  # O(n) memory
```

**Flash Attention 2** ([Dao, 2023](https://arxiv.org/abs/2307.08691)) improved parallelism along the sequence dimension, achieving 50–73% MFU on A100.

---

## 7. vLLM and PagedAttention

[vLLM](https://docs.vllm.ai/) is the dominant open-source LLM serving framework for production. Its key innovation is **PagedAttention** ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)).

The problem with contiguous KV cache: traditional KV cache is allocated as one large contiguous block per request. Since maximum sequence length is unknown at request time, you must either pre-allocate the maximum (wasting memory) or risk running out (causing OOM). Memory fragmentation between requests further reduces effective GPU memory.

**PagedAttention** allocates KV cache in small fixed-size pages (blocks of 16 tokens), like virtual memory paging in an OS. Each request's KV cache is stored in non-contiguous pages, tracked via a page table. Benefits:
- No memory wasted on pre-allocation
- Copy-on-write for beam search (shared prefixes)
- Near-zero fragmentation overhead

```python
# pip install vllm
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,                    # maximum sequence length
    gpu_memory_utilization=0.9             # use 90% of GPU memory for KV pages
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)

prompts = [
    "Explain KV cache in one paragraph.",
    "What is PagedAttention?",
    "How does continuous batching work?"
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Output: {output.outputs[0].text[:100]}...")
```

vLLM also provides an OpenAI-compatible REST API, enabling drop-in replacement for OpenAI clients.

---

## 8. Continuous Batching

Static batching (the naive approach) waits for N requests to accumulate, runs them as a batch, and waits for all to finish before accepting new requests. If one request generates 1000 tokens and another generates 10, the fast request's GPU slot sits idle for 99% of the time.

**Continuous batching** (iteration-level scheduling): at each decode iteration, the scheduler can insert new requests into vacant slots and remove completed requests. The batch composition changes dynamically every token.

```
Static batching:
  Batch [Req1(1000), Req2(10), Req3(500)] → wait 1000 steps before ANY slot freed

Continuous batching:
  Step 10:  Req2 finishes → insert Req4 immediately
  Step 500: Req3 finishes → insert Req5 immediately
  Step 1000: Req1 finishes → insert Req6 immediately
```

GPU utilization increases from ~50% (static) to ~90%+ (continuous) at typical production load. All major serving frameworks (vLLM, TGI, SGLang) implement continuous batching.

---

## 9. Speculative Decoding

The bottleneck in LLM decoding is memory bandwidth — loading billions of weight parameters from HBM to compute a single output token. The GPU's compute units sit mostly idle.

**Speculative decoding** ([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192)) exploits this idle compute: a small "draft" model proposes k tokens speculatively; the large "target" model verifies all k in a single parallel forward pass (since verification is compute-bound, not memory-bound):

1. Draft model generates k tokens autoregressively (~10ms)
2. Target model runs one forward pass over all k+1 positions in parallel (~15ms vs 15×15=225ms)
3. Accept the first j ≤ k tokens that the target model agrees with
4. Reject from position j+1 onward, generate position j+1 with the target model

If most tokens are accepted, throughput improves 2–3× at the same output quality.

```python
def speculative_decode_step(draft_tokens: list[int], draft_probs: list[float],
                             target_probs: list[float]) -> tuple[list[int], int]:
    """Speculative decoding acceptance: accept tokens where target ≥ draft probability."""
    import random
    accepted = []
    for i, (dt, dp, tp) in enumerate(zip(draft_tokens, draft_probs, target_probs)):
        accept_prob = min(1.0, tp / (dp + 1e-9))              # acceptance probability
        if random.random() < accept_prob:
            accepted.append(dt)                                # accept this draft token
        else:
            return accepted, i                                  # reject from here
    return accepted, len(draft_tokens)                         # all accepted
```

**Medusa** adds multiple parallel draft heads to the base model itself — no separate draft model needed. **Eagle** uses a lightweight drafter trained with feature alignment to the target model.

---

## 10. Prompt Caching

When many requests share the same long prefix (system prompt, few-shot examples, document context), recomputing the KV cache for the prefix on every request is wasteful.

**Prompt caching** (prefix caching): compute the KV cache for the shared prefix once, store it, and reuse it across requests. The first request pays full prefill cost; subsequent requests skip the cached portion.

**Anthropic prompt caching** ([Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)):
- Cache hit: 10% of normal input token cost
- Cache TTL: 5 minutes (resettable by re-sending the cached prefix)
- Minimum cacheable length: 1024 tokens

```python
from anthropic import Anthropic

client = Anthropic()

# Long system prompt / document to cache
long_document = "..." * 500  # ~1000 tokens

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    system=[{
        "type": "text",
        "text": long_document,
        "cache_control": {"type": "ephemeral"}               # mark for caching
    }],
    messages=[{"role": "user", "content": "Summarize the key points."}]
)
# First call: full price. Subsequent calls within 5 min: 90% discount on cached tokens
print(f"Cache write tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

**vLLM radix attention:** automatically caches common prefixes across concurrent requests using a radix tree structure — no explicit API needed.

---

## 11. Tensor and Pipeline Parallelism

For models too large for a single GPU, model parallelism distributes the model across multiple GPUs.

**Tensor Parallelism (TP):** split individual weight matrices across GPUs. For attention:
- Q, K, V weight matrices split column-wise across tp_size GPUs
- Each GPU computes one shard of Q, K, V
- After output projection, all-reduce across GPUs to sum results

```
GPU 0: W_Q[:, 0:d/2], W_K[:, 0:d/2], W_V[:, 0:d/2]
GPU 1: W_Q[:, d/2:d], W_K[:, d/2:d], W_V[:, d/2:d]
```

TP requires fast intra-node NVLink — the all-reduce communication per layer is latency-sensitive. Typically limited to 4 or 8 GPUs (one node).

**Pipeline Parallelism (PP):** assign consecutive transformer layers to consecutive GPUs. A batch is split into micro-batches; while GPU 1 processes micro-batch 2 in layers 0–10, GPU 2 processes micro-batch 1 in layers 11–20. 

Pipeline bubble: the fraction of time GPUs are idle waiting for the previous stage. Minimized with more micro-batches (GPipe schedule) or 1F1B (one forward pass, one backward pass interleaved).

```python
# Hugging Face accelerate handles basic TP/PP automatically
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    from transformers import LlamaForCausalLM, LlamaConfig
    model = LlamaForCausalLM(LlamaConfig())

model = load_checkpoint_and_dispatch(
    model,
    checkpoint="meta-llama/Llama-3.1-70B",
    device_map="auto",                     # auto-assign layers to GPUs
    max_memory={0: "40GB", 1: "40GB"}     # per-GPU memory budget
)
```

---

## 12. Expert Parallelism for MoE

Mixture of Experts models (Mixtral, DeepSeek-V3) route each token to a small subset of expert FFN layers. With many experts (DeepSeek-V3 has 256), not all experts fit on one GPU.

**Expert Parallelism (EP):** place different experts on different GPUs. Each GPU processes the tokens routed to its experts. An all-to-all communication operation dispatches tokens to the right GPU and collects results.

The EP communication pattern is very different from TP's all-reduce: EP requires all-to-all (each GPU sends data to potentially all other GPUs), which is much more sensitive to network bandwidth. InfiniBand is typically required for efficient EP across nodes.

---

## 13. Constrained Decoding

Generating valid JSON, SQL, or structured formats by post-processing LLM output is fragile — the model may produce syntactically invalid output that breaks parsing. **Constrained decoding** enforces structure at the token level during generation.

**Outlines** ([github.com/outlines-dev/outlines](https://github.com/outlines-dev/outlines)) converts a JSON schema or regex into a finite state machine (FSM) and masks out tokens that would violate the constraint at each generation step:

```python
import outlines
import outlines.models as models

model = models.transformers("meta-llama/Llama-3.1-8B-Instruct")

# Enforce JSON schema at token level — guaranteed valid JSON output
schema = '{"type": "object", "properties": {"name": {"type": "string"}, "score": {"type": "number"}}, "required": ["name", "score"]}'
generator = outlines.generate.json(model, schema)
result = generator("Extract the person and their score: Alice scored 95 on the exam.")
print(result)  # Always valid JSON: {"name": "Alice", "score": 95.0}
```

**xgrammar** ([mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)) provides faster FSM-based constrained decoding with persistent state compilation — ~10× faster than Outlines for complex grammars.

---

## 14. Disaggregated Prefill and Decode

The prefill phase (processing the input prompt) is compute-bound — the GPU is doing dense matrix multiplications across all input positions simultaneously. The decode phase is memory-bandwidth-bound — loading weights and KV cache to generate one token at a time.

These two phases have fundamentally different hardware requirements. Running them together in the same GPU cluster forces a compromise.

**Disaggregated prefill** ([Splitwise, 2023](https://arxiv.org/abs/2311.18677)): run prefill on a separate pool of "prefill machines" (optimized for compute) and decode on "decode machines" (optimized for memory bandwidth). Transfer the computed KV cache over NVLink/InfiniBand.

This enables:
- Scaling prefill and decode capacity independently
- Using different hardware (e.g., H100 for prefill, cheaper A10 for decode)
- Better latency for long-prompt requests (dedicated prefill resources)

**Mooncake** (Kimi's production system) takes this further with a distributed KV cache store accessible by all serving nodes.

---

## 15. TensorRT-LLM and ONNX

**TensorRT-LLM** ([NVIDIA](https://github.com/NVIDIA/TensorRT-LLM)): NVIDIA's high-performance LLM inference library. Compiles model weights into a TensorRT engine with custom CUDA kernels:
- FP8 on H100 (highest throughput on Hopper GPUs)
- INT4 weight-only with AWQ
- In-flight batching (continuous batching)
- Custom attention kernels and quantized KV cache

Typically 2–3× faster than FP16 HuggingFace inference on NVIDIA GPUs. Requires a compilation step (model → TRT engine) that can take 10–30 minutes.

```bash
# Convert and serve with TRT-LLM
# trtllm-build --checkpoint_dir ./llama3-8b-fp8 \
#              --output_dir ./llama3-8b-trt-engine \
#              --gemm_plugin float16 \
#              --max_batch_size 32 \
#              --max_input_len 2048 \
#              --max_seq_len 4096
```

**ONNX export** enables cross-framework deployment (e.g., deploy a PyTorch model via ONNX Runtime on CPU). Less useful for large decoder-only LLMs (TRT-LLM is better for NVIDIA GPUs) but valuable for encoder models (BERT, embedding models) and edge/mobile.

---

## 16. LoRA Serving at Scale

A large deployment may have one base LLM but thousands of fine-tuned LoRA adapters (one per customer, one per language, one per task). Loading a new base model for each adapter is impractical.

**S-LoRA** ([Sheng et al., 2023](https://arxiv.org/abs/2311.03285)) serves thousands of LoRA adapters on a single base model:
- The base model weights stay fixed in GPU memory
- Each LoRA adapter is stored separately (small — typically 10–500MB)
- When a request arrives, the relevant adapter is loaded into memory dynamically
- Multiple adapters can run in the same batch using unified paging

**vLLM LoRA support:**

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,                      # enable LoRA support
    max_loras=10                           # max simultaneous LoRA adapters
)

# Different requests can use different LoRA adapters in the same batch
outputs = llm.generate(
    ["Translate to French: Hello world",
     "Write a haiku about Python"],
    SamplingParams(max_tokens=100),
    lora_request=[
        LoRARequest("french-adapter", 1, "./adapters/french"),
        LoRARequest("poetry-adapter", 2, "./adapters/poetry")
    ]
)
```

---

## 17. KV Cache Quantization and Compression

The KV cache grows with sequence length and is stored in fp16 by default. For long contexts, it can exceed model weight size.

**KV cache quantization:** store K and V in INT8 or FP8 instead of FP16. Halves KV cache memory with minimal quality loss on most tasks.

**Attention sinks (StreamingLLM)** ([Xiao et al., 2023](https://arxiv.org/abs/2307.03170)): LLMs consistently assign high attention weight to the first few tokens ("attention sinks") — even when they're not semantically relevant. StreamingLLM always keeps these sink tokens in the KV cache, then uses a sliding window for the rest, enabling infinite context length (new tokens evict old ones beyond the window).

**KV cache eviction policies:** for cache sizes beyond StreamingLLM, more sophisticated policies evict tokens based on attention score history, preferring to keep tokens with high cumulative attention.

```python
def compute_kv_memory_savings(
    seq_len: int, n_layers: int, n_kv_heads: int, d_head: int,
    batch_size: int
) -> dict:
    """Compare KV cache memory at different quantization levels."""
    elements = 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size
    return {
        "fp16_gb": elements * 2 / (1024**3),
        "int8_gb": elements * 1 / (1024**3),
        "int4_gb": elements * 0.5 / (1024**3)
    }

result = compute_kv_memory_savings(4096, 32, 8, 128, 16)
print("KV cache memory comparison:")
for precision, gb in result.items():
    print(f"  {precision}: {gb:.2f} GB")
```

**GQA as architectural compression** (Section 3 in the LLM blog): using fewer KV heads (GQA) reduces KV cache proportionally — LLaMA-3.1 70B uses 8 KV heads (vs 64 query heads), giving 8× smaller KV cache than full MHA.

---

## 18. Benchmarking and Profiling

Before optimizing inference, measure what's actually slow.

**Key metrics to profile:**
- **TTFT (ms)** at P50/P99 — prefill performance
- **TPOT (ms/token)** at P50/P99 — decode performance
- **GPU memory utilization** — headroom for larger batches
- **MFU (Model FLOPS Utilization)** — fraction of peak GPU FLOPs actually used
- **GPU bandwidth utilization** — fraction of peak HBM bandwidth used

**MFU calculation:**

```python
def compute_mfu(model_params: float, tokens_per_sec: float,
                gpu_tflops: float) -> float:
    """Model FLOPS Utilization: actual FLOPS / peak GPU FLOPS."""
    # FLOPs per token ≈ 6 × parameters (for decoder-only models, including matmuls)
    flops_per_token = 6 * model_params
    actual_tflops = tokens_per_sec * flops_per_token / 1e12
    return actual_tflops / gpu_tflops

# LLaMA-3.1-8B on A100 80GB
mfu = compute_mfu(
    model_params=8e9,
    tokens_per_sec=2000,                   # throughput tokens/sec
    gpu_tflops=312                         # A100 peak BF16 TFLOPs
)
print(f"MFU: {mfu:.1%}")                  # typical range: 30-60% for well-tuned systems
```

**Tools:**
- **llmperf** ([github.com/ray-project/llmperf](https://github.com/ray-project/llmperf)): standardized benchmarking of any OpenAI-compatible endpoint
- **GenAI-Perf** (NVIDIA): detailed profiling with percentile latencies, TTFT/TPOT breakdown
- **nvitop / nvidia-smi**: GPU utilization, memory, SM occupancy

---

## 19. References

### Attention and Architecture

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.](https://arxiv.org/abs/2205.14135)
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.](https://arxiv.org/abs/2307.08691)

### Quantization

- [Dettmers et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.](https://arxiv.org/abs/2208.07339)
- [Frantar et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.](https://arxiv.org/abs/2210.17323)
- [Lin et al. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.](https://arxiv.org/abs/2306.00978)

### Serving Infrastructure

- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention.](https://arxiv.org/abs/2309.06180)
- [Leviathan et al. (2022). Fast Inference from Transformers via Speculative Decoding.](https://arxiv.org/abs/2211.17192)
- [Sheng et al. (2023). S-LoRA: Serving Thousands of Concurrent LoRA Adapters.](https://arxiv.org/abs/2311.03285)
- [Splitwise (2023). Disaggregated Prefill and Decode for LLM Inference.](https://arxiv.org/abs/2311.18677)

### KV Cache

- [Xiao et al. (2023). Efficient Streaming Language Models with Attention Sinks (StreamingLLM).](https://arxiv.org/abs/2307.03170)

### Parallelism

- [Shoeybi et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.](https://arxiv.org/abs/1909.08053)
- [Huang et al. (2018). GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism.](https://arxiv.org/abs/1811.06965)

### Production Resources

- [vLLM Documentation](https://docs.vllm.ai/en/stable/)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llmperf Benchmarking](https://github.com/ray-project/llmperf)
