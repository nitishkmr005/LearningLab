# 09 — LLM Inferencing

Exhaustive learning path for deploying and optimizing LLMs: quantization, serving, batching, and hardware efficiency.

---

## 01 — Autoregressive Decoding & KV Cache
Token-by-token generation; how KV cache avoids recomputing all attention; memory footprint per token.
- https://jalammar.github.io/illustrated-gpt2/
- https://medium.com/@joaolages/kv-caching-explained-276520203249

## 02 — Latency Metrics: TTFT, TPOT, Throughput
Time-to-first-token (prefill dominated); time-per-output-token (decode dominated); tokens/sec; P99.
- https://www.anyscale.com/blog/llm-performance-benchmarks

## 03 — Mixed Precision: FP16 / BF16
Half-precision inference; BF16 vs FP16 numerical range; torch.autocast; memory savings with no quality loss.
- https://pytorch.org/docs/stable/amp.html

## 04 — Quantization Basics: INT8 (LLM.int8)
Absmax / zero-point quantization; outlier-aware INT8 (bitsandbytes); dynamic vs static quantization.
- https://arxiv.org/abs/2208.07339
- https://huggingface.co/docs/transformers/quantization/bitsandbytes

## 05 — GPTQ (Weight-Only INT4)
Layerwise quantization using Hessian; INT4 weights + FP16 activations; AutoGPTQ library.
- https://arxiv.org/abs/2210.17323
- https://github.com/AutoGPTQ/AutoGPTQ

## 06 — AWQ (Activation-Aware Weight Quantization)
Protect salient channels via activation-aware scaling; often better than GPTQ at 4-bit; llm-awq.
- https://arxiv.org/abs/2306.00978
- https://github.com/mit-han-lab/llm-awq

## 07 — GGUF & llama.cpp
Local CPU/Metal/CUDA inference; Q4_K_M, Q5_K_M, Q8_0 quant levels; server mode; Python bindings.
- https://github.com/ggerganov/llama.cpp
- https://huggingface.co/docs/hub/gguf

## 08 — Flash Attention
IO-aware tiling; fused kernel for attention + softmax; no O(n²) memory materialization; FA2/FA3.
- https://arxiv.org/abs/2205.14135
- https://arxiv.org/abs/2307.08691 (FlashAttention-2)

## 09 — vLLM & PagedAttention
Non-contiguous KV cache blocks; copy-on-write for beam search; OpenAI-compatible serving; high throughput.
- https://arxiv.org/abs/2309.06180
- https://docs.vllm.ai/en/stable/

## 10 — Continuous Batching
Slot new requests into running batches at decode step boundary; vs static batching; GPU utilization gain.
- https://www.anyscale.com/blog/continuous-batching-llm-inference

## 11 — Speculative Decoding
Draft model proposes k tokens; target model verifies all in one forward pass; 2–3× speedup.
- https://arxiv.org/abs/2211.17192
- https://huggingface.co/blog/assisted-generation

## 12 — Medusa & Eagle Decoding
Multiple draft heads on base model (Medusa); Eagle uses a separate drafter with feature alignment.
- https://arxiv.org/abs/2401.10774 (Eagle)
- https://github.com/FasterDecoding/Medusa

## 13 — Prompt Caching
Reuse KV states for shared prefixes; Anthropic cache_control; vLLM radix attention; cost reduction.
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- https://blog.vllm.ai/2024/03/13/prefix-caching.html

## 14 — Tensor Parallelism
Shard attention heads and FFN across GPUs; Megatron-LM style; all-reduce communication; tp_size.
- https://arxiv.org/abs/1909.08053
- https://huggingface.co/docs/transformers/parallelism

## 15 — Pipeline Parallelism
Split layers across GPUs; micro-batching to hide bubble; GPipe vs 1F1B schedule.
- https://arxiv.org/abs/1811.06965

## 16 — Expert Parallelism (MoE)
Route tokens to experts on different GPUs; dispatch + combine; DeepSeek-V3 EP group.
- https://arxiv.org/abs/2412.19437

## 17 — Constrained / Structured Decoding
Token-level grammar (FSM / regex); JSON schema enforcement; Outlines; xgrammar; no retry needed.
- https://github.com/outlines-dev/outlines
- https://github.com/mlc-ai/xgrammar

## 18 — Disaggregated Prefill & Decode
Separate prefill nodes (compute-bound) from decode nodes (memory-bound); Splitwise; Mooncake.
- https://arxiv.org/abs/2311.18677

## 19 — TensorRT-LLM & ONNX Export
NVIDIA TRT-LLM for FP8 + INT4 engine compilation; ONNX export for cross-framework serving.
- https://github.com/NVIDIA/TensorRT-LLM
- https://onnxruntime.ai/docs/performance/model-optimizations/float16.html

## 20 — LoRA Serving (Multiple Adapters)
Serve one base + N LoRA adapters; dynamic loading; S-LoRA; Punica; minimal memory overhead.
- https://arxiv.org/abs/2311.03285 (S-LoRA)
- https://docs.vllm.ai/en/stable/models/lora.html

## 21 — MLC-LLM: Mobile & Edge Inference
Apache TVM compilation; run Llama on iOS/Android/WebGPU; INT4 group quantization; chat app.
- https://llm.mlc.ai/
- https://arxiv.org/abs/2406.02539

## 22 — DeepSpeed Inference
Kernel injection; tensor parallelism via DeepSpeed; ZeRO-Inference for CPU offload; int8 kernel.
- https://www.deepspeed.ai/tutorials/inference-tutorial/

## 23 — LLM Benchmarking & Profiling
llmperf, GenAI-Perf; measure TTFT/TPOT at P50/P99; GPU utilization; model FLOPS utilization (MFU).
- https://github.com/ray-project/llmperf
- https://github.com/NVIDIA/GenerativeAIExamples
