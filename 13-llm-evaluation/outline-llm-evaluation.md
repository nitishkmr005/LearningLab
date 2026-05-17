# 12 — LLM Evaluation

Exhaustive learning path for evaluating large language models: benchmarks, metrics, human evaluation, and automated pipelines.

---

## 01 — Why LLM Evaluation is Hard
Distribution shift from training; open-ended outputs; no single ground truth; Goodhart's law; benchmark contamination; leaderboard gaming.
- https://huggingface.co/blog/evaluating-llm-bias
- https://arxiv.org/abs/2307.09009

## 02 — Evaluation Taxonomy
Intrinsic (perplexity, likelihood) vs extrinsic (task performance); automated vs human; reference-based vs reference-free; offline vs online.
- https://arxiv.org/abs/2307.03109

## 03 — Perplexity & Log-Likelihood
Bits-per-character, bits-per-byte; perplexity as a fluency proxy; limitations (tokenizer-dependence); stride evaluation on long docs.
- https://huggingface.co/docs/transformers/perplexity

## 04 — Academic Benchmarks
MMLU, HellaSwag, ARC, WinoGrande, GSM8K, HumanEval, MATH, BIG-Bench Hard; prompt format sensitivity; few-shot vs zero-shot.
- https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- https://github.com/EleutherAI/lm-evaluation-harness

## 05 — Text Generation Metrics
BLEU, ROUGE-L, METEOR; BERTScore; BLEURT; MoverScore; when each metric is appropriate; their blind spots.
- https://huggingface.co/spaces/evaluate-metric/bertscore
- https://arxiv.org/abs/1904.09675

## 06 — Factuality & Hallucination Detection
TruthfulQA; FactScore; entity-level hallucination; SelfCheckGPT; retrieval-augmented verification; consistency checks across samples.
- https://arxiv.org/abs/2307.11988
- https://arxiv.org/abs/2303.08896

## 07 — Instruction Following & Alignment Metrics
IFEval; prompt adherence; constraint satisfaction; AlpacaEval; MT-Bench; chatbot arena ELO ratings.
- https://arxiv.org/abs/2311.07911
- https://chat.lmsys.org/

## 08 — LLM-as-a-Judge
GPT-4 / Claude as evaluator; pairwise vs pointwise scoring; position bias; verbosity bias; calibration; self-evaluation pitfalls.
- https://arxiv.org/abs/2306.05685
- https://arxiv.org/abs/2307.09009

## 09 — Human Evaluation Protocols
Annotation guidelines; inter-annotator agreement (Cohen's κ, Fleiss' κ, Krippendorff's α); preference studies; A/B testing; crowdsourcing via Mechanical Turk.
- https://aclanthology.org/2021.findings-acl.141/

## 10 — Safety & Bias Evaluation
BBQ bias benchmark; ToxiGen; WinoBias; red-teaming; jailbreak success rate; refusal rate; representation parity.
- https://arxiv.org/abs/2110.08193
- https://arxiv.org/abs/2211.09527

## 11 — Code & Reasoning Evaluation
HumanEval, MBPP, LiveCodeBench; execution-based pass@k; chain-of-thought faithfulness; math proof verification; tool-use accuracy.
- https://github.com/openai/human-eval
- https://arxiv.org/abs/2108.07732

## 12 — RAG & Retrieval Evaluation
RAGAS (faithfulness, answer relevancy, context precision/recall); ARES; retriever hit@k; MRR; end-to-end vs component-level eval.
- https://docs.ragas.io/en/latest/
- https://arxiv.org/abs/2309.15217

## 13 — Evaluation Pipelines & Frameworks
EleutherAI lm-evaluation-harness; HuggingFace evaluate library; Promptfoo; DeepEval; HELM; running evals at scale; CI integration.
- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/confident-ai/deepeval

## 14 — Continuous & Production Evaluation
Shadow deployment; online A/B testing; implicit feedback signals (thumbs, retention); drift detection; evaluation datasets curation; version-controlled eval sets.
- https://eugeneyan.com/writing/llm-patterns/

## 15 — Building a Custom Eval Suite
Defining task-specific rubrics; creating golden datasets; bootstrapping from production logs; preventing eval set leakage; versioning and reproducibility.
- https://www.anthropic.com/research/evaluating-ai-systems
