# 06 — RAG (Retrieval Augmented Generation)

Exhaustive learning path from basic pipelines to production-grade, multi-modal, agentic RAG.

---

## 01 — Basic RAG Pipeline
Embed corpus → store in vector DB → retrieve top-k → stuff into prompt → generate. Minimal end-to-end.
- https://python.langchain.com/docs/tutorials/rag/
- https://docs.llamaindex.ai/en/stable/getting_started/concepts/

## 02 — Document Loading & Parsing
PDF (PyMuPDF), HTML, Markdown, CSV loaders; OCR for scanned docs; table extraction; LlamaParse.
- https://python.langchain.com/docs/how_to/#document-loaders
- https://github.com/run-llama/llama_parse

## 03 — Chunking: Fixed-Size & Recursive
Character splitter; RecursiveCharacterTextSplitter; chunk_size and chunk_overlap trade-offs.
- https://www.pinecone.io/learn/chunking-strategies/

## 04 — Chunking: Semantic & Proposition Splitting
Embed sentences, split on semantic drift; proposition chunking for atomic facts; benchmark vs fixed-size.
- https://research.trychroma.com/evaluating-chunking

## 05 — Vector Store: FAISS
In-memory index; flat (exact) vs IVF vs HNSW; cosine vs L2; save/load; LangChain FAISS wrapper.
- https://faiss.ai/index.html
- https://www.pinecone.io/learn/faiss-tutorial/

## 06 — Vector Store: Chroma
Persistent local store; metadata filtering; multiple collections; embedding function plug-in.
- https://docs.trychroma.com/

## 07 — Vector Store: Qdrant
Collections, payloads, named vectors; filtered ANN search; sparse+dense; local Docker or cloud.
- https://qdrant.tech/documentation/quick-start/

## 08 — Vector Store: Weaviate
Schema-based with classes; multi-tenancy; BM25+vector hybrid built-in; generative search module.
- https://weaviate.io/developers/weaviate/quickstart

## 09 — Embedding Models for RAG
Dense models (E5, BGE, OpenAI ada-002); choosing by domain; MTEB retrieval score; latency vs quality.
- https://huggingface.co/spaces/mteb/leaderboard

## 10 — Hybrid Search: BM25 + Dense
BM25 for keyword recall; dense for semantics; reciprocal rank fusion (RRF) to merge ranked lists.
- https://www.pinecone.io/learn/hybrid-search-intro/
- https://arxiv.org/abs/2009.01513

## 11 — Reranking with Cross-Encoders
Retrieve top-20 with bi-encoder → rerank to top-5 with cross-encoder; Cohere Rerank, BGE-Reranker.
- https://www.sbert.net/examples/applications/cross-encoder/README.html
- https://docs.cohere.com/docs/reranking

## 12 — Contextual Compression
LLM-based or embedding-based chunk filtering; return only relevant sentences, not full chunks.
- https://python.langchain.com/docs/how_to/contextual_compression/

## 13 — Parent-Child (Multi-Vector) Retrieval
Index small child chunks for precision; retrieve larger parent chunk for context window richness.
- https://python.langchain.com/docs/how_to/parent_document_retriever/

## 14 — HyDE (Hypothetical Document Embeddings)
Generate hypothetical answer → embed it → retrieve against real corpus; improves abstract query recall.
- https://arxiv.org/abs/2212.10496

## 15 — Query Expansion & Rewriting
Multi-query paraphrases; step-back prompting; decompose multi-hop questions into sub-queries.
- https://arxiv.org/abs/2305.14283

## 16 — RAG-Fusion
Generate N query variants → retrieve per variant → fuse with RRF; increases recall.
- https://arxiv.org/abs/2402.03367

## 17 — RAPTOR (Recursive Abstractive Processing)
Cluster + summarize chunks recursively to build a tree; retrieve at multiple abstraction levels.
- https://arxiv.org/abs/2401.18059

## 18 — Self-RAG
LLM decides when to retrieve; critiques relevance (isREL) and support (isSUP); selective retrieval.
- https://arxiv.org/abs/2310.11511

## 19 — Corrective RAG (CRAG)
Score retrieved doc quality; fall back to web search if docs are poor; knowledge refinement step.
- https://arxiv.org/abs/2401.15884

## 20 — GraphRAG (Microsoft)
Extract entities + relations → knowledge graph; community detection; community summary retrieval.
- https://arxiv.org/abs/2404.16130
- https://github.com/microsoft/graphrag

## 21 — Multi-Modal RAG
CLIP embeddings for image retrieval; GPT-4V/LLaVA for image-grounded answers; mixed corpus indexing.
- https://blog.llamaindex.ai/multimodal-rag-pipeline-with-llamaindex-1e8a6f45a39c

## 22 — Agentic RAG
LLM decides retrieval strategy; iterative search loops; LangGraph + RAG; tool-calling retriever.
- https://arxiv.org/abs/2303.17651
- https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/

## 23 — Long-Context RAG vs Stuffing
When to stuff everything (Gemini 1M token) vs retrieve; lost-in-the-middle problem; hybrid approach.
- https://arxiv.org/abs/2307.03172

## 24 — Generating Synthetic QA for Evaluation
Use LLM to create question-answer-context triples from documents; build golden eval set.
- https://docs.ragas.io/en/stable/getstarted/testset_generation/

## 25 — RAG Evaluation: RAGAS
Faithfulness, answer relevancy, context precision, context recall; no-reference metrics; LLM-as-judge.
- https://docs.ragas.io/en/stable/getstarted/evaluation/
- https://arxiv.org/abs/2309.15217

## 26 — RAG Evaluation: DeepEval
Test cases in code; hallucination, answer correctness, BERTScore metrics; CI/CD integration.
- https://docs.confident-ai.com/

## 27 — Prompt Engineering for RAG
Grounding system prompt; citation instructions; "I don't know" handling; structured output format.
- https://www.anthropic.com/research/building-effective-agents

## 28 — Production RAG: Latency, Cost & Caching
Async retrieval; prompt caching for repeated prefixes; semantic caching (GPTCache); monitoring.
- https://github.com/zilliztech/GPTCache
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
