# Retrieval-Augmented Generation: From Basic Pipelines to Agentic RAG

> A production-first deep dive into RAG architecture, chunking, retrieval, reranking, evaluation, and advanced patterns — for ML Engineers and AI Engineers.

---

## Table of Contents

1. [Basic RAG Pipeline](#1-basic-rag-pipeline)
2. [Document Loading and Parsing](#2-document-loading-and-parsing)
3. [Chunking: Fixed-Size and Recursive](#3-chunking-fixed-size-and-recursive)
4. [Chunking: Semantic and Proposition Splitting](#4-chunking-semantic-and-proposition-splitting)
5. [Vector Stores: FAISS, Chroma, Qdrant, Weaviate](#5-vector-stores-faiss-chroma-qdrant-weaviate)
6. [Embedding Models for RAG](#6-embedding-models-for-rag)
7. [Hybrid Search: BM25 and Dense](#7-hybrid-search-bm25-and-dense)
8. [Reranking with Cross-Encoders](#8-reranking-with-cross-encoders)
9. [Contextual Compression](#9-contextual-compression)
10. [Parent-Child Retrieval](#10-parent-child-retrieval)
11. [HyDE: Hypothetical Document Embeddings](#11-hyde-hypothetical-document-embeddings)
12. [Query Expansion and Rewriting](#12-query-expansion-and-rewriting)
13. [RAG-Fusion](#13-rag-fusion)
14. [RAPTOR: Recursive Abstractive Processing](#14-raptor-recursive-abstractive-processing)
15. [Self-RAG and Corrective RAG](#15-self-rag-and-corrective-rag)
16. [GraphRAG](#16-graphrag)
17. [Multi-Modal RAG](#17-multi-modal-rag)
18. [Agentic RAG](#18-agentic-rag)
19. [Long-Context RAG vs Stuffing](#19-long-context-rag-vs-stuffing)
20. [Synthetic QA Generation and Evaluation](#20-synthetic-qa-generation-and-evaluation)
21. [RAG Evaluation: RAGAS and DeepEval](#21-rag-evaluation-ragas-and-deepeval)
22. [Prompt Engineering for RAG](#22-prompt-engineering-for-rag)
23. [Contextual Retrieval](#23-contextual-retrieval)
24. [Production RAG: Latency, Cost, and Caching](#24-production-rag-latency-cost-and-caching)
25. [Agentic Document Parsing](#25-agentic-document-parsing)
26. [References](#26-references)

---

## 1. Basic RAG Pipeline

Large language models hallucinate when they don't know something. They fill the gap with plausible-sounding text. Retrieval-Augmented Generation (RAG) was introduced by [Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) as a way to ground LLM answers in a retrieved document corpus, reducing hallucination and enabling knowledge updates without retraining.

The basic RAG pipeline has five steps:

```
1. Index:   Embed each document chunk → store in vector database
2. Retrieve: Embed query → find top-k similar chunks by cosine similarity
3. Augment: Prepend retrieved chunks to the prompt
4. Generate: LLM produces answer conditioned on retrieved context
5. Return:  Return answer (optionally with citations)
```

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from anthropic import Anthropic

def build_basic_rag(documents: list[str]) -> tuple:
    """Build a minimal RAG system: embed docs, store in FAISS."""
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")        # lightweight embedding model
    embeddings = model.encode(documents, normalize_embeddings=True) # L2-normalized for cosine
    index = faiss.IndexFlatIP(embeddings.shape[1])                 # inner product = cosine (normalized)
    index.add(embeddings.astype(np.float32))                       # add all doc embeddings
    return model, index, documents

def rag_query(query: str, model, index, documents: list[str],
              k: int = 3) -> str:
    """Retrieve top-k chunks and generate an answer."""
    q_emb = model.encode([query], normalize_embeddings=True)       # embed the query
    _, indices = index.search(q_emb.astype(np.float32), k)        # retrieve top-k
    context = "\n\n".join(documents[i] for i in indices[0])       # join retrieved chunks
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="Answer questions using only the provided context. If unsure, say so.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )
    return response.content[0].text
```

🎯 **Interview prep:** RAG beats fine-tuning when the knowledge domain changes frequently, when you need source citations, or when the corpus is too large to fit in a fine-tuning dataset. Fine-tuning beats RAG when you need to change the model's style, format, or reasoning behavior.

---

## 2. Document Loading and Parsing

Before you can embed documents, you need to parse them into clean text. Most production corpora mix file formats, and each format has its own parsing pitfalls.

**PDF parsing** is the hardest case. PDFs are layout-based, not semantic. Text extraction order can be wrong (multi-column layouts, footnotes, headers). Tables are typically not parsed as tables — they come out as mangled text.

- **PyMuPDF (fitz)** — fastest PDF text extractor; handles most standard PDFs.
- **pdfplumber** — better table extraction via character-level bounding boxes.
- **LlamaParse** — LLM-powered parser for complex PDFs with tables, charts, and formulas. Slow and expensive but significantly more accurate on structured documents.
- **Docling** — open-source layout-aware parser from IBM Research that converts PDFs to clean Markdown with table and figure detection.

```python
import fitz  # PyMuPDF

def extract_pdf_text(pdf_path: str) -> list[str]:
    """Extract text by page from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)                                     # open PDF
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")                              # extract plain text
        if text.strip():                                          # skip blank pages
            pages.append(f"[Page {page_num + 1}]\n{text.strip()}")
    return pages

def extract_markdown(md_path: str) -> list[str]:
    """Split Markdown by headers into logical sections."""
    with open(md_path) as f:
        content = f.read()
    sections = content.split("\n## ")                             # split on H2 headers
    return [s.strip() for s in sections if s.strip()]
```

🏭 **Production note:** For any production RAG system with PDFs, budget time for parser evaluation. Run 50 representative PDFs through your parser and manually inspect 10% of the output. The quality of parsing is the single biggest determinant of RAG answer quality on document-heavy corpora.

---

## 3. Chunking: Fixed-Size and Recursive

Chunking determines the retrieval unit. Too small and chunks lack context; too large and retrieved chunks dilute signal with irrelevant content.

**Fixed-size character splitting:** split on character count, with overlap to preserve context across chunk boundaries.

```python
def fixed_size_chunks(text: str, chunk_size: int = 500,
                      overlap: int = 50) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size                                  # chunk end position
        chunks.append(text[start:end])                           # add this chunk
        start = end - overlap                                     # slide forward with overlap
    return chunks
```

**RecursiveCharacterTextSplitter** (from LangChain) tries to split on natural boundaries in priority order: `\n\n` (paragraphs), `\n` (lines), ` ` (words), then characters. This preserves semantic units better than pure character splitting.

**Practical chunk size guidance:**

| Use Case | chunk_size | chunk_overlap |
|---|---|---|
| Dense technical docs | 256–512 chars | 50–100 |
| General web content | 512–1024 chars | 100–200 |
| Long narrative text | 1024–2048 chars | 200–400 |
| Code snippets | File-level or function-level | 0 |

🎯 **Interview prep:** chunk_size and chunk_overlap are the most frequently tuned hyperparameters in RAG. Small chunks = more precise retrieval but less context per chunk. Large chunks = more context but noisier retrieval. Start with 512 chars / 50 overlap and tune on your evaluation set.

---

## 4. Chunking: Semantic and Proposition Splitting

Fixed-size chunking ignores semantic boundaries. A 500-character chunk might cut a sentence in half, or combine two unrelated paragraphs.

**Semantic chunking** embeds individual sentences, then splits where the cosine similarity between consecutive sentences drops below a threshold — signaling a topic shift ([Chroma research](https://research.trychroma.com/evaluating-chunking)).

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text: str, threshold: float = 0.5) -> list[str]:
    """Split text at semantic boundaries detected by embedding cosine drift."""
    sentences = text.split(". ")                                  # rough sentence split
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embs = model.encode(sentences, normalize_embeddings=True)     # embed all sentences
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i-1], embs[i]))                  # cosine similarity
        if sim < threshold:                                       # topic drift detected
            chunks.append(" ".join(current))                      # flush current chunk
            current = [sentences[i]]                              # start new chunk
        else:
            current.append(sentences[i])                          # continue current chunk
    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Proposition chunking** takes semantic chunking further: convert each chunk into standalone atomic facts (propositions) using an LLM. Each proposition is a self-contained sentence that can be understood without the surrounding context. This maximizes retrieval precision — you retrieve exactly the facts relevant to the query, not surrounding filler.

Proposition chunking is expensive (one LLM call per chunk) but produces the highest retrieval precision on factoid questions.

---

## 5. Vector Stores: FAISS, Chroma, Qdrant, Weaviate

Choosing a vector store is a practical decision based on scale, infrastructure, and filtering needs.

| Store | Best For | Key Features |
|---|---|---|
| **FAISS** | Local, high-performance | In-memory, IVF/HNSW indexes, no metadata filtering |
| **Chroma** | Local development | Persistent SQLite, metadata filtering, easy setup |
| **Qdrant** | Production cloud/on-prem | Sparse+dense, payload filtering, named vectors |
| **Weaviate** | Multi-tenant production | Built-in BM25+vector hybrid, generative modules |

**FAISS index types:**
- `IndexFlatL2` / `IndexFlatIP` — exact search, no approximation; accurate but O(n) per query
- `IndexIVFFlat` — inverted file index; partition space into clusters, search only nearby clusters; fast but needs training
- `IndexHNSWFlat` — graph-based ANN; excellent recall/speed tradeoff; recommended for production

```python
import faiss
import numpy as np

def build_hnsw_index(embeddings: np.ndarray, M: int = 32) -> faiss.Index:
    """Build an HNSW index for fast approximate nearest neighbor search."""
    d = embeddings.shape[1]                                       # embedding dimension
    index = faiss.IndexHNSWFlat(d, M)                            # M = edges per node
    index.hnsw.efConstruction = 200                               # build quality (higher = slower build, better recall)
    index.hnsw.efSearch = 50                                      # search quality (higher = slower query, better recall)
    index.add(embeddings.astype(np.float32))                      # populate index
    return index

# Save and reload for production
def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)                                # serialize to disk

def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)                                 # deserialize from disk
```

🏭 **Production note:** For most teams starting out, Qdrant's Docker image (single node) or Chroma's persistent mode handles millions of vectors without infrastructure complexity. Move to managed cloud (Qdrant Cloud, Pinecone, Weaviate Cloud) when you need SLAs and horizontal scaling.

---

## 6. Embedding Models for RAG

The embedding model determines what "similarity" means in your vector space. A model trained on general web text may not understand medical or legal terminology.

**MTEB (Massive Text Embedding Benchmark)** ([Muennighoff et al., 2022](https://arxiv.org/abs/2210.07316)) benchmarks embedding models across retrieval, classification, clustering, and semantic similarity tasks. The retrieval sub-task (nDCG@10) is most relevant for RAG.

**Top retrieval models (as of mid-2025):**

| Model | Size | MTEB Retrieval | Notes |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | API | ~56 nDCG@10 | High quality, pay-per-use |
| `BAAI/bge-large-en-v1.5` | 335M | ~54 | Strong open-source baseline |
| `E5-mistral-7b-instruct` | 7B | ~57 | Best open-source, expensive |
| `BAAI/bge-small-en-v1.5` | 33M | ~51 | Fast, cheap, good for prototyping |
| `Cohere embed-english-v3.0` | API | ~55 | Supports int8 quantization |

**Domain adaptation:** if your corpus is technical (medical, legal, scientific), fine-tune a base model with domain-specific query-document pairs using contrastive loss (InfoNCE):

```python
import torch
import torch.nn.functional as F

def infonce_loss(query_emb: torch.Tensor, pos_emb: torch.Tensor,
                 neg_embs: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss for embedding fine-tuning."""
    # query_emb: (batch, dim); pos_emb: (batch, dim); neg_embs: (batch, n_neg, dim)
    pos_sim = F.cosine_similarity(query_emb, pos_emb)              # (batch,)
    neg_sim = F.cosine_similarity(                                  # (batch, n_neg)
        query_emb.unsqueeze(1).expand_as(neg_embs), neg_embs
    )
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)    # (batch, 1+n_neg)
    logits = logits / temperature                                   # scale by temperature
    labels = torch.zeros(len(query_emb), dtype=torch.long)        # positive is at index 0
    return F.cross_entropy(logits, labels)                         # cross-entropy loss
```

---

## 7. Hybrid Search: BM25 and Dense

Dense retrieval excels at semantic matching ("canine companions" matches "dogs"). BM25 excels at exact keyword matching — critical when the query contains specific product codes, names, or technical terms that embeddings might smear.

**BM25** (Best Match 25) is a TF-IDF variant with term saturation and document length normalization ([Robertson & Zaragoza, 2009](https://dl.acm.org/doi/10.1561/1500000019)):

```
BM25(q, d) = Σ_t IDF(t) · (TF(t,d) · (k1+1)) / (TF(t,d) + k1·(1-b+b·|d|/avgdl))
```

where k1=1.5 and b=0.75 are standard defaults.

**Reciprocal Rank Fusion (RRF)** merges ranked lists from BM25 and dense retrieval without needing score calibration ([Cormack et al., 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)):

```
RRF_score(doc) = Σ_ranker 1 / (k + rank(doc))
```

where k=60 is the standard constant that dampens the impact of very high ranks.

```python
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_rrf(query: str, documents: list[str],
               dense_scores: dict[int, float], k: int = 60) -> list[int]:
    """Combine BM25 and dense retrieval ranks with RRF."""
    # BM25 ranking
    tokenized = [doc.lower().split() for doc in documents]        # tokenize for BM25
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())          # BM25 scores
    bm25_ranks = np.argsort(-bm25_scores)                         # rank by score (descending)

    # Dense ranking (already provided as dict[idx → score])
    dense_ranks = sorted(dense_scores.keys(),
                         key=lambda i: -dense_scores[i])          # sort by dense score

    # RRF fusion
    rrf = {}
    for rank, doc_idx in enumerate(bm25_ranks):
        rrf[doc_idx] = rrf.get(doc_idx, 0) + 1 / (k + rank + 1) # BM25 contribution
    for rank, doc_idx in enumerate(dense_ranks):
        rrf[doc_idx] = rrf.get(doc_idx, 0) + 1 / (k + rank + 1) # dense contribution

    return sorted(rrf.keys(), key=lambda i: -rrf[i])              # final ranked list
```

🎯 **Interview prep:** "What's the difference between BM25 and embedding-based retrieval?" — BM25 is exact term matching with frequency weighting; embeddings capture semantic similarity. Hybrid systems using RRF consistently outperform either alone ([Pinecone hybrid search](https://www.pinecone.io/learn/hybrid-search-intro/)).

---

## 8. Reranking with Cross-Encoders

The retrieval stage uses bi-encoders (query and document embedded independently, similarity computed with dot product). This is fast but less accurate than cross-encoding.

A **cross-encoder** takes the query and document together as a single input and outputs a relevance score. This is dramatically more accurate — the model can attend to interactions between query and document tokens — but cannot be pre-computed, so it can only be applied to a small set of candidates.

The standard two-stage pipeline:
1. **Retrieve** top-20 with bi-encoder (fast, approximate)
2. **Rerank** top-20 with cross-encoder → return top-5

```python
from sentence_transformers import CrossEncoder

def rerank_with_cross_encoder(query: str, candidates: list[str],
                               top_k: int = 5) -> list[tuple[str, float]]:
    """Rerank candidate documents using a cross-encoder."""
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # fast cross-encoder
    pairs = [(query, doc) for doc in candidates]                   # (query, doc) pairs
    scores = model.predict(pairs)                                  # relevance scores
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1]) # sort by score
    return ranked[:top_k]                                          # return top-k
```

**Production options:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, 6-layer model; good for latency-sensitive systems
- `BAAI/bge-reranker-large` — higher quality, larger model
- **Cohere Rerank API** — managed service, very strong performance, no hosting required ([Cohere docs](https://docs.cohere.com/docs/reranking))

---

## 9. Contextual Compression

Even after reranking, retrieved chunks may contain paragraphs irrelevant to the specific query. Contextual compression filters each chunk down to only the relevant sentences before adding them to the prompt.

Two approaches:
- **LLM-based compression:** prompt the LLM to extract relevant sentences from the chunk given the query
- **Embedding-based compression:** embed each sentence; keep only sentences with cosine similarity to the query above a threshold

```python
def embedding_compress(query: str, chunk: str,
                        model, threshold: float = 0.4) -> str:
    """Keep only sentences in chunk that are semantically similar to the query."""
    sentences = [s.strip() for s in chunk.split(".") if s.strip()] # split into sentences
    if not sentences:
        return chunk
    query_emb = model.encode([query], normalize_embeddings=True)[0] # query embedding
    sent_embs = model.encode(sentences, normalize_embeddings=True)   # sentence embeddings
    scores = sent_embs @ query_emb                                   # cosine similarity
    relevant = [s for s, sc in zip(sentences, scores) if sc >= threshold]
    return ". ".join(relevant) if relevant else chunk               # fallback to full chunk
```

---

## 10. Parent-Child Retrieval

Precision and recall pull in opposite directions for chunk sizing. Small chunks are precise — they match queries closely. Large chunks are rich — they provide more context for generation.

**Parent-child retrieval** resolves this tension: index small "child" chunks for retrieval precision, but return the larger "parent" chunk (or document section) as the context for generation.

```python
def build_parent_child_store(document: str,
                              parent_size: int = 2000,
                              child_size: int = 400) -> dict:
    """Build a parent-child chunk store."""
    # Split into parent chunks
    parents = [document[i:i+parent_size]
               for i in range(0, len(document), parent_size)]

    # For each parent, create smaller child chunks and link back to parent
    child_to_parent = {}
    all_children = []
    for parent_idx, parent in enumerate(parents):
        children = [parent[j:j+child_size]
                    for j in range(0, len(parent), child_size)]
        for child in children:
            child_to_parent[child] = parent_idx  # child → parent index
            all_children.append(child)

    return {"parents": parents, "children": all_children,
            "child_to_parent": child_to_parent}

def retrieve_with_parent(query: str, store: dict, model, index,
                          k: int = 3) -> list[str]:
    """Retrieve child chunks, then return their parent chunks."""
    q_emb = model.encode([query], normalize_embeddings=True)
    import faiss, numpy as np
    _, child_indices = index.search(q_emb.astype(np.float32), k)
    children = store["children"]
    parents = store["parents"]
    parent_indices = set()
    for ci in child_indices[0]:
        parent_idx = store["child_to_parent"].get(children[ci])
        if parent_idx is not None:
            parent_indices.add(parent_idx)                       # deduplicate parent chunks
    return [parents[pi] for pi in parent_indices]
```

---

## 11. HyDE: Hypothetical Document Embeddings

Standard retrieval embeds the query and finds similar documents. But queries are short and terse while documents are long and descriptive — they live in different regions of the embedding space.

**HyDE** ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)) addresses this mismatch: use an LLM to generate a hypothetical document that *answers* the query, then embed the hypothetical document and retrieve against the real corpus. The generated document is much closer in style to real documents, improving retrieval recall.

```python
from anthropic import Anthropic

def hyde_retrieve(query: str, model, index, documents: list[str],
                  k: int = 3) -> list[str]:
    """HyDE: generate hypothetical answer, embed it, retrieve real docs."""
    client = Anthropic()
    # Step 1: generate a hypothetical answer
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",                        # cheap model for generation
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Write a detailed passage that answers: {query}"
        }]
    )
    hypothetical_doc = response.content[0].text                   # the generated document
    # Step 2: embed hypothetical doc and retrieve
    import numpy as np
    hyp_emb = model.encode([hypothetical_doc], normalize_embeddings=True)
    _, indices = index.search(hyp_emb.astype(np.float32), k)
    return [documents[i] for i in indices[0]]
```

HyDE works well for abstract or conceptual queries ("What causes market volatility?") and underperforms on specific factoid queries where the hypothetical answer may drift from the real answer.

---

## 12. Query Expansion and Rewriting

A single query may not capture all aspects of what the user needs. Query expansion generates alternative phrasings and retrieves for each.

**Multi-query expansion:** generate N paraphrases of the query, retrieve for each, merge and deduplicate results. More retrieval calls but higher recall for ambiguous queries.

**Step-back prompting** ([Zheng et al., 2023](https://arxiv.org/abs/2310.06117)): before answering a specific question, first answer a more abstract, general version. "What is Einstein's nationality?" → step-back: "What is the personal background of Albert Einstein?" — the abstract query retrieves broader context that helps answer the specific one.

**Sub-query decomposition:** for multi-hop questions ("Which company was founded first, Apple or Microsoft?"), decompose into atomic sub-queries, answer each, then synthesize.

```python
def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Generate N query paraphrases using an LLM."""
    client = Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (f"Generate {n} different phrasings of this query, "
                        f"one per line:\n{query}")
        }]
    )
    lines = response.content[0].text.strip().split("\n")          # split into lines
    variants = [q.strip() for q in lines if q.strip()][:n]       # keep up to n variants
    return [query] + variants                                      # include original
```

---

## 13. RAG-Fusion

RAG-Fusion ([Raudaschl, 2024](https://arxiv.org/abs/2402.03367)) combines multi-query expansion with RRF fusion:

1. Generate N query variants from the original query
2. Retrieve top-k documents for each variant independently
3. Fuse all result lists with RRF
4. Pass the highest-scoring documents to the LLM

The key insight is that RRF promotes documents that appear in multiple query-variant result sets — these are likely more robustly relevant than documents that appear in only one.

```python
def rag_fusion(query: str, model, index, documents: list[str],
               n_variants: int = 4, top_k: int = 5) -> list[str]:
    """RAG-Fusion: multi-query retrieval fused with RRF."""
    import numpy as np
    variants = generate_query_variants(query, n=n_variants)       # generate query variants
    all_rankings = []
    for variant in variants:
        q_emb = model.encode([variant], normalize_embeddings=True)
        scores, indices = index.search(q_emb.astype(np.float32), 20) # retrieve 20 per variant
        all_rankings.append(indices[0].tolist())                   # store ranked doc indices

    # RRF fusion
    rrf_scores: dict[int, float] = {}
    for ranked_list in all_rankings:
        for rank, doc_idx in enumerate(ranked_list):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (60 + rank + 1)

    top_indices = sorted(rrf_scores.keys(),
                         key=lambda i: -rrf_scores[i])[:top_k]   # top docs by RRF score
    return [documents[i] for i in top_indices]
```

---

## 14. RAPTOR: Recursive Abstractive Processing

RAPTOR ([Sarthi et al., 2024](https://arxiv.org/abs/2401.18059)) builds a hierarchical tree of summaries over the document corpus:

1. Embed all chunks
2. Cluster chunks with Gaussian Mixture Models
3. Summarize each cluster with an LLM → these become higher-level "nodes"
4. Embed and cluster the summary nodes
5. Repeat until a single root summary remains

At query time, retrieve from all tree levels. Short factoid queries match leaf chunks; broad thematic queries match high-level summaries.

RAPTOR is computationally expensive to build (O(n) LLM calls for n chunks, then recursively) but significantly improves performance on questions that require synthesizing information across multiple documents.

---

## 15. Self-RAG and Corrective RAG

Standard RAG always retrieves, even for questions the LLM could answer from parametric memory. Both Self-RAG and CRAG add adaptive retrieval logic.

**Self-RAG** ([Asai et al., 2023](https://arxiv.org/abs/2310.11511)) fine-tunes an LLM to generate special reflection tokens:
- `[Retrieve]` / `[No Retrieve]` — should we retrieve for this query?
- `[Relevant]` / `[Irrelevant]` — is the retrieved passage relevant?
- `[Fully Supported]` / `[Partially Supported]` / `[No Support]` — does the generation follow from the retrieved passage?

This makes retrieval selective and quality-controlled, at the cost of requiring fine-tuning.

**CRAG (Corrective RAG)** ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)) uses a lightweight retrieval evaluator to score retrieved documents. If all documents are scored "irrelevant" or "ambiguous," CRAG falls back to web search for fresher or broader sources. Documents are then refined (decomposed, noise-filtered, recomposed) before generation.

```
CRAG flow:
  Query → Retrieve top-k
           ↓
     Score relevance of each doc
           ↓
  If any doc is Correct → proceed to generation with filtered content
  If all are Ambiguous → combine internal + web search
  If all are Incorrect → web search only
           ↓
     Knowledge refinement (extract key segments)
           ↓
     Generate with refined context
```

---

## 16. GraphRAG

Standard RAG retrieves document chunks in isolation. Relationships between entities — who founded what, which law applies to which jurisdiction, which drug interacts with which compound — are scattered across chunks and hard to aggregate via vector similarity.

**GraphRAG** ([Edge et al., 2024](https://arxiv.org/abs/2404.16130)) from Microsoft Research:

1. Extract entities and relationships from the corpus with an LLM
2. Build a knowledge graph of (entity, relation, entity) triples
3. Run community detection (Leiden algorithm) on the graph
4. Summarize each community into a "community report"
5. At query time: map query to relevant communities; retrieve community reports + relevant entity triples

GraphRAG dramatically improves performance on "global" queries that require aggregating information across the entire corpus ("What are the main themes in these financial reports?"). Standard RAG with chunk retrieval cannot answer these coherently.

The cost: GraphRAG requires expensive LLM extraction at index time (many LLM calls per document). Microsoft's open-source implementation is at [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag).

---

## 17. Multi-Modal RAG

Many production corpora contain images, charts, diagrams, and tables that carry critical information not captured in the surrounding text.

**Approaches to multi-modal RAG:**

1. **CLIP embeddings:** embed both images and text in a shared embedding space using CLIP ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)). Retrieve images relevant to a text query.

2. **Caption-based indexing:** generate captions for each image using a vision-language model, embed the captions, and retrieve as text.

3. **GPT-4V / Claude for grounded generation:** retrieve relevant pages/images, pass them directly to a multimodal LLM that can read charts and diagrams.

```python
from anthropic import Anthropic
import base64

def multimodal_rag_query(query: str, image_paths: list[str]) -> str:
    """Query a multimodal RAG system with retrieved images."""
    client = Anthropic()
    content = [{"type": "text", "text": f"Question: {query}\n\nAnalyze the following retrieved images:"}]
    for path in image_paths:
        with open(path, "rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_data}
        })
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text
```

---

## 18. Agentic RAG

In standard RAG, the retrieval strategy is fixed: embed query, retrieve top-k, generate. Agentic RAG gives the LLM control over the retrieval process itself.

The LLM can:
- Decide whether to retrieve at all (or answer from memory)
- Choose which retrieval tool to use (vector search, keyword search, structured query, web search)
- Issue follow-up queries if the initial results are insufficient
- Synthesize across multiple retrieval rounds

```python
from anthropic import Anthropic

tools = [
    {
        "name": "search_documents",
        "description": "Search the internal document corpus for relevant information",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for current information not in the document corpus",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
]

def agentic_rag(user_question: str, retriever) -> str:
    """Agentic RAG: LLM controls retrieval strategy via tool calls."""
    client = Anthropic()
    messages = [{"role": "user", "content": user_question}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024, tools=tools,
            messages=messages
        )
        if response.stop_reason == "end_turn":                    # LLM finished
            return response.content[0].text
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "search_documents":
                    result = retriever(block.input["query"])      # hit vector store
                else:
                    result = f"[Web search result for: {block.input['query']}]"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

---

## 19. Long-Context RAG vs Stuffing

Modern LLMs have 128K–1M token context windows. A natural question: why not just stuff the entire document corpus into the context?

**The lost-in-the-middle problem** ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)): LLMs perform worse on information placed in the middle of long contexts. Performance is best at the beginning and end of the context window. For a 100-document corpus stuffed into one prompt, most documents are in the "lost" zone.

**When to stuff vs retrieve:**

| Situation | Recommendation |
|---|---|
| Small corpus (< 20 docs), latency insensitive | Stuff all docs |
| Medium corpus, single domain | RAG with hybrid search |
| Large corpus (> 1M tokens), diverse | RAG with HNSW index |
| Question requires cross-doc synthesis | GraphRAG or RAPTOR |
| Real-time information needed | Agentic RAG with web search |

**Hybrid approach:** retrieve top-20 chunks, stuff all 20 into context. Put the most relevant chunks at the beginning and end of the context (not the middle) to work with the attention bias.

---

## 20. Synthetic QA Generation and Evaluation

Before shipping a RAG system, you need a golden evaluation set. Manual annotation is expensive. Synthetic QA generation uses an LLM to create (question, answer, context) triples from your documents.

```python
from anthropic import Anthropic

def generate_qa_pairs(document_chunk: str, n: int = 3) -> list[dict]:
    """Generate synthetic QA pairs from a document chunk."""
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} question-answer pairs from this document chunk. "
                f"Questions should be specific and answerable from the text only. "
                f"Format as JSON list: [{{\"question\": ..., \"answer\": ...}}]\n\n"
                f"Document:\n{document_chunk}"
            )
        }]
    )
    import json
    text = response.content[0].text
    start = text.find("[")                                        # find JSON list
    end = text.rfind("]") + 1
    pairs = json.loads(text[start:end])                          # parse QA pairs
    return [{"question": p["question"], "answer": p["answer"],
             "context": document_chunk} for p in pairs]           # add context
```

**Quality of synthetic QA:** validate the generated pairs manually before using them as golden set. Common issues: questions that are too easy (keyword matching suffices), questions that are unanswerable from the chunk, or answers that are too long.

---

## 21. RAG Evaluation: RAGAS and DeepEval

**RAGAS** ([Es et al., 2023](https://arxiv.org/abs/2309.15217)) measures four dimensions without needing ground-truth answers:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Does the answer make claims supported by the retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question asked? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |
| **Context Recall** | Are all ground-truth facts present in retrieved chunks? |

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# Evaluation dataset
data = {
    "question": ["What is RAG?", "What is HNSW?"],
    "answer": ["RAG is retrieval-augmented generation...", "HNSW is a graph-based ANN index..."],
    "contexts": [["RAG combines retrieval with LLM generation..."],
                 ["HNSW uses hierarchical navigable small world graphs..."]],
    "ground_truth": ["RAG stands for retrieval-augmented generation",
                     "HNSW is an approximate nearest neighbor algorithm"]
}
dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
```

**DeepEval** provides test-case-based evaluation that integrates with CI/CD pipelines ([Confident AI](https://docs.confident-ai.com/)). Write RAG quality assertions like unit tests: hallucination score < 0.1, answer correctness > 0.8.

---

## 22. Prompt Engineering for RAG

The system prompt for a RAG application sets the rules for how the LLM uses retrieved context.

**Core principles:**
1. **Grounding instruction:** "Answer only using the provided context. If the context doesn't contain the answer, say you don't know."
2. **Citation format:** "After every claim, cite the source in [brackets] using the Source ID provided with each context chunk."
3. **Uncertainty handling:** "If the context is ambiguous or contradictory, acknowledge the ambiguity."
4. **No-answer handling:** "If no relevant context was retrieved, say: 'I don't have information about this in my knowledge base.'"

```python
SYSTEM_PROMPT = """You are a precise, citation-focused assistant. You answer questions using only the provided context documents.

Rules:
1. Base every claim on the provided context. Do not use prior knowledge.
2. Cite each claim using [Source N] notation matching the source IDs below.
3. If the context does not contain the answer, respond: "I don't have this information in my knowledge base."
4. If context is ambiguous, say so explicitly before giving your best interpretation.
5. Keep answers concise — 2-4 sentences per point."""

def format_context_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks with source IDs for citation."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Source {i+1}] {chunk.get('source', 'Unknown')}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
```

---

## 23. Contextual Retrieval

Anthropic's **contextual retrieval** ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval)) addresses a fundamental problem: chunks lose context when extracted from their parent document. A chunk saying "The company was founded in 1985" doesn't say which company.

The fix: before embedding each chunk, prepend a context summary generated by an LLM explaining what the chunk is about in the context of the full document.

```
Chunk (before):
  "Revenue grew 15% year-over-year."

Contextual chunk (after):
  "This passage is from Acme Corp's Q3 2024 earnings report discussing
   financial performance. Revenue grew 15% year-over-year."
```

Anthropic found this approach reduced retrieval failures by 49% when combined with BM25 hybrid search. The main cost is the LLM call to generate context for each chunk (mitigated by using prompt caching — the document is the cached prefix).

```python
def add_chunk_context(document: str, chunk: str) -> str:
    """Prepend situating context to a chunk using an LLM."""
    client = Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system="Give a brief context sentence for the following chunk within its document.",
        messages=[{
            "role": "user",
            "content": f"<document>{document[:2000]}</document>\n\n<chunk>{chunk}</chunk>"
        }]
    )
    context_sentence = response.content[0].text.strip()           # the context sentence
    return f"{context_sentence}\n\n{chunk}"                       # prepend to chunk
```

---

## 24. Production RAG: Latency, Cost, and Caching

A production RAG endpoint must answer in < 2 seconds end-to-end. The budget:

| Step | Target Latency |
|---|---|
| Query embedding | 20–50ms |
| Vector search | 5–20ms |
| Reranking (optional) | 100–200ms |
| LLM generation | 500–1500ms |

**Async retrieval:** if you use multiple retrieval strategies (dense + BM25 + web), fire them in parallel with `asyncio.gather`. Don't wait for one to finish before starting the next.

**Prompt caching:** for documents that appear repeatedly in context (FAQ pages, policy documents), use prompt caching to avoid re-processing their tokens on every request. Anthropic's prompt caching ([Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)) charges 10% of normal input token cost for cache hits.

**Semantic caching:** cache full RAG responses for common queries. **GPTCache** ([Zilliz](https://github.com/zilliztech/GPTCache)) embeds the incoming query and checks if a semantically similar query was answered recently. If similarity > threshold, return the cached answer. Effective when users ask similar questions repeatedly (help desks, FAQ bots).

```python
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    """Simple semantic cache using cosine similarity on query embeddings."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 threshold: float = 0.95):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold                                 # similarity threshold for cache hit
        self.cache: list[tuple[np.ndarray, str]] = []             # (query_emb, answer) pairs

    def get(self, query: str) -> str | None:
        """Return cached answer if a similar query was answered before."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        for cached_emb, answer in self.cache:
            sim = float(np.dot(q_emb, cached_emb))               # cosine similarity
            if sim >= self.threshold:
                return answer                                      # cache hit
        return None                                               # cache miss

    def set(self, query: str, answer: str):
        """Cache a query-answer pair."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        self.cache.append((q_emb, answer))                        # store in memory cache
```

---

## 25. Agentic Document Parsing

For complex documents with embedded charts, multi-column tables, and mixed layouts, rule-based parsers fail. Vision models can read documents as images.

**Docling** ([IBM Research](https://github.com/DS4SD/docling)) — open-source pipeline that combines layout detection, OCR, and table structure recognition to convert PDFs to clean Markdown. Handles multi-column layouts and table cell boundaries correctly without LLM calls.

**LlamaParse** ([LlamaIndex](https://github.com/run-llama/llama_parse)) — managed service that uses LLMs and vision models for high-fidelity parsing of complex documents. Returns structured representations of tables, figures, and formulas.

**Vision model parsing** — send PDF pages as images to Claude or GPT-4V and ask for structured extraction. Best for documents where every detail matters (financial statements, legal contracts, scientific papers with formulas).

```python
import fitz  # PyMuPDF
import base64
from anthropic import Anthropic

def parse_pdf_with_vision(pdf_path: str, page_range: range) -> list[str]:
    """Extract text from PDF pages using Claude vision model."""
    doc = fitz.open(pdf_path)
    client = Anthropic()
    results = []
    for page_num in page_range:
        page = doc[page_num]
        # Render page as PNG image
        mat = fitz.Matrix(2.0, 2.0)                              # 2x scale for quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": img_b64}},
                    {"type": "text",
                     "text": "Extract all text from this page. For tables, use Markdown table format. Preserve structure."}
                ]
            }]
        )
        results.append(f"[Page {page_num+1}]\n{response.content[0].text}")
    return results
```

🎯 **Interview prep:** "How do you handle PDFs with complex tables?" — the answer is a decision tree: simple PDFs → PyMuPDF; tables with clear boundaries → pdfplumber; complex layout → Docling; critical high-value documents → vision model parsing with Claude/GPT-4V.

---

## 26. References

### Foundational RAG

- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.](https://arxiv.org/abs/2005.11401)
- [Gao et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). *ACL*.](https://arxiv.org/abs/2212.10496)

### Retrieval and Ranking

- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark.](https://arxiv.org/abs/2210.07316)
- [Cormack et al. (2009). Reciprocal Rank Fusion. *SIGIR*.](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.](https://dl.acm.org/doi/10.1561/1500000019)
- [SBERT Cross-Encoder Documentation](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)

### Advanced RAG Architectures

- [Sarthi et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.](https://arxiv.org/abs/2401.18059)
- [Asai et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique.](https://arxiv.org/abs/2310.11511)
- [Yan et al. (2024). Corrective Retrieval Augmented Generation (CRAG).](https://arxiv.org/abs/2401.15884)
- [Edge et al. (2024). From Local to Global: A Graph RAG Approach (Microsoft).](https://arxiv.org/abs/2404.16130)
- [Raudaschl (2024). RAG-Fusion.](https://arxiv.org/abs/2402.03367)

### Query Rewriting

- [Zheng et al. (2023). Take a Step Back: Evoking Reasoning via Abstraction.](https://arxiv.org/abs/2310.06117)
- [Ma et al. (2023). Query Rewriting for Retrieval-Augmented Large Language Models.](https://arxiv.org/abs/2305.14283)

### Evaluation

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation.](https://arxiv.org/abs/2309.15217)
- [RAGAS Documentation](https://docs.ragas.io/en/stable/)
- [DeepEval / Confident AI Documentation](https://docs.confident-ai.com/)

### Chunking and Retrieval

- [Chroma Research: Evaluating Chunking Strategies](https://research.trychroma.com/evaluating-chunking)
- [Pinecone Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Pinecone Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)

### Long Context

- [Liu et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.](https://arxiv.org/abs/2307.03172)

### Production

- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic: Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [GPTCache (Zilliz)](https://github.com/zilliztech/GPTCache)
- [Docling (IBM Research)](https://github.com/DS4SD/docling)
- [LlamaParse (LlamaIndex)](https://github.com/run-llama/llama_parse)
