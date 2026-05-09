# 05 — Embedding Models

Exhaustive learning path from classic word vectors to modern multimodal and fine-tuned embeddings.

---

## 01 — Word2Vec: Skip-gram
Predict surrounding words from center; negative sampling; train from scratch; analogy tasks (king-man+woman).
- https://arxiv.org/abs/1301.3781
- https://jalammar.github.io/illustrated-word2vec/

## 02 — Word2Vec: CBOW
Predict center from context; faster training; compare CBOW vs Skip-gram on analogy benchmarks.
- https://code.google.com/archive/p/word2vec/

## 03 — GloVe (Global Vectors)
Co-occurrence matrix factorization; combine global statistics with local window; load pretrained 100d/300d vectors.
- https://nlp.stanford.edu/projects/glove/
- https://arxiv.org/abs/1405.0312

## 04 — FastText (Subword Embeddings)
Character n-gram embeddings; OOV handling; morphology-aware; fasttext Python bindings.
- https://fasttext.cc/docs/en/unsupervised-tutorial.html
- https://arxiv.org/abs/1607.04606

## 05 — ELMo (Contextual Word Embeddings)
Bi-LSTM language model; same word gets different vectors per context; layer weighting for downstream tasks.
- https://arxiv.org/abs/1802.05365

## 06 — BERT Embeddings
[CLS] token vs mean-pool; contextual representations; off-the-shelf with HuggingFace; layer selection.
- https://arxiv.org/abs/1810.04805
- https://huggingface.co/docs/transformers/model_doc/bert

## 07 — Sentence Transformers (SBERT)
Siamese network with mean pooling; MultipleNegativesRankingLoss; semantic search; all-MiniLM, all-mpnet.
- https://arxiv.org/abs/1908.10084
- https://www.sbert.net/

## 08 — Bi-Encoder vs Cross-Encoder
Bi-encoder: encode independently → fast retrieval. Cross-encoder: encode together → accurate reranking.
- https://www.sbert.net/examples/applications/cross-encoder/README.html

## 09 — Dense Passage Retrieval (DPR)
Dual-encoder for open-domain QA; question encoder + passage encoder; hard negative training.
- https://arxiv.org/abs/2004.04906

## 10 — SimCSE: Contrastive Sentence Embeddings
Dropout as data augmentation; in-batch negatives; unsupervised + NLI-supervised variants.
- https://arxiv.org/abs/2104.08821

## 11 — Modern Embeddings: E5, BGE, GTE
Instruction-aware (E5: "query: …" / "passage: …"); weakly supervised training; MTEB top performers.
- https://arxiv.org/abs/2212.03533 (E5)
- https://arxiv.org/abs/2309.07597 (BGE)

## 12 — Matryoshka Representation Learning (MRL)
Single model produces valid embeddings at 1536→256→64 dims; trade storage/cost for quality.
- https://arxiv.org/abs/2205.13147
- https://huggingface.co/blog/matryoshka

## 13 — Late Interaction: ColBERT
Token-level embeddings stored per document; MaxSim scoring; 100× cheaper than cross-encoder at scale.
- https://arxiv.org/abs/2004.12832
- https://github.com/stanford-futuredata/ColBERT

## 14 — Sparse Embeddings: SPLADE
Learned sparse vectors in vocabulary space; combine keyword recall + neural semantics; inverted index.
- https://arxiv.org/abs/2109.10086

## 15 — Hybrid Dense + Sparse Search
SPLADE/BM25 + dense vectors; reciprocal rank fusion (RRF); Qdrant sparse-dense; Weaviate hybrid.
- https://qdrant.tech/articles/sparse-vectors/
- https://www.pinecone.io/learn/hybrid-search-intro/

## 16 — Fine-Tuning Embedding Models
MultipleNegativesRankingLoss; CosineSimilarityLoss; triplet loss; hard negatives with mining; SBERT fine-tune.
- https://www.sbert.net/docs/training/overview.html

## 17 — CLIP: Image-Text Multimodal Embeddings
Align image (ViT) and text (Transformer) in shared space; zero-shot classification; image search by text.
- https://arxiv.org/abs/2103.00020
- https://github.com/openai/CLIP

## 18 — ImageBind (Meta)
6-modality alignment (image, text, audio, depth, IMU, thermal) in one space; emergent cross-modal retrieval.
- https://arxiv.org/abs/2305.05665

## 19 — Embedding Quantization (Binary / INT8)
Binary embeddings (32× size reduction); Hamming distance; INT8 with re-scoring; minimal quality loss.
- https://huggingface.co/blog/embedding-quantization

## 20 — Evaluating Embeddings: MTEB
Benchmark over retrieval, clustering, classification, reranking, STS; how to read leaderboard scores.
- https://huggingface.co/spaces/mteb/leaderboard
- https://arxiv.org/abs/2210.07316

## 21 — Embedding Visualization
UMAP / t-SNE for embedding space exploration; cluster analysis; Nomic Atlas and TF Projector.
- https://projector.tensorflow.org/
- https://atlas.nomic.ai/

## 22 — Approximate Nearest Neighbor (ANN) Indexes
HNSW, IVF-PQ, Flat; FAISS, hnswlib, ScaNN; recall-latency trade-offs; choosing index type.
- https://faiss.ai/
- https://github.com/nmslib/hnswlib
