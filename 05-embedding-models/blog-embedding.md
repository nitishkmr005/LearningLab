# How Embedding Models Are Trained: From Word2Vec to State-of-the-Art

*A complete guide to loss functions, dataset preparation, training, evaluation, and the MTEB benchmark.*

---

## 1. The Problem: Meaning Lives in High-Dimensional Space

Language is slippery. "The bank was steep" and "I withdrew cash from the bank" use the same word but mean entirely different things. A search engine that treats text as bags of keywords will fail you the moment the user phrases their query differently from how the document was written.

The fundamental goal of embedding models is to map text — words, sentences, or entire documents — into a dense vector space where **semantic similarity corresponds to geometric proximity**. Sentences that mean the same thing should land near each other; sentences with opposite or unrelated meanings should be far apart.

Getting there took a decade of research, a series of breakthrough loss functions, and the development of large-scale benchmark suites. This blog walks through the entire story: where embeddings came from, how they are trained today, what makes a loss function good or bad, how to evaluate what you've built, and how to pick the right pre-trained model for your use case.

---

## 2. A Brief History: From Counting to Contextual Understanding

### 2.1 The Counting Era (pre-2013)

The earliest text representations were built on counting. **Bag of Words (BoW)** represented a document as a vector of word counts, ignoring word order entirely. **TF-IDF** improved on this by down-weighting words that appear everywhere (*the*, *is*) and up-weighting words that are rare and therefore informative.

These approaches worked for keyword matching but were fundamentally limited: they had no concept of synonymy (laptop ≈ notebook), no understanding of context, and produced enormous sparse vectors (one dimension per vocabulary word).

### 2.2 Word2Vec: Learning from Co-occurrence (2013)

Mikolov et al.'s **Word2Vec** ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)) was the first great leap. The key insight was the **distributional hypothesis**: words that appear in similar contexts have similar meanings. Word2Vec trained a shallow neural network to either:

- **Skip-gram**: predict surrounding context words given a center word, or
- **CBOW** (Continuous Bag of Words): predict the center word from surrounding context.

The hidden layer weights became the embedding vectors. After training on billions of tokens, these 300-dimensional vectors encoded extraordinary structure — the famous analogy `king − man + woman ≈ queen` fell out of the geometry naturally.

But Word2Vec had a fatal flaw: every word had **one fixed vector**, regardless of context. "Bank" in "river bank" and "bank account" got the same embedding.

### 2.3 GloVe and FastText: Filling the Gaps (2014–2017)

**GloVe** ([Pennington et al., 2014](https://aclanthology.org/D14-1162/)) took a different route — instead of local context windows, it factored the global word co-occurrence matrix, combining the strengths of matrix factorization (global statistics) with Word2Vec (local context).

**FastText** ([Bojanowski et al., 2017](https://arxiv.org/abs/1607.04606)) addressed a different problem: unknown words (OOV). By decomposing each word into character n-grams and summing their embeddings, FastText could construct a reasonable vector for any word, even misspelled ones. This made it especially powerful for morphologically rich languages.

Both still produced **static embeddings** — the same vector for a word regardless of where it appeared.

### 2.4 ELMo: Context Finally Enters (2018)

**ELMo** ([Peters et al., 2018](https://arxiv.org/abs/1802.05365)) was the first major shift toward contextual representations. A two-layer bidirectional LSTM language model produced representations that changed based on surrounding text. The word "bank" now got a different vector in a financial sentence versus a geographical one.

ELMo's embeddings were used by adding them on top of existing model inputs — they were a feature extraction layer, not a fine-tuning target.

### 2.5 BERT: The Foundation Model (2018)

**BERT** ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)) changed everything. Trained on two tasks — **Masked Language Modeling** (predict randomly masked tokens) and **Next Sentence Prediction** — on a 3.3B word corpus, BERT produced deep bidirectional contextual representations that became the backbone of virtually every NLP benchmark.

But BERT had a serious problem for similarity tasks.

---

## 3. The Sentence Embedding Problem and the SBERT Breakthrough

### 3.1 Why BERT's [CLS] Token Fails

The natural instinct when using BERT for semantic similarity is to take the `[CLS]` token's representation and compare two sentences. But this turns out to produce embeddings that perform **worse than averaging GloVe vectors** on semantic similarity benchmarks. The `[CLS]` token was trained for classification tasks — it was never optimized to encode sentence-level semantics.

The bigger problem is computational. To find the most similar pair among 10,000 sentences with BERT, you need to feed every possible pair as a single input (`[CLS] sentence_A [SEP] sentence_B [SEP]`) through the network. That's **50 million inference passes**, which takes roughly **65 hours** on a modern GPU.

### 3.2 SBERT: Siamese Networks to the Rescue (2019)

**Sentence-BERT** ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)) solved both problems elegantly. The key innovation was using a **Siamese network architecture**: two identical BERT encoders (sharing weights) processing sentences independently, with a pooling layer (mean pooling over all token embeddings) producing a fixed-size sentence vector.

```
Sentence A ──► BERT ──► Mean Pool ──► embedding_a ─┐
                                                    ├──► similarity / loss
Sentence B ──► BERT ──► Mean Pool ──► embedding_b ─┘
(shared weights)
```

With SBERT, you encode each sentence **once**. Comparing 10,000 sentences drops from 65 hours to **5 seconds** — using the same BERT-level quality but with pre-computed vectors and cosine similarity lookups.

This architectural shift made semantic search, clustering, and retrieval practical at scale. The rest of this blog is about *how* to train a model like this well.

---

## 4. What Training an Embedding Model Actually Means

Training an embedding model means learning a function `f(text) → ℝᵈ` such that the geometry of the output space reflects semantic meaning. You do this by:

1. **Preparing pairs or triplets** of related/unrelated texts
2. **Defining a loss function** that pushes semantically similar texts closer and dissimilar texts apart
3. **Fine-tuning a pre-trained transformer** (BERT, RoBERTa, etc.) with this loss

The quality of your loss function is the single most important factor in the quality of your embeddings. The field has gone through several generations of increasingly powerful losses.

---

## 5. Dataset Preparation

The shape of your training data is dictated by the loss function you choose. There are three fundamental formats:

### 5.1 Scored Pairs `(text_a, text_b, score)`

Used for regression-style objectives. Each pair has a continuous similarity score (e.g., 0–1 from human annotation). The most common sources are STS (Semantic Textual Similarity) datasets:

| Dataset | HuggingFace Link | Description |
|---|---|---|
| STSbenchmark | [`mteb/stsbenchmark-sts`](https://huggingface.co/datasets/mteb/stsbenchmark-sts) | 8,628 sentence pairs rated 0–5 by crowdworkers; primary STS eval dataset |
| STS12 | [`mteb/sts12-sts`](https://huggingface.co/datasets/mteb/sts12-sts) | SemEval 2012 — MSRpar, MSRvid, SMTeuroparl subsets |
| STS13 | [`mteb/sts13-sts`](https://huggingface.co/datasets/mteb/sts13-sts) | SemEval 2013 — headlines, OnWN, FNWN |
| STS14 | [`mteb/sts14-sts`](https://huggingface.co/datasets/mteb/sts14-sts) | SemEval 2014 — tweets, news headlines, image captions |
| STS15 | [`mteb/sts15-sts`](https://huggingface.co/datasets/mteb/sts15-sts) | SemEval 2015 — answers-forums, answers-students |
| STS16 | [`mteb/sts16-sts`](https://huggingface.co/datasets/mteb/sts16-sts) | SemEval 2016 — news headlines, plagiarism, Q&A |

Annotators rated sentence similarity on a 0–5 scale (normalized to 0–1 for training).

```python
from datasets import Dataset

data = [
    {"text_a": "A man is playing guitar.", "text_b": "A person strums a guitar.", "score": 0.92},
    {"text_a": "A cat sits on a mat.",    "text_b": "The stock market crashed.", "score": 0.02},
]
dataset = Dataset.from_list(data)
```

### 5.2 Positive Pairs `(anchor, positive)`

Used for contrastive objectives. You only provide related pairs; negatives are constructed automatically from other items in the batch (called **in-batch negatives**). Sources include:

| Category | Dataset | HuggingFace Link | Pair Type |
|---|---|---|---|
| NLI | SNLI | [`stanfordnlp/snli`](https://huggingface.co/datasets/stanfordnlp/snli) | entailment = positive; contradiction = hard negative |
| NLI | MultiNLI | [`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli) | same as SNLI; covers 10 genres |
| NLI | AllNLI (combined) | [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) | SNLI + MNLI merged; ready-to-use triplets |
| Q&A | MS MARCO | [`microsoft/ms_marco`](https://huggingface.co/datasets/microsoft/ms_marco) | (query, relevant passage) from Bing search logs |
| Q&A | Natural Questions | [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions) | (question, Wikipedia passage) |
| Q&A | PAQ | [`sentence-transformers/paq`](https://huggingface.co/datasets/sentence-transformers/paq) | 65M (question, answer) pairs auto-generated from Wikipedia |
| Paraphrase | Quora Question Pairs | [`quora`](https://huggingface.co/datasets/quora) | 400K question pairs labeled duplicate / not-duplicate |
| Paraphrase | PAWS | [`google-research-datasets/paws`](https://huggingface.co/datasets/google-research-datasets/paws) | Adversarial paraphrases — high lexical overlap, different meaning |
| Weak supervision | Wikipedia sentences | [`sentence-transformers/wikipedia-en-sentences`](https://huggingface.co/datasets/sentence-transformers/wikipedia-en-sentences) | consecutive sentences as loose positives |
| Weak supervision | Reddit title–body | [`sentence-transformers/reddit-title-body`](https://huggingface.co/datasets/sentence-transformers/reddit-title-body) | post title as query; body as positive passage |

```python
data = [
    {"anchor": "What is the capital of France?", "positive": "Paris is the capital of France."},
    {"anchor": "How does photosynthesis work?",  "positive": "Plants convert sunlight to energy via chlorophyll."},
]
```

### 5.3 Triplets `(anchor, positive, negative)`

Used for triplet loss. The negative is explicitly provided, giving the model a harder signal. The challenge is choosing *good* negatives.

| Dataset | HuggingFace Link | Description |
|---|---|---|
| AllNLI Triplets | [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) | NLI-derived triplets: premise = anchor, entailment = positive, contradiction = hard negative |
| Quora Triplets | [`embedding-data/QQP_triplets`](https://huggingface.co/datasets/embedding-data/QQP_triplets) | Quora duplicate questions formatted as (query, duplicate, non-duplicate) |
| MS MARCO Triplets | [`sentence-transformers/msmarco-bm25`](https://huggingface.co/datasets/sentence-transformers/msmarco-bm25) | (query, relevant passage, BM25 hard negative) from Bing search logs |
| NLI + Hard Negatives | [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) | `triplet-all` split includes hard negatives mined with cross-encoder scoring |
| GooAQ Triplets | [`sentence-transformers/gooaq-hard-negatives`](https://huggingface.co/datasets/sentence-transformers/gooaq-hard-negatives) | Google auto-complete (question, answer) pairs with BM25 hard negatives |

```python
from datasets import load_dataset

# AllNLI triplets: ready-to-use anchor/positive/negative columns
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
# Columns: anchor, positive, negative

data = [
    {
        "anchor":   "What is the capital of France?",
        "positive": "Paris is the capital of France.",
        "negative": "The French Revolution began in 1789.",
    },
]
```

### 5.4 Hard Negative Mining

Random negatives are easy — the model quickly learns to separate them. **Hard negatives** are samples that are *semantically close but not the correct answer*, and they force the model to make finer distinctions.

Common mining strategies:

- **BM25 mining**: Retrieve top BM25 results for a query. These are lexically similar but may not be the true positive — powerful hard negatives.
- **Cross-encoder mining**: Use a strong cross-encoder reranker to score candidates; take high-scoring but non-positive pairs.
- **Model-guided mining** (GPL — Generative Pseudo Labeling, [Wang et al., 2021](https://arxiv.org/abs/2112.09118)): Generate queries for documents with a T5 model, then mine negatives with BM25, then score with a cross-encoder to get soft labels.
- **Augmented SBERT** ([Reimers & Gurevych, 2021](https://arxiv.org/abs/2010.08240)): When labeled data is scarce (1K–3K pairs), train a cross-encoder on your gold labels → use BM25 or semantic search to generate new candidate pairs → have the cross-encoder score them as *silver labels* → train the bi-encoder on gold + silver combined. Works in two modes: **in-domain** (small labeled set + same-domain unlabeled corpus) and **cross-domain** (source-domain labels transferred to an unlabeled target domain via the cross-encoder). Particularly effective when you cannot afford to annotate thousands of pairs but have access to a related labeled dataset.

```python
# BM25-based hard negative mining (conceptual)
from rank_bm25 import BM25Okapi

corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(corpus)

for query, true_positive_idx in query_pairs:
    top_bm25 = bm25.get_top_n(query.split(), documents, n=10)
    # Filter out true positive, take the rest as hard negatives
    hard_negatives = [d for d in top_bm25 if d != documents[true_positive_idx]][:3]
```

### 5.5 Synthetic Data Generation

When you have no labeled pairs, you can generate them. A common recipe:

1. Chunk your target corpus into passages
2. Use a T5/GPT model to generate questions for each passage
3. These (question, passage) pairs are your training data
4. Optionally run cross-encoder scoring to filter weak pairs

This is the **GPL** ([Generative Pseudo-Labeling](https://arxiv.org/abs/2112.09118)) approach and is highly effective for domain adaptation.

### 5.6 Multi-Dataset Training

One of the most powerful features in sentence-transformers v3 is training on **multiple datasets simultaneously with a different loss function per dataset** ([Tom Aarsen, HF blog 2024](https://huggingface.co/blog/train-sentence-transformers)). This removes the need to force all your data into one format — each loss does what it is best at.

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss, CoSENTLoss
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from datasets import load_dataset

model = SentenceTransformer("microsoft/mpnet-base")

# Dataset 1: anchor-positive pairs → MultipleNegativesRankingLoss
nli_dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train[:50000]")

# Dataset 2: scored sentence pairs → CoSENTLoss
sts_dataset = load_dataset("mteb/stsbenchmark-sts", split="train")
sts_dataset = sts_dataset.select_columns(["sentence1", "sentence2", "score"])

mnrl_loss   = MultipleNegativesRankingLoss(model)
cosent_loss = CoSENTLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="output/multi-dataset-model",
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset={"nli": nli_dataset, "sts": sts_dataset},
    loss={"nli": mnrl_loss, "sts": cosent_loss},
)
trainer.train()
```

**Sampling strategies:**
- `PROPORTIONAL` (default) — samples proportional to dataset size; larger datasets dominate
- `ROUND_ROBIN` — alternates equally between datasets; prevents small datasets from being ignored

**Critical format rule:** Column *order* matters, not just names. For scored-pair losses expecting `(text1, text2, score)`, columns must appear in that order. Columns named `label` or `score` are treated as targets; all others as inputs. Always call `dataset.remove_columns(["id", "split", ...])` to strip non-input metadata columns before training.

---

## 6. Loss Functions: The Evolution

### 6.1 Generation 1 — Softmax Classification Loss

The original SBERT paper ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)) used a **3-class softmax** over NLI data. Given embeddings `u` and `v`, the model concatenated `[u, v, |u-v|]` and passed it through a linear layer with softmax to predict Entailment / Neutral / Contradiction.

**Formula:**
```
input  = [u; v; |u - v|]   (3×d dimensional vector)
logits = W · input
loss   = CrossEntropy(logits, label)
```

**Why it's limited:** The loss never directly optimizes cosine similarity between sentence vectors. You're training a classifier on top, and the geometric structure of the embedding space is an indirect consequence.

### 6.2 Generation 2 — Cosine Similarity Loss (Regression)

A simple MSE loss between predicted cosine similarity and the gold score from STS annotations:

**Formula:**
```
ŷ = cosine_similarity(u, v)
loss = MSE(ŷ, y_gold)
```

This *does* directly optimize the cosine space. But MSE loss has a known weakness: it treats all prediction errors equally, regardless of whether correcting them would change the ranking of pairs. For retrieval, ranking is what matters.

### 6.3 Generation 3 — Contrastive Loss

Inspired by metric learning, contrastive loss explicitly pulls similar pairs together and pushes dissimilar pairs apart:

**Formula:**
```
loss = y · d(u,v)² + (1 - y) · max(margin - d(u,v), 0)²
```
Where:
- `y = 1` for similar pairs, `y = 0` for dissimilar pairs
- `d(u,v)` is Euclidean distance
- `margin` is a hyperparameter (minimum separation for negatives)

**Limitation:** Requires explicit negative pairs (no in-batch negatives). Also, the margin hyperparameter is sensitive — set it too low and the model stops learning; too high and training is noisy.

### 6.4 Generation 4 — Triplet Loss

Instead of working on pairs, triplet loss considers three samples at once: an **anchor**, a **positive** (same class), and a **negative** (different class):

**Formula:**
```
loss = max(d(a, p) - d(a, n) + margin, 0)
```

Where:
- `d` is a distance function (cosine or Euclidean)
- `margin` ensures the anchor is closer to the positive than the negative by at least `m`

**Advantage over contrastive loss:** Triplet loss doesn't force all positives to converge to a single point — it only requires that the anchor is *closer* to the positive than to the negative. This preserves intra-class variance (two correct answers can still differ from each other).

**Online triplet mining:** Rather than pre-mining triplets, compute all valid triplets within a batch on-the-fly. Semi-hard mining selects negatives that violate the margin but not by too much — the sweet spot for gradient signal.

### 6.4.5 — Batch Triplet Losses (for Labeled Class Data)

A separate family of losses handles a distinct data format: **single sentences with an integer class label** `(sentence, class_id)`. Rather than requiring pre-built pairs, these losses automatically mine triplets from within each batch by treating same-class samples as positives and different-class samples as negatives.

| Loss | Triplet Mining Strategy | Best For |
|---|---|---|
| `BatchAllTripletLoss` | Averages loss over *all* valid triplets in the batch | Datasets with many classes per batch |
| `BatchHardTripletLoss` | Uses the *hardest* negative (closest) and *hardest* positive (farthest) per anchor | Maximum gradient signal; risk of noisy gradients early in training |
| `BatchSemiHardTripletLoss` | Negatives that violate the margin but are not the absolute hardest | Stable training; best default for most labeled-class scenarios |
| `BatchHardSoftMarginTripletLoss` | Hard mining with a soft (learned) margin instead of a fixed constant | Avoids the need to tune the margin hyperparameter |

```python
from sentence_transformers.losses import BatchSemiHardTripletLoss
from datasets import load_dataset

# Data format: (sentence, label) where label is an integer class id
# e.g. TREC question-type classification, news topic datasets
dataset = load_dataset("trec", split="train")
# Rename columns to match expected "sentence" and "label"

loss = BatchSemiHardTripletLoss(model)
# The trainer automatically groups same-label samples in each batch
```

**When to use:** You have a labeled corpus (news categories, product types, intent labels) and want items from the same category embedded close together — without building explicit pairs.

### 6.5 Generation 5 — MultipleNegativesRankingLoss ⭐ (Modern Standard)

This is the breakthrough that dominates modern embedding training. The idea: for a batch of N positive pairs `{(a₁, p₁), (a₂, p₂), ..., (aₙ, pₙ)}`, treat all *other* positives in the batch as negatives for each anchor.

**Formula:**
```
similarity_matrix[i][j] = cosine(aᵢ, pⱼ)   for all i, j

loss = CrossEntropy(similarity_matrix, diagonal_labels)
```

The diagonal of the similarity matrix is the "correct" answer (anchor matches its own positive). Everything else is treated as a negative. This means a batch of 64 pairs gives you 63 negatives per anchor — for free, with no manual negative curation.

**Why it works so well:**
- As batch size grows, you get stronger negative signals. A batch of 512 gives 511 negatives per anchor.
- The loss optimizes *ranking* directly, not just individual pair scores.
- With hard negatives added explicitly, you get the best of both worlds.

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss

model = SentenceTransformer("bert-base-uncased")
loss = MultipleNegativesRankingLoss(model)
```

**Important caveat:** If your batch accidentally contains pairs where `aᵢ` is actually semantically close to `pⱼ` for some `i ≠ j`, you're training on false negatives. This is where **GISTEmbedLoss** comes in.

### 6.6 Generation 6 — CoSENTLoss and AnglELoss

**CoSENTLoss** ([Su Jianlin, 2022](https://kexue.fm/archives/8847)) fixes a theoretical weakness in regression losses: MSE on cosine similarity doesn't guarantee the *ordering* of pairs is correct.

CoSENTLoss operates on all pairs within a batch simultaneously and penalizes cases where a pair with a lower gold score has a higher predicted similarity than a pair with a higher gold score:

**Formula:**
```
For all pairs (i,j) and (k,l) where score(i,j) > score(k,l):

loss = log(1 + Σ exp(cosine(k,l) - cosine(i,j)))
```

This is a **list-wise** loss — it considers the relative ordering of all pairs, not just individual predictions. Empirically, it outperforms CosineSimilarityLoss consistently.

**AnglELoss** ([Li & Li, 2023](https://arxiv.org/abs/2309.12871)) is a drop-in upgrade: it uses **angle distance** instead of cosine similarity, which is better behaved on the unit hypersphere and avoids saturation at extreme similarity values:

```
angle_similarity(u, v) = angle between complex projections of u and v
```

In practice, AnglELoss outperforms CoSENTLoss, especially for long texts. **Use AnglELoss as your default for scored-pair data.**

### 6.7 Generation 7 — MatryoshkaLoss (Flexible Dimensions)

Named after Russian nesting dolls, **Matryoshka Representation Learning** ([Kusupati et al., 2022](https://arxiv.org/abs/2205.13147)) trains a single model to produce valid embeddings at *multiple dimensions simultaneously*.

The trick: apply your base loss not just to the full 768-dimensional embedding, but also to the first 512, 256, 128, 64, and 32 dimensions. The model learns to front-load the most important information into the first dimensions.

**Formula:**
```
loss = Σ wₘ · base_loss(embeddings[:m])    for m in [768, 512, 256, 128, 64, 32]
```

Where `wₘ` are dimension-level weights (typically uniform).

**Practical impact:** You train once but can serve at 1/24th the size (32d) with acceptable quality, or at full quality when precision matters. This is especially valuable for large-scale retrieval systems where storage and ANN speed matter.

```python
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

base_loss = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=[768, 512, 256, 128, 64])
```

### 6.8 Generation 8 — GISTEmbedLoss (Guide Model Filtering)

**GISTEmbedLoss** ([Solatorio, 2024](https://arxiv.org/abs/2402.16829)) addresses the false-negative problem in MultipleNegativesRankingLoss. A **guide model** (a strong pre-trained embedding model) evaluates all in-batch candidate negatives. Candidates that the guide model considers semantically similar to the anchor are *excluded* from being treated as negatives — they would be false negatives.

```python
from sentence_transformers.losses import GISTEmbedLoss

guide_model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # strong guide
loss = GISTEmbedLoss(model, guide=guide_model)
```

The trade-off is compute: every training step runs inference through both models. But for high-quality domain-specific fine-tuning, the cleaner negatives are worth it.

### Summary: Complete Loss Function Reference

| Loss | Data Format | When to Use | Quality |
|------|------------|-------------|---------|
| `SoftmaxLoss` | (a, b, class_label) | NLI-style classification over pairs | ★★☆ |
| `CosineSimilarityLoss` | (a, b, float_score) | STS regression; simple but weak ordering | ★★☆ |
| `ContrastiveLoss` | (a, b, 0/1) | Binary similar/dissimilar pairs | ★★☆ |
| `OnlineContrastiveLoss` | (a, b, 0/1) | Contrastive with dynamic hard pair mining | ★★★ |
| `TripletLoss` | (a, p, n) | Pre-curated triplets with explicit negatives | ★★★ |
| `BatchSemiHardTripletLoss` | (sentence, class_id) | Labeled class data; auto-mines semi-hard triplets | ★★★ |
| `BatchHardTripletLoss` | (sentence, class_id) | Labeled class data; hardest triplet per anchor | ★★★ |
| `BatchAllTripletLoss` | (sentence, class_id) | Labeled class data; all valid triplets averaged | ★★★ |
| `MultipleNegativesRankingLoss` | (a, p) | General-purpose; large in-batch negatives | ★★★★ |
| `CachedMultipleNegativesRankingLoss` | (a, p) | Same as MNRL but supports very large virtual batches | ★★★★ |
| `CoSENTLoss` | (a, b, float_score) | Scored pairs; rank-aware, better than MSE | ★★★★ |
| `AnglELoss` | (a, b, float_score) | Best for scored pairs; angle-based, avoids saturation | ★★★★★ |
| `MatryoshkaLoss` | wraps any loss | Variable-dimension serving from one model | ★★★★★ |
| `Matryoshka2dLoss` | wraps any loss | Variable dims + variable layers simultaneously | ★★★★★ |
| `GISTEmbedLoss` | (a, p) + guide model | Filters false negatives via a guide model | ★★★★★ |
| `CachedGISTEmbedLoss` | (a, p) + guide model | GISTEmbed with large virtual batch caching | ★★★★★ |
| `AdaptiveLayerLoss` | wraps any loss | Train model to work well with fewer transformer layers | ★★★★ |
| `DenoisingAutoEncoderLoss` | (corrupted, original) | Unsupervised; no labeled data needed | ★★☆ |
| `MSELoss` | (student_emb, teacher_emb) | Knowledge distillation from a larger teacher model | ★★★ |
| `MarginMSELoss` | (a, p, n) + teacher scores | Distillation preserving teacher's margin between pairs | ★★★★ |

---

## 7. Training Script

Here is a complete, minimal training script using the modern SBERT training API:

```python
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.training_args import BatchSamplers

# --- 1. Load base model ---
model = SentenceTransformer("microsoft/mpnet-base")

# --- 2. Prepare dataset ---
# Format: (anchor, positive) pairs — in-batch negatives will be auto-constructed
train_data = [
    {"anchor": "How do I reset my password?", "positive": "You can reset your password via the login page."},
    {"anchor": "What is the return policy?",  "positive": "Items can be returned within 30 days of purchase."},
    {"anchor": "Where is my order?",           "positive": "Track your order using the tracking number in your email."},
    # ... more pairs
]
eval_data = [
    {"anchor": "How to change my email?",     "positive": "Update your email in account settings."},
]

train_dataset = Dataset.from_list(train_data)
eval_dataset  = Dataset.from_list(eval_data)

# --- 3. Define loss ---
# Matryoshka wraps MNR — trains flexible-dimension embeddings
base_loss = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=[768, 512, 256, 128, 64])

# --- 4. Training arguments ---
args = SentenceTransformerTrainingArguments(
    output_dir="output/my-embedding-model",
    num_train_epochs=3,
    per_device_train_batch_size=64,   # larger = more in-batch negatives
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # ensures no false negatives in batch
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# --- 5. Train ---
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()

# --- 6. Save ---
model.save_pretrained("output/my-embedding-model/final")
```

**Key training choices explained:**
- `per_device_train_batch_size=64`: With MNR loss, each item gets 63 negatives. More is better; limited only by GPU VRAM.
- `BatchSamplers.NO_DUPLICATES`: Prevents the same sentence from appearing twice in a batch, which would create false negatives.
- `warmup_ratio=0.1`: Gradually increase LR for the first 10% of steps — critical for transformer fine-tuning stability.
- `fp16=True`: Half-precision saves ~40% VRAM with negligible quality loss.

---

## 8. Evaluation on Sample Data

Before running MTEB, evaluate on your own held-out data to catch issues early.

### 8.1 Semantic Similarity Evaluator (for STS-style tasks)

```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("output/my-embedding-model/final")

# Provide sentence pairs with gold similarity scores (0–1)
sentences_a = ["A dog is playing in the park.", "It is raining heavily."]
sentences_b = ["A puppy runs on the grass.",    "The sun is shining brightly."]
scores       = [0.85,                            0.05]

evaluator = EmbeddingSimilarityEvaluator(
    sentences1=sentences_a,
    sentences2=sentences_b,
    scores=scores,
    name="my-sts-eval",
)
result = evaluator(model)
# Returns Spearman and Pearson correlation between predicted and gold similarity
print(result)
```

### 8.2 Information Retrieval Evaluator (for search/RAG tasks)

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

queries = {
    "q1": "What causes climate change?",
    "q2": "How is steel manufactured?",
}
corpus = {
    "d1": "Greenhouse gases trap heat in the atmosphere, causing global temperatures to rise.",
    "d2": "CO2 emissions from fossil fuels are the primary driver of climate change.",
    "d3": "Steel is produced by smelting iron ore with carbon in a blast furnace.",
    "d4": "The Eiffel Tower is located in Paris, France.",
}
relevant_docs = {
    "q1": {"d1", "d2"},
    "q2": {"d3"},
}

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="my-retrieval-eval",
    score_functions={"cosine": lambda u, v: (u @ v.T)},
)
result = evaluator(model)
# Returns nDCG@10, MAP@100, Recall@100, Precision@10
print(result)
```

### 8.3 Triplet Evaluator

```python
from sentence_transformers.evaluation import TripletEvaluator

anchors   = ["A man is eating an apple."]
positives = ["A person is consuming fruit."]
negatives = ["A car drives on a highway."]

evaluator = TripletEvaluator(
    anchors=anchors, positives=positives, negatives=negatives,
    name="my-triplet-eval",
)
result = evaluator(model)
# Returns accuracy: fraction of triplets where sim(a,p) > sim(a,n)
print(result)
```

### 8.4 Additional Evaluators: Complete Reference

The sentence-transformers library ships more evaluators than the three shown above. Choose based on your task:

| Evaluator | Input Format | Primary Metric | Use Case |
|---|---|---|---|
| `EmbeddingSimilarityEvaluator` | (s1, s2, score) | Spearman + Pearson correlation | STS-style similarity scoring |
| `InformationRetrievalEvaluator` | queries, corpus, relevant_docs | nDCG@10, MAP, Recall@k | RAG retrieval, semantic search |
| `TripletEvaluator` | (anchor, positive, negative) | Accuracy: sim(a,p) > sim(a,n) | Triplet-trained models |
| `BinaryClassificationEvaluator` | (s1, s2, label 0/1) | AP, F1, Accuracy | Paraphrase / duplicate detection |
| `ParaphraseMiningEvaluator` | sentences + duplicates list | Average Precision | Finding all paraphrase pairs in a large corpus |
| `RerankingEvaluator` | queries + candidate lists with labels | MAP | Reranking quality evaluation |
| `MSEEvaluator` | source sentences + target model | MSE of embedding distances | Knowledge distillation quality |
| `TranslationEvaluator` | source-lang + target-lang sentence lists | Accuracy | Cross-lingual alignment quality |
| `SequentialEvaluator` | list of evaluators | Configurable (first / mean) | Combine multiple evaluators; one score for early stopping |

**Combining evaluators with SequentialEvaluator:**

```python
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator, SequentialEvaluator,
)
from datasets import load_dataset

# STS evaluator
sts = load_dataset("mteb/stsbenchmark-sts", split="validation")
sts_eval = EmbeddingSimilarityEvaluator(
    sentences1=sts["sentence1"], sentences2=sts["sentence2"],
    scores=[s / 5.0 for s in sts["score"]], name="sts-val",
)

# IR evaluator (small retrieval set)
ir_eval = InformationRetrievalEvaluator(
    queries={"q1": "climate change causes"},
    corpus={"d1": "Greenhouse gases trap heat.", "d2": "CO2 drives warming."},
    relevant_docs={"q1": {"d1", "d2"}},
    name="ir-val",
)

# Combine: early stopping uses STS score (index 0)
combined = SequentialEvaluator(
    [sts_eval, ir_eval],
    main_score_function=lambda scores: scores[0],
)
result = combined(model)
```

### 8.5 Quick Sanity Check

Always do this before anything else:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("output/my-embedding-model/final")

test_pairs = [
    ("The cat sat on the mat.", "A kitten rested on the rug."),  # should be HIGH
    ("The cat sat on the mat.", "Quarterly earnings beat expectations."),  # should be LOW
]

for s1, s2 in test_pairs:
    e1 = model.encode(s1)
    e2 = model.encode(s2)
    sim = cosine_similarity([e1], [e2])[0][0]
    print(f"sim={sim:.3f}  |  '{s1[:40]}' <-> '{s2[:40]}'")
```

---

## 9. Inferencing

### 9.1 Basic Encoding

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Single sentence
embedding = model.encode("What is machine learning?")
print(embedding.shape)  # (1024,) for large models, (768,) for base

# Batch encoding — always prefer batches over looping
sentences = ["First sentence.", "Second sentence.", "Third sentence."]
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)
print(embeddings.shape)  # (3, 1024)
```

### 9.2 Semantic Search

```python
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Encode corpus once, store
corpus = [
    "Python is a high-level programming language.",
    "Neural networks are inspired by the human brain.",
    "The Eiffel Tower was built in 1889.",
    "Gradient descent minimizes the loss function.",
]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

# At query time
query = "How do neural nets learn?"
query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

# Cosine similarity (dot product since normalized)
scores = util.dot_score(query_embedding, corpus_embeddings)[0]
top_k = scores.topk(2)

for score, idx in zip(top_k.values, top_k.indices):
    print(f"Score: {score:.4f}  |  {corpus[idx]}")
```

### 9.3 Instruction-Aware Models (E5, BGE)

Modern top-ranked models like **E5-large-v2** and **BGE-large** require instruction prefixes to signal query vs passage intent:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

# Queries must be prefixed with "query: "
# Passages/documents must be prefixed with "passage: "
queries   = ["query: What causes inflation?"]
passages  = ["passage: Inflation is caused by excess money supply chasing limited goods."]

q_emb = model.encode(queries,  normalize_embeddings=True)
p_emb = model.encode(passages, normalize_embeddings=True)

similarity = (q_emb @ p_emb.T)[0][0]
print(f"Similarity: {similarity:.4f}")
```

### 9.4 Matryoshka Truncation

If you trained with MatryoshkaLoss, you can freely truncate at inference:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
emb_full = model.encode("Hello world", normalize_embeddings=True)  # 768d

# Use only first 256 dimensions — still valid!
emb_small = emb_full[:256]
emb_small = emb_small / np.linalg.norm(emb_small)  # re-normalize after truncation
print(emb_small.shape)  # (256,)
```

---

## 10. The MTEB Benchmark: What It Measures and How

**MTEB** ([Massive Text Embedding Benchmark, Muennighoff et al., 2022](https://arxiv.org/abs/2210.07316)) is the gold standard for evaluating embedding models. It covers **112 languages** across **58 datasets** spanning **8 task categories**, each testing a different facet of embedding quality.

Understanding what each task measures — and which metric to use — is the key to picking the right model for your problem.

### 10.1 Retrieval (58 datasets, PRIMARY task)

**What it tests:** Given a query, retrieve the most relevant documents from a large corpus. Datasets include MS MARCO, BEIR benchmarks (TREC-COVID, NFCorpus, HotpotQA, etc.), and domain-specific collections.

**Primary metric: nDCG@10 (Normalized Discounted Cumulative Gain at 10)**

The intuition: not all positions in a ranked list are equal. The first result is more valuable than the tenth. nDCG accounts for this with a logarithmic position discount.

**Step-by-step calculation:**

*Step 1 — Compute DCG (Discounted Cumulative Gain):*
```
DCG@K = Σᵢ₌₁ᴷ  relevance(i) / log₂(i + 1)

Position 1: relevance / log₂(2) = relevance / 1.0
Position 2: relevance / log₂(3) = relevance / 1.585
Position 3: relevance / log₂(4) = relevance / 2.0
...
```

*Step 2 — Compute IDCG (Ideal DCG):*
Reorder results so all relevant items appear first. Compute DCG of this ideal ranking.

*Step 3 — Normalize:*
```
nDCG@K = DCG@K / IDCG@K
```

**Example:**
```
Query: "climate change causes"
Results returned: [relevant, irrelevant, relevant, irrelevant, relevant]

DCG@5 = 1/log₂(2) + 0/log₂(3) + 1/log₂(4) + 0/log₂(5) + 1/log₂(6)
      = 1.0 + 0 + 0.5 + 0 + 0.387 = 1.887

Ideal order: [relevant, relevant, relevant, irrelevant, irrelevant]
IDCG@5 = 1.0 + 0.631 + 0.5 = 2.131

nDCG@5 = 1.887 / 2.131 = 0.886
```

**Other retrieval metrics:**
- **MAP@100** (Mean Average Precision): Average of precision measured at each relevant item position. More sensitive to early relevant items.
- **Recall@100**: Fraction of all relevant documents retrieved in top 100.
- **MRR@10** (Mean Reciprocal Rank): Reciprocal of the rank of the first relevant result. MRR = 1 if the first result is relevant, 0.5 if the second, etc.

### 10.2 Semantic Textual Similarity (STS)

**What it tests:** Given pairs of sentences, predict their semantic similarity score. The model's cosine similarity is correlated against human-annotated gold scores.

**Primary metric: Spearman Correlation**

Rather than comparing raw cosine similarity values to gold scores (Pearson correlation), Spearman correlation compares their *ranks*. This is more robust to score distribution mismatches.

**Calculation:**
```
1. Compute cosine_sim(a, b) for all pairs → predicted scores
2. Rank predicted scores
3. Rank gold scores
4. Compute Pearson correlation on the ranks

ρ = 1 - (6 · Σdᵢ²) / (n · (n² - 1))

where dᵢ = difference in ranks for pair i
      n   = number of pairs
```

**Range:** -1 to 1. Higher is better. State-of-the-art models score ~0.90–0.92 on STSbenchmark.

### 10.3 Clustering

**What it tests:** Embed a collection of texts, cluster them (e.g., with K-Means), and measure whether the discovered clusters match the true categories.

**Primary metric: V-Measure**

V-Measure is the harmonic mean of **homogeneity** and **completeness** — two complementary clustering properties:

- **Homogeneity**: Each cluster contains only members of a single true class.
  ```
  h = 1 - H(C|K) / H(C)
  where H(C|K) = conditional entropy of class distribution given cluster assignments
        H(C)   = entropy of the class distribution
  ```

- **Completeness**: All members of a true class are assigned to the same cluster.
  ```
  c = 1 - H(K|C) / H(K)
  ```

- **V-Measure:**
  ```
  V = (1 + β) · (h · c) / (β · h + c)    [β=1 gives equal weight]
  ```

**Range:** 0 to 1. Higher is better. A perfect clustering has V=1.0.

### 10.4 Classification

**What it tests:** Use embeddings as features for a downstream classification task (linear probe — logistic regression on top of frozen embeddings). Tests whether the embedding space is linearly separable by class.

**Primary metric: Accuracy (or F1 for imbalanced classes)**

```
Accuracy = correct predictions / total predictions
```

### 10.5 Reranking

**What it tests:** Given a query, a list of relevant documents, and a list of irrelevant documents, measure how well the model reorders them (relevant first).

**Primary metric: MAP (Mean Average Precision)**

```
AP for one query = (1/R) · Σₖ P@k · rel(k)

where R   = total relevant documents
      P@k = precision at position k
      rel(k) = 1 if document at position k is relevant, else 0

MAP = mean of AP across all queries
```

**Example:**
```
Query, 2 relevant docs in corpus.
Returned order: [relevant, irrelevant, relevant, irrelevant, irrelevant]

P@1 = 1/1 = 1.0  (relevant at position 1)
P@3 = 2/3 = 0.667 (relevant at position 3)

AP = (1.0 + 0.667) / 2 = 0.833
```

### 10.6 Pair Classification

**What it tests:** Given a pair of texts, classify whether they are semantically equivalent (paraphrases), contradictory, etc. Models produce a similarity score; a threshold is tuned to classify.

**Primary metric: Average Precision (AP) on the binary label**

```
AP = Σₙ (Rₙ - Rₙ₋₁) · Pₙ
```

This is the area under the precision-recall curve — robust to threshold choice.

### 10.7 Bitext Mining

**What it tests:** Given parallel bilingual corpora, identify matching translations. Tests cross-lingual alignment quality.

**Primary metric: F1 score** on correctly identified translation pairs.

### 10.8 Summarization

**What it tests:** Embed generated summaries and reference summaries; correlate their cosine similarity with human quality scores.

**Primary metric: Spearman correlation** (same calculation as STS).

### 10.9 BEIR: Zero-Shot Retrieval Benchmark

**BEIR** ([Thakur et al., 2021](https://arxiv.org/abs/2104.08663)) is the standard framework for testing embedding models on **zero-shot domain transfer** — no fine-tuning on any of the 18 test domains. It reveals whether a model that tops MS MARCO leaderboards actually generalises to new domains.

Key BEIR datasets, all available via MTEB:

| Dataset | Domain | Queries | Corpus Size | HuggingFace |
|---|---|---|---|---|
| TREC-COVID | Biomedical | 50 | 171K docs | [`mteb/trec-covid`](https://huggingface.co/datasets/mteb/trec-covid) |
| NFCorpus | Medical IR | 323 | 3.6K docs | [`mteb/nfcorpus`](https://huggingface.co/datasets/mteb/nfcorpus) |
| HotpotQA | Multi-hop Wikipedia QA | 7.4K | 5.2M docs | [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa) |
| FiQA-2018 | Finance QA | 648 | 57K docs | [`mteb/fiqa`](https://huggingface.co/datasets/mteb/fiqa) |
| ArguAna | Counter-argument retrieval | 1.4K | 8.7K docs | [`mteb/arguana`](https://huggingface.co/datasets/mteb/arguana) |
| SciFact | Scientific claim verification | 300 | 5K docs | [`mteb/scifact`](https://huggingface.co/datasets/mteb/scifact) |
| DBPedia | Entity retrieval | 400 | 4.6M docs | [`mteb/dbpedia-entity`](https://huggingface.co/datasets/mteb/dbpedia-entity) |

**Key BEIR finding:** Models that rank highest on in-domain benchmarks (e.g., MS MARCO-trained models) often show poor zero-shot transfer on BEIR biomedical or legal datasets. This is why BGE-M3 and GTE use diverse-domain pre-training corpora.

**Running MTEB / BEIR locally:**

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Run a targeted subset (BEIR retrieval tasks)
evaluation = MTEB(tasks=["NFCorpus", "TREC-COVID", "FiQA2018", "SciFact"])
results = evaluation.run(model, output_folder="results/bge-large")

# Run full English MTEB (takes several hours)
# evaluation = MTEB(language="en")
# results = evaluation.run(model, output_folder="results/bge-large-full")
```

Submit to the public leaderboard by adding the MTEB results metadata to your model card on HuggingFace Hub.

---

## 11. How to Choose the Right Model for Your Use Case

The MTEB leaderboard lists hundreds of models. Here is a principled framework for selecting one:

### Step 1: Identify your primary task type

| Your Use Case | MTEB Task | Primary Metric |
|---|---|---|
| Semantic search / RAG | Retrieval | nDCG@10 |
| Duplicate detection | Pair Classification | AP |
| Sentence similarity scoring | STS | Spearman |
| Topic/intent clustering | Clustering | V-Measure |
| Document classification | Classification | Accuracy |
| Re-ranking search results | Reranking | MAP |
| Cross-lingual alignment | Bitext Mining | F1 |

### Step 2: Filter by dimension and language

- If you're serving at scale and VRAM is limited → prefer 384d models (e.g., `all-MiniLM-L6-v2`, `bge-small-en-v1.5`)
- If quality is paramount → use 1024d models (e.g., `bge-large-en-v1.5`, `e5-large-v2`)
- If multilingual → filter to the multilingual MTEB track (`multilingual-e5-large`, `paraphrase-multilingual-mpnet`)
- If variable dimension is needed → use Matryoshka models (`nomic-embed-text-v1.5`, `mxbai-embed-large-v1`)

### Step 3: Check domain fit

MTEB averages over many datasets, but your domain may differ. A model that tops the MTEB retrieval average on Wikipedia-based queries may underperform on medical or legal text. Always:

1. Pick the top-3 MTEB models for your task
2. Evaluate on a **100–500 sample** of your actual query/document pairs
3. Select based on your in-domain evaluation, not just MTEB rank

### Step 4: Consider the query/passage instruction requirement

Models like E5 and BGE require instruction prefixes:
- E5: `"query: ..."` / `"passage: ..."`
- BGE-EN-ICL: full instruction like `"Represent this sentence for retrieval: ..."`
- Nomic: `"search_query: ..."` / `"search_document: ..."`

If you forget these prefixes, performance drops significantly.

### Quick Selection Guide

| Scenario | Recommended Model |
|---|---|
| General English retrieval (high quality) | `BAAI/bge-large-en-v1.5` |
| General English retrieval (fast/small) | `BAAI/bge-small-en-v1.5` |
| Sentence similarity / STS | `sentence-transformers/all-mpnet-base-v2` |
| Variable dimensions (Matryoshka) | `nomic-ai/nomic-embed-text-v1.5` |
| Multilingual | `intfloat/multilingual-e5-large` |
| Latest frontier (as of 2025) | `BAAI/bge-en-icl`, `Alibaba-NLP/gte-large-en-v1.5` |

---

## 12. State-of-the-Art Embedding Models: Full Comparison

The table below covers the most widely used and highest-performing embedding models as of mid-2025 — open-source and proprietary — with training details, dimensions, and recommended use cases.

A few notes on reading this table:
- **Dims** is the default output dimension; models marked with `†` support Matryoshka truncation to smaller sizes.
- **Training Loss** for closed-source models is inferred from technical blog posts and papers where available; otherwise marked *Proprietary*.
- **Data Format** describes the input structure used during fine-tuning (pre-training data formats are omitted for brevity).

---

### 12.1 Open-Source Models

*Popularity tier: 🔥 >1M downloads/month · ⭐ 100K–1M · 📈 10K–100K · 🆕 <10K (HuggingFace monthly downloads, May 2025)*

| Model | HF Link | Dims | Provider | Released | Best Use Cases | Training Loss | Data Format | Popularity |
|---|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | [🤗](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | SBERT / HuggingFace | 2021 | Lightweight semantic search, mobile, edge | MultipleNegativesRankingLoss + knowledge distillation | (anchor, positive) pairs | 🔥 251M/mo |
| `all-mpnet-base-v2` | [🤗](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 768 | SBERT / HuggingFace | 2021 | Sentence similarity, clustering, paraphrase detection | MultipleNegativesRankingLoss on 1B+ curated pairs | (anchor, positive) pairs | 🔥 37M/mo |
| `bge-small-en-v1.5` | [🤗](https://huggingface.co/BAAI/bge-small-en-v1.5) | 512 | BAAI | Sept 2023 | Fast retrieval, low-latency RAG | Contrastive loss + BM25 hard negatives | (anchor, positive, hard-negative) triplets | 🔥 38M/mo |
| `bge-base-en-v1.5` | [🤗](https://huggingface.co/BAAI/bge-base-en-v1.5) | 768 | BAAI | Sept 2023 | Balanced quality/speed retrieval | Contrastive + hard negatives; RetroMAE pre-train | (anchor, positive, hard-negative) triplets | 🔥 ~20M/mo |
| `bge-large-en-v1.5` | [🤗](https://huggingface.co/BAAI/bge-large-en-v1.5) | 1024 | BAAI | Sept 2023 | High-accuracy English retrieval, reranking | Contrastive + hard negatives; RetroMAE pre-train | (anchor, positive, hard-negative) triplets | 🔥 15M/mo |
| `bge-m3` | [🤗](https://huggingface.co/BAAI/bge-m3) | 1024 | BAAI | Jan 2024 | Dense+sparse+multi-vector; 100+ languages; 8K context | Self-knowledge distillation across retrieval heads | (anchor, positive) + hard negatives | 🔥 22M/mo |
| `nomic-embed-text-v1.5` | [🤗](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 768 `†` (64–768) | Nomic AI | Feb 2024 | Variable-dimension RAG, storage-constrained search | MatryoshkaLoss + MultipleNegativesRankingLoss | (anchor, positive) pairs; task-prefix instructions | 🔥 16M/mo |
| `multilingual-e5-large` | [🤗](https://huggingface.co/intfloat/multilingual-e5-large) | 1024 | Microsoft | 2023 | Multilingual retrieval (100+ languages) | Weak-supervision contrastive on multilingual web data | (`query: …`, `passage: …`) pairs | 🔥 7.3M/mo |
| `mxbai-embed-large-v1` | [🤗](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | 1024 `†` | Mixedbread AI | 2024 | Variable-dim serving, quantization-aware deployment | MatryoshkaLoss + quantization-aware contrastive training | (anchor, positive) pairs | 🔥 4.4M/mo |
| `e5-large-v2` | [🤗](https://huggingface.co/intfloat/e5-large-v2) | 1024 | Microsoft | 2023 | General retrieval, semantic search, ranking | Weak-supervision contrastive on web (title, body) pairs | (`query: …`, `passage: …`) pairs | 🔥 1.3M/mo |
| `gte-large-en-v1.5` | [🤗](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) | 1024 | Alibaba NLP | 2024 | Long-context retrieval (8K tokens), domain-general | Contrastive + hard negative mining; large-scale pre-train | (anchor, positive) + hard negatives | 🔥 1M/mo |
| `SFR-Embedding-2_R` | [🤗](https://huggingface.co/Salesforce/SFR-Embedding-2_R) | 4096 | Salesforce | 2024 | Long-context (4K) retrieval, reranking | Contrastive fine-tuning on Mistral/Llama backbone | (anchor, positive) + hard negatives | ⭐ 175K/mo |
| `stella_en_1.5B_v5` | [🤗](https://huggingface.co/dunzhang/stella_en_1.5B_v5) | 8192 `†` (512–8192) | DunZhang | 2024 | MTEB-topping flexible retrieval; large capacity | Matryoshka Representation Learning (MRL) | (anchor, positive) + hard negatives | 📈 54K/mo |
| `NV-Embed-v2` | [🤗](https://huggingface.co/nvidia/NV-Embed-v2) | 4096 | NVIDIA | 2024 | SOTA retrieval, long-context (32K), instruction-following | Contrastive + instruction tuning; two-stage training | (instruction, anchor, positive) triplets | 📈 24K/mo |
| `bge-en-icl` | [🤗](https://huggingface.co/BAAI/bge-en-icl) | 4096 | BAAI | Jul 2024 | In-context learning retrieval; few-shot adaptation | Contrastive with in-context examples in prompt | (instruction, few-shot examples, anchor, positive) | 🆕 1.8K/mo |

---

### 12.2 Closed-Source / API Models

| Model | API / Docs | Dims | Provider | Released | Best Use Cases | Training Loss | Data Format | Popularity |
|---|---|---|---|---|---|---|---|---|
| `text-embedding-ada-002` | [OpenAI](https://platform.openai.com/docs/models/text-embedding) | 1536 | OpenAI | Dec 2022 | General search, code similarity, classification | Proprietary contrastive | Proprietary | 💼 API only |
| `text-embedding-3-small` | [OpenAI](https://platform.openai.com/docs/models/text-embedding) | 1536 `†` (256–1536) | OpenAI | Jan 2024 | Cost-efficient search, RAG at scale | Proprietary + Matryoshka (variable dims at inference) | Proprietary | 💼 API only |
| `text-embedding-3-large` | [OpenAI](https://platform.openai.com/docs/models/text-embedding) | 3072 `†` (256–3072) | OpenAI | Jan 2024 | High-accuracy semantic search, complex RAG | Proprietary + Matryoshka | Proprietary | 💼 API only |
| `embed-english-v3.0` | [Cohere](https://docs.cohere.com/docs/embed-2) | 1024 | Cohere | Nov 2023 | Enterprise search, document retrieval, RAG | Proprietary contrastive with hard negatives | Proprietary | 💼 API only |
| `voyage-3` | [Voyage AI](https://docs.voyageai.com/docs/embeddings) | 1024 | Voyage AI | Sept 2024 | Dense retrieval, general-purpose RAG | Proprietary contrastive | Proprietary | 💼 API only |
| `voyage-3-large` | [Voyage AI](https://docs.voyageai.com/docs/embeddings) | 2048 `†` (256–2048) | Voyage AI | Jan 2025 | SOTA dense retrieval, code search, long-context | Proprietary + Matryoshka (quantization-aware) | Proprietary | 💼 API only |
| `text-embedding-004` | [Google](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings) | 768 `†` (1–768) | Google | May 2024 | Retrieval, semantic similarity, classification | Proprietary + variable-dim (Matryoshka-style) | Proprietary | 💼 API only |
| `amazon-titan-embed-text-v2` | [Amazon](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) | 1024 `†` (256–1024) | Amazon | Apr 2024 | RAG on AWS, classification, document search | Proprietary contrastive | Proprietary | 💼 API only |

---

### 12.3 Choosing by Dimension vs Quality Trade-off

This chart shows where each open-source model sits on the size-vs-quality spectrum:

```
Quality (MTEB Retrieval nDCG@10)
  ▲
  │                                              ● stella_en_1.5B_v5 (8192d)
  │                              ● NV-Embed-v2 (4096d)
  │              ● bge-m3 (1024d)  ● gte-large-en-v1.5 (1024d)
  │     ● bge-large-en-v1.5 (1024d) ● mxbai-embed-large-v1 (1024d)
  │  ● e5-large-v2 (1024d)
  │          ● nomic-embed-text-v1.5 (768d†)
  │  ● all-mpnet-base-v2 (768d)
  │      ● bge-small-en-v1.5 (512d)
  │  ● all-MiniLM-L6-v2 (384d)
  └─────────────────────────────────────────────────► Model Size / Latency
     Fast                                         Slow / Expensive
```

`†` = supports Matryoshka truncation — one model, multiple operating points.

---

### 12.4 Key Trends Across the Landscape

**1. Fixed → Matryoshka dimensions.** Earlier models produced one fixed-size vector. Modern models (nomic, mxbai, stella, voyage-3-large, text-embedding-3) use Matryoshka loss so a single model can serve at 64d, 256d, or 1024d depending on your cost/quality budget.

**2. 512-token → 8K+ context.** BGE-M3, GTE, NV-Embed-v2 support 8K–32K tokens, enabling whole-document embeddings without chunking.

**3. Dense-only → Dense + Sparse + Multi-vector.** BGE-M3 unifies all three retrieval modes in a single model — dense vectors for semantic recall, learned sparse for keyword precision, and multi-vector (ColBERT-style) for fine-grained matching.

**4. Anonymous encoding → Instruction-aware.** BGE-en-ICL and NV-Embed-v2 accept task instructions in the prompt, allowing the same model to be steered toward retrieval vs classification vs clustering without retraining.

**5. Training data scale.** The gap between good and great models is increasingly determined by the *quality* of training pairs (hard negatives, deduplication, cross-encoder scoring) rather than raw data volume.

---

## 13. Putting It All Together: The Modern Recipe

If you are training or fine-tuning an embedding model today, this is the recipe most likely to give you state-of-the-art results:

1. **Start from a strong base**: `BAAI/bge-base-en-v1.5` or `microsoft/mpnet-base`
2. **Loss**: `MatryoshkaLoss(GISTEmbedLoss(...))` for maximum flexibility and clean negatives
3. **Data**: Mix of NLI entailment pairs + domain-specific (anchor, positive) pairs + hard negatives mined with BM25
4. **Batch size**: As large as VRAM allows (64–256)
5. **Sampler**: `NO_DUPLICATES` to prevent false negatives
6. **Evaluate during training**: `InformationRetrievalEvaluator` on a held-out retrieval set
7. **Post-training**: Run on MTEB retrieval + STS subsets to verify no regression

The field has moved fast — the loss functions and training infrastructure described in this blog represent the accumulated learning of 6+ years of research since the original SBERT paper. But the core insight hasn't changed: good embeddings come from exposing the model to the right signal about what "similar" means, and the loss function is how you deliver that signal.

---

## References

### Foundational Models
- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space (Word2Vec).* https://arxiv.org/abs/1301.3781
- Pennington et al. (2014). *GloVe: Global Vectors for Word Representation.* https://aclanthology.org/D14-1162/
- Bojanowski et al. (2017). *Enriching Word Vectors with Subword Information (FastText).* https://arxiv.org/abs/1607.04606
- Peters et al. (2018). *Deep Contextualized Word Representations (ELMo).* https://arxiv.org/abs/1802.05365
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* https://arxiv.org/abs/1810.04805

### Sentence Embeddings & SBERT
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* https://arxiv.org/abs/1908.10084
- Reimers & Gurevych (2021). *Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders.* https://arxiv.org/abs/2010.08240

### Loss Functions
- Su Jianlin (2022). *CoSENT: A More Effective Sentence Vector Training Method.* https://kexue.fm/archives/8847
- Li & Li (2023). *AnglE-optimized Text Embeddings.* https://arxiv.org/abs/2309.12871
- Kusupati et al. (2022). *Matryoshka Representation Learning.* https://arxiv.org/abs/2205.13147
- Solatorio (2024). *GISTEmbed: Guided In-sample Selection of Training Negatives.* https://arxiv.org/abs/2402.16829

### Training Data & Domain Adaptation
- Wang et al. (2021). *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval.* https://arxiv.org/abs/2112.09118

### Evaluation Benchmarks
- Muennighoff et al. (2022). *MTEB: Massive Text Embedding Benchmark.* https://arxiv.org/abs/2210.07316
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* https://arxiv.org/abs/2104.08663

### State-of-the-Art Models
- Xiao et al. (2023). *C-Pack: Packaged Resources to Advance General Chinese Embedding (BGE).* https://arxiv.org/abs/2309.07597
- Wang et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* https://arxiv.org/abs/2212.03533
- Chen et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity.* https://arxiv.org/abs/2402.03216
- Lee et al. (2024). *NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models.* https://arxiv.org/abs/2405.17428

### Training Infrastructure & Blogs
- Tom Aarsen (2024). *Train and Fine-Tune Sentence Transformers Models.* https://huggingface.co/blog/train-sentence-transformers
- Sentence Transformers Documentation: https://www.sbert.net/
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
