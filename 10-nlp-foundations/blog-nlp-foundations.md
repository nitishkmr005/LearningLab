# NLP Foundations: From TF-IDF to LSTMs and Attention

> The complete pre-Transformer NLP stack — every concept that shows up in DS, ML, and AI interviews.

---

## Table of Contents

1. [Why NLP Foundations Still Matter](#1-why-nlp-foundations-still-matter)
2. [Text Preprocessing — The Invisible Foundation](#2-text-preprocessing--the-invisible-foundation)
3. [Bag of Words — The Simplest Representation](#3-bag-of-words--the-simplest-representation)
4. [TF-IDF — Smarter Term Weighting](#4-tf-idf--smarter-term-weighting)
5. [N-grams — Capturing Local Context](#5-n-grams--capturing-local-context)
6. [Word Embeddings — Distributed Representations](#6-word-embeddings--distributed-representations)
7. [Recurrent Neural Networks — Sequential Processing](#7-recurrent-neural-networks--sequential-processing)
8. [The Vanishing Gradient Problem](#8-the-vanishing-gradient-problem)
9. [Long Short-Term Memory (LSTM)](#9-long-short-term-memory-lstm)
10. [Gated Recurrent Unit (GRU)](#10-gated-recurrent-unit-gru)
11. [Architecture Comparison Table](#11-architecture-comparison-table)
12. [Seq2Seq — Encoder-Decoder Architecture](#12-seq2seq--encoder-decoder-architecture)
13. [Bahdanau Attention — Fixing the Bottleneck](#13-bahdanau-attention--fixing-the-bottleneck)
14. [Production Notes — When to Use These in 2026](#14-production-notes--when-to-use-these-in-2026)
15. [Interview Q&A](#15-interview-qa)
16. [References](#16-references)

---

## 1. Why NLP Foundations Still Matter

In an era of GPT-4 and Claude, you might wonder why anyone learns TF-IDF or LSTMs. The answer is threefold. First, interviewers test these concepts heavily — they reveal whether you understand *why* Transformers work by testing whether you understand what they replaced. Second, LSTMs and GRUs are still deployed in latency-sensitive production systems where a full Transformer is overkill: IoT sensors, embedded speech recognition, real-time financial tick data, and mobile keyboards. Third, the attention mechanism that powers every modern LLM was born directly from the failure modes of LSTMs — you cannot understand attention without understanding the problem it solved.

This blog traces the full arc: sparse representations (BoW, TF-IDF, n-grams) → dense embeddings (Word2Vec, GloVe) → recurrent models (RNN, LSTM, GRU) → sequence-to-sequence learning → attention. Everything Transformers build on lives in these concepts.

---

## 2. Text Preprocessing — The Invisible Foundation

Raw text is noise. Every character, every extra space, every morphological variant is a dimension that doesn't generalize. The preprocessing pipeline is the unglamorous but load-bearing part of every NLP system — bugs here silently poison every downstream model.

### 2.1 Tokenization

Tokenization splits a string into units (tokens). The choice of tokenizer is consequential:

```
"Don't run into the woods!" 

Word-level:    ["Don't", "run", "into", "the", "woods", "!"]
Char-level:    ['D','o','n',"'",'t',' ','r','u','n',...]
Subword (BPE): ["don", "'t", "run", "into", "the", "wood", "s", "!"]
```

- **Word-level**: simple, large vocabulary, OOV problem for rare words.
- **Character-level**: zero OOV, very long sequences.
- **Subword (BPE / WordPiece)**: the modern default — balances vocabulary size against OOV coverage. Used in every LLM.

> 🎯 **Interview prep**: "What's the OOV problem and how do subword tokenizers solve it?" — Word-level tokenizers assign `<UNK>` to unseen words; BPE encodes them as subword fragments that were seen during training.

### 2.2 Normalization

```python
import re

text = "  Running FAST is great!! 😊  "
text = text.lower()                      # "  running fast is great!! 😊  "
text = re.sub(r'[^a-z\s]', '', text)    # "  running fast is great  "
text = text.strip()                      # "running fast is great"
```

Key choices:
- **Lowercasing**: always correct for classical NLP; wrong for cased models like BERT-cased.
- **Punctuation removal**: safe for sentiment; dangerous for tasks like NER or parsing.
- **Unicode normalization**: `café` vs `cafe` — matters for multilingual systems.

### 2.3 Stemming vs Lemmatization

Both reduce inflected forms to a common base. They differ in quality:

| | Stemming | Lemmatization |
|---|---|---|
| Method | Rule-based truncation | Dictionary + POS lookup |
| `running` → | `run` | `run` |
| `better` → | `better` | `good` |
| `studies` → | `studi` | `study` |
| Speed | Fast | Slower |
| Library | NLTK Porter/Snowball | spaCy, NLTK WordNet |

> 🏭 **Production note**: In practice, neither is used with modern neural models — subword tokenizers implicitly handle morphology. You encounter stemming/lemmatization in legacy search systems (Elasticsearch analyzers) and TF-IDF pipelines for document retrieval.

### 2.4 Stop Word Removal

Stop words (`the`, `is`, `at`, `which`) carry low information for classification and retrieval tasks. Removing them reduces vocabulary size and noise for classical models.

**When to remove**: TF-IDF for document retrieval, classical ML features, keyword extraction.  
**When NOT to remove**: Sentiment tasks (`not` is a stop word), QA systems, language modeling. The sentence "this is not good" loses its meaning without `not`.

**Resources**
- [Stanford IR Book Chapter 2](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) — authoritative treatment of preprocessing
- [spaCy linguistic features](https://spacy.io/usage/linguistic-features) — production-grade lemmatization

---

## 3. Bag of Words — The Simplest Representation

Bag of Words treats a document as an unordered set of word counts, discarding all positional and grammatical structure. Despite its brutality, it works surprisingly well for short-document classification.

Given a vocabulary `V = {cat, sat, mat, dog, ran}` and two documents:

```
d1 = "the cat sat on the mat"
d2 = "the cat ran away"
```

The BoW vectors are:

```
         cat  sat  mat  dog  ran
d1   →  [ 1,   1,   1,   0,   0 ]
d2   →  [ 1,   0,   0,   0,   1 ]
```

(Stop words like `the`, `on`, `away` removed first.)

### BoW problems

1. **No word order**: "dog bites man" = "man bites dog" in BoW.
2. **High dimensionality**: vocabulary can be 100k+ words → sparse, high-dim vectors.
3. **No semantics**: `good` and `great` are orthogonal vectors; no similarity captured.
4. **Frequency bias**: common words dominate without normalization — hence TF-IDF.

> 🎯 **Interview prep**: "Why does BoW fail for semantic similarity?" — Orthogonal vectors for synonyms: cosine(`good`, `great`) = 0 in BoW. Solved by word embeddings.

---

## 4. TF-IDF — Smarter Term Weighting

TF-IDF (Term Frequency–Inverse Document Frequency) reweights BoW counts to penalize words that appear in many documents (so they carry less discriminative signal) and reward words that appear frequently in one document but rarely across the corpus.

The concept was developed by Spärck Jones (1972) for IDF and formalized by Salton & Buckley (Salton & Buckley, 1988) into the combined TF-IDF weighting scheme that became the industry standard for information retrieval.

### 4.1 Formula

**Term Frequency (TF)** — how often word `t` appears in document `d`:

```
TF(t, d) = count(t in d) / total words in d
```

**Inverse Document Frequency (IDF)** — penalizes words common across all documents:

```
IDF(t) = log( N / df(t) )
```

where `N` = total number of documents, `df(t)` = number of documents containing `t`.

**TF-IDF**:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### 4.2 Worked Numerical Example

Corpus of 3 documents, vocabulary: `{cat, sat, dog, bark, loud}`.

```
N = 3
d1 = "cat sat cat"           → word counts: {cat:2, sat:1}
d2 = "dog bark loud bark"    → word counts: {dog:1, bark:2, loud:1}
d3 = "cat bark"              → word counts: {cat:1, bark:1}

Document frequencies:
  df(cat)  = 2 (d1, d3)
  df(sat)  = 1 (d1)
  df(bark) = 2 (d2, d3)
  df(dog)  = 1 (d2)
```

**TF("cat", d1)**:
```
TF = 2 / 3 = 0.667
```

**IDF("cat")**:
```
IDF = log(3 / 2) = log(1.5) = 0.405
```

**TF-IDF("cat", d1)**:
```
TF-IDF = 0.667 × 0.405 = 0.270
```

**IDF("sat")** (only in d1 — rare, high IDF):
```
IDF = log(3 / 1) = log(3) = 1.099
```

**TF-IDF("sat", d1)**:
```
TF-IDF = (1/3) × 1.099 = 0.366
```

Interpretation: `sat` scores *higher* than `cat` in d1 despite lower raw frequency, because `cat` appears in multiple documents and is thus less discriminative for identifying d1.

> 🎯 **Interview prep**: "A word that appears in every document gets IDF = log(N/N) = log(1) = 0. What does that mean?" — It contributes nothing to TF-IDF regardless of its local frequency — stop word removal is baked into the math.

### 4.3 Variants

- **Sublinear TF**: `TF = 1 + log(count)` — dampens the effect of word repetition.
- **BM25**: the production default for search engines (Elasticsearch, Lucene). Adds document length normalization and saturates TF. Outperforms raw TF-IDF on retrieval benchmarks.
- **Smooth IDF**: `IDF = log((1+N)/(1+df)) + 1` — avoids division by zero for new terms.

**Resources**
- [Stanford IR Book Chapter 6](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf) — canonical TF-IDF treatment with proof
- [Salton & Buckley, 1988](https://www.researchgate.net/publication/228572832) — original term weighting paper

---

## 5. N-grams — Capturing Local Context

BoW destroys all word order. N-grams partially restore it by treating sequences of `n` consecutive words as atomic units. This preserves local syntactic structure that single words cannot.

```
sentence = "the quick brown fox"

unigrams (n=1):  ["the", "quick", "brown", "fox"]
bigrams  (n=2):  ["the quick", "quick brown", "brown fox"]
trigrams (n=3):  ["the quick brown", "quick brown fox"]
```

A bigram model learns that "New York" and "not good" are different units from "New" and "York" separately. The phrase "not good" is a bigram — removing stop words would destroy it, which is why stop-word removal must be done carefully.

### Language Model with N-grams

N-gram language models estimate the probability of a word given the previous `n-1` words:

```
P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)

Example:
P("fox" | "quick", "brown") = count("quick brown fox") / count("quick brown")
```

**Limitations**: data sparsity (unseen trigrams get zero probability — requires Kneser-Ney smoothing), no generalization across synonyms, fixed-width context window.

> 🎯 **Interview prep**: "What's the sparsity problem with n-grams?" — Most n-gram sequences in test data never appear in training data. Smoothing techniques (Laplace, Kneser-Ney) redistribute probability mass to unseen sequences, but fundamentally the problem is solved by neural language models with continuous representations.

> 🏭 **Production note**: N-gram language models are still used inside spell checkers, mobile keyboard prediction (SwiftKey historically), and as the language model component of classic ASR decoders. They are extremely fast and interpretable.

**Resources**
- [Stanford NLP: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — Chapter 3 of Jurafsky & Martin, the canonical NLP textbook

---

## 6. Word Embeddings — Distributed Representations

TF-IDF and BoW represent words as orthogonal one-hot vectors: `cosine(king, queen) = 0`. This is semantically wrong. Word embeddings solve this by learning dense, low-dimensional vectors where semantic similarity corresponds to geometric proximity.

### 6.1 Word2Vec

([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)) proposed two self-supervised training objectives:

**CBOW (Continuous Bag of Words)**: predict the center word from surrounding context words.
**Skip-gram**: predict surrounding context words given the center word (better for rare words).

```
Skip-gram objective — maximize:
log P(context | center) = Σ log P(w_c | w_t)

Famous analogy: vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

The key insight: words that appear in similar contexts have similar meanings — the distributional hypothesis (Harris, 1954). Word2Vec operationalizes this with a simple 2-layer neural network trained on next-word prediction.

### 6.2 GloVe

([Pennington et al., 2014](https://aclanthology.org/D14-1162/)) addresses Word2Vec's limitation of training on local context windows only. GloVe uses a global co-occurrence matrix and trains embeddings to predict log co-occurrence ratios:

```
J = Σ f(Xij) (wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²
```

where `Xij` is the co-occurrence count of words `i` and `j`, and `f` is a weighting function that down-weights very frequent pairs.

GloVe pre-trained vectors are available at [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/).

> 🎯 **Interview prep**: "What's the difference between Word2Vec and GloVe?" — Word2Vec: local context window, prediction-based (neural), efficient online training. GloVe: global co-occurrence statistics, matrix-factorization-based, better on analogy tasks. In practice, they perform similarly on downstream tasks; GloVe is more interpretable theoretically.

> 🏭 **Production note**: Both are largely superseded by contextual embeddings (BERT, etc.) for most NLP tasks, but pre-trained static embeddings are still used in lightweight feature pipelines, recommendation systems, and domain-specific NLP where fine-tuning a full LLM is not feasible.

**Resources**
- [Word2Vec arXiv](https://arxiv.org/abs/1301.3781) — original paper
- [GloVe project page](https://nlp.stanford.edu/projects/glove/) — pre-trained vectors + paper

---

## 7. Recurrent Neural Networks — Sequential Processing

Standard feedforward networks have no memory — they process each input independently. For sequential data (text, time series, speech), this is fatal: the meaning of "bank" depends on whether the previous words were "river" or "savings". Recurrent Neural Networks (RNNs) solve this by maintaining a hidden state `h` that accumulates information across time steps.

### 7.1 Architecture

At each time step `t`, the RNN takes the current input `xₜ` and the previous hidden state `hₜ₋₁`, and produces a new hidden state `hₜ`:

```
hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + b)
yₜ = Wᵧ · hₜ                         (if output at each step)
```

The same weight matrices `Wₕ`, `Wₓ`, `Wᵧ` are shared across all time steps — this is parameter efficiency and the source of the vanishing gradient problem.

**Unrolled RNN across time:**

```
         x(0)         x(1)         x(2)         x(3)
          │            │            │            │
     ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
h(0)→│  RNN   │→ │  RNN   │→ │  RNN   │→ │  RNN   │→ h(4)
     │  cell  │  │  cell  │  │  cell  │  │  cell  │
     └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
          │            │            │            │
         y(0)         y(1)         y(2)         y(3)

Each cell applies: hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)
Same weights W at every timestep — this is weight sharing.
```

This unrolled view makes it clear why RNNs are trained with **Backpropagation Through Time (BPTT)**: gradients flow backwards through every time step, multiplying the Jacobian of the recurrence at each step.

### 7.2 Backpropagation Through Time (BPTT)

For a loss `L` at the final step, the gradient with respect to the initial hidden state involves:

```
∂L/∂h₀ = ∂L/∂hₜ · ∏ᵢ₌₁ᵗ ∂hᵢ/∂hᵢ₋₁
        = ∂L/∂hₜ · ∏ᵢ₌₁ᵗ (Wₕ · diag(tanh'(·)))
```

Each term in the product is the recurrent weight matrix scaled by the derivative of tanh. For sequences of length `t`, this product has `t` terms — and the problem becomes explosive.

> 🎯 **Interview prep**: "Why is BPTT expensive?" — Computing the full gradient requires materializing all hidden states in memory (O(T) space) and doing O(T) matrix multiplications. Truncated BPTT limits the unrolling depth, trading gradient accuracy for compute.

**Resources**
- [Dive into Deep Learning: RNN](https://arxiv.org/pdf/2106.11342) — rigorous derivation of BPTT

---

## 8. The Vanishing Gradient Problem

The vanishing gradient problem is the central reason vanilla RNNs cannot learn long-range dependencies. It was first formally analyzed by ([Bengio, Simard & Frasconi, 1994](https://doi.org/10.1109/72.279181)) and later given a comprehensive treatment by ([Pascanu, Mikolov & Bengio, 2013](https://arxiv.org/abs/1211.5063)).

### 8.1 Intuition

Consider predicting the last word in:

```
"I grew up in France, went to school there, learned the culture, and now I speak fluent ___"
```

The relevant information (`France`) is many time steps before the prediction site. The gradient of the loss with respect to `h_France` involves multiplying many Jacobians — one per time step in between.

If the largest eigenvalue of `Wₕ` is `λ`:
- `λ < 1` → gradient shrinks exponentially as it travels back: **vanishing gradients** (most common)
- `λ > 1` → gradient grows exponentially: **exploding gradients**

```
Gradient magnitude at step k (propagating back T steps):

   |∂L/∂h_k| ≈ λᵀ⁻ᵏ × |∂L/∂h_T|

λ = 0.9, T-k = 20:  0.9²⁰ ≈ 0.12   (12% signal remaining)
λ = 0.9, T-k = 50:  0.9⁵⁰ ≈ 0.005  (0.5% signal remaining — effectively dead)
λ = 1.1, T-k = 20:  1.1²⁰ ≈ 6.7    (exploding)
```

### 8.2 tanh Saturates

The tanh activation compounds the problem. In saturation regions (|x| large), `tanh'(x) ≈ 0`. Gradients passing through a saturated tanh are killed regardless of eigenvalue magnitude.

```
tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)
tanh'(x) = 1 - tanh²(x)

tanh'(0) = 1.0      (gradient fully preserved at center)
tanh'(2) = 0.07     (93% of gradient killed)
tanh'(3) = 0.01     (99% killed)
```

### 8.3 Solutions

1. **Gradient clipping** (for exploding gradients): clip the global gradient norm to a threshold. If `||g|| > threshold`, rescale: `g ← g × (threshold / ||g||)`. Proposed by Pascanu et al. (2013). Implemented as `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`.

2. **Gating mechanisms** (for vanishing gradients): LSTMs and GRUs replace the multiplicative recurrence with additive updates through learned gates — the core innovation that makes deep sequential learning tractable.

> 🎯 **Interview prep**: "What's the difference between vanishing and exploding gradients?" — Vanishing: model can't update early-layer weights → long-range dependencies ignored, training stalls silently. Exploding: gradients blow up to NaN → training crashes visibly. Clipping fixes exploding; gating (LSTM/GRU) or residual connections fix vanishing.

**Resources**
- [Pascanu et al., 2013](https://arxiv.org/abs/1211.5063) — gradient clipping analysis and prescription

---

## 9. Long Short-Term Memory (LSTM)

([Hochreiter & Schmidhuber, 1997](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)) introduced the LSTM to directly address vanishing gradients. The key insight is the **cell state**: a separate memory channel that flows through the network with only multiplicative (gate) interactions — no repeated matrix multiplication, no squashing activation on the main path.

Christopher Olah's [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is the definitive visual explanation; the diagrams below are adapted from his notation.

### 9.1 The Core Idea: Cell State as Conveyor Belt

A vanilla RNN's hidden state `hₜ` is fully overwritten at each step. An LSTM separates two streams:

- **Cell state `Cₜ`** — the long-term memory. Flows mostly unchanged; gates modulate small additions and subtractions. Colah describes it as "a conveyor belt running straight down the entire chain, with only some minor linear interactions."
- **Hidden state `hₜ`** — the short-term working memory, produced fresh each step.

```
Cell State (long-term memory — the "conveyor belt"):

C(t-1) ──────────×──────────────────────+──────────► C(t)
                  │                      │
            [forget gate]         [input gate × candidate]

Hidden State (output at each step):

h(t) = output_gate × tanh( C(t) )
```

The additive update `C(t) = f⊙C(t-1) + i⊙C̃` is the key: gradients flow backwards through addition unimpeded (gradient of addition is 1), not through repeated matrix multiplications.

### 9.2 The Three Gates

Every gate is a sigmoid layer (`σ`) outputting values in `[0, 1]`. A value of 0 means "block completely", 1 means "pass through completely". The input to every gate is the concatenation of the previous hidden state `hₜ₋₁` and the current input `xₜ`.

---

#### Gate 1: Forget Gate `fₜ`

*"What fraction of the old cell state should we keep?"*

```
         hₜ₋₁ ──┐
                 ├─► [ σ(Wf · [hₜ₋₁, xₜ] + bf) ] ──► fₜ ∈ [0, 1]
           xₜ ──┘
                                  ↓
                          Multiply with C(t-1):
                          fₜ ⊙ C(t-1)
                          (0 = forget everything, 1 = keep everything)
```

In a language model, when the subject changes from "Alice" to "Bob", the forget gate would push `fₜ ≈ 0` for the slot encoding the old subject's gender, erasing that information.

---

#### Gate 2: Input Gate `iₜ` + Candidate Cell State `C̃ₜ`

*"What new information should we write into the cell state?"*

Two sub-operations work together:

```
         hₜ₋₁ ──┐
                 ├─► [ σ(Wi · [hₜ₋₁, xₜ] + bi) ] ──► iₜ  ∈ [0,1]  (how much to write)
           xₜ ──┘

         hₜ₋₁ ──┐
                 ├─► [ tanh(Wg · [hₜ₋₁, xₜ] + bg) ] ─► C̃ₜ ∈ [-1,1] (what to write)
           xₜ ──┘

                          New contribution: iₜ ⊙ C̃ₜ
```

The `tanh` creates candidate values in `[-1, 1]`; the sigmoid `iₜ` decides how much of each candidate value to actually write.

---

#### Cell State Update

Putting forget and input gates together:

```
Cₜ = fₜ ⊙ Cₜ₋₁  +  iₜ ⊙ C̃ₜ
     └──────────┘    └────────┘
      selective        selective
      forgetting        writing

┌──────────────────────────────────────────────────────────────┐
│ C(t-1) ──×──────────────────────────────+──────────► C(t)  │
│          │                              │                    │
│         f(t)                        i(t)·C̃(t)              │
│         [σ]                        [σ] × [tanh]             │
│          │                              │                    │
│       [h(t-1), x(t)] ──────────── [h(t-1), x(t)]           │
└──────────────────────────────────────────────────────────────┘
```

---

#### Gate 3: Output Gate `oₜ`

*"What part of the cell state should we expose as the hidden state?"*

```
         hₜ₋₁ ──┐
                 ├─► [ σ(Wo · [hₜ₋₁, xₜ] + bo) ] ──► oₜ ∈ [0, 1]
           xₜ ──┘

         hₜ = oₜ ⊙ tanh(Cₜ)
              └──────────────────────────────────────────────┐
                                                             │
         (tanh squashes cell state to [-1,1];                │
          sigmoid output gate selects which parts to expose) │
```

### 9.3 Complete LSTM Equations

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)          ← forget gate
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)          ← input gate
C̃ₜ = tanh(Wg · [hₜ₋₁, xₜ] + bg)       ← candidate cell state
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ             ← cell state update
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)          ← output gate
hₜ = oₜ ⊙ tanh(Cₜ)                     ← hidden state (output)
```

### 9.4 Worked Numerical Example

Scalars, 1-dimensional, no bias, to show the gate mechanics:

```
Inputs: xₜ = 0.5, hₜ₋₁ = 0.3, Cₜ₋₁ = 0.8

Weights (simplified): Wf = 0.6, Wi = 0.7, Wg = 0.9, Wo = 0.5

Step 1 — Forget gate:
  fₜ = σ(Wf · [hₜ₋₁, xₜ]) = σ(0.6 × (0.3 + 0.5)) = σ(0.48) = 0.618

Step 2 — Input gate:
  iₜ = σ(Wi · [hₜ₋₁, xₜ]) = σ(0.7 × 0.8) = σ(0.56) = 0.637

Step 3 — Candidate:
  C̃ₜ = tanh(Wg × 0.8) = tanh(0.72) = 0.616

Step 4 — Cell state update:
  Cₜ = fₜ · Cₜ₋₁ + iₜ · C̃ₜ
     = 0.618 × 0.8 + 0.637 × 0.616
     = 0.494 + 0.392
     = 0.886

Step 5 — Output gate:
  oₜ = σ(Wo × 0.8) = σ(0.40) = 0.599

Step 6 — Hidden state:
  hₜ = oₜ × tanh(Cₜ) = 0.599 × tanh(0.886) = 0.599 × 0.706 = 0.423
```

The cell state went from 0.8 to 0.886 — a modest additive update, not a full overwrite. This is why gradients flow cleanly: ∂Cₜ/∂Cₜ₋₁ = fₜ (just a scalar multiplier around 0.618), not a full matrix multiplication through tanh saturation.

### 9.5 LSTM Variants

Greff et al. (2015) evaluated eight LSTM variants and found no single modification consistently outperforms the standard LSTM:

1. **Peephole connections** (Gers & Schmidhuber, 2000): gates also look at the cell state `Cₜ₋₁` directly.
2. **Coupled forget/input gates**: `iₜ = 1 - fₜ` — one decision controls both forgetting and writing.
3. **Bidirectional LSTM (BiLSTM)**: run two LSTMs, one forward, one backward; concatenate `hₜ` — captures context from both directions. Essential for sequence labeling (NER, POS tagging).
4. **Stacked LSTM**: multiple LSTM layers where output of layer `l` is input to layer `l+1`. Each layer learns higher-level abstractions.

> 🎯 **Interview prep**: "Why does adding cell state solve vanishing gradients?" — The gradient of the cell state update `∂Cₜ/∂Cₜ₋₁ = fₜ` is just a learned scalar near 1 (during normal operation), not a product of Jacobians through tanh. This creates what the paper calls a "constant error carousel" — gradients can flow back through hundreds of steps unchanged.

**Resources**
- [Hochreiter & Schmidhuber, 1997](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) — original LSTM paper
- [Colah's LSTM blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — best visual intuition

---

## 10. Gated Recurrent Unit (GRU)

([Cho et al., 2014](https://arxiv.org/abs/1406.1078)) introduced the GRU as a deliberate simplification of the LSTM. The motivation: LSTMs have 4 weight matrices and 2 separate state vectors — can we get similar performance with less?

The GRU merges the cell state and hidden state into one vector `hₜ`, and reduces three gates to two.

### 10.1 GRU Architecture

```
GRU vs LSTM — what changed:

LSTM: forget gate + input gate + output gate + cell state Cₜ + hidden state hₜ
GRU:  update gate + reset gate + hidden state hₜ only

─────────────────────────────────────────────────────────────────
LSTM cell state: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ   (additive update)
GRU hidden:      hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ (linear interpolation)
─────────────────────────────────────────────────────────────────
```

### 10.2 GRU Equations

```
zₜ = σ(Wz · [hₜ₋₁, xₜ])           ← update gate (how much to update)
rₜ = σ(Wr · [hₜ₋₁, xₜ])           ← reset gate  (how much past to forget)
h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ])   ← candidate hidden state
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ  ← hidden state update
```

**Update gate `zₜ`**: controls how much of the old hidden state to carry over vs. overwrite with the candidate. `zₜ = 0` → completely copy old state; `zₜ = 1` → completely replace with new candidate.

**Reset gate `rₜ`**: controls how much of the past hidden state to mix into the candidate. `rₜ = 0` → candidate is computed from input only (forget the past); `rₜ = 1` → full access to past hidden state.

```
GRU Data Flow:

                       hₜ₋₁ ──────────────────────────────────────┐
                         │                                          │
              ┌──────────┤                                          │
              │          │                                ┌─────────▼──────────┐
           zₜ │       rₜ │                                │   (1-z)·hₜ₋₁      │
           [σ]│       [σ]│                                │    + z·h̃ₜ         │ → hₜ
              │          │                                └─────────┬──────────┘
              │     ┌────▼───────┐                                  │
              │     │  rₜ ⊙ hₜ₋₁│                                  │
              │     └────┬───────┘                                  │
              │          │                                          │
              │     ┌────▼──────┐                                   │
              │     │ tanh(W·[·,xₜ]) ──────────────────── h̃ₜ ─────┘
              │     └───────────┘
              │
           [h(t-1), x(t)] ─────────────────────── same for both gates
```

### 10.3 GRU vs LSTM — When Does Each Win?

GRU typically matches LSTM on language modeling and translation. Where LSTM retains an edge: tasks requiring very long-range selective memory (e.g., reading a document and answering a question about its first sentence). Where GRU wins: faster training, lower memory, simpler hyperparameter tuning.

> 🎯 **Interview prep**: "What is the update gate in a GRU equivalent to in an LSTM?" — The update gate `zₜ` is the combination of the LSTM's forget gate and input gate operating in complementary fashion: when `zₜ` is high, new information is written (like LSTM input gate = 1); when `zₜ` is low, old state is preserved (like LSTM forget gate = 1). The LSTM separates these decisions; the GRU couples them.

**Resources**
- [Cho et al., 2014](https://arxiv.org/abs/1406.1078) — original GRU paper

---

## 11. Architecture Comparison Table

| | Vanilla RNN | LSTM | GRU | Transformer |
|---|---|---|---|---|
| **Memory type** | Hidden state `hₜ` | Cell `Cₜ` + hidden `hₜ` | Hidden `hₜ` | Attention over all positions |
| **Gates** | None | 3 (forget, input, output) | 2 (update, reset) | N/A (attention weights) |
| **Parameters** | `d²` (small) | `4d²` (4x more than RNN) | `3d²` | `O(d²)` per layer, but many layers |
| **Context window** | Theoretically ∞, effectively ~10 | Theoretically ∞, effectively ~200-500 | Same as LSTM | Full sequence (memory O(n²)) |
| **Training speed** | Fastest | Moderate | ~1.3x faster than LSTM | Highly parallelizable, fast on GPU |
| **Vanishing gradient** | Severe | Mostly solved | Mostly solved | Not an issue (attention is direct) |
| **Sequential dependency** | Yes — cannot parallelize | Yes | Yes | No — full parallelism |
| **Long-range deps** | Poor | Good | Good | Excellent |
| **Best for** | Very short sequences, baselines | Time series with long patterns, NLP | Speed-critical sequential tasks | NLP, long documents, generation |
| **Still used in 2026?** | Rarely | Yes (edge, time series) | Yes (edge, speed-critical) | Dominant for NLP |
| **Example use case** | Stock price regression (short) | ECG anomaly detection | Mobile keyboard prediction | ChatGPT, BERT, translation |

> 🎯 **Interview prep**: "Would you ever use an LSTM over a Transformer in production today?" — Yes: (1) streaming/real-time inputs where you can't batch the full sequence before predicting; (2) very long time series (hourly sensor data for months) where O(n²) attention is prohibitive; (3) edge deployment where a 100KB LSTM fits in SRAM but a 500MB Transformer doesn't; (4) interpretable gating — you can inspect gate activations to understand what the model is tracking.

---

## 12. Seq2Seq — Encoder-Decoder Architecture

([Sutskever, Vinyals & Le, 2014](https://arxiv.org/abs/1409.3215)) introduced the sequence-to-sequence framework that enabled end-to-end neural machine translation, summarization, and any task that maps one variable-length sequence to another.

The architecture has two components:

```
Input sequence (source):   "Je suis étudiant"
                                    │
                             ┌──────▼──────┐
                             │  ENCODER   │  (LSTM processes each input token)
                             │  LSTM      │
                             └──────┬──────┘
                                    │ final hidden state h_T (context vector)
                                    │ ← entire source sentence compressed here
                             ┌──────▼──────┐
                             │  DECODER   │  (LSTM generates output token by token)
                             │  LSTM      │
                             └──────┬──────┘
                                    │
Output sequence (target):   "I am a student"
                             (generated autoregressively, one token at a time)
```

**Encoder**: reads the source sequence and compresses it into a single fixed-length context vector (the final hidden state). No outputs are emitted during encoding.

**Decoder**: initialized with the context vector, generates the target sequence one token at a time, feeding each predicted token back as the next input (teacher forcing during training).

### 12.1 The Bottleneck Problem

The entire source sentence must be compressed into a single fixed-length vector regardless of source length. For short sentences this is fine. For long sentences (50+ words), the encoder must squeeze arbitrarily complex meaning into a vector of, say, 512 dimensions. Empirically, translation quality degrades sharply as source length increases.

```
BLEU score vs. source sentence length (Bahdanau et al., 2014):

Length 10-20:  RNNenc-dec ≈ 35 BLEU  (acceptable)
Length 30-40:  RNNenc-dec ≈ 20 BLEU  (degraded)
Length 50+:    RNNenc-dec ≈ 10 BLEU  (near failure)

With attention:  nearly flat across all lengths ← this is the motivation for attention
```

> 🎯 **Interview prep**: "What's the bottleneck problem in seq2seq, and what's the solution?" — A single fixed-size vector must represent the entire input. Attention solves this by giving the decoder access to all encoder hidden states, not just the last one, weighted by relevance to the current decoding step.

**Resources**
- [Sutskever et al., 2014](https://arxiv.org/abs/1409.3215) — seq2seq paper

---

## 13. Bahdanau Attention — Fixing the Bottleneck

([Bahdanau, Cho & Bengio, 2014](https://arxiv.org/abs/1409.0473)) replaced the single context vector with a dynamic, step-by-step weighted sum over all encoder hidden states. This is the direct ancestor of the self-attention used in Transformers.

### 13.1 Core Idea

Instead of using only the final encoder hidden state, the decoder at each step `t` looks at **all encoder hidden states** `h₁, h₂, ..., hₙ` and learns to attend to the most relevant ones:

```
At decoding step t:
  1. Score each encoder state hᵢ against current decoder state sₜ₋₁
  2. Normalize scores to get attention weights αₜᵢ (sum to 1)
  3. Weighted sum of encoder states = context vector cₜ
  4. Decoder uses cₜ (not just h_T) to generate next token

                  Encoder hidden states
         h₁    h₂    h₃    h₄    h₅
          │     │     │     │     │
          │     │     │     │     │
      ┌───▼─────▼─────▼─────▼─────▼───┐
      │    Score (alignment model)     │   ← eₜᵢ = score(sₜ₋₁, hᵢ)
      └───────────────┬────────────────┘
                      │
                   softmax
                      │
                   αₜ = [α₁, α₂, α₃, α₄, α₅]   (attention weights, sum to 1)
                      │
                   cₜ = Σᵢ αₜᵢ · hᵢ              (context vector, weighted sum)
                      │
              ┌───────▼───────┐
              │  Decoder step │  → next token
              └───────────────┘
```

### 13.2 Attention Score (Additive / Bahdanau Attention)

The alignment score between decoder state `sₜ` and encoder state `hᵢ` is computed by a small feedforward network:

```
eₜᵢ = vᵃ · tanh(Wₐ · sₜ₋₁ + Uₐ · hᵢ)    ← alignment energy

αₜᵢ = softmax(eₜᵢ) = exp(eₜᵢ) / Σⱼ exp(eₜⱼ)  ← attention weight

cₜ  = Σᵢ αₜᵢ · hᵢ                            ← context vector
```

This is called **additive attention** because the decoder and encoder states are combined by addition after linear projection. (Contrast with Luong's **multiplicative/dot-product attention**: `eₜᵢ = sₜᵀ · hᵢ`, which Transformers generalize into Q·Kᵀ/√d.)

### 13.3 Worked Example

Translating "Je suis" → "I am". Simplified 2-encoder-state example:

```
Encoder states: h₁ = [1.0, 0.0]  (encodes "Je")
                h₂ = [0.0, 1.0]  (encodes "suis")
Decoder state at t=1 (generating "I"): s₀ = [0.8, 0.2]

Alignment scores (dot product for simplicity):
  e₁ = s₀ · h₁ = 0.8×1.0 + 0.2×0.0 = 0.8
  e₂ = s₀ · h₂ = 0.8×0.0 + 0.2×1.0 = 0.2

Softmax:
  α₁ = exp(0.8) / (exp(0.8) + exp(0.2)) = 2.225 / (2.225 + 1.221) = 0.645
  α₂ = 1.221 / 3.446 = 0.355

Context vector:
  c₁ = 0.645 × [1.0, 0.0] + 0.355 × [0.0, 1.0] = [0.645, 0.355]
```

The decoder at step 1 (generating "I", which corresponds to "Je") attends heavily to `h₁` (weight 0.645) — the model has learned that the first French word aligns with the first English word.

### 13.4 Why Attention Is the Foundation of Transformers

The Transformer (Vaswani et al., 2017 — "Attention Is All You Need") removes the recurrence entirely and uses self-attention: every position attends to every other position simultaneously. This enables full parallelism during training and captures arbitrary-range dependencies in a single layer. The Q/K/V formulation is a parametric, scaled extension of dot-product attention.

```
Bahdanau → Luong → Self-Attention → Multi-Head Attention → Transformer

Bahdanau:  eₜᵢ = vᵀ·tanh(Wₐ·s + Uₐ·h)   (additive, O(n))
Dot-prod:  eₜᵢ = sₜᵀ·hᵢ                   (multiplicative, faster)
Scaled:    eₜᵢ = (Q·Kᵀ)/√d_k             (what Transformers use)
```

> 🎯 **Interview prep**: "What's the difference between Bahdanau and Luong (dot-product) attention?" — Bahdanau: decoder and encoder states combined via a learned additive network, more expressive but more parameters. Luong: simple dot product between decoder state and encoder states, faster, generalizes easily to self-attention. Scaled dot-product attention divides by √d_k to prevent vanishing gradients when d_k is large.

**Resources**
- [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) — original attention paper

---

## 14. Production Notes — When to Use These in 2026

The honest answer: for most NLP classification tasks, a fine-tuned BERT-class model will beat TF-IDF + LSTM. But the world is not only classification tasks at Google-scale data centers.

### Still actively used

**TF-IDF + classical ML**:
- Document retrieval pipelines in search engines (BM25 variant, Elasticsearch default).
- Lightweight text classification where training data is scarce and a 1M-parameter BERT is overkill.
- Interpretable features for compliance/legal NLP where model decisions must be explainable.

**LSTMs and GRUs**:
- **Streaming / real-time inference**: LSTMs process one token at a time and emit an output immediately. Transformers need the full context. This matters for real-time fraud scoring, live caption generation, and industrial sensor monitoring.
- **Edge/embedded devices**: an LSTM for keyboard prediction weighs ~1 MB. A small Transformer is 50–200 MB.
- **Time-series forecasting**: LSTM-based models (e.g., DeepAR) outperform Transformers on many time-series benchmarks with irregular sampling.
- **Long time series** where O(n²) Transformer attention is prohibitive — hourly sensor data for 3 months = 2160 steps; self-attention is expensive.

**Bidirectional LSTMs (BiLSTMs)**:
- Still used inside token-level NLP models for NER and POS tagging in latency-sensitive pipelines.
- spaCy's medium English model (`en_core_web_md`) uses a CNN/BiLSTM stack, not a Transformer.

### Common failure modes to know for interviews

1. **Gradient exploding during training**: symptoms are NaN loss. Fix: clip gradient norm (`torch.nn.utils.clip_grad_norm_`).
2. **LSTM forgetting relevant context**: check if the sequence is too long and the forget gate is collapsing (all near 0). Fix: truncate sequences, stack layers, or switch to attention.
3. **Slow convergence with small batches**: LSTMs do not benefit from batch normalization (sequential dependency). Use layer normalization instead.
4. **Overfitting on small datasets**: LSTMs have 4x the parameters of vanilla RNNs. Use higher dropout (variational dropout on hidden state, not just on inputs).

---

## 15. Interview Q&A

### Q1: Explain TF-IDF and when you'd use it over embeddings.

**A**: TF-IDF is a term-weighting scheme that scores each word `w` in document `d` by its local frequency (TF) multiplied by the log inverse of how many documents contain `w` (IDF). This down-weights common words and up-weights discriminative words. It produces sparse, high-dimensional vectors (one dimension per vocabulary word).

Use TF-IDF when: training data is small (embeddings need more data to generalize), interpretability matters (you can inspect which terms drove a decision), the task is retrieval or keyword matching rather than semantic similarity, or inference latency requirements preclude embedding lookup.

Use embeddings when: semantic similarity matters (synonyms should be close), data is abundant, or the task requires understanding paraphrases and context.

---

### Q2: Why can't vanilla RNNs learn long-range dependencies?

**A**: During backpropagation, gradients must flow through every time step between the loss and the relevant input. Each step multiplies by the recurrent weight matrix `Wₕ` and the derivative of tanh. If the dominant eigenvalue of `Wₕ` is less than 1, this product shrinks exponentially. After 20-50 steps, the gradient reaching early time steps is numerically zero — the model cannot update weights to learn from distant context. This is the vanishing gradient problem (Bengio et al., 1994).

---

### Q3: Walk through the three LSTM gates with a concrete example.

**A**: Consider a language model reading "Alice was hungry. She ate... **the**". At the word "the", the model must predict "food"/"sandwich"/etc., using "hungry" from 3 steps back.

- **Forget gate**: when reading "Alice was hungry", the forget gate keeps the "subject=Alice" slot in cell state. When a new sentence starts (period), forget gate ≈ 0 for subject slot.
- **Input gate**: when reading "hungry", input gate ≈ 1 for the "state=hungry" dimension, writing this into cell state.
- **Output gate**: when generating the next word prediction, output gate selects the "hunger-related food" dimensions from cell state to produce `hₜ`, which influences the output softmax.

---

### Q4: What's the difference between GRU and LSTM, and when would you choose each?

**A**: GRU merges the forget and input gates into a single update gate `zₜ`, and removes the separate cell state (hidden state serves both roles). This reduces parameter count by 25% and speeds up training.

Choose GRU when: compute budget is tight, sequence lengths are moderate, or you want faster iteration. Choose LSTM when: the task requires fine-grained control over what to remember vs. forget (rare but sometimes important), when the dataset is large enough to benefit from the additional expressivity, or when using transfer learning from pre-trained LSTM weights.

In practice for NLP, if you're choosing between LSTM and GRU, you should probably just use a pre-trained Transformer instead.

---

### Q5: Explain the seq2seq bottleneck and how attention solves it.

**A**: In the original seq2seq model (Sutskever et al., 2014), the entire source sequence is compressed into the encoder's final hidden state — a fixed-size vector (e.g., 512 dimensions) regardless of source length. For long sentences, this bottleneck causes information loss and BLEU scores drop sharply with increasing source length.

Bahdanau attention (2014) solves this by retaining all encoder hidden states `h₁...hₙ` and, at each decoding step `t`, computing a dynamic context vector as a weighted sum of all encoder states. The weights (attention scores `αₜᵢ`) are learned: the model learns which source positions are most relevant for each decoding position. This removes the compression constraint entirely.

---

### Q6: Why does the LSTM cell state solve vanishing gradients but the hidden state doesn't?

**A**: The cell state update is additive: `Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ`. The gradient of `Cₜ` with respect to `Cₜ₋₁` is just `fₜ` — a scalar learned to be near 1 when the memory should persist. No matrix multiplication, no tanh squashing on the backward path. Information can flow unchanged for hundreds of steps (constant error carousel).

The hidden state `hₜ = oₜ ⊙ tanh(Cₜ)` still uses tanh and gating, but it's not on the critical gradient path between distant time steps — that path goes through the cell state.

---

### Q7: What is teacher forcing and what problem does it cause?

**A**: During training, teacher forcing feeds the ground-truth previous token as input to the decoder at each step, rather than the model's own (possibly wrong) prediction. This makes training stable and fast.

The problem: at inference time, the decoder uses its own predictions as inputs. If the model makes an error at step 3, step 4 receives a wrong input — a distribution shift the model was never trained on. This is called **exposure bias**. Solutions include scheduled sampling (gradually replacing ground truth with model predictions during training) or using a beam search with diverse candidates.

---

### Q8: What is BM25 and how does it improve on TF-IDF?

**A**: BM25 (Best Match 25, Robertson et al.) is a probabilistic extension of TF-IDF used in Elasticsearch, Lucene, and most production search engines. It adds two improvements:

1. **TF saturation**: BM25 uses `TF/(TF + k₁)` instead of raw TF. After a word appears a few times in a document, additional occurrences contribute diminishing returns (k₁ ≈ 1.2–2.0 typically).
2. **Document length normalization**: short documents with a relevant word should score higher than long documents with the same count. BM25 divides by a length normalization factor controlled by parameter `b` (b=0.75 typically).

In practice, BM25 consistently outperforms raw TF-IDF on retrieval benchmarks.

---

### Q9: What is the GLUE benchmark and what does it measure?

**A**: GLUE (General Language Understanding Evaluation, Wang et al. 2018) is a collection of 9 NLP tasks including sentiment analysis (SST-2), textual entailment (MNLI), question answering (QNLI), paraphrase detection (MRPC/QQP), and grammaticality (CoLA). It was introduced to measure general NLP capability rather than task-specific optimization.

Models report a single GLUE score (average across tasks). BERT (88.5) and RoBERTa (90.2) exceeded human performance (87.1) on GLUE. SuperGLUE was then introduced as a harder successor.

---

### Q10: How does gradient clipping work and when do you use it?

**A**: Gradient clipping rescales the global gradient norm when it exceeds a threshold. The global norm is `||g|| = sqrt(Σᵢ gᵢ²)`. If `||g|| > max_norm`, all gradient components are multiplied by `max_norm / ||g||`.

This preserves the gradient direction while preventing magnitude explosions. It's standard practice when training RNNs/LSTMs. In PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`, called between `loss.backward()` and `optimizer.step()`.

---

## 16. References

### Foundational Papers

| Paper | Year | Key Contribution |
|---|---|---|
| [Salton & Buckley, "Term Weighting Approaches"](https://www.researchgate.net/publication/228572832) | 1988 | TF-IDF formalization |
| [Bengio, Simard & Frasconi, IEEE Trans. Neural Networks](https://doi.org/10.1109/72.279181) | 1994 | Vanishing gradient problem analysis |
| [Hochreiter & Schmidhuber, Neural Computation](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) | 1997 | Long Short-Term Memory |
| [Mikolov et al., arXiv:1301.3781](https://arxiv.org/abs/1301.3781) | 2013 | Word2Vec (CBOW + Skip-gram) |
| [Pascanu, Mikolov & Bengio, arXiv:1211.5063](https://arxiv.org/abs/1211.5063) | 2013 | Gradient clipping, RNN training |
| [Cho et al., arXiv:1406.1078](https://arxiv.org/abs/1406.1078) | 2014 | GRU + RNN Encoder-Decoder |
| [Bahdanau, Cho & Bengio, arXiv:1409.0473](https://arxiv.org/abs/1409.0473) | 2014 | Additive attention mechanism |
| [Sutskever, Vinyals & Le, arXiv:1409.3215](https://arxiv.org/abs/1409.3215) | 2014 | Sequence-to-sequence learning |
| [Pennington, Socher & Manning, ACL 2014](https://aclanthology.org/D14-1162/) | 2014 | GloVe word vectors |

### Blogs and Tutorials

| Resource | What it covers |
|---|---|
| [Colah's Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Best visual explanation of LSTM gates |
| [Stanford IR Book Chapter 6](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf) | TF-IDF and vector space model |
| [Dive into Deep Learning: RNN](https://arxiv.org/pdf/2106.11342) | BPTT derivation and implementation |
| [GloVe project page](https://nlp.stanford.edu/projects/glove/) | Pre-trained vectors + paper |

### Datasets (HuggingFace)

| Dataset | Task | Link |
|---|---|---|
| `stanfordnlp/sst` | Sentiment analysis (5-class) | [HF link](https://huggingface.co/datasets/stanfordnlp/sst) |
| `imdb` | Binary sentiment | [HF link](https://huggingface.co/datasets/imdb) |
| `Salesforce/wikitext` | Language modeling | [HF link](https://huggingface.co/datasets/Salesforce/wikitext) |

### Benchmarks

| Benchmark | Year | What it measures | Link |
|---|---|---|---|
| GLUE ([Wang et al., arXiv:1804.07461](https://arxiv.org/abs/1804.07461)) | 2018 | General NLU across 9 tasks | [gluebenchmark.com](https://gluebenchmark.com/) |
| SuperGLUE | 2019 | Harder NLU tasks post-BERT saturation | [super.gluebenchmark.com](https://super.gluebenchmark.com/) |

---

*This blog covers the full pre-Transformer NLP stack. The natural next step is [11-embedding-models](../11-embedding-models/blog-embedding-models.md) (contextual embeddings, sentence transformers) and then [14-llm](../14-llm/blog-llm.md) (Transformer architecture in depth).*
