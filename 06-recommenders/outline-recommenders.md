# 04 — Recommender Systems

Exhaustive learning path from classic collaborative filtering to production two-stage neural recsys.

---

## 01 — Types of Recommender Systems
Collaborative filtering, content-based, hybrid, knowledge-based, session-based; when to use each.
- https://developers.google.com/machine-learning/recommendation

## 02 — User-Based Collaborative Filtering
User-user cosine similarity; k-NN neighborhood; weighted average of neighbor ratings; sparsity problems.
- https://surprise.readthedocs.io/en/stable/getting_started.html
- https://developers.google.com/machine-learning/recommendation/collaborative/basics

## 03 — Item-Based Collaborative Filtering
Item-item similarity; adjusted cosine similarity; more stable than user-based; Amazon's approach.
- https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf

## 04 — Matrix Factorization: SVD / FunkSVD
Latent factor model; SGD training; user/item biases; regularization; Surprise SVD.
- https://surprise.readthedocs.io/en/stable/matrix_factorization.html
- https://arxiv.org/abs/0907.2648

## 05 — NMF (Non-Negative Matrix Factorization)
Non-negative constraints → interpretable topics; sklearn NMF; parts-based decomposition.
- https://scikit-learn.org/stable/modules/decomposition.html#nmf

## 06 — ALS for Implicit Feedback
Confidence weighting on clicks/plays/views; weighted regularized matrix factorization; implicit library.
- http://yifanhu.net/PUB/cf.pdf
- https://implicit.readthedocs.io/en/latest/

## 07 — BPR (Bayesian Personalized Ranking)
Pairwise training objective for implicit feedback; negative sampling; maximize observed > unobserved.
- https://arxiv.org/abs/1205.2618

## 08 — Content-Based Filtering
TF-IDF / embedding item features; user profile from history; cosine similarity; cold-start advantage.
- https://developers.google.com/machine-learning/recommendation/content-based/basics

## 09 — Hybrid Recommenders & LightFM
Weighted, switching, cascade hybrids; LightFM joint embedding of content + collab signals.
- https://making.lyst.com/lightfm/docs/home.html

## 10 — Neural Collaborative Filtering (NCF)
Generalize dot product with MLP; embed user/item IDs; train on implicit feedback; BCEloss.
- https://arxiv.org/abs/1708.05031

## 11 — Wide & Deep Learning
Memorization (wide / linear) + generalization (deep DNN); feature crosses; Google Play RecSys.
- https://arxiv.org/abs/1606.07792

## 12 — DeepFM
FM layer for second-order interactions + DNN for higher-order; no manual feature engineering.
- https://arxiv.org/abs/1703.04247

## 13 — Two-Tower Models (Dual Encoder)
User tower + item tower; in-batch negatives; hard negative mining; ANN retrieval at serving time.
- https://research.google/pubs/pub48840/
- https://www.tensorflow.org/recommenders/examples/basic_retrieval

## 14 — Sequential Recommendations: SASRec
Self-attention over ordered item history; causal masking; next-item prediction.
- https://arxiv.org/abs/1808.09781

## 15 — Sequential Recommendations: BERT4Rec
Bidirectional transformer; cloze task (mask random items); fine-tune for next-item prediction.
- https://arxiv.org/abs/1904.06690

## 16 — Graph-Based Recommendations: PinSage
GraphSAGE on item-item graph; random-walk neighbor sampling; Pinterest production deployment.
- https://arxiv.org/abs/1806.01973

## 17 — Knowledge Graph Embeddings for Recommendations
Entity/relation embeddings (TransE, RotatE); propagate user preferences over KG paths.
- https://arxiv.org/abs/1905.08049

## 18 — Bandit Algorithms: Exploration-Exploitation
ε-greedy, UCB1, Thompson Sampling; contextual bandits; LinUCB; personalized exploration.
- https://arxiv.org/abs/1904.10040
- https://vowpalwabbit.org/

## 19 — Candidate Retrieval with ANN (FAISS / ScaNN)
IVF-PQ, HNSW for billion-scale ANN; latency vs recall trade-off; embedding refresh cadence.
- https://faiss.ai/
- https://github.com/google-research/google-research/tree/master/scann

## 20 — Re-Ranking Layer
Pointwise (cross-entropy), pairwise (BPR, RankNet), listwise (ListNet, LambdaMART); business rule fusion.
- https://arxiv.org/abs/1212.0702 (BPR)
- https://en.wikipedia.org/wiki/Learning_to_rank

## 21 — Evaluation Metrics
Precision@K, Recall@K, NDCG@K, MAP, MRR, Hit Rate; online vs offline gap; catalog coverage.
- https://eugeneyan.com/writing/evaluation-metrics-for-recommender-systems/

## 22 — Cold Start Strategies
New user: popularity/trending, onboarding quiz; new item: content-based, injection into explore slots.
- https://dl.acm.org/doi/10.1145/3109859.3109862

## 23 — Diversity, Novelty & Serendipity
Intra-list diversity; maximal marginal relevance (MMR); novelty filtering; reducing popularity bias.
- https://dl.acm.org/doi/10.1145/2792838.2800183

## 24 — Debiasing: Popularity & Exposure Bias
Inverse propensity scoring (IPS); unbiased LTR; position bias in click logs; counterfactual learning.
- https://arxiv.org/abs/1602.05352

## 25 — A/B Testing for Recommenders
Interleaving tests; holdout design; metric selection (CTR, dwell time, long-term retention).
- https://exp-platform.com/

## 26 — Production Architecture
Two-stage pipeline (retrieval → ranking); feature stores; real-time vs batch serving; monitoring drift.
- https://eugeneyan.com/writing/system-design-for-recommendations-and-search/
