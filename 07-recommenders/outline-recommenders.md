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

## 16 — Graph Neural Networks for RecSys
Most popular GNN recsys: LightGCN (simplified GCN — no feature transformation, only neighborhood aggregation; state-of-art on implicit feedback); NGCF (Neural Graph CF — adds non-linearity); PinSage (GraphSAGE + random-walk sampling, rich item features, Pinterest scale); user-item bipartite graph construction; multi-hop propagation; LightGCN + BPR loss is the default production GNN baseline; SotA: UltraGCN, SimGCL (contrastive augmentation).
- https://arxiv.org/abs/2002.02126 (LightGCN)
- https://arxiv.org/abs/1806.01973 (PinSage)
- https://github.com/recommenders-team/recommenders

## 17 — Knowledge Graph Embeddings for Recommendations
Entity/relation embeddings (TransE, RotatE); propagate user preferences over KG paths; KGAT (Knowledge Graph Attention Network).
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

## 21 — Evaluation Metrics: Accuracy
Precision@K, Recall@K, NDCG@K (graded relevance, position-aware; most used in industry), MAP, MRR, Hit Rate@K; RMSE/MAE for explicit ratings; how to compute each formula; online vs offline gap; Jurity for computing NDCG and fairness metrics in one library.
- https://eugeneyan.com/writing/evaluation-metrics-for-recommender-systems/
- https://jurity.readthedocs.io/en/latest/

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

## 27 — Multi-Task Learning in RecSys
Shared bottom + task-specific towers; MMoE (Mixture of Experts); PLE (Progressive Layered Extraction); YouTube multi-objective ranking; calibrating engagement vs satisfaction; loss weighting strategies.
- https://arxiv.org/abs/1904.05862
- https://dl.acm.org/doi/10.1145/3383313.3412236

## 28 — LLM-Based Recommendations
LLMRec; prompting LLMs for zero-shot recommendations; LLM as feature encoder for cold-start; instruction tuning on interaction history; trade-offs vs traditional two-tower at scale.
- https://arxiv.org/abs/2307.15780
- https://arxiv.org/abs/2305.07001

## 29 — Contextual Bandits in Production
LinGreedy (ε-greedy with linear reward model); LinUCB (upper confidence bound); Thompson Sampling; MABWiser — 9 learning policies (EpsilonGreedy, LinTS, LinUCB, UCB1, Thompson, Softmax) + 5 neighborhood policies (KNearest, LSH, Radius, TreeBandit, Clusters) for contextual variants; parallel MAB support; Simulator for policy comparison and HPO; Mab2Rec wraps MABWiser — train()/score() interface, benchmark() for algorithm comparison, outputs ranked DataFrames; offline evaluation via replay and IPS; Jurity for NDCG and fairness metrics on bandit outputs; when bandits outperform two-tower (sparse data, rapidly-changing catalog, cold-start); comparison: bandits vs RL vs A/B testing.
- https://github.com/fidelity/mabwiser
- https://fidelity.github.io/mabwiser/
- https://github.com/fidelity/mab2rec
- https://arxiv.org/abs/2012.01780

## 30 — Dataset Preparation per Algorithm
How input data format differs across algorithm families:
- **CF / MF / ALS**: (user_id, item_id, rating/count) triples; explicit (star ratings) vs implicit (clicks, plays); confidence weighting for implicit; temporal holdout split; MovieLens, Amazon Reviews, LastFM datasets.
- **Content-based**: item feature matrix (TF-IDF text, categorical OHE, numerical features); user profile built from history; no user features needed at train time.
- **Two-tower / NCF**: (user, pos_item, neg_item) triples; in-batch negatives; hard negative mining from popular items; feature encoding for user/item side; MIND news, Criteo, Kuaishou datasets.
- **Sequential (SASRec, BERT4Rec)**: ordered interaction sequences per user sorted by timestamp; sliding window augmentation for long sequences; filter users/items below min-interaction threshold; max sequence length cap.
- **Bandits (MABWiser/Mab2Rec)**: (context_vector, arm/item_id, reward) rows where context = user feature vector; context-free variant needs only (arm, reward); reward typically binary (click/purchase); warm-start from historical logs.
- **Graph (LightGCN, PinSage)**: user-item bipartite edge list; sparse adjacency matrix; degree-normalized Laplacian; PinSage needs rich item node features (images/text embeddings); LightGCN works with graph structure alone.
- **Microsoft Recommenders repo**: reference implementations and dataset loaders for MovieLens, MIND, Amazon, Criteo.
- https://github.com/recommenders-team/recommenders
- https://github.com/fidelity/mab2rec

## 31 — Inference & Serving Patterns
How recommendations are actually generated at serving time differs fundamentally by model type:
- **Two-tower**: offline batch — generate all item embeddings; store in FAISS/ScaNN; online — compute user embedding in real time, ANN lookup; embedding refresh cadence (hourly/daily).
- **Matrix factorization / ALS**: precompute user+item factor matrices; lookup by user_id; batch rescore nightly; no real-time model serving.
- **Re-ranking (LambdaMART / DNN ranker)**: low-latency pointwise scoring on candidate set from retrieval; feature joins from feature store at request time.
- **Sequential (SASRec)**: online inference with last-N item history as input; cache attention KV for frequent users.
- **Bandits (MABWiser/Mab2Rec)**: `.score()` at request time with current context vector; periodic `.train()` or warm-start updates; no GPU needed.
- **LightGCN**: offline — propagate graph, compute user/item embeddings; serve like two-tower; graph incremental update strategies.
- Key production concerns: SLA (< 50 ms for ranking), stale embedding risk, feature skew between training and serving.
- https://eugeneyan.com/writing/system-design-for-recommendations-and-search/
- https://arxiv.org/abs/2309.06180

## 32 — Beyond-Accuracy Evaluation: Diversity, Fairness & Jurity
Accuracy alone misses user experience and fairness: catalog coverage (% of items ever recommended); intra-list diversity (avg pairwise distance in embedding space); novelty (popularity-adjusted surprise); serendipity (unexpectedly relevant); fairness metrics — demographic parity, equalized odds across user groups, provider fairness (item exposure); Jurity library covers all of these + NDCG, CTR, MAP in one API; offline vs online gap (high-NDCG models can hurt long-term retention); evaluation dimensions: accuracy, diversity, fairness, novelty, coverage.
- https://jurity.readthedocs.io/en/latest/
- https://arxiv.org/abs/2107.14601
- https://dl.acm.org/doi/10.1145/2792838.2800183
