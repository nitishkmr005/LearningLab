# 07 — Recommender Systems

Production-first study outline for recommendation systems, with explicit links to statistics, bias, ranking, and online evaluation.

---

## Format Used In This Outline
- `Concept`: what to learn.
- `Why it matters`: where it is used.
- `Production example`: how it appears in a real recommender.

## 01 — Problem Framing
- `Concept`: retrieval vs ranking vs re-ranking; explicit vs implicit feedback; short-term vs long-term objective.
- `Why it matters`: not all recommendation problems are "predict the next click".
- `Production example`: course recommendation may optimize enrollment, watch time, completion, and satisfaction differently.

## 02 — Statistics That Matter Most In Recommenders
- `Concept`: CTR, CVR, dwell time, watch time, NDCG, coverage, novelty, calibration, popularity share.
- `Why it matters`: recommendation is as much measurement as modeling.
- `Production example`: a model can improve CTR but worsen coverage and long-tail discovery.

## 03 — Basic Probability In Recommendation
- `Concept`: `P(click | user, item, context)`, `P(purchase | exposure)`, conditional probability and exposure bias.
- `Why it matters`: the model only sees outcomes after exposure, which makes naive learning biased.
- `Production example`: an unseen item has no clicks not because it is bad, but because it was never shown.

## 04 — Collaborative Filtering
- `Concept`: user-user and item-item similarity, matrix factorization.
- `Why it matters`: still strong baselines for many applications.
- `Production example`: recommend similar courses based on co-enrollment or co-watch behavior.

## 05 — Implicit Feedback Statistics
- `Concept`: clicks, views, plays, skips, add-to-cart, dwell time; confidence weighting.
- `Why it matters`: most production recommenders do not have explicit star ratings.
- `Production example`: a 5-second view and a 30-minute watch should not carry the same signal.

## 06 — Two-Stage Architecture
- `Concept`: candidate retrieval first, then ranking, then business-rule re-ranking.
- `Why it matters`: this is the common production design.
- `Production example`: retrieve 500 candidates with embeddings, rank top 50 with richer features, re-rank final 10 for diversity and policy rules.

## 07 — Feature Engineering For Ranking Models
- `Concept`: user features, item features, user-item interaction features, freshness, popularity, context features.
- `Why it matters`: even neural recommenders depend heavily on good features.
- `Production example`: user's last 7-day category mix, item's 30-day CTR, user-item category affinity, hour-of-day.

## 08 — Popularity Bias and Exposure Bias
- `Concept`: popular items receive more exposure, therefore more clicks, which reinforces popularity.
- `Why it matters`: this can crowd out relevant niche items and make offline metrics misleading.
- `Production example`: the home page keeps recommending already-popular items because historical logs are biased toward them.

## 09 — Propensity and Debiasing
- `Concept`: inverse propensity weighting, counterfactual evaluation, position bias correction.
- `Why it matters`: raw logged clicks are not unbiased labels.
- `Production example`: item rank position is a confounder, so CTR alone overstates quality of top slots.

## 10 — A/B Testing For Recommenders
- `Concept`: randomized experiments, guardrail metrics, long-term effects, novelty effects.
- `Why it matters`: online testing is the final judge for recommendation quality.
- `Production example`: a new ranking model improves CTR in week 1, but retention drops after week 3 because diversity collapsed.

## 11 — Offline vs Online Metrics
- `Concept`: NDCG and Recall@K are offline; CTR and conversion are online; they are related but not identical.
- `Why it matters`: teams often overtrust offline wins.
- `Production example`: a model with slightly worse offline NDCG may still win online if it improves freshness and relevance under real UI constraints.

## 12 — Calibration In Recommenders
- `Concept`: predicted probabilities should match observed probabilities within score bands.
- `Why it matters`: calibrated scores help budget allocation, triggering, and exploration policies.
- `Production example`: if score band `0.7 to 0.8` only clicks at `0.3`, business planning based on expected lifts breaks.

## 13 — Diversity, Novelty, and Serendipity
- `Concept`: beyond-accuracy metrics that improve user experience.
- `Why it matters`: a good recommender should not feel repetitive and narrow.
- `Production example`: mix highly relevant items with a controlled amount of exploration or long-tail content.

## 14 — Fairness In Recommenders
- `Concept`: user fairness, provider fairness, exposure fairness, popularity fairness.
- `Why it matters`: recommendation systems can systematically underexpose creators, products, or user groups.
- `Production example`: a marketplace recommender overexposes already-large sellers and starves smaller sellers of impressions.

## 15 — Sequential Recommendation
- `Concept`: model order and recency of interactions.
- `Why it matters`: next-item prediction often depends more on recent actions than on old global preference.
- `Production example`: a user watching interview-prep content this week should see different recommendations than their historic preference profile alone suggests.

## 16 — Bandits and Exploration
- `Concept`: epsilon-greedy, UCB, Thompson sampling, contextual bandits.
- `Why it matters`: recommenders need exploration to learn about new users, new items, and uncertain candidates.
- `Production example`: reserve one slot for exploration so the system learns whether a cold-start item is actually attractive.

## 17 — Recommended Statistical Review For Any Recommender Launch
- `Check 1`: overall CTR and CVR.
- `Check 2`: by-position metrics.
- `Check 3`: by-user segment performance.
- `Check 4`: popularity concentration and catalog coverage.
- `Check 5`: calibration by score bucket.
- `Check 6`: long-tail item exposure.

## 18 — Practical Build Order
- `Step 1`: define business objective and counter-metric.
- `Step 2`: choose base candidate generation method.
- `Step 3`: build ranking features and time-aware splits.
- `Step 4`: evaluate offline ranking metrics and calibration.
- `Step 5`: check popularity bias, diversity, and fairness slices.
- `Step 6`: run online A/B test with guardrails.

## References
- Google recommendation guides: https://developers.google.com/machine-learning/recommendation
- popularity bias survey: https://link.springer.com/article/10.1007/s11257-024-09406-0
- Google fairness overview: https://developers.google.com/machine-learning/guides/intro-responsible-ai/fairness
