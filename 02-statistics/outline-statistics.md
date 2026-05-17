# 02 — Statistics

Production-first study outline for statistics in machine learning, experimentation, and recommender systems.

---

## Format Used In This Outline
- `Concept`: what you need to understand.
- `Where it is used`: ML, data science, A/B testing, or recommender-system usage.
- `Production example`: the kind of business problem where it shows up.
- `A/B angle`: how the same idea appears in experimentation.

## 01 — Descriptive Statistics
- `Concept`: mean, median, mode, variance, standard deviation, percentiles, IQR, skewness, kurtosis.
- `Where it is used`: feature profiling, anomaly checks, drift detection, dashboarding.
- `Production example`: before training a credit-risk model, compare mean income, median income, and 95th percentile income to see whether a few extreme users will distort scaling or model behavior.
- `A/B angle`: compare average order value, but also median order value when the metric has heavy tails.

## 02 — Probability Basics
- `Concept`: events, conditional probability, independence, Bayes' theorem.
- `Where it is used`: probabilistic models, classifier outputs, Bayesian reasoning, recommender ranking.
- `Production example`: probability that a user clicks a recommendation given they opened the app in the evening and saw the widget.
- `A/B angle`: probability of conversion conditioned on treatment vs control.

## 03 — Common Distributions
- `Concept`: Bernoulli, Binomial, Poisson, Normal, Log-normal, Exponential, Beta, Gamma.
- `Where it is used`: label generation, count modeling, Bayesian priors, confidence intervals.
- `Production example`: purchases per day often look count-like, so Poisson intuition is useful; conversion-rate priors in Bayesian A/B testing are often Beta.
- `A/B angle`: user converts or not is Bernoulli; total conversions in a cohort are Binomial.

## 04 — Sampling, CLT, and Standard Error
- `Concept`: sample vs population, sampling variability, central limit theorem, standard error.
- `Where it is used`: uncertainty estimation, experiment readouts, model metric confidence bands.
- `Production example`: your model's precision is 0.62 on a validation sample, but the standard error tells you how noisy that estimate is.
- `A/B angle`: the difference between two observed conversion rates needs a standard error before you call it a win.

## 05 — Confidence Intervals
- `Concept`: interval estimate around a mean, proportion, lift, or model metric.
- `Where it is used`: experiment readouts, offline model comparison, calibration review.
- `Production example`: report ROC-AUC as `0.78 +/- uncertainty` instead of just `0.78`.
- `A/B angle`: if treatment lift confidence interval crosses zero, the launch decision is weak.

## 06 — Hypothesis Testing
- `Concept`: null hypothesis, alternative, p-value, Type I error, Type II error.
- `Where it is used`: A/B testing, feature screening, statistical validation.
- `Production example`: test whether a new fraud model reduces false positives without hurting recall.
- `A/B angle`: this is the base language of fixed-horizon online experiments.

## 07 — t-Tests, z-Tests, and Non-Parametric Tests
- `Concept`: one-sample, two-sample, paired tests; Mann-Whitney and Wilcoxon when assumptions break.
- `Where it is used`: metric comparison, model benchmarking, experiment readouts.
- `Production example`: compare average latency before and after a model-serving change.
- `A/B angle`: paired tests are useful when the same users or entities are observed before and after an intervention.

## 08 — Chi-Square and Categorical Statistics
- `Concept`: contingency tables, chi-square test, expected counts, Cramer's V.
- `Where it is used`: feature-target dependence checks, bias audits, recommender exposure analysis.
- `Production example`: test whether product-category exposure differs by user segment more than chance would suggest.
- `A/B angle`: check whether treatment assignment is independent of device type or geography.

## 09 — Correlation, Covariance, and Association
- `Concept`: Pearson, Spearman, Kendall, covariance, partial correlation.
- `Where it is used`: feature pruning, multicollinearity analysis, ranking diagnostics.
- `Production example`: two demand features are both high-cardinality proxies for the same behavior; correlation pruning avoids unstable importances.
- `A/B angle`: correlated guardrail metrics can move together even if only one matters causally.

## 10 — Regression Statistics
- `Concept`: OLS, coefficient interpretation, residuals, heteroscedasticity, p-values, confidence intervals.
- `Where it is used`: interpretable baselines, causal adjustment, elasticity estimation.
- `Production example`: estimate how price, discount, and seasonality relate to purchase probability before you jump to boosted trees.
- `A/B angle`: CUPED-style variance reduction and covariate adjustment build on regression ideas.

## 11 — Logistic Regression Statistics
- `Concept`: log-odds, maximum likelihood, odds ratio, calibration.
- `Where it is used`: binary classification, scorecards, interpretable baselines.
- `Production example`: churn propensity model where every coefficient can be explained to business stakeholders.
- `A/B angle`: treatment effect can be modeled with treatment and interaction terms.

## 12 — Bayesian Statistics
- `Concept`: prior, likelihood, posterior, conjugate priors, posterior predictive reasoning.
- `Where it is used`: Bayesian A/B tests, hierarchical recommenders, uncertainty-aware decision systems.
- `Production example`: low-traffic campaigns need shrinkage so noisy groups do not look falsely strong.
- `A/B angle`: Beta-Binomial updating is a clean way to compare conversion rates continuously.

## 13 — Bootstrap and Permutation Testing
- `Concept`: resampling for uncertainty and assumption-light testing.
- `Where it is used`: confidence intervals for custom metrics, lift curves, recommender metrics like NDCG.
- `Production example`: bootstrap the top-decile lift of a response model instead of pretending it is normally distributed.
- `A/B angle`: permutation tests are useful when metric distributions are ugly or sample sizes are uneven.

## 14 — Effect Size and Power
- `Concept`: Cohen's d, uplift magnitude, minimum detectable effect, power analysis.
- `Where it is used`: experiment planning and launch prioritization.
- `Production example`: a 0.1% CTR gain may be statistically significant but economically meaningless.
- `A/B angle`: decide experiment duration based on baseline rate, desired MDE, and traffic.

## 15 — Multiple Testing
- `Concept`: Bonferroni, Holm, Benjamini-Hochberg, false discovery rate.
- `Where it is used`: feature screening, many-slice analysis, many-metric experimentation.
- `Production example`: if you compare 200 features to the target, some will look significant by luck.
- `A/B angle`: if a product team checks ten variants and fifteen secondary metrics, correction matters.

## 16 — Missing Data Statistics
- `Concept`: MCAR, MAR, MNAR, missingness indicators, imputation bias.
- `Where it is used`: feature engineering, data quality pipelines, fairness review.
- `Production example`: income missingness itself may be predictive of churn or fraud.
- `A/B angle`: telemetry loss can bias experiment readouts if missingness differs by treatment.

## 17 — Outlier Statistics
- `Concept`: z-score, robust z-score, IQR fences, winsorization, heavy tails.
- `Where it is used`: preprocessing, anomaly handling, metric stabilization.
- `Production example`: a few corporate customers with extreme spend can dominate average revenue models.
- `A/B angle`: revenue metrics often need robust summaries because a few whales distort means.

## 18 — Calibration Statistics
- `Concept`: reliability, expected vs observed probability, Brier score, calibration curves.
- `Where it is used`: risk models, uplift models, recommendation/ranking scores.
- `Production example`: if users scored 0.8 only convert 0.4 of the time, thresholding and business planning become unreliable.
- `A/B angle`: calibrated predictions support better decision policies, not just better ranking.

## 19 — Ranking and Recommender Statistics
- `Concept`: CTR, CVR, watch time, NDCG, MAP, MRR, hit rate, coverage, novelty, diversity, calibration.
- `Where it is used`: retrieval/ranking systems and recommendation quality evaluation.
- `Production example`: a recommender can have high CTR but poor diversity, high popularity bias, and low long-term retention.
- `A/B angle`: online experiments are needed because offline ranking metrics only partially predict real user value.

## 20 — Popularity Bias and Propensity in Recommenders
- `Concept`: exposure bias, position bias, inverse propensity weighting, long-tail effects.
- `Where it is used`: debiased training and evaluation for recommender systems.
- `Production example`: items shown at the top get more clicks partly because of position, not because they are better.
- `A/B angle`: experiment interpretation must separate ranking quality from UI placement changes.

## 21 — Time Series Statistics for ML
- `Concept`: trends, seasonality, rolling averages, lag structure, autocorrelation, leakage risk.
- `Where it is used`: demand forecasting, fraud, engagement, response modeling.
- `Production example`: monthly attendance or purchase history needs trailing-window features like 3-month mean, 6-month max, 30-day count.
- `A/B angle`: pre-period covariates and post-period windows must be aligned carefully.

## 22 — Causal Inference Basics
- `Concept`: correlation vs causation, confounding, treatment effect, selection bias.
- `Where it is used`: policy evaluation, uplift modeling, recommender interventions.
- `Production example`: users who receive coupons may already be high intent, so naive response comparisons overstate coupon impact.
- `A/B angle`: randomization is the cleanest way to estimate causality.

## 23 — Statistics You Should Connect Directly To ML Metrics
- `Concept`: precision, recall, F1, ROC-AUC, PR-AUC, log loss, KS, Gini, Brier score.
- `Where it is used`: model validation and business threshold setting.
- `Production example`: fraud detection usually optimizes recall at a fixed review capacity, not plain accuracy.
- `A/B angle`: the offline metric you optimize should align with the online metric you read out.

## 24 — Example-Driven Practice Blocks
- `Binary classification`: predict who will respond to a campaign; use proportions, lift, deciles, calibration, and cutoff selection.
- `Regression`: forecast next-month spend; use residual statistics, outlier checks, and interval estimates.
- `Recommender`: rank courses or products; use CTR, exposure bias, propensity weighting, NDCG, diversity, and online experiments.
- `A/B test`: compare a new ranking model to old; use power, sample ratio mismatch checks, primary metric, guardrails, and practical significance.

## 25 — Minimal Example Ideas To Add When Expanding This Outline
- `Attendance example`: "What is the 3-month maximum number of seminars attended by a user?"
- `Campaign example`: "Which decile of users should receive the offer if the call-center budget reaches only 20% of users?"
- `Recommender example`: "Did the new ranking model increase CTR only for popular items, or also improve long-tail discovery?"

## Current References
- Scikit-learn feature selection docs: https://scikit-learn.org/stable/modules/feature_selection.html
- PyTorch distributed overview: https://docs.pytorch.org/tutorials/distributed.html
- Google responsible AI fairness overview: https://developers.google.com/machine-learning/guides/intro-responsible-ai/fairness
- AIF360 fairness metrics docs: https://aif360.readthedocs.io/en/stable/
- Recent causal feature selection survey: https://arxiv.org/abs/2402.02696
