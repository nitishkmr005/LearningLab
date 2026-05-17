# 02 — Statistics

Exhaustive learning path covering probability, inference, experimentation, and causal reasoning for data scientists.

---

## 01 — Probability Fundamentals
Events, sample space, conditional probability, independence, law of total probability, Bayes' theorem.
- https://seeing-theory.brown.edu/
- https://www.probabilitycourse.com/

## 02 — Combinatorics & Counting
Permutations, combinations, multinomial; birthday problem; inclusion-exclusion principle.
- https://brilliant.org/courses/probability/

## 03 — Discrete Probability Distributions
Bernoulli, Binomial, Poisson, Geometric, Hypergeometric, Negative Binomial; PMF, CDF, E[X], Var[X].
- https://docs.scipy.org/doc/scipy/reference/stats.html

## 04 — Continuous Probability Distributions
Normal, Uniform, Exponential, Gamma, Beta, Log-normal, Student-t, Chi-squared, F; PDF, CDF, quantile functions.
- https://docs.scipy.org/doc/scipy/reference/stats.html
- https://seeing-theory.brown.edu/probability-distributions/index.html

## 05 — Descriptive Statistics
Mean, median, mode, variance, std, IQR, skewness, kurtosis; outlier detection (z-score, Tukey fences); pandas describe().
- https://numpy.org/doc/stable/reference/routines.statistics.html

## 06 — Central Limit Theorem
Sampling distribution of the mean; CLT simulation; standard error; why CLT underpins frequentist inference.
- https://seeing-theory.brown.edu/frequentist-inference/index.html

## 07 — Confidence Intervals
CI construction via CLT; t-distribution for small samples; interpretation pitfall; Wilson interval for proportions.
- https://www.statsmodels.org/stable/stats.html

## 08 — Hypothesis Testing Framework
Null/alternative hypothesis; p-value; Type I (α) and Type II (β) errors; one-tailed vs two-tailed; steps.
- https://www.statsmodels.org/stable/stats.html
- https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample

## 09 — t-Tests
One-sample, two-sample (Welch's), paired t-test; assumptions; when to use; scipy.stats.ttest_*.
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

## 10 — Chi-Square Tests
Goodness-of-fit; test of independence (contingency tables); Cramér's V effect size.
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

## 11 — ANOVA & Post-Hoc Tests
One-way ANOVA; F-statistic; Tukey HSD post-hoc; Welch's ANOVA when variances differ.
- https://www.statsmodels.org/stable/anova.html

## 12 — Non-Parametric Tests
Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Kolmogorov-Smirnov; when to abandon normality.
- https://docs.scipy.org/doc/scipy/reference/stats.html

## 13 — Effect Size
Cohen's d, Cohen's h, η², Pearson's r as effect size; why significance ≠ importance.
- https://en.wikipedia.org/wiki/Effect_size

## 14 — Power Analysis & Sample Size Calculation
Statistical power (1-β); sample size for t-test and proportions test; power curves; statsmodels TTestPower.
- https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations
- https://www.evanmiller.org/ab-testing/sample-size.html

## 15 — A/B Testing End-to-End
Randomization unit; SRM (sample ratio mismatch) checks; metric selection; duration; novelty effect; readout.
- https://netflixtechblog.com/experimentation-is-a-major-focus-of-data-science-across-netflix-f67923f8e985
- https://www.exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf

## 16 — Sequential Testing & Peeking Problem
Fixed-horizon vs sequential testing; alpha spending; always-valid p-values (mixture sequential probability ratio test).
- https://www.evanmiller.org/sequential-ab-testing.html

## 17 — Multiple Testing Correction
FWER (Bonferroni, Holm-Bonferroni); FDR (Benjamini-Hochberg); Benjamini-Yekutieli for dependency.
- https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

## 18 — Bayesian Inference Basics
Prior × likelihood ∝ posterior; conjugate priors; Beta-Binomial; credible intervals; MCMC overview.
- https://www.bayesrulesbook.com/
- https://docs.pymc.io/en/stable/learn.html

## 19 — Bayesian A/B Testing
Model conversion rates with Beta; compute P(B > A) analytically or via sampling; expected loss.
- https://www.evanmiller.org/bayesian-ab-testing.html

## 20 — Bootstrap & Permutation Tests
Bootstrap confidence intervals for any statistic; permutation test as assumption-free hypothesis test.
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

## 21 — Correlation & Covariance
Pearson, Spearman, Kendall tau; partial correlation; VIF for multicollinearity; correlation ≠ causation.
- https://realpython.com/numpy-scipy-pandas-correlation-python/

## 22 — Linear Regression (statsmodels)
OLS; interpret coefficients; R², adjusted R²; residual diagnostics; heteroscedasticity (Breusch-Pagan).
- https://www.statsmodels.org/stable/regression.html
- https://www.statlearning.com/

## 23 — Regularization: Ridge & Lasso
L2 (Ridge) shrinks; L1 (Lasso) zeroes out; cross-validate λ; path plot; sklearn RidgeCV/LassoCV.
- https://scikit-learn.org/stable/modules/linear_model.html

## 24 — Logistic Regression & Classification Metrics
Log-odds; MLE; ROC-AUC; precision-recall curve; calibration; sklearn LogisticRegression.
- https://www.statlearning.com/

## 25 — Time Series: Stationarity & ARIMA
ADF test; differencing; ACF/PACF; ARIMA(p,d,q); seasonal SARIMA; statsmodels ARIMA.
- https://otexts.com/fpp3/
- https://www.statsmodels.org/stable/tsa.html

## 26 — Survival Analysis
Kaplan-Meier estimator; log-rank test; Cox proportional hazards; censored data; lifelines library.
- https://lifelines.readthedocs.io/en/latest/
- https://en.wikipedia.org/wiki/Survival_analysis

## 27 — Causal Inference: Potential Outcomes
Counterfactuals; ATE, ATT; confounding; selection bias; propensity scores; overlap assumption.
- https://matheusfacure.github.io/python-causality-handbook/

## 28 — Causal Inference: Methods
Difference-in-differences; regression discontinuity; instrumental variables; matching; doubly robust estimators.
- https://www.causalinferencebook.net/
- https://matheusfacure.github.io/python-causality-handbook/

## 29 — Information Theory for ML
Entropy, joint entropy, conditional entropy; KL divergence and why it appears in VAEs, RL, and fine-tuning; mutual information; cross-entropy loss derivation; sklearn mutual_info_classif.
- https://www.deeplearningbook.org/contents/prob.html
- https://arxiv.org/abs/2106.09685

## 30 — Missing Data Mechanisms (MCAR, MAR, MNAR)
Missing completely at random vs informative missingness; listwise deletion pitfalls; multiple imputation (MICE); sklearn SimpleImputer vs IterativeImputer; missingness as a feature.
- https://scikit-learn.org/stable/modules/impute.html

## 31 — Gaussian Mixture Models & EM Algorithm
Soft clustering; E-step / M-step derivation; log-likelihood optimization; BIC/AIC for k selection; sklearn GaussianMixture; use cases: user segmentation, density estimation, anomaly detection.
- https://scikit-learn.org/stable/modules/mixture.html
- https://arxiv.org/abs/1111.0352
