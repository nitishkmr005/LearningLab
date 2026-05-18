# Statistics for Data Scientists & ML Engineers: From Probability to Causal Inference

*The complete reference covering every statistical concept that appears in DS/ML interviews and production systems — probability theory, hypothesis testing, A/B experimentation, Bayesian methods, regression, time series, causal inference, and information theory.*

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [A Brief History](#2-a-brief-history)
3. [Probability Foundations](#3-probability-foundations)
4. [Distributions: Discrete & Continuous](#4-distributions-discrete--continuous)
5. [Descriptive Statistics](#5-descriptive-statistics)
6. [Central Limit Theorem](#6-central-limit-theorem)
7. [Confidence Intervals](#7-confidence-intervals)
8. [Hypothesis Testing Framework](#8-hypothesis-testing-framework)
9. [The Major Tests](#9-the-major-tests)
10. [Effect Size & Power Analysis](#10-effect-size--power-analysis)
11. [A/B Testing End-to-End](#11-ab-testing-end-to-end)
12. [Sequential Testing & Multiple Corrections](#12-sequential-testing--multiple-corrections)
13. [Bayesian Inference & Bayesian A/B Testing](#13-bayesian-inference--bayesian-ab-testing)
14. [Bootstrap & Permutation Tests](#14-bootstrap--permutation-tests)
15. [Correlation & Regression](#15-correlation--regression)
16. [Time Series: Stationarity & ARIMA](#16-time-series-stationarity--arima)
17. [Survival Analysis](#17-survival-analysis)
18. [Causal Inference](#18-causal-inference)
19. [Information Theory for ML](#19-information-theory-for-ml)
20. [Missing Data & EM Algorithm](#20-missing-data--em-algorithm)
21. [The Modern Recipe](#21-the-modern-recipe)
22. [References](#22-references)

---

## 1. The Problem

Statistics is the invisible substrate under every data science project. You can have the fanciest gradient boosting pipeline in the world, but if you chose the wrong evaluation metric, ran your A/B test for too short a duration, or confused correlation with causation in your feature selection, you will ship something that actively hurts the product. The consequences range from mild (a model that doesn't generalize) to catastrophic (rolling out a harmful policy because your experiment was underpowered and you declared a false positive).

The classic failure is the underpowered A/B test. A data scientist runs an experiment, peeks at the results after three days, sees p=0.03, calls it a win, and ships. What they don't know: the test needed 10,000 users per arm and they only had 2,000; the observed effect was pure noise; and the company now has a "winning" feature that will quietly drain engagement for six months before anyone figures it out. The fix is not more data — it's knowing how to compute sample size before you start, how to interpret a p-value correctly, and why peeking at results invalidates frequentist guarantees.

This blog covers every statistical concept a data scientist needs: from the law of total probability to causal difference-in-differences, from t-tests to Bayesian A/B testing, from ARIMA to the EM algorithm. Each concept includes the formula, a worked numerical example, and runnable Python code.

### Statistics in Practice: Real-World Use Cases

Statistics is not abstract — every concept maps to a concrete production scenario. Use this table as a decision guide:

| Scenario | Statistical Tool | Why This Tool |
|---|---|---|
| "Did our new recommendation model increase CTR?" | Two-sample t-test or z-test | Compare means between control and treatment groups |
| "Is the CTR difference large enough to matter?" | Cohen's d (effect size) + power analysis | Statistical significance ≠ practical significance |
| "We peeked at results after 3 days — is that okay?" | Sequential testing (mSPRT / O'Brien-Fleming) | Peeking inflates false positive rate with fixed-horizon tests |
| "Our churn model outputs probabilities — are they reliable?" | Calibration curve + ECE | Model scores 0.7 should mean 70% churn rate |
| "Feature X and Y are correlated 0.85 — should we use both?" | Pearson/Spearman correlation + VIF | Multicollinearity inflates coefficient variance in linear models |
| "We rolled out a feature 6 months ago — did it improve retention?" | Difference-in-Differences (DiD) | Removes confounding trends with a control group over time |
| "We can't A/B test pricing changes fairly (network effects)" | Instrumental Variables (IV) | Estimate causal effect when random assignment isn't possible |
| "Revenue is seasonal — are our metrics improving or just holiday-driven?" | STL decomposition + stationarity tests (ADF) | Separate trend/seasonality from signal |
| "We have 200 features — which actually matter?" | Mutual information, SHAP, feature importance with p-values | Distinguish signal from noise statistically |
| "How long until 50% of new users churn?" | Survival analysis (Kaplan-Meier) | Handles censoring (users still active at analysis time) |
| "We have 15 A/B variants — how do we avoid false positives?" | Bonferroni or Benjamini-Hochberg correction | Multiple comparisons inflate Type I error |
| "Small user base — can't wait for classical A/B test to hit power" | Bayesian A/B test with Beta-Binomial | Works at any sample size, reports probability of winning |
| "We want to know P(user churns | they logged in < 3 times)" | Conditional probability + Bayes' theorem | Direct probabilistic reasoning from observed data |
| "How uncertain is our estimated conversion rate?" | Confidence interval (Wilson for proportions) | Report range, not just point estimate |
| "Two algorithms, same accuracy — which is truly better?" | Paired t-test or McNemar's test on predictions | Test if difference is systematic, not due to random test-set variation |

**Concrete example — SaaS churn prediction project:**
A data scientist building a B2B SaaS churn model would use statistics at every step:
1. **EDA:** descriptive statistics (median contract value, IQR of login frequency), correlation matrix to spot redundant features
2. **Feature selection:** mutual information against the churn label, VIF to detect multicollinearity
3. **Model calibration:** calibration curve to verify predicted churn probability = observed churn rate
4. **Intervention design:** power analysis to determine how many accounts to target in a save campaign to detect a 5% retention improvement
5. **Campaign evaluation:** two-sample z-test on retention rates between saved and control accounts; Kaplan-Meier curves to compare time-to-churn
6. **Long-term impact:** DiD to separate campaign effect from seasonal retention improvement

---

## 2. A Brief History

Statistics as a discipline emerged from practical problems — Graunt counting deaths in London (1662), Gauss fitting orbits with least squares (1809), Galton discovering regression to the mean (1886). The frequentist framework crystallized through Fisher's work on significance testing (1925), Neyman and Pearson's hypothesis testing framework (1933), and Wald's sequential analysis during World War II.

Bayesian inference has older roots — Bayes' theorem was published posthumously in 1763 — but remained computationally intractable until MCMC methods (Gelfand & Smith, 1990) made posterior sampling practical. The modern era is characterized by a pragmatic fusion: frequentist methods dominate industrial A/B testing (Netflix, Airbnb, Spotify all use frequentist frameworks with sequential corrections), while Bayesian methods dominate when prior information matters or when sample sizes are small.

The causal inference revolution came later: Rubin's potential outcomes framework (1974), Pearl's do-calculus (1995), and Angrist & Imbens' econometric work on instrumental variables (1994 Nobel Prize-adjacent work, 2021 Nobel Prize) gave data scientists a rigorous vocabulary for asking "what caused what" rather than just "what correlates with what."

---

## 3. Probability Foundations

Probability theory provides the mathematical language for uncertainty. Every statistical test, every ML loss function, and every Bayesian posterior is built on these foundations.

### 3.1 Sample Space, Events, and Axioms

The **sample space** Ω is the set of all possible outcomes. An **event** is a subset of Ω. Probability P maps events to [0,1] satisfying Kolmogorov's axioms: P(Ω)=1, P(A)≥0 for all A, and P(A∪B) = P(A)+P(B) for disjoint A,B.

### 3.2 Conditional Probability

P(A|B) = P(A∩B) / P(B), the probability of A given that B has occurred.

**Worked example**: A dataset has 1000 users. 200 are premium users. Of those 200, 160 made a purchase. Of the 800 free users, 80 made a purchase. P(purchase|premium) = 160/200 = 0.80. P(purchase|free) = 80/800 = 0.10.

### 3.3 Bayes' Theorem

```
P(A|B) = P(B|A) × P(A) / P(B)
```

where P(B) = P(B|A)P(A) + P(B|Aᶜ)P(Aᶜ) by the **Law of Total Probability**.

**Worked example — Spam filter**: P(spam)=0.20, P(word "free"|spam)=0.80, P("free"|not spam)=0.10.

```
P(spam|"free") = (0.80 × 0.20) / (0.80×0.20 + 0.10×0.80)
               = 0.16 / (0.16 + 0.08)
               = 0.16 / 0.24
               = 0.667
```

A message containing "free" has a 66.7% chance of being spam.

### 3.4 Independence

Events A and B are independent if P(A∩B) = P(A)×P(B), equivalently P(A|B) = P(A). Independence is a strong assumption — most real-world features are not independent, which is why Naive Bayes ("naively" assumes feature independence) is an approximation.

### 3.5 Combinatorics

- **Permutations**: ordered arrangements. P(n,k) = n!/(n-k)!
- **Combinations**: unordered. C(n,k) = n!/(k!(n-k)!)
- **Birthday problem**: With n people, P(at least two share a birthday) = 1 - 365!/((365-n)! × 365ⁿ). With n=23, this exceeds 50%. Counterintuitive because we're checking all pairs, not one specific pair.

```python
import math
from scipy.special import comb

n, k = 10, 3
perms = math.perm(n, k)  # 720
combos = math.comb(n, k)  # 120

# Birthday problem
n_people = 23
prob_no_shared = math.prod((365-i)/365 for i in range(n_people))
prob_shared = 1 - prob_no_shared  # ~0.507
print(f"P(shared birthday, n=23): {prob_shared:.3f}")
```

> 🎯 **Interview prep**: "What is the birthday paradox?" — With 23 people, there's a >50% chance two share a birthday, because you're checking C(23,2)=253 pairs, not one specific pair against a fixed birthday.

**Resources**
- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/) — stunning visual probability explanations
- [Probability Course](https://www.probabilitycourse.com/) — rigorous free textbook

---

## 4. Distributions: Discrete & Continuous

Understanding which distribution governs your data is the difference between a model that works and one that silently fails.

### 4.1 Discrete Distributions

**Bernoulli(p)**: single binary trial. E[X]=p, Var[X]=p(1-p).

**Binomial(n,p)**: number of successes in n independent Bernoulli trials.
- PMF: P(X=k) = C(n,k) × pᵏ × (1-p)ⁿ⁻ᵏ
- E[X]=np, Var[X]=np(1-p)
- Use case: number of clicked ads in 100 impressions

**Poisson(λ)**: number of events in a fixed interval when events occur at rate λ.
- PMF: P(X=k) = e⁻λ × λᵏ / k!
- E[X]=λ, Var[X]=λ
- Use case: support tickets per hour, server errors per day

**Geometric(p)**: number of trials until first success.
- PMF: P(X=k) = (1-p)ᵏ⁻¹ × p
- E[X]=1/p, Var[X]=(1-p)/p²

```python
from scipy import stats
import numpy as np

# Poisson: expected 3 clicks per minute
poisson = stats.poisson(mu=3)
print(f"P(X=5): {poisson.pmf(5):.4f}")   # 0.1008
print(f"P(X<=3): {poisson.cdf(3):.4f}")  # 0.6472

# Binomial: 100 impressions, 5% CTR
binom = stats.binom(n=100, p=0.05)
print(f"P(>=10 clicks): {1 - binom.cdf(9):.4f}")  # 0.0282
```

### 4.2 Continuous Distributions

**Normal N(μ,σ²)**: the workhorse. PDF: f(x) = (1/σ√2π)exp(-(x-μ)²/2σ²). The 68-95-99.7 rule: 68% of data within ±1σ, 95% within ±2σ, 99.7% within ±3σ.

**Standard Normal N(0,1)**: z = (x-μ)/σ. z-scores tell you how many standard deviations from the mean.

**Student-t(ν)**: heavier tails than Normal. Used when sample size is small and population σ is unknown. Approaches Normal as ν→∞.

**Chi-squared(k)**: sum of k squared standard normal variables. Appears in goodness-of-fit tests.

**F(d1,d2)**: ratio of two chi-squared variables. Appears in ANOVA and regression.

**Beta(α,β)**: defined on [0,1]. Natural for modeling probabilities and proportions. E[X]=α/(α+β). Conjugate prior to Binomial.

**Exponential(λ)**: time between Poisson events. Memoryless property: P(X>s+t|X>s) = P(X>t). E[X]=1/λ.

```python
from scipy import stats
import matplotlib.pyplot as plt

# Standard normal
z = stats.norm(0, 1)
print(f"P(|Z|>1.96): {2*(1-z.cdf(1.96)):.4f}")  # 0.0500

# t-distribution with 10 df vs normal
t10 = stats.t(df=10)
print(f"t critical (0.025, df=10): {t10.ppf(0.975):.4f}")  # 2.228 > 1.96

# Beta distribution
beta = stats.beta(a=10, b=90)  # prior: 10% conversion rate
print(f"Beta(10,90) mean: {beta.mean():.3f}")  # 0.100
print(f"95% CI: ({beta.ppf(0.025):.3f}, {beta.ppf(0.975):.3f})")
```

> 🎯 **Interview prep**: "When do you use t vs z?" — Use t when sample size is small (n<30 as a rule of thumb) or when population σ is unknown. In practice with n>100, t and z give essentially identical results.

> 🏭 **Production note**: Real data almost never follows a Normal distribution — it's often right-skewed (incomes, transaction amounts) or zero-inflated (activity counts). Always visualize your data before assuming Normality.

**Resources**
- [SciPy stats module](https://docs.scipy.org/doc/scipy/reference/stats.html) — all distributions with CDF/PPF/PMF
- [Seeing Theory: Distributions](https://seeing-theory.brown.edu/probability-distributions/index.html) — interactive visualizations

---

## 5. Descriptive Statistics

Before any model, you must understand your data. Descriptive statistics are not just exploratory — they reveal data quality issues, guide feature engineering, and catch leakage.

### 5.1 Central Tendency

- **Mean**: sensitive to outliers. Mean of [1,2,3,4,100] = 22.
- **Median**: robust to outliers. Median = 3.
- **Mode**: most frequent value. Useful for categorical data.

### 5.2 Spread

- **Variance**: σ² = E[(X-μ)²] = (1/n)Σ(xᵢ-x̄)²  (population) or s² = (1/(n-1))Σ(xᵢ-x̄)² (sample, Bessel's correction)
- **Standard deviation**: σ = √Var
- **IQR**: Q3-Q1. Robust measure of spread. Used in Tukey's fence for outlier detection: outlier if x < Q1-1.5×IQR or x > Q3+1.5×IQR

### 5.3 Shape

- **Skewness**: asymmetry. Positive skew = tail right (income distributions). Negative = tail left.
- **Kurtosis**: tail heaviness. Normal has kurtosis=3. Excess kurtosis = kurtosis-3. Heavy tails (excess kurtosis >0) mean more extreme values.

```python
import pandas as pd
import numpy as np
from scipy import stats

# Generate skewed data
np.random.seed(42)
data = np.random.lognormal(mean=0, sigma=1, size=1000)

df = pd.Series(data)
print(f"Mean:     {df.mean():.2f}")
print(f"Median:   {df.median():.2f}")
print(f"Std:      {df.std():.2f}")
print(f"IQR:      {df.quantile(0.75) - df.quantile(0.25):.2f}")
print(f"Skewness: {df.skew():.2f}")    # strongly positive
print(f"Kurtosis: {df.kurt():.2f}")   # excess kurtosis

# Outlier detection: z-score
z_scores = np.abs(stats.zscore(data))
outliers_z = np.sum(z_scores > 3)

# Tukey fences
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
outliers_tukey = np.sum((data < Q1-1.5*IQR) | (data > Q3+1.5*IQR))
print(f"Outliers (z>3): {outliers_z}, Outliers (Tukey): {outliers_tukey}")
```

> 🏭 **Production note**: Always check skewness before using mean-based features. Log-transforming right-skewed revenue/amount features often dramatically improves model performance. Check for bimodality too — it often signals a data population issue (two different user types mixed together).

---

## 6. Central Limit Theorem

The Central Limit Theorem (CLT) is the foundation of frequentist inference. Without it, we couldn't compute p-values or confidence intervals for most real-world statistics.

**CLT**: If X₁, X₂, ..., Xₙ are i.i.d. random variables with mean μ and finite variance σ², then the sample mean X̄ = (1/n)ΣXᵢ is approximately normally distributed as n→∞:

```
X̄ ~ N(μ, σ²/n)
```

The standard error of the mean is SE = σ/√n.

**Worked example**: A population of page load times has μ=2.0s, σ=1.5s. We take samples of n=100 users. The sample mean follows N(2.0, (1.5/√100)²) = N(2.0, 0.0225). SE = 0.15s. There's a 95% chance the sample mean falls within ±1.96×0.15 = ±0.294s of the true mean.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Simulate CLT with a skewed population
np.random.seed(42)
population = np.random.exponential(scale=2.0, size=100000)  # skewed

# Sample means of size n=50
n = 50
sample_means = [np.mean(np.random.choice(population, size=n)) for _ in range(10000)]

# Despite skewed population, sample means are approximately normal
print(f"Population mean: {population.mean():.3f}")
print(f"Sample means: mean={np.mean(sample_means):.3f}, std={np.std(sample_means):.4f}")
print(f"Theoretical SE: {population.std()/np.sqrt(n):.4f}")
_, p_value = stats.normaltest(sample_means)
print(f"Normality test p-value: {p_value:.4f}")  # should be high
```

> 🎯 **Interview prep**: "Why does the CLT matter?" — It lets us use normal-distribution math (z-tests, t-tests, confidence intervals) even when the underlying data isn't normal, as long as the sample size is large enough (typically n≥30 for moderately skewed distributions, more for heavy tails).

**Resources**
- [Seeing Theory: Frequentist Inference](https://seeing-theory.brown.edu/frequentist-inference/index.html) — CLT visualization

---

## 7. Confidence Intervals

A confidence interval (CI) is a range of values that, if we repeated our sampling procedure many times, would contain the true parameter in 95% (or whatever level) of those repetitions. The single most common misinterpretation: "there's a 95% chance the true mean is in this interval." Wrong — the true mean is fixed; it's the interval that's random.

### 7.1 CI via CLT

For a large sample with known (or estimated) σ:

```
CI: x̄ ± z_(α/2) × (σ/√n)
```

For α=0.05 (95% CI), z_(0.025) = 1.96.

**Worked example**: n=400 users, sample mean conversion=0.12, sample std=0.08.
- SE = 0.08/√400 = 0.004
- 95% CI = 0.12 ± 1.96×0.004 = (0.1122, 0.1278)

### 7.2 t-Distribution CI for Small Samples

When n<30 or σ unknown:
```
CI: x̄ ± t_(α/2, n-1) × (s/√n)
```

### 7.3 Wilson Interval for Proportions

The standard CI for proportions p̂ ± z√(p̂(1-p̂)/n) can give nonsensical values (below 0 or above 1) when p̂ is near 0 or 1, or n is small. Wilson interval is preferred:

```
CI: (p̂ + z²/2n ± z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)
```

```python
import numpy as np
from scipy import stats
import statsmodels.stats.proportion as smp

# Standard CI
n, successes = 400, 48
p_hat = successes / n  # 0.12
se = np.sqrt(p_hat * (1-p_hat) / n)
ci_standard = (p_hat - 1.96*se, p_hat + 1.96*se)
print(f"Standard CI: {ci_standard}")

# Wilson CI (use for small n or extreme proportions)
ci_wilson = smp.proportion_confint(successes, n, alpha=0.05, method='wilson')
print(f"Wilson CI: {ci_wilson}")

# t-CI for small sample
sample = np.random.normal(5.2, 1.5, size=20)
ci_t = stats.t.interval(0.95, df=len(sample)-1,
                         loc=np.mean(sample),
                         scale=stats.sem(sample))
print(f"t-CI (n=20): {ci_t}")
```

> 🏭 **Production note**: Bootstrap CIs (Section 14) are often better in production than CLT-based CIs when the statistic isn't a mean — e.g., for median, 90th percentile latency, or revenue per user which is heavily skewed.

---

## 8. Hypothesis Testing Framework

Hypothesis testing is a formal decision procedure for evaluating claims about data. It does not tell you that your hypothesis is true — it tells you whether the data are surprising under the null hypothesis.

### 8.1 The Five Steps

1. **State H₀ (null) and H₁ (alternative)**: H₀ is the boring "nothing happened" hypothesis. Example: H₀: μ_control = μ_treatment; H₁: μ_control ≠ μ_treatment.
2. **Choose significance level α**: typically 0.05 (5% Type I error rate).
3. **Compute test statistic**: standardize the observed effect.
4. **Compute p-value**: P(seeing data this extreme | H₀ is true).
5. **Decision**: if p < α, reject H₀. Otherwise, fail to reject H₀ (not the same as accepting H₀).

### 8.2 Type I and Type II Errors

| Decision \ Truth | H₀ True | H₀ False |
|---|---|---|
| Reject H₀ | **Type I error** (α) | Correct (Power = 1-β) |
| Fail to reject | Correct | **Type II error** (β) |

- **Type I error rate (α)**: false positive — concluding an effect exists when it doesn't. Controlled by choice of α.
- **Type II error rate (β)**: false negative — missing a real effect. Controlled by sample size.
- **Power (1-β)**: probability of correctly detecting a real effect.

### 8.3 p-value Interpretation

The p-value is NOT the probability that H₀ is true. It is P(test statistic ≥ observed | H₀ true). A p-value of 0.04 means: if there truly were no effect, you'd see data this extreme 4% of the time by chance.

> 🎯 **Interview prep**: "What is a p-value?" — The probability of observing a test statistic as extreme as the one computed, assuming the null hypothesis is true. Common wrong answers: "the probability the null is true" or "the probability the result is due to chance."

**Resources**
- [Statsmodels stats module](https://www.statsmodels.org/stable/stats.html)

---

## 9. The Major Tests

### 9.1 t-Tests

The t-test compares means. The test statistic follows a t-distribution under H₀.

**One-sample t-test**: test if population mean equals μ₀.
```
t = (x̄ - μ₀) / (s / √n)
```

**Two-sample (Welch's) t-test**: compare means of two independent groups. Welch's handles unequal variances:
```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

**Paired t-test**: compare means of matched/related samples (e.g., before/after the same users).

**Worked example**: Control conversion: n₁=200, x̄₁=0.10, s₁=0.08. Treatment: n₂=200, x̄₂=0.13, s₂=0.09.
```
SE = √(0.08²/200 + 0.09²/200) = √(0.000032 + 0.0000405) = √0.0000725 = 0.00851
t = (0.13-0.10)/0.00851 = 0.03/0.00851 = 3.525
df ≈ 390  →  p < 0.001  →  reject H₀
```

```python
from scipy import stats
import numpy as np

np.random.seed(42)
control = np.random.binomial(1, 0.10, 200).astype(float)
treatment = np.random.binomial(1, 0.13, 200).astype(float)

# Welch's t-test (unequal variances, default)
t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
print(f"t={t_stat:.3f}, p={p_val:.4f}")

# Paired t-test
before = np.random.normal(50, 10, 50)
after = before + np.random.normal(3, 5, 50)  # some improvement
t_stat, p_val = stats.ttest_rel(after, before)
print(f"Paired t: t={t_stat:.3f}, p={p_val:.4f}")
```

### 9.2 Chi-Square Tests

**Goodness-of-fit**: does observed distribution match expected?
```
χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ
```

**Independence test**: are two categorical variables independent?

**Worked example**: 1000 users split by device (mobile/desktop) and purchase (yes/no).

| | Purchased | Not Purchased |
|---|---|---|
| Mobile | 60 | 440 |
| Desktop | 90 | 410 |

Expected under independence: E(mobile, purchase) = (500×150)/1000 = 75.

χ² = (60-75)²/75 + (90-75)²/75 + (440-425)²/425 + (410-425)²/425 = 3+3+0.53+0.53 = 7.06, df=1, p=0.008 → reject independence.

```python
from scipy.stats import chi2_contingency
import numpy as np

observed = np.array([[60, 440], [90, 410]])
chi2, p, dof, expected = chi2_contingency(observed)
print(f"χ²={chi2:.3f}, p={p:.4f}, df={dof}")
print(f"Expected:\n{expected}")

# Cramér's V (effect size for chi-square)
n = observed.sum()
cramers_v = np.sqrt(chi2 / (n * (min(observed.shape)-1)))
print(f"Cramér's V: {cramers_v:.3f}")  # 0=no assoc, 1=perfect
```

### 9.3 ANOVA

One-way ANOVA tests if means differ across ≥3 groups. The F-statistic:
```
F = (Between-group variance) / (Within-group variance)
  = [Σnᵢ(x̄ᵢ - x̄)² / (k-1)] / [ΣΣ(xᵢⱼ - x̄ᵢ)² / (n-k)]
```

If F is large, between-group variance dominates → groups differ. After ANOVA, use **Tukey HSD** for post-hoc pairwise comparisons with family-wise error rate control.

```python
from scipy import stats
import numpy as np

# Three email subject lines, measuring open rates
np.random.seed(42)
a = np.random.normal(0.20, 0.05, 100)
b = np.random.normal(0.24, 0.05, 100)
c = np.random.normal(0.22, 0.05, 100)

f_stat, p_val = stats.f_oneway(a, b, c)
print(f"F={f_stat:.3f}, p={p_val:.4f}")

# Post-hoc: Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

data = np.concatenate([a, b, c])
groups = ['A']*100 + ['B']*100 + ['C']*100
result = pairwise_tukeyhsd(data, groups, alpha=0.05)
print(result)
```

### 9.4 Non-Parametric Tests

Use these when you can't assume normality, have ordinal data, or have small samples.

| Parametric | Non-Parametric Alternative |
|---|---|
| one-sample t | Wilcoxon signed-rank |
| two-sample t | Mann-Whitney U |
| paired t | Wilcoxon signed-rank (paired) |
| one-way ANOVA | Kruskal-Wallis |
| Pearson correlation | Spearman rank correlation |

**Mann-Whitney U** tests whether one distribution stochastically dominates the other (not just means). Good for skewed distributions like revenue.

```python
from scipy import stats

# Revenue comparison: non-normal, right-skewed
np.random.seed(42)
control = np.random.exponential(100, 500)
treatment = np.random.exponential(115, 500)

# Mann-Whitney U
u_stat, p_val = stats.mannwhitneyu(treatment, control, alternative='two-sided')
print(f"Mann-Whitney U: u={u_stat:.0f}, p={p_val:.4f}")

# Kolmogorov-Smirnov: tests if two distributions are the same
ks_stat, p_val = stats.ks_2samp(control, treatment)
print(f"KS test: D={ks_stat:.3f}, p={p_val:.4f}")
```

> 🏭 **Production note**: For revenue or LTV metrics in A/B tests, use Mann-Whitney U or a log-transform before t-test. Revenue is almost always right-skewed with high-value outliers that inflate variance and reduce power. Many teams also use bootstrapped CIs for revenue metrics.

**Resources**
- [SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html) — all tests
- [Statsmodels](https://www.statsmodels.org/stable/stats.html) — ANOVA, multiple comparison

---

## 10. Effect Size & Power Analysis

Statistical significance tells you whether an effect is real. Effect size tells you whether it matters. A treatment with p=0.001 could have a trivially small effect — you just had enough users to detect it.

### 10.1 Effect Size Measures

**Cohen's d**: standardized mean difference.
```
d = (μ₁ - μ₂) / s_pooled
where s_pooled = √((s₁² + s₂²) / 2)
```
- Small: d=0.2, Medium: d=0.5, Large: d=0.8

**Worked example**: Control mean=10.0, s₁=4.0. Treatment mean=10.8, s₂=4.2.
```
s_pooled = √((16+17.64)/2) = √16.82 = 4.10
d = (10.8-10.0)/4.10 = 0.195  → small effect
```

**Cohen's h**: effect size for proportions. h = 2arcsin(√p₁) - 2arcsin(√p₂)

**η² (eta-squared)**: proportion of variance explained in ANOVA. η² = SS_between / SS_total.

```python
import numpy as np
from scipy import stats

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s_pooled = np.sqrt(((n1-1)*group1.var() + (n2-1)*group2.var()) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / s_pooled

np.random.seed(42)
ctrl = np.random.normal(10.0, 4.0, 500)
trt  = np.random.normal(10.8, 4.2, 500)
d = cohens_d(trt, ctrl)
print(f"Cohen's d = {d:.3f}")  # ~0.195
```

### 10.2 Power Analysis & Sample Size

**Power** = P(reject H₀ | H₁ is true) = 1 - β. Typically target 80% or 90%.

For a two-sample t-test, the required sample size per group:
```
n = (z_α/2 + z_β)² × 2σ² / δ²
```
where δ is the minimum detectable effect (MDE).

**Worked example**: Baseline conversion 10%, MDE=2pp (want to detect 10%→12%), α=0.05, power=80%.
- σ² ≈ p(1-p) = 0.10×0.90 = 0.09 per arm
- z_0.025 = 1.96, z_0.20 = 0.842
- n = (1.96+0.842)² × 2×0.09 / (0.02)² = 7.854 × 0.18 / 0.0004 = 3535 per arm

```python
from statsmodels.stats.power import TTestIndPower, NormalIndPower

# Power analysis for t-test
analysis = TTestIndPower()

# Given effect size and α, compute sample size for 80% power
n = analysis.solve_power(effect_size=0.2, alpha=0.05, power=0.8)
print(f"Required n per group: {n:.0f}")  # 394

# For proportions
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

p1, p2 = 0.10, 0.12
es = proportion_effectsize(p1, p2)
analysis_prop = NormalIndPower()
n_prop = analysis_prop.solve_power(effect_size=es, alpha=0.05, power=0.80)
print(f"Proportions test n per group: {n_prop:.0f}")  # ~3560

# Power curve
import matplotlib.pyplot as plt
ns = np.arange(100, 5000, 100)
powers = [analysis.solve_power(effect_size=0.2, alpha=0.05, nobs1=n) for n in ns]
```

> 🎯 **Interview prep**: "How do you decide experiment duration?" — Compute required sample size from MDE and base rate, then divide by daily traffic. Never end early based on interim results without sequential correction (see Section 12).

**Resources**
- [Statsmodels power](https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations)
- [Evan Miller sample size calculator](https://www.evanmiller.org/ab-testing/sample-size.html)

---

## 11. A/B Testing End-to-End

A/B testing is the gold standard for causal claims in product decisions. The statistics are simple; the hard part is avoiding the subtle mistakes that invalidate results.

### 11.1 Randomization Unit

Choose the unit of randomization carefully. Options: user-level, session-level, cookie-level, device-level. Key principle: **the randomization unit should be at least as coarse as the analysis unit**. If you analyze at the user level, randomize at the user level — not the session level, which allows the same user to see both variants (SUTVA violation: Stable Unit Treatment Value Assumption).

### 11.2 Sample Ratio Mismatch (SRM)

Before trusting any results, check that the actual split ratio matches the intended ratio. An SRM (p<0.01 on a chi-square test of actual vs expected counts) indicates a technical bug — bot traffic filtered differently, assignment algorithm error, etc.

```python
from scipy.stats import chisquare

# Intended: 50/50 split
control_users = 10243
treatment_users = 9891
total = control_users + treatment_users

chi2, p = chisquare([control_users, treatment_users],
                    f_exp=[total/2, total/2])
print(f"SRM check: χ²={chi2:.3f}, p={p:.4f}")
# p<0.01 → investigate before trusting results
```

### 11.3 Novelty Effect

Users behave differently when they encounter something new. A new UI feature may show a short-term engagement spike that doesn't persist. For major UI changes, run the experiment long enough (often 2-4 weeks) to let novelty wear off, and analyze long-term metrics separately.

### 11.4 CUPED: Variance Reduction

Controlled-experiment Using Pre-Experiment Data (CUPED) reduces variance without increasing sample size, effectively increasing power. The idea: regress the outcome on a pre-experiment covariate to remove its variance.

```
Y_cuped = Y - θ × (X - E[X])
where θ = Cov(Y, X) / Var(X)
```

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 1000
# Pre-experiment: past week's revenue per user
pre_exp = np.random.normal(50, 20, n)
# Outcome: current week's revenue
noise = np.random.normal(0, 15, n)
outcome_ctrl = 0.8 * pre_exp + noise  # correlated with pre-exp
outcome_trt = 0.8 * pre_exp + noise + 2.0  # +$2 treatment effect

# CUPED
theta_ctrl = np.cov(outcome_ctrl, pre_exp)[0,1] / np.var(pre_exp)
theta_trt  = np.cov(outcome_trt, pre_exp)[0,1] / np.var(pre_exp)

cuped_ctrl = outcome_ctrl - theta_ctrl * (pre_exp - pre_exp.mean())
cuped_trt  = outcome_trt  - theta_trt  * (pre_exp - pre_exp.mean())

# Compare variance reduction
print(f"Original std: {outcome_ctrl.std():.2f}")
print(f"CUPED std:    {cuped_ctrl.std():.2f}")  # lower → more power

# t-test on CUPED metric
t, p = stats.ttest_ind(cuped_trt, cuped_ctrl)
print(f"CUPED test: t={t:.3f}, p={p:.4f}")
```

> 🏭 **Production note**: Netflix, Airbnb, and Microsoft all use CUPED-style variance reduction in their experimentation platforms. If your pre-experiment metric correlates ≥0.3 with the outcome, CUPED can cut required sample size by 30-50%.

**Resources**
- [Netflix Tech Blog: Experimentation](https://netflixtechblog.com/experimentation-is-a-major-focus-of-data-science-across-netflix-f67923f8e985)
- [Microsoft Experimentation Rules of Thumb](https://www.exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf)

---

## 12. Sequential Testing & Multiple Corrections

### 12.1 The Peeking Problem

In a fixed-horizon test, you compute the p-value once after collecting the pre-specified sample size. If you peek at results mid-experiment and stop early when p<0.05, you inflate Type I error dramatically. With 20 peeks, the true Type I error can exceed 25% even when using α=0.05.

([Spotify Engineering, 2023](https://engineering.atspotify.com/2023/03/choosing-sequential-testing-framework-comparisons-and-discussions))

### 12.2 Sequential Testing Solutions

**Alpha spending functions** allocate your total α budget across planned interim looks. The O'Brien-Fleming spending function spends little α early (conservative early stopping) and more later.

**Always-valid p-values (mSPRT)**: mixture Sequential Probability Ratio Test. These p-values are valid at any stopping time — you can look as often as you want without inflating Type I error. Statsig, Optimizely, and VWO use variants of this approach.

```python
# Approximate sequential test using alpha spending
import numpy as np
from scipy import stats

def sequential_test(control_data, treatment_data, n_looks=5, alpha=0.05):
    """Simple alpha spending with equal allocation (Pocock boundaries)."""
    n_total = len(control_data)
    look_points = [n_total * (i+1) // n_looks for i in range(n_looks)]
    
    # Pocock: same alpha at each look → alpha_i = alpha × f(k,i)
    alpha_spent = alpha / n_looks  # simplified equal spending
    
    results = []
    for n_so_far in look_points:
        c = control_data[:n_so_far]
        t = treatment_data[:n_so_far]
        _, p = stats.ttest_ind(t, c, equal_var=False)
        results.append({'n': n_so_far, 'p': p, 'alpha_boundary': alpha_spent,
                        'significant': p < alpha_spent})
    return results

np.random.seed(42)
ctrl = np.random.normal(0, 1, 2000)
trt  = np.random.normal(0.3, 1, 2000)  # real effect
results = sequential_test(ctrl, trt)
for r in results:
    print(f"n={r['n']:4d}: p={r['p']:.4f} (boundary={r['alpha_boundary']:.4f}) → {'STOP' if r['significant'] else 'continue'}")
```

### 12.3 Multiple Testing Correction

When running many tests (multiple metrics, multiple segments), false positives multiply. If you test 20 independent hypotheses at α=0.05, you expect 1 false positive even when nothing is true.

**Family-wise error rate (FWER) control**:
- **Bonferroni**: α_adjusted = α/m. Conservative, easy to compute. For m=20 tests, α_adj=0.0025.
- **Holm-Bonferroni**: step-down procedure, less conservative than Bonferroni.

**False Discovery Rate (FDR) control** (preferred for exploratory analysis):
- **Benjamini-Hochberg (1995)**: sort p-values; reject H(i) if p(i) ≤ (i/m)×α. Controls the expected proportion of false discoveries among rejections. Less conservative than FWER methods.
- **Benjamini-Yekutieli**: valid when tests are positively dependent.

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

# 20 tests: 18 truly null, 2 truly alternative
np.random.seed(42)
p_null = np.random.uniform(0.01, 1.0, 18)
p_true = np.array([0.001, 0.008])  # real effects
p_values = np.concatenate([p_null, p_true])

# Bonferroni (FWER)
reject_bonf, p_bonf, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
# Benjamini-Hochberg (FDR)
reject_bh, p_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print(f"Bonferroni rejections: {reject_bonf.sum()}")  # fewer, more conservative
print(f"BH rejections: {reject_bh.sum()}")            # more, accepts some FDR
```

> 🎯 **Interview prep**: "When do you use Bonferroni vs Benjamini-Hochberg?" — Bonferroni for confirmatory tests where a single false positive is unacceptable (e.g., medical trials, regulatory approval). BH for exploratory analysis where you're willing to accept that 5% of "discoveries" are false (e.g., feature selection, genome-wide association).

**Resources**
- [Spotify: Choosing a Sequential Testing Framework](https://engineering.atspotify.com/2023/03/choosing-sequential-testing-framework-comparisons-and-discussions)
- [Statsmodels multipletests](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)

---

## 13. Bayesian Inference & Bayesian A/B Testing

Bayesian inference expresses uncertainty as probability distributions over parameters rather than fixed point estimates. The key update rule:

```
posterior ∝ likelihood × prior
P(θ|data) ∝ P(data|θ) × P(θ)
```

### 13.1 Conjugate Priors

A conjugate prior is one where the posterior has the same functional form as the prior, making computation closed-form.

**Beta-Binomial conjugacy** (most useful for conversion rates):
- Prior: θ ~ Beta(α, β)
- Likelihood: k successes in n trials ~ Binomial(n, θ)
- Posterior: θ|data ~ Beta(α+k, β+n-k)

**Worked example**: Prior: Beta(2,18) — represents prior belief of ~10% conversion from 20 historical observations. Observe 13 conversions in 100 trials.
- Posterior: Beta(2+13, 18+87) = Beta(15, 105)
- Posterior mean = 15/(15+105) = 0.125

```python
from scipy import stats
import numpy as np

# Beta-Binomial update
alpha_prior, beta_prior = 2, 18   # prior belief ~10%
n_trials, n_success = 100, 13

alpha_post = alpha_prior + n_success
beta_post  = beta_prior + (n_trials - n_success)

posterior = stats.beta(alpha_post, beta_post)
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% Credible Interval: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
```

### 13.2 Credible Intervals vs Confidence Intervals

The Bayesian **credible interval** [a,b] means "there is a 95% probability the parameter lies in [a,b]" — which is what most people *think* a frequentist CI means. The frequentist CI means something subtler (about long-run coverage of the procedure).

### 13.3 Bayesian A/B Testing

Instead of p-values, compute P(B > A) directly by sampling from posteriors.

```python
from scipy import stats
import numpy as np

# Control: 1000 users, 100 conversions
# Treatment: 1000 users, 130 conversions
alpha_A, beta_A = 1 + 100, 1 + 900   # posterior Beta
alpha_B, beta_B = 1 + 130, 1 + 870

# Monte Carlo P(B > A)
n_samples = 100_000
samples_A = np.random.beta(alpha_A, beta_A, n_samples)
samples_B = np.random.beta(alpha_B, beta_B, n_samples)

prob_B_better = (samples_B > samples_A).mean()
expected_lift = (samples_B - samples_A).mean() / samples_A.mean()

print(f"P(B > A): {prob_B_better:.3f}")   # e.g., 0.978
print(f"Expected relative lift: {expected_lift:.3f}")

# Expected loss: how bad if we choose wrong
loss_if_choose_A = np.maximum(samples_B - samples_A, 0).mean()
loss_if_choose_B = np.maximum(samples_A - samples_B, 0).mean()
print(f"Expected loss (choose A): {loss_if_choose_A:.4f}")
print(f"Expected loss (choose B): {loss_if_choose_B:.4f}")
```

> 🏭 **Production note**: Bayesian A/B testing is more intuitive for business stakeholders ("98% chance B is better") vs frequentist ("we cannot reject the null at α=0.05"). But be careful: Bayesian tests still require adequate sample sizes to be meaningful. A prior can dominate a small sample inappropriately.

| | Frequentist | Bayesian |
|---|---|---|
| Output | p-value, CI | P(B>A), credible interval |
| Requires pre-specified n | Yes (for validity) | No (but recommended) |
| Incorporates prior knowledge | No | Yes |
| Early stopping | Requires correction | Natural with loss threshold |
| Business communication | Harder | Easier |

---

## 14. Bootstrap & Permutation Tests

### 14.1 Bootstrap Confidence Intervals

The bootstrap treats the sample as a proxy for the population and resamples from it repeatedly to estimate sampling variability. This works for **any statistic** — no distributional assumption needed.

```python
import numpy as np
from scipy import stats

np.random.seed(42)
data = np.random.lognormal(0, 1, 500)  # skewed data

# Bootstrap CI for the median
n_bootstrap = 10_000
bootstrap_medians = np.array([
    np.median(np.random.choice(data, size=len(data), replace=True))
    for _ in range(n_bootstrap)
])

# Percentile method
ci_percentile = np.percentile(bootstrap_medians, [2.5, 97.5])
print(f"Bootstrap 95% CI for median: ({ci_percentile[0]:.3f}, {ci_percentile[1]:.3f})")

# scipy bootstrap (more sophisticated methods)
result = stats.bootstrap((data,), np.median, n_resamples=10_000,
                          confidence_level=0.95)
print(f"scipy bootstrap CI: {result.confidence_interval}")
```

### 14.2 Permutation Test

A permutation test is a non-parametric hypothesis test. Under H₀ (no difference between groups), group labels are interchangeable. Compute the test statistic on random relabelings to build the null distribution.

```python
import numpy as np

np.random.seed(42)
control = np.random.exponential(100, 300)
treatment = np.random.exponential(110, 300)

observed_diff = treatment.mean() - control.mean()
combined = np.concatenate([control, treatment])
n_control = len(control)

# Permutation test
n_permutations = 10_000
perm_diffs = np.array([
    np.random.permutation(combined)[:n_control].mean() -
    np.random.permutation(combined)[n_control:].mean()
    for _ in range(n_permutations)
])

p_value = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()
print(f"Observed difference: {observed_diff:.2f}")
print(f"Permutation p-value: {p_value:.4f}")
```

> 🎯 **Interview prep**: "Why use a permutation test over a t-test?" — Permutation tests make no distributional assumptions, are exact for any sample size, and work for any statistic (not just means). For revenue metrics that are severely skewed, permutation tests are more reliable than t-tests.

---

## 15. Correlation & Regression

### 15.1 Correlation

**Pearson r**: linear relationship between two continuous variables.
```
r = Cov(X,Y) / (σ_X × σ_Y) = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[Σ(xᵢ-x̄)² × Σ(yᵢ-ȳ)²]
```
Ranges -1 to +1. Pearson r = 0 doesn't mean independence — it means no **linear** relationship.

**Spearman ρ**: rank-based. Captures monotonic relationships. Robust to outliers and non-normality.

**Kendall τ**: another rank-based measure, better for small samples.

**Partial correlation**: correlation between X and Y after removing the effect of confounders Z.

**Variance Inflation Factor (VIF)**: measures multicollinearity among predictors. VIF > 10 indicates problematic collinearity.

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)
X = np.random.normal(0, 1, 200)
Y_linear = 2*X + np.random.normal(0, 0.5, 200)
Y_nonlinear = X**2 + np.random.normal(0, 0.5, 200)

pearson_r, p = stats.pearsonr(X, Y_linear)
spearman_r, p = stats.spearmanr(X, Y_nonlinear)
print(f"Linear: Pearson r={pearson_r:.3f}")
print(f"Non-linear: Spearman r={spearman_r:.3f}")  # captures X² relationship

# VIF
df_X = pd.DataFrame({'x1': X, 'x2': X + np.random.normal(0, 0.1, 200),  # nearly collinear
                     'x3': np.random.normal(0, 1, 200)})
vifs = [variance_inflation_factor(df_X.values, i) for i in range(df_X.shape[1])]
print(f"VIFs: {[f'{v:.1f}' for v in vifs]}")  # x1, x2 very high
```

### 15.2 Linear Regression (OLS)

OLS minimizes sum of squared residuals: minimize Σ(yᵢ - ŷᵢ)². The closed-form solution: β = (XᵀX)⁻¹Xᵀy.

Key diagnostics:
- **R²**: proportion of variance explained. Adjusted R² penalizes for additional predictors.
- **Residual plots**: residuals should be random (no pattern). Patterns indicate model misspecification.
- **Breusch-Pagan test**: tests for heteroscedasticity (non-constant variance).

```python
import statsmodels.api as sm
import numpy as np

np.random.seed(42)
X = np.random.normal(0, 1, (200, 3))
beta_true = [1.5, -0.8, 0.3]
y = X @ beta_true + np.random.normal(0, 1, 200)

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())

# Key outputs: coeff estimates, std errors, t-stats, p-values, R²
# Residual diagnostics
residuals = model.resid
_, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(residuals, X_with_const)
print(f"Breusch-Pagan p-value: {bp_p:.4f}")  # p<0.05 → heteroscedasticity
```

### 15.3 Regularization: Ridge (L2) and Lasso (L1)

When predictors are many or collinear, OLS overfits. Regularization adds a penalty:
- **Ridge**: minimize Σ(yᵢ-ŷᵢ)² + λΣβⱼ². Shrinks coefficients toward 0 but doesn't zero them out.
- **Lasso**: minimize Σ(yᵢ-ŷᵢ)² + λΣ|βⱼ|. Zeros out some coefficients → automatic feature selection.
- **ElasticNet**: combines L1 and L2.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.seed(42)
n, p = 200, 50
X = np.random.normal(0, 1, (n, p))
beta = np.zeros(p); beta[:5] = [1.5, -1, 0.8, -0.5, 0.3]  # sparse true model
y = X @ beta + np.random.normal(0, 1, n)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with cross-validation for λ
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
print(f"Lasso best lambda: {lasso.alpha_:.4f}")
print(f"Non-zero coefficients: {(lasso.coef_ != 0).sum()} / {p}")

# Ridge
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=np.logspace(-2, 3, 100), cv=5).fit(X_scaled, y)
print(f"Ridge best lambda: {ridge.alpha_:.4f}")
print(f"Non-zero coefficients: {(ridge.coef_ != 0).sum()} / {p}")  # all non-zero
```

### 15.4 Logistic Regression

Models P(y=1|X) = σ(Xβ) = 1/(1+exp(-Xβ)). The log-odds are linear: log(p/(1-p)) = Xβ.

Key metrics: ROC-AUC (ranking quality), precision-recall (imbalanced classes), calibration (probability accuracy).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibrationDisplay
import numpy as np

np.random.seed(42)
X = np.random.normal(0, 1, (1000, 5))
beta = [1.2, -0.8, 0.5, 0.3, -0.2]
p = 1 / (1 + np.exp(-X @ beta))
y = np.random.binomial(1, p)

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X, y)
probs = model.predict_proba(X)[:,1]

print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
print(classification_report(y, model.predict(X)))
```

---

## 16. Time Series: Stationarity & ARIMA

Time series data violates i.i.d. assumptions — observations are temporally dependent. Special methods are required.

### 16.1 Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time. Most models require stationarity.

**Augmented Dickey-Fuller (ADF) test**: H₀ = unit root (non-stationary). p<0.05 → stationary.

**Making stationary**: differencing (subtract previous value), log transform, seasonal differencing.

### 16.2 ACF and PACF

- **ACF (Autocorrelation Function)**: correlation of series with its own lagged values at lag k.
- **PACF (Partial ACF)**: correlation at lag k after removing the effect of shorter lags.

Use these to identify p (AR order from PACF) and q (MA order from ACF) for ARIMA(p,d,q).

### 16.3 ARIMA(p,d,q)

- **AR(p)**: autoregressive — current value depends on p past values.
- **I(d)**: integrated — number of differences needed to achieve stationarity.
- **MA(q)**: moving average — current value depends on q past error terms.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Generate sample time series
np.random.seed(42)
n = 200
# ARIMA(1,1,1) process
errors = np.random.normal(0, 1, n)
y = np.zeros(n)
for t in range(2, n):
    y[t] = y[t-1] + 0.6*(y[t-1]-y[t-2]) + errors[t] - 0.3*errors[t-1]
series = pd.Series(y)

# Stationarity test
adf_result = adfuller(series)
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"ADF p-value: {adf_result[1]:.4f}")  # p<0.05 → stationary

# Difference if needed
diff_series = series.diff().dropna()

# Fit ARIMA
model = ARIMA(series, order=(1,1,1))
fitted = model.fit()
print(fitted.summary())

# Forecast
forecast = fitted.forecast(steps=10)
print(f"Next 10 periods forecast:\n{forecast.values}")
```

> 🏭 **Production note**: For most production forecasting, consider Prophet (Facebook/Meta) or SARIMA for seasonal data. ARIMA works well for stationary series but requires manual order selection. For modern ML-based forecasting, N-BEATS, Temporal Fusion Transformers, or even XGBoost with lag features often outperform ARIMA.

---

## 17. Survival Analysis

Survival analysis handles time-to-event data with **censoring** — observations where the event hasn't occurred by the end of the study period.

### 17.1 Kaplan-Meier Estimator

Non-parametric estimate of the survival function S(t) = P(T > t):
```
Ŝ(t) = Π_{tᵢ≤t} (1 - dᵢ/nᵢ)
```
where dᵢ = number of events at time tᵢ, nᵢ = number at risk just before tᵢ.

### 17.2 Log-Rank Test

Tests H₀: no difference between survival curves. Used to compare survival between groups.

### 17.3 Cox Proportional Hazards

Semi-parametric model: h(t|X) = h₀(t) × exp(Xβ), where h₀(t) is the unspecified baseline hazard. Coefficient exp(β) is a hazard ratio — how much each unit of X multiplies the hazard.

```python
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300
# Simulate churn data
tenure = np.random.exponential(scale=180, size=n)  # days until churn
is_premium = np.random.binomial(1, 0.4, n)
# Premium users have lower hazard (longer tenure)
tenure_adj = tenure * np.where(is_premium, 1.8, 1.0)
observed = (tenure_adj < 365).astype(int)  # churned within 1 year
duration = np.minimum(tenure_adj, 365)

df = pd.DataFrame({'duration': duration, 'churned': observed,
                   'is_premium': is_premium, 'age': np.random.randint(18, 65, n)})

# Kaplan-Meier
kmf = KaplanMeierFitter()
for group, data in df.groupby('is_premium'):
    kmf.fit(data['duration'], data['churned'], label=f'Premium={group}')
    print(f"Median survival (Premium={group}): {kmf.median_survival_time_:.1f} days")

# Log-rank test
premium = df[df.is_premium==1]
free = df[df.is_premium==0]
result = logrank_test(premium.duration, free.duration,
                       premium.churned, free.churned)
print(f"Log-rank p-value: {result.p_value:.4f}")

# Cox PH
cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='churned')
cph.print_summary()
# exp(coef) is the hazard ratio
```

> 🎯 **Interview prep**: "What is censoring in survival analysis?" — Censoring occurs when a subject leaves the study before the event occurs (e.g., a user who doesn't churn before the study ends). Ignoring censored observations (just treating them as "didn't churn") biases estimates downward. KM and Cox handle this correctly.

---

## 18. Causal Inference

### 18.1 Potential Outcomes Framework

For each unit i, define Y_i(1) (outcome if treated) and Y_i(0) (outcome if untreated). The individual treatment effect is Y_i(1)-Y_i(0), but we can never observe both. We observe only the **factual** outcome — the other is the **counterfactual**.

- **ATE** (Average Treatment Effect): E[Y(1)-Y(0)]
- **ATT** (Average Treatment Effect on the Treated): E[Y(1)-Y(0)|T=1]

The fundamental problem of causal inference: we can never estimate individual causal effects, only averages under strong assumptions.

### 18.2 Confounding & Selection Bias

A **confounder** affects both treatment and outcome, creating spurious associations. Example: people who exercise more tend to eat healthier. Naively attributing health outcomes to exercise ignores diet as a confounder.

**Selection bias**: units that select into treatment systematically differ from controls. A restaurant that's been open 10 years isn't a good comparison for a new restaurant — the survivors are different.

### 18.3 Causal Inference Methods

**Difference-in-Differences (DiD)**: compare treatment vs control before and after treatment. Assumes **parallel trends** — without the intervention, both groups would have evolved similarly.

```
DiD = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
```

```python
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

np.random.seed(42)
n_units = 200
unit_id = np.repeat(np.arange(n_units), 2)  # 2 time periods each
time = np.tile([0, 1], n_units)   # 0=pre, 1=post
treatment = (np.arange(n_units) >= n_units//2).astype(int)
treated_unit = np.repeat(treatment, 2)

# DGP: treatment effect = 5
outcome = (2.0 + 3.0*time + 5.0*treated_unit*time
           + np.random.normal(0, 2, n_units*2))

df = pd.DataFrame({'outcome': outcome, 'time': time,
                   'treated': treated_unit,
                   'unit_id': unit_id})

# DiD regression
model = smf.ols('outcome ~ time + treated + time:treated', data=df).fit()
print(model.summary().tables[1])  # interaction term = DiD estimate (~5)
```

**Regression Discontinuity (RD)**: treatment is assigned based on a threshold of a running variable. Units just above and below the threshold are comparable.

**Instrumental Variables (IV)**: find a variable Z that affects treatment T but has no direct effect on outcome Y (exclusion restriction). Z "instruments" exogenous variation in T.

**Propensity Score Matching**: estimate P(T=1|X) and match treated units to controls with similar propensity scores.

```python
# Propensity score matching (simplified)
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
# Confounded: more active users (X) more likely to get treatment AND have better outcomes
X = np.random.normal(0, 1, n)
T = (X + np.random.normal(0, 0.5, n) > 0).astype(int)
Y = 2*X + 3*T + np.random.normal(0, 1, n)  # true ATE=3

# Naive estimate (biased by confounding)
naive_ate = Y[T==1].mean() - Y[T==0].mean()
print(f"Naive ATE: {naive_ate:.2f}")  # will be >3 due to confounding

# Propensity score
ps_model = LogisticRegression()
ps_model.fit(X.reshape(-1,1), T)
propensity = ps_model.predict_proba(X.reshape(-1,1))[:,1]

# IPW (Inverse Propensity Weighting) estimate
weights = np.where(T==1, 1/propensity, 1/(1-propensity))
ipw_ate = (weights * T * Y).sum() / (weights * T).sum() - \
           (weights * (1-T) * Y).sum() / (weights * (1-T)).sum()
print(f"IPW ATE: {ipw_ate:.2f}")  # closer to 3
```

> 🏭 **Production note**: DiD is heavily used at tech companies for evaluating policy changes that can't be A/B tested — product launches in specific cities, pricing changes for specific user segments. Always test the parallel trends assumption using pre-treatment data.

**Resources**
- [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) — excellent free textbook with Python
- [Causal Inference: The Mixtape](https://www.causalinferencebook.net/)

---

## 19. Information Theory for ML

Information theory concepts appear throughout ML: in loss functions, variational inference, model selection, and RL.

### 19.1 Entropy

Shannon entropy measures uncertainty in a probability distribution:
```
H(X) = -Σ p(x) log₂ p(x)
```

**Worked example**: Fair coin (p=0.5): H = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1 bit. Biased coin (p=0.9): H = -0.9×log₂(0.9) - 0.1×log₂(0.1) = 0.469 bits. A certain outcome has H=0.

### 19.2 KL Divergence

KL(P||Q) = Σ P(x) log(P(x)/Q(x)). Measures "how different Q is from P." Always ≥0; equals 0 iff P=Q. **Not symmetric**: KL(P||Q) ≠ KL(Q||P).

Appears in: VAE loss (KL between posterior and prior), RL (PPO's KL penalty), fine-tuning (KL from reference model).

### 19.3 Cross-Entropy

H(P,Q) = -Σ P(x) log Q(x) = H(P) + KL(P||Q)

The **cross-entropy loss** in classification: minimize H(y_true, y_pred). When y_true is one-hot, this reduces to -log(p_correct_class).

### 19.4 Mutual Information

I(X;Y) = H(X) - H(X|Y) = KL(P(X,Y)||P(X)P(Y)). Measures how much knowing Y reduces uncertainty about X. Used in feature selection (sklearn.feature_selection.mutual_info_classif) and information bottleneck.

```python
import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif

def kl_divergence(P, Q):
    """KL(P||Q) in nats."""
    P, Q = np.asarray(P), np.asarray(Q)
    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / Q[mask]))

# Entropy of a discrete distribution
probs_uniform = np.array([0.25, 0.25, 0.25, 0.25])
probs_skewed = np.array([0.7, 0.1, 0.1, 0.1])
print(f"H(uniform): {entropy(probs_uniform, base=2):.3f} bits")  # 2.0
print(f"H(skewed): {entropy(probs_skewed, base=2):.3f} bits")   # ~1.36

# KL divergence
print(f"KL(uniform||skewed): {kl_divergence(probs_uniform, probs_skewed):.3f}")

# Cross-entropy loss (equivalent to NLL)
y_true = np.array([0, 1, 0, 0])  # one-hot
y_pred = np.array([0.1, 0.7, 0.1, 0.1])  # softmax output
cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-8))
print(f"Cross-entropy loss: {cross_entropy:.4f}")  # -log(0.7) ≈ 0.357

# Mutual information for feature selection
np.random.seed(42)
X = np.random.normal(0, 1, (500, 5))
y = (X[:,0] + X[:,1] > 0).astype(int)  # only first 2 features matter
mi = mutual_info_classif(X, y)
print(f"Mutual information: {mi.round(3)}")  # first 2 highest
```

> 🎯 **Interview prep**: "Why do we use cross-entropy loss for classification?" — Because minimizing cross-entropy is equivalent to maximum likelihood estimation under a categorical distribution, and it penalizes confident wrong predictions heavily (log(near-0) → large penalty).

---

## 20. Missing Data & EM Algorithm

### 20.1 Missing Data Mechanisms

Understanding **why** data is missing determines the right strategy:

- **MCAR** (Missing Completely At Random): missingness unrelated to data values. Safe to delete rows, but reduces power.
- **MAR** (Missing At Random): missingness depends on observed variables but not on the missing value itself. Multiple imputation works.
- **MNAR** (Missing Not At Random): missingness depends on the missing value itself (e.g., high-income people don't report income). Most dangerous. Requires domain knowledge to handle.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer

np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age': np.random.randint(20, 70, n),
    'income': np.random.lognormal(10, 0.5, n),
    'score': np.random.normal(50, 15, n)
})

# Introduce MAR missingness (income missing more for younger)
missing_mask = (df['age'] < 35) & (np.random.uniform(0, 1, n) > 0.5)
df.loc[missing_mask, 'income'] = np.nan

print(f"Missing income: {df['income'].isna().sum()} ({df['income'].isna().mean():.2%})")

# Simple imputation (median)
simple_imp = SimpleImputer(strategy='median')
df_simple = df.copy()
df_simple['income'] = simple_imp.fit_transform(df[['income']])

# MICE (Iterative Imputer) — models each feature using others
mice_imp = IterativeImputer(random_state=42, max_iter=10)
df_mice = pd.DataFrame(mice_imp.fit_transform(df), columns=df.columns)

# Add missingness indicator (can carry signal)
df['income_missing'] = df['income'].isna().astype(int)
```

### 20.2 Gaussian Mixture Models & EM Algorithm

GMMs model data as a mixture of K Gaussian components. The EM algorithm iteratively estimates:
- **E-step**: compute P(component k | xᵢ) for each point (soft cluster assignments).
- **M-step**: update parameters (means, covariances, mixture weights) using soft assignments.

Log-likelihood is guaranteed to increase each iteration.

```python
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Generate 3-component mixture
comp1 = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], 200)
comp2 = np.random.multivariate_normal([5, 5], [[1.5, -0.3],[-0.3, 1.5]], 150)
comp3 = np.random.multivariate_normal([0, 8], [[0.8, 0],[0, 0.8]], 100)
X = np.vstack([comp1, comp2, comp3])

# Fit GMM
# BIC/AIC for number of components
for k in [2, 3, 4, 5]:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X)
    print(f"K={k}: BIC={gmm.bic(X):.2f}, AIC={gmm.aic(X):.2f}")

# Fit best model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)  # soft assignments

print(f"Converged: {gmm.converged_}")
print(f"Cluster means:\n{gmm.means_}")
print(f"Mixture weights: {gmm.weights_}")
```

> 🏭 **Production note**: Use BIC (not AIC) for GMM component selection — BIC penalizes complexity more heavily and tends to select more parsimonious models. GMMs are particularly useful for anomaly detection: flag observations with low P(data | model) as anomalies.

---

## 21. The Modern Recipe

Here is the opinionated workflow for applying statistics in production data science:

1. **Always do EDA first**: check distributions, missing rates, correlations. The histogram that saves you from a 3-month false win is worth more than the fanciest model.

2. **For A/B tests**: compute sample size *before* the experiment using `statsmodels.stats.power`. Target 80% power. Run the full duration. Use CUPED to reduce variance. Check SRM before reading results.

3. **For metric choice**: favor metrics with known distributions and low variance (conversion rate > revenue per user for primary metrics). Use bootstrapped CIs for skewed metrics.

4. **For multiple comparisons**: use Benjamini-Hochberg for exploratory feature selection. Use Bonferroni for primary confirmatory tests where false positives are very costly.

5. **For causal claims**: distinguish correlation (coefficient from regression) from causation (requires an experiment or causal method like DiD, IV, or RD). Never say "Feature X causes outcome Y" without a causal design.

6. **For time series**: always test stationarity with ADF before fitting ARIMA. For production forecasting, start with Prophet or ML-based approaches before committing to ARIMA.

7. **For missing data**: never silently drop rows without understanding the missingness mechanism. Add a missingness indicator column when missingness is informative.

8. **Default test selection**:

| Situation | Test |
|---|---|
| Compare 2 group means, normal | Welch's t-test |
| Compare 2 group means, skewed | Mann-Whitney U |
| Compare ≥3 group means | ANOVA + Tukey HSD |
| Compare proportions | z-test for proportions or chi-square |
| Test independence of categoricals | Chi-square independence test |
| Any statistic, no distribution assumed | Bootstrap or permutation test |
| Conversion rate experiment | Bayesian Beta-Binomial or z-test |
| Peeking is required | Sequential test with alpha spending |

---

## 22. References

### Foundational Papers
- Fisher, R.A. (1925). *Statistical Methods for Research Workers.* — the original frequentist significance testing framework
- Neyman, J. & Pearson, E.S. (1933). *On the problem of the most efficient tests of statistical hypotheses.* — hypothesis testing framework
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.* JRSS-B. — the FDR method
- Rubin, D.B. (1974). *Estimating Causal Effects of Treatments.* — potential outcomes framework

### Causal Inference
- Angrist, J. & Pischke, J.S. (2009). *Mostly Harmless Econometrics.* — DiD, IV, RD
- Pearl, J. (2009). *Causality.* — do-calculus and DAGs
- [Matheus Facure: Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/)

### A/B Testing & Experimentation
- [Netflix Tech Blog: Experimentation](https://netflixtechblog.com/experimentation-is-a-major-focus-of-data-science-across-netflix-f67923f8e985)
- [Spotify Engineering: Sequential Testing](https://engineering.atspotify.com/2023/03/choosing-sequential-testing-framework-comparisons-and-discussions)
- [Microsoft: Rules of Thumb for Experimenters](https://www.exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf)
- [Evan Miller: Sample Size Calculator](https://www.evanmiller.org/ab-testing/sample-size.html)

### Libraries & Tools
- [SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html) — the canonical Python stats library
- [Statsmodels](https://www.statsmodels.org/stable/) — regression, time series, power analysis
- [Lifelines](https://lifelines.readthedocs.io/en/latest/) — survival analysis
- [PyMC](https://docs.pymc.io/) — probabilistic programming and MCMC
- [Scikit-learn: impute](https://scikit-learn.org/stable/modules/impute.html) — MICE and simple imputation

### Interactive Learning
- [Seeing Theory (Brown)](https://seeing-theory.brown.edu/) — best visual introduction to probability and statistics
- [Causal Inference: The Mixtape](https://www.causalinferencebook.net/) — free causal inference textbook
