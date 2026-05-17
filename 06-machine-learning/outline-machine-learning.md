# 06 — Machine Learning

Production-first study outline for tabular ML, feature engineering, validation, thresholding, explainability, fairness, and boosting systems.

---

## Format Used In This Outline
- `Concept`: what to learn.
- `Why it matters`: where teams use it.
- `Production example`: the business shape of the problem.
- `Implementation note`: what to build or watch for.

## 01 — Problem Framing and Label Definition
- `Concept`: define target, prediction point, observation window, performance window, and action window.
- `Why it matters`: most bad ML projects are badly framed before they are badly modeled.
- `Production example`: predict whether a user will attend a seminar in the next 30 days using data available up to the end of the previous month.
- `Implementation note`: write down `as_of_date`, `lookback_window`, `label_window`, and intervention timing explicitly.

## 02 — How To Choose One-Year vs Two-Year History
- `Concept`: choose historical depth based on business cycle, seasonality, user frequency, and data staleness.
- `Why it matters`: too little history misses signal; too much history adds stale behavior and leakage risk.
- `Production example`: for annual renewal prediction, 12 to 24 months may be useful; for fast-moving grocery demand, 90 to 180 days may dominate.
- `Implementation note`: compare model performance and feature stability across 3, 6, 12, and 24 month windows.

## 03 — Time-Based Splits
- `Concept`: train, validation, and test should respect time.
- `Why it matters`: random splits often overestimate performance on behavioral data.
- `Production example`: train on Jan to Sep, validate on Oct, test on Nov to Dec.
- `Implementation note`: never let future events leak into feature creation for earlier rows.

## 04 — Base Population Design
- `Concept`: define who enters the modeling table and when.
- `Why it matters`: the base population determines business applicability and leakage exposure.
- `Production example`: include only active users as of the scoring date, not users who had already churned earlier.

## 05 — Time-Based Feature Engineering
- `Concept`: recency, frequency, rolling windows, lag features, trailing averages, trailing max/min, trend features.
- `Why it matters`: this is the core of most high-performing tabular models.
- `Production example`: 3-month average seminar attendance, 6-month max attendance, days since last login, 12-month spend trend.
- `Implementation note`: use only data strictly before the prediction date.

```python
agg = (
    events[events["event_date"] < events["as_of_date"]]
    .groupby(["user_id", "as_of_date"])
    .agg(
        seminars_3m_sum=("seminars_last_3m", "sum"),
        seminars_3m_max=("seminars_last_3m", "max"),
        seminars_6m_min=("seminars_last_6m", "min"),
    )
)
```

## 06 — Feature Types To Cover
- `Concept`: static features, behavioral aggregates, target-adjacent features, interaction features, category encodings.
- `Why it matters`: strong models usually mix stable identity/context features with recent behavior.
- `Production example`: demographic attributes plus recent engagement plus product-category preferences.

## 07 — Data Leakage
- `Concept`: future information, post-outcome fields, leakage through aggregation windows, leakage through imputation or scaling outside cross-validation.
- `Why it matters`: leakage produces fake wins and failed production launches.
- `Production example`: using "last seminar attended date" when that seminar occurred after the scoring date.
- `Implementation note`: leakage review should happen before model training, not after.

## 08 — Recommended Order Of Operations
- `Concept`: split first, then fit preprocessing only on training data, then transform validation/test.
- `Why it matters`: this avoids train-test contamination.
- `Production example`: time split -> imputation fit on train -> outlier handling thresholds fit on train -> scaling fit on train -> model fit.
- `Implementation note`: put preprocessing inside a pipeline.

## 09 — Imputation
- `Concept`: mean/median for numeric, mode/constant for categorical, model-based imputation, missing indicators.
- `Why it matters`: different missingness mechanisms imply different choices.
- `Production example`: median imputation for transaction amount, `"UNKNOWN"` for occupation, missing flag for income.
- `Implementation note`: add a missingness indicator when missingness itself carries signal.

## 10 — Outlier Handling
- `Concept`: clipping, winsorization, robust scaling, log transform, isolation strategies.
- `Why it matters`: necessary mainly for linear models, distance-based models, and unstable business metrics.
- `Production example`: clip annual spend at the 99.5th percentile before logistic regression.
- `Implementation note`: tree models often tolerate outliers better than linear models.

## 11 — Scaling
- `Concept`: standard scaling, min-max scaling, robust scaling, log scaling.
- `Why it matters`: scaling is important for linear models, SVM, KNN, neural nets; usually not required for tree boosting.
- `Production example`: use `StandardScaler` for logistic regression, skip it for LightGBM.
- `Implementation note`: choose preprocessing based on model family, not habit.

## 12 — Feature Selection: Practical Taxonomy
- `Concept`: filter, wrapper, embedded, permutation-based, SHAP-based, causal/stability-aware methods.
- `Why it matters`: feature selection is not one method; it is a family of tradeoffs among speed, stability, interpretability, and model dependence.
- `Implementation note`: start simple and only escalate when dimensionality, cost, or regulation requires it.

## 13 — Filter Methods
- `Concept`: variance threshold, correlation pruning, mutual information, chi-square, ANOVA F-score, mRMR-style thinking.
- `Why it matters`: fast first pass for wide datasets.
- `Production example`: remove constant fields, near-duplicates, and weak univariate signals before wrapper methods.
- `Implementation note`: do this inside the training fold only.

## 14 — Wrapper Methods
- `Concept`: RFE, RFECV, sequential forward/backward selection.
- `Why it matters`: useful when feature interactions matter and feature count is manageable.
- `Production example`: reduce a 150-feature credit model to a more interpretable 30-feature version.
- `Implementation note`: expensive; use with smaller feature sets or strong priors.

## 15 — Embedded Methods
- `Concept`: Lasso, ElasticNet, tree-based feature importance, `SelectFromModel`.
- `Why it matters`: good balance of usefulness and cost.
- `Production example`: use L1 logistic regression for sparse selection or boosted-tree importances for nonlinear ranking.

## 16 — Modern and State-Of-The-Art Feature Selection Directions
- `Concept`: stability-aware selection, SHAP-guided pruning, permutation importance, causal feature selection, differentiable selection in deep models.
- `Why it matters`: the current direction is not just "which features correlate", but "which features remain stable, actionable, and less spurious".
- `Production example`: use SHAP plus business review to remove proxy features; use causal selection when spurious correlations harm robustness.
- `Implementation note`: recent literature especially emphasizes stability and causal relevance for responsible ML.

## 17 — Recommended Practical Feature-Selection Stack
- `Step 1`: remove constant and duplicate features.
- `Step 2`: correlation pruning plus domain review.
- `Step 3`: mutual information or univariate ranking for a fast screen.
- `Step 4`: tree model plus permutation importance or SHAP.
- `Step 5`: optional RFECV or L1-based selection if simplification is required.
- `Best for tabular production`: correlation pruning + tree model + permutation/SHAP + business review is usually a strong combination.

## 18 — XGBoost and LightGBM Intuition
- `Concept`: gradient boosting builds trees sequentially to correct previous errors.
- `Why it matters`: these are dominant baselines for tabular supervised learning.
- `Production example`: response modeling, churn, fraud, propensity, risk, pricing.
- `Implementation note`: LightGBM is often faster; XGBoost is often the most familiar and stable baseline.

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_valid)[:, 1]
```

## 19 — What Metrics To Calculate After Training
- `Classification`: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, log loss, KS, lift, calibration.
- `Regression`: RMSE, MAE, MAPE, SMAPE, R-squared, residual diagnostics.
- `Ranking/recommenders`: NDCG, MAP, MRR, Recall@K, CTR, coverage.
- `Implementation note`: metrics should reflect actionability, not just abstract score quality.

## 20 — Calibration and Probability Quality
- `Concept`: a good ranking model is not always a well-calibrated probability model.
- `Why it matters`: cutoff selection, cost modeling, and capacity planning depend on good probabilities.
- `Production example`: a score of 0.8 should mean roughly 80% response in that band, not just "high rank".
- `Implementation note`: review calibration plots and consider Platt or isotonic calibration when needed.

## 21 — Lift Report and Decile Analysis
- `Concept`: bucket predictions into 10 or 20 bins from high score to low score and summarize each bucket.
- `Why it matters`: this is how many business teams consume classification models.
- `Production example`: top decile contains 28% responders with 4.1x lift; bottom decile contains almost none.
- `Implementation note`: compute users, responders, response rate, cumulative responders, precision, recall, and lift per bucket.

```python
report = scored.assign(
    decile=pd.qcut(scored["proba"], 10, labels=False, duplicates="drop")
).groupby("decile").agg(
    users=("target", "size"),
    responders=("target", "sum"),
    min_score=("proba", "min"),
    max_score=("proba", "max"),
)
```

## 22 — Selecting Probability Cutoff
- `Concept`: choose threshold from business capacity, cost-benefit, recall needs, or precision targets.
- `Why it matters`: the best threshold is rarely `0.5`.
- `Production example`: marketing can contact only 20% of users, so choose the top two deciles or the threshold that maps to that capacity.
- `Implementation note`: use precision-recall curve, cost matrix, and decile table together.

## 23 — Feature Importance vs Feature Insight
- `Concept`: importance tells you which features matter; insight tells you how they affect the target.
- `Why it matters`: stakeholders ask "what drives the prediction?" not just "what ranked high?".
- `Production example`: feature `days_since_last_attendance` is important, and increasing values decrease response probability sharply after day 45.
- `Implementation note`: combine SHAP summary, dependence plots, monotonicity checks, and bucketed trend tables.

## 24 — Bias, Fairness, and Sensitive Features
- `Concept`: demographic parity, equalized odds, calibration by group, proxy features, fairness slices.
- `Why it matters`: removing only `gender` is not enough if other proxy features recreate the same bias.
- `Production example`: even after removing gender, location and spending proxies may reproduce disparate approval rates.
- `Implementation note`: audit performance by sensitive groups and intersections, not only overall.

## 25 — Leakage and Fairness Together
- `Concept`: some proxy variables are both leaky and unfair.
- `Why it matters`: post-outcome or policy-driven variables can encode both future info and historical bias.
- `Production example`: prior manual review outcome may be highly predictive but can hard-code human bias.

## 26 — Recommended End-To-End Build Order
- `Step 1`: define the prediction date and label window.
- `Step 2`: build the base population.
- `Step 3`: create leakage-safe historical features.
- `Step 4`: split by time.
- `Step 5`: fit preprocessing on train only.
- `Step 6`: train XGBoost or LightGBM baseline.
- `Step 7`: evaluate metrics, calibration, lift, and fairness.
- `Step 8`: create feature insights and threshold recommendations.

## 27 — Minimal Code Assets Worth Adding Later
- `Pipeline with train-only preprocessing`
- `Rolling feature generator`
- `Lift report function`
- `Threshold optimization function`
- `Feature insight report with SHAP`

## Current References
- Scikit-learn feature selection docs: https://scikit-learn.org/stable/modules/feature_selection.html
- XGBoost Python API: https://xgboost.readthedocs.io/en/latest/python/python_api.html
- LightGBM parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- SHAP docs: https://shap.readthedocs.io/en/latest/
- Causal feature selection survey: https://arxiv.org/abs/2402.02696
- Filter-method benchmark review: https://arxiv.org/abs/2111.12140
