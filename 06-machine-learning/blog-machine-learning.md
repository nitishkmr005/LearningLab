# Machine Learning for Data Scientists and ML Engineers

*A production-first deep-dive covering problem framing, feature engineering, gradient boosting, model evaluation, calibration, and fairness — the end-to-end ML knowledge that separates engineers who train models from engineers who ship them.*

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [A Brief History](#2-a-brief-history)
3. [Problem Framing and Label Definition](#3-problem-framing-and-label-definition)
4. [Data Splits and Leakage Prevention](#4-data-splits-and-leakage-prevention)
5. [Feature Engineering](#5-feature-engineering)
6. [Preprocessing: Imputation, Outliers, and Scaling](#6-preprocessing-imputation-outliers-and-scaling)
7. [Feature Selection](#7-feature-selection)
8. [Gradient Boosting: XGBoost, LightGBM, and CatBoost](#8-gradient-boosting-xgboost-lightgbm-and-catboost)
9. [Model Evaluation Metrics](#9-model-evaluation-metrics)
10. [Calibration and Probability Quality](#10-calibration-and-probability-quality)
11. [Lift Reports and Threshold Selection](#11-lift-reports-and-threshold-selection)
12. [Feature Importance and Model Insight](#12-feature-importance-and-model-insight)
13. [Bias, Fairness, and Sensitive Features](#13-bias-fairness-and-sensitive-features)
14. [The Modern Recipe](#14-the-modern-recipe)
15. [References](#15-references)

---

## 1. The Problem

Most ML projects don't fail because of the model. They fail in the two hours before and the two hours after training. Before: the problem is poorly framed, the labels are leaky, or the features encode future information. After: the evaluation metrics don't match business goals, the model's probabilities are miscalibrated, or no one thought to check whether the model is systematically wrong for certain demographic groups.

A real example: a team builds a churn model with 6-month trailing engagement features. Validation AUC is 0.91 — impressive. But they forgot to exclude users who had already churned before the scoring date from the base population. The model learned to predict past churn, not future churn. It deployed, it failed to drive retention interventions, and it took three months to figure out why. The model was perfectly fine; the problem framing was broken.

This blog covers the full ML pipeline from problem framing to production — not just "here's how to train XGBoost," but how to build ML systems that actually work when deployed.

---

## 2. A Brief History

The history of applied ML is a story of two parallel tracks merging. The academic track evolved from Rosenblatt's perceptron (1958) through backpropagation (Rumelhart et al., 1986), SVMs (Vapnik, 1995), and random forests (Breiman, 2001). The practical data mining track, shaped by the KDD Cup competitions and later Kaggle (founded 2010), developed gradient boosting through Friedman's seminal work (2001), eventually producing XGBoost ([Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)) and LightGBM ([Ke et al., 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)) — the tools that dominate tabular ML production today.

The deep learning revolution (2012–present) captured image, audio, and text. But for tabular data with millions of rows and hundreds of engineered features — churn prediction, fraud detection, propensity modeling, demand forecasting — gradient boosting remains the production default. Surveys of Kaggle competition winners consistently show XGBoost or LightGBM as the backbone of winning solutions for tabular tasks. Recent challenges from TabNet, FT-Transformer, and NODE have not unseated gradient boosting at scale in practice.

---

## 3. Problem Framing and Label Definition

Before any feature engineering or model training, you must precisely define four things. Getting any one wrong will doom the project.

### 3.1 The Four Temporal Anchors

- **Prediction date (as_of_date)**: the date at which the model makes its prediction. Features must use only data before this date.
- **Lookback window**: how far back you look to build features (e.g., "last 12 months of purchase history").
- **Label window (performance window)**: the future period that defines the outcome (e.g., "did the user churn in the next 30 days?").
- **Action window**: when the business can act on the prediction (e.g., "if we predict churn, we'll send a retention email in the next 7 days").

**Example**: Predict whether a SaaS user will cancel their subscription in the next 30 days.

```
Timeline:
|<---- 12 months of features ---->|<---- 30 days ----->|
                                   ^                    ^
                              as_of_date           label_cutoff
```

```python
import pandas as pd
import numpy as np

def create_modeling_table(events_df: pd.DataFrame,
                           scoring_dates: pd.DatetimeIndex,
                           lookback_days: int = 365,
                           label_days: int = 30) -> pd.DataFrame:
    """
    Build a time-aware modeling table.
    Events before as_of_date → features.
    Events after as_of_date but before label_cutoff → labels.
    """
    rows = []
    for as_of_date in scoring_dates:
        label_cutoff = as_of_date + pd.Timedelta(days=label_days)
        feature_start = as_of_date - pd.Timedelta(days=lookback_days)

        # Feature data: strictly before as_of_date
        feature_events = events_df[
            (events_df['event_date'] >= feature_start) &
            (events_df['event_date'] < as_of_date)
        ]
        # Label data: strictly between as_of_date and label_cutoff
        label_events = events_df[
            (events_df['event_date'] >= as_of_date) &
            (events_df['event_date'] < label_cutoff)
        ]

        # Build features per user
        user_features = feature_events.groupby('user_id').agg(
            n_events=('event_id', 'count'),
            last_event=('event_date', 'max'),
        ).reset_index()
        user_features['as_of_date'] = as_of_date
        user_features['recency_days'] = (as_of_date - user_features['last_event']).dt.days

        # Build labels
        churned_users = set(label_events[label_events['event_type']=='cancel']['user_id'])
        user_features['label'] = user_features['user_id'].isin(churned_users).astype(int)

        rows.append(user_features)

    return pd.concat(rows, ignore_index=True)
```

### 3.2 Base Population Design

The base population defines who enters the model. Critically, only include users who are in-scope for the intervention. If you're predicting subscription churn, include only active subscribers as of the scoring date — not already-churned users (leakage), not users on free plans (out of scope).

```python
def build_base_population(users_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """Active subscribers as of as_of_date."""
    as_of = pd.Timestamp(as_of_date)
    return users_df[
        (users_df['subscription_start'] < as_of) &
        ((users_df['subscription_end'].isna()) | (users_df['subscription_end'] >= as_of)) &
        (users_df['plan_type'] != 'free')
    ].copy()
```

> 🎯 **Interview prep**: "How do you define the label for a churn model?" — You need: (1) a clear scoring date (when you make the prediction), (2) a label window (how many days into the future defines "churn"), and (3) a base population (only users who were active at the scoring date). Common mistakes: including already-churned users, using a label that depends on future data, or not aligning the label window with the business's intervention capability.

---

## 4. Data Splits and Leakage Prevention

For time series behavioral data, always split by time — never randomly.

```python
import pandas as pd

def time_based_split(df: pd.DataFrame, date_col: str,
                      train_end: str, val_end: str) -> tuple:
    """
    Train: everything before train_end
    Val: train_end to val_end
    Test: after val_end
    """
    df = df.sort_values(date_col)
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train = df[df[date_col] < train_end_dt]
    val = df[(df[date_col] >= train_end_dt) & (df[date_col] < val_end_dt)]
    test = df[df[date_col] >= val_end_dt]

    print(f"Train: {len(train):,} rows ({df[date_col].min()} to {train_end_dt.date()})")
    print(f"Val:   {len(val):,} rows ({train_end_dt.date()} to {val_end_dt.date()})")
    print(f"Test:  {len(test):,} rows ({val_end_dt.date()} to {df[date_col].max()})")
    return train, val, test

# CRITICAL: fit preprocessing on train only, transform val/test
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

X_train_processed = preprocessing.fit_transform(X_train)  # fit + transform on train
X_val_processed = preprocessing.transform(X_val)           # transform ONLY on val
X_test_processed = preprocessing.transform(X_test)         # transform ONLY on test
```

> 🏭 **Production note**: The most common leakage patterns in production: (1) fitting the scaler/imputer on the full dataset before splitting; (2) computing aggregate statistics on the full dataset and joining to train/val/test; (3) using post-outcome fields as features (e.g., "number of support tickets filed after the churn event"). Always ask: could this feature value be different if the outcome had been different?

---

## 5. Feature Engineering

Time-based feature engineering is the core skill for tabular ML. The pattern: for each user at each scoring date, compute aggregates of historical events.

```python
import pandas as pd
import numpy as np

def build_user_features(events: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Build user-level features using only data before as_of_date."""

    # Safety check: enforce no future leakage
    assert events['event_date'].max() < as_of_date, "Leakage detected!"

    features = (
        events
        .groupby('user_id')
        .agg(
            # Recency
            days_since_last_event=('event_date', lambda x: (as_of_date - x.max()).days),
            # Frequency
            total_events_12m=('event_id', 'count'),
            # Monetary / value
            total_spend_12m=('amount', 'sum'),
            avg_spend=('amount', 'mean'),
            max_spend=('amount', 'max'),
            spend_std=('amount', 'std'),
            # Behavioral diversity
            n_unique_categories=('category', 'nunique'),
            # Trend
            n_events_last_3m=('event_date',
                lambda x: (x >= (as_of_date - pd.Timedelta(days=90))).sum()),
        )
        .reset_index()
    )

    # Derived features
    features['events_per_month'] = features['total_events_12m'] / 12
    features['spend_trend'] = (
        features['n_events_last_3m'] * 4 / features['total_events_12m'].clip(1)
    )  # annualized 3-month rate / full-year rate > 1 means accelerating
    features['log_total_spend'] = np.log1p(features['total_spend_12m'])
    features['recency_bucket'] = pd.cut(
        features['days_since_last_event'],
        bins=[0, 30, 90, 180, 365, np.inf],
        labels=['<30d', '30-90d', '90-180d', '180-365d', '>365d']
    )

    return features
```

### 5.1 Feature Types

| Type | Examples | Notes |
|---|---|---|
| Static | age, country, account_type | Don't change; safe to use as-is |
| Behavioral aggregate | avg_spend_3m, n_logins_30d | Time-window aggregates; main signal |
| Recency | days_since_last_purchase | Often the strongest single predictor |
| Ratio/trend | current_period / prior_period | Captures change, often better than absolute |
| Categorical encoding | target encoding, binary flags | OHE for LR; raw label for trees |

---

## 6. Preprocessing: Imputation, Outliers, and Scaling

### 6.1 Imputation

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer

# Simple imputation (default, fast)
# Numeric: median is more robust than mean for skewed distributions
# Categorical: mode or "UNKNOWN"
from sklearn.compose import ColumnTransformer

numeric_cols = ['age', 'income', 'spend_3m']
categorical_cols = ['category', 'region']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric_cols),
    ('cat', SimpleImputer(strategy='constant', fill_value='UNKNOWN'), categorical_cols),
])

# Always add missingness indicator when missingness carries signal
def add_missing_indicators(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
    return df

# MICE (Iterative Imputer): models each feature using all others
# Better than simple imputation but 10-50× slower
mice = IterativeImputer(random_state=42, max_iter=10, initial_strategy='median')
```

### 6.2 Outlier Handling

```python
import numpy as np

def winsorize(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Clip values to [lower_pct, upper_pct] percentiles. Fit-transform on train only."""
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    return series.clip(lo, hi)

# Log transform for right-skewed monetary features
# log1p handles zero values
df['log_amount'] = np.log1p(df['amount'])

# When to handle outliers:
# Linear/logistic regression: yes, scale-sensitive
# SVM: yes
# Tree-based (XGBoost/LightGBM): mostly no — splits handle extremes naturally
# K-NN, clustering: yes — distance-based
```

### 6.3 Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance — for linear models, neural nets
# RobustScaler: uses median and IQR — robust to outliers
# MinMaxScaler: [0,1] — for neural nets, when distribution bounded
# NOT needed: XGBoost, LightGBM, Random Forest — tree splits are scale-invariant

# Production pattern: use Pipeline to prevent leakage
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('model', LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)  # scaler fits only on X_train
pipe.predict_proba(X_val)   # uses train statistics on val
```

---

## 7. Feature Selection

Start simple. Only add complexity if it's needed.

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectFromModel, mutual_info_classif
)
from lightgbm import LGBMClassifier
import shap

def select_features_pipeline(X_train, y_train, threshold_corr=0.95):
    """
    Multi-step feature selection:
    1. Remove constants
    2. Remove highly correlated pairs
    3. Tree model importance
    4. SHAP for final review
    """
    feature_names = list(X_train.columns)

    # Step 1: Remove zero/near-zero variance features
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X_train)
    feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"After variance filter: {len(feature_names)} features")

    # Step 2: Correlation-based removal (keep one from each correlated pair)
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold_corr)]
    X_df = X_df.drop(columns=to_drop)
    feature_names = list(X_df.columns)
    print(f"After correlation filter: {len(feature_names)} features")

    # Step 3: Tree-based importance
    model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    model.fit(X_df, y_train)

    # Keep top features by permutation importance
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(50).index.tolist()
    print(f"After importance filter: {len(top_features)} features")

    # Step 4: SHAP for final review
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df[top_features].iloc[:1000])
    shap_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=top_features
    ).sort_values(ascending=False)

    return top_features, shap_importance

# Result: typically reduce from 200+ to 30-60 meaningful features
```

---

## 8. Gradient Boosting: XGBoost, LightGBM, and CatBoost

Gradient boosting builds an ensemble of decision trees sequentially. Each new tree corrects the errors of the previous ensemble. The key innovation in XGBoost was the second-order Taylor expansion of the loss, which gives more precise gradient estimates.

### 8.1 The Algorithm

At each step t, fit a new tree h_t to the negative gradient of the loss:
```
F_t(x) = F_{t-1}(x) + η × h_t(x)

where h_t = argmin_h Σ L(yᵢ, F_{t-1}(xᵢ) + h(xᵢ))
```

XGBoost uses a regularized objective:
```
Obj = Σ L(yᵢ, ŷᵢ) + Ω(F)
where Ω(F) = γT + ½λΣwⱼ²
```
T = number of leaves, wⱼ = leaf weights. Regularization prevents overfitting to individual trees.

### 8.2 LightGBM: The Production Default

LightGBM is typically 5-10× faster than XGBoost on large datasets through two innovations:
- **GOSS** (Gradient-based One-Side Sampling): retains high-gradient instances, randomly drops low-gradient instances.
- **EFB** (Exclusive Feature Bundling): bundles mutually exclusive sparse features.

```python
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# LightGBM: the production default for tabular classification
params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 1000,          # use early stopping to find optimal
    'learning_rate': 0.05,
    'num_leaves': 31,              # controls tree complexity (< 2^max_depth)
    'min_child_samples': 20,       # regularization: min samples per leaf
    'subsample': 0.8,              # row subsampling per tree
    'colsample_bytree': 0.8,       # feature subsampling per tree
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 1.0,             # L2 regularization
    'class_weight': 'balanced',    # for imbalanced datasets
    'random_state': 42,
    'verbose': -1,
}

model = lgb.LGBMClassifier(**params)

# Early stopping: train until val AUC stops improving
callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

print(f"Best iteration: {model.best_iteration_}")
print(f"Best val AUC: {model.best_score_['valid_0']['auc']:.4f}")

proba = model.predict_proba(X_val)[:, 1]
```

### 8.3 XGBoost, LightGBM, CatBoost Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Speed on large data | Moderate | Fast (5-10×) | Moderate |
| Categorical handling | Manual encoding | Native (set `categorical_feature`) | Best native (ordered target encoding) |
| Default for most tasks | Good | ✅ Best | Good |
| Tree growth | Level-wise | Leaf-wise (faster, needs tuning) | Symmetric (balanced) |
| Missing value handling | Built-in | Built-in | Built-in |
| GPU support | Yes | Yes | Yes |
| Key advantage | Stability, documentation | Speed, memory | Categoricals, no tuning |

```python
# XGBoost
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', early_stopping_rounds=50,
    enable_categorical=True, tree_method='hist',
    random_state=42, verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# CatBoost: best for datasets with high-cardinality categoricals
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    cat_features=categorical_cols,  # specify categorical columns by name
    eval_metric='AUC', early_stopping_rounds=50,
    verbose=0, random_seed=42
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

---

## 9. Model Evaluation Metrics

### 9.1 Classification Metrics

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
import numpy as np

def evaluate_classifier(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    # Ranking metrics (threshold-independent)
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)  # better for imbalanced classes
    ll = log_loss(y_true, y_proba)

    # KS statistic: max separation between positive and negative CDF
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks = max(tpr - fpr)

    # Threshold-dependent
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

    return {
        'auc': auc, 'pr_auc': pr_auc, 'log_loss': ll, 'ks': ks,
        'precision': precision, 'recall': recall, 'f1': f1,
        'confusion_matrix': cm
    }
```

**Formula: ROC-AUC**
AUC = P(score of random positive > score of random negative). An AUC of 0.87 means: if you pick a random positive (churner) and a random negative (non-churner), the model ranks the churner higher 87% of the time.

**PR-AUC (Average Precision)** is better than ROC-AUC for highly imbalanced datasets (< 5% positive rate). A model with high AUC can have terrible PR-AUC if it outputs well-separated scores but the positives are very rare.

> 🎯 **Interview prep**: "When do you use PR-AUC vs ROC-AUC?" — ROC-AUC can be misleadingly optimistic for very imbalanced classes (e.g., 0.1% fraud rate) because it's influenced by the large number of true negatives. PR-AUC focuses only on the positive class — it's the right metric for fraud detection, rare disease diagnosis, and any task where false positives are acceptable but false negatives are costly.

### 9.2 Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_regressor(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE: sensitive to near-zero values — use with caution
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None)))

    # SMAPE: symmetric MAPE, bounded [0, 2]
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'smape': smape}
```

---

## 10. Calibration and Probability Quality

A model with AUC 0.87 is good at ranking. But if a score of 0.8 only corresponds to 40% actual positives in that band, the model's probabilities are miscalibrated — and any downstream system that trusts those probabilities (capacity planning, cost modeling, email send triggers) will make wrong decisions.

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import numpy as np

def check_calibration(y_true, y_proba, n_bins=10):
    """Reliability diagram: predicted probability vs actual frequency."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    print("Calibration check:")
    print(f"{'Predicted Band':<20} {'Actual Rate':<15} {'Bias':<10}")
    for pred, true in zip(prob_pred, prob_true):
        bias = pred - true
        indicator = "✓" if abs(bias) < 0.05 else "⚠" if abs(bias) < 0.1 else "✗"
        print(f"  {pred:.2f}               {true:.2f}          {bias:+.2f}  {indicator}")

    # Expected Calibration Error (ECE)
    ece = np.abs(prob_true - prob_pred).mean()
    print(f"\nECE (lower is better): {ece:.4f}")
    return prob_true, prob_pred

# Fix miscalibration with Platt scaling or isotonic regression
from sklearn.calibration import CalibratedClassifierCV

# Calibrate using a held-out calibration set (NOT the val set used for training decisions)
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_cal, y_cal)

proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
```

> 🏭 **Production note**: Tree-based models (XGBoost, LightGBM) are typically poorly calibrated by default — they tend to push predictions toward 0 and 1. Always check calibration before using model probabilities for business decisions. Logistic regression is well-calibrated by construction.

---

## 11. Lift Reports and Threshold Selection

### 11.1 Lift Report (Decile Analysis)

Business teams don't think in AUC — they think in "how many responders does the top 10% of scores capture?" Build a lift report for every model.

```python
import pandas as pd
import numpy as np

def lift_report(y_true: pd.Series, y_proba: pd.Series, n_deciles: int = 10) -> pd.DataFrame:
    """
    Create a decile lift report.
    Scores sorted from highest to lowest.
    Decile 1 = highest-scored users.
    """
    df = pd.DataFrame({'target': y_true.values, 'score': y_proba.values})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    df['decile'] = pd.qcut(df.index, q=n_deciles, labels=False) + 1

    baseline_rate = df['target'].mean()

    report = (
        df.groupby('decile')
        .agg(
            n_users=('target', 'count'),
            n_responders=('target', 'sum'),
            min_score=('score', 'min'),
            max_score=('score', 'max'),
        )
        .reset_index()
    )

    report['response_rate'] = report['n_responders'] / report['n_users']
    report['lift'] = report['response_rate'] / baseline_rate
    report['cumulative_responders'] = report['n_responders'].cumsum()
    report['cumulative_capture_rate'] = report['cumulative_responders'] / df['target'].sum()

    return report

# Worked example
np.random.seed(42)
y_true = np.random.binomial(1, 0.15, 10000)
y_proba = np.clip(y_true * 0.5 + np.random.beta(1, 5, 10000), 0, 1)

report = lift_report(pd.Series(y_true), pd.Series(y_proba))
print(report[['decile', 'n_users', 'n_responders', 'response_rate', 'lift',
              'cumulative_capture_rate']].to_string(index=False))
```

### 11.2 Threshold Selection

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

def find_optimal_threshold(y_true, y_proba, business_constraint: str = 'capacity',
                            capacity_pct: float = 0.2) -> float:
    """
    Find threshold under various business constraints.
    """
    if business_constraint == 'capacity':
        # Contact only the top capacity_pct of users
        threshold = np.percentile(y_proba, 100 * (1 - capacity_pct))
    elif business_constraint == 'min_precision':
        # Minimum precision threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # Find threshold with precision >= 0.5
        valid = thresholds[precision[:-1] >= 0.5]
        threshold = valid.min() if len(valid) > 0 else 0.5
    elif business_constraint == 'f1':
        # Maximum F1
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        threshold = thresholds[np.argmax(f1_scores[:-1])]
    return threshold
```

---

## 12. Feature Importance and Model Insight

SHAP (SHapley Additive exPlanations) gives consistent, theoretically grounded feature attributions that explain individual predictions.

```python
import shap
import numpy as np
import pandas as pd

# SHAP: TreeExplainer for tree models (fast, exact)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)  # shape (n_samples, n_features)

# Global feature importance
shap_importance = pd.DataFrame({
    'feature': X_val.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
print(shap_importance.head(20))

# Individual explanation
idx = 42  # a specific user
print(f"User {idx}: predicted probability = {model.predict_proba(X_val.iloc[[idx]])[:, 1][0]:.3f}")
print("Top 5 factors:")
user_shap = pd.Series(shap_values[idx], index=X_val.columns)
print(user_shap.abs().nlargest(5).map(lambda v: f"{v:+.4f}"))

# SHAP dependence plot: how a feature affects predictions
# shap.dependence_plot("days_since_last_login", shap_values, X_val, interaction_index="age")
```

> 🎯 **Interview prep**: "How would you explain why a specific user got a high churn score?" — Use SHAP force plots or waterfall plots to show which features pushed the prediction above/below the baseline expectation. Stakeholders understand "this user got a high score primarily because they haven't logged in for 95 days and their spend dropped 60% last month" far better than "feature importance says recency is important."

---

## 13. Bias, Fairness, and Sensitive Features

A model can be statistically accurate but systematically wrong for specific groups. Auditing fairness is not optional for production ML.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def fairness_audit(y_true, y_proba, sensitive_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance metrics by sensitive group.
    """
    results = []
    threshold = 0.5

    for col in sensitive_features.columns:
        groups = sensitive_features[col].unique()
        for group in sorted(groups):
            mask = sensitive_features[col] == group
            n = mask.sum()
            if n < 50:  # skip very small groups
                continue

            y_t = y_true[mask]
            y_p = y_proba[mask]
            y_pred = (y_p >= threshold).astype(int)

            tp = ((y_pred == 1) & (y_t == 1)).sum()
            fp = ((y_pred == 1) & (y_t == 0)).sum()
            fn = ((y_pred == 0) & (y_t == 1)).sum()

            results.append({
                'feature': col,
                'group': group,
                'n': n,
                'positive_rate': y_t.mean(),
                'auc': roc_auc_score(y_t, y_p) if y_t.nunique() > 1 else np.nan,
                'approval_rate': y_pred.mean(),    # demographic parity
                'precision': tp/(tp+fp) if (tp+fp)>0 else np.nan,
                'recall': tp/(tp+fn) if (tp+fn)>0 else np.nan,   # equalized odds
            })

    return pd.DataFrame(results)

# Check:
# Demographic parity: approval_rate similar across groups
# Equalized odds: recall (TPR) and FPR similar across groups
# Calibration: positive_rate similar to approval_rate within each group
```

> 🏭 **Production note**: Removing a protected attribute (e.g., gender) from the feature set is insufficient if proxy features remain (e.g., occupation, zip code, spending patterns). Always audit the model's output by sensitive group, not just the feature list. Intersectional analysis (gender × age × location) often reveals more severe disparities than single-variable analysis.

---

## 14. The Modern Recipe

The opinionated end-to-end ML build order for tabular classification:

1. **Problem framing first**: write down as_of_date, lookback_window, label_window, and base population definition before touching data.

2. **Time-based split**: train on Jan-Sep, validate on Oct, test on Nov-Dec. Never use random splits for temporal data.

3. **Feature engineering**: recency, frequency, trailing aggregates (3m, 6m, 12m), category diversity, trend ratios. Target ~30-100 features.

4. **Preprocessing inside sklearn Pipeline**: fit imputer + scaler on train only.

5. **Start with LightGBM**: `LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31)` with early stopping. This is the baseline for tabular ML.

6. **Evaluate with multiple metrics**: AUC, PR-AUC, log loss, calibration curve, lift report.

7. **Check calibration**: use isotonic calibration if probabilities are used for business decisions.

8. **SHAP analysis**: identify top features, check for proxy features, and validate feature directions make business sense.

9. **Fairness audit**: check performance by key sensitive groups before deploying.

10. **Threshold selection**: choose based on business capacity constraint, not 0.5.

**Algorithm comparison for tabular data**:

| Algorithm | Speed | Interpretability | Best for |
|---|---|---|---|
| Logistic Regression | Fast | High | Baseline, regulated environments |
| Random Forest | Moderate | Medium | Robust baseline |
| **LightGBM** | **Fast** | **Medium** | **Production default** |
| XGBoost | Moderate | Medium | Stable alternative |
| CatBoost | Moderate | Medium | High-cardinality categoricals |
| Neural Net (MLP) | Slow | Low | Large datasets, many features |
| Linear SVM | Fast | Low | Text classification |

---

## 15. References

### Key Papers
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions.* [SHAP paper](https://arxiv.org/abs/1705.07874)
- Friedman, J. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* — foundational GBM paper

### Libraries & Docs
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
- [SHAP documentation](https://shap.readthedocs.io/en/latest/)
- [Scikit-learn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html)

### Feature Selection Research
- Frénay, B. et al. (2021). *Filter-method benchmark review.* [arXiv:2111.12140](https://arxiv.org/abs/2111.12140)
- Causal feature selection survey: [arXiv:2402.02696](https://arxiv.org/abs/2402.02696)
