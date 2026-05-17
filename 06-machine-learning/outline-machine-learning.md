# 03 — Machine Learning

Exhaustive learning path for classical and modern ML: algorithms, evaluation, feature engineering, and production workflows.

---

## 01 — ML Workflow & Scikit-Learn Pipeline
fit/predict/transform API; Pipeline + ColumnTransformer; train-test split; preventing data leakage.
- https://scikit-learn.org/stable/modules/compose.html
- https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

## 02 — Bias-Variance Tradeoff
Decompose MSE into bias², variance, and irreducible error; underfitting vs overfitting; learning curves.
- https://scott.fortmann-roe.com/docs/BiasVariance.html

## 03 — Linear Regression (implementation)
Closed-form OLS, gradient descent, feature scaling; interpret coefficients; residual analysis.
- https://scikit-learn.org/stable/modules/linear_model.html

## 04 — Logistic Regression (implementation)
Binary classification; MLE; decision boundary; multi-class (OvR, softmax); regularization.
- https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

## 05 — Regularization: Ridge, Lasso, ElasticNet
L2 shrinks; L1 zeroes; ElasticNet blends; RidgeCV/LassoCV for λ selection; sparse feature selection.
- https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression

## 06 — Decision Trees
Information gain, Gini impurity; max_depth, min_samples_leaf; decision boundary visualization; tree pruning.
- https://scikit-learn.org/stable/modules/tree.html
- https://explained.ai/decision-tree-viz/

## 07 — Random Forests
Bagging + feature subsampling; OOB error; feature importances (MDI, permutation); variance reduction.
- https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees

## 08 — Gradient Boosting (XGBoost)
Additive model; gradient in function space; shrinkage; tree depth; early stopping; XGBoost API.
- https://xgboost.readthedocs.io/en/stable/tutorials/model.html
- https://arxiv.org/abs/1603.02754

## 09 — Gradient Boosting (LightGBM & CatBoost)
Histogram-based splitting (LightGBM); native categorical handling (CatBoost); DART; comparison with XGBoost.
- https://lightgbm.readthedocs.io/en/stable/
- https://catboost.ai/docs/

## 10 — Support Vector Machines
Maximum margin classifier; kernel trick (RBF, polynomial); C vs γ trade-off; SVR for regression.
- https://scikit-learn.org/stable/modules/svm.html
- https://cs229.stanford.edu/notes2022fall/cs229-notes3.pdf

## 11 — K-Nearest Neighbors
Distance metrics (Euclidean, Manhattan, cosine); k selection via CV; curse of dimensionality; KD-tree.
- https://scikit-learn.org/stable/modules/neighbors.html

## 12 — Naive Bayes
Gaussian, Multinomial, Bernoulli variants; conditional independence assumption; text classification use case.
- https://scikit-learn.org/stable/modules/naive_bayes.html

## 13 — Model Evaluation Metrics
Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC. Regression: MAE, RMSE, MAPE, R².
- https://scikit-learn.org/stable/modules/model_evaluation.html

## 14 — Cross-Validation
K-fold, stratified K-fold, leave-one-out, time-series split; nested CV for unbiased eval.
- https://scikit-learn.org/stable/modules/cross_validation.html

## 15 — Hyperparameter Tuning
Grid search, random search, Bayesian optimization (Optuna); early stopping; search space design.
- https://optuna.readthedocs.io/en/stable/
- https://scikit-learn.org/stable/modules/grid_search.html

## 16 — Feature Engineering
Polynomial features; interaction terms; target encoding; frequency encoding; date feature extraction.
- https://feature-engine.trainindata.com/

## 17 — Feature Selection
Univariate (chi2, ANOVA F); model-based (RFE, feature importance); SHAP-based; correlation pruning.
- https://scikit-learn.org/stable/modules/feature_selection.html

## 18 — Handling Imbalanced Datasets
Class weights; oversampling (SMOTE, ADASYN); undersampling; PR-AUC over ROC-AUC; threshold tuning.
- https://imbalanced-learn.org/stable/

## 19 — Dimensionality Reduction: PCA
Eigendecomposition; explained variance; scree plot; whitening; when to apply PCA vs feature selection.
- https://scikit-learn.org/stable/modules/decomposition.html#pca

## 20 — Dimensionality Reduction: t-SNE & UMAP
Non-linear reduction for visualization; perplexity in t-SNE; UMAP for faster and structure-preserving reduction.
- https://umap-learn.readthedocs.io/en/latest/
- https://distill.pub/2016/misread-tsne/

## 21 — K-Means Clustering
Lloyd's algorithm; elbow method + silhouette score; k-means++; limitations with non-convex clusters.
- https://scikit-learn.org/stable/modules/clustering.html#k-means

## 22 — DBSCAN & Hierarchical Clustering
Density-based; no need to specify k; detects noise; hierarchical with dendrogram; agglomerative strategies.
- https://scikit-learn.org/stable/modules/clustering.html#dbscan

## 23 — Anomaly Detection
Isolation Forest, Local Outlier Factor, One-Class SVM; unsupervised vs semi-supervised; threshold selection.
- https://scikit-learn.org/stable/modules/outlier_detection.html

## 24 — Calibration
Probability calibration with Platt scaling and isotonic regression; reliability diagrams; when raw model probs are miscalibrated.
- https://scikit-learn.org/stable/modules/calibration.html

## 25 — SHAP & Model Interpretability
SHAP values from game theory; TreeSHAP; waterfall, beeswarm, dependence plots; global vs local explanations.
- https://shap.readthedocs.io/en/latest/
- https://christophm.github.io/interpretable-ml-book/

## 26 — Time Series ML (ARIMA, Prophet)
ARIMA; seasonal decomposition; Prophet for trend + seasonality + holidays; cross-validation for TS.
- https://otexts.com/fpp3/
- https://facebook.github.io/prophet/

## 27 — Neural Networks with PyTorch (Basics)
MLP for tabular data; forward pass, loss, backprop; training loop; BatchNorm, Dropout.
- https://pytorch.org/tutorials/beginner/basics/intro.html

## 28 — MLflow: Experiment Tracking
Log parameters, metrics, artifacts; compare runs; model registry; reproducible experiments.
- https://mlflow.org/docs/latest/index.html

## 29 — Gradient Descent Variants
SGD + momentum; Adagrad (adaptive per-parameter lr); RMSprop; Adam (moment estimates); AdamW (decoupled weight decay); learning rate warm-up; cosine annealing; gradient clipping.
- https://pytorch.org/docs/stable/optim.html
- https://www.ruder.io/optimizing-gradient-descent/

## 30 — Tabular Deep Learning
TabNet (attention masks for feature selection); FT-Transformer (feature tokenizer + transformer); SAINT (intersample attention); when DL beats GBDT on tabular; benchmark on OpenML-CC18.
- https://arxiv.org/abs/1908.07442
- https://arxiv.org/abs/2106.11959

## 31 — AutoML
AutoGluon (stack ensembles, best-in-class); TPOT (genetic programming pipelines); H2O AutoML; meta-learning; when AutoML beats manual tuning; production use at scale.
- https://auto.gluon.ai/stable/index.html
- https://epistasislab.github.io/tpot/

## 32 — Data Validation & Pipeline Testing
Great Expectations (expectations suites, data docs); Pandera (schema + statistical checks on DataFrames); Evidently for data drift detection; testing data pipelines like code.
- https://docs.greatexpectations.io/docs/
- https://pandera.readthedocs.io/en/stable/

## 33 — Feature Stores
Online vs offline store; point-in-time correct joins (prevent leakage); Feast (open source); Tecton; Hopsworks; how feature stores eliminate training-serving skew.
- https://docs.feast.dev/
- https://www.tecton.ai/blog/what-is-a-feature-store/

## 34 — Model Serving & MLOps Patterns
BentoML for packaging models as services; Modal for serverless GPU; Docker + uvicorn for FastAPI model APIs; model versioning; canary deploys; monitoring with Evidently / Arize.
- https://docs.bentoml.com/en/latest/
- https://modal.com/docs/guide
