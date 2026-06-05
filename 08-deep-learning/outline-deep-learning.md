# Deep Learning — Outline

A comprehensive roadmap covering deep learning fundamentals, architectures, and training techniques. Bridges the gap between classical ML (`07-machine-learning`) and the PyTorch framework (`05-pytorch`), focusing on theory, intuition, and design choices that appear in DS/ML/AI interviews.

---

## 01 — Neural Network Fundamentals

Universal approximation theorem, biological motivation, feedforward networks, capacity vs generalisation.

Most popular: MLP with ReLU activations. SotA understanding: scaling laws (Chinchilla) show depth matters less than parameter count × data volume.

- [Universal Approximation Theorem (Hornik et al., 1989)](https://www.sciencedirect.com/science/article/pii/0893608089900208)
- [Deep Learning textbook — Goodfellow, Bengio, Courville (Ch. 6)](https://www.deeplearningbook.org/)

---

## 02 — Activation Functions

Purpose: introduce non-linearity so the network can learn any mapping beyond affine transforms.

| Activation | Formula | Range | Intuition | Use case |
|---|---|---|---|---|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | (0,1) | Squashes to probability; vanishing gradient for |x|≫0 | Binary output layer |
| **Tanh** | tanh(x) = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (−1,1) | Zero-centered sigmoid; still saturates | RNNs, hidden layers |
| **ReLU** | max(0,x) | [0,∞) | No saturation for x>0; dead neurons when x<0 | Default hidden layers |
| **Leaky ReLU** | max(αx, x), α≈0.01 | (−∞,∞) | Fixes dead ReLU with small negative slope | When dead neurons are observed |
| **GELU** | x·Φ(x) | ≈(−0.17,∞) | Smooth, probabilistic gating; better than ReLU for Transformers | BERT, GPT, ViT |
| **SiLU/Swish** | x·σ(x) | ≈(−0.28,∞) | Self-gated, smooth; consistently outperforms ReLU in deep nets | LLaMA, EfficientNet |
| **Softmax** | eˣⁱ/Σeˣʲ | (0,1), sums to 1 | Normalised probability distribution over classes | Multi-class output layer |

SotA: GELU and SiLU dominate modern architectures. ReLU remains default for CNNs due to speed.

- [Gaussian Error Linear Units — Hendrycks & Gimpel (2016)](https://arxiv.org/abs/1606.08415)
- [Searching for Activation Functions (Swish) — Ramachandran et al. (2017)](https://arxiv.org/abs/1710.05941)

---

## 03 — Backpropagation and Automatic Differentiation

Chain rule applied layer-by-layer. Forward pass: compute activations. Backward pass: propagate gradients via `∂L/∂W = ∂L/∂a · ∂a/∂W`. Autograd builds a computation graph and differentiates it.

Key problems: **vanishing gradient** (sigmoid/tanh saturate → gradients → 0 in early layers), **exploding gradient** (gradient grows exponentially). Solutions: residual connections, gradient clipping, better activations.

- [Learning representations by back-propagating errors — Rumelhart et al. (1986)](https://www.nature.com/articles/323533a0)
- [PyTorch Autograd docs](https://pytorch.org/docs/stable/autograd.html)

---

## 04 — Loss Functions

> **The loss function defines what the network optimises — choosing the wrong one leads to training instability, poor calibration, or misaligned objectives.**

### 04.1 — Regression Losses

#### Mean Squared Error (MSE / L2 Loss)
$$L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Intuition:** Penalises large errors quadratically — a prediction off by 2 contributes 4× more than one off by 1. This makes it sensitive to outliers (a single large error dominates the loss). Smooth, everywhere-differentiable gradient makes optimisation easy. **Default for regression.**

---

#### Mean Absolute Error (MAE / L1 Loss)
$$L = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**Intuition:** Each error contributes linearly regardless of magnitude — robust to outliers. Gradient is constant (±1) everywhere except 0, which can slow convergence near the solution since there's no signal about "how wrong" the prediction is. Use when your target has heavy-tailed noise.

---

#### Huber Loss (Smooth L1)
$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta\left(|y - \hat{y}| - \frac{\delta}{2}\right) & \text{otherwise} \end{cases}$$

**Intuition:** Best of both worlds — MSE behaviour for small errors (smooth gradient, fast convergence) and MAE behaviour for large errors (bounded gradient, outlier robustness). δ is a hyperparameter: smaller δ → more like MAE, larger δ → more like MSE. **Standard in object detection** (bounding box regression).

---

#### Log-Cosh Loss
$$L = \sum_{i=1}^n \log\left(\cosh(\hat{y}_i - y_i)\right)$$

**Intuition:** `log(cosh(x)) ≈ x²/2` for small x (like MSE) and `≈ |x| − log(2)` for large x (like MAE). Unlike Huber, it is twice differentiable everywhere — useful for second-order optimisation (L-BFGS). Slightly smoother than Huber at the transition point.

---

#### Quantile Loss (Pinball Loss)
$$L_q = \begin{cases} q \cdot |y - \hat{y}| & \text{if } y \geq \hat{y} \\ (1-q) \cdot |y - \hat{y}| & \text{otherwise} \end{cases}$$

**Intuition:** Optimise for a specific quantile q instead of the mean. q=0.5 gives MAE (median). q=0.9 means: over-predictions are penalised 0.9× while under-predictions are penalised only 0.1× — forcing the model to predict the 90th percentile. **Used in demand forecasting and uncertainty estimation.**

---

### 04.2 — Classification Losses

#### Binary Cross-Entropy (BCE)
$$L = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$$

**Intuition:** Measures the divergence between the true label distribution and the predicted probability. If the model confidently predicts the wrong class (p=0.01 when y=1), the log term → −∞, giving a very large loss. Each prediction is scored independently — positive and negative classes contribute separate log terms. Always use `BCEWithLogitsLoss` in PyTorch (applies sigmoid + BCE numerically stably via log-sum-exp).

---

#### Categorical Cross-Entropy (CCE)
$$L = -\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log p_{i,c}$$

**Intuition:** Since targets are one-hot, only the log-probability of the true class contributes: `L = −log(p_true)`. Equivalent to maximum likelihood estimation. The loss is 0 when the model assigns probability 1 to the correct class and ∞ when it assigns 0. **Default loss for multi-class classification.**

---

#### Sparse Categorical Cross-Entropy (SCCE)
$$L = -\frac{1}{n}\sum_{i=1}^n \log p_{i,\, y_i}$$

where `y_i ∈ {0, 1, …, C−1}` is the **integer class index** (not a one-hot vector).

**Intuition:** Mathematically identical to CCE — only the log-probability of the correct class is summed. The "sparse" refers to the label format: instead of allocating a full one-hot vector of size C for every sample (expensive for large C, e.g. vocabulary of 50,000 tokens), you pass the integer index directly. The loss function looks up `p_{i, y_i}` with a gather operation.

**Gradient (same as CCE):** After softmax + cross-entropy, the gradient w.r.t. the pre-softmax logit `z_j` is:
$$\frac{\partial L}{\partial z_j} = p_j - \mathbf{1}[j = y_i]$$
This is the softmax output minus 1 at the true class position and minus 0 everywhere else. The gradient flows back cleanly with no one-hot materialisation needed.

**When to use SCCE vs CCE:**

| | CCE | SCCE |
|---|---|---|
| Label format | One-hot vector `[0,0,1,0,…]` | Integer index `2` |
| Memory | O(N × C) — costly for large C | O(N) — always small |
| Numerically equivalent | Yes (same math) | Yes |
| Typical use | Segmentation (spatial labels) | Language modelling, image classification |
| Framework | `tf.keras.losses.CategoricalCrossentropy` | `tf.keras.losses.SparseCategoricalCrossentropy` / `nn.CrossEntropyLoss` (PyTorch uses integer targets by default) |

**Note for PyTorch users:** `nn.CrossEntropyLoss` in PyTorch **already expects integer targets** — it is equivalent to Sparse CCE + no one-hot needed. TensorFlow has both CCE and SCCE as separate classes.

---

#### Focal Loss
$$L = -\frac{1}{n}\sum_{i=1}^n \alpha_t (1 - p_t)^\gamma \log(p_t)$$

where `p_t = p` if y=1, else `1−p`.

**Intuition:** Extends BCE with a **modulating factor** `(1−p_t)^γ`. When the model is confident and correct (`p_t → 1`), the factor → 0, down-weighting easy examples. Hard examples (low `p_t`) retain their full gradient signal. α handles class imbalance; γ (typically 2.0) controls how strongly easy examples are suppressed. **Invented for RetinaNet** to handle the extreme foreground/background imbalance in object detection.

- [Focal Loss for Dense Object Detection — Lin et al., 2017](https://arxiv.org/abs/1708.02002)

---

#### Hinge Loss (Multi-class SVM Loss)
$$L = \frac{1}{n}\sum_{i=1}^n \max(0,\ 1 - y_i \cdot f(x_i))$$

Multi-class (Weston-Watkins):
$$L = \frac{1}{n}\sum_{i=1}^n \sum_{j \neq y_i} \max(0,\ s_j - s_{y_i} + \Delta)$$

**Intuition:** The model is penalised only when the margin between the correct class score and wrong class scores is less than Δ (usually 1). If the model already separates the correct class by ≥ Δ, loss = 0 (sparse gradients). Compared to cross-entropy, hinge loss ignores easy examples completely once the margin is satisfied, making it potentially faster but less calibrated.

---

#### Label Smoothing Cross-Entropy
$$L = (1 - \varepsilon) \cdot \text{CE}(y, p) + \frac{\varepsilon}{K} \sum_c \text{CE}(e_c, p)$$

Equivalently: replace hard targets `y=1` with soft targets `y = 1 − ε + ε/K`.

**Intuition:** Prevents the model from becoming overconfident (assigning probability → 1 to one class). A model trained with hard labels is incentivised to push logits to ±∞; label smoothing regularises this, improving calibration and generalisation. ε=0.1 is standard. Introduced in Inception-v3, used in all modern Transformer training.

- [Rethinking the Inception Architecture — Szegedy et al., 2015](https://arxiv.org/abs/1512.00567)

---

### 04.3 — Metric Learning Losses

#### Contrastive Loss
$$L = y \cdot d^2 + (1 - y) \cdot \max(\text{margin} - d,\ 0)^2$$

where `d = ||f(x_1) - f(x_2)||₂`, y=1 if same class.

**Intuition:** For positive pairs (y=1), minimise distance d. For negative pairs (y=0), maximise distance but only if within the margin — pairs already farther apart contribute zero loss. Margin prevents the network from trivially pushing all negatives to ∞. Requires **paired data**, which is often the bottleneck.

---

#### Triplet Loss
$$L = \max\left(d(a, p) - d(a, n) + \text{margin},\ 0\right)$$

where a=anchor, p=positive (same class), n=negative (different class).

**Intuition:** Forces `d(anchor, positive) + margin < d(anchor, negative)` — the anchor must be closer to same-class examples than to different-class examples by at least the margin. Positive triplets (already satisfying the constraint) contribute no loss. **Hard negative mining** is critical — randomly sampled triplets are mostly easy and useless. Used in FaceNet, signature verification.

- [FaceNet — Schroff et al., 2015](https://arxiv.org/abs/1503.03832)

---

#### NT-Xent Loss (SimCLR Contrastive Loss)
$$L_i = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1, k\neq i}^{2N} \exp(\text{sim}(z_i, z_k)/\tau)}$$

**Intuition:** Given a batch of N examples each augmented twice (2N views total), treat each augmented pair as a positive pair and all other 2(N−1) examples as negatives. τ (temperature) controls sharpness: low τ → model focuses on hard negatives; high τ → smoother gradients. Requires **large batch sizes** (4096+) since negatives come from the batch. Foundation of self-supervised vision learning.

- [A Simple Framework for Contrastive Learning — Chen et al., 2020](https://arxiv.org/abs/2002.05709)

---

### 04.4 — Sequence and NLP Losses

#### Negative Log-Likelihood (NLL) Loss
$$L = -\frac{1}{n}\sum_{i=1}^n \log p_\theta(y_i | x_i)$$

**Intuition:** Maximise the probability the model assigns to the correct label. Equivalent to cross-entropy after applying log-softmax. For language models, summed over all tokens in a sequence. Directly interpretable: `exp(NLL) = perplexity`.

---

#### Perplexity
$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log p(w_t | w_{<t})\right)$$

**Intuition:** Geometric mean of the inverse probability the model assigns to each token. PPL=10 means the model is as uncertain as if it were choosing uniformly among 10 options at each step. Lower is better. Not a loss per se but the primary evaluation metric for language models. **Cannot compare PPL across models with different tokenisers.**

---

#### CTC Loss (Connectionist Temporal Classification)
$$L = -\log P(y|x) = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi|x)$$

**Intuition:** For speech/OCR where input length ≫ output length and alignment is unknown. The loss sums over **all valid alignments** between input frames and output tokens (including the blank token). Forward-backward algorithm computes this efficiently. Allows end-to-end training without forced alignment. Powers Whisper, DeepSpeech.

---

### 04.5 — Generative Model Losses

#### KL Divergence
$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

**Intuition:** Measures how much Q diverges from P — extra bits needed to encode P using Q's encoding. **Not symmetric**: `D_KL(P‖Q) ≠ D_KL(Q‖P)`. Zero iff P=Q. Used in VAEs, knowledge distillation, RL policy optimisation. In VAEs: the KL term regularises the latent space to stay close to the prior N(0,I).

---

#### ELBO (VAE Loss / Evidence Lower Bound)
$$\mathcal{L}_\text{ELBO} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{regularisation}}$$

**Intuition:** Two competing terms — the **reconstruction term** pushes the decoder to reproduce the input accurately; the **KL term** pushes the encoder's posterior `q(z|x)` toward the prior `p(z) = N(0,I)`. β-VAE adds a coefficient β>1 to the KL term to enforce a more disentangled latent space. The ELBO is a lower bound on `log p(x)` — maximising it tightens the bound.

- [Auto-Encoding Variational Bayes — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)

---

#### GAN Minimax Loss
$$\min_G \max_D \mathbb{E}_{x\sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]$$

**Intuition:** A two-player game: the discriminator D tries to distinguish real from fake; the generator G tries to fool D. In theory this converges to the generator matching the data distribution. In practice: **mode collapse** (G generates few modes), **vanishing gradients** (D becomes too good), **training instability**.

---

#### Wasserstein Loss (WGAN)
$$L = \underbrace{\mathbb{E}[D(x_\text{real})]}_{\text{maximise}} - \underbrace{\mathbb{E}[D(G(z))]}_{\text{minimise}}$$

Subject to: D must be 1-Lipschitz (enforced via gradient penalty or weight clipping).

**Intuition:** Instead of training D as a classifier, train it as a **critic** scoring realness. The difference of expectations approximates the **Wasserstein-1 distance** (Earth Mover's Distance) between real and generated distributions — the cost of "moving" probability mass to match one distribution to another. Unlike JS divergence, Wasserstein is continuous and provides meaningful gradients even when distributions don't overlap. Dramatically stabilises GAN training.

- [Wasserstein GAN — Arjovsky et al., 2017](https://arxiv.org/abs/1701.04862)

---

### 04.6 — Detection and Segmentation Losses

#### IoU Loss
$$L_\text{IoU} = 1 - \frac{|A \cap B|}{|A \cup B|}$$

**Intuition:** Directly optimises the metric used at evaluation time. `|A∩B|/|A∪B|` is 1 when boxes are identical, 0 when they don't overlap. Problem: gradient is 0 when boxes don't overlap at all — the network gets no signal about direction to move. Solved by GIoU/DIoU/CIoU.

---

#### GIoU Loss (Generalised IoU)
$$L_\text{GIoU} = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|}$$

where C is the smallest enclosing box of A and B.

**Intuition:** Adds a penalty for the gap between the union and the enclosing box. When boxes don't overlap, the penalty still provides a gradient signal that pushes them toward overlap. GIoU ∈ (−1, 1]; IoU ∈ [0, 1].

---

#### Dice Loss
$$L_\text{Dice} = 1 - \frac{2|A \cap B|}{|A| + |B|}$$

**Intuition:** Optimises the F1-score of pixel-level segmentation directly. `2|A∩B|/(|A|+|B|)` is the Sørensen–Dice coefficient, which equals F1. Naturally handles **class imbalance** since it normalises by the total predicted and true foreground pixels — small objects get equal weight. Standard for medical image segmentation. Often combined with BCE: `L = 0.5·BCE + 0.5·Dice`.

---

### 04.7 — RLHF / Alignment Losses

#### RLHF PPO Objective
$$L_\text{PPO} = \mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)A_t\right)\right]$$

where `r_t = π_θ(a|s)/π_\text{old}(a|s)` is the probability ratio and A_t is the advantage.

**Intuition:** Clipping prevents the updated policy from straying too far from the old one — keeping training stable. The min ensures we only use the clipped version when it would be more conservative. PPO-style RLHF adds a KL penalty from the reference model to prevent reward hacking.

---

#### DPO Loss (Direct Preference Optimisation)
$$L_\text{DPO} = -\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

where y_w = preferred response, y_l = less preferred response, β controls deviation from reference.

**Intuition:** Skips the reward model entirely. Directly optimise a classification loss on preference pairs — reward the model for giving higher likelihood to preferred responses relative to the reference policy, while penalising higher likelihood to rejected responses. β=0 means no constraint on deviation from reference; higher β → stays closer to the reference model. Simpler than PPO, widely adopted for instruction tuning.

- [Direct Preference Optimization — Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)

---

### 04.8 — Loss Comparison Table

| Loss | Task | Outlier robust | Calibrated | Sparse gradient | Key hyperparameter |
|---|---|---|---|---|---|
| MSE | Regression | No | Yes | No | — |
| MAE | Regression | Yes | No | Yes | — |
| Huber | Regression | Yes | Yes | No | δ |
| BCE | Binary classification | — | Yes | No | pos_weight |
| CCE | Multi-class | — | Yes | No | — |
| Focal | Imbalanced classification | — | No | Yes | γ, α |
| Hinge | Classification (SVM-style) | — | No | Yes | margin |
| Triplet | Metric learning | — | — | Yes | margin |
| NT-Xent | Self-supervised | — | — | No | τ |
| Dice | Segmentation | — | — | No | — |
| ELBO | VAE | — | — | No | β |
| Wasserstein | GAN | — | — | No | — |
| DPO | LLM alignment | — | — | No | β |

---

## 05 — Optimizers

> **An optimizer updates model parameters to minimise the loss. The choice of optimizer determines training speed, stability, and final model quality.**

### 05.1 — Gradient Descent Variants

| Variant | Update rule | Update frequency | Pros | Cons |
|---|---|---|---|---|
| **Batch GD** | w ← w − η·∇L(all data) | Once per epoch | Stable, exact gradient | Slow, memory intensive |
| **Stochastic GD (SGD)** | w ← w − η·∇L(one sample) | Every sample | Fast updates, escapes local minima | Noisy, high variance |
| **Mini-batch GD** | w ← w − η·∇L(batch of B) | Every B samples | Balance of speed + stability | Requires tuning B |

Mini-batch GD is what "SGD" means in every deep learning framework. Typical batch sizes: 32–512 for images, 512–4096 for language models.

---

### 05.2 — SGD with Momentum

$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

**Intuition:** Accumulates a velocity vector in directions of persistent gradient. The momentum term β (typically 0.9) means 10% new gradient + 90% old velocity — like a ball rolling downhill, accelerating in consistent directions and dampening oscillations in others. Escapes sharp local minima better than vanilla SGD.

**Nesterov Momentum (NAG):** compute gradient at the anticipated position `θ - β·v` instead of current position — "look before you leap." Slightly faster convergence in practice.

---

### 05.3 — RMSProp

$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1-\rho) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

**Intuition:** Adapts the learning rate per-parameter based on a running average of squared gradients. Parameters with large recent gradients get a smaller effective LR; parameters with small gradients get a larger effective LR. Solves AdaGrad's problem of monotonically decreasing LR (uses exponential moving average instead of sum). **Default for RNNs** before Adam became standard.

---

### 05.4 — Adam (Adaptive Moment Estimation)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1st moment — mean)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2nd moment — uncentered variance)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8.

**Intuition:** Combines momentum (m_t tracks gradient direction) with RMSProp (v_t tracks gradient magnitude). Bias correction divides by `(1−β^t)` to account for the fact that m and v are initialised at 0 — without it, early updates are severely biased toward zero. Adam generally converges faster than SGD and requires less LR tuning. **Standard for most tasks.**

- [Adam — Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)

---

### 05.5 — AdamW (Decoupled Weight Decay)

$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

**Intuition:** In Adam, L2 regularisation (`loss += λ||θ||²`) adds `λθ` to the gradient before adaptive scaling — the weight decay is inadvertently rescaled by `1/√v̂`. AdamW decouples weight decay from the gradient update: the penalty `λθ_t` is applied directly to the parameter, not scaled by the adaptive learning rate. This makes weight decay behave the same regardless of gradient magnitude. **Ubiquitous for transformer training; standard in HuggingFace Transformers.**

- [Decoupled Weight Decay (AdamW) — Loshchilov & Hutter, 2017](https://arxiv.org/abs/1711.05101)

---

### 05.6 — LAMB (Layer-wise Adaptive Moments)

$$\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \cdot \frac{\|\theta_t^{(l)}\|}{\|u_t^{(l)}\|} \cdot u_t^{(l)}$$

where `u_t` is the Adam update before the ratio scaling.

**Intuition:** Designed for very large batch training (batch size 32k+). Adam's per-parameter LR causes different layers to have wildly different update norms — LAMB normalises by the ratio of the parameter norm to the update norm, keeping per-layer update scales similar. Enables BERT training in 76 minutes with batch size 65,536.

- [Large Batch Optimization for Deep Learning (LAMB) — You et al., 2019](https://arxiv.org/abs/1904.00962)

---

### 05.7 — Lion (EvoLved Sign Momentum)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(\beta_2 m_{t-1} + (1-\beta_2) g_t) - \lambda \theta_t$$

**Intuition:** Uses sign of the update (±1) rather than the raw gradient magnitude — every parameter gets the same magnitude update. More memory-efficient than Adam (no v_t second moment). Found via evolution search. Competitive with AdamW on vision; less established for LLMs.

- [Symbolic Discovery of Optimization Algorithms (Lion) — Chen et al., 2023](https://arxiv.org/abs/2302.06675)

---

### 05.8 — Learning Rate Schedulers

| Scheduler | Behaviour | Best for |
|---|---|---|
| **Step decay** | LR × factor every N epochs | CNNs, simple training |
| **Cosine annealing** | LR follows cosine from η_max to η_min | Transformers, fine-tuning |
| **Warmup + cosine** | Linear warmup then cosine decay | BERT, GPT, ViT pre-training |
| **One-cycle** | Rise then fall within one cycle | fast.ai, discriminative LR |
| **Polynomial decay** | LR decays as (1−step/total)^p | T5, original Transformer |
| **ReduceLROnPlateau** | Halve LR when val loss stops improving | When training dynamics are unknown |

**Warmup intuition:** at step 0 the model's weights are random — a large LR causes chaotic parameter jumps. Linear warmup (typically 4–10% of total steps) lets the model find a stable region before applying the full LR.

---

### 05.9 — Optimizer Comparison Table

| Optimizer | Adaptive LR | Memory overhead | Best use case |
|---|---|---|---|
| SGD | No | Minimal | CNNs from scratch; ImageNet |
| SGD + Momentum | No | +1 buffer | CNNs, stable convergence |
| RMSProp | Yes | +1 buffer | RNNs, non-stationary problems |
| Adam | Yes | +2 buffers | Default for most tasks |
| AdamW | Yes | +2 buffers | Transformer pre-training & fine-tuning |
| LAMB | Yes | +2 buffers | Large-batch distributed training |
| Lion | Partial (sign) | +1 buffer | Memory-efficient alternative to AdamW |

---

## 06 — Batching, Epochs, and the Training Loop

> **The training loop, batch size, and epoch count are the three knobs that control how the optimizer sees the dataset — they interact with each other and with the optimizer's behaviour.**

### 06.1 — Epoch

An **epoch** is one complete pass through the entire training dataset — every sample has been seen exactly once.

$$\text{Steps per epoch} = \left\lceil \frac{N}{B} \right\rceil$$

where N = total training samples, B = batch size.

**Intuition:** One epoch is not one gradient update — it's N/B updates. A model trains for multiple epochs so each sample is seen many times, allowing the optimizer to refine weights progressively. How many epochs?

- **Too few:** underfitting — the model hasn't seen enough of the data distribution.
- **Too many:** overfitting — the model memorises training data, validation loss diverges from training loss.
- **Early stopping:** monitor validation loss; stop when it stops improving for `patience` epochs. Standard practice to avoid over-specifying epoch count.

**Epoch vs iteration vs step:**

| Term | Meaning |
|---|---|
| **Epoch** | Full pass over the training set |
| **Iteration / step** | One gradient update on one mini-batch |
| **Global step** | Total number of gradient updates since training began |

Example: N=50,000, B=100 → 500 steps per epoch. After 10 epochs: 5,000 total gradient updates.

---

### 06.2 — Batch Size and Its Effects

**Batch size (B)** is the number of training samples used in one forward + backward pass before updating weights.

| Batch size | Gradient quality | Memory | Convergence speed | Generalisation |
|---|---|---|---|---|
| **Small (8–64)** | Noisy (high variance) | Low | Slow (more steps) | Often better (noise as regulariser) |
| **Large (512–8192)** | Low noise (close to true gradient) | High | Fast (fewer steps) | Often worse — sharp minima |
| **Full batch** | Exact gradient | Maximised | 1 step/epoch | Tends to overfit |

**Why large batches can hurt generalisation (sharp vs flat minima):** Large-batch training converges to **sharp minima** — narrow valleys in loss landscape where small perturbations cause large loss increases. Small-batch training's noise "escapes" sharp minima and finds **flat minima** — wide valleys where the model generalises better (Keskar et al., 2017).

**Linear scaling rule:** when multiplying batch size by k, multiply learning rate by k (to compensate for less noisy gradient and fewer steps). Used in ResNet and BERT distributed training.

$$\eta_{\text{new}} = \eta_{\text{base}} \times \frac{B_{\text{new}}}{B_{\text{base}}}$$

**Gradient accumulation:** simulate a larger effective batch by accumulating gradients over multiple forward passes before calling `optimizer.step()`. Enables large-batch training without requiring proportionally large GPU memory.

$$\text{Effective batch size} = B \times \text{accumulation\_steps}$$

---

### 06.3 — Mini-batch Gradient Descent in Detail

**Full forward + backward pass for one batch:**

```
for batch (x, y) in dataloader:
    logits = model(x)          # forward pass
    loss   = criterion(logits, y)
    loss.backward()            # backward pass: compute ∂L/∂θ
    optimizer.step()           # update: θ ← θ - η·∇L
    optimizer.zero_grad()      # clear gradients for next batch
```

**Why `zero_grad()` is required:** PyTorch accumulates gradients by default. Forgetting to zero is a common bug — gradients from previous batches corrupt the current update.

---

### 06.4 — Data Shuffling

**Why shuffle between epochs:** without shuffling the model sees samples in the same order every epoch — leading to correlated mini-batches and slower convergence. Shuffling ensures each mini-batch is a random sample of the dataset, giving an unbiased gradient estimate.

**Drop last:** if `N mod B ≠ 0`, the final batch is smaller. `drop_last=True` discards it — important for Batch Normalisation (which behaves poorly on batch size 1).

---

### 06.5 — Practical Training Loop Decisions

| Decision | Rule of thumb |
|---|---|
| **Batch size** | Start at 32–256; scale up with GPU VRAM |
| **Epochs** | Use early stopping with patience=10–20 |
| **LR warmup steps** | 4–10% of total training steps |
| **Gradient clipping** | `max_norm=1.0` for transformers, optional for CNNs |
| **Validation frequency** | Every epoch for small datasets; every N steps for large |

- [On Large-Batch Training for Deep Learning — Keskar et al., 2017](https://arxiv.org/abs/1609.04836)

---

## 07 — Regularisation

Dropout (standard, variational, DropPath/Stochastic Depth), Batch Normalisation, Layer Normalisation, Group Norm, RMSNorm, Weight Decay (L2), Data Augmentation as implicit regularisation.

Most popular: **LayerNorm** for Transformers, **BatchNorm** for CNNs. SotA: **RMSNorm** (simpler than LayerNorm, used in LLaMA). **DropPath** for vision transformers.

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)
- [Batch Normalization — Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)

---

## 08 — CNN Architectures

AlexNet → VGGNet → Inception → ResNet → EfficientNet → ConvNeXt. Skip connections as the key innovation. Depthwise separable convolutions (MobileNet). Neural architecture search (EfficientNet). Modern: **ConvNeXt** matches ViT on many benchmarks.

- [Deep Residual Learning — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [EfficientNet — Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- [ConvNeXt — Liu et al., 2022](https://arxiv.org/abs/2201.03545)

---

## 09 — Sequence Models: RNN, LSTM, GRU

Vanishing gradient in vanilla RNNs. LSTM gates (input, forget, output) as the solution. GRU as a simpler two-gate alternative. Bidirectional RNNs. Seq2Seq with attention. Largely superseded by Transformers but still used in streaming/low-latency settings.

Most popular: **GRU** for lightweight sequence tasks, **LSTM** for legacy systems. SotA for sequences: Transformer, Mamba (SSM), RWKV.

- [Long Short-Term Memory — Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Mamba — Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)

---

## 10 — Attention and Transformers (Foundations)

Scaled dot-product attention: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`. Multi-head attention. Positional encodings (sinusoidal, rotary/RoPE, ALiBi). Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5). O(n²) complexity problem → sparse attention, FlashAttention.

Covered in depth in `13-llm`.

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [FlashAttention — Dao et al., 2022](https://arxiv.org/abs/2205.14135)

---

## 11 — Transfer Learning and Fine-Tuning

Pretrain on large data, fine-tune on task-specific data. Feature extraction vs full fine-tuning. Catastrophic forgetting and elastic weight consolidation (EWC). Linear probing before fine-tuning. PEFT methods (LoRA, adapters) covered in `13-llm`.

Most popular: **full fine-tuning** for small models, **LoRA** for LLMs. Probing: widely used to understand what representations encode.

- [How transferable are features in deep neural networks? — Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)

---

## 12 — Training Best Practices

Gradient clipping (prevent exploding), learning rate warmup, mixed precision (FP16/BF16), gradient accumulation, early stopping, model EMA, weight initialisation (Xavier/Glorot, He/Kaiming). Debugging: loss NaN, dead neurons, overfit check.

Most popular: **AdamW + cosine schedule + warmup** as universal starting point. **He init** for ReLU networks, **Xavier** for tanh/sigmoid.

- [Practical Deep Learning for Coders — fast.ai](https://course.fast.ai/)
- [Delving Deep into Rectifiers (He init) — He et al., 2015](https://arxiv.org/abs/1502.01852)

---

## 13 — Generative Architectures

GANs (DCGAN, StyleGAN, CycleGAN), VAEs, Normalising Flows, Diffusion Models. DDPM denoising process, DDIM deterministic sampling, classifier-free guidance. Stable Diffusion architecture.

Most popular production: **Diffusion models** (Stable Diffusion, DALL-E 3, Imagen). SotA: **Flux**, **SD3** (flow-matching). GANs largely replaced by diffusion for image generation.

- [Denoising Diffusion Probabilistic Models — Ho et al., 2020](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with LDMs — Rombach et al., 2022](https://arxiv.org/abs/2112.10752)

---

## References

- [Deep Learning textbook — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
- [Dive into Deep Learning — d2l.ai](https://d2l.ai/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Stanford CS231n — CNNs for Visual Recognition](https://cs231n.stanford.edu/)
- [Lilian Weng's blog — Loss Functions](https://lilianweng.github.io/posts/2022-01-11-supervised-contrastive-loss/)
