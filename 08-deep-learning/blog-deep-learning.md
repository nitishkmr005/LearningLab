# Deep Learning: The Complete Guide

*From neurons and backprop to loss functions, optimizers, CNNs, RNNs, and diffusion models — with formulas, runnable code, and the intuition that matters in interviews and production.*

---

## Table of Contents

1. [Why Deep Learning?](#1-why-deep-learning)
2. [Neural Network Fundamentals](#2-neural-network-fundamentals)
3. [Activation Functions](#3-activation-functions)
4. [Backpropagation and Automatic Differentiation](#4-backpropagation-and-automatic-differentiation)
5. [Loss Functions](#5-loss-functions)
   - [Regression Losses](#51-regression-losses)
   - [Classification Losses](#52-classification-losses)
   - [Metric Learning Losses](#53-metric-learning-losses)
   - [Sequence and NLP Losses](#54-sequence-and-nlp-losses)
   - [Generative Model Losses](#55-generative-model-losses)
   - [Detection and Segmentation Losses](#56-detection-and-segmentation-losses)
   - [RLHF and Alignment Losses](#57-rlhf-and-alignment-losses)
6. [Optimizers](#6-optimizers)
7. [Batching, Epochs, and the Training Loop](#7-batching-epochs-and-the-training-loop)
8. [Regularisation](#8-regularisation)
9. [CNN Architectures](#9-cnn-architectures)
10. [Sequence Models: RNN, LSTM, GRU](#10-sequence-models-rnn-lstm-gru)
11. [Attention and Transformers (Foundations)](#11-attention-and-transformers-foundations)
12. [Transfer Learning and Fine-Tuning](#12-transfer-learning-and-fine-tuning)
13. [Training Best Practices](#13-training-best-practices)
14. [Generative Architectures](#14-generative-architectures)
15. [References](#15-references)

---

## 1. Why Deep Learning?

Classical machine learning requires human-designed features: you hand-engineer edge detectors for images, n-gram counts for text, or frequency features for audio. Deep learning replaces this with a hierarchy of learned representations — each layer of a neural network transforms its input into progressively more abstract features, automatically discovering what matters for the task. A 2012 ImageNet result by Krizhevsky, Sutskever, and Hinton dropped the top-5 error rate from 26% to 15% in one year, a gap that had taken the prior decade to close. Everything since — transformers, diffusion models, foundation models — extends the same core idea: stack parameterised transformations, define a loss, and let gradient descent find the parameters.

The reason deep learning works is the **universal approximation theorem** ([Hornik et al., 1989](https://www.sciencedirect.com/science/article/pii/0893608089900208)): a single hidden layer with enough neurons can approximate any continuous function to arbitrary precision. Depth, rather than width, buys **compositionality** — it is far more parameter-efficient to compose many simple functions than to enumerate all possibilities in a single flat layer. Scaling laws ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)) later quantified this: model loss follows a power law in parameters, data, and compute, giving the field a roadmap for progress.

> 🎯 **Interview prep**: "Why do we need depth if wide networks can also approximate anything?" Answer: efficiency and inductive bias. Composing simple transformations captures the hierarchical structure of natural data (edges → textures → objects) with exponentially fewer parameters than a flat network of equivalent capacity.

---

## 2. Neural Network Fundamentals

A neural network is a directed graph of parameterised operations. The fundamental building block is a **fully connected layer** (also called a dense or linear layer):

$$\mathbf{a} = f\!\left(\mathbf{W}\mathbf{x} + \mathbf{b}\right)$$

where **x** ∈ ℝⁿⁱⁿ is the input, **W** ∈ ℝⁿᵒᵘᵗ×ⁿⁱⁿ is the weight matrix, **b** ∈ ℝⁿᵒᵘᵗ is the bias vector, f is a non-linear activation function, and **a** ∈ ℝⁿᵒᵘᵗ is the output (activation).

**Why non-linearity is not optional.** Consider three layers with identity activation f(x)=x. The composition collapses:

$$\hat{y} = W_3(W_2(W_1 x)) = \underbrace{(W_3 W_2 W_1)}_{\text{one matrix}} x$$

Any depth of linear layers is mathematically equivalent to a single-layer linear model. ReLU's kink at x=0 breaks this — you cannot fold `ReLU(W₂ · ReLU(W₁x))` into a single matrix product because the ReLU applied to intermediate results depends on the actual flowing values, not just the weights.

**Variance and signal stability.** At each layer you compute z = w₁x₁ + … + wₙxₙ. For n independent inputs with variance Var(x) and weights with mean 0 and variance Var(w):

$$\text{Var}(z) = n \cdot \text{Var}(w) \cdot \text{Var}(x)$$

If n · Var(w) ≠ 1, variance compounds geometrically with depth:

| n · Var(w) | After 10 layers | After 50 layers | After 100 layers |
|---|---|---|---|
| 0.9 | 0.35 | 0.005 | vanished |
| 1.0 | 1.0 | 1.0 | 1.0 |
| 1.1 | 2.6 | 117 | 13,780 |

This is the mathematical basis for weight initialisation schemes — they choose Var(w) to make this product ≈ 1 at initialisation.

```python
import torch
import torch.nn as nn

# Simple MLP: 3 hidden layers, 256 units each
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = MLP(in_dim=784, hidden=256, out_dim=10)
x = torch.randn(32, 784)           # batch of 32 MNIST-sized images
logits = model(x)                   # (32, 10) — one score per class
print(logits.shape)                 # torch.Size([32, 10])

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")     # Parameters: 267,018
```

> 🏭 **Production note**: Always verify your initial loss matches theory before training. For 10-class softmax with random init, the expected loss is `−log(1/10) ≈ 2.3`. If you see 0.01 or 100, something is wrong with your data pipeline or initialisation before you've taken a single gradient step. Karpathy calls this "the most important sanity check."

**Resources**
- [Deep Learning textbook — Goodfellow et al.](https://www.deeplearningbook.org/) — Chapters 6-8 cover feedforward networks, regularisation, and optimisation
- [Neural Networks: Zero to Hero — Karpathy](https://karpathy.ai/zero-to-hero.html) — builds backprop from scalar derivatives to GPT, the clearest bottom-up treatment
- [Dive into Deep Learning — d2l.ai](https://d2l.ai/chapter_multilayer-perceptrons/index.html) — interactive code-first textbook

---

## 3. Activation Functions

Non-linear activation functions are what give deep networks their expressive power. The choice of activation has cascading effects on gradient flow, training stability, and final performance — and it's one of the most reliably probed interview topics.

### 3.1 — The Full Roster

| Activation | Formula | Output range | Max gradient | Use case |
|---|---|---|---|---|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | (0, 1) | 0.25 at x=0 | Binary output, gates |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (−1, 1) | 1.0 at x=0 | RNN hidden states |
| **ReLU** | max(0, x) | [0, ∞) | 1 for x>0 | Default CNN/MLP |
| **Leaky ReLU** | max(αx, x), α=0.01 | (−∞, ∞) | 1 for x>0 | When dead neurons observed |
| **ELU** | x if x>0 else α(eˣ−1) | (−α, ∞) | smooth | Faster convergence than Leaky |
| **GELU** | x·Φ(x) | ≈(−0.17, ∞) | smooth | Transformers (BERT, GPT, ViT) |
| **SiLU/Swish** | x·σ(x) | ≈(−0.28, ∞) | smooth | LLaMA, EfficientNet |
| **Softmax** | eˣⁱ/Σeˣʲ | (0,1), sum=1 | — | Multi-class output layer |

### 3.2 — Sigmoid: The Vanishing Gradient Problem

The sigmoid derivative is:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Proof:** Let σ(x) = (1+e⁻ˣ)⁻¹. By the chain rule:

$$\sigma'(x) = -(1+e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x) \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$$

The maximum is 0.25 at x=0 (where σ(0)=0.5, and 0.5×0.5=0.25). This has a devastating consequence for deep networks:

| Sigmoid layers | Maximum gradient scale |
|---|---|
| 4 | 0.25⁴ = 0.004 |
| 8 | 0.25⁸ ≈ 1.5×10⁻⁵ |
| 16 | 0.25¹⁶ ≈ 2.3×10⁻¹⁰ |

Early layers receive near-zero gradients and essentially stop learning. This was the central blocker for deep networks until ReLU and residual connections.

**Tanh is better but not solved.** Tanh's maximum derivative is 1.0, so 1.0ⁿ = 1 — no decay for unsaturated neurons. But tanh still saturates for |x|≫0, just with a 4× slower onset. Tanh also has the zero-centering advantage: sigmoid outputs are always positive, forcing weight gradients to always have the same sign as the upstream error signal — causing zig-zag updates. Tanh is zero-centred, allowing more direct descent.

### 3.3 — ReLU: Simple, Fast, Not Dead

$$f(x) = \max(0, x), \quad f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

For any active neuron, the gradient passes through unchanged — no exponential decay with depth. This is the single most important reason ReLU enabled training of networks deeper than 5-10 layers.

**The dying ReLU problem.** If a neuron receives a large negative weight update, its output becomes permanently 0 (gradient also 0, weights never update). Causes: high learning rates, bad initialisation, no bias. Fixes:

- **Leaky ReLU**: `max(0.01x, x)` — always non-zero gradient
- **ELU**: smooth negative side, mean activations closer to zero
- **GELU**: smooth everywhere, no hard threshold

### 3.4 — GELU and SiLU: The Modern Default

GELU ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)) multiplies the input by its CDF under the standard normal:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where Φ(x) = P(X ≤ x) for X ~ N(0,1). Approximation: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.

SiLU/Swish ([Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)) uses the sigmoid as the gate:

$$\text{SiLU}(x) = x \cdot \sigma(x)$$

Both are smooth everywhere (no kink at zero), which provides cleaner gradient signal and better optimisation landscapes in deep networks. GELU is used in BERT, GPT-2, ViT; SiLU in LLaMA, EfficientNet, and most models after 2023.

```python
import torch
import torch.nn.functional as F
import matplotlib

x = torch.linspace(-4, 4, 200)

activations = {
    "sigmoid": torch.sigmoid(x),
    "tanh":    torch.tanh(x),
    "relu":    F.relu(x),
    "gelu":    F.gelu(x),
    "silu":    F.silu(x),
}

for name, vals in activations.items():
    print(f"{name:10s}  range: [{vals.min():.3f}, {vals.max():.3f}]")
```

### 3.5 — Which Activation to Pick

| Use case | Recommended | Why |
|---|---|---|
| Hidden layers (MLP, CNN) | ReLU | Fast, sparse activations, no vanishing |
| Deep networks / unstable training | Leaky ReLU or ELU | Prevents dead neurons |
| Transformer hidden layers | GELU | Used in every major transformer |
| RNN hidden state | Tanh | Bounded output stabilises recurrent signal |
| LLM hidden layers (2023+) | SiLU | LLaMA-1 onwards |
| Binary output | Sigmoid | Outputs probability in (0,1) |
| Multi-class output | Softmax | Normalised probability distribution |
| Regression output | None (linear) | No squashing needed |

> 🎯 **Interview prep**: "Why is ReLU non-linear if it's just two straight lines?" ReLU violates additivity: `ReLU(5 + (−5)) = 0 ≠ ReLU(5) + ReLU(−5) = 5`. Its kink at x=0 is the source of non-linearity. Stacking thousands of such kinks across a network lets it approximate arbitrarily complex boundaries.

**Resources**
- [GELU paper — Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)
- [Swish / SiLU — Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)
- [d2l.ai activation functions chapter](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)

---

## 4. Backpropagation and Automatic Differentiation

Backpropagation is the algorithm that makes training possible. It efficiently computes the gradient of the loss with respect to every parameter in the network using the chain rule of calculus. The key insight, often missed: backprop is not a special algorithm — it is the chain rule applied systematically to a computation graph.

### 4.1 — Forward and Backward Passes

For a two-layer network with parameters W₁, W₂ and loss L:

**Forward:** z₁ = W₁x, a₁ = ReLU(z₁), z₂ = W₂a₁, L = CrossEntropy(z₂, y)

**Backward (chain rule):**

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot a_1^T$$

$$\frac{\partial L}{\partial W_1} = \left(W_2^T \cdot \frac{\partial L}{\partial z_2}\right) \odot \mathbf{1}[z_1 > 0] \cdot x^T$$

The gradient flows backwards through each operation — each layer "passes" a gradient signal to the layer below, scaled by its own local gradient.

**Why PyTorch's autograd works.** Every tensor operation builds a node in a directed acyclic computation graph. When you call `loss.backward()`, PyTorch traverses this graph in reverse topological order, accumulating `∂L/∂param` for each leaf tensor (parameter) using the stored local gradients from the forward pass.

```python
import torch

# Manual backprop on a scalar graph — matching Karpathy's micrograd
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

z = w * x + b          # z = 3*2 + 1 = 7
a = z.tanh()           # a = tanh(7) ≈ 0.9999
L = (a - 1.0) ** 2    # loss

L.backward()

print(f"dL/dw = {w.grad:.6f}")   # chain rule: dL/da * da/dz * dz/dw
print(f"dL/dx = {x.grad:.6f}")
print(f"dL/db = {b.grad:.6f}")
```

> 🎯 **Interview prep**: "What is the computational complexity of backprop?" The backward pass costs approximately the same as 2× the forward pass — the forward pass stores intermediate activations, and the backward pass computes gradients using them. Memory scales with depth (all activations must be stored), which is why gradient checkpointing trades compute for memory.

### 4.2 — Vanishing and Exploding Gradients

**Vanishing:** Sigmoid/tanh layers multiply gradients by ≤ 0.25 / ≤ 1.0 per layer. At depth 8 with sigmoid, the gradient reaching the first layer is ≤ 0.25⁸ ≈ 1.5×10⁻⁵.

**Exploding:** Weight matrices with spectral radius >1 amplify gradients exponentially going backward. RNNs are especially vulnerable — they apply the *same* weight matrix W at every time step, meaning backpropagating T steps multiplies W by itself T times.

**Solutions:**
- **Better activations**: ReLU passes gradient of 1 for active neurons
- **Residual connections**: gradient has a direct path bypassing blocks (Section 9)
- **Gradient clipping**: cap gradient L2 norm before the update
- **Batch Normalisation**: resets activation variance to ≈1 at each layer (Section 8)
- **LSTM gating**: additive cell state avoids repeated matrix multiplication (Section 10)

### 4.3 — Weight Initialisation

Initialisation sets Var(w) so that n·Var(w) ≈ 1 at the start of training.

**Xavier / Glorot initialisation** (for sigmoid/tanh):

$$\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

Forward pass needs Var(w) = 1/nᵢₙ; backward pass needs 1/nₒᵤₜ. Xavier takes the harmonic mean, balancing both directions for symmetric activations.

**He / Kaiming initialisation** (for ReLU):

$$\text{Var}(w) = \frac{2}{n_{\text{in}}}$$

ReLU zeroes ≈50% of neurons — halving the effective variance every layer. He doubles the variance to compensate for this halving.

**Rule of thumb**: *Does the activation discard half the signal? If yes, use He to compensate. If the activation is symmetric (sigmoid, tanh), use Xavier.*

```python
import torch.nn as nn

# PyTorch applies appropriate init by default, but you can set explicitly
linear = nn.Linear(256, 128)
nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')  # He for ReLU
nn.init.zeros_(linear.bias)

linear_tanh = nn.Linear(256, 128)
nn.init.xavier_uniform_(linear_tanh.weight)   # Xavier for tanh/sigmoid
```

**Resources**
- [Neural Networks: Zero to Hero — Lecture 4 (manual backprop)](https://karpathy.ai/zero-to-hero.html) — the most step-by-step treatment
- [Delving Deep into Rectifiers — He et al., 2015](https://arxiv.org/abs/1502.01852) — derives He init
- [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

---

## 5. Loss Functions

The loss function is the objective the network optimises — it translates your task into a number that gradient descent can minimise. Choosing the wrong loss leads to training instability, poor calibration, or a model that optimises the wrong thing entirely. The loss function is not an afterthought; it *is* the specification of what you want the model to learn.

### 5.1 — Regression Losses

#### Mean Squared Error (MSE / L2 Loss)

$$L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Worked example:** true=[3.0, 1.0], pred=[2.5, 1.5] → MSE = ((3−2.5)² + (1−1.5)²)/2 = (0.25+0.25)/2 = **0.25**

**Intuition:** Penalises large errors quadratically — an error of 2 contributes 4× more than an error of 1. Makes the model minimise the *mean* of the squared errors, which equals the conditional mean E[y|x]. This also means one large outlier can dominate the loss. Smooth gradient everywhere: ∂L/∂ŷᵢ = −2(yᵢ−ŷᵢ)/n. **Default for regression.**

---

#### Mean Absolute Error (MAE / L1 Loss)

$$L = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**Worked example:** true=[3.0, 1.0], pred=[2.5, 1.5] → MAE = (|3−2.5| + |1−1.5|)/2 = (0.5+0.5)/2 = **0.5**

**Intuition:** Each error contributes linearly — robust to outliers. The gradient is a constant ±1 regardless of error magnitude, which can slow convergence near the solution (no signal about "how wrong"). Optimises for the conditional median E[median(y)|x], not the mean. Use when targets have heavy-tailed noise.

---

#### Huber Loss (Smooth L1)

$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta\!\left(|y - \hat{y}| - \frac{\delta}{2}\right) & \text{otherwise} \end{cases}$$

**Worked example** (δ=1): error=0.5 → MSE regime: ½(0.5)² = **0.125**. Error=2.0 → MAE regime: 1×(2.0−0.5) = **1.5**.

**Intuition:** MSE behaviour for small errors (smooth gradient, fast convergence near solution), MAE behaviour for large errors (bounded gradient, outlier-robust). δ is a hyperparameter: smaller δ → more MAE-like, larger δ → more MSE-like. Default δ=1. **Standard in object detection bounding-box regression.**

---

#### Log-Cosh Loss

$$L = \sum_{i=1}^n \log\!\left(\cosh(\hat{y}_i - y_i)\right)$$

**Intuition:** log(cosh(x)) ≈ x²/2 for small x (like MSE) and ≈ |x|−log(2) for large x (like MAE). Unlike Huber, it is *twice differentiable everywhere* — useful for second-order optimisers (L-BFGS). Slightly smoother transition than Huber at the δ boundary.

---

#### Quantile Loss (Pinball Loss)

$$L_q = \frac{1}{n}\sum_i \begin{cases} q \cdot |y_i - \hat{y}_i| & y_i \geq \hat{y}_i \\ (1-q) \cdot |y_i - \hat{y}_i| & y_i < \hat{y}_i \end{cases}$$

**Intuition:** Optimise for a specific quantile. q=0.5 gives MAE (predicts the median). q=0.9 means over-predictions are penalised 0.9× while under-predictions are penalised only 0.1× — forcing the model to predict the 90th percentile. **Used in probabilistic forecasting (demand planning, prediction intervals).**

**Regression loss comparison:**

| Loss | Outlier robust | Gradient near solution | Optimises for | Typical δ / q |
|---|---|---|---|---|
| MSE | No | Smooth | Mean | — |
| MAE | Yes | Constant ±1 | Median | — |
| Huber | Yes | Smooth | Mean (soft) | δ=1 |
| Log-Cosh | Yes | Smooth | Mean (soft) | — |
| Quantile | — | Asymmetric | q-th quantile | q∈(0,1) |

```python
import torch
import torch.nn as nn

y_true = torch.tensor([3.0, 1.0, 2.0])
y_pred = torch.tensor([2.5, 1.5, 2.8])

mse  = nn.MSELoss()(y_pred, y_true)
mae  = nn.L1Loss()(y_pred, y_true)
hub  = nn.HuberLoss(delta=1.0)(y_pred, y_true)

print(f"MSE:   {mse.item():.4f}")    # MSE:   0.2967
print(f"MAE:   {mae.item():.4f}")    # MAE:   0.4333
print(f"Huber: {hub.item():.4f}")    # Huber: 0.1483
```

---

### 5.2 — Classification Losses

#### Binary Cross-Entropy (BCE)

$$L = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$$

**Worked example:** y=1, p=0.8 → L = −[1·log(0.8) + 0] = −log(0.8) ≈ **0.223**. y=1, p=0.01 (confident wrong) → L = −log(0.01) ≈ **4.6** — the loss grows logarithmically, forcing the model to move.

**Intuition:** Measures divergence between the true label distribution and the predicted probability. The positive and negative class contribute separate log terms — each independent prediction is scored independently. Always use `BCEWithLogitsLoss` in PyTorch (applies sigmoid + BCE via numerically stable log-sum-exp; plain `BCELoss` can produce NaN).

---

#### Categorical Cross-Entropy (CCE)

$$L = -\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log p_{i,c}$$

**Worked example:** C=3, y=[0,1,0], p=[0.1, 0.7, 0.2] → L = −log(0.7) ≈ **0.357**. Only the true class contributes since y is one-hot.

**Intuition:** Generalises BCE to C classes. Since targets are one-hot, this simplifies to L = −log(p_true) — equivalent to maximum likelihood estimation on a categorical distribution. The loss is 0 when the model assigns probability 1 to the correct class and ∞ when it assigns 0. **Default loss for multi-class classification.**

---

#### Sparse Categorical Cross-Entropy (SCCE)

$$L = -\frac{1}{n}\sum_{i=1}^n \log p_{i,\, y_i}$$

where yᵢ ∈ {0, 1, …, C−1} is the **integer class index**.

**Intuition:** Mathematically identical to CCE — the "sparse" refers to the label format. Instead of a one-hot vector of size C, you pass the integer index. The loss function looks up `p_{i, y_i}` with a gather operation. For large C (vocabulary of 50,000 tokens), this saves enormous memory: O(N) labels vs O(N×C) one-hot vectors.

**Gradient:** After softmax + cross-entropy, the gradient w.r.t. logit zⱼ is:

$$\frac{\partial L}{\partial z_j} = p_j - \mathbf{1}[j = y_i]$$

Softmax output minus 1 at the true class position, minus 0 everywhere else. The gradient flows cleanly without materialising any one-hot tensor.

| | CCE | SCCE |
|---|---|---|
| Label format | One-hot `[0,0,1,0,…]` | Integer `2` |
| Memory | O(N × C) | O(N) |
| Numerically equivalent | Yes | Yes |
| PyTorch equivalent | `nn.CrossEntropyLoss` expects integer targets (= SCCE) | Same |
| TensorFlow | `CategoricalCrossentropy` | `SparseCategoricalCrossentropy` |

> 🎯 **Interview prep**: "What's the difference between CCE and SCCE?" Same math, different label format. SCCE is strictly more memory-efficient for large output spaces. PyTorch's `nn.CrossEntropyLoss` uses integer targets by default — it *is* SCCE.

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1],    # batch of 2
                        [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])              # integer class indices (SCCE)

criterion = nn.CrossEntropyLoss()           # applies log_softmax + NLL internally
loss = criterion(logits, targets)
print(f"SCCE loss: {loss.item():.4f}")      # SCCE loss: 0.4741

# Verify manually
probs = torch.softmax(logits, dim=1)
manual = -torch.log(probs[0, 0]) - torch.log(probs[1, 1])
print(f"Manual:    {(manual / 2).item():.4f}")   # same: 0.4741
```

---

#### Focal Loss

$$L = -\frac{1}{n}\sum_{i=1}^n \alpha_t (1 - p_t)^\gamma \log(p_t)$$

where `p_t = p` if y=1, else `1−p`. Default: α=0.25, γ=2.

**Worked example:** γ=2, correct prediction with p=0.9 → modulating factor (1−0.9)²=0.01 — this easy example contributes only 1% of its original loss. Hard example p=0.3 → (1−0.3)²=0.49 — nearly unchanged.

**Intuition:** Extends BCE with a modulating factor `(1−pₜ)^γ`. When the model is confident and correct (pₜ→1), the factor→0, effectively down-weighting easy examples. Hard examples (low pₜ) retain their full gradient signal. Invented for RetinaNet ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) to handle the extreme foreground/background imbalance in object detection: in a 640×640 image, there may be 3 foreground objects among 200,000 background anchor boxes.

---

#### Hinge Loss

$$L = \frac{1}{n}\sum_{i=1}^n \max(0,\ 1 - y_i \cdot f(x_i))$$

Multi-class (Weston-Watkins): $L = \frac{1}{n}\sum_i \sum_{j \neq y_i} \max(0,\ s_j - s_{y_i} + \Delta)$

**Intuition:** Penalises only when the margin between correct and incorrect class scores is below Δ (usually 1). If the model already separates classes by ≥ Δ, gradient is exactly 0. Sparse gradients: easy examples contribute nothing. The model is incentivised to find a maximum-margin separator — the SVM objective.

---

#### Label Smoothing Cross-Entropy

Replaces hard targets y=1 with soft targets y = (1−ε) + ε/C, and y=0 with ε/C.

$$L = (1-\varepsilon) \cdot \text{CE}(y, p) + \frac{\varepsilon}{C} \sum_c \text{CE}(e_c, p)$$

**Intuition:** Prevents the model from becoming overconfident. Without smoothing, the model is incentivised to push logits to ±∞ (probability 1 for the true class), which harms calibration. ε=0.1 is standard. Introduced in Inception-v3 ([Szegedy et al., 2015](https://arxiv.org/abs/1512.00567)); used in every modern transformer training recipe.

```python
# Label smoothing is built into PyTorch's CrossEntropyLoss
criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_smooth = criterion_smooth(logits, targets)
print(f"Label-smoothed loss: {loss_smooth.item():.4f}")
```

---

### 5.3 — Metric Learning Losses

#### Contrastive Loss

$$L = y \cdot d^2 + (1-y) \cdot \max(\text{margin} - d,\ 0)^2$$

where d = ||f(x₁) − f(x₂)||₂, y=1 if same class.

**Intuition:** For positive pairs (y=1), minimise embedding distance d. For negative pairs (y=0), push distance beyond a margin — pairs already farther apart contribute zero loss. Requires *paired* labelled data, which is often the bottleneck.

---

#### Triplet Loss

$$L = \max\!\left(d(a, p) - d(a, n) + \text{margin},\ 0\right)$$

where a=anchor, p=positive, n=negative.

**Worked example:** d(a,p)=0.4, d(a,n)=0.9, margin=0.3 → max(0.4−0.9+0.3, 0) = max(−0.2, 0) = **0** (satisfied). d(a,n)=0.5 → max(0.4−0.5+0.3, 0) = **0.2** (violated, loss>0).

**Intuition:** Forces the anchor to be closer to positive examples than negative examples by at least the margin. Hard negative mining is critical — randomly sampled triplets are mostly easy and provide zero gradient. Used in FaceNet ([Schroff et al., 2015](https://arxiv.org/abs/1503.03832)).

---

#### NT-Xent Loss (SimCLR)

$$L_i = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1, k\neq i}^{2N} \exp(\text{sim}(z_i, z_k)/\tau)}$$

**Intuition:** Given N examples each augmented twice (2N views), treat each augmented pair as a positive pair and all other 2(N−1) examples as negatives. τ (temperature) controls sharpness: low τ→model focuses on hard negatives; high τ→smoother gradients. Requires large batch sizes (4096+ in the original paper) since negatives come from the batch. Foundation of self-supervised vision representation learning ([Chen et al., 2020](https://arxiv.org/abs/2002.05709)).

---

### 5.4 — Sequence and NLP Losses

#### Negative Log-Likelihood (NLL) Loss

$$L = -\frac{1}{n}\sum_{i=1}^n \log p_\theta(y_i | x_i)$$

**Intuition:** Maximise the probability the model assigns to correct labels. Equivalent to cross-entropy after log-softmax. `exp(NLL) = perplexity`.

---

#### Perplexity

$$\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log p(w_t | w_{<t})\right)$$

**Worked example:** If a language model assigns average log-probability −2.3 (log₁₀) per token, perplexity ≈ 10 — the model is as uncertain as if choosing uniformly among 10 options per token. Lower is better.

**Caution:** Cannot compare perplexity across models with different tokenisers — the token count T differs even on the same text.

---

#### CTC Loss (Connectionist Temporal Classification)

$$L = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi | x)$$

**Intuition:** For tasks where input and output lengths differ and alignment is unknown (speech → text, image → text). Sums over *all valid alignments* using a blank token. The forward-backward algorithm computes this efficiently in O(T·L) where T is input length and L is output length. Powers Whisper, DeepSpeech, end-to-end OCR.

---

### 5.5 — Generative Model Losses

#### KL Divergence

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

**Worked example:** P=[0.9,0.1], Q=[0.5,0.5] → D_KL = 0.9·log(0.9/0.5) + 0.1·log(0.1/0.5) ≈ 0.9×0.588 + 0.1×(−1.609) ≈ 0.529 − 0.161 ≈ **0.368** nats.

**Intuition:** Extra bits needed to encode samples from P using Q's code. *Not symmetric*: D_KL(P‖Q) ≠ D_KL(Q‖P). Zero iff P=Q. Used in VAEs (regularise posterior to match prior), knowledge distillation (match student to teacher), and RL policy optimisation.

---

#### ELBO — VAE Loss

$$\mathcal{L}_\text{ELBO} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}\!\left(q(z|x) \| p(z)\right)}_{\text{regularisation}}$$

**Intuition:** Two competing forces. The **reconstruction term** pushes the decoder to reproduce the input; the **KL term** pushes the encoder's posterior q(z|x) toward the prior p(z)=N(0,I), keeping the latent space structured. β-VAE multiplies the KL term by β>1 to enforce disentanglement — higher β → more independent latent dimensions, worse reconstruction quality. The ELBO is a lower bound on log p(x); maximising it tightens the bound ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)).

---

#### GAN Minimax Loss

$$\min_G \max_D \mathbb{E}_{x\sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]$$

**Intuition:** A zero-sum game: D tries to distinguish real from fake, G tries to fool D. In theory, the Nash equilibrium has G match the data distribution. In practice: **mode collapse** (G generates only a few modes), **vanishing gradients** (D becomes too good, G gets no signal), **training instability**. These failures motivated Wasserstein GAN.

---

#### Wasserstein Loss (WGAN)

$$L = \mathbb{E}[D(x_\text{real})] - \mathbb{E}[D(G(z))]$$

Subject to: D must be 1-Lipschitz (enforced via gradient penalty).

**Intuition:** Trains D as a *critic* that scores realness, not a binary classifier. The difference of expectations approximates the Wasserstein-1 distance — the cost of "transporting" one distribution to match the other. Unlike JS divergence, Wasserstein is continuous and provides meaningful gradients even when distributions don't overlap. **Dramatically stabilises GAN training** ([Arjovsky et al., 2017](https://arxiv.org/abs/1701.04862)).

---

### 5.6 — Detection and Segmentation Losses

#### IoU Loss

$$L_\text{IoU} = 1 - \frac{|A \cap B|}{|A \cup B|}$$

**Intuition:** Directly optimises the metric used at test time. Problem: gradient is 0 when boxes don't overlap at all — no direction signal for the network to move.

---

#### GIoU Loss

$$L_\text{GIoU} = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|}$$

where C is the smallest enclosing box. Adds a penalty for the gap between the union and the enclosing box — provides gradient even when boxes don't overlap.

---

#### Dice Loss

$$L_\text{Dice} = 1 - \frac{2|A \cap B|}{|A| + |B|}$$

**Intuition:** Optimises the F1 score of pixel-level segmentation directly. Naturally handles class imbalance — small objects get equal weight since the loss normalises by total predicted and true foreground pixels. Standard for medical image segmentation. Often combined with BCE: `L = 0.5·BCE + 0.5·Dice`.

---

### 5.7 — RLHF and Alignment Losses

#### PPO Clip Objective (RLHF)

$$L_\text{PPO} = \mathbb{E}\!\left[\min\!\left(r_t A_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)A_t\right)\right]$$

where `r_t = π_θ(a|s)/π_old(a|s)` is the probability ratio and Aₜ is the advantage.

**Intuition:** Clipping prevents the updated policy from straying too far from the old one. The min ensures we only use the clipped version when it would be more conservative. RLHF adds a KL penalty from the reference model to prevent reward hacking.

---

#### DPO Loss (Direct Preference Optimisation)

$$L_\text{DPO} = -\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

where y_w = preferred response, y_l = rejected response.

**Intuition:** Skips the separate reward model entirely. Directly optimises a classification loss on preference pairs — reward the model for giving higher likelihood to preferred responses *relative to the reference policy*, while penalising higher likelihood to rejected responses. β controls deviation from reference (β→0 means unconstrained). Matches or exceeds PPO on summarisation and dialogue tasks with far less implementation complexity ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

> 🎯 **Interview prep**: "When would you use Focal Loss vs standard BCE?" When the dataset has severe class imbalance — specifically when the ratio of easy negatives to hard positives is >100:1. If the imbalance is mild, class weighting via pos_weight in BCEWithLogitsLoss is simpler. Focal Loss is the standard choice for one-stage object detection.

> 🏭 **Production note**: Loss NaN is almost always caused by one of three things: (1) log(0) from predicting exactly 0 probability — use log-sum-exp stability tricks; (2) exploding gradients — add gradient clipping; (3) learning rate too high — reduce by 10× and check loss at step 0.

**Complete loss selection guide:**

| Task | Loss | Notes |
|---|---|---|
| Regression, balanced | MSE | Default |
| Regression, outliers present | Huber | δ=1 |
| Regression, probabilistic | Quantile | For prediction intervals |
| Binary classification | BCEWithLogitsLoss | Never use BCELoss directly |
| Multi-class, one-hot labels | CCE | |
| Multi-class, integer labels | SCCE / CrossEntropyLoss | More memory-efficient |
| Imbalanced classification | Focal (γ=2) | Detection, rare events |
| Embedding learning | Triplet or NT-Xent | With hard negative mining |
| Self-supervised vision | NT-Xent | Large batch (4096+) |
| Segmentation | Dice + BCE | Medical imaging |
| Generative (VAE) | ELBO | β-VAE for disentanglement |
| Generative (GAN) | Wasserstein + GP | Stable over minimax |
| LLM training | NLL / cross-entropy | = SCCE on token predictions |
| LLM alignment | DPO | Simpler than PPO+RM |

---

## 6. Optimizers

The optimizer translates gradients into parameter updates. A bad optimizer choice can mean training takes 10× longer, gets stuck in sharp minima, or diverges entirely. The choices have compounded over the years from vanilla SGD to adaptive methods that require almost no learning-rate tuning.

### 6.1 — Gradient Descent Variants

The fundamental distinction is *how much data* you use per update:

| Variant | Update rule | Gradient quality | Memory | When to use |
|---|---|---|---|---|
| **Batch GD** | w ← w − η·∇L(all data) | Exact | Full dataset in memory | Rare — small datasets only |
| **Stochastic GD** | w ← w − η·∇L(one sample) | Very noisy | Minimal | Not used directly in DL |
| **Mini-batch GD** | w ← w − η·∇L(batch B) | Good, unbiased | One batch | Universal — this is what "SGD" means |

### 6.2 — SGD with Momentum

$$v_t = \beta v_{t-1} + \nabla L(\theta_t), \quad \theta_{t+1} = \theta_t - \eta \cdot v_t$$

**Worked example:** β=0.9, ηg=0.1, gradient=1.0 each step.
- Step 1: v₁ = 0 + 1.0 = 1.0, update = 0.1
- Step 2: v₂ = 0.9×1.0 + 1.0 = 1.9, update = 0.19
- Step 10: vₜ → 1/(1−0.9) = 10.0 — effective LR 10× higher in consistent directions

**Intuition:** Like a ball rolling downhill — accelerates in directions of consistent gradient, damps oscillations in directions of conflicting gradients. β=0.9 means 90% old velocity, 10% new gradient. **Nesterov** computes the gradient at the *anticipated* position (θ − β·v) — "look before you leap" — converging slightly faster.

### 6.3 — RMSProp

$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1-\rho) \cdot g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

**Intuition:** Adapts the learning rate per-parameter based on a running average of squared gradients. High recent gradient → smaller effective LR. Low recent gradient → larger effective LR. Solves AdaGrad's problem of monotonically shrinking LR (uses exponential moving average). **Default for RNNs** before Adam.

### 6.4 — Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1st moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2nd moment)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default: β₁=0.9, β₂=0.999, ε=1e-8.

**Worked example** (step 1): m₁=0.1·g, v₁=0.001·g². With bias correction: m̂₁=m₁/0.1=g, v̂₁=v₁/0.001=g². Update = η·g/|g| — pure sign! Adam normalises magnitude completely on step 1.

**Intuition:** Combines momentum (mₜ tracks direction) with RMSProp (vₜ tracks magnitude). Bias correction divides by (1−βᵗ) to account for zero-initialisation — without it, early updates would be biased toward zero. Adam converges faster than SGD and requires less LR tuning. Karpathy's rule: *"Adam at LR=3e-4 is a safe default for early-stage experiments."* ([Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980))

### 6.5 — AdamW

$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

**Intuition:** In Adam with L2 regularisation, weight decay is added to the gradient *before* adaptive scaling — the decay is inadvertently rescaled by 1/√v̂. AdamW **decouples** weight decay from the gradient update: the penalty λθₜ is applied directly to parameters, not scaled adaptively. This makes weight decay behave consistently regardless of gradient magnitude. **Ubiquitous for transformer training** — every HuggingFace training recipe uses AdamW ([Loshchilov & Hutter, 2017](https://arxiv.org/abs/1711.05101)).

### 6.6 — LAMB

$$\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \cdot \frac{\|\theta_t^{(l)}\|}{\|u_t^{(l)}\|} \cdot u_t^{(l)}$$

where uₜ is the Adam update. **Intuition:** Normalises updates layer-wise so every layer receives updates of similar relative magnitude — essential for large-batch training (32k+). LAMB enabled BERT pre-training in 76 minutes with batch size 65,536 ([You et al., 2019](https://arxiv.org/abs/1904.00962)).

### 6.7 — Lion

Uses the *sign* of the gradient moment, not the magnitude:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(\beta_2 m_{t-1} + (1-\beta_2) g_t) - \lambda \theta_t$$

**Intuition:** Every parameter gets the same magnitude update (±η). More memory-efficient than Adam (no v second-moment buffer). Competitive with AdamW on vision; less established for LLMs ([Chen et al., 2023](https://arxiv.org/abs/2302.06675)).

### 6.8 — Learning Rate Schedulers

| Scheduler | Behaviour | Best for |
|---|---|---|
| Constant | No decay | Exploration, early experiments |
| Step decay | LR × factor every N epochs | CNNs, simple training |
| Cosine annealing | Follows cosine from η_max to η_min | Transformers |
| Warmup + cosine | Linear warmup → cosine decay | BERT, GPT pre-training |
| One-cycle | Rise then fall in one cycle | fast.ai discriminative LR |
| ReduceLROnPlateau | Halve on plateau | When schedule is unknown |

**Warmup intuition:** At step 0 the model weights are random — a large LR causes chaotic updates that can permanently damage the initialisation. Linear warmup (4–10% of total steps) lets the model find a stable region first. **This is not optional for transformer training** — skipping warmup reliably causes divergence in the first few hundred steps.

```python
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

model = MLP(784, 256, 10)  # from earlier

# AdamW: the standard for transformer fine-tuning
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Warmup + cosine: the standard schedule
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,           # warmup over first 100 steps
    num_training_steps=1000         # total training steps
)

# One step of the training loop
optimizer.zero_grad()
loss = torch.tensor(2.3, requires_grad=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
optimizer.step()
scheduler.step()
```

> 🎯 **Interview prep**: "Why does AdamW fix weight decay vs Adam?" In Adam, L2 regularisation adds λ·θ to the gradient *before* the adaptive scaling divides by √v̂ — the effective decay is λ/√v̂, not λ. Parameters with large recent gradients get less decay. AdamW applies decay directly to parameters after the adaptive update, so decay is always exactly λ.

**Optimizer comparison:**

| Optimizer | Adaptive LR | Memory overhead | Best use case |
|---|---|---|---|
| SGD + Momentum | No | +1 buffer (v) | CNN ImageNet training |
| RMSProp | Yes | +1 buffer | RNNs |
| Adam | Yes | +2 buffers (m, v) | Default most tasks |
| AdamW | Yes | +2 buffers | Transformer pre-training/fine-tuning |
| LAMB | Yes | +2 buffers | Large-batch distributed training |
| Lion | Partial (sign) | +1 buffer | Memory-efficient alternative |

> 🏭 **Production note**: Karpathy's recipe — use **Adam with LR=3e-4** for initial experiments. Once architecture and data are validated, switch to AdamW + cosine schedule for final training runs. Don't tune the optimizer until the model can overfit a small batch.

**Resources**
- [Adam — Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- [AdamW — Loshchilov & Hutter, 2017](https://arxiv.org/abs/1711.05101)
- [LAMB — You et al., 2019](https://arxiv.org/abs/1904.00962)
- [Lion — Chen et al., 2023](https://arxiv.org/abs/2302.06675)

---

## 7. Batching, Epochs, and the Training Loop

The training loop seems mechanical — loop, forward, backward, step — but the choices of batch size, epoch count, and data ordering have substantial effects on generalisation and training speed. These details appear in every ML interview because they reveal whether a candidate has actually trained models at scale.

### 7.1 — What Is an Epoch?

An **epoch** is one complete pass through the entire training dataset:

$$\text{Steps per epoch} = \left\lceil \frac{N}{B} \right\rceil$$

Example: N=50,000 samples, B=256 batch size → 196 steps per epoch.

**Too few epochs:** underfitting — the model hasn't seen the full distribution. **Too many epochs:** overfitting — validation loss diverges upward while training loss keeps falling. **Early stopping:** monitor validation loss; stop when it hasn't improved for `patience` epochs (typically 5–20). Save the checkpoint from the best validation step, not the last.

**Epoch vs step vocabulary:**

| Term | Meaning |
|---|---|
| Epoch | One full pass over training data |
| Step / iteration | One forward + backward + update on one mini-batch |
| Global step | Total updates since training started |
| Warmup steps | Steps at the start with linearly increasing LR |

### 7.2 — Batch Size Effects on Generalisation

Batch size is not just a memory constraint — it directly affects what the model learns.

| Batch size | Gradient quality | Generalisation | Intuition |
|---|---|---|---|
| Small (8–64) | Noisy, high variance | Often **better** | Noise as regulariser |
| Large (512–4096) | Low noise, near-true gradient | Often **worse** | Converges to sharp minima |
| Full batch | Exact gradient | Tends to overfit | No stochastic regularisation |

**Why large batches generalise worse** ([Keskar et al., 2017](https://arxiv.org/abs/1609.04836)): Large-batch training converges to **sharp minima** — narrow loss valleys where small perturbations cause large accuracy drops. Small-batch training's gradient noise "escapes" sharp minima and finds **flat minima** — wide valleys that generalise better across the test distribution.

**Linear scaling rule:** when multiplying batch size by k, multiply LR by k (ResNet, BERT training):

$$\eta_\text{new} = \eta_\text{base} \times \frac{B_\text{new}}{B_\text{base}}$$

This works because the expected gradient magnitude is the same, but with k× less gradient noise — so you can take k× larger steps.

**Gradient accumulation** simulates a larger batch without more memory:

```python
accumulation_steps = 4              # simulate 4× the actual batch size
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    logits = model(x)
    loss = criterion(logits, y) / accumulation_steps   # normalise by steps
    loss.backward()                 # gradients accumulate in .grad buffers

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()            # update after accumulating K batches
        optimizer.zero_grad()
```

### 7.3 — The Complete Training Loop

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Toy dataset
X = torch.randn(10_000, 784)
y = torch.randint(0, 10, (10_000,))
dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

model     = MLP(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for epoch in range(10):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()                    # always clear before backward
        logits = model(batch_x)
        loss   = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip exploding grads
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1:2d} | loss: {avg_loss:.4f}")
```

**drop_last=True:** if N mod B ≠ 0, the final batch is smaller. With Batch Normalisation, a batch of size 1 makes BN statistics meaningless — always use `drop_last=True` when BN is in the model.

**shuffle=True between epochs:** without shuffling, the model sees identical mini-batches every epoch — correlated gradients slow convergence and can create periodic loss oscillations.

> 🏭 **Production note**: The single most important debugging step for a new training pipeline is to **overfit one batch**. Comment out shuffling, take one batch of 2–32 samples, and train until loss → 0. If it doesn't reach near-zero, you have a model or loss bug — not a hyperparameter problem.

---

## 8. Regularisation

Regularisation reduces the gap between training and validation performance. The most effective techniques don't penalise specific parameters — they reshape the training dynamics or inject structured noise that prevents the model from relying on any single pattern.

### 8.1 — Dropout

During training, randomly zero each neuron's output with probability p:

$$\tilde{h} = h \odot \text{Bernoulli}(1-p) \cdot \frac{1}{1-p}$$

The `1/(1−p)` scale factor (inverted dropout) means no rescaling is needed at test time — expected output is the same at train and test.

**Intuition:** Prevents neurons from *co-adapting* — learning to depend on specific other neurons that might be absent. Acts as approximate ensemble averaging over 2ⁿ sub-networks. p=0.1–0.3 for large models; p=0.5 was standard for FC layers in AlexNet/VGGNet era ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)).

> 🏭 **Production note**: Do not use standard dropout with Batch Normalisation — they interfere. BN tracks running statistics across the batch, while dropout randomly alters activations, making BN statistics noisy and causing train/test discrepancy. Use one or the other, not both.

### 8.2 — Batch Normalisation

For a mini-batch B = {x₁, …, xₘ}, normalise and re-scale:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

where μ_B, σ²_B are the batch mean and variance, and γ, β are learned scale and shift parameters.

**Why γ and β?** Pure normalisation would prevent the network from learning identity mappings or using the full range of the activation function. γ and β give the network the freedom to undo the normalisation if optimal — the safety comes from the normalisation during early training, not from constraining the final representation ([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)).

At inference time, BN uses running statistics collected during training (exponential moving average of batch μ and σ²), not the current batch.

**Why BN enables higher learning rates:** by resetting activation variance to ≈1 at every layer, BN prevents the runaway activation growth that causes gradient explosion. The network is far more tolerant of aggressive learning rates.

### 8.3 — Layer Normalisation

Normalises across the feature dimension (not the batch dimension):

$$\text{LN}(x) = \frac{x - \mu_\text{feat}}{\sqrt{\sigma_\text{feat}^2 + \epsilon}} \cdot \gamma + \beta$$

**Key difference:** Layer Norm is independent of batch size — it normalises each sample by its own statistics, not by statistics across other samples. This makes it ideal for transformers where batch size can be 1 during inference, and for variable-length sequences where different positions shouldn't share normalisation statistics.

**RMSNorm** (used in LLaMA, Mistral, Qwen, DeepSeek) removes the mean-subtraction step:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

**Intuition:** Models with LayerNorm spontaneously learn representations orthogonal to the uniform vector, making mean subtraction redundant. RMSNorm drops this step with negligible quality loss but real speed gains that compound over billions of training tokens.

**Normalisation comparison:**

| Method | Normalises over | Batch size independent | Use case |
|---|---|---|---|
| Batch Norm | Batch + spatial | No | CNNs, large batches |
| Layer Norm | Feature dim | Yes | Transformers (pre-2023) |
| RMSNorm | Feature dim (no mean) | Yes | LLaMA, Mistral, all 2023+ LLMs |
| Group Norm | Feature groups | Yes | Detection, small batches |
| Instance Norm | Spatial, per-sample | Yes | Style transfer |

### 8.4 — Weight Decay (L2 Regularisation)

$$L_\text{reg} = L + \frac{\lambda}{2}\|\theta\|^2, \quad \text{gradient: } \nabla L_\text{reg} = \nabla L + \lambda\theta$$

**Intuition:** Penalises large weights, keeping the model's effective capacity limited and encouraging smaller, more distributed representations. With AdamW (Section 6.5), weight decay is applied directly to parameters — the correct way to regularise adaptive optimisers.

```python
# Batch norm vs layer norm usage
bn = nn.BatchNorm1d(256)    # for CNNs, normalise across batch
ln = nn.LayerNorm(256)      # for transformers, normalise across features

# RMSNorm (manual — standard in LLaMA-style models)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
```

**Resources**
- [Dropout — Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)
- [Batch Normalisation — Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)
- [RMSNorm vs LayerNorm comparison — mljourney.com](https://mljourney.com/batch-normalization-vs-layer-normalization-vs-rmsnorm-which-to-use-and-when/)

---

## 9. CNN Architectures

Convolutional neural networks apply shared, spatially localised filters to exploit the fact that natural images have translation-invariant structure — an edge detector useful in the top-left corner of an image is equally useful everywhere else. A convolution with kernel size k×k and C_out output channels has only k²·C_in·C_out parameters, compared to C_out·H·W·C_in for a fully connected layer — a massive parameter reduction for typical image sizes.

### 9.1 — The Architecture Arms Race

**AlexNet (2012)**: First to use ReLU activations and dropout at scale. Won ImageNet by a 10-point margin, launching the deep learning era. Key: GPU training was essential.

**VGGNet (2014)**: Showed that stacking 3×3 convolutions (two 3×3 convolutions have the same receptive field as one 5×5 but fewer parameters) is more effective than using large kernels. Simple, deep, and reproducible — still used as a backbone.

**Inception/GoogLeNet (2014)**: Introduced the **inception module** — apply multiple filter sizes (1×1, 3×3, 5×5) in parallel and concatenate. 1×1 convolutions act as bottlenecks to reduce channel dimension before expensive larger convolutions.

**ResNet (2015)**: The pivotal innovation — **skip connections** that add the block's input directly to its output ([He et al., 2015](https://arxiv.org/abs/1512.03385)):

$$\text{output} = F(\mathbf{x}) + \mathbf{x}$$

During backpropagation, the gradient through a residual block is:

$$\frac{\partial \text{output}}{\partial \mathbf{x}} = \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}} + 1$$

The `+1` term creates a **direct gradient highway** — even if F's gradient vanishes, the gradient still flows. This allowed training of 152-layer networks (ResNet-152 won ILSVRC 2015 with 3.57% top-5 error) when 20-layer plain networks couldn't train at all.

**EfficientNet (2019)**: Neural architecture search to find the optimal compound scaling — simultaneously scale depth, width, and image resolution with a fixed resource budget. EfficientNet-B7 matched the best models with 8.4× fewer parameters ([Tan & Le, 2019](https://arxiv.org/abs/1905.11946)).

**ConvNeXt (2022)**: Modernised ResNet with lessons from Transformers — larger kernels (7×7), fewer normalisation/activation layers, inverted bottleneck, replaced BatchNorm with LayerNorm. Matches ViT on many benchmarks while keeping the CNN inductive bias ([Liu et al., 2022](https://arxiv.org/abs/2201.03545)).

### 9.2 — MobileNet: Depthwise Separable Convolutions

Standard convolution: C_out × C_in × k × k parameters. MobileNet factorises this into:
- **Depthwise conv**: one filter per input channel = k²·C_in
- **Pointwise conv**: 1×1 conv mixing channels = C_in·C_out

Total: k²·C_in + C_in·C_out. For k=3, this is ~8-9× fewer operations. The accuracy drop is modest; the speed gain is massive for mobile/edge deployment.

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                              # save input for skip connection
        out = self.relu(self.bn1(self.conv1(x)))  # first conv+bn+relu
        out = self.bn2(self.conv2(out))           # second conv+bn (no relu yet)
        out = self.relu(out + residual)           # add skip, then relu
        return out

block = ResidualBlock(64)
x = torch.randn(4, 64, 32, 32)         # (batch, channels, H, W)
print(block(x).shape)                   # torch.Size([4, 64, 32, 32])
```

> 🎯 **Interview prep**: "Why did ResNet work where plain deep networks failed?" The skip connection ensures the gradient of the loss with respect to any layer's input includes a direct +1 term — gradient can't vanish completely even in very deep networks. The network also learns *residuals* (corrections to identity) rather than complete mappings, which is an easier optimisation problem.

**Resources**
- [ResNet — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [EfficientNet — Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- [ConvNeXt — Liu et al., 2022](https://arxiv.org/abs/2201.03545)

---

## 10. Sequence Models: RNN, LSTM, GRU

Recurrent networks process sequences by maintaining a hidden state that is updated at each step. They are largely superseded by Transformers for offline tasks but remain relevant in streaming, low-latency, or resource-constrained settings.

### 10.1 — Vanilla RNN and Its Problem

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

Backpropagating through T time steps means multiplying by W_h T times. If the spectral radius of W_h is >1, gradients explode; if <1, they vanish. A sequence of length 100 with spectral radius 0.9 gives a gradient of 0.9¹⁰⁰ ≈ 2.66×10⁻⁵. Practically, vanilla RNNs can't learn dependencies beyond ~10–20 steps.

### 10.2 — LSTM: Additive Cell State

LSTM replaces pure multiplicative recurrence with an **additive cell state** cₜ ([Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)):

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate cell)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}$$
$$h_t = o_t \odot \tanh(c_t)$$

The key is **cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ** — the `+` means gradient flows back through the cell state *additively* (like a highway), not through repeated matrix multiplication. If the forget gate stays near 1 (fₜ≈1), the gradient propagates essentially unchanged through many steps.

### 10.3 — GRU: Simpler Two-Gate Alternative

GRU merges the cell state and hidden state, using only reset and update gates:

$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(reset gate)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(W_h [r_t \odot h_{t-1}, x_t])$$

**Comparison:**

| | RNN | LSTM | GRU |
|---|---|---|---|
| Parameters | Fewest | Most (3× RNN) | Middle (2× RNN) |
| Long-range dependencies | Poor | Excellent | Good |
| Training speed | Fastest | Slowest | Middle |
| Use today | Avoid | Legacy NLP | Lightweight streaming |
| Replaced by | Transformer | Transformer | Transformer |

**Resources**
- [LSTM — Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Mamba SSM — Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) — the modern RNN-alternative for long sequences

---

## 11. Attention and Transformers (Foundations)

Attention solves the fundamental problem of recurrent networks: the hidden state is a fixed-size bottleneck that must compress the entire sequence history. Attention instead allows the model to *directly attend* to any position in the sequence with arbitrary distance.

**Scaled dot-product attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Q (query), K (key), V (value) are linear projections of the input. The dot product QKᵀ computes pairwise compatibility scores; dividing by √d_k prevents softmax saturation when d_k is large. The output is a weighted sum of values, where weights are the attention probabilities.

**Multi-head attention** runs h attention heads in parallel, each with different projections:

$$\text{MH}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Each head can attend to different aspects of the sequence (syntax, semantics, coreference). The original Transformer used h=8 heads with d_k=64 ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

**O(n²) complexity** is the key limitation: the attention matrix is n×n for sequence length n. FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) reorders the computation to avoid materialising the full n×n matrix in HBM, achieving 2–4× speedup and reducing memory from O(n²) to O(n).

*Transformers, BERT, GPT, ViT, positional encodings (RoPE, ALiBi), and fine-tuning methods (LoRA) are covered in depth in* `13-llm`.

---

## 12. Transfer Learning and Fine-Tuning

Training large models from scratch requires enormous compute and data. Transfer learning starts from a model pre-trained on a large dataset and adapts it to a new, smaller task — amortising the expensive pre-training across many downstream uses.

### 12.1 — What Transfers and Why

([Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)) showed empirically that early layers of CNNs learn universal features (Gabor filters, colour blobs) that transfer across domains, while later layers are increasingly task-specific. The insight generalises to transformers: early attention layers capture syntax; later layers capture semantics; the final layers are task-specific.

### 12.2 — Fine-Tuning Strategies

| Strategy | What you do | When to use |
|---|---|---|
| **Feature extraction** | Freeze backbone, train only head | Very little data (<1k samples) |
| **Partial fine-tuning** | Unfreeze top k layers | Moderate data, domain mismatch |
| **Full fine-tuning** | Update all parameters | Sufficient data, same domain |
| **Linear probing** | Freeze backbone, train linear head | Understanding what backbone encodes |
| **LoRA/PEFT** | Add low-rank adapters to frozen model | LLMs, large models (covered in 13-llm) |

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")

# Stage 1: freeze all, train only a classification head
for param in model.parameters():
    param.requires_grad_(False)

classifier = nn.Linear(768, 5)     # task-specific head

# Stage 2: unfreeze last 4 transformer layers with lower LR
for layer in model.encoder.layer[-4:]:
    for param in layer.parameters():
        param.requires_grad_(True)

optimizer = optim.AdamW([
    {"params": model.encoder.layer[-4:].parameters(), "lr": 1e-5},  # pretrained: low LR
    {"params": classifier.parameters(),               "lr": 1e-3},  # random init: high LR
])
```

**Catastrophic forgetting:** full fine-tuning on a small dataset can make the model forget its pre-trained knowledge. Mitigations: lower LR for pretrained layers, early stopping, Elastic Weight Consolidation (EWC), or PEFT methods that leave backbone weights unchanged.

> 🏭 **Production note**: Always start with linear probing to understand your task's alignment with the pretrained representations. If probing gives >80% accuracy, full fine-tuning is likely unnecessary — a 2-layer head with frozen backbone will generalise better on small datasets.

---

## 13. Training Best Practices

Experienced practitioners develop a mental checklist that prevents the most common, costly mistakes. Karpathy's training recipe — data inspection → pipeline validation → overfit a small set → regularise → tune — maps exactly to what goes wrong in production.

### 13.1 — The Pre-Training Checklist

```
□ Fix random seed (reproducibility)
□ Verify loss at step 0 matches theory (-log(1/C) for C classes = 2.3 for 10 classes)
□ Overfit single batch to near-zero loss (model/loss correctness check)
□ Training loss should decrease monotonically for first few epochs
□ Validation loss should be close to training loss early (large gap = overfitting)
□ Visualise inputs immediately before the model (catch data bugs)
□ Monitor gradient norms (should be roughly constant, not exploding/vanishing)
```

### 13.2 — Mixed Precision Training

FP16 uses 2 bytes per parameter (vs 4 for FP32), halving memory and doubling throughput on modern GPUs. BF16 has the same memory as FP16 but with a larger dynamic range (same exponent bits as FP32), making it more numerically stable for training.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()           # scales loss to avoid FP16 underflow

for x, y in loader:
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):    # forward in BF16
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()           # scale before backward
    scaler.unscale_(optimizer)              # unscale for gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                  # update (unscales internally)
    scaler.update()                         # adjust scale factor
```

### 13.3 — Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss = NaN at step 1 | LR too high / log(0) / bad init | Lower LR; check label distribution |
| Loss = NaN after many steps | Exploding gradients | Add `clip_grad_norm_` with max_norm=1 |
| Training loss decreases, val stays flat | Overfitting | Add dropout, weight decay, or more data |
| Both losses flat from start | LR too low / dead neurons | Increase LR 10×; check activation saturations |
| Loss oscillates wildly | LR too high | Halve LR; add warmup |
| Dead neurons (ReLU outputs always 0) | Bad init or high LR | Use He init; lower LR; try Leaky ReLU |
| Slow convergence near solution | MAE loss + small errors | Switch to Huber or MSE |

> 🎯 **Interview prep**: "How do you debug a training run where loss doesn't decrease?" Karpathy's approach: (1) overfit a single batch — if it fails, you have a model/loss bug; (2) verify loss at step 0; (3) train with no regularisation and plenty of capacity to confirm the model can learn; (4) only then add regularisation.

**Resources**
- [A Recipe for Training Neural Networks — Karpathy, 2019](https://karpathy.github.io/2019/04/25/recipe/) — the definitive practical guide
- [Practical Deep Learning for Coders — fast.ai](https://course.fast.ai/) — top-down, code-first approach
- [Delving Deep into Rectifiers — He et al., 2015](https://arxiv.org/abs/1502.01852) — He initialisation derivation

---

## 14. Generative Architectures

Generative models learn the data distribution p(x) to enable sampling new examples. Three families have dominated: GANs, VAEs, and diffusion models. As of 2024, diffusion models have essentially replaced GANs for image generation.

### 14.1 — Variational Autoencoders (VAE)

A VAE consists of an encoder q(z|x) that maps data to a latent distribution and a decoder p(x|z) that reconstructs from latent samples. The ELBO loss (Section 5.5) balances reconstruction quality against latent space structure.

The **reparameterisation trick** enables backprop through sampling: instead of sampling z ~ q(z|x) = N(μ, σ²) (not differentiable), sample ε ~ N(0,1) and compute z = μ + σ·ε. The gradient flows through μ and σ, not the sampling operation.

### 14.2 — Generative Adversarial Networks (GAN)

The minimax game between generator and discriminator (Section 5.5) produces sharp, photorealistic images when it converges. **DCGAN** (2015) stabilised training with batch normalisation and strided transposed convolutions. **StyleGAN** (2019-2020) enabled style control via adaptive instance normalisation (AdaIN). **CycleGAN** enables unpaired image-to-image translation using cycle consistency loss.

**Why GANs fell out of favour:** training instability (mode collapse, discriminator/generator imbalance), sensitivity to hyperparameters, and the WGAN fix only partially addresses these issues. Diffusion models achieve better diversity with simpler training.

### 14.3 — Diffusion Models

Diffusion models define a forward process that gradually adds Gaussian noise to data, then train a network to reverse it ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)).

**Forward process:** q(xₜ|xₜ₋₁) = N(xₜ; √(1−βₜ)xₜ₋₁, βₜI) — at each step t, add a small amount of noise β. After T steps (typically 1000), x_T ≈ N(0,I) is pure noise.

**Training objective:** Instead of learning the full reverse distribution, DDPM trains a noise predictor εθ(xₜ, t) to predict the noise ε that was added:

$$L = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\, t)\|^2\right]$$

where ᾱₜ = ∏ᵢ₌₁ᵗ(1−βᵢ) is the cumulative noise schedule.

**Sampling:** Run the reverse process for T steps: at each step, predict the noise, subtract it, and add a small amount of Gaussian noise.

**DDIM** ([Song et al., 2020](https://arxiv.org/abs/2010.02502)) reformulates as a deterministic ODE, reducing steps from 1000 to 20–50 without retraining. **Latent Diffusion Models** (Stable Diffusion, [Rombach et al., 2022](https://arxiv.org/abs/2112.10752)) run diffusion in the compressed latent space of a VAE rather than pixel space — 8× spatial compression makes generation practical.

**Classifier-free guidance (CFG)** enables conditional generation without a separate classifier:

$$\tilde\epsilon_\theta(x_t, c) = (1+s)\epsilon_\theta(x_t, c) - s\epsilon_\theta(x_t, \emptyset)$$

where c is the conditioning signal (text prompt), ∅ is the unconditional embedding, and s is the guidance scale. Higher s → stronger condition adherence, less diversity.

**Generative model comparison:**

| | GAN | VAE | Diffusion |
|---|---|---|---|
| Sample quality | High (sharp) | Low-medium (blurry) | Highest |
| Training stability | Poor | Good | Good |
| Mode coverage | Mode collapse risk | Good | Excellent |
| Sample speed | Fast (one forward pass) | Fast | Slow (T steps) |
| Controllability | Moderate | Good (β-VAE) | Excellent (CFG) |
| SotA systems | StyleGAN3 | — | Stable Diffusion 3, DALL-E 3, Flux |

> 🎯 **Interview prep**: "Why did diffusion models replace GANs?" Three reasons: (1) no adversarial training → no mode collapse or training instability; (2) likelihood-based training gives better diversity; (3) the iterative denoising process allows quality–speed tradeoff at inference (DDIM, consistency distillation). The only remaining GANadvantage is inference speed — a single forward pass vs 20–50 denoising steps.

> 🏭 **Production note**: Stable Diffusion 1.5 runs at 512×512 in ~1s on a consumer GPU. For production image generation APIs, Flux and SD3 (flow-matching, not DDPM) are the current standard — they use continuous-time formulations that generate in 4–8 steps.

**Resources**
- [DDPM — Ho et al., 2020](https://arxiv.org/abs/2006.11239) — the denoising diffusion probabilistic model paper
- [Latent Diffusion Models — Rombach et al., 2022](https://arxiv.org/abs/2112.10752) — Stable Diffusion architecture
- [Wasserstein GAN — Arjovsky et al., 2017](https://arxiv.org/abs/1701.04862)
- [VAE — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)

---

## 15. References

### Foundational Papers
- [Rumelhart et al., 1986 — Backpropagation](https://www.nature.com/articles/323533a0)
- [Hochreiter & Schmidhuber, 1997 — LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Srivastava et al., 2014 — Dropout (JMLR)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Ioffe & Szegedy, 2015 — Batch Normalisation](https://arxiv.org/abs/1502.03167)
- [He et al., 2015 — Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [He et al., 2015 — Delving Deep into Rectifiers (He Init)](https://arxiv.org/abs/1502.01852)
- [Szegedy et al., 2015 — Inception / Label Smoothing](https://arxiv.org/abs/1512.00567)

### Activation Functions
- [Hendrycks & Gimpel, 2016 — GELU](https://arxiv.org/abs/1606.08415)
- [Ramachandran et al., 2017 — Swish / SiLU](https://arxiv.org/abs/1710.05941)

### Optimizers
- [Kingma & Ba, 2014 — Adam](https://arxiv.org/abs/1412.6980)
- [Loshchilov & Hutter, 2017 — AdamW](https://arxiv.org/abs/1711.05101)
- [You et al., 2019 — LAMB](https://arxiv.org/abs/1904.00962)
- [Chen et al., 2023 — Lion](https://arxiv.org/abs/2302.06675)

### Loss Functions
- [Lin et al., 2017 — Focal Loss / RetinaNet](https://arxiv.org/abs/1708.02002)
- [Schroff et al., 2015 — FaceNet / Triplet Loss](https://arxiv.org/abs/1503.03832)
- [Chen et al., 2020 — SimCLR / NT-Xent](https://arxiv.org/abs/2002.05709)
- [Kingma & Welling, 2013 — VAE / ELBO](https://arxiv.org/abs/1312.6114)
- [Arjovsky et al., 2017 — WGAN](https://arxiv.org/abs/1701.04862)
- [Rafailov et al., 2023 — DPO](https://arxiv.org/abs/2305.18290)

### Architectures
- [Tan & Le, 2019 — EfficientNet](https://arxiv.org/abs/1905.11946)
- [Liu et al., 2022 — ConvNeXt](https://arxiv.org/abs/2201.03545)
- [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Dao et al., 2022 — FlashAttention](https://arxiv.org/abs/2205.14135)
- [Gu & Dao, 2023 — Mamba](https://arxiv.org/abs/2312.00752)

### Generative Models
- [Ho et al., 2020 — DDPM](https://arxiv.org/abs/2006.11239)
- [Song et al., 2020 — DDIM](https://arxiv.org/abs/2010.02502)
- [Rombach et al., 2022 — Latent Diffusion / Stable Diffusion](https://arxiv.org/abs/2112.10752)

### Training and Practical Guidance
- [Keskar et al., 2017 — On Large-Batch Training](https://arxiv.org/abs/1609.04836)
- [Yosinski et al., 2014 — How Transferable are Features?](https://arxiv.org/abs/1411.1792)
- [Kaplan et al., 2020 — Scaling Laws](https://arxiv.org/abs/2001.08361)
- [Karpathy, 2019 — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/)
- [Deep Learning textbook — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
- [Dive into Deep Learning — d2l.ai](https://d2l.ai/)
