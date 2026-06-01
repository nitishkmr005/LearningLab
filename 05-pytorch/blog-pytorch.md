# PyTorch for ML Engineers and AI Engineers

*A production-first deep-dive covering tensors, autograd, training loops, distributed training, mixed precision, debugging, and deployment — the PyTorch knowledge that separates engineers who can write training code from engineers who can own it.*

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [A Brief History](#2-a-brief-history)
3. [Tensors, Devices, and Shapes](#3-tensors-devices-and-shapes)
4. [Autograd: The Computation Graph](#4-autograd-the-computation-graph)
5. [nn.Module: Building Models](#5-nnmodule-building-models)
6. [Datasets and DataLoaders](#6-datasets-and-dataloaders)
7. [The Training Loop](#7-the-training-loop)
8. [Loss Functions and Optimizers](#8-loss-functions-and-optimizers)
9. [Saving, Loading, and Checkpointing](#9-saving-loading-and-checkpointing)
10. [Fine-Tuning Pretrained Models](#10-fine-tuning-pretrained-models)
11. [Mixed Precision and Throughput Optimization](#11-mixed-precision-and-throughput-optimization)
12. [Gradient Accumulation and Clipping](#12-gradient-accumulation-and-clipping)
13. [Debugging Training](#13-debugging-training)
14. [Profiling and Eliminating Bottlenecks](#14-profiling-and-eliminating-bottlenecks)
15. [Distributed Training: DDP and FSDP](#15-distributed-training-ddp-and-fsdp)
16. [Export and Deployment](#16-export-and-deployment)
17. [The Modern Recipe](#17-the-modern-recipe)
18. [References](#18-references)

---

## 1. The Problem

PyTorch is the lingua franca of ML research and AI engineering, but knowing how to call `model(x)` is not the same as being able to own a training pipeline. The engineers who cause outages are the ones who didn't understand that `model.eval()` disables dropout and batch norm during inference — so their deployed model behaved differently than the one they evaluated. The engineers who waste GPU days are the ones who didn't know that their DataLoader was the bottleneck, not the model. The engineers whose training runs diverge mysteriously are the ones who didn't know that in-place operations break the computation graph.

The gap between "I can train a model in a notebook" and "I can build a reliable, efficient training system" is large, and it's built from a dozen small technical details: how autograd tracks operations, how gradient accumulation works with loss scaling, why `state_dict` is the right checkpoint format, why `pin_memory=True` and `num_workers > 0` matter, and how DDP handles gradient synchronization across processes.

This blog covers all of it — from the mechanics of automatic differentiation to distributed training across multiple GPUs.

---

## 2. A Brief History

Before PyTorch, training neural networks meant either fighting TensorFlow 1.x's static computation graph — where you defined the graph first and executed it separately, making debugging nearly impossible — or using Theano, which had similar limitations. The "define-by-run" (dynamic graph) paradigm, pioneered by Chainer (2015), changed everything: the computation graph is built at runtime, so you can use native Python control flow, debug with pdb, and inspect intermediate values like normal variables.

Facebook AI Research released PyTorch 0.1 in 2017, and its adoption was explosive in the research community because it made experimentation dramatically faster. TensorFlow responded with eager execution (TF 2.0, 2019), but by then PyTorch had won the research mindset share. The key events in PyTorch's evolution:

- **2017**: PyTorch 0.1 — dynamic graphs, automatic differentiation
- **2018**: TorchScript and JIT compilation for production deployment
- **2020**: `DistributedDataParallel` stabilized — standard for multi-GPU training
- **2022**: `FullyShardedDataParallel` (FSDP) for training models too large for single-GPU DDP
- **2023**: PyTorch 2.0 — `torch.compile()` with TorchInductor backend, significant training speedup without code changes
- **2024**: PyTorch 2.x — `torch.export()` for ahead-of-time compilation, FSDP2 improvements

---

## 3. Tensors, Devices, and Shapes

Every PyTorch computation operates on tensors — n-dimensional arrays with a dtype, shape, and device. Getting these details wrong causes the most beginner bugs.

```python
import torch
import numpy as np

# Creation
t_zeros = torch.zeros(3, 4)            # shape (3,4), float32
t_rand = torch.rand(2, 3, dtype=torch.float32)
t_from_list = torch.tensor([1.0, 2.0, 3.0])
t_from_numpy = torch.from_numpy(np.array([1, 2, 3], dtype=np.float32))

# Shapes: the most important debugging tool
x = torch.rand(32, 128, 768)  # batch=32, seq_len=128, d_model=768
print(x.shape)         # → torch.Size([32, 128, 768])
print(x.dtype)         # → torch.float32
print(x.device)        # → cpu  (or cuda:0 if GPU available)

# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
print(x.device)        # → cuda:0  (if GPU available)

# Broadcasting: two tensors are compatible if dimensions match or one is 1
a = torch.rand(32, 1, 768)   # (32, 1, 768)
b = torch.rand(1, 128, 768)  # (1, 128, 768)
c = a + b                    # (32, 128, 768) — broadcast both dims
print(c.shape)         # → torch.Size([32, 128, 768])

# Reshaping: view vs reshape
# view: contiguous memory only, no copy
# reshape: works always, may copy
flat = x.view(32, -1)          # (32, 128*768)
print(flat.shape)      # → torch.Size([32, 98304])
transposed = x.transpose(1, 2)  # (32, 768, 128)
print(transposed.shape)# → torch.Size([32, 768, 128])
permuted = x.permute(2, 0, 1)  # (768, 32, 128)
print(permuted.shape)  # → torch.Size([768, 32, 128])

# Squeeze/unsqueeze: remove or add dimensions of size 1
squeezed = torch.rand(1, 128, 1).squeeze()
print(squeezed.shape)  # → torch.Size([128])  ← both size-1 dims removed
unsqueezed = torch.rand(128).unsqueeze(0)
print(unsqueezed.shape)# → torch.Size([1, 128])
```

> 🎯 **Interview prep**: "What is the difference between `.view()` and `.reshape()`?" — `view` requires contiguous memory and never copies; if memory isn't contiguous (e.g., after transpose), it raises an error. `reshape` always works and makes a copy if needed. In practice, call `.contiguous().view()` after operations that break contiguity, or just use `reshape`.

---

## 4. Autograd: The Computation Graph

Autograd is PyTorch's automatic differentiation engine. It builds a computation graph dynamically — every operation on a tensor with `requires_grad=True` is recorded, and `backward()` traverses this graph in reverse to compute gradients.

### 4.1 How the Graph is Built

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([1.5, -0.5], requires_grad=True)

# Each operation creates a node in the graph
y = (x * w).sum()  # y = 2*1.5 + 3*(-0.5) = 3.0 - 1.5 = 1.5
z = y ** 2         # z = 2.25

print(f"y={y.item():.2f}, z={z.item():.2f}")
print(f"z.grad_fn: {z.grad_fn}")  # PowBackward0

z.backward()  # computes gradients via chain rule

# dz/dx = dz/dy * dy/dx = 2y * w = 2*1.5 * [1.5, -0.5] = [4.5, -1.5]
print(f"x.grad = {x.grad}")   # tensor([ 4.5000, -1.5000])
print(f"w.grad = {w.grad}")   # tensor([ 6., -3.])
```

### 4.2 Common Autograd Bugs

```python
import torch

# Bug 1: in-place operations on leaf tensors with grad
x = torch.tensor([1.0], requires_grad=True)
try:
    x += 1.0  # RuntimeError: in-place operation on tensor with gradient
except RuntimeError as e:
    print(f"In-place error: {e}")

# Fix: use new tensor
x = x + 1.0  # creates new tensor, safe

# Bug 2: forgetting to zero gradients
x = torch.tensor([2.0], requires_grad=True)
for _ in range(3):
    y = x ** 2
    y.backward()
    print(f"x.grad = {x.grad}")  # 4, 8, 12 — accumulating!
    x.grad.zero_()  # must zero before next backward

# Bug 3: detach when you need values without grad tracking
output = model(x)
loss = criterion(output, target)
logged_loss = loss.detach().item()  # safe to log; no graph reference kept

# Bug 4: torch.no_grad() for inference
with torch.no_grad():
    predictions = model(test_x)  # no graph built, saves memory
```

### 4.3 Freezing Layers

```python
import torch.nn as nn

# Freeze backbone, train only the head
def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if 'head' not in name:  # freeze everything except head layers
            param.requires_grad_(False)
    # Verify
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
```

> 🏭 **Production note**: Always verify parameter counts after freezing. A common mistake is freezing at the wrong level of the module hierarchy, resulting in either all parameters frozen or none frozen.

---

## 5. nn.Module: Building Models

`nn.Module` is PyTorch's building block for all neural network components. Every model, layer, loss function, and utility you build should subclass it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        self.dropout = nn.Dropout(dropout)
        self.batchnorms = nn.ModuleList([
            nn.BatchNorm1d(d) for d in hidden_dims
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.batchnorms)):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.layers[-1](x)  # no activation on output for flexibility

# Critical: use nn.ModuleList, not plain Python list
# nn.ModuleList registers sub-modules so their params appear in model.parameters()

model = MLP(input_dim=100, hidden_dims=[256, 128], output_dim=10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

x = torch.rand(32, 100)
out = model(x)
print(f"Output shape: {out.shape}")  # (32, 10)

# state_dict: the canonical way to inspect and save model weights
sd = model.state_dict()
for key, val in list(sd.items())[:3]:
    print(f"{key}: {val.shape}")
```

> 🎯 **Interview prep**: "Why use `nn.ModuleList` instead of a Python list of layers?" — `nn.ModuleList` registers submodules, so their parameters appear in `model.parameters()`, are moved by `model.to(device)`, and are saved by `model.state_dict()`. A plain Python list of `nn.Module` objects is invisible to PyTorch's parameter tracking.

---

## 6. Datasets and DataLoaders

The DataLoader is frequently the training throughput bottleneck. Getting it right means understanding workers, pin_memory, and prefetching.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert once at construction time, not in __getitem__
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# Production DataLoader settings
def make_dataloader(dataset: Dataset, batch_size: int = 256, train: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,        # parallel data loading (set to num CPU cores or 4)
        pin_memory=True,      # faster CPU→GPU transfer (if CUDA available)
        persistent_workers=True,  # keep workers alive between epochs
        prefetch_factor=2,    # each worker prefetches 2 batches
        drop_last=train,      # avoid variable-size last batch during training
    )

# Example
X = np.random.randn(10_000, 50)
y = np.random.randint(0, 2, 10_000)
dataset = TabularDataset(X, y)
loader = make_dataloader(dataset, batch_size=256, train=True)

# Test one batch
xb, yb = next(iter(loader))
print(f"Batch shapes: X={xb.shape}, y={yb.shape}")  # (256, 50), (256,)
```

> 🏭 **Production note**: Set `num_workers=0` first when debugging DataLoader issues (crashes often become clearer in the main process). Then increase to 4. `pin_memory=True` only helps with CUDA — it pre-allocates page-locked memory on the host so the GPU can DMA from it directly, saving ~20% of transfer time.

---

## 7. The Training Loop

The canonical PyTorch training loop with validation:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()  # CRITICAL: enables dropout, batch norm in training mode
    total_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad()          # clear previous gradients
        logits = model(xb)             # forward pass
        loss = criterion(logits, yb)   # compute loss
        loss.backward()               # backprop
        optimizer.step()              # update parameters

        total_loss += loss.item() * len(xb)  # use .item() to detach scalar

    return total_loss / len(loader.dataset)

def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> dict:
    model.eval()  # CRITICAL: disables dropout, uses running stats for batch norm
    total_loss = 0.0
    correct = 0

    with torch.no_grad():  # no gradient computation needed
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()

    n = len(loader.dataset)
    return {"loss": total_loss / n, "accuracy": correct / n}

def train(model, train_loader, val_loader, n_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), "best_model.pt")

# --- Expected training output ---
# Epoch  1/10 | Train Loss: 2.3012 | Val Loss: 2.2847 | Val Acc: 0.1125 | Time: 12.4s
# Epoch  2/10 | Train Loss: 2.1534 | Val Loss: 2.0913 | Val Acc: 0.2347 | Time: 11.9s
# Epoch  3/10 | Train Loss: 1.9821 | Val Loss: 1.9152 | Val Acc: 0.3241 | Time: 12.1s
# Epoch  5/10 | Train Loss: 1.6203 | Val Loss: 1.6498 | Val Acc: 0.4587 | Time: 12.0s
# Epoch 10/10 | Train Loss: 1.1847 | Val Loss: 1.2103 | Val Acc: 0.5912 | Time: 12.2s
#
# Key signals to watch:
# - train_loss decreasing every epoch → model is learning
# - val_loss tracks train_loss closely → no overfitting yet
# - if val_loss starts increasing while train_loss decreases → overfitting → add dropout/regularization
# - if both losses plateau → learning rate too small or model capacity too small
```

---

## 8. Loss Functions and Optimizers

### 8.1 Choosing the Right Loss

```python
import torch
import torch.nn as nn

# Multiclass classification: CrossEntropyLoss
# Input: (batch, num_classes) logits; target: (batch,) class indices
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 5.0, 3.0]),  # class weights for imbalance
    label_smoothing=0.1                     # smoothing regularization
)

# Binary classification: BCEWithLogitsLoss (NOT BCELoss — numerically stable)
# Input: (batch,) logits; target: (batch,) float 0/1
criterion_binary = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([10.0])  # weight positive class (imbalanced data)
)

# Regression: MSELoss, L1Loss, HuberLoss
criterion_reg = nn.HuberLoss(delta=1.0)  # robust to outliers

# Loss shape gotcha: always check input/target shapes match expectations
logits = torch.rand(32, 10)   # (batch, num_classes)
target = torch.randint(0, 10, (32,))  # (batch,) NOT (batch, num_classes)
loss = criterion(logits, target)
print(f"Loss: {loss.item():.4f}")
```

### 8.2 Optimizers and Learning Rate Scheduling

```python
import torch.optim as optim

# AdamW: the default for transformer fine-tuning
# Weight decay applies to weights but NOT biases/layer norms (unlike plain Adam)
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Discriminative learning rates: lower lr for pretrained layers
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # pretrained: lower lr
    {'params': model.head.parameters(), 'lr': 1e-3},      # random init: higher lr
])

# Warmup + cosine annealing: the standard for transformer training
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

# Each step in training
optimizer.step()
scheduler.step()
```

---

## 9. Saving, Loading, and Checkpointing

Always save `state_dict`, never the full model. Full model pickling ties you to a specific code structure — if you refactor a class, your checkpoint breaks.

```python
import torch
from pathlib import Path

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

# Inference-only loading (no optimizer state needed)
def load_for_inference(model_class, path, device, **model_kwargs):
    model = model_class(**model_kwargs)
    state = torch.load(path, map_location=device)
    # Handle state_dict wrapped in checkpoint
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()  # always set eval mode for inference
    model.to(device)
    return model
```

---

## 10. Fine-Tuning Pretrained Models

Most applied PyTorch work is fine-tuning, not training from scratch. The key decisions: which layers to freeze, learning rate, and how to avoid catastrophic forgetting.

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# Stage 1: freeze backbone, train only head (fast convergence)
model = TextClassifier("bert-base-uncased", num_classes=5)
for param in model.backbone.parameters():
    param.requires_grad_(False)

# Stage 2: unfreeze top layers with lower lr
for param in model.backbone.encoder.layer[-4:].parameters():
    param.requires_grad_(True)

# Verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Stage 2: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")
```

> 🏭 **Production note**: For LoRA-style fine-tuning (PEFT), which is now the standard for LLMs (7B+ parameters), use the `peft` library rather than manual layer freezing. But understanding the manual approach is essential for smaller models and for debugging PEFT configurations.

---

## 11. Mixed Precision and Throughput Optimization

Mixed precision training uses FP16/BF16 for most operations (faster, less memory) while maintaining FP32 for numerically sensitive parts (loss scaling, optimizer state).

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, loader, criterion, optimizer, device):
    model.train()
    scaler = GradScaler()  # scales loss to avoid FP16 underflow

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        # autocast: automatically selects FP16 or BF16 for operations
        with autocast(device_type='cuda', dtype=torch.bfloat16):  # BF16 preferred on Ampere+
            logits = model(xb)
            loss = criterion(logits, yb)

        # Scale loss to prevent underflow, then backward
        scaler.scale(loss).backward()
        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update: scaler checks for inf/nan gradients
        scaler.step(optimizer)
        scaler.update()

    # Memory savings with mixed precision: roughly 2x for activations
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {allocated:.2f} GB")
```

> 🎯 **Interview prep**: "BF16 vs FP16 — which is better for training?" — BF16 has the same dynamic range as FP32 (8 exponent bits) but lower precision (7 mantissa bits). FP16 has higher precision but narrower range — it can underflow to zero for small gradients. BF16 is preferred on Ampere+ GPUs (A100, H100) and avoids loss scaling in most cases. FP16 is needed on Volta/Turing GPUs.

---

## 12. Gradient Accumulation and Clipping

When your desired batch size doesn't fit in GPU memory, accumulate gradients over multiple microbatches before updating.

```python
import torch

def train_with_accumulation(model, loader, criterion, optimizer, device,
                             accumulation_steps: int = 4):
    """Effective batch size = batch_size × accumulation_steps."""
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()

    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)

        with torch.cuda.amp.autocast():
            logits = model(xb)
            # Divide loss by accumulation steps to maintain correct scale
            loss = criterion(logits, yb) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping: prevents exploding gradients (common in RNNs, transformers)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Log gradient norms for monitoring
            grad_norm = sum(p.grad.norm()**2 for p in model.parameters()
                          if p.grad is not None) ** 0.5
            print(f"Step {step+1}: grad_norm={grad_norm:.4f}")
```

> 🏭 **Production note**: Gradient clipping at `max_norm=1.0` is standard for transformer training. If you see gradient norms consistently above 10, something is wrong (too high lr, bad data, numerical instability). If norms collapse to near-zero, your model is stuck — lower the weight decay or check for dead neurons.

---

## 13. Debugging Training

The most reliable debugging technique is "overfit on one batch." If your model can't memorize a single batch with tiny loss after 100 steps, something fundamental is broken — wrong loss, wrong shapes, wrong target format.

```python
import torch
import torch.nn as nn
import numpy as np

def debug_one_batch(model, loader, criterion, optimizer, device, n_steps=100):
    """Debug: can the model overfit a single batch?"""
    model.train()
    model.to(device)

    # Get one batch and hold it fixed
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)

    print(f"Input shape: {xb.shape}, Target shape: {yb.shape}")
    print(f"Target range: min={yb.min().item()}, max={yb.max().item()}")

    # Check for NaN in input
    assert not torch.isnan(xb).any(), "NaN in inputs!"
    assert not torch.isinf(xb).any(), "Inf in inputs!"

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(xb)

        # Check output shape and range
        if step == 0:
            print(f"Output shape: {out.shape}")
            print(f"Output range: min={out.min().item():.4f}, max={out.max().item():.4f}")

        loss = criterion(out, yb)

        # Check for NaN loss early
        if torch.isnan(loss):
            print(f"NaN loss at step {step}! Check loss function and target format.")
            break

        loss.backward()

        # Check gradients
        if step == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN gradient in {name}!")

        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.6f}")

# If loss doesn't approach near-zero after 100 steps on one batch:
# 1. Check target format (class indices vs one-hot vs float)
# 2. Check loss function matches task
# 3. Check learning rate (try 1e-2 for debugging)
# 4. Check model architecture (is output dimension correct?)
```

---

## 14. Profiling and Eliminating Bottlenecks

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# The key question: is my GPU starving (waiting for data)?
def check_gpu_utilization(model, loader, device, n_batches=20):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        for i, (xb, yb) in enumerate(loader):
            if i >= n_batches:
                break
            with record_function("data_transfer"):
                xb, yb = xb.to(device), yb.to(device)
            with record_function("forward"):
                out = model(xb)
                loss = criterion(out, yb)
            with record_function("backward"):
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Fast diagnosis: compare these two measurements
# If num_workers=0 and increasing to 4 massively speeds up training → DataLoader bottleneck
# If GPU utilization < 50% → CPU/data bottleneck
# If GPU OOM → reduce batch size or enable gradient checkpointing
```

---

## 15. Distributed Training: DDP and FSDP

### 15.1 DistributedDataParallel (DDP)

DDP replicates the full model on each GPU and averages gradients after each backward pass. It's the standard for models that fit on a single GPU.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank: int, world_size: int, dataset, model_class, **model_kwargs):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    model = model_class(**model_kwargs).to(device)
    model = DDP(model, device_ids=[rank])

    # DistributedSampler ensures each GPU sees different data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)  # CRITICAL: ensures different shuffling each epoch
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()  # gradients averaged across GPUs automatically
            optimizer.step()

        # Save checkpoint only on rank 0
        if rank == 0:
            torch.save(model.module.state_dict(), f"checkpoint_epoch{epoch}.pt")

    dist.destroy_process_group()

# Launch with: torchrun --nproc_per_node=4 train.py
```

### 15.2 FSDP for Large Models

FSDP shards model parameters, gradients, and optimizer state across GPUs. Required when the model doesn't fit on a single GPU even in FP16.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.bert.modeling_bert import BertLayer
import functools

def setup_fsdp(model, rank):
    # Mixed precision: params in BF16, reduction in FP32
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    # Wrap individual transformer layers (not the whole model at once)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BertLayer}
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=rank,
    )
    return model
```

---

## 16. Export and Deployment

```python
import torch

# torch.compile: the 2.0+ way to accelerate training/inference
# No code changes needed — just wrap the model
model = torch.compile(model, mode="default")
# mode options: "default", "reduce-overhead" (fastest), "max-autotune" (slower compile, fastest run)

# ONNX export for cross-framework serving
def export_to_onnx(model, dummy_input, output_path: str):
    model.eval()
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {output_path}")

# Production inference: load once, reuse
class ModelServer:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = load_model(checkpoint_path, self.device)
        self.model.eval()

    @torch.inference_mode()  # more aggressive than no_grad — disables autograd globally
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        return self.model(inputs).cpu()

    def predict_batch(self, texts: list, tokenizer, max_length: int = 128) -> list:
        encodings = tokenizer(texts, padding=True, truncation=True,
                             max_length=max_length, return_tensors='pt')
        with torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in encodings.items()})
        return logits.softmax(dim=-1).cpu().numpy().tolist()
```

---

## 17. The Modern Recipe

The opinionated PyTorch workflow for production ML in 2025:

1. **Start single-GPU**: master the training loop, validation, and checkpointing before adding complexity.
2. **Mixed precision by default**: `torch.autocast` + `GradScaler` gives 2× memory and 1.5-2× speed with zero accuracy loss on Volta+ GPUs.
3. **Debug first**: overfit one batch to verify the model architecture and loss function before running a full training.
4. **Profile DataLoader first**: 80% of "GPU is slow" complaints are actually DataLoader CPU bottlenecks. Use `num_workers=4, pin_memory=True, persistent_workers=True`.
5. **AdamW + cosine warmup**: the default optimizer for most tasks. Warmup prevents large gradients early; cosine decay prevents oscillation near convergence.
6. **Gradient clipping at max_norm=1.0**: mandatory for transformer training.
7. **torch.compile for free speed**: wrapping with `torch.compile(model)` gives 10-30% training speedup on PyTorch 2.0+ with no code changes.
8. **DDP before FSDP**: use DDP when model fits on one GPU. Switch to FSDP only for models that don't.

**PyTorch vs alternatives**:

| Framework | Strengths | When to use |
|---|---|---|
| PyTorch | Full control, debugging, research | Default for custom models |
| JAX/Flax | Functional, JIT, TPU-native | Google/DeepMind ecosystem |
| TF/Keras | Legacy, mobile, TFLite | Existing TF deployments |
| Lightning | Boilerplate reduction | When loop management is overhead |

---

## 18. References

### Core Documentation
- [PyTorch official docs](https://pytorch.org/docs/stable/) — authoritative reference
- [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Data loading tutorial](https://pytorch.org/docs/stable/data.html)
- [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

### Key Blogs and Papers
- [Andrej Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the best PyTorch intro for LLMs
- [Sebastian Raschka: PyTorch Performance Tips](https://sebastianraschka.com/blog/) — practical training optimization
- PyTorch 2.0 paper: Ansel et al. (2024). *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.*
- [torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
