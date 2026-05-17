# 04 — PyTorch

Production-first study outline for training, fine-tuning, inference, and multi-GPU workloads in PyTorch.

---

## Format Used In This Outline
- `Concept`: what to learn.
- `Why it matters`: where developers actually use it.
- `Typical code`: the minimum pattern you should know.

## 01 — Tensors, Devices, and Shapes
- `Concept`: tensor creation, dtype, shape, broadcasting, CPU vs CUDA, `to(device)`.
- `Why it matters`: most beginner bugs are shape bugs or device mismatches.
- `Typical code`: move batch tensors and model to GPU safely.

## 02 — Autograd
- `Concept`: `requires_grad`, computation graph, `backward`, `no_grad`, `detach`.
- `Why it matters`: training, fine-tuning, gradient accumulation, frozen layers all depend on this.
- `Typical code`: freeze backbone, train only the classifier head.

## 03 — `nn.Module` and Model Structure
- `Concept`: build modules, register parameters, `forward`, `state_dict`.
- `Why it matters`: every reusable model, wrapper, adapter, or head is an `nn.Module`.
- `Typical code`: define encoder, pooling, classification head.

## 04 — Datasets and DataLoaders
- `Concept`: `Dataset`, `IterableDataset`, `DataLoader`, `collate_fn`, `pin_memory`, `num_workers`.
- `Why it matters`: data input pipelines frequently dominate training throughput.
- `Typical code`: tokenized text dataset, image dataset, or tabular dataset with labels.

```python
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## 05 — Training Loop Fundamentals
- `Concept`: `model.train()`, `optimizer.zero_grad()`, forward pass, loss, `backward`, `step`.
- `Why it matters`: this is the muscle memory of PyTorch.
- `Typical code`: epoch loop with validation after each epoch.

```python
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()
```

## 06 — Validation and Inference
- `Concept`: `model.eval()`, `torch.no_grad()`, logits vs probabilities, batch inference.
- `Why it matters`: evaluation bugs often come from forgetting eval mode or applying the wrong output transform.
- `Typical code`: `softmax` for multiclass, `sigmoid` for binary logits.

## 07 — Loss Functions
- `Concept`: `CrossEntropyLoss`, `BCEWithLogitsLoss`, `MSELoss`, class weights, label smoothing.
- `Why it matters`: picking the wrong loss or wrong target shape causes silent failure.
- `Typical code`: imbalanced classification with weighted BCE or focal-loss-style variants.

## 08 — Optimizers and LR Scheduling
- `Concept`: SGD, Adam, AdamW, weight decay, warmup, cosine schedule, OneCycle.
- `Why it matters`: training stability and final performance depend heavily on optimization choices.
- `Typical code`: AdamW plus warmup-cosine for transformer fine-tuning.

## 09 — Saving and Loading Models
- `Concept`: save `state_dict`, load checkpoint, resume training, store optimizer and scheduler state.
- `Why it matters`: full-model pickles are brittle; `state_dict` is the default professional path.
- `Typical code`: checkpoint best validation model and restore later for inference.

```python
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")
```

## 10 — Fine-Tuning Patterns
- `Concept`: freeze/unfreeze, head replacement, low learning rates, discriminative learning rates.
- `Why it matters`: most applied PyTorch work is fine-tuning, not training from scratch.
- `Typical code`: load pretrained vision or language backbone, replace task head, train head first, then unfreeze selectively.

## 11 — Mixed Precision and Throughput Optimization
- `Concept`: `torch.autocast`, `GradScaler`, memory savings, faster training.
- `Why it matters`: common default on modern GPUs.
- `Typical code`: mixed precision inside the training loop when CUDA is available.

## 12 — Gradient Accumulation and Clipping
- `Concept`: accumulate gradients across microbatches, clip exploding gradients.
- `Why it matters`: large effective batch sizes and stable training when memory is tight.
- `Typical code`: accumulate every `k` steps, then optimizer step.

## 13 — Debugging Training
- `Concept`: overfit one batch, inspect gradients, detect NaNs, check target ranges, watch learning curves.
- `Why it matters`: this is how experienced developers debug most training failures.
- `Typical code`: verify the model can drive loss near zero on a tiny sample.

## 14 — Profiling and Input Bottlenecks
- `Concept`: `torch.profiler`, CPU wait time, data stalls, CUDA memory inspection.
- `Why it matters`: many "slow model" complaints are really slow input pipelines.
- `Typical code`: compare GPU utilization before and after increasing `num_workers` or using pinned memory.

## 15 — Common PyTorch Tips Used In Real Projects
- `Concept`: seed control, deterministic flags, gradient checkpointing, compiled graphs, avoiding accidental `.cpu()` calls.
- `Why it matters`: these are the boring details that make runs reproducible and efficient.
- `Typical code`: log seed, library versions, device name, parameter count, and tokenizer version.

## 16 — Transfer Learning and PEFT Mindset
- `Concept`: adapter layers, LoRA-style thinking, frozen backbone plus lightweight trainable modules.
- `Why it matters`: AI engineering increasingly favors parameter-efficient fine-tuning.
- `Typical code`: train a small head or adapter instead of updating the full model.

## 17 — Export and Deployment
- `Concept`: inference-only module loading, TorchScript or ONNX when needed, batching and latency.
- `Why it matters`: model quality is irrelevant if serving is unstable or too slow.
- `Typical code`: load checkpoint once at service startup and reuse it across requests.

## 18 — Distributed Training: DDP First
- `Concept`: `DistributedDataParallel`, `torchrun`, one process per GPU, distributed sampler.
- `Why it matters`: DDP is the standard baseline for multi-GPU training; it scales better than legacy `DataParallel`.
- `Typical code`: launch a training script across 4 GPUs with `torchrun`.

```bash
torchrun --nproc_per_node=4 train.py
```

## 19 — Multi-GPU Data Loading Details
- `Concept`: `DistributedSampler`, `set_epoch`, rank-aware checkpointing.
- `Why it matters`: without this, workers duplicate data or shuffle inconsistently.
- `Typical code`: save checkpoints only on rank 0 and call `sampler.set_epoch(epoch)`.

## 20 — FSDP and Larger-Than-Memory Training
- `Concept`: Fully Sharded Data Parallel, parameter sharding, memory reduction.
- `Why it matters`: relevant once models are too large for naive DDP.
- `Typical code`: keep this after you are already comfortable with DDP.

## 21 — Recommended Progression for ML/AI Engineers
- `Stage 1`: write a clean single-GPU training loop.
- `Stage 2`: add validation, checkpointing, and inference utilities.
- `Stage 3`: optimize throughput with mixed precision and better dataloading.
- `Stage 4`: fine-tune pretrained backbones safely.
- `Stage 5`: move to DDP and then FSDP if scale demands it.

## 22 — Minimal Production Snippets To Add Later
- `Tabular training loop`
- `Text classification fine-tuning`
- `Checkpoint resume`
- `Batch inference service`
- `DDP training template`

## References
- Saving/loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- Data loading: https://pytorch.org/docs/stable/data.html
- Distributed overview: https://docs.pytorch.org/tutorials/distributed.html
- DDP tutorial: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
