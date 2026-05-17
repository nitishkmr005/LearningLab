# 04 — PyTorch

Exhaustive learning path for deep learning with PyTorch: tensors, autograd, model building, training, and deployment.

---

## 01 — Tensors & Operations
Tensor creation (zeros, ones, rand, arange); dtypes; shape/view/reshape/squeeze/unsqueeze; device (CPU/CUDA); in-place ops.
- https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- https://pytorch.org/docs/stable/tensors.html

## 02 — Autograd & Computational Graphs
requires_grad; backward pass; grad_fn; torch.no_grad; detach; accumulating vs zeroing gradients; retain_graph.
- https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- https://pytorch.org/docs/stable/autograd.html

## 03 — nn.Module & Layer Building Blocks
Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Embedding; forward(); parameter registration; children/modules; state_dict.
- https://pytorch.org/docs/stable/nn.html
- https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## 04 — Loss Functions
MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, NLLLoss, HuberLoss; reduction modes; custom loss; label smoothing.
- https://pytorch.org/docs/stable/nn.html#loss-functions

## 05 — Optimizers & Learning Rate Schedules
SGD (momentum, weight_decay), Adam, AdamW, RMSProp; step(), zero_grad(); LR schedulers (StepLR, CosineAnnealingLR, OneCycleLR); warm-up.
- https://pytorch.org/docs/stable/optim.html

## 06 — Dataset & DataLoader
Custom Dataset (__len__, __getitem__); DataLoader (batch_size, shuffle, num_workers, pin_memory); collate_fn; IterableDataset.
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- https://pytorch.org/docs/stable/data.html

## 07 — Training Loop Patterns
Train/eval mode; typical epoch loop; mixed precision with torch.cuda.amp (autocast, GradScaler); gradient clipping; early stopping.
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

## 08 — Convolutional Neural Networks
Conv2d, MaxPool2d, AdaptiveAvgPool; receptive field; LeNet → ResNet skeleton; torchvision models; transfer learning.
- https://pytorch.org/vision/stable/models.html
- https://cs231n.github.io/convolutional-networks/

## 09 — Recurrent Networks & LSTM
RNN, LSTM, GRU; hidden state; pack_padded_sequence; bidirectional; seq2seq encoder-decoder; gradient vanishing.
- https://pytorch.org/docs/stable/nn.html#recurrent-layers
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## 10 — Attention & Transformers in PyTorch
MultiheadAttention; scaled dot-product attention; PositionalEncoding; nn.Transformer; torch.nn.functional.scaled_dot_product_attention.
- https://pytorch.org/docs/stable/nn.html#transformer-layers
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html

## 11 — Model Checkpointing & Serialization
save/load state_dict; full model save; torch.jit.script (TorchScript); ONNX export; versioning best practices.
- https://pytorch.org/tutorials/beginner/saving_loading_models.html

## 12 — Debugging & Profiling
torch.autograd.set_detect_anomaly; nan/inf detection; torch.profiler (CPU+CUDA traces); memory_allocated; torchinfo (model summary).
- https://pytorch.org/docs/stable/profiler.html
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

## 13 — Distributed Training
DataParallel vs DistributedDataParallel; torchrun; process groups; all_reduce; gradient sync; FSDP overview.
- https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- https://pytorch.org/docs/stable/fsdp.html

## 14 — Quantization & Pruning
Post-training quantization (dynamic, static); QAT; torch.ao.quantization; unstructured/structured pruning; torch.nn.utils.prune.
- https://pytorch.org/docs/stable/quantization.html
- https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

## 15 — PyTorch Lightning & Ecosystem
LightningModule (training_step, configure_optimizers); Trainer (devices, precision, callbacks); WandB/TensorBoard logging; torchmetrics.
- https://lightning.ai/docs/pytorch/stable/
- https://torchmetrics.readthedocs.io/en/stable/
