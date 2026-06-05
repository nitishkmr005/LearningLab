# Computer Vision and OCR: From CNNs to Vision Transformers and Document Understanding

> A comprehensive guide to image processing, detection, segmentation, ViT, CLIP, OCR, and video understanding — for ML Engineers and AI Engineers.

---

## Table of Contents

1. [Image Fundamentals](#1-image-fundamentals)
2. [Classical Image Processing](#2-classical-image-processing)
3. [CNN Architecture Foundations](#3-cnn-architecture-foundations)
4. [Landmark CNN Architectures](#4-landmark-cnn-architectures)
5. [Transfer Learning and Fine-Tuning](#5-transfer-learning-and-fine-tuning)
6. [Data Augmentation for Vision](#6-data-augmentation-for-vision)
7. [Object Detection](#7-object-detection)
8. [Image Segmentation](#8-image-segmentation)
9. [Vision Transformers (ViT)](#9-vision-transformers-vit)
10. [Multimodal Vision-Language Models](#10-multimodal-vision-language-models)
11. [OCR Fundamentals](#11-ocr-fundamentals)
12. [OCR Libraries and Document Processing](#12-ocr-libraries-and-document-processing)
13. [Document Understanding](#13-document-understanding)
14. [SAM and SAM 2: Segment Anything](#14-sam-and-sam-2-segment-anything)
15. [DINOv2: Self-Supervised Vision Features](#15-dinov2-self-supervised-vision-features)
16. [Video Understanding](#16-video-understanding)
17. [Vision Evaluation Metrics and Production](#17-vision-evaluation-metrics-and-production)
18. [References](#18-references)

---

## 1. Image Fundamentals

A digital image is a 2D array of pixels. Each pixel stores one intensity value (grayscale) or three values (color).

**Color spaces:**
- **RGB:** Red, Green, Blue — the standard for displays and most DL models
- **BGR:** OpenCV's default (historical convention from Windows BMP)
- **HSV:** Hue (color), Saturation (vividness), Value (brightness) — useful for color-based thresholding
- **LAB:** perceptually uniform; L = lightness, A = green-red, B = blue-yellow

**Data types:** raw images are uint8 (0–255 per channel). DL models expect float32 normalized to [0,1] or standardized with ImageNet mean/std.

```python
import numpy as np
from PIL import Image
import cv2

# PIL (Pillow) — more Pythonic, used in torchvision
img_pil = Image.open("photo.jpg").convert("RGB")
img_arr = np.array(img_pil)                                  # HWC, uint8, RGB
print(f"Shape: {img_arr.shape}, dtype: {img_arr.dtype}")     # e.g., (480, 640, 3), uint8

# OpenCV — faster for classical processing
img_bgr = cv2.imread("photo.jpg")                            # HWC, uint8, BGR by default
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)           # convert BGR → RGB
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)           # convert to HSV

# Normalize for DL
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])
img_float = img_rgb.astype(np.float32) / 255.0              # scale to [0, 1]
img_norm = (img_float - imagenet_mean) / imagenet_std        # ImageNet normalization

# Resize
img_resized = cv2.resize(img_bgr, (224, 224))                # resize to 224×224
```

---

## 2. Classical Image Processing

Before deep learning, computer vision relied on handcrafted operations. These remain useful for preprocessing, augmentation, and document processing pipelines.

**Convolution and filters:**

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)          # load as grayscale

# Gaussian blur: remove noise
blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)          # 5×5 kernel, σ=1.0

# Sobel edge detection: gradient magnitude
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)         # horizontal gradient
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)         # vertical gradient
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)                 # gradient magnitude

# Laplacian: second-order edge detection
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Otsu's thresholding: automatic threshold selection from histogram
_, binary = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operations: useful for text preprocessing in OCR
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)             # shrink white regions
dilated = cv2.dilate(binary, kernel, iterations=1)           # expand white regions
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)    # remove noise (erode then dilate)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)   # close holes (dilate then erode)

# CLAHE: contrast-limited adaptive histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)                                  # improves local contrast
```

---

## 3. CNN Architecture Foundations

A **Convolutional Neural Network (CNN)** applies learnable filters across the image. Unlike fully connected layers, convolutions:
1. Share weights (the same filter applied everywhere) → fewer parameters
2. Preserve spatial structure → maintain locality
3. Are equivariant to translation → recognizing a cat works regardless of where it is

**Key hyperparameters:**

```
Output size = floor((Input + 2*Padding - Kernel) / Stride) + 1
```

```python
import torch
import torch.nn as nn

# Convolution layer breakdown
conv = nn.Conv2d(
    in_channels=3,                         # RGB input
    out_channels=64,                       # 64 feature maps
    kernel_size=3,                         # 3×3 filter
    stride=1,                              # move 1 pixel at a time
    padding=1                              # pad edges to preserve spatial size
)

# Parameter count: 64 filters × (3 × 3 × 3 weights + 1 bias) = 1,792
params = sum(p.numel() for p in conv.parameters())
print(f"Conv2d parameters: {params}")

# Receptive field: how much of the input each output pixel "sees"
# Each 3×3 conv adds 2 pixels to the receptive field
# After 5 stacked 3×3 convs: RF = 1 + 5×(3-1) = 11×11

class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),   # 224×224
            nn.MaxPool2d(2),                                                   # 112×112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 112×112
            nn.MaxPool2d(2),                                                   # 56×56
            nn.AdaptiveAvgPool2d((1, 1))                                       # 1×1 (global average pool)
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).squeeze(-1).squeeze(-1)  # (batch, 64)
        return self.classifier(features)
```

🎯 **Interview prep:** "Why use BatchNorm?" — normalizes activations to zero mean/unit variance within a batch, reducing internal covariate shift. Allows higher learning rates and acts as a regularizer. Insert after Conv, before activation.

---

## 4. Landmark CNN Architectures

The history of CNN architectures is a story of scale and efficiency:

| Architecture | Year | Params | Top-1 ImageNet | Key Innovation |
|---|---|---|---|---|
| LeNet-5 | 1998 | 60K | ~99% on MNIST | First successful CNN |
| AlexNet | 2012 | 60M | 63.3% | Deep CNN, ReLU, dropout |
| VGG-16 | 2014 | 138M | 74.4% | Small 3×3 filters, depth |
| GoogLeNet (Inception) | 2014 | 7M | 74.8% | Inception modules, 1×1 convs |
| ResNet-50 | 2015 | 25M | 76.0% | Skip connections (residuals) |
| EfficientNet-B0 | 2019 | 5.3M | 77.0% | Compound scaling (depth/width/resolution) |

**ResNet skip connections** ([He et al., 2016](https://arxiv.org/abs/1512.03385)) solved the vanishing gradient problem in very deep networks. The residual block learns `H(x) = F(x) + x` instead of `F(x)` directly. If the optimal transformation is close to the identity, it's much easier to learn F(x) ≈ 0 than H(x) ≈ x:

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual                                  # skip connection
        return self.relu(out)
```

---

## 5. Transfer Learning and Fine-Tuning

Training a vision model from scratch requires millions of labeled examples. Transfer learning starts from ImageNet-pretrained weights and adapts them to your task.

**Feature extraction:** freeze all layers, replace the final classifier, train only the classifier. Works when your domain is similar to ImageNet (natural images).

**Full fine-tuning:** start from pretrained weights, unfreeze all layers, train with a small learning rate. Works better when your domain differs (medical, satellite, documents).

**Layer freezing strategy:**
1. First, train only the new head (fast, avoids catastrophic forgetting)
2. Then unfreeze the last few convolutional blocks
3. Finally, unfreeze all layers with a very small learning rate (1e-5)

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for your task (e.g., 5 classes)
n_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(n_features, 5)                                 # 5-class head
)
# Only the new head has requires_grad=True

# Phase 2: unfreeze last residual block
for param in model.layer4.parameters():
    param.requires_grad = True
```

---

## 6. Data Augmentation for Vision

Augmentation is the cheapest regularization technique for vision. Applied on-the-fly during training, it creates variability that reduces overfitting.

**Standard augmentations:**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Training augmentation pipeline
train_transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),  # crop 70-100% of image
    A.HorizontalFlip(p=0.5),                                        # mirror image
    A.ColorJitter(brightness=0.3, contrast=0.3,
                  saturation=0.3, hue=0.1, p=0.8),                 # color augmentation
    A.GaussNoise(p=0.3),                                            # add noise
    A.Rotate(limit=15, p=0.5),                                     # random rotation
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3), # cutout
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),                         # ImageNet normalization
    ToTensorV2()                                                     # HWC → CHW tensor
])

# Minimal validation transform (no augmentation — just resize + normalize)
val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

**MixUp** ([Zhang et al., 2017](https://arxiv.org/abs/1710.09412)): linearly interpolate two training examples and their labels. Forces the model to learn linear behavior between classes.

**CutMix:** cut a rectangular region from one image and paste it onto another, mixing labels proportionally to the area.

---

## 7. Object Detection

Detection predicts bounding boxes `[x1, y1, x2, y2]` plus class labels for each object.

**Two-stage detectors (Faster R-CNN):** region proposal network (RPN) → RoI pooling → classifier + box regressor. Accurate but slower.

**One-stage detectors (YOLO):** directly predict boxes and classes from grid cells. Fast, suitable for real-time.

**YOLO series:** YOLOv8/v11 (Ultralytics) is the production standard for real-time detection. mAP@50:95 on COCO reaches ~55% for YOLOv8x.

```python
from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")                                   # nano model (fast, 3.2M params)

# Inference
results = model("photo.jpg", conf=0.5, iou=0.45)             # confidence threshold, NMS IoU
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()                       # (N, 4) bounding boxes
    classes = r.boxes.cls.cpu().numpy().astype(int)          # class IDs
    confs = r.boxes.conf.cpu().numpy()                       # confidence scores
    for box, cls, conf in zip(boxes, classes, confs):
        label = model.names[cls]
        print(f"{label} ({conf:.2f}): [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

# Fine-tune on custom data
# model.train(data="custom_dataset.yaml", epochs=50, imgsz=640)
```

**IoU (Intersection over Union):** primary detection metric. IoU = area of intersection / area of union. IoU > 0.5 is typically counted as a correct detection.

**mAP (mean Average Precision):** average AP across classes and IoU thresholds. mAP@50 = mAP at IoU 0.5; mAP@50:95 = averaged over 0.5 to 0.95 in steps of 0.05.

---

## 8. Image Segmentation

Segmentation predicts a class label for every pixel.

**Semantic segmentation:** all cats are one class, all dogs another. No instance distinction.

**Instance segmentation:** each individual object is a separate segment.

**Panoptic segmentation:** combines both — "stuff" (sky, road) as semantic, "things" (cars, people) as instances.

**U-Net** ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)): encoder-decoder with skip connections. The encoder downsamples (captures global context); the decoder upsamples (recovers spatial detail); skip connections add the encoder's fine details back to the decoder. Standard for medical image segmentation.

```python
# Segmentation Model Pytorch (SMP) library
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="resnet34",                                 # pretrained encoder backbone
    encoder_weights="imagenet",                              # ImageNet pretraining
    in_channels=3,                                           # RGB input
    classes=2                                                # binary segmentation (foreground/background)
)

# Loss for segmentation
criterion = smp.losses.DiceLoss(mode="binary")               # Dice loss for class imbalance

x = torch.randn(4, 3, 256, 256)                             # batch of 4 images
logits = model(x)                                            # (4, 2, 256, 256) pixel-wise logits
print(f"Output shape: {logits.shape}")
```

**Dice coefficient:** 2|A∩B|/(|A|+|B|) — measures overlap between predicted and ground truth masks. More robust to class imbalance than pixel accuracy.

---

## 9. Vision Transformers (ViT)

**ViT** ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) applies the Transformer architecture directly to images. The image is divided into non-overlapping patches (typically 16×16), each flattened and linearly projected into an embedding. A [CLS] token aggregates global information for classification.

For a 224×224 image with 16×16 patches: 196 patches + 1 CLS = 197 sequence positions. This sequence is fed into a standard Transformer encoder.

```python
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

image = Image.open("photo.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    print(f"Predicted class: {model.config.id2label[predicted_class]}")
```

**Swin Transformer** ([Liu et al., 2021](https://arxiv.org/abs/2103.14030)): introduces hierarchical feature maps (like CNNs) and shifted window attention (attention computed within local windows instead of globally). Reduces O(n²) attention to O(n) per layer while maintaining strong performance on dense tasks (detection, segmentation).

ViT vs CNN:
- ViT: better at global context, weaker inductive bias, needs more data or strong augmentation
- CNN: stronger inductive bias (translation equivariance), better on small datasets
- Modern hybrids (ConvNeXt, EfficientViT) combine both

---

## 10. Multimodal Vision-Language Models

**CLIP** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)): trained on 400M (image, text) pairs from the web using contrastive loss. Image encoder and text encoder are trained to produce similar embeddings for matching pairs.

Zero-shot classification: encode class names as text, encode the image, find the closest class embedding.

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)
labels = ["a photo of a dog", "a photo of a cat", "a photo of a car"]
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)               # image embedding
    text_features = model.encode_text(text)                  # text embeddings

# Cosine similarity → softmax → probabilities
logits_per_image = image_features @ text_features.T          # dot product
probs = logits_per_image.softmax(dim=-1).cpu().numpy()
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.3f}")
```

**LLaVA** ([Liu et al., 2023](https://arxiv.org/abs/2301.12597)): projects CLIP image features into LLaMA's embedding space via a linear layer, enabling visual question answering with a language model. The minimal architecture makes it easy to fine-tune.

**BLIP-2** ([Li et al., 2023](https://arxiv.org/abs/2301.12597)): Q-Former architecture — a learnable query module bridges the frozen image encoder and frozen LLM. The Q-Former distills visual information into a fixed number of query tokens fed to the LLM.

---

## 11. OCR Fundamentals

OCR (Optical Character Recognition) converts document images to machine-readable text. Two-stage pipeline:

**Text detection:** find where text is in the image (bounding boxes around words or lines).
- **EAST** ([Zhou et al., 2017](https://arxiv.org/abs/1704.03155)): anchor-free, rotated bounding boxes; fast
- **DBNet:** differentiable binarization; strong for curved and oriented text
- **CRAFT:** character-level detection with affinity maps

**Text recognition:** read the text within each detected region.
- **CRNN + CTC** ([Shi et al., 2017](https://arxiv.org/abs/1507.05717)): CNN features → BiLSTM → CTC; baseline for scene text recognition
- **ASTER:** attention-based with spatial transformer for curved text
- **TrOCR:** Transformer encoder-decoder (ViT + BERT-style decoder)

**Key challenges:**
- Rotated, curved, or perspective-distorted text
- Low resolution or blurry images
- Multiple fonts, handwriting
- Language mixing

---

## 12. OCR Libraries and Document Processing

```python
# EasyOCR: supports 80+ languages, no setup required
import easyocr
import cv2

reader = easyocr.Reader(['en'])                               # English OCR
results = reader.readtext("document.jpg")

for (bbox, text, confidence) in results:
    print(f"Text: '{text}' (conf: {confidence:.2f})")
    # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] quadrilateral

# PaddleOCR: multi-language, strong on Chinese, table detection
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')               # auto-correct text angle
result = ocr.ocr("document.jpg", cls=True)
for line in result[0]:
    print(f"Bbox: {line[0]}, Text: {line[1][0]}, Conf: {line[1][1]:.2f}")

# Tesseract (pytesseract): traditional, good for clean documents
import pytesseract
from PIL import Image

img = Image.open("document.jpg")
text = pytesseract.image_to_string(img, lang='eng')          # full page text
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)  # word-level bboxes
```

**Preprocessing for better OCR:**
1. Grayscale conversion
2. Deskew (correct rotation)
3. CLAHE or binarization to improve contrast
4. Denoise (Gaussian or median filter)

```python
def preprocess_for_ocr(img_path: str) -> np.ndarray:
    """Standard preprocessing pipeline for document OCR."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Binarize with Otsu
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Deskew using Hough transform
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]                      # rotation angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    deskewed = cv2.warpAffine(binary, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed
```

---

## 13. Document Understanding

Raw OCR gives text but no structure — it doesn't know that "Invoice #1234" is the invoice number or that "Amount Due: $500" is a financial field. Document understanding models combine text, layout (bounding box positions), and visual features.

**LayoutLM** ([Xu et al., 2019](https://arxiv.org/abs/1912.13318)) and **LayoutLMv3** ([Huang et al., 2022](https://huggingface.co/microsoft/layoutlmv3-base)): BERT-like models that add 2D positional embeddings for bounding box coordinates. Input: text tokens + their (x1, y1, x2, y2) positions + optionally image patches. Fine-tuned on document understanding tasks like key-value extraction, form understanding, and invoice parsing.

**Benchmarks:**
- **FUNSD:** form understanding (name-value pairs in noisy scanned forms)
- **CORD:** receipt parsing (structured JSON extraction from receipts)
- **DocVQA:** visual question answering over document images

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base",
                                                  apply_ocr=True)  # built-in OCR
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=7                                              # e.g., B-KEY, I-KEY, B-VALUE, etc.
)

image = Image.open("invoice.jpg").convert("RGB")
encoding = processor(image, return_tensors="pt")             # runs OCR internally

with torch.no_grad():
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze()        # token-level label predictions
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())
    for token, pred in zip(tokens, predictions.tolist()):
        print(f"{token:20s} → {model.config.id2label[pred]}")
```

---

## 14. SAM and SAM 2: Segment Anything

**SAM (Segment Anything Model)** ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)) from Meta enables zero-shot, promptable segmentation. Given a point, bounding box, or rough mask as a prompt, SAM segments any object in the image — without being trained on the specific object class.

Architecture: ViT image encoder → lightweight prompt encoder (for point/box/mask prompts) → mask decoder (2-layer Transformer).

**SAM 2** ([Ravi et al., 2024](https://github.com/facebookresearch/sam2)) extends SAM to video with a memory attention module that tracks object appearances across frames — enabling consistent segmentation across a video sequence.

```python
# pip install segment-anything
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

image = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)                               # precompute image embedding

# Segment from a point prompt (click at x=300, y=200)
input_point = np.array([[300, 200]])
input_label = np.array([1])                                  # 1 = foreground

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True                                     # return 3 candidate masks
)
# masks: (3, H, W) boolean arrays
best_mask = masks[np.argmax(scores)]                         # pick highest-confidence mask
```

SAM use cases: annotation acceleration (replace polygon annotation), interactive segmentation, fine-grained data curation, document layout analysis.

---

## 15. DINOv2: Self-Supervised Vision Features

**DINOv2** ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)) from Meta learns rich visual features without any labels, using self-distillation with a student-teacher framework. The ViT backbone trained with DINOv2 produces features that work as a strong backbone for dense tasks without fine-tuning.

Key finding: DINOv2 frozen features outperform ImageNet-supervised features on:
- Semantic segmentation (linear probe on ADE20K)
- Monocular depth estimation
- K-NN image retrieval
- Few-shot classification

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

image = Image.open("photo.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# patch_embeddings: (1, 196, 768) — one embedding per 16×16 patch
patch_embeddings = outputs.last_hidden_state[:, 1:, :]       # exclude CLS token
cls_embedding = outputs.last_hidden_state[:, 0, :]           # CLS = global image representation

# Use patch embeddings for dense tasks (segmentation, depth)
# Use CLS for image-level tasks (classification, retrieval)
print(f"Patch embeddings: {patch_embeddings.shape}")         # (1, 196, 768)
print(f"CLS embedding: {cls_embedding.shape}")               # (1, 768)
```

DINOv2 is particularly useful when labeled data is scarce — fine-tune just the head on 10 labeled examples per class and achieve strong results.

---

## 16. Video Understanding

Video is a sequence of frames. The challenge: temporal modeling (what changes across frames) + spatial modeling (what's in each frame) + efficiency (videos are large).

**TimeSformer** ([Bertasius et al., 2021](https://arxiv.org/abs/2102.05095)): factorized space-time attention — apply spatial attention within each frame, then temporal attention across frames at the same spatial position. More efficient than full space-time attention.

**VideoMAE** ([Tong et al., 2022](https://arxiv.org/abs/2203.12602)): self-supervised video pre-training via masked autoencoding — mask 90% of video tokens, reconstruct them. The very high masking ratio forces the model to learn temporal consistency.

**Frame sampling strategies:**
- **Uniform sampling:** select K frames evenly spaced. Misses fast motions.
- **Keyframe sampling:** detect scene changes, sample one frame per scene. Efficient.
- **Dense sampling:** all frames, sliding window. Accurate but slow.

```python
import cv2
import torch
from torchvision.transforms import functional as TF

def sample_video_frames(video_path: str, n_frames: int = 8) -> torch.Tensor:
    """Sample N frames uniformly from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / n_frames) for i in range(n_frames)]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)                # seek to frame
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = TF.to_tensor(
                TF.resize(TF.to_pil_image(frame_rgb), [224, 224])
            )
            frames.append(tensor)
    cap.release()

    video_tensor = torch.stack(frames)                       # (T, C, H, W)
    return video_tensor

# video_tensor shape: (8, 3, 224, 224) for 8 sampled frames
```

**Action recognition benchmarks:**
- **Kinetics-400/700:** large-scale action recognition (400/700 classes)
- **Something-Something v2:** hand-object interactions (tests temporal reasoning, not just appearance)

---

## 17. Vision Evaluation Metrics and Production

**Classification:**
- Top-1 accuracy: fraction where the highest-probability class is correct
- Top-5 accuracy: fraction where the correct class is in the top 5

**Detection:**
- **IoU (Intersection over Union):** area of overlap / area of union; threshold 0.5 for "correct"
- **mAP@50:** mean AP at IoU 0.5 threshold across all classes
- **mAP@50:95:** mean AP averaged over IoU thresholds 0.5–0.95

**Segmentation:**
- **mIoU:** mean IoU across all semantic classes
- **Dice coefficient:** 2|A∩B|/(|A|+|B|); equivalent to F1 for binary segmentation

**Production serving with ONNX/TensorRT:**

```python
import torch
import torchvision.models as models

# Export to ONNX
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"},               # dynamic batch dimension
                  "output": {0: "batch_size"}},
    opset_version=17
)

# Run with ONNX Runtime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("resnet50.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_data = np.random.randn(4, 3, 224, 224).astype(np.float32)  # batch of 4
outputs = session.run(["output"], {"input": input_data})
print(f"Output shape: {outputs[0].shape}")                  # (4, 1000)
```

**NVIDIA Triton Inference Server** provides GPU-accelerated serving with dynamic batching, model ensemble, and concurrent model execution. Supports ONNX, TensorRT, PyTorch, and TensorFlow backends.

---

## 18. References

### CNN Architectures

- [He et al. (2015). Deep Residual Learning for Image Recognition (ResNet).](https://arxiv.org/abs/1512.03385)
- [Zhang et al. (2017). mixup: Beyond Empirical Risk Minimization.](https://arxiv.org/abs/1710.09412)

### Vision Transformers

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT).](https://arxiv.org/abs/2010.11929)
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.](https://arxiv.org/abs/2103.14030)

### Object Detection and Segmentation

- [Ren et al. (2015). Faster R-CNN.](https://arxiv.org/abs/1506.01497)
- [Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

### Vision-Language Models

- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP).](https://arxiv.org/abs/2103.00020)
- [Liu et al. (2023). LLaVA: Visual Instruction Tuning.](https://arxiv.org/abs/2301.12597)

### OCR and Document Understanding

- [Shi et al. (2017). CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN).](https://arxiv.org/abs/1507.05717)
- [Xu et al. (2019). LayoutLM: Pre-training of Text and Layout for Document Image Understanding.](https://arxiv.org/abs/1912.13318)
- [LayoutLMv3 (HuggingFace)](https://huggingface.co/microsoft/layoutlmv3-base)

### Segment Anything and Self-Supervised Vision

- [Kirillov et al. (2023). Segment Anything (SAM).](https://arxiv.org/abs/2304.02643)
- [SAM 2 (Meta)](https://github.com/facebookresearch/sam2)
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision.](https://arxiv.org/abs/2304.07193)

### Video Understanding

- [Bertasius et al. (2021). Is Space-Time Attention All You Need? (TimeSformer).](https://arxiv.org/abs/2102.05095)
- [Tong et al. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.](https://arxiv.org/abs/2203.12602)

### Production

- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Ultralytics Triton Guide](https://docs.ultralytics.com/guides/triton-inference-server/)
- [Segmentation Models PyTorch (SMP)](https://smp.readthedocs.io/en/latest/)
