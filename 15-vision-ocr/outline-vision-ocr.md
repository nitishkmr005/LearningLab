# 15 — Vision & OCR

Exhaustive learning path for computer vision and OCR: image processing, CNNs, detection, segmentation, vision transformers, and document understanding.

---

## 01 — Image Fundamentals
Pixels, channels, color spaces (RGB, BGR, HSV, LAB); image I/O with PIL/OpenCV; normalization; data types (uint8 vs float32).
- https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- https://pillow.readthedocs.io/en/stable/handbook/tutorial.html

## 02 — Classical Image Processing
Convolution & filters (Gaussian, Sobel, Laplacian); morphological ops (erosion, dilation); thresholding (Otsu); histogram equalization; CLAHE.
- https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html

## 03 — Feature Extraction: SIFT, HOG, ORB
SIFT keypoints & descriptors; HOG for person detection; ORB (fast alternative); feature matching; homography; applications before deep learning.
- https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html

## 04 — CNN Architecture Foundations
Convolutional layer; receptive field; stride, padding; pooling; flattening to FC; parameter count; inductive biases (translation equivariance).
- https://cs231n.github.io/convolutional-networks/

## 05 — Landmark CNN Architectures
LeNet → AlexNet → VGG → GoogLeNet (Inception) → ResNet (skip connections) → EfficientNet (compound scaling); torchvision pretrained models.
- https://pytorch.org/vision/stable/models.html
- https://arxiv.org/abs/1512.03385

## 06 — Transfer Learning & Fine-Tuning
ImageNet pretrained weights; feature extraction vs full fine-tuning; layer freezing; learning rate warm-up; domain adaptation; data augmentation strategies.
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## 07 — Data Augmentation for Vision
Random crop, flip, color jitter, rotation, cutout, mixup, cutmix; Albumentations library; test-time augmentation (TTA); augmentation pipelines.
- https://albumentations.ai/docs/
- https://arxiv.org/abs/1710.09412

## 08 — Object Detection
Anchor-based: RCNN family (Faster RCNN), YOLO, SSD; anchor-free: FCOS, CenterNet; mAP, IoU; NMS; torchvision detection API.
- https://arxiv.org/abs/1506.01497
- https://docs.ultralytics.com/

## 09 — Image Segmentation
Semantic (FCN, DeepLab); instance (Mask RCNN); panoptic; U-Net for medical imaging; IoU/Dice metrics; loss functions (focal, dice).
- https://arxiv.org/abs/1505.04597
- https://smp.readthedocs.io/en/latest/

## 10 — Vision Transformers (ViT)
Patch embedding; position encoding; CLS token; ViT vs CNN; DeiT (data-efficient); Swin Transformer (hierarchical, shifted windows).
- https://arxiv.org/abs/2010.11929
- https://arxiv.org/abs/2103.14030

## 11 — Multimodal Vision-Language Models
CLIP (contrastive image-text); BLIP/BLIP-2; LLaVA; Flamingo; image captioning; visual question answering (VQA); zero-shot classification.
- https://arxiv.org/abs/2103.00020
- https://arxiv.org/abs/2301.12597

## 12 — OCR Fundamentals
Text detection (EAST, DBNet, CRAFT); text recognition (CRNN + CTC, ASTER); end-to-end OCR; character-level vs word-level; language-specific challenges.
- https://arxiv.org/abs/1507.05717
- https://github.com/clovaai/deep-text-recognition-benchmark

## 13 — OCR Libraries & Document Processing
Tesseract (pytesseract); EasyOCR; PaddleOCR; AWS Textract / Google Vision API; table extraction; layout analysis with LayoutParser.
- https://github.com/JaidedAI/EasyOCR
- https://github.com/PaddlePaddle/PaddleOCR

## 14 — Document Understanding
LayoutLM / LayoutLMv3 (text + layout + image); DocFormer; key-value extraction; form understanding; invoice parsing; benchmarks (FUNSD, CORD, DocVQA).
- https://arxiv.org/abs/1912.13318
- https://huggingface.co/microsoft/layoutlmv3-base

## 15 — Vision Evaluation Metrics & Production
Image classification (Top-1/5 accuracy); detection (mAP@50, mAP@50:95); segmentation (mIoU, Dice); model latency; ONNX/TensorRT export; serving with Triton.
- https://docs.ultralytics.com/guides/triton-inference-server/
- https://docs.nvidia.com/deeplearning/triton-inference-server/
