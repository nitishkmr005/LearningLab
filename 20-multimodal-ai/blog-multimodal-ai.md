# Multimodal AI: From CLIP to Gemini 1.5 and Beyond

*A complete guide to vision-language models, audio-language models, text-to-image generation, cross-modal retrieval, multimodal benchmarks, and production deployment — for DS/ML/AI/GenAI engineers.*

---

## Table of Contents

1. [The Problem: Why Unimodal Models Hit a Ceiling](#1-the-problem)
2. [History: From Hand-Crafted Features to Foundation Models](#2-history)
3. [Core Architectures: How Multimodal Models Actually Work](#3-core-architectures)
4. [Vision-Language Models](#4-vision-language-models)
5. [Audio-Language Models](#5-audio-language-models)
6. [Video Understanding](#6-video-understanding)
7. [Text-to-Image Generation](#7-text-to-image-generation)
8. [Multimodal Embeddings and Cross-Modal Retrieval](#8-multimodal-embeddings-and-cross-modal-retrieval)
9. [Training Data and Alignment Techniques](#9-training-data-and-alignment-techniques)
10. [Evaluation Benchmarks](#10-evaluation-benchmarks)
11. [Model Comparison: Open-Source and Closed](#11-model-comparison)
12. [Inference and Serving Multimodal Models](#12-inference-and-serving-multimodal-models)
13. [The Modern Recipe](#13-the-modern-recipe)
14. [Interview Q&A](#14-interview-qa)
15. [References](#15-references)

---

## 1. The Problem

A radiologist reads a chest X-ray. Her knowledge integrates visual anatomy — the density of a shadow, the curve of a rib — with years of textual training: case notes, textbooks, clinical guidelines. She doesn't see an image and then separately recall language. The two modalities are fused in a single cognitive act. For decades, AI couldn't come close to this.

The first generation of deep learning systems was strictly unimodal: a convolutional network processed pixels, a recurrent network processed words, and the two worlds never met. An image classifier trained on ImageNet knew nothing about language. A language model trained on books was blind to visual context. This caused cascading failures in real applications: a visual search engine that couldn't find "a red dress similar to the one in this photo," a medical AI that could classify a scan but couldn't answer a doctor's natural-language follow-up question, a voice assistant that couldn't understand what a user was pointing at on screen.

The deeper problem is that intelligence itself is multimodal. The physical world generates aligned signals — a dog barks (audio) and you see its mouth move (vision) and you describe it in words (language). Models trained on any single channel miss the alignment structure that makes generalization possible. A vision model has no way to leverage the fact that "apple" and a picture of an apple refer to the same concept until it has been explicitly taught the correspondence.

Multimodal AI is the field that builds this correspondence at scale. This blog traces how we got from clumsy feature concatenation to models that can analyze an hour-long video, answer questions about a medical image in any language, and generate photorealistic images from a paragraph of text. It covers the architectures, the training tricks, the benchmarks, and the practical engineering decisions you need to deploy these systems in production.

---

## 2. History: From Hand-Crafted Features to Foundation Models

### 2.1 Pre-Deep Learning: Feature Concatenation (2000–2012)

The earliest multimodal systems worked by extracting features separately from each modality — SIFT features for images, bag-of-words for text — and concatenating the resulting vectors into a single classifier. The Visual-Semantic Embedding (VSE) approach from this era learned linear projections to align image and text features in a shared space. The limitation was brutal: the features were not learned jointly, so the shared space captured superficial correlations rather than deep semantics. "Dog" and a photo of a dog aligned because both appeared together in training documents, not because the model understood what a dog *is*.

### 2.2 Deep Visual-Semantic Embeddings (2013–2016)

**Fang et al. (2014)** and contemporaries replaced hand-crafted features with CNN image representations and RNN text representations, training both jointly on image-caption pairs. The DeViSE model ([Frome et al., 2013](https://papers.nips.cc/paper_files/paper/2013/hash/7cce53cf90577442771720a370c3c723-Abstract.html)) mapped ImageNet classifiers into word2vec space, enabling zero-shot visual recognition by finding the nearest word embedding to an image representation. This was the first demonstration that visual and linguistic representations could be placed in the same geometric space — but the space was still small, shallow, and brittle.

The key limitation: the models were discriminative (could match, not generate) and required labeled image-caption pairs, which are expensive and scarce.

### 2.3 Contrastive Pre-training at Scale: CLIP and ALIGN (2021)

The breakthrough insight was that the web already contained billions of aligned image-text pairs in the form of images with alt-text, captions, and surrounding paragraphs. You didn't need human annotations — you needed scale.

**CLIP** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) trained a vision transformer and a text transformer jointly on 400 million (image, text) pairs using contrastive learning. The objective was simple: for a batch of N pairs, make the correct image-text pairs similar and all N²-N incorrect pairs dissimilar. The result was a shared embedding space so powerful that you could describe a new class in natural language and perform zero-shot classification — without a single labeled example — achieving ResNet-50-level accuracy on ImageNet zero-shot. CLIP remains the most widely deployed vision-language backbone in production today.

**ALIGN** ([Jia et al., 2021](https://arxiv.org/abs/2102.05918)) from Google showed the same principle at even larger scale: 1.8 billion noisy image-text pairs, with noise handled by scale rather than filtering. The lesson: contrastive learning is surprisingly robust to label noise when the dataset is large enough.

**Limitation:** CLIP and ALIGN produce embeddings but not language. They can tell you whether an image matches a text but cannot describe the image, answer questions, or reason across multiple steps.

### 2.4 Bridging to Language Models: BLIP, BLIP-2, and Flamingo (2022–2023)

The next era asked: how do we connect CLIP-style vision encoders to the new generation of powerful language models without retraining everything from scratch?

**Flamingo** ([Alayrac et al., 2022](https://arxiv.org/abs/2204.14198)) from DeepMind introduced the **Perceiver Resampler** — a fixed set of learned query tokens that compress a variable-length sequence of image patch features into a small, fixed-size representation. Gated cross-attention layers interspersed within a frozen LLM let the model condition on images without disturbing the language model's internal representations. Trained on interleaved image-text web documents, Flamingo achieved dramatic few-shot results, outperforming models fine-tuned on thousands of labeled examples using just 32 in-context demonstrations. Published at NeurIPS 2022.

**BLIP-2** ([Li et al., 2023](https://arxiv.org/abs/2301.12597)) took a more parameter-efficient approach. Its **Q-Former** is a lightweight transformer with N learned query tokens that attend to frozen image encoder features and output N fixed-size visual tokens. Training happens in two stages: first the Q-Former learns to align visual and language representations using a frozen image encoder; then the Q-Former's outputs are fed as a soft prompt to a frozen LLM. BLIP-2 outperformed Flamingo-80B by 8.7% on zero-shot VQAv2 using 54x fewer trainable parameters — a remarkable efficiency gain.

### 2.5 Instruction Tuning: LLaVA and the Chat Era (2023)

**LLaVA** ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)) asked a different question: what if we used GPT-4 to generate multimodal instruction-following training data, then fine-tuned a smaller model on it? Using CLIP as the vision encoder and Vicuna as the language model with a simple linear projection layer in between, LLaVA achieved 85.1% relative performance vs GPT-4 on synthetic instruction-following tasks. The key insight: the *alignment data* matters more than the architecture complexity.

**LLaVA-1.5** ([Liu et al., 2023](https://arxiv.org/abs/2310.03744)) refined this further — replacing the linear projection with an MLP, using a higher-resolution CLIP ViT-L-336px, and adding academic VQA data. It achieved SotA across 11 benchmarks using only 1.2M publicly available examples, training in one day on 8 A100s. LLaVA-1.5 remains the most widely cited open-source VLM baseline.

> 📚 **Go deeper**: [LLaVA project page](https://llava-vl.github.io/) — interactive demos and all model variants

**Resources**
- [CLIP paper](https://arxiv.org/abs/2103.00020) — foundational contrastive vision-language pre-training
- [BLIP-2 paper](https://arxiv.org/abs/2301.12597) — Q-Former bridging frozen encoders efficiently
- [LLaVA paper](https://arxiv.org/abs/2304.08485) — instruction tuning for multimodal chat

---

## 3. Core Architectures: How Multimodal Models Actually Work

Three architectural patterns dominate production multimodal systems today. Understanding which pattern a model uses tells you its trade-offs before you read a single benchmark number.

### 3.1 Dual-Encoder (Contrastive)

```
Image ──► [Vision Encoder] ──► image_embedding ──────┐
                                                       ▼ cosine_sim
Text  ──► [Text Encoder]  ──► text_embedding  ────────┘
```

The vision and text encoders are trained independently with a contrastive loss to place matching pairs nearby and non-matching pairs far apart in a shared embedding space. At inference, either modality can be the query.

**When to use:** Cross-modal retrieval (image search from text query, text search from image), zero-shot classification, multimodal embeddings for vector databases.

**When NOT to use:** When you need generative output (captioning, VQA with reasoning, multi-step dialogue).

**The contrastive loss (InfoNCE):**

```
L = -log[ exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ) ]

Where:
  I_i    = image embedding for the i-th pair (L2-normalized)
  T_j    = text embedding for the j-th item in the batch
  τ      = temperature parameter (CLIP uses τ = 0.07)
  sim()  = dot product of unit vectors (cosine similarity)
```

**Worked example:**

```
Batch of 3 pairs: (cat_img, "cat"), (dog_img, "dog"), (car_img, "car")

Similarities after encoder (τ=0.07):
  sim(cat_img, "cat")  = 0.9  →  exp(0.9/0.07) = exp(12.86) = 388,000
  sim(cat_img, "dog")  = 0.2  →  exp(0.2/0.07) = exp(2.86)  = 17.5
  sim(cat_img, "car")  = 0.1  →  exp(0.1/0.07) = exp(1.43)  = 4.2

L(cat_img) = -log[ 388000 / (388000 + 17.5 + 4.2) ] ≈ -log(0.9999) ≈ 0.0001

(Perfect match → loss ≈ 0. Wrong match would give loss closer to log(3) ≈ 1.1)
```

> 🎯 **Interview prep**: "Why does CLIP use a temperature parameter τ?" — τ controls the sharpness of the distribution. A small τ (like 0.07) makes the model very confident, concentrating probability mass on the most similar pair. Too small → training instability. Too large → gradients vanish because all pairs look equally similar.

### 3.2 Encoder-Decoder with Cross-Attention Bridge

```
Image ──► [Frozen Vision Encoder] ──► patch_tokens
                                           │
                                    [Q-Former / Perceiver Resampler]
                                           │  (N learned query tokens)
                                           ▼
Text  ──► [Frozen Language Model] ◄── visual_soft_prompt
              │
              ▼
           Generated text
```

A lightweight bridge module (Q-Former in BLIP-2, Perceiver Resampler in Flamingo) compresses the image patches into a fixed number of tokens that the LLM can consume. Both the vision encoder and the LLM stay frozen; only the bridge is trained. This is the most parameter-efficient path to a capable VLM.

**When to use:** When you want to add vision to an existing powerful LLM without retraining it. Great for constrained compute budgets.

**When NOT to use:** The frozen LLM cannot learn new visual concepts that weren't in its text pre-training. Long-chain visual reasoning suffers.

### 3.3 Unified Early Fusion (Native Multimodal)

```
Image ──► [Patchify + Embed] ──► image_tokens ──┐
                                                  ├──► [Shared Transformer] ──► output
Text  ──► [Tokenize + Embed] ──► text_tokens  ──┘
```

Image patches and text tokens are treated as a single interleaved sequence fed into one transformer from the start. The model learns visual and linguistic structure jointly from scratch. GPT-4o, Gemini, and Meta's Llama 4 Maverick use this approach.

**When to use:** When you have the compute to train from scratch and want maximum capability — especially for tasks requiring tight image-text co-reasoning (reading text in images, document understanding, long-context video).

**When NOT to use:** Expensive to train. Fine-tuning risks catastrophic forgetting of either modality if not done carefully.

> 🏭 **Production note**: The dominant production pattern as of 2025 is still the bridge approach (LLaVA-style) for open-source deployments because you can swap out the LLM for a newer one without retraining the visual components. Native early fusion models (GPT-4o, Gemini) are strictly closed API.

**Resources**
- [Flamingo paper](https://arxiv.org/abs/2204.14198) — Perceiver Resampler and gated cross-attention details
- [The Evolution of Multimodal Model Architectures](https://arxiv.org/pdf/2405.17927) — survey comparing all three patterns

---

## 4. Vision-Language Models

Vision-language models (VLMs) take images and text as input and produce text as output. They are the workhorses of multimodal AI: powering visual question answering, image captioning, document understanding, medical image analysis, and chart reasoning.

### 4.1 The Visual Instruction Tuning Recipe

The dominant open-source recipe, established by LLaVA and widely adopted, is:

```
Vision Encoder  (CLIP ViT-L/14 or ViT-L/14@336)
      │
      ▼
  MLP Projector  (2-layer linear with GELU)
      │
      ▼
Language Model  (LLaMA-3, Mistral, Qwen2, etc.)
```

**Stage 1 — Pre-training:** Freeze the LLM and vision encoder, train only the MLP projector on image-caption pairs (CC3M, LAION-CC-SBU). Goal: align the visual feature space to the LLM's token space.

**Stage 2 — Visual instruction tuning:** Unfreeze the MLP and the LLM (or apply LoRA), train on multimodal instruction-following data (LLaVA-Instruct-150K, ShareGPT4V, etc.). Goal: teach the model to respond conversationally to image+text prompts.

```python
# Minimal LLaVA-style inference with HuggingFace Transformers
# pip install transformers accelerate torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch, requests

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"  # real HF model ID
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

image = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png", stream=True).raw)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What objects are in this image? Describe the scene."}
        ]
    }
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)  # format as chat
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=200)  # generate up to 200 tokens
print(processor.decode(output[0], skip_special_tokens=True))  # decode and print
```

> 🏭 **Production note**: Always use `apply_chat_template` — LLaVA-style models have specific `[INST]` tags and `<image>` token placement that must match training. Getting this wrong silently degrades quality without raising an error.

### 4.2 GPT-4V and GPT-4o

OpenAI's GPT-4V ([OpenAI, 2023](https://openai.com/research/gpt-4v-system-card)) and GPT-4o are closed-source unified models that process images natively. GPT-4o processes both images and text in the same token sequence. Key capability gap vs open-source: GPT-4o excels at reading text embedded in images (charts, receipts, handwriting), a task that requires high-resolution visual processing where most 7B open-source models still struggle.

**MMMU accuracy (2024):** GPT-4o ~69%, GPT-4V ~56%, LLaVA-1.5-13B ~36%.

### 4.3 Google Gemini

Gemini ([Google DeepMind, 2023](https://arxiv.org/abs/2312.11805)) is a natively multimodal model trained from the start on interleaved text, images, audio, and video. Gemini 1.5 Pro ([Google Team, 2024](https://arxiv.org/abs/2403.05530)) extended this with a million-token context window using a multimodal mixture-of-experts architecture. Key capability: processing an entire 45-minute film (684K tokens at 1fps) and answering specific questions about scenes that appear only once — near-perfect recall (>99.7%) on long-context retrieval tasks.

> 🎯 **Interview prep**: "What's the difference between CLIP-based VLMs and native multimodal models like GPT-4o/Gemini?" — CLIP-based models use a frozen or separately-trained vision encoder connected to an LLM via a bridge; the two modalities never fully co-train. Native multimodal models train on interleaved visual and text tokens from scratch, learning tighter cross-modal dependencies. The trade-off: native models are far more capable but require enormous compute to train.

**Resources**
- [LLaVA-1.5 paper](https://arxiv.org/abs/2310.03744) — systematic study of design choices (encoder size, projection type, data mix)
- [Gemini 1.5 paper](https://arxiv.org/abs/2403.05530) — long-context multimodal architecture details
- [GPT-4V system card](https://openai.com/research/gpt-4v-system-card) — capability overview and safety evaluation

---

## 5. Audio-Language Models

Audio is the most neglected modality in most ML curricula — but voice is the most natural human interface. Audio-language models connect speech recognition, speech synthesis, and language understanding into pipelines (or unified models) capable of real-time voice conversation, translation, and audio captioning.

### 5.1 Whisper: Robust Speech Recognition at Scale

**Whisper** ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) from OpenAI demonstrated that a simple encoder-decoder transformer trained on 680,000 hours of weakly supervised internet audio transcripts achieves human-level robustness on speech recognition — in a zero-shot setting, without per-domain fine-tuning. The key architectural choice: treat all speech tasks (transcription, translation, language identification) as a text prediction problem with special prefix tokens.

```
Audio (mel spectrogram) ──► [Encoder] ──► audio_repr
                                              │
Task prefix tokens ──────────────────► [Decoder] ──► output tokens
```

Special prefix tokens: `<|transcribe|>`, `<|translate|>`, `<|en|>` (language), `<|nospeech|>`. The same model handles all tasks by varying the prefix.

```python
# Whisper transcription via HuggingFace
# pip install transformers torch datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch

model_id = "openai/whisper-large-v3"  # best quality; use "whisper-base" for speed
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load sample audio from HF datasets
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
sample = ds[0]["audio"]  # dict with {"array": ..., "sampling_rate": ...}

inputs = processor(
    sample["array"],
    sampling_rate=sample["sampling_rate"],
    return_tensors="pt"
).input_features.to(model.device)

predicted_ids = model.generate(inputs, language="en", task="transcribe")  # force English transcription
print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
```

> 🏭 **Production note**: Whisper's real-time factor (RTF) on large-v3 is ~0.3x on an A10G GPU (3 seconds of audio takes 1 second to transcribe). For latency-sensitive applications, use `whisper-medium` (RTF ~0.1x) or the Faster-Whisper CTranslate2 backend which is 4x faster with the same accuracy.

### 5.2 AudioPaLM: Speaking and Listening as One Model

**AudioPaLM** ([Rubenstein et al., 2023](https://arxiv.org/abs/2306.12925)) combined a speech tokenizer (AudioLM's SoundStream codec) with PaLM's language model to create a unified model that processes and generates both text and speech tokens in the same vocabulary. AudioPaLM achieves significantly better translation quality for speech-to-speech tasks than cascaded systems (ASR → MT → TTS) because errors don't accumulate across stages. The model also preserves speaker identity and prosody in translation — something impossible in text-only MT systems.

### 5.3 SeamlessM4T: Massively Multilingual Multimodal Translation

**SeamlessM4T** ([Meta AI, 2023](https://arxiv.org/abs/2308.11596)) unified five translation tasks (S2ST, S2TT, T2ST, T2TT, ASR) into a single model supporting up to 100 languages. The key innovation: using 1 million hours of open speech audio processed through w2v-BERT 2.0 for self-supervised representation learning, combined with multimodal parallel corpora. The result was a 20% BLEU improvement over the previous SotA in direct speech-to-text translation, while the end-to-end model preserves prosody that cascaded systems lose.

> 🎯 **Interview prep**: "Why are end-to-end speech translation models better than cascaded ASR+MT?" — Three reasons: (1) no error propagation between stages, (2) prosody and speaker information are preserved (lost once you go to text), (3) lower latency since you avoid two separate model calls. The downside: harder to debug because you can't inspect an intermediate transcript.

**Resources**
- [Whisper paper](https://arxiv.org/abs/2212.04356) — weak supervision at scale for robust ASR
- [AudioPaLM paper](https://arxiv.org/abs/2306.12925) — unified speech and language token vocabulary
- [SeamlessM4T paper](https://arxiv.org/abs/2308.11596) — massively multilingual speech translation

---

## 6. Video Understanding

Video is images plus time. The challenge is not just processing many frames — it's understanding *change*: motion, causality, temporal ordering, narrative. A model that understands individual frames perfectly can still fail to answer "what happened after the man entered the building?" if it can't reason across a 2-minute clip.

### 6.1 The Temporal Challenge

Naive approaches (average frame embeddings, process every 10th frame) miss temporal dependencies. The field developed two main strategies:

1. **Sparse sampling with temporal positional encoding:** Sample frames uniformly or based on scene boundaries, encode each with a vision encoder, and add learned temporal position embeddings before feeding to an LLM. Fast but misses fine-grained motion.

2. **Video-specific temporal models:** Inflate 2D convolutions to 3D (C3D, SlowFast) or apply temporal attention across frames (TimeSFormer). Better for action recognition but expensive to scale to long videos.

### 6.2 Flamingo for Video

Flamingo naturally handles video by treating a video as an ordered sequence of frames — each compressed via the Perceiver Resampler into a fixed number of tokens, then interleaved with text tokens in the LLM. This was among the first demonstration that a single model could handle both images and video without task-specific architectures.

### 6.3 Gemini 1.5 Pro: Long-Context Video at Scale

**Gemini 1.5 Pro** ([Google Team, 2024](https://arxiv.org/abs/2403.05530)) extended video understanding to an entirely different scale. With a context window supporting up to 10 million tokens and a multimodal MoE architecture, Gemini 1.5 Pro can process hours of video at 1fps (a 45-minute film at 1fps ≈ 684K tokens), demonstrate near-perfect recall (>99.7%) on long-context retrieval tasks, and answer questions about events that happen only once across the full video. This is qualitatively different from models that process a few dozen frames — it's closer to watching the whole video the way a human would.

```python
# Video understanding with Gemini via the google-generativeai SDK
# pip install google-generativeai
import google.generativeai as genai
import time

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")  # or gemini-1.5-flash for lower cost

# Upload a video file (required for videos >20MB)
video_file = genai.upload_file(path="my_video.mp4", mime_type="video/mp4")
while video_file.state.name == "PROCESSING":  # wait for processing
    time.sleep(5)
    video_file = genai.get_file(video_file.name)

response = model.generate_content([
    video_file,
    "Describe the key events in this video in chronological order. "
    "Focus on any actions or transitions between scenes."
])
print(response.text)
```

> 🏭 **Production note**: For most production video QA tasks, processing 1fps is sufficient and dramatically reduces token cost. For sports analysis or action recognition requiring frame-level precision, 4–8fps may be needed. Use scene detection (e.g., PySceneDetect) to extract keyframes rather than uniform sampling — you get better coverage per token.

**Resources**
- [Gemini 1.5 technical report](https://arxiv.org/abs/2403.05530) — long-context multimodal MoE architecture
- [Flamingo paper](https://arxiv.org/abs/2204.14198) — original video+image few-shot learning

---

## 7. Text-to-Image Generation

Text-to-image models are generative multimodal systems that map natural language descriptions to photorealistic images. They represent the inverse problem from VQA: rather than understanding an image, they create one conditioned on text.

### 7.1 Diffusion Models: The Dominant Paradigm

Modern text-to-image generation is dominated by **latent diffusion models** (LDMs). The key insight: run the diffusion process in a compressed latent space (encoded by a VAE) rather than pixel space. This reduces compute by 64x while preserving perceptual quality.

```
Text ──► [Text Encoder (T5 or CLIP)] ──► text_embedding
                                              │
                                    cross-attention conditioning
                                              │
Noisy latent z_T ──► [UNet or DiT] ──► ... ──► z_0 ──► [VAE Decoder] ──► Image
```

At each denoising step, the UNet (or DiT — Diffusion Transformer) takes the current noisy latent and the text embedding as cross-attention conditioning to predict the noise. After T steps, the clean latent z_0 is decoded by the VAE into a pixel-space image.

**Classifier-free guidance (CFG):** At inference, the model evaluates two predictions — one conditioned on the text prompt and one unconditioned — then amplifies the difference:

```
z_guided = z_uncond + guidance_scale × (z_cond - z_uncond)

Where:
  guidance_scale  = 7.5 is typical (higher → more prompt adherence, less diversity)
  z_cond          = model prediction given text prompt
  z_uncond        = model prediction given empty text
```

**Worked example:**

```
guidance_scale = 7.5
z_uncond       = [0.1, -0.2, 0.3]  (arbitrary latent prediction)
z_cond         = [0.4, 0.1, 0.5]   (conditioned on "a red rose")

z_guided = [0.1, -0.2, 0.3] + 7.5 × ([0.4, 0.1, 0.5] - [0.1, -0.2, 0.3])
         = [0.1, -0.2, 0.3] + 7.5 × [0.3, 0.3, 0.2]
         = [0.1, -0.2, 0.3] + [2.25, 2.25, 1.5]
         = [2.35, 2.05, 1.8]

Result: latent strongly pushed toward the "red rose" direction
```

### 7.2 SDXL: Open-Source SotA

**Stable Diffusion XL (SDXL)** ([Podell et al., 2023](https://arxiv.org/abs/2307.01952)) improved latent diffusion through: (1) a 3x larger UNet backbone, (2) dual text encoder (OpenCLIP-ViT/G + CLIP-ViT/L) providing richer conditioning, (3) multi-aspect-ratio training, and (4) a refinement model for post-hoc enhancement. SDXL is competitive with closed proprietary models and runs on a single A100 or consumer GPUs with 8-bit quantization.

### 7.3 Imagen: Language Encoder is the Key

**Imagen** ([Saharia et al., 2022](https://arxiv.org/abs/2205.11487)) from Google made a counterintuitive finding: scaling the text encoder (T5-XXL) improves image quality *more* than scaling the diffusion model itself. Large language models pretrained purely on text turn out to be excellent image conditioning signals. Imagen achieved FID 7.27 on MS-COCO zero-shot — better than human-rater scores at the time — without ever training on COCO.

### 7.4 DALL-E 3: Caption Quality Over Scale

**DALL-E 3** ([Betker et al., 2023](https://cdn.openai.com/papers/dall-e-3.pdf)) from OpenAI took a different angle: the primary improvement was *recaptioning* the training data with a detailed image captioner (a VLM), replacing the noisy alt-text web captions with precise, detailed descriptions. This dramatically improved prompt adherence — the model's ability to actually draw what you ask for, including specific counts, spatial relationships, and unusual compositions. The lesson: for generative models, caption quality matters more than image quantity.

```python
# SDXL inference via diffusers
# pip install diffusers transformers accelerate torch
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # real HF model ID
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A photorealistic portrait of a red fox sitting in autumn leaves, golden hour lighting, shallow depth of field",
    negative_prompt="blurry, low quality, cartoon, illustration",  # push away from unwanted styles
    num_inference_steps=30,  # 20-40 is typical quality/speed tradeoff
    guidance_scale=7.5,       # 6-8 for photorealism
    width=1024,
    height=1024
).images[0]

image.save("fox_portrait.png")
```

> 🎯 **Interview prep**: "What is classifier-free guidance and why does it matter?" — CFG lets you control the trade-off between prompt adherence and image diversity at *inference* time without retraining. High guidance scale makes the model strictly follow the prompt but reduces diversity; low scale allows creativity but may drift from the prompt. The model is trained with randomly dropped text conditioning so it learns both conditioned and unconditioned generation simultaneously.

> 🏭 **Production note**: SDXL requires ~10GB VRAM for base inference. For production at lower cost, SDXL-Turbo (adversarial distillation) produces quality images in 1-4 steps instead of 30, giving a 10x latency reduction. FLUX.1 (Black Forest Labs, 2024) has largely superseded SDXL as the open-source default for quality-critical applications.

**Resources**
- [SDXL paper](https://arxiv.org/abs/2307.01952) — architecture improvements and multi-aspect training
- [Imagen paper](https://arxiv.org/abs/2205.11487) — text encoder scaling insight
- [DALL-E 3 paper](https://cdn.openai.com/papers/dall-e-3.pdf) — synthetic recaptioning technique

---

## 8. Multimodal Embeddings and Cross-Modal Retrieval

Cross-modal retrieval is the task of finding items in one modality given a query in another — "find images that match this text description" or "find the most similar caption to this image." It is the backbone of visual search engines, product discovery systems, and multimodal RAG.

### 8.1 The CLIP Embedding Space

CLIP's contrastive training produces a shared vector space where semantically similar concepts from different modalities cluster together. An image of a dog and the text "a golden retriever" will have similar embeddings; an image of a cat will be farther away. This makes CLIP the default backbone for cross-modal retrieval.

```python
# Cross-modal image-text retrieval with CLIP + FAISS
# pip install transformers torch faiss-gpu pillow requests
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
import numpy as np
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # real HF model ID
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_images(image_list):
    """Embed a list of PIL images into CLIP's shared space."""
    inputs = processor(images=image_list, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)  # L2 normalize

def embed_texts(text_list):
    """Embed a list of strings into CLIP's shared space."""
    inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)  # L2 normalize

# Build index over image embeddings
image_embeddings = embed_images([img1, img2, img3]).numpy()  # shape: [N, 768]
index = faiss.IndexFlatIP(image_embeddings.shape[1])  # inner product = cosine sim (normalized)
index.add(image_embeddings)

# Query with text
query_embedding = embed_texts(["a golden retriever playing in snow"]).numpy()
distances, indices = index.search(query_embedding, k=5)  # top-5 nearest images
print(f"Top-5 matching image indices: {indices[0]}")
```

### 8.2 Beyond CLIP: Jina CLIP and SigLIP

CLIP has two weaknesses in production: (1) it was trained on English-dominant data and struggles on multilingual queries; (2) its text encoder has a 77-token limit (roughly 60 words), making it unsuitable for detailed, long-form captions.

**Jina CLIP v2** ([Jina AI, 2024](https://huggingface.co/jinaai/jina-clip-v2)) extends CLIP with multilingual support (89 languages) and a 512-token text limit via a JinaBERT text encoder. It uses multi-task contrastive training to handle both text-image and text-text retrieval tasks in a single model — crucial for multimodal RAG where you may query across mixed document types.

**SigLIP** ([Zhai et al., 2023](https://arxiv.org/abs/2303.15343)) replaces the softmax contrastive loss with a sigmoid loss that treats each image-text pair independently rather than normalizing across the batch. This makes training less sensitive to batch size and improves performance on fine-grained retrieval.

### 8.3 Multimodal RAG

In multimodal RAG, documents may contain both text and images (PDFs, presentations, product catalogs). The retrieval stage needs to match a text query against both text chunks and image chunks in the same vector index.

```
Strategy A (text-only): OCR images → embed all as text → standard RAG
  Pro: simple. Con: loses visual information (charts, diagrams)

Strategy B (separate indexes): embed text chunks and images separately → merge results
  Pro: modality-appropriate embeddings. Con: complex ranking fusion

Strategy C (unified CLIP index): embed everything with CLIP → single index
  Pro: simple cross-modal matching. Con: CLIP's 77-token limit loses long document context
```

> 🏭 **Production note**: The most reliable production pattern for PDF/document RAG is Strategy A + B hybrid: use LLaVA to generate rich captions for all figures/charts (run offline), embed the captions as text, and keep the original image stored for retrieval. This way your index is all text (cheap, fast) but retrieval is semantically rich because the captions were generated by a multimodal model.

**Resources**
- [CLIP](https://arxiv.org/abs/2103.00020) — foundational shared embedding space
- [HuggingFace CLIP + FAISS cookbook](https://huggingface.co/learn/cookbook/en/faiss_with_hf_datasets_and_clip) — practical multimodal retrieval
- [`jinaai/jina-clip-v2`](https://huggingface.co/jinaai/jina-clip-v2) — multilingual, long-text CLIP variant

---

## 9. Training Data and Alignment Techniques

The quality of a multimodal model is determined more by its training data than by its architecture. This is the section most practitioners overlook, and it's where the real competitive moats are built.

### 9.1 Dataset Progression

The field progressed by trading annotation quality for scale:

| Dataset | Size | Source | Quality | Key Feature |
|---|---|---|---|---|
| MS-COCO Captions | 1.5M | Human annotated | High | 5 captions per image |
| Conceptual Captions 3M (CC3M) | 3M | Web alt-text, filtered | Medium | First large auto-labeled set |
| CC12M | 12M | Relaxed CC3M filter | Medium-low | Scale over quality |
| WIT (Wikipedia Image-Text) | 5.5M | Wikipedia | High | Encyclopedic breadth |
| LAION-400M | 400M | CLIP-filtered web | Low | Scale; used to train open CLIP |
| LAION-5B ([Schuhmann et al., 2022](https://arxiv.org/abs/2210.08402)) | 5.85B | CLIP-filtered web | Low-medium | Largest public image-text corpus |
| OBELICS ([Laurençon et al., 2023](https://arxiv.org/abs/2306.16527)) | 115B tokens | Interleaved web docs | Medium | Native interleaved text+image |

**LAION-5B** consists of 5.85 billion CLIP-filtered image-text pairs — 20x larger than any previous public dataset. It used CLIP cosine similarity ≥ 0.28 to filter the noisiest pairs, still yielding only ~10% of the original scraped data. Models trained on LAION-400M (a subset) matched OpenAI's original CLIP.

> 🏭 **Production note**: LAION-5B was taken offline in late 2023 after a security research disclosure about CSAM content in the dataset. The current recommended large-scale training set is DataComp ([Gadre et al., 2024](https://arxiv.org/abs/2304.14108)) — a competition-format dataset designed with better filtering, available via the DataComp benchmark.

### 9.2 Instruction-Tuning Data

For conversational VLMs, instruction-following data is as important as pre-training data. The key insight from LLaVA was that GPT-4 could generate high-quality multimodal instruction-following examples from image captions alone (no human labeling required):

1. Feed GPT-4 the image caption + bounding box annotations as text
2. Ask GPT-4 to generate instruction-response pairs (conversations, detailed descriptions, complex reasoning)
3. Use these pairs to fine-tune a smaller VLM

ShareGPT4V expanded this to 100K GPT-4V-generated captions directly from images, giving richer visual grounding than caption-only generation.

### 9.3 Alignment via RLHF and DPO

Raw instruction-tuned VLMs hallucinate confidently — they fabricate objects, quantities, and relationships that aren't in the image. Two alignment techniques reduce this:

**RLHF for VLMs:** Train a reward model on human preferences over image-response pairs, then optimize the VLM with PPO. Effective but expensive.

**Hallucination-specific DPO:** Collect pairs where one response correctly describes the image and another hallucinates an element. Use DPO (Direct Preference Optimization) to push the model toward accurate responses. LLaVA-RLHF and RLHF-V use this approach, reducing hallucination rates by 20-30% over supervised fine-tuning alone.

> 🎯 **Interview prep**: "How do you reduce hallucination in VLMs?" — Three levels: (1) data — use higher-quality captions and include negative examples during training; (2) architecture — high-resolution visual encoders that preserve fine-grained detail; (3) alignment — DPO on hallucination-specific preference pairs. In production, also add a post-generation grounding step: ask the model to point to the evidence for each claim in the image.

**Resources**
- [LAION-5B paper](https://arxiv.org/abs/2210.08402) — dataset construction and CLIP-filtering methodology
- [DataComp](https://arxiv.org/abs/2304.14108) — principled data filtering competition
- [RLHF-V paper](https://arxiv.org/abs/2312.00849) — hallucination reduction via preference learning

---

## 10. Evaluation Benchmarks

Evaluating multimodal models is harder than evaluating text-only LLMs because tasks span perception, reasoning, knowledge, and generation. The community has converged on a set of canonical benchmarks, each measuring different capabilities.

### 10.1 MMMU — College-Level Expert Reasoning

**MMMU** ([Yue et al., 2024](https://arxiv.org/abs/2311.16502)) — Massive Multi-discipline Multimodal Understanding — tests expert-level knowledge across 6 disciplines, 30 subjects, and 183 subfields. 11,500 questions drawn from college exams and textbooks, covering 30 heterogeneous image types (diagrams, chemical structures, music sheets, medical images, charts).

**Why it matters:** MMMU cannot be answered by pattern-matching visual surface features — it requires integrating discipline-specific knowledge with visual understanding. A question might show a circuit diagram and ask you to calculate the output voltage.

**Scores (zero-shot, 2024):**

| Model | MMMU Accuracy |
|---|---|
| GPT-4o | ~69% |
| Gemini 1.5 Pro | ~62% |
| InternVL2.5-78B | ~70% |
| LLaVA-1.5-13B | ~36% |
| Human expert | ~88% |

```python
# Running MMMU evaluation locally
# pip install lmms-eval
from lmms_eval import evaluator
from lmms_eval.models import get_model

model = get_model("llava", pretrained="llava-hf/llava-v1.6-mistral-7b-hf")
results = evaluator.simple_evaluate(
    model=model,
    tasks=["mmmu_val"],   # MMMU validation set
    batch_size=1,
    log_samples=True
)
print(results["results"]["mmmu_val"])
```

### 10.2 MMBench — Systematic Ability Coverage

**MMBench** ([Liu et al., 2023](https://arxiv.org/abs/2307.06281)) evaluates VLMs across 20 multimodal ability dimensions organized in a hierarchical taxonomy — from low-level perception (object recognition, attribute detection) to high-level cognition (commonsense reasoning, relation understanding). The bilingual design (English + Chinese) makes it the standard benchmark for evaluating multilingual VLMs.

Key innovation: MMBench uses **CircularEval** — rotating the answer choices across multiple-choice questions — to prevent models from gaming the benchmark by always picking "A".

### 10.3 SEED-Bench — Generative Comprehension

**SEED-Bench** ([Li et al., 2023](https://arxiv.org/abs/2307.16125)) evaluates generative comprehension with 19K multiple-choice questions across 12 dimensions including spatial understanding, visual reasoning, and temporal understanding (for video). The SEED-Bench-2 variant extended this to interleaved image-text inputs.

### 10.4 VQAv2 and TextVQA

**VQAv2** ([Goyal et al., 2017](https://arxiv.org/abs/1612.00837)) is the oldest major visual QA benchmark — 265K images with 1.1M natural-language questions and ground-truth answers. Despite being superseded by MMMU for frontier model evaluation, it remains the canonical benchmark for basic VQA and is widely reported.

**TextVQA** ([Singh et al., 2019](https://arxiv.org/abs/1904.08920)) specifically tests whether models can read and reason about text embedded in images (signs, menus, product labels). This is where CLIP-based models with 224px encoders traditionally fail — too low resolution to read small text.

> 🎯 **Interview prep**: "If you had to pick one benchmark to evaluate a VLM for enterprise document understanding, which would you choose?" — TextVQA or DocVQA, since document understanding requires reading embedded text, reasoning about layout, and extracting structured information — all different from natural image understanding tested by VQAv2 or MMBench.

**Resources**
- [MMMU paper](https://arxiv.org/abs/2311.16502) — benchmark construction methodology
- [MMBench paper](https://arxiv.org/abs/2307.06281) — CircularEval methodology and taxonomy
- [SEED-Bench GitHub](https://github.com/AILab-CVC/SEED-Bench) — code and leaderboard

---

## 11. Model Comparison: Open-Source and Closed

### 11.1 Vision-Language Model Landscape (2024–2025)

*Popularity: 🔥 >1M/mo · ⭐ 100K–1M · 📈 10K–100K · 🆕 <10K (HuggingFace monthly downloads, Jun 2025)*

**Open-source models:**

| Model | HF Link | Size | Provider | Released | Best Use Cases | Key Innovation | MMMU | Pop. |
|---|---|---|---|---|---|---|---|---|
| `LLaVA-1.5-13B` | [🤗](https://huggingface.co/llava-hf/llava-1.5-13b-hf) | 13B | UW-Madison | Oct 2023 | General VQA, baseline research | MLP projection + high-res CLIP | 36% | 🔥 |
| `LLaVA-NeXT-7B` | [🤗](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | 7B | UW-Madison | Jan 2024 | Document understanding, OCR | Dynamic high-resolution tiling | 35% | ⭐ |
| `PaliGemma-3B` | [🤗](https://huggingface.co/google/paligemma-3b-pt-224) | 3B | Google | May 2024 | Efficient edge deployment | SigLIP + Gemma-2B, compact | 34% | 📈 |
| `Qwen2-VL-7B` | [🤗](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | 7B | Alibaba | Sep 2024 | Charts, documents, multilingual | Naive dynamic resolution | 54% | ⭐ |
| `InternVL2.5-8B` | [🤗](https://huggingface.co/OpenGVLab/InternVL2_5-8B) | 8B | OpenGVLab | Dec 2024 | Balanced quality/efficiency | InternViT-300M encoder | 56% | 📈 |
| `InternVL2.5-78B` | [🤗](https://huggingface.co/OpenGVLab/InternVL2_5-78B) | 78B | OpenGVLab | Dec 2024 | Max open-source quality | First open model >70% MMMU | 70%+ | 🆕 |

> Models with MMMU >60% require InternViT-6B or equivalent high-capacity visual encoder — encoder size matters as much as LLM size for multimodal tasks.

**Closed-source models:**

| Model | API Docs | Provider | Released | Best Use Cases | Multimodal Scope | Pricing |
|---|---|---|---|---|---|---|
| GPT-4o | [OpenAI](https://platform.openai.com/docs) | OpenAI | May 2024 | Complex visual reasoning, OCR | Image + text | $5/1M in, $15/1M out |
| GPT-4o-mini | [OpenAI](https://platform.openai.com/docs) | OpenAI | Jul 2024 | Cost-sensitive image tasks | Image + text | $0.15/1M in |
| Gemini 1.5 Pro | [Google](https://ai.google.dev/docs) | Google | Feb 2024 | Long video, million-token context | Image+video+audio+text | $3.5/1M in |
| Gemini 1.5 Flash | [Google](https://ai.google.dev/docs) | Google | May 2024 | High-throughput multimodal | Image+video+audio+text | $0.075/1M in |
| Claude 3.5 Sonnet | [Anthropic](https://docs.anthropic.com) | Anthropic | Oct 2024 | Document analysis, charts | Image + text | $3/1M in |

### 11.2 Quality vs. Size / Cost

```
MMMU Score
  ▲
70%│                           ● InternVL2.5-78B
   │                     ● Gemini 1.5 Pro
   │               ● GPT-4o
60%│         ● Qwen2-VL-72B
   │     ● InternVL2.5-8B  ● Qwen2-VL-7B
50%│
   │    ● LLaVA-1.5-13B  ● PaliGemma-3B
35%│
   └──────────────────────────────────────► Parameters / Cost
     Small/Fast                  Large/Slow
```

### 11.3 Key Trends (2024–2025)

1. **High-resolution visual encoding is the bottleneck**: Models that tile images into multiple high-resolution crops (LLaVA-NeXT, Qwen2-VL's naive dynamic resolution) dramatically outperform fixed-resolution models on document and chart tasks. The visual encoder quality matters as much as the LLM.

2. **Open-source is closing the gap**: InternVL2.5-78B (70%+ on MMMU) matches closed proprietary models. The gap that remains is primarily on tasks requiring massive pretraining compute (long video, audio, code generation from screenshots).

3. **Efficiency specialization**: The 3–8B range (PaliGemma, Qwen2-VL-7B) is now good enough for most production VQA tasks and runs on a single A10G. Full 70B models are only needed for expert-level reasoning or document understanding.

---

## 12. Inference and Serving Multimodal Models

Deploying multimodal models in production introduces challenges beyond those of text-only LLMs: image preprocessing overhead, variable input sizes, higher memory requirements, and the need to efficiently batch mixed image+text requests.

### 12.1 Latency Breakdown

For a typical LLaVA-style VLM (7B, fp16, serving via vLLM):

```
Total request latency ≈ 230ms  (example: 1 image + 100 token prompt, 200 token output)

  Image preprocessing (resize, normalize)  :   5ms   (~2%)
  Vision encoder (CLIP ViT-L forward pass) :  25ms   (~11%)
  MLP projection                           :   2ms    (~1%)
  LLM prefill (image + text tokens)        :  48ms   (~21%)
  LLM decode (200 tokens @ ~50 tok/s)      : 150ms   (~65%)
```

The bottleneck is LLM decode — same as text-only LLMs. Image processing is cheap. This means optimizations for LLM inference (batching, speculative decoding, tensor parallelism) apply directly.

### 12.2 Serving with vLLM

vLLM v0.4+ natively supports LLaVA-compatible models with `--image-input-type` flags. PagedAttention still applies to the text KV cache; image features are cached separately.

```python
# vLLM multimodal inference
# pip install vllm
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

image = Image.open("chart.png")

outputs = llm.generate(
    {
        "prompt": "[INST] <image>\nWhat is the trend shown in this chart? [/INST]",
        "multi_modal_data": {"image": image}
    },
    SamplingParams(temperature=0.0, max_tokens=300)  # temperature=0 for factual tasks
)
print(outputs[0].outputs[0].text)
```

### 12.3 Quantization for Memory Reduction

A 7B VLM in fp16 requires ~14GB VRAM. Quantization paths:

| Method | VRAM (7B) | Quality Loss | When to Use |
|---|---|---|---|
| fp16 (default) | 14GB | None | A100/H100 serving |
| 4-bit NF4 (bitsandbytes) | 4.5GB | ~2% on benchmarks | Single A10G/3090 |
| AWQ 4-bit | 4GB | ~1% — best quantization | Production serving |
| GGUF Q4_K_M (llama.cpp) | 4.5GB | ~2% | CPU or edge inference |

```python
# 4-bit quantized LLaVA with bitsandbytes
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",       # NormalFloat4 — better than fp4 for LLMs
    bnb_4bit_use_double_quant=True   # nested quantization for extra memory savings
)

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
# ~4.5GB VRAM vs ~14GB for fp16
```

### 12.4 Production Failure Modes

1. **Token budget blowouts**: Tiling a high-resolution image (e.g., LLaVA-NeXT 4-tile) can consume 2,304 visual tokens per image. At 4 images per request, 9,216 tokens are visual before a word of text. Always cap `max_image_tiles` and log token counts.

2. **Chat template mismatches**: Every model has a different format for the `<image>` placeholder, `[INST]` tags, and system prompts. One missed tag → silent quality regression. Always test against the model card's exact format.

3. **VRAM spikes on variable-resolution inputs**: Dynamic tiling means image token count varies per request. A single 4K image can OOM a server that handles 720p images fine. Implement hard resolution caps (max 1024x1024 per tile) in preprocessing.

> 🏭 **Production note**: For cost-sensitive production serving (high QPS visual classification or captioning), consider a two-stage pipeline: first run CLIP to filter or route requests (cheap, ~5ms), then only invoke the heavy VLM for requests that need it. This cuts VLM calls by 40-70% in typical classification-heavy workloads.

**Resources**
- [vLLM multimodal documentation](https://docs.vllm.ai/en/latest/models/vlm.html) — serving LLaVA and other VLMs
- [LMDeploy](https://github.com/InternLM/lmdeploy) — optimized serving for InternVL and Qwen-VL

---

## 13. The Modern Recipe

What to do today to build and deploy a production-quality multimodal system.

### For Visual Question Answering / Document Understanding

1. **Start with the right model for your compute**: 
   - < 8GB VRAM → `Qwen2-VL-7B-Instruct` (best quality/size ratio at 7B, 54% MMMU)
   - < 24GB VRAM → `InternVL2.5-8B` (56% MMMU, strong OCR)
   - < 80GB VRAM → `InternVL2.5-78B` (70%+ MMMU, matches GPT-4o)
   - No GPU → GPT-4o-mini API ($0.15/1M in) or Gemini 1.5 Flash ($0.075/1M in)

2. **Use high-resolution input**: Always use the 448px or higher resolution variant. LLaVA-NeXT's dynamic tiling gives 1.5-2x improvement on OCR and chart tasks.

3. **Evaluate on MMMU, TextVQA, and ChartQA** before shipping. Domain-specific evals (medical, legal, financial) always outperform generic benchmark ranks.

4. **Reduce hallucination**: Apply the "grounding prompt" at inference: "Describe only what you can directly see in the image. If you're uncertain, say so." Then post-process with a CLIP score to verify response-image alignment.

5. **Serve with vLLM** for throughput (3-5x vs HuggingFace naive inference). Use AWQ 4-bit quantization for memory efficiency with minimal quality loss.

6. **Implement request-level token budgets**: Cap image resolution at 1024x1024, limit to 4 tiles per image. Log token counts per request. Set alarm at p99 > 4K tokens.

### For Cross-Modal Retrieval

1. Use `openai/clip-vit-large-patch14` (best open CLIP, 768-dim) or `jinaai/jina-clip-v2` for multilingual/long-text.
2. Index with FAISS `IndexFlatIP` for exact search (<1M items) or `IndexIVFPQ` for approximate search (>1M items).
3. Always L2-normalize embeddings before indexing (cosine similarity via inner product).
4. For PDF/document RAG: caption images with a VLM offline, embed captions as text, store original image linked to caption embedding.

### For Text-to-Image Generation

1. Use SDXL-Base + SDXL-Refiner pipeline for quality. Use SDXL-Turbo or FLUX.1-schnell for speed.
2. Guidance scale: 7–8 for photorealism, 4–6 for creative/artistic styles.
3. Write negative prompts: always include `"blurry, low quality, watermark, text"` as a baseline.
4. For production, serve with diffusers + Gradio or ComfyUI backend. Cache the VAE separately — it's 1.5GB and never changes.

> 🎯 **Interview prep**: "Design a multimodal search system for an e-commerce platform." This recipe is your answer: CLIP for cross-modal retrieval, a VLM for generating rich product descriptions from product images, FAISS for the index, a VQA endpoint for "find me shoes that look like this" queries. The senior-engineer nuance: handle the cold start by pre-computing CLIP embeddings for all catalog images offline, and implement an online path that generates embeddings for user-uploaded query images in real time.

---

## 14. Interview Q&A

**Q: What is the difference between CLIP and LLaVA?**  
CLIP is an embedding model — it maps images and text into a shared vector space for retrieval and classification but produces no text output. LLaVA is a generative VLM — it processes an image and a text prompt and generates a text response. LLaVA uses CLIP's vision encoder internally to produce image features, then feeds them into an LLM through a projection layer.

**Q: How does classifier-free guidance work in diffusion models?**  
During training, the model randomly drops the text conditioning with some probability p (typically 10%), learning both conditioned and unconditioned generation. At inference, you run the model twice — once with the text prompt and once with an empty prompt — then interpolate: `z_guided = z_uncond + scale × (z_cond - z_uncond)`. Higher scale amplifies the "direction" from unconditioned to conditioned, making the image more faithfully match the prompt at the cost of diversity.

**Q: What is the Q-Former in BLIP-2 and why does it exist?**  
The Q-Former is a lightweight transformer with N learned query tokens (typically 32) that cross-attend to the frozen image encoder's output. Its purpose: compress a variable-length sequence of thousands of image patch tokens into a fixed-size representation (32 tokens) that a frozen LLM can consume as a soft prompt. This allows connecting any image encoder to any LLM without training either, making it extremely parameter-efficient.

**Q: How would you handle hallucination in a VLM deployed in production?**  
Four complementary strategies: (1) alignment data — fine-tune with DPO on preference pairs where one response hallucinates; (2) prompt engineering — "describe only what you can directly see"; (3) post-hoc grounding — use CLIP to verify that mentioned objects actually appear in the image; (4) ensemble/sampling — sample 3 responses and take the one with highest cross-response consistency. The most practical production approach is (2) + (3).

**Q: What makes Gemini 1.5 Pro different from GPT-4V for video tasks?**  
Context length. GPT-4V can process a handful of frames. Gemini 1.5 Pro's 1M+ token context window allows processing hours of video at 1fps — an entire film — maintaining near-perfect recall (>99.7%) across the full context. This is qualitatively different: GPT-4V sees a movie trailer; Gemini 1.5 Pro watches the whole film.

**Q: What is early fusion vs late fusion in multimodal models?**  
Early fusion: interleave raw image tokens and text tokens before any processing, letting a shared transformer learn cross-modal interactions from the start (GPT-4o, Gemini). Late fusion: process each modality with separate specialized encoders, then combine representations at a higher level (original CLIP-based VLMs). Early fusion enables tighter cross-modal dependencies but requires massive compute to train from scratch.

**Q: Why does DALL-E 3 produce better prompt adherence than SDXL?**  
DALL-E 3 was trained on data where images were recaptioned by a powerful VLM (a GPT-4-style captioner) to produce detailed, accurate descriptions — replacing noisy web alt-text. The model learned to generate images matching detailed captions, so when you provide a detailed prompt, it knows exactly what to produce. SDXL was trained on noisier alt-text, so it's better at style matching than precise compositional adherence.

---

## 15. References

### Foundational Vision-Language

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* https://arxiv.org/abs/2103.00020
- Jia et al. (2021). *Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN).* https://arxiv.org/abs/2102.05918
- Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning.* https://arxiv.org/abs/2204.14198
- Li et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training.* https://arxiv.org/abs/2201.12086
- Li et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* https://arxiv.org/abs/2301.12597
- Liu et al. (2023). *Visual Instruction Tuning (LLaVA).* https://arxiv.org/abs/2304.08485
- Liu et al. (2023). *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5).* https://arxiv.org/abs/2310.03744

### Closed Frontier Models

- Google Team (2024). *Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context.* https://arxiv.org/abs/2403.05530
- Google Team (2023). *Gemini: A Family of Highly Capable Multimodal Models.* https://arxiv.org/abs/2312.11805
- OpenAI (2023). *GPT-4V(ision) System Card.* https://openai.com/research/gpt-4v-system-card
- Betker et al. (2023). *Improving image generation with better captions (DALL-E 3).* https://cdn.openai.com/papers/dall-e-3.pdf

### Audio and Speech

- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).* https://arxiv.org/abs/2212.04356
- Rubenstein et al. (2023). *AudioPaLM: A Large Language Model That Can Speak and Listen.* https://arxiv.org/abs/2306.12925
- Meta AI (2023). *SeamlessM4T: Massively Multilingual & Multimodal Machine Translation.* https://arxiv.org/abs/2308.11596

### Text-to-Image Generation

- Saharia et al. (2022). *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen).* https://arxiv.org/abs/2205.11487
- Podell et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* https://arxiv.org/abs/2307.01952
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* https://arxiv.org/abs/2112.10752

### Training Data and Alignment

- Schuhmann et al. (2022). *LAION-5B: An open large-scale dataset for training next generation image-text models.* https://arxiv.org/abs/2210.08402
- Gadre et al. (2024). *DataComp: In search of the next generation of multimodal datasets.* https://arxiv.org/abs/2304.14108
- Laurençon et al. (2023). *OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents.* https://arxiv.org/abs/2306.16527
- Zhai et al. (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP).* https://arxiv.org/abs/2303.15343

### Evaluation Benchmarks

- Yue et al. (2024). *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark.* https://arxiv.org/abs/2311.16502
- Liu et al. (2023). *MMBench: Is Your Multi-modal Model an All-around Player?* https://arxiv.org/abs/2307.06281
- Li et al. (2023). *SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension.* https://arxiv.org/abs/2307.16125
- Goyal et al. (2017). *Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering (VQAv2).* https://arxiv.org/abs/1612.00837
- Singh et al. (2019). *Towards VQA Models That Can Read (TextVQA).* https://arxiv.org/abs/1904.08920

### Libraries and Tools

- [HuggingFace Transformers — VLM documentation](https://huggingface.co/docs/transformers/model_doc/llava_next)
- [vLLM multimodal inference](https://docs.vllm.ai/en/latest/models/vlm.html)
- [diffusers library](https://github.com/huggingface/diffusers) — Stable Diffusion, SDXL, FLUX serving
- [openai/CLIP](https://github.com/openai/CLIP) — original CLIP inference code
- [`jinaai/jina-clip-v2`](https://huggingface.co/jinaai/jina-clip-v2) — multilingual CLIP variant

### Surveys and Blogs

- Liu et al. (2024). *A Comprehensive Survey and Guide to Multimodal Large Language Models in Vision-Language Tasks.* https://arxiv.org/abs/2411.06284
- Weng, L. (2022). *Generative Models.* https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ — diffusion model fundamentals
- HuggingFace (2023). *IDEFICS: An Open Reproduction of Flamingo.* https://huggingface.co/blog/idefics
