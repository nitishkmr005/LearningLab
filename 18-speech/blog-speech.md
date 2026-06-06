# Speech Processing: From Audio Fundamentals to Voice Agent Architecture

> A comprehensive guide to ASR, TTS, speaker recognition, and production voice pipelines — for ML Engineers and AI Engineers.

---

## Table of Contents

1. [Audio Fundamentals](#1-audio-fundamentals)
2. [Signal Processing: STFT and Spectrograms](#2-signal-processing-stft-and-spectrograms)
3. [Mel Spectrograms and MFCCs](#3-mel-spectrograms-and-mfccs)
4. [Classical ASR Pipeline](#4-classical-asr-pipeline)
5. [CTC and Sequence-to-Sequence ASR](#5-ctc-and-sequence-to-sequence-asr)
6. [Whisper and Modern ASR](#6-whisper-and-modern-asr)
7. [Wav2Vec 2.0 and Self-Supervised Speech](#7-wav2vec-20-and-self-supervised-speech)
8. [ASR Evaluation](#8-asr-evaluation)
9. [Text-to-Speech Fundamentals](#9-text-to-speech-fundamentals)
10. [Neural TTS: Tacotron and FastSpeech](#10-neural-tts-tacotron-and-fastspeech)
11. [Vocoders](#11-vocoders)
12. [Speaker Recognition and Diarization](#12-speaker-recognition-and-diarization)
13. [Voice Conversion and Cloning](#13-voice-conversion-and-cloning)
14. [Streaming ASR and Real-Time Inference](#14-streaming-asr-and-real-time-inference)
15. [Audio Foundation Models](#15-audio-foundation-models)
16. [Voice Agent Architecture](#16-voice-agent-architecture)
17. [Speech Libraries and Ecosystem](#17-speech-libraries-and-ecosystem)
18. [References](#18-references)

---

## 1. Audio Fundamentals

Digital audio is a sampled representation of continuous sound waves. Understanding the sampling parameters is essential for building any speech system.

**Sampling rate (Hz):** how many samples per second. Human speech contains frequencies up to ~8 kHz; telephone quality uses 8 kHz; high-quality audio uses 16–44.1 kHz. The **Nyquist theorem** states you need at least 2× the highest frequency: to capture 8 kHz speech, sample at ≥ 16 kHz.

**Bit depth:** resolution of each sample. 16-bit is standard (65,536 levels). 8-bit (256 levels) is noticeably degraded.

**PCM (Pulse Code Modulation):** the raw uncompressed format. A 16kHz mono 16-bit audio file uses 16,000 × 2 bytes = 32,000 bytes/second = 1.9 MB/minute.

**Loudness metrics:**
- **dBFS (decibels relative to full scale):** 0 dBFS = maximum digital amplitude. Speech is typically at -20 to -6 dBFS
- **RMS energy:** root mean square of amplitude — measures average loudness over a window

```python
import librosa
import numpy as np
import soundfile as sf

# Load audio file
audio, sr = librosa.load("speech.wav", sr=16000, mono=True)  # resample to 16kHz, mono
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")
print(f"Samples: {len(audio)}")

# Loudness metrics
rms = np.sqrt(np.mean(audio**2))                             # RMS energy
db_rms = 20 * np.log10(rms + 1e-10)                         # convert to dB
print(f"RMS energy: {db_rms:.1f} dBFS")

# Normalize audio to -20 dBFS
target_db = -20
gain = 10**((target_db - db_rms) / 20)                      # gain factor
audio_normalized = audio * gain

# Save
sf.write("speech_normalized.wav", audio_normalized, sr)
```

---

## 2. Signal Processing: STFT and Spectrograms

Raw audio waveforms are hard to model — a 1-second clip at 16kHz is 16,000 samples with complex temporal patterns. The **Short-Time Fourier Transform (STFT)** converts audio into a time-frequency representation that reveals which frequencies are present at each moment.

STFT divides the signal into overlapping frames using a window (typically Hann window), applies FFT to each frame, and stacks the results:

```
STFT[t, f] = Σ_n x[n] · w[n - t·hop] · e^(-j2πfn/N)
```

Parameters:
- **n_fft:** FFT size (e.g., 1024) — determines frequency resolution
- **hop_length:** step between frames (e.g., 256) — determines time resolution
- **win_length:** window size (typically = n_fft)

```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio, sr = librosa.load("speech.wav", sr=16000, mono=True)

# Compute STFT
D = librosa.stft(audio, n_fft=1024, hop_length=256, win_length=1024)

# Separate magnitude and phase
magnitude = np.abs(D)                                        # |STFT|
phase = np.angle(D)                                          # phase angle

# Power spectrogram in dB
power_db = librosa.amplitude_to_db(magnitude, ref=np.max)

# Reconstruct waveform from magnitude (Griffin-Lim)
magnitude_only = np.abs(D)
audio_reconstructed = librosa.griffinlim(magnitude_only, n_iter=32, hop_length=256)

print(f"Spectrogram shape: {magnitude.shape}")               # (freq_bins, time_frames)
print(f"Freq bins: {magnitude.shape[0]}, Time frames: {magnitude.shape[1]}")
```

**Griffin-Lim reconstruction** iteratively estimates phase from a magnitude-only spectrogram. Modern neural vocoders (HiFi-GAN) are far better for audio synthesis.

---

## 3. Mel Spectrograms and MFCCs

Linear-frequency spectrograms don't match human auditory perception. The cochlea responds logarithmically — we perceive differences between 200 Hz and 300 Hz as similar to differences between 2000 Hz and 3000 Hz, even though the raw frequency difference is much larger.

The **Mel scale** maps frequencies to a perceptually uniform space:
```
mel(f) = 2595 · log10(1 + f/700)
```

A **Mel spectrogram** applies a filterbank of M triangular filters (typically 80 or 128) spaced on the mel scale, averaging the power in each filter's frequency range.

**MFCCs (Mel-Frequency Cepstral Coefficients)** apply a DCT to the log mel spectrogram, decorrelating the filter bank outputs into compact coefficients. The first 13 MFCCs capture most of the speech information:

```python
import librosa
import numpy as np

audio, sr = librosa.load("speech.wav", sr=16000)

# Mel spectrogram: the backbone of modern ASR preprocessing
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=sr,
    n_mels=80,                                               # 80 mel bins (Whisper uses 80)
    n_fft=1024,
    hop_length=256,
    fmin=0,
    fmax=8000                                                # max frequency (Nyquist for 16kHz)
)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)          # convert to log scale

# MFCCs: classic handcrafted feature for speech
mfccs = librosa.feature.mfcc(
    y=audio, sr=sr,
    n_mfcc=13,                                               # 13 coefficients
    n_mels=40
)

# Delta features: model temporal dynamics
delta_mfccs = librosa.feature.delta(mfccs)                  # first derivative
delta2_mfccs = librosa.feature.delta(mfccs, order=2)        # second derivative

# Stack into 39-dim feature vector per frame (classic HMM-GMM feature)
features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
print(f"MFCC shape: {mfccs.shape}")                         # (13, time_frames)
print(f"Full feature shape: {features.shape}")               # (39, time_frames)
```

🎯 **Interview prep:** "Why do we use mel spectrograms instead of raw waveforms for ASR?" — Mel spectrograms compress audio into a compact, perceptually relevant representation (~80×~300 floats for 3 seconds vs 48,000 raw samples). They also remove phase information, which is largely irrelevant for speech recognition. However, modern end-to-end models (wav2vec 2.0) increasingly process raw waveforms.

---

## 4. Classical ASR Pipeline

Before deep learning, ASR used a pipeline of separately trained components:

1. **Acoustic Model (AM):** Gaussian Mixture Model (GMM) or DNN mapping acoustic features to HMM state probabilities
2. **Language Model (LM):** n-gram model giving P(word | previous words)
3. **Pronunciation Lexicon:** mapping words to phoneme sequences
4. **Decoder:** Viterbi search combining AM and LM scores

**HMM-GMM architecture:**
- Each phoneme is modeled as a 3-state HMM (silence, closure, release for stops; or onset, steady-state, offset for others)
- Each HMM state emits observations from a GMM over MFCC features
- Decoding: find the sequence of words W that maximizes P(W|X) ∝ P(X|W) · P(W) (acoustic × language model)

**WER (Word Error Rate):** the primary metric for ASR quality:

```
WER = (S + D + I) / N
```

where S = substitutions, D = deletions, I = insertions, N = number of reference words.

```python
def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using dynamic programming (edit distance)."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    n, m = len(ref), len(hyp)

    # Edit distance matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i                                          # deletions
    for j in range(m + 1):
        dp[0][j] = j                                          # insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]                      # no change
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],                               # deletion
                    dp[i][j-1],                               # insertion
                    dp[i-1][j-1]                              # substitution
                )
    return dp[n][m] / n                                       # normalize by reference length

wer = compute_wer(
    "the cat sat on the mat",
    "the cat sat on a mat"
)
print(f"WER: {wer:.2%}")                                     # 1 substitution / 6 words = 16.7%
```

---

## 5. CTC and Sequence-to-Sequence ASR

**CTC (Connectionist Temporal Classification)** ([Graves et al., 2006](https://arxiv.org/abs/1412.5567)) solved the alignment problem in sequence-to-sequence tasks. The key insight: add a special blank token that can be emitted at any time step. CTC marginalizes over all valid alignments (all sequences of characters and blanks that reduce to the target string):

```
P(y|x) = Σ_{π: B(π)=y} Π_t p(π_t|x)
```

where B collapses consecutive identical labels and removes blanks.

CTC advantages: no need for frame-level alignments in training data; allows efficient forward-backward algorithm. Disadvantages: assumes conditional independence of output labels (no language model).

**Sequence-to-sequence with attention** (like Whisper) explicitly models output label dependencies using an autoregressive decoder and cross-attention over encoder outputs. More accurate but slower at inference.

```python
# CTC decoding (greedy)
def ctc_greedy_decode(logits: "torch.Tensor", blank_id: int = 0) -> list[int]:
    """Greedy CTC decoding: argmax over each time step, collapse, remove blanks."""
    import torch
    predictions = torch.argmax(logits, dim=-1)                # (time,) best label per step
    collapsed = []
    prev = None
    for p in predictions.tolist():
        if p != prev:                                          # collapse consecutive same labels
            collapsed.append(p)
        prev = p
    return [c for c in collapsed if c != blank_id]            # remove blank tokens
```

---

## 6. Whisper and Modern ASR

**Whisper** ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) is an encoder-decoder Transformer trained by OpenAI on 680,000 hours of web-scraped multilingual audio. It is currently the strongest openly available ASR model.

**Architecture:**
- 30-second audio segments converted to 80-channel mel spectrograms
- Encoder: Transformer encoder over spectrogram frames
- Decoder: autoregressive Transformer, conditioned on encoder output via cross-attention
- Multitask training: transcription, translation, language identification, VAD, timestamp prediction — all via special tokens

**Model sizes:** Whisper tiny (39M), base (74M), small (244M), medium (769M), large-v3 (1.5B).

```python
# pip install faster-whisper
from faster_whisper import WhisperModel

# faster-whisper: 4× faster than openai-whisper, same accuracy
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"                                    # use fp16 for speed
)

segments, info = model.transcribe(
    "speech.wav",
    beam_size=5,                                              # beam search
    word_timestamps=True,                                     # word-level timing
    language="en"                                             # force language (skip detection)
)

for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
    if hasattr(segment, 'words') and segment.words:
        for word in segment.words:
            print(f"  Word: '{word.word}' [{word.start:.2f}s → {word.end:.2f}s]")
```

**Fine-tuning Whisper on custom data:**

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Prepare custom dataset (requires audio + transcription pairs)
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch
```

🏭 **Production note:** For production ASR, faster-whisper with CTranslate2 backend is the standard. At 16kHz audio, large-v3 runs in ~0.3× real-time on a single A10G GPU (i.e., 1 minute of audio transcribed in ~18 seconds).

---

## 7. Wav2Vec 2.0 and Self-Supervised Speech

**Wav2Vec 2.0** ([Baevski et al., 2020](https://arxiv.org/abs/2006.11477)) learns speech representations without transcriptions, using **self-supervised learning** — crucial for low-resource languages where labeled data is scarce.

The approach:
1. Pass raw waveform through a convolutional feature encoder → latent speech representations
2. Quantize representations into discrete speech units using a codebook (Gumbel-softmax)
3. Mask 50% of encoded representations (like BERT's masked tokens)
4. Train a Transformer encoder to predict the correct quantized unit at masked positions using contrastive loss

After pre-training, fine-tune on just 10 minutes of labeled audio to achieve competitive WER.

**HuBERT** and **WavLM** extend this framework: HuBERT uses k-means cluster IDs as pseudo-labels instead of a learned quantizer; WavLM adds denoising to the objective, improving performance on downstream tasks.

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio, sr = librosa.load("speech.wav", sr=16000)             # must be 16kHz
inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(**inputs).logits                           # (batch, time, vocab)

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(f"Transcription: {transcription[0]}")
```

---

## 8. ASR Evaluation

**WER** is the primary metric but has blind spots:

| Metric | Formula | Sensitivity |
|---|---|---|
| WER | (S+D+I)/N | Character errors count as full word errors |
| CER | (S+D+I)/N on characters | Better for logographic languages |
| MER (Match Error Rate) | 1 - matches/max(|ref|,|hyp|) | Handles very short references |

**Normalization matters:** WER depends heavily on text normalization — should "Dr." be "doctor"? Should numbers be spelled out? "five" vs "5"? Define your normalization strategy before reporting WER.

**Domain-specific benchmarks:**
- **LibriSpeech** test-clean: clean audiobook reading; top models achieve 1.9% WER
- **LibriSpeech** test-other: harder, noisy conditions; top models achieve 4–5% WER
- **Common Voice**: multilingual, crowdsourced; more representative of real-world accents
- **FLEURS**: Google's benchmark across 102 languages

```python
# pip install evaluate jiwer
import evaluate

wer_metric = evaluate.load("wer")

references = ["the cat sat on the mat", "hello world"]
hypotheses = ["the cat sat on a mat", "hello word"]

wer = wer_metric.compute(references=references, predictions=hypotheses)
print(f"WER: {wer:.3f}")                                     # 0.083 (1 error / 12 words)
```

---

## 9. Text-to-Speech Fundamentals

TTS converts text to natural-sounding speech. The pipeline:

1. **Text analysis:** tokenize, normalize (expand "Dr." → "doctor", "123" → "one two three")
2. **Grapheme-to-Phoneme (G2P):** convert text to phoneme sequences (e.g., "read" → /r ɛ d/ or /r iː d/ — context-dependent)
3. **Prosody prediction:** predict duration, pitch (F0), energy for each phoneme
4. **Acoustic model:** predict mel spectrogram from phoneme sequence + prosody
5. **Vocoder:** convert mel spectrogram to waveform

**Prosody** is what makes speech sound natural — the rhythm, melody, and stress patterns. Getting prosody right is what separates lifelike TTS from robotic TTS.

**Key quality metrics:**
- **MOS (Mean Opinion Score):** human listeners rate naturalness 1–5; human speech ≈ 4.5; state-of-the-art TTS ≈ 4.3
- **Intelligibility (WER with forced alignment):** is it understandable?

---

## 10. Neural TTS: Tacotron and FastSpeech

**Tacotron 2** ([Shen et al., 2018](https://arxiv.org/abs/1712.05884)): the first competitive neural TTS system. Uses a sequence-to-sequence Transformer with attention to predict mel spectrograms from character/phoneme sequences. Autoregressive — generates one mel frame at a time, conditioned on all previous frames. High quality, slow inference.

**FastSpeech 2** ([Ren et al., 2020](https://arxiv.org/abs/2006.04558)): non-autoregressive (parallel) mel spectrogram prediction. A duration predictor learns to align phonemes to mel frames, then all frames are generated in parallel. 30× faster inference than Tacotron 2 at similar quality.

**VITS** ([Kim et al., 2021](https://arxiv.org/abs/2106.06103)): end-to-end variational model that generates waveforms directly from text, bypassing the mel spectrogram intermediate. Combines a posterior encoder, a flow-based decoder, and a discriminator. Competitive with two-stage systems at half the latency.

```python
# pip install TTS
from TTS.api import TTS

# Coqui TTS: production-ready TTS library
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

# Generate speech
tts.tts_to_file(
    text="The mel spectrogram is a time-frequency representation of audio.",
    file_path="output.wav"
)

# Multi-speaker TTS with voice cloning
tts_multi = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts_multi.tts_to_file(
    text="Bonjour, comment allez-vous?",
    speaker_wav="reference_speaker.wav",                      # reference for voice cloning
    language="fr",
    file_path="output_cloned.wav"
)
```

---

## 11. Vocoders

A vocoder converts mel spectrograms to waveforms. This is the final synthesis step in most TTS pipelines.

**WaveNet** ([van den Oord et al., 2016](https://arxiv.org/abs/1609.03499)): autoregressive dilated causal convolutions. Extremely high quality but 1000× slower than real-time without optimizations.

**HiFi-GAN** ([Kong et al., 2020](https://arxiv.org/abs/2010.05646)): GAN-based vocoder with multi-period and multi-scale discriminators. Real-time on CPU, near-WaveNet quality. The standard production vocoder.

**WaveGlow** (NVIDIA): normalizing flow model; parallel generation, GPU-efficient.

**UnivNet:** multi-resolution discriminator for improved high-frequency reproduction.

```python
# HiFi-GAN via torchaudio
import torch
import torchaudio

# Load pretrained HiFi-GAN from NVIDIA
bundle = torchaudio.pipelines.HIFIGAN_VOCODER_V3_LJSPEECH
vocoder = bundle.get_vocoder()

# Convert mel spectrogram to waveform
mel_spec = torch.randn(1, 80, 100)                           # (batch, mel_bins, frames)
with torch.inference_mode():
    waveform = vocoder(mel_spec)                             # (batch, 1, time)
print(f"Waveform shape: {waveform.shape}")
torchaudio.save("synthesized.wav", waveform.squeeze(0), 22050)
```

---

## 12. Speaker Recognition and Diarization

**Speaker verification:** given two utterances, are they from the same speaker? Binary decision.

**Speaker identification:** given an utterance and a gallery of known speakers, which speaker is it?

**Speaker embedding models:**
- **x-vectors** (TDNN-based): extract a fixed-dimensional speaker embedding (usually 512-dim) from variable-length utterances by mean-pooling over time
- **ECAPA-TDNN** ([Desplanques et al., 2020](https://arxiv.org/abs/2005.07143)): channel- and context-dependent attention; state-of-the-art for speaker verification

**EER (Equal Error Rate):** the threshold where false accept rate = false reject rate. Lower EER = better.

**Speaker diarization:** "who spoke when?" — segment audio by speaker without prior knowledge of how many speakers are present. Pipeline:
1. Segment audio into speech/non-speech regions (VAD)
2. Extract speaker embeddings per segment
3. Cluster embeddings (agglomerative clustering)
4. Assign speaker labels to segments

```python
# pip install pyannote.audio
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

diarization = pipeline("audio.wav", num_speakers=2)          # optional speaker count hint

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {turn.start:.1f}s → {turn.end:.1f}s")
```

---

## 13. Voice Conversion and Cloning

**Voice conversion:** transform speech from speaker A to sound like speaker B while preserving linguistic content. Any-to-any conversion requires only a reference sample of the target speaker — no retraining.

**VALL-E** ([Wang et al., 2023](https://arxiv.org/abs/2301.02111)): treats TTS as a conditional language modeling problem over acoustic tokens. Given a 3-second prompt from the target speaker, generates high-quality speech that preserves the speaker's voice, acoustic environment, and emotion. Zero-shot voice cloning.

**XTTS (Coqui TTS v2):** open-source multilingual voice cloning. Fine-tuning optional — works zero-shot with a 6-second reference audio.

**Ethical considerations:** voice cloning technology can be used for fraud and disinformation. Production systems should:
- Watermark synthesized speech (SynthID)
- Require explicit consent from the voice owner
- Detect AI-generated speech (deepfake audio detection)
- Limit cloning to authenticated users/speakers

---

## 14. Streaming ASR and Real-Time Inference

Batch ASR processes the complete audio file. Voice agents need streaming ASR — produce transcriptions incrementally as audio arrives, with low latency.

**Key latency budget for voice interfaces:**
- ASR latency: < 300ms from end of speech to final transcription
- LLM first-token latency: < 500ms
- TTS start latency: < 200ms
- **Total: < 1 second** for a natural conversational feel

**CTC-based streaming:** CTC models can decode greedily on each chunk without waiting for future context. Partial hypotheses can be emitted immediately.

**Streaming Whisper:** Whisper is not natively streaming, but with VAD-based chunking (segment on silence) and sliding window approaches, near-streaming behavior is achievable.

**Silero VAD** ([snakers4/silero-models](https://github.com/snakers4/silero-models)): fast, lightweight Voice Activity Detection (50ms latency) — determines when the user has stopped speaking to trigger ASR.

```python
import torch
import numpy as np

# Silero VAD for endpoint detection
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)
get_speech_timestamps, _, read_audio, *_ = utils

audio = read_audio("speech.wav", sampling_rate=16000)
speech_timestamps = get_speech_timestamps(
    audio, model,
    threshold=0.5,                                           # VAD confidence threshold
    sampling_rate=16000,
    min_speech_duration_ms=250,                              # ignore very short speech bursts
    min_silence_duration_ms=100                              # merge gaps shorter than this
)

for ts in speech_timestamps:
    print(f"Speech: {ts['start']/16000:.2f}s → {ts['end']/16000:.2f}s")
```

---

## 15. Audio Foundation Models

**SeamlessM4T** ([Barrault et al., 2023](https://arxiv.org/abs/2308.11596)) — Meta's unified model for speech-to-text, text-to-speech, and speech-to-speech translation across 100+ languages. One model for the full multilingual translation pipeline:

```python
from transformers import SeamlessM4TModel, AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large")

# Speech-to-text translation: Spanish audio → English text
audio, sr = librosa.load("spanish_audio.wav", sr=16000)
inputs = processor(audios=audio, return_tensors="pt")
output_tokens = model.generate(**inputs, tgt_lang="eng")
translation = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(f"Translation: {translation}")
```

**Whisper large-v3-turbo:** 2× faster than large-v3 via distillation with minimal WER degradation. Best cost-quality tradeoff for production transcription.

**Benchmarks:** CoVoST-2 (speech translation) and FLEURS (multilingual ASR across 102 languages) are the standard evaluation benchmarks for multilingual speech models.

---

## 16. Voice Agent Architecture

A production voice agent connects ASR → LLM → TTS in a low-latency pipeline. The architecture determines end-to-end latency, naturalness, and reliability.

**Component breakdown:**

```
Microphone → VAD → Streaming ASR → LLM → Streaming TTS → Speaker
              ↓
         Endpoint detection: when did the user finish speaking?
```

**Full pipeline latency budget (target < 1 second):**

| Stage | Target Latency | Notes |
|---|---|---|
| VAD endpoint detection | 100–200ms | Detect end of speech |
| ASR (Whisper small, GPU) | 150–300ms | Transcribe utterance |
| LLM first token | 300–500ms | Start TTS as soon as first sentence complete |
| TTS first audio chunk | 100–200ms | Stream audio while generating |

**Streaming TTS:** start playing audio as soon as the first sentence is available, while the LLM continues generating the rest of the response. Reduces perceived latency by 50%.

**Interrupt handling:** if the user starts speaking while the agent is talking (barge-in), immediately stop TTS playback and process the new input.

```python
# Pipecat: production voice agent framework
# pip install pipecat-ai

from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.network.websocket_server import WebsocketServerTransport

transport = WebsocketServerTransport(host="0.0.0.0", port=8765)

pipeline = Pipeline([
    transport.input(),                                       # WebSocket audio input
    WhisperSTTService(model="large-v3-turbo"),               # streaming ASR
    AnthropicLLMService(model="claude-haiku-4-5-20251001"),  # LLM
    CartesiaTTSService(),                                    # streaming TTS
    transport.output()                                       # WebSocket audio output
])
```

**LiveKit** ([docs.livekit.io/agents](https://docs.livekit.io/agents)): WebRTC-based platform with built-in support for ASR, TTS, and LLM pipeline composition. Handles audio routing, echo cancellation, and multi-party sessions.

🏭 **Production note:** Echo cancellation (AEC) is critical for voice agents. Without it, the agent hears its own TTS output and tries to transcribe it. WebRTC includes built-in AEC; platforms like Daily.co and LiveKit expose it through their SDK.

---

## 17. Speech Libraries and Ecosystem

| Library | Purpose | Best For |
|---|---|---|
| **librosa** | Audio analysis, feature extraction | Research, preprocessing |
| **soundfile** | Fast audio file I/O (WAV, FLAC, OGG) | Production file handling |
| **torchaudio** | PyTorch-native audio processing | Training, transforms, datasets |
| **faster-whisper** | Optimized Whisper inference | Production ASR |
| **pyannote.audio** | Speaker diarization, VAD | Multi-speaker scenarios |
| **Coqui TTS** | Neural TTS, voice cloning | Open-source TTS |
| **Pipecat** | Voice agent pipeline framework | Voice agent orchestration |
| **LiveKit Agents** | WebRTC + agent hosting | Production voice agents |

```python
import torchaudio
import torchaudio.transforms as T

# torchaudio: GPU-accelerated audio transforms
waveform, sr = torchaudio.load("speech.wav")                 # (channels, time)

# Resample to 16kHz
resampler = T.Resample(orig_freq=sr, new_freq=16000)
waveform_16k = resampler(waveform)

# MelSpectrogram transform
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)
mel = mel_transform(waveform_16k)                            # (channels, n_mels, time)
print(f"Mel shape: {mel.shape}")

# Augmentation: SpecAugment for ASR training
spec_aug = T.SpecAugment(
    n_time_masks=2,
    time_mask_param=40,                                      # mask up to 40 time steps
    n_freq_masks=2,
    freq_mask_param=27                                       # mask up to 27 frequency bins
)
mel_aug = spec_aug(mel)
```

---

## 18. References

### Audio Processing

- [librosa Documentation](https://librosa.org/doc/latest/tutorial.html)
- [torchaudio Documentation](https://pytorch.org/audio/stable/index.html)

### ASR

- [Graves et al. (2006). Connectionist Temporal Classification (CTC).](https://arxiv.org/abs/1412.5567)
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).](https://arxiv.org/abs/2212.04356)
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.](https://arxiv.org/abs/2006.11477)

### TTS

- [Shen et al. (2018). Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Tacotron 2).](https://arxiv.org/abs/1712.05884)
- [Ren et al. (2020). FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.](https://arxiv.org/abs/2006.04558)
- [van den Oord et al. (2016). WaveNet: A Generative Model for Raw Audio.](https://arxiv.org/abs/1609.03499)
- [Kong et al. (2020). HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.](https://arxiv.org/abs/2010.05646)

### Speaker Recognition

- [Desplanques et al. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation, and Aggregation in TDNN.](https://arxiv.org/abs/2005.07143)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)

### Voice Cloning

- [Wang et al. (2023). Neural Codec Language Models Are Zero-Shot Text to Speech Synthesizers (VALL-E).](https://arxiv.org/abs/2301.02111)
- [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

### Multilingual

- [Barrault et al. (2023). SeamlessM4T: Massively Multilingual and Multimodal Machine Translation.](https://arxiv.org/abs/2308.11596)
- [Whisper large-v3-turbo (HuggingFace)](https://huggingface.co/openai/whisper-large-v3-turbo)

### Production

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Pipecat (voice agent framework)](https://github.com/pipecat-ai/pipecat)
- [Silero VAD](https://github.com/snakers4/silero-models)
