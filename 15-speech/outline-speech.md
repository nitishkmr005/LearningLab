# 14 — Speech

Exhaustive learning path for speech processing: audio fundamentals, ASR, TTS, speaker recognition, and modern end-to-end models.

---

## 01 — Audio Fundamentals
Sampling rate, bit depth, PCM; Nyquist theorem; mono vs stereo; loudness (dBFS, RMS); reading audio with librosa and soundfile.
- https://librosa.org/doc/latest/tutorial.html
- https://realpython.com/playing-and-recording-sound-python/

## 02 — Signal Processing: STFT & Spectrograms
Fourier transform; Short-Time Fourier Transform (STFT); magnitude/phase spectrogram; hop length, window size; Griffin-Lim reconstruction.
- https://librosa.org/doc/latest/generated/librosa.stft.html

## 03 — Mel Spectrograms & MFCCs
Mel filterbank; mel spectrogram; MFCC (cepstral coefficients); delta/delta-delta features; why MFCCs approximate auditory perception.
- https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
- https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

## 04 — Classical ASR Pipeline
GMM-HMM architecture; acoustic model, language model, pronunciation lexicon; Viterbi decoding; beam search; WER metric.
- https://kaldi-asr.org/doc/

## 05 — CTC & Sequence-to-Sequence for ASR
Connectionist Temporal Classification (CTC) loss; blank token; greedy vs beam-search decoding; attention-based encoder-decoder; LM shallow fusion.
- https://arxiv.org/abs/1412.5567
- https://distill.pub/2017/ctc/

## 06 — Whisper & Modern ASR
Whisper architecture (encoder-decoder Transformer); multilingual, multitask training; word-level timestamps; faster-whisper; fine-tuning on custom data.
- https://arxiv.org/abs/2212.04356
- https://huggingface.co/openai/whisper-large-v3

## 07 — Wav2Vec 2.0 & Self-Supervised Speech
Masked feature prediction; quantization; contrastive loss; fine-tuning on low-resource languages; HuBERT; WavLM.
- https://arxiv.org/abs/2006.11477
- https://huggingface.co/docs/transformers/model_doc/wav2vec2

## 08 — ASR Evaluation
Word Error Rate (WER), Character Error Rate (CER), Match Error Rate; normalization (case, punctuation); domain-specific benchmarks (LibriSpeech, Common Voice).
- https://huggingface.co/spaces/evaluate-metric/wer
- https://commonvoice.mozilla.org/en/datasets

## 09 — Text-to-Speech (TTS) Fundamentals
Phonemes and grapheme-to-phoneme (G2P); prosody (pitch, duration, energy); mel spectrogram prediction; vocoder; naturalness vs intelligibility.
- https://google.github.io/tacotron/

## 10 — Neural TTS: Tacotron & FastSpeech
Tacotron 2 (attention-based mel prediction); FastSpeech 2 (non-autoregressive, duration predictor); VITS (end-to-end variational); comparison.
- https://arxiv.org/abs/1712.05884
- https://arxiv.org/abs/2006.04558

## 11 — Vocoders
WaveNet; WaveGlow; HiFi-GAN; UnivNet; mel-to-waveform; trade-offs between quality, speed, and memory.
- https://arxiv.org/abs/1609.03499
- https://arxiv.org/abs/2010.05646

## 12 — Speaker Recognition & Diarization
Speaker verification (x-vectors, d-vectors, ECAPA-TDNN); Equal Error Rate (EER); speaker diarization pipeline; pyannote.audio.
- https://github.com/pyannote/pyannote-audio
- https://arxiv.org/abs/2005.07143

## 13 — Voice Conversion & Cloning
Any-to-any voice conversion; YourTTS; VALL-E; speaker embedding conditioning; zero-shot TTS; ethical considerations.
- https://arxiv.org/abs/2301.02111
- https://huggingface.co/coqui/XTTS-v2

## 14 — Streaming ASR & Real-Time Inference
Online decoding; chunked inference; CTC with partial hypothesis; latency vs accuracy trade-off; streaming Whisper; WebRTC audio.
- https://github.com/snakers4/silero-models

## 15 — Speech Libraries & Ecosystem
librosa (analysis); soundfile (I/O); torchaudio (transforms, datasets); HuggingFace transformers (ASR/TTS pipelines); pyannote; Coqui TTS.
- https://pytorch.org/audio/stable/index.html
- https://huggingface.co/docs/transformers/index

## 16 — Audio Foundation Models
SeamlessM4T (Meta): speech-to-speech, speech-to-text in 100+ languages; AudioPaLM; Whisper large-v3-turbo; zero-shot cross-lingual transfer; evaluate on CoVoST-2 and FLEURS benchmarks.
- https://arxiv.org/abs/2308.11596
- https://huggingface.co/openai/whisper-large-v3-turbo

## 17 — Voice Agent Architecture
Full pipeline: VAD (Voice Activity Detection) → ASR → LLM → TTS; latency budget per stage; streaming ASR + streaming TTS for low-latency response; WebSocket / WebRTC transport; interrupt handling; LiveKit, Daily.co SDKs.
- https://docs.livekit.io/agents/
- https://github.com/pipecat-ai/pipecat
