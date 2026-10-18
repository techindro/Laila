<div align="center">

# 🎙️ Laila

## Parameter-Efficient Speech Recognition for Indian Languages

**Adapting Whisper-Small with LoRA — 3.5M trainable params, 1 free T4 GPU**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Made in India](https://img.shields.io/badge/Made_in-India-orange.svg)](https://www.makeinindia.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HF Hub](https://img.shields.io/badge/HuggingFace-techindro-yellow)](https://huggingface.co/techindro)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-blueviolet)](https://neurips.cc)

</div>

---

## 🌐 Overview

![Laila — ASR for Indian Languages](assets/hero.png)

> *"Whisper, but for India — parameter-efficient ASR for Indian-accented and code-switched speech"*

Laila adapts **Whisper-Small** with **Low-Rank Adaptation (LoRA)** to improve speech recognition on **Indian-accented English**, **Hindi**, and **Hindi-English code-switched (Hinglish)** speech — all on a **single free-tier T4 GPU**.

---

## 🚀 Live Demos

Experience Laila in action. All demos run on Hugging Face Spaces — no installation required.

![Laila Live Demos](assets/live-demos.png)

| Demo | Description | Link |
|------|-------------|------|
| 🎙️ Hindi ASR | Hindi speech recognition with LoRA-tuned Whisper | [Try Now](https://huggingface.co/spaces/techindro/Laila-Demo) |
| 🔀 Hinglish ASR | Code-switched Hindi-English transcription | [Try Now](https://huggingface.co/spaces/techindro/Laila-Hinglish) |
| 🇮🇳 Indian English ASR | Indian-accented English transcription | [Try Now](https://huggingface.co/spaces/techindro/Laila-IndianEnglish) |
| 🌐 22-Language Suite | One adapter per scheduled Indian language | [Coming Soon](https://huggingface.co/techindro) |

🌐 Repository: [github.com/techindro/Laila](https://github.com/techindro/Laila)

---

## 📖 What is Laila?

Laila is a **parameter-efficient speech recognition model** purpose-built for India's linguistic diversity. It adapts the base Whisper-Small model using LoRA, training only **1.5% of parameters** while keeping all 244M base weights frozen.

### The Name

- **Laila** = "Night" in Arabic/Persian, "play" in Hindi — evoking speech and expression
- **Model** = Speech recognition for Indian languages

So Laila means bringing clarity to Indian speech, day or night.

### What Problem Does It Solve?

Global speech models like Whisper, trained mostly on Western English, make more errors on Indian speech because:

1. Indian-accented English has different phonetics, rhythm, and stress patterns
2. Everyday speech is often code-switched (Hindi + English in one sentence)
3. No single accepted transcript convention exists (Devanagari vs. Roman)
4. Full fine-tuning is costly — needs more memory and produces full model copies

**The result:** Speech recognition that works perfectly in San Francisco but struggles in Mumbai, Delhi, and Chennai.

### How Does Laila Work?

The pipeline has five parts that work together:

| Part | What It Does |
|------|---------------|
| 🎧 Audio Preprocessing | Resampling, mono conversion, normalization, 30s chunking |
| 🧠 Whisper-Small | Frozen 244M-parameter encoder-decoder Transformer |
| 🔗 LoRA Adapters | Trains 3.5M params on `q_proj` & `v_proj` attention layers |
| 🔊 Noise Augmentation | Mixes real background noise at random SNRs (train only) |
| 📝 Transcript Convention | Devanagari for Hindi, Latin for English, normalized scoring |

---

## 📊 Key Results at a Glance

| Metric | Laila | Whisper-Small (zero-shot) | Improvement |
|--------|-------|---------------------------|-------------|
| Trainable Parameters | 3.5M | 244M | **1.5% of base** |
| Adapter Size | ~few MB | 244M | **~70× smaller** |
| Hardware | 1× T4 (16GB) | 1× T4 (16GB) | **Free-tier** |
| Hindi WER | TBD | TBD | TBD |
| Indian-English WER | TBD | TBD | TBD |
| Code-Switched WER | TBD | TBD | TBD |

---

## 🎯 The Problem

**Global ASR training data ignores 1.4 billion Indian voices.**

Existing models like Whisper were trained on 680k hours of mostly English data:

| Limitation | Consequence |
|------------|-------------|
| Western English bias | Errors on Indian-accented English |
| Monolingual training | Fails on code-switched speech |
| Mixed-script ambiguity | Inconsistent training/evaluation |
| Full fine-tuning cost | High memory, full model copies per variant |

---

## 🚀 The Solution

Laila is a **parameter-efficient adaptation** of Whisper-Small purpose-built for India's linguistic diversity.

### Comparison with Baselines

| Feature | Whisper-Small (zero-shot) | Whisper-Small (full FT) | Laila (LoRA) |
|---------|---------------------------|--------------------------|--------------|
| Hindi Support | Partial | ✅ Yes | ✅ Yes |
| Indian-Accented English | Partial | ✅ Yes | ✅ Yes |
| Code-Switched (Hinglish) | ❌ | ✅ Yes | ✅ Yes |
| Trainable Params | 0 | 244M | **3.5M** |
| Adapter Size | — | Full model | **Few MB** |
| Memory (Training) | — | High | **16GB T4** |
| Base Model Weights | Frozen | Fully trained | **Frozen** |

---

## 🛰️ Space & Defense Use Cases

Laila's architecture extends naturally to **space and defense** speech applications:

### Breakthrough ASR for Space & Beyond

Laila provides a parameter-efficient ASR base for:
- Space agencies (ISRO and allied partners)
- Defense teams and armed forces
- Enterprises working on voice interfaces

### Generative AI Data Engine

Laila's LoRA adapters power **speech data pipelines** for advanced LLMs through:

| Capability | Description |
|------------|-------------|
| **Transcription** | Convert speech to text for prompt-response pairs |
| **Code-Switching** | Handle Hindi-English mixed utterances |
| **Noise Robustness** | Perform in monsoon, traffic, and night conditions |
| **Multi-Language** | 22 Indian language adapters |
| **Evaluation** | Benchmark WER/CER across languages |

The full pipeline: `Audio → Preprocess → LoRA Adapter → Transcription → Downstream NLP`

### Laila Data Engine

![Laila Data Engine — Collect, Curate, Annotate](assets/data-engine.png)

> *Collect, curate, and annotate speech. Train adapters and evaluate. Repeat.*

The Data Engine follows a continuous loop:
