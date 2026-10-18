<div align="center">

<!-- YAHAN APNE GITHUB LOGO KA LINK DAALEIN (Upar wala logo) -->
<img src="https://raw.githubusercontent.com/techindro/Laila/main/assets/logo.png" alt="Laila Logo" width="300"/>

### Parameter-Efficient Adaptation of Whisper-Small for Indian-Accented and Code-Switched Hindi-English Speech

[![GitHub](https://img.shields.io/badge/GitHub-techindro-black?logo=github)](https://github.com/techindro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-purple)](https://github.com/huggingface/peft)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-blueviolet)](https://neurips.cc)

**3.5M trainable params · 1.5% of base model · Trains on 1 free T4 GPU**

</div>

---

## 🌐 Platform Overview

![Laila — ASR for Indian Languages](https://raw.githubusercontent.com/techindro/Laila/main/assets/hero.png)

> *"Whisper, but for India — parameter-efficient ASR for Indian-accented and code-switched speech"*

Laila provides high-quality speech recognition, model evaluation, and AI tools designed for Indian users, enterprises, and researchers working on next-generation voice applications.

---

## 🚀 Live Demos

Experience Laila in action. All demos run on Hugging Face Spaces — no installation required.

![Laila Live Demos](https://raw.githubusercontent.com/techindro/Laila/main/assets/live-demos.png)

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

## 🧠 AI Knowledge Hub

![AI Knowledge Hub — Speech AI & Parameter-Efficient AI](https://raw.githubusercontent.com/techindro/Laila/main/assets/ai-knowledge-hub.png)

Laila explores core pillars of modern speech AI and how LoRA applies each discipline to solve critical language challenges.

### Core AI Disciplines

**Parameter-Efficient Fine-Tuning** — *Train less. Do more.*
> LoRA writes the update to a weight matrix as the product of two low-rank matrices and trains only these. This lowers memory use and the size of the stored update. Laila leverages LoRA to adapt Whisper-Small with just 1.5% of trainable parameters.
- 3.5M trainable params vs. 244M base
- Single free-tier T4 GPU
- Adapter file of a few MB

**Code-Switched Speech** — *Mixed language. Unified model.*
> Code-switched speech mixes languages within a sentence — e.g., "kal meeting ke baad report bhej dena." Laila handles Hindi grammar with English nouns using separate test sets and a fixed transcript convention.
- Separate Hindi, English, and code-switched test sets
- Devanagari + Latin script convention
- Noise-robust training

---

## 📂 Case Studies & Resources

![Case Studies & Resources](https://raw.githubusercontent.com/techindro/Laila/main/assets/case-studies.png)

### Highlighted Case Studies

| Case Study | Category | Description |
|------------|----------|-------------|
| **Hindi ASR with LoRA** | ASR | Adapting Whisper-Small for Hindi speech on a single free T4 GPU |
| **Hinglish Code-Switching** | Code-Switching | Building a transcription pipeline for Hindi-English mixed speech |
| **Noise-Robust ASR** | Robustness | Training with background noise for real-world conditions |

### Learning Resources

![Learn More About The Data Engine](https://raw.githubusercontent.com/techindro/Laila/main/assets/resources.png)

| Type | Title |
|------|-------|
| Paper | Laila: Parameter-Efficient Adaptation of Whisper-Small |
| Guide | Guide to LoRA Fine-Tuning |
| Guide | Guide: Whisper Architecture |
| Guide | Guide: WER & CER Evaluation |

---

## 📊 Benchmark Results

### Word Error Rate (WER, %)

| Model | Hindi | Indian English | Code-Switched | Trainable Params |
|-------|-------|----------------|---------------|------------------|
| Whisper-Small (zero-shot) | TBD | TBD | TBD | 0 |
| Whisper-Small (full fine-tune) | TBD | TBD | TBD | 244M |
| **Laila (LoRA, r=32)** | **TBD** | **TBD** | **TBD** | **3.5M** |

### Character Error Rate (CER, %)

| Model | Hindi | Indian English | Code-Switched |
|-------|-------|----------------|---------------|
| Whisper-Small (zero-shot) | TBD | TBD | TBD |
| Whisper-Small (full fine-tune) | TBD | TBD | TBD |
| **Laila (LoRA, r=32)** | **TBD** | **TBD** | **TBD** |

### Ablation: LoRA Rank

| Rank | Trainable Params | WER (%) |
|------|------------------|---------|
| 8 | TBD | TBD |
| 16 | TBD | TBD |
| **32** | **3.5M** | **TBD** |
| 64 | TBD | TBD |

### Ablation: Adapted Modules & Noise Augmentation

| Adapted Modules | Noise | Hindi | English | Code-Switched |
|-----------------|-------|-------|---------|---------------|
| `q_proj`, `v_proj` | Off | TBD | TBD | TBD |
| `q_proj`, `v_proj` | On | TBD | TBD | TBD |
| `q_proj`, `k_proj`, `v_proj`, `out_proj` | On | TBD | TBD | TBD |
| Decoder only | On | TBD | TBD | TBD |

---

## 🎙️ Indian Language Support (22 Adapters Planned)

Laila plans to support all 22 scheduled languages of India:

| Category | Languages |
|----------|-----------|
| **Indo-Aryan** | Hindi (हिन्दी), Bengali (বাংলা), Marathi (मराठी), Gujarati (ગુજરાતી), Odia (ଓଡ଼ିଆ), Punjabi (ਪੰਜਾਬੀ), Assamese (অসমীয়া), Urdu (اردو), Konkani (कोंकणी), Nepali (नेपाली), Maithili (मैथिली), Dogri (डोगरी), Sindhi (سنڌي) |
| **Dravidian** | Tamil (தமிழ்), Telugu (తెలుగు), Kannada (ಕನ್ನಡ), Malayalam (മലയാളം) |
| **Other** | Sanskrit (संस्कृतम्), Manipuri (মণিপুরী), Bodo (बड़ो), Santali (ᱥᱟᱱᱛᱟᱲᱤ), Kashmiri (کٲشُر) |
| **Code-Switching** | Hinglish, Tanglish, Benglish, Tinglish, Kanglish, Manglish, Punglish, Marathi-English |

---

## 📁 Dataset v1.0 Statistics

| Split | Modality | Samples | Annotated Labels |
|-------|----------|---------|------------------|
| Train | Hindi (Common Voice + AI4Bharat) | TBD | TBD |
| Val | Hindi | TBD | TBD |
| Train | Indian-Accented English | TBD | TBD |
| Val | Indian-Accented English | TBD | TBD |
| Train | Code-Switched Hindi-English | TBD | TBD |
| Val | Code-Switched Hindi-English | TBD | TBD |
| **Total** | **All** | **TBD** | **TBD** |

---

## 🏗️ Technology Stack

| Layer | Technologies |
|-------|--------------|
| Base Model | Whisper-Small (244M parameters) |
| Adaptation | LoRA (PEFT), rank r=32, α=64, dropout=0.05 |
| Deep Learning | PyTorch 2.0+, HuggingFace Transformers |
| Audio Processing | torchaudio, librosa, soundfile |
| Evaluation | jiwer (WER/CER), HuggingFace Evaluate |
| Training | HuggingFace Trainer, FP16, gradient checkpointing |
| Inference | PEFT + Transformers, serverless worker architecture |
| Infrastructure | 1× free-tier T4 GPU (16GB) |

---

## 🔮 What's Next?

- Support for all 22 Indian languages
- Real-time speech data processing
- API for companies to use
- Expansion to other code-switched pairs
- Deeper integration with AI4Bharat and Bhashini
- Expansion of noise-robust ASR for defense applications

---

## 📜 License

MIT — Free to use, modify, and distribute.

---

## 🤝 Contributing

PRs welcome! Let's build speech AI for Bharat 🇮🇳
