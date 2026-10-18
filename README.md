<div align="center">

# 🎙️ Laila

### Parameter-Efficient Adaptation of Whisper-Small for Indian-Accented and Code-Switched Hindi-English Speech

[![GitHub](https://img.shields.io/badge/GitHub-techindro-black?logo=github)](https://github.com/techindro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-purple)](https://github.com/huggingface/peft)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-blueviolet)](https://neurips.cc)

**3.5M trainable params · 1.5% of base model · Trains on 1 free T4 GPU**

[Overview](#-overview) · [Features](#-features) · [Installation](#-installation) · [Training](#-training) · [Evaluation](#-evaluation) · [Inference](#-inference) · [Results](#-results) · [Citation](#-citation)

</div>

---

## 📖 Overview

Speech recognition models trained on broad global data often make more errors on **Indian-accented English** and on **code-switched Hindi-English speech**, where speakers switch between languages within a single sentence.

**Laila** addresses this by adapting **Whisper-Small** with **Low-Rank Adaptation (LoRA)**. Only the query (`q_proj`) and value (`v_proj`) projections in the attention layers are trained, keeping all 244M base parameters frozen.

| Property | Value |
|---|---|
| **Base Model** | Whisper-Small (244M parameters) |
| **Trainable Parameters** | 3,538,944 (~1.5%) |
| **Adapter Size** | A few MB |
| **Hardware** | 1× free-tier 16GB T4 GPU |
| **Languages** | Hindi, Indian-accented English, Hindi-English code-switched |

The adapter is a small file (a few MB), making it easy to share, version, and deploy.

---

## ✨ Features

- **Parameter-Efficient** — Trains only 1.5% of the model's parameters.
- **Low-Resource Friendly** — Fits entirely on a free-tier T4 GPU.
- **Tiny Adapter** — Weights are only a few megabytes.
- **Noise-Robust** — Trained with background noise augmentation (fan hum, traffic, crowd chatter).
- **Multi-Test Evaluation** — Separate test sets for Hindi, Indian-accented English, and code-switched speech.
- **Ablation Studies** — Analyzes LoRA rank, adapted modules, and noise augmentation.
- **Reproducible** — Code, adapter weights, dependency versions, and data splits are released.

---

## 🏗️ Architecture

Laila applies LoRA to the attention layers of Whisper-Small. For a frozen weight matrix \( W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \), LoRA learns two small matrices \( A \in \mathbb{R}^{r \times d_{\text{in}}} \) and \( B \in \mathbb{R}^{d_{\text{out}} \times r} \):

\[
W' = W + \frac{\alpha}{r} BA
\]

| Component | Configuration |
|---|---|
| **LoRA Rank (\( r \))** | 32 |
| **LoRA Alpha (\( \alpha \))** | 64 |
| **LoRA Dropout** | 0.05 |
| **Adapted Modules** | `q_proj`, `v_proj` (encoder & decoder) |
| **Precision** | FP16 mixed precision |
| **Bias** | Not trained |

**Adapted layers:**
- Encoder self-attention: 24 layers (12 layers × 2 projections)
- Decoder self-attention: 24 layers
- Decoder cross-attention: 24 layers

Each adapter adds \( r(d_{\text{in}} + d_{\text{out}}) = 32 \times (768 + 768) = 49,152 \) parameters per layer.

---

## ⚙️ Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (T4 or better recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/techindro/Laila.git
cd Laila

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
