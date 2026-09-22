# ReLU-KAN-RAG-LLM: 124M End-to-End Language Model & RAG Engine

A lightweight, consumer-hardware-optimized **124.2M parameter Language Model** based on **Kolmogorov-Arnold Networks (ReLU-KAN)**, paired with a local **Vector Retrieval-Augmented Generation (RAG)** engine. 

Engineered from scratch to train and run entirely on an **11 GB NVIDIA GeForce RTX 2080 Ti**, featuring **1024-token context length**, **memory-mapped dataset streaming** (host RAM < 350 MB), and **activation checkpointing** (peak VRAM ~4.8 GB).

---

## Highlights

* **Pure ReLU-KAN Architecture:** Replaces standard dense Feed-Forward Networks (FFNs) with Gated KAN layers using learnable 1D quadratic spline basis functions ($\text{ReLU}^2$).
* **Engineered for 11 GB VRAM:** Uses unified end-to-end backpropagation with activation checkpointing to fit 124M parameters and a 1024-token context inside **~4.8 GB VRAM**, leaving 6+ GB of headroom.
* **Zero-RAM Data Pipeline:** Implements streaming binary chunking directly to memory-mapped files (`np.memmap`), allowing you to tokenize and train on **2.5+ Billion tokens** while keeping host system RAM usage strictly under **350 MB** (eliminating Python `MemoryError` crashes).
* **Deterministic RAG Precision:** Overcomes the factual memory limits of sub-billion parameter models by pairing the KAN generator with **FAISS** vector search and **all-MiniLM-L6-v2** embeddings.
* **Pre-Flight Diagnostics:** Includes `train_advisor.py` to calculate exact memory footprints, parameter breakdowns, Chinchilla scaling ratios, and runtime estimates before launching.

---

## Model Specifications

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **124,226,560 (124.2M)** | Fully tied input/output embedding weights |
| **Hidden Dimension ($d$)** | **640** | Multiple of 64 for optimal Tensor Core MMA tiling |
| **Layers** | **14 Transformer-KAN Blocks** | End-to-end unified backpropagation |
| **Attention Heads** | **10 Heads ($d_{\text{head}} = 64$)** | Flash Attention / PyTorch SDPA with RoPE |
| **Spline Grid ($k$)** | **4 Basis Functions** | $\text{ReLU}(\tanh(x) - \text{grid})^2$ across $[-1.5, 1.5]$ |
| **Context Length** | **1,024 Tokens** | Rotary Position Embeddings (RoPE up to 2048) |
| **Vocabulary Size** | **32,768** | Custom Byte-Level BPE trained on educational text |
| **Pretraining Data** | **HuggingFaceFW/fineweb-edu** | `sample-10BT` filtered educational web text |
| **Precision** | **Mixed Precision FP16** | PyTorch Native AMP + Fused AdamW |
| **Peak Training VRAM** | **~4.8 GB** | Measured on NVIDIA GeForce RTX 2080 Ti (11 GB) |

---

## Project Structure

```text
ReLU-KAN-RAG-LLM/
│
├── train_kan_llm_e2e.py    # Main 124M End-to-End pretraining script with CLI
├── train_advisor.py        # Pre-flight diagnostic, VRAM budget & runtime calculator
├── finetune_rag.py         # SFT instruction-tuning script for in-context extraction
├── rag_inference.py        # Complete RAG engine (FAISS + MiniLM + KAN Generator)
│
├── config_e2e.json         # Effective configuration file (auto-generated)
├── requirements.txt        # Pinned dependencies with CUDA 12.1 index
├── .gitignore              # Ignores large checkpoints and binary token caches
└── README.md               # Project documentation
```

---

## Installation & Environment Setup

### 1. Prerequisites
* **OS:** Windows 10/11 (PowerShell) or Linux (Ubuntu 22.04+)
* **GPU:** NVIDIA GPU with $\ge 8$ GB VRAM (RTX 2080 Ti, 3070, 3080, 4070+)
* **Python:** 3.10, 3.11, or 3.12 (Python 3.13 is **not** supported by CUDA wheels)

### 2. Setup Virtual Environment
```powershell
# Clone or create directory
cd D:\Projects
mkdir KAN-LLM
cd KAN-LLM

# Create clean virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux:
# source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify CUDA Installation
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"
```

---

## Step-by-Step Workflow

### Step 1: Run Pre-Flight Diagnostics
Before launching a long training run, inspect your hardware headroom, parameter counts, and estimated runtime:

```powershell
python train_advisor.py
```

Example advisor output:
```text
===========================================================================
           RELU-KAN LLM PRE-FLIGHT TRAINING ADVISOR
===========================================================================
Hardware Detected      : NVIDIA GeForce RTX 2080 Ti (11,264 MB physical VRAM)
Model Configuration    : dim=640 | layers=14 | k=4 | context=1024
Total Parameters       : 124,226,560 (124.2M params, tied weights)
Estimated Peak VRAM    : ~4,846 MB (4.73 GB)
VRAM Headroom Available: ~6,418 MB
Tokens Per Step        : 16,384 tokens (8 micro-batch × 2 grad-accum)
Total Tokens Processed : 655,360,000 tokens (40,000 steps)
Token/Parameter Ratio  : 5.28x
Estimated Runtime      : ~24.2 hours (@ 7,500 tok/s)
[advisor check] VRAM Budget: EXCELLENT. Comfortable headroom, zero paging risk.
===========================================================================
```

---

### Step 2: Pretraining on FineWeb-Edu

The pretraining engine streams educational web pages from Hugging Face, trains a custom 32k BPE tokenizer, writes binary `int32` token arrays directly to disk in chunks, and trains the model end-to-end.

```powershell
# Launch default 24-hour marathon (~655M tokens, 40,000 steps)
python train_kan_llm_e2e.py

# Or launch with custom CLI switches:
python train_kan_llm_e2e.py --steps 40000 --batch-size 8 --grad-accum 2 --save-config
```

#### CLI Switches Available:
* `--steps INT`: Total training steps.
* `--batch-size INT`: Micro-batch size per forward pass.
* `--grad-accum INT`: Gradient accumulation steps (Effective batch = `batch-size * grad-accum`).
* `--lr FLOAT`: Base learning rate (default: `3e-4`).
* `--min-lr FLOAT`: Final cosine decayed learning rate (default: `3e-5`).
* `--max-articles INT`: Number of articles to stream and cache (e.g. `2500000` for full Chinchilla).
* `--fresh`: Ignore existing checkpoints and start clean from scratch.
* `--save-config`: Save effective CLI switches back into `config_e2e.json`.

#### Resuming Interrupted Runs:
The script automatically serializes `model_state`, `optimizer_state`, `scaler_state`, and **full RNG states (Python, Torch, CUDA)** into `checkpoints_e2e/kan_e2e_124m_latest.pt`. Re-running the script automatically resumes training forward without repeating random token slices.

---

### Step 3: Instruction Tuning for RAG (SFT)

A base pre-trained model knows grammar and language structure, but treats prompts as continuous essays. **Supervised Fine-Tuning (SFT)** teaches the 12-layer attention stack how to inspect a `Context:`, identify the matching clause for a `Question:`, and extract the factual `Answer:`.

```powershell
python finetune_rag.py
```

* Streams 12,000 question-answer passages from **`rajpurkar/squad`**.
* Evaluates on a held-out validation set every 100 steps.
* Automatically saves the checkpoint at the point of **minimum validation loss** to `checkpoints/kan_model_rag_instruct.pt`, preventing memorization and overfitting.

---

### Step 4: Interactive RAG Inference Engine

Run the integrated RAG engine to query your knowledge base:

```powershell
python rag_inference.py
```

#### How the Pipeline Works at Runtime:
1. **Query Ingestion:** User inputs a natural language question.
2. **Embedding:** `all-MiniLM-L6-v2` embeds the query into a 384-dimensional vector in ~5ms.
3. **FAISS Search:** Fast inner-product vector similarity retrieves the top-1 or top-2 relevant document chunks.
4. **Context Injection:** Formats the prompt cleanly within the 1024-token budget:
   ```text
   Context: {retrieved_knowledge_passage}
   Question: {user_query}
   Answer:
   ```
5. **Autoregressive Generation:** The 124M ReLU-KAN model generates the factual answer using greedy decoding (`argmax`), stopping cleanly upon generating `<|endoftext|>`.

---

## Technical Deep-Dive & Architecture Decisions

### Why ReLU-KAN?
Traditional Kolmogorov-Arnold Networks use B-splines. While mathematically elegant, B-splines are notoriously slow on modern GPUs because spline basis recursion creates irregular memory accesses.

**ReLU-KAN** replaces B-splines with an outer-product basis expansion:
$$\phi_i(x) = \text{ReLU}\big(\tanh(x) - g_i\big)^2$$
Where $\{g_i\}_{i=1}^k$ are fixed, linearly spaced grid knots. This reformulates the non-linear curve fitting into standard GPU-friendly matrix operations:
$$\text{Input} \longrightarrow \text{Tanh} \longrightarrow \text{ReLU} \longrightarrow \text{Square} \longrightarrow \text{Linear Matmul}$$
This delivers the continuous non-linear expressiveness of KANs while maximizing Tensor Core utilization on Turing/Ampere architectures.

### The Memory-Mapped (`np.memmap`) Streaming Pipeline
Standard PyTorch datasets load tokens into memory as lists or tensors. For a Chinchilla-compliant dataset (2.5 Billion tokens $\times$ 4 bytes per `int32`), raw storage in RAM requires **over 10 GB of continuous RAM**, plus duplicate buffers during dataset creation, triggering `MemoryError` on 16 GB and 32 GB machines.

`train_kan_llm_e2e.py` solves this using memory-mapped binary disk arrays:
1. Articles stream in small bursts of 4,000.
2. Each burst is encoded and appended directly as raw bytes to `fineweb_train.bin`.
3. Memory is immediately cleared via Python garbage collection (`gc.collect()`).
4. Training uses `np.memmap`, which maps the multi-gigabyte binary file into virtual address space. PyTorch reads 32 KB slices on-the-fly, keeping host system RAM usage **under 350 MB**.

---

## Hardware Sizing & Chinchilla Scaling

According to Chinchilla scaling laws ($D \approx 20 \times N$):
* **Model Parameters ($N$):** $124.23 \times 10^6$
* **Chinchilla Optimal Token Count ($D$):** $\approx \mathbf{2.48 \text{ Billion tokens}}$
* **Documents in FineWeb-Edu:** $\approx \mathbf{2.5 \text{ Million articles}}$

### Training Timeline Options on an RTX 2080 Ti (~7,500 tok/s):
* **1-Day Training Run (Recommended Starting Point):**
  * `max_articles: 700000` $\longrightarrow$ **~655M tokens** (40,000 steps) $\approx$ **24.2 hours** (~5.3 tokens/param).
  * Validates the architecture, achieves solid convergence (Val Loss ~2.2 – 2.4), and produces a highly functional model.
* **Full Chinchilla Run:**
  * `max_articles: 2500000` $\longrightarrow$ **~2.48B tokens** (152,000 steps) $\approx$ **92 hours** (~3.8 days).
  * Reaches full compute-optimal convergence (Val Loss ~1.9 – 2.1).

---

## License

This project is released under the **MIT License**. Pretrained weights and datasets inherit licensing terms from [HuggingFace FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).