# GPT-Style Transformer from Scratch

> A transparent PyTorch implementation of a decoder-only transformer using GPT-2-Small-like block dimensions.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository implements the core components of a GPT-style language model directly in PyTorch. The goal is to make the transformer pipeline understandable by keeping tokenization, embeddings, causal multi-head self-attention, normalization, feed-forward layers, residual connections, and autoregressive generation explicit in the code.

The current model uses the same number of layers, attention heads, and hidden dimensions commonly associated with GPT-2 Small. However, it uses the `o200k_base` tokenizer and separate input and output embedding matrices, so it is **not an exact GPT-2 Small reproduction or checkpoint-compatible implementation**.

The main script initializes the model with random weights and performs greedy next-token generation. It demonstrates the complete inference path, but meaningful language generation requires training the model.

## What This Project Implements

- Text tokenization with `tiktoken`
- Sliding-window next-token training samples
- Learned token and positional embeddings
- Scaled dot-product causal self-attention
- Multi-head attention with 12 attention heads
- Custom layer normalization and GELU activation
- Pre-normalization transformer blocks
- Residual connections and dropout
- Position-wise feed-forward networks
- Autoregressive text generation using greedy decoding
- Educational notebooks for individual transformer components

## Architecture

```text
Input token IDs
      │
      ▼
Token Embeddings + Positional Embeddings
      │
      ▼
12 × Transformer Blocks
      ├── LayerNorm
      ├── 12-Head Causal Self-Attention
      ├── Residual Connection
      ├── LayerNorm
      ├── Feed-Forward Network (768 → 3072 → 768)
      └── Residual Connection
      │
      ▼
Final LayerNorm
      │
      ▼
Vocabulary Projection
      │
      ▼
Next-Token Logits
```

### Current Model Configuration

| Component | Value |
|---|---:|
| Tokenizer | `tiktoken` `o200k_base` |
| Vocabulary size | 200,019 tokens |
| Context length | 1,024 tokens |
| Hidden dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Feed-forward dimension | 3,072 |
| Dropout | 0.1 |
| QKV bias | Disabled |
| Input/output weight tying | No |
| Generation strategy | Greedy decoding |
| Trainable parameters | 393,043,968 (~393M) |

### Why the Model Has Approximately 393M Parameters

The transformer blocks use GPT-2-Small-like dimensions, but the `o200k_base` vocabulary contains 200,019 tokens. The implementation also uses separate token-embedding and output-projection matrices.

| Parameter group | Parameters |
|---|---:|
| Token embedding | 153,614,592 |
| Output projection | 153,614,592 |
| Transformer blocks, position embeddings, and normalization | 85,814,784 |
| **Total** | **393,043,968** |

The current configuration is therefore a ~393M model. A 125M-class configuration would require a substantially smaller vocabulary and/or shared input-output embedding weights.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/NayeemHossenJim/GPT2-From-Scratch.git
cd GPT2-From-Scratch
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the architecture demo

```bash
python src/compact_gpt_architecture.py
```

The script will:

1. Load the `o200k_base` tokenizer.
2. Initialize the model with random weights using seed `123`.
3. Tokenize an example prompt.
4. Run autoregressive greedy decoding.
5. Print the generated token IDs and decoded text.

> **Important:** Because the model is randomly initialized, the generated text is not expected to be coherent. The demo verifies the architecture and generation pipeline; it does not represent pretrained model quality.

## Hardware Note

The current configuration contains approximately 393M float32 parameters, requiring roughly 1.46 GiB for model weights alone. Training requires additional memory for gradients, activations, and optimizer states. A smaller vocabulary or reduced model configuration is recommended for local experimentation on limited hardware.

## Repository Structure

```text
GPT2-From-Scratch/
├── src/
│   ├── compact_gpt_architecture.py   # End-to-end model and generation demo
│   ├── 1_dataloader.ipynb            # Tokenization and data loading
│   ├── 2_multihead_attention.ipynb   # Attention implementation
│   └── __init__.py
├── test/                              # Experiment and validation notebooks
├── docs/                              # Architecture and learning documentation
├── extra/                             # Supplementary PyTorch material
├── small-text-sample.txt              # Small text sample
├── the-verdict.txt                    # Example training corpus
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Documentation

| Document | Description | Best for |
|---|---|---|
| [Getting Started](docs/GETTING_STARTED.md) | Installation and first run | New users |
| [Architecture](docs/ARCHITECTURE.md) | End-to-end model design | Visual learners |
| [Learning Path](docs/LEARNING_PATH.md) | Suggested study sequence | Structured learning |
| [Concepts](docs/CONCEPTS.md) | Transformer concepts and mathematics | Theory review |
| [Components](docs/COMPONENTS.md) | Explanation of individual modules | Code readers |
| [API Reference](docs/API_REFERENCE.md) | Classes and functions | Developers |
| [Examples](docs/EXAMPLES.md) | Usage examples | Hands-on learning |
| [FAQ](docs/FAQ.md) | Common questions and troubleshooting | Debugging |
| [Contributing](docs/CONTRIBUTING.md) | Contribution workflow | Contributors |

## Core Dependencies

- Python 3.11 or newer recommended
- PyTorch 2.4.0
- `tiktoken` 0.12.0
- NumPy 1.26.4
- Jupyter and IPython for notebooks

See [`requirements.txt`](requirements.txt) for the complete environment.

## Current Scope and Limitations

- The main script demonstrates an untrained, randomly initialized model.
- The repository does not currently provide pretrained weights.
- No benchmark, validation perplexity, or downstream-task score is claimed.
- Greedy decoding is implemented; top-k, top-p, temperature sampling, and beam search are not included in the main script.
- The implementation is educational and is not optimized for production inference.

These limitations are stated explicitly so that the repository represents the implemented work accurately.

## Attribution

This project was developed for educational study using concepts and implementation patterns from Sebastian Raschka's *Build a Large Language Model (From Scratch)* and its [official code repository](https://github.com/rasbt/LLMs-from-scratch).

The original instructional approach and reference implementation are credited to Sebastian Raschka. This repository reorganizes the material into a compact implementation with additional documentation and experiments for learning purposes.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019.
- Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020.
- Sebastian Raschka, [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch).
- [PyTorch Documentation](https://pytorch.org/docs/).
- [tiktoken](https://github.com/openai/tiktoken).

## Contributing

Contributions that improve correctness, testing, documentation, or reproducibility are welcome. See [CONTRIBUTING.md](docs/CONTRIBUTING.md) before opening a pull request.

## License

This repository is distributed under the [MIT License](LICENSE).
