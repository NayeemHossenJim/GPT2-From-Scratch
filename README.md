# 🚀 LLM From Scratch

> **Building Large Language Models from First Principles**

An educational implementation of a GPT-style language model in **PyTorch**, designed to teach how transformers and LLMs work through clean, understandable code and step-by-step learning materials.

---

## 📋 Table of Contents

- [🎯 What Is This?](#-what-is-this)
- [⚡ Quick Start](#-quick-start)
- [📚 Documentation](#-documentation)
- [🏗️ Architecture](#️-architecture)
- [📂 Project Structure](#-project-structure)
- [🔗 Key Resources](#-key-resources)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 What Is This?

This project implements a GPT-3 style language model from first principles using PyTorch. Each component is explicit and educational:

- **Tokenization & Embeddings**: Converting text to vectors
- **Self-Attention**: The core mechanism that makes transformers work
- **Transformer Blocks**: Stacking attention and feed-forward layers
- **Text Generation**: Inference and decoding strategies

Perfect for:
- 🎓 Learning how LLMs work internally
- 🔍 Understanding transformer architecture
- 💻 Studying clean, readable ML code
- 🚀 Building custom language models

---

## ⚡ Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/LLM-From-Scratch.git
cd LLM-From-Scratch
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the demo
python src/3_compact_gpt_architecture.py
```

**Next steps:**
- 📖 Read [Getting Started](docs/GETTING_STARTED.md) for detailed setup
- 🗺️ Pick a [Learning Path](docs/LEARNING_PATH.md) based on your style
- 📚 Explore [Architecture Overview](docs/ARCHITECTURE.md)

---

## 📚 Documentation

Complete documentation is available in the `docs/` directory:

| Doc | Purpose |
|-----|---------|
| [Getting Started](docs/GETTING_STARTED.md) | Installation and first run |
| [Architecture](docs/ARCHITECTURE.md) | Model design and overview |
| [Learning Path](docs/LEARNING_PATH.md) | Structured learning routes |
| [Components](docs/COMPONENTS.md) | Deep dives into each module |
| [Concepts](docs/CONCEPTS.md) | Fundamental LLM/transformer concepts |
| [API Reference](docs/API_REFERENCE.md) | Formal class and function documentation |
| [Examples](docs/EXAMPLES.md) | Usage patterns and code samples |
| [Contributing](docs/CONTRIBUTING.md) | How to contribute and develop |
| [FAQ](docs/FAQ.md) | Common questions |

---

## 🏗️ Architecture

**The Model**:
```
Input (tokens) 
  ↓
Token + Position Embeddings
  ↓
[12 Transformer Blocks] ← Attention + Feed-Forward
  ↓
Layer Normalization
  ↓
Output (logits for next token)
```

**Key Features**:
- Multi-head self-attention (12 heads)
- Causal masking (can't look ahead)
- Pre-norm residual connections
- Position-wise feed-forward networks
- Learnable embeddings

**Configuration**:
- Vocabulary: 200K tokens (tiktoken)
- Context length: 1024 tokens
- Hidden dimension: 768
- Layers: 12
- Heads: 12

See [Architecture Guide](docs/ARCHITECTURE.md) for detailed explanation.

---

## 📂 Project Structure

```
LLM-From-Scratch/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
│
├── src/
│   ├── 3_compact_gpt_architecture.py    # Main model implementation
│   ├── 1_dataloader.ipynb               # Educational notebook
│   ├── 2_attention.ipynb                # Educational notebook
│   └── ...
│
├── test/
│   ├── Dummy_GPT.ipynb         # Test notebooks
│   ├── simple_self_attention.ipynb
│   └── tokenizer.ipynb
│
├── extra/
│   └── Appendix_A.ipynb        # Tensor fundamentals
│
├── docs/                        # Comprehensive documentation
│   ├── GETTING_STARTED.md
│   ├── ARCHITECTURE.md
│   ├── LEARNING_PATH.md
│   ├── COMPONENTS.md
│   ├── CONCEPTS.md
│   ├── API_REFERENCE.md
│   ├── EXAMPLES.md
│   ├── CONTRIBUTING.md
│   ├── FAQ.md
│   └── _assets/
│
├── the-verdict.txt             # Sample training data
└── small-text-sample.txt       # Sample training data
```

---

## 🔗 Key Resources

### Learning References

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Original transformer paper
- **[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** - GPT-2 paper
- **[PyTorch Docs](https://pytorch.org/docs/)** - Deep learning framework reference

### Related Projects

- **[Hugging Face Transformers](https://huggingface.co/transformers/)** - Production-grade library
- **[LLaMA](https://ai.meta.com/llama/)** - Meta's open-source LLM
- **[Pythia](https://github.com/EleutherAI/pythia)** - Interpretability suite

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.


