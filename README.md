# ⟫ LLM From Scratch

> **Building Large Language Models from First Principles**

An educational implementation of a GPT-style language model in **PyTorch**, designed to teach how transformers and LLMs work through clean, understandable code and structured learning materials.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**⚡ PyTorch** | **🐍 Python 3.8+** | **📓 Jupyter** | **⎇ Git** | **ⓘ MIT License**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Quick Navigation

▶ **[What Is This?](#what-is-this)** — The project in one sentence  
▶ **[Quick Start](#quick-start)** — Get running in 2 minutes  
▶ **[Learning Paths](#learning-paths)** — Find your entry point  
▶ **[Documentation](#documentation)** — Comprehensive guides  
▶ **[Architecture](#architecture)** — How the model works  

---

## What Is This?

A **ground-up implementation of a GPT-3 style transformer language model** that teaches how modern LLMs work internally. Every component is explicit, well-documented, and built for understanding.

**Core Components:**  
⊕ **Tokenization & Embeddings** — Converting text into learnable vectors  
⊙ **Self-Attention Mechanism** — The foundation of transformer models  
⟡ **Transformer Blocks** — Stacked layers of attention + feed-forward networks  
◈ **Text Generation** — Inference and decoding strategies  

**Ideal for:**  
▶ 🎓 Students learning transformer architecture  
▶ 🔬 ML practitioners exploring language models  
▶ 📊 Researchers studying interpretable implementations  
▶ 💼 Developers building custom LLM applications  

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/NayeemHossenJim/LLM-From-Scratch.git
cd LLM-From-Scratch

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the model
python src/compact_gpt_architecture.py
```

**Next Steps:**  
▶ Read [Getting Started](docs/GETTING_STARTED.md) for detailed setup instructions  
▶ Choose a [Learning Path](#learning-paths) below based on your style  
▶ Explore [Architecture Overview](docs/ARCHITECTURE.md) for deep dives

---

## Learning Paths

Choose how you want to learn. Different paths for different learning styles:

### 🔷 Visual Learners: Architecture First  
Start by understanding the big picture and how components connect.  
▶ **Entry Point:** [Architecture Overview](docs/ARCHITECTURE.md)  
▶ **Then:** [Concepts Explained](docs/CONCEPTS.md) → [Component Deep Dives](docs/COMPONENTS.md)  

### 📐 Theory-Focused: Deep Dive into Concepts  
Understand the mathematical foundations and why transformers work.  
▶ **Entry Point:** [Transformer Concepts](docs/CONCEPTS.md)  
▶ **Then:** [Architecture](docs/ARCHITECTURE.md) → [Read Papers](#key-references) → [Code](src/compact_gpt_architecture.py)  

### 💻 Hands-On Coders: Learn by Doing  
Run code immediately and modify as you go.  
▶ **Entry Point:** Launch Jupyter notebooks in `src/` and `test/`  
▶ **Then:** [API Reference](docs/API_REFERENCE.md) → [Examples](docs/EXAMPLES.md)  

### ⟫ API Users: Just Show Me the Code  
Go straight to the implementation and reference docs.  
▶ **Entry Point:** [src/compact_gpt_architecture.py](src/compact_gpt_architecture.py) (348 lines)  
▶ **Reference:** [API Documentation](docs/API_REFERENCE.md) → [Usage Examples](docs/EXAMPLES.md)  

---

## Architecture

**The Model Pipeline:**
```
Input (tokens) 
    │
    ├─→ ⊕ Token + Position Embeddings
    │
    ├─→ ⟡ [12 Transformer Blocks]
    │       ├─ ⊙ Multi-Head Self-Attention (12 heads)
    │       ├─ ◈ Causal Masking (can't look ahead)
    │       └─ ▲ Position-wise Feed-Forward Network
    │
    ├─→ ◇ Layer Normalization
    │
    └─→ Output (logits for next token prediction)
```

**Model Specifications:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **⟫ Vocabulary Size** | 200K tokens | Using tiktoken o200k_base encoding |
| **◇ Context Length** | 1024 tokens | Maximum input sequence length |
| **⊕ Hidden Dimension** | 768 | Feature representation size |
| **⊙ Attention Heads** | 12 | Multi-head attention mechanism |
| **⟡ Layers** | 12 | Transformer blocks stacked |
| **◈ Dropout** | 0.1 | Regularization during training |
| **⚡ Approx. Parameters** | ~125M | Similar to GPT-2 Small |

**Key Design Choices:**  
⊙ **Pre-norm architecture** — Layer norm before attention/FFN (more stable)  
⊕ **Learnable embeddings** — Position embeddings are learned, not fixed  
◈ **Causal masking** — Prevents attending to future tokens  
▲ **Residual connections** — Skip connections between layers  

See [detailed architecture guide](docs/ARCHITECTURE.md) for complete explanations.

---

## Project Structure

```
LLM-From-Scratch/
│
├── src/                                  ⟫ Source Code
│   ├── compact_gpt_architecture.py       ⚡ Main implementation (348 lines)
│   ├── 1_dataloader.ipynb                📓 Data handling tutorial
│   ├── 2_multihead_attention.ipynb       ⊙ Attention mechanism
│   └── __init__.py
│
├── test/                                 🔬 Testing & Experiments
│   ├── Dummy_GPT.ipynb                   ⟡ Model testing
│   ├── custom_gpt_training_notebook.ipynb 🔥 Training examples
│   ├── simple_self_attention.ipynb       ⊙ Attention demos
│   ├── tokenizer.ipynb                   ◇ Tokenization
│   └── the-verdict.txt                   📄 Training data (465KB)
│
├── docs/                                 📚 Documentation
│   ├── GETTING_STARTED.md                ▶ Setup & installation
│   ├── ARCHITECTURE.md                   ⟡ Model design deep-dive
│   ├── COMPONENTS.md                     ⊕ Individual class docs
│   ├── LEARNING_PATH.md                  🗺️ Structured learning
│   ├── CONCEPTS.md                       📐 Transformer math
│   ├── API_REFERENCE.md                  ◇ Complete API
│   ├── EXAMPLES.md                       💡 Usage patterns
│   ├── CONTRIBUTING.md                   🤝 Development guide
│   ├── FAQ.md                            ❓ Common questions
│   └── _assets/                          🖼️ Documentation images
│
├── extra/                                📖 Supplementary
│   ├── Appendix_A.ipynb                  📐 Tensor fundamentals
│   └── ...
│
├── README.md                             ⟫ This file
├── requirements.txt                      🐍 Python dependencies
├── LICENSE                               ⓘ MIT License
└── .venv/                                🔧 Virtual environment
```

---

## Documentation

Complete learning materials organized by topic and learning style:

| Document | Purpose | Best For |
|----------|---------|----------|
| **▶ Getting Started** | [Link](docs/GETTING_STARTED.md) | Installation, setup, first run | New users |
| **⟡ Architecture** | [Link](docs/ARCHITECTURE.md) | Model design and component overview | Visual learners |
| **🗺️ Learning Path** | [Link](docs/LEARNING_PATH.md) | Structured learning routes | Planning your study |
| **📐 Concepts** | [Link](docs/CONCEPTS.md) | Transformer fundamentals and math | Theory-focused learners |
| **⊕ Components** | [Link](docs/COMPONENTS.md) | Deep dive into each class and module | Code readers |
| **◇ API Reference** | [Link](docs/API_REFERENCE.md) | Complete function and class documentation | Developers building on this |
| **💡 Examples** | [Link](docs/EXAMPLES.md) | Usage patterns and code samples | Hands-on learners |
| **🤝 Contributing** | [Link](docs/CONTRIBUTING.md) | Development guidelines and standards | Contributors |
| **❓ FAQ** | [Link](docs/FAQ.md) | Common questions and troubleshooting | When stuck |

---

## Dependencies

**⚡ Core ML Framework:**  
⚡ **PyTorch** 2.4.0 — Deep learning framework for neural networks  

**🐍 Python Utilities:**  
🐍 **Python** 3.8+ — Core language  
◇ **tiktoken** 0.12.0 — OpenAI's tokenizer (o200k_base encoding)  
🔢 **NumPy** 1.26.4 — Numerical computing  

**📓 Interactive Learning:**  
📓 **Jupyter** — Notebooks for step-by-step exploration  
📓 **IPython** — Enhanced interactive Python shell  

**🖼️ Visualization:**  
📊 **Matplotlib** — Plotting and visualization  
🖼️ **Pillow** — Image processing  

See [requirements.txt](requirements.txt) for complete list.

---

## Key References

**📄 Papers:**  
▶ [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — ⟡ Original Transformer architecture  
▶ [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1906.04341) (2019) — ⟫ GPT-2 foundation  
▶ [Pre-Norm vs Post-Norm](https://arxiv.org/abs/2002.07239) — ⊙ Training stability research  

**🔗 Resources:**  
▶ ⚡ [PyTorch Documentation](https://pytorch.org/docs/) — Framework reference  
▶ 🤖 [Hugging Face Transformers](https://huggingface.co/transformers/) — Production library  
▶ 📺 [3Blue1Brown Attention](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R4_67m3IWEibEi8ecQc) — Visual explanation  

---

## Contributing

Found a bug or want to improve the documentation? Contributions are welcome!

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for:  
▶ 🔧 Development setup  
▶ 📝 Code style guidelines  
▶ 🧪 Testing requirements  
▶ 🤝 How to submit changes  

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

You're free to use, modify, and distribute this code for educational and commercial purposes.
