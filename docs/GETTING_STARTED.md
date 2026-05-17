# Getting Started

Welcome! This guide will have you up and running in about 5 minutes.

## Prerequisites

- **Python 3.8+** — Check with `python --version`
- **pip** — Usually comes with Python
- **~2GB disk space** for dependencies
- **Optional**: GPU for faster inference (CUDA 11.8+)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LLM-From-Scratch.git
cd LLM-From-Scratch
```

### 2. Create a Virtual Environment

A virtual environment keeps project dependencies isolated:

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` — PyTorch (deep learning framework)
- `tiktoken` — OpenAI's tokenizer
- `jupyter` — For notebooks (optional but recommended)

**GPU Users**: If you have CUDA, PyTorch may already be GPU-enabled from the pip install. Verify:

```bash
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## Your First Run

### Run the Demo

```bash
python src/3_compact_gpt_architecture.py
```

You should see:
```
Input: "Hello, I am Elon Musk"
Generated: "Hello, I am Elon Musk. I was born in 1971. I am the CEO of Tesla ..."
```

This loads a pre-trained GPT model and generates 10 tokens (words/subwords) of text.

### What Just Happened?

1. The model was initialized with 12 transformer blocks, 12 attention heads, and 768 hidden dimensions
2. Your prompt was tokenized (converted to numbers)
3. The model predicted the next token 10 times
4. The output was decoded back to text

## Understanding the Code

### Main File

The entire model lives in **`src/3_compact_gpt_architecture.py`** (224 lines).

Open it and scan for these classes:

```python
# Data handling
class GPTDataset: ...           # Turns raw text into batches

# Model components
class MultiHeadAttention: ...   # The core of transformers
class TransformerBlock: ...     # Attention + Feed-Forward
class GPTModel: ...             # Stacks everything together

# Utilities
def generate_text_simple(): ... # Creates text from a prompt
def main(): ...                 # Demo script
```

**→ See [Components](COMPONENTS.md) for detailed explanations of each class.**

### Educational Notebooks

If you prefer interactive learning, start with notebooks:

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter
jupyter notebook

# Open any notebook in your browser:
```

- **`extra/Appendix_A.ipynb`** — Tensor fundamentals (if math is new to you)
- **`src/1_dataloader.ipynb`** — How text becomes model inputs
- **`src/2_dataloader_multihead-attention.ipynb`** — Attention mechanism visualized

## Next Steps

### Pick Your Learning Style

**I want to understand the architecture first**
→ Read [Architecture](ARCHITECTURE.md), then [Components](COMPONENTS.md)

**I want to learn step-by-step**
→ Follow [Learning Path](LEARNING_PATH.md)

**I want to modify the model**
→ Jump to [Examples](../docs/EXAMPLES.md) and [API Reference](API_REFERENCE.md)

**I want to understand the concepts deeply**
→ Read [Concepts](CONCEPTS.md)

**I have questions**
→ Check [FAQ](FAQ.md)

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

You haven't activated your virtual environment or installed dependencies.

```bash
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'tiktoken'`

Make sure you've run:

```bash
pip install -r requirements.txt
```

### The script runs but output looks wrong

Make sure you're in the project root directory:

```bash
cd LLM-From-Scratch
python src/3_compact_gpt_architecture.py
```

### Slow inference / low memory warning

The model runs on CPU by default. GPU acceleration is optional:

```python
# In the main file, look for device assignment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## File Structure

Here's what each directory contains:

```
├── src/                 # Main code and notebooks
│   ├── 3_compact_gpt_architecture.py    ← START HERE
│   ├── 1_dataloader.ipynb               ← Educational
│   └── 2_attention.ipynb                ← Educational
│
├── test/               # Test notebooks (optional)
│   ├── Dummy_GPT.ipynb
│   ├── tokenizer.ipynb
│   └── simple_self_attention.ipynb
│
├── extra/              # Appendix materials
│   └── Appendix_A.ipynb                 ← Tensor basics
│
├── docs/               # Documentation you're reading now
│   ├── ARCHITECTURE.md
│   ├── LEARNING_PATH.md
│   ├── COMPONENTS.md
│   ├── API_REFERENCE.md
│   └── ...
│
├── README.md           # Project overview
└── requirements.txt    # Dependencies
```

## Want to Dive Deeper?

**Understand how attention works**
→ [Concepts: Self-Attention](CONCEPTS.md#self-attention-mechanism)

**Learn the full model architecture**
→ [Architecture Guide](ARCHITECTURE.md)

**See code examples**
→ [Examples](EXAMPLES.md)

**Contribute improvements**
→ [Contributing Guide](CONTRIBUTING.md)

## You're All Set!

You now have a working GPT implementation on your machine. The next step depends on your goal:

- **Learning**: Pick a path from [Learning Path](LEARNING_PATH.md)
- **Experimenting**: Jump to [Examples](EXAMPLES.md)
- **Contributing**: Check [Contributing](CONTRIBUTING.md)

Happy learning! 🚀
