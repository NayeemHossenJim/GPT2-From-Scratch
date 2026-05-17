# FAQ

Frequently asked questions and answers.

## Installation & Setup

### Q: How do I install the project?

A: Follow the [Getting Started](GETTING_STARTED.md) guide. Quick version:

```bash
git clone <repo>
cd LLM-From-Scratch
python -m venv .venv
source .venv/bin/activate  # macOS/Linux, or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Q: What Python version do I need?

A: Python 3.8+. Check with `python --version`.

### Q: I get "ModuleNotFoundError: No module named 'torch'"

A: You haven't activated your virtual environment or haven't installed dependencies.

```bash
source .venv/bin/activate  # Activate venv
pip install -r requirements.txt
```

### Q: Can I use this on Windows?

A: Yes! Use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

### Q: Do I need a GPU?

A: No, CPU works fine. GPU is optional for faster inference. If you have GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: How much memory does this use?

A: Roughly 2-3GB for the full model (768-dim embeddings, 12 layers). Smaller configs use much less.

## Understanding the Code

### Q: What's a transformer?

A: A deep neural network architecture using attention mechanisms instead of recurrence. See [ARCHITECTURE.md](ARCHITECTURE.md).

### Q: What's attention?

A: A mechanism that computes weighted combinations of values based on query-key similarity. See [CONCEPTS.md](CONCEPTS.md).

### Q: What's causal masking?

A: Prevents the model from looking at future tokens during training/generation. Essential for language modeling. See [ARCHITECTURE.md#causal-masking](ARCHITECTURE.md#causal-masking).

### Q: Why 12 heads?

A: Allows the model to attend to different types of relationships simultaneously. Each head learns different patterns. See [ARCHITECTURE.md#multi-head-attention](ARCHITECTURE.md#multi-head-attention).

### Q: What does "pre-norm" mean?

A: Layer normalization applied before (pre) each sub-layer, not after (post). Improves stability in deep networks.

### Q: Why does the FFN expand to 4x then contract?

A: Expansion allows learning more complex non-linear transformations. 4x is conventional from the original transformer paper.

## Modifying the Model

### Q: How do I make a smaller model?

A: Reduce `emb_dim`, `n_heads`, `n_layers`, or `context_length`:

```python
cfg = {
    "vocab_size": 200_000,
    "context_length": 256,    # was 1024
    "emb_dim": 128,           # was 768
    "n_heads": 2,             # was 12
    "n_layers": 2,            # was 12
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

### Q: How do I make a larger model?

A: Increase `emb_dim`, `n_heads`, `n_layers`. Warning: much slower!

```python
cfg = {
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    ...
}
```

### Q: What if I change the context length?

A: Shorter = faster but limited long-range reasoning. Longer = slower but better for long-range dependencies.

### Q: Can I change the vocab size?

A: Yes, but it needs to match your tokenizer. The default is 200K (tiktoken's o200k_base).

### Q: How do I add a new layer type?

A: Create a new class inheriting `nn.Module` and add it to the transformer blocks. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Training & Fine-tuning

### Q: Can I train this model?

A: The code currently only does inference. Training would require:
- Defining a loss function (next-token prediction)
- Setting up an optimizer
- Implementing training loop

This is not in scope for this educational project, but you can add it!

### Q: Can I fine-tune on custom data?

A: Not built-in, but you can use the `GPTDataset` class to tokenize your data and write a training loop. See [EXAMPLES.md](EXAMPLES.md).

### Q: Where do I get training data?

A: Common sources:
- WikiText-103
- The Pile
- Common Crawl
- Project Gutenberg (free books)

### Q: How long does training take?

A: Depends on data size, model size, and compute. This 125M-param model on 1 GPU: weeks to months for meaningful results.

## Text Generation

### Q: Why is the output sometimes nonsensical?

A: The model hasn't been trained on real data (only initialized with random weights). Train it first!

### Q: Can I use sampling instead of greedy?

A: Yes! See [EXAMPLES.md#sampling](EXAMPLES.md#example-8-sampling-instead-of-greedy).

### Q: How do I control output diversity?

A: Use temperature:
- `temperature=0.5` → More deterministic
- `temperature=1.0` → Default (natural)
- `temperature=2.0` → More random

### Q: Why is generation slow?

A: Generating one token requires a full forward pass through the model. 12 layers × 12 attention heads = lots of computation. GPU helps significantly.

### Q: Can I generate multiple texts in parallel?

A: Yes, batch multiple prompts. See [EXAMPLES.md#batching](EXAMPLES.md#example-2-batch-generation).

## Performance & Optimization

### Q: Is this optimized?

A: No, it prioritizes clarity over performance. Production implementations (Hugging Face, etc.) have many optimizations.

### Q: How do I make it faster?

A: Options:
- Use a smaller model (fewer layers/heads)
- Use a GPU
- Use batch processing
- Use float16 precision (GPU only): `model.half()`

### Q: What's the inference speed?

A: Depends on hardware. Rough estimates:
- CPU (1 token): 100-500ms
- GPU (1 token): 10-50ms
- Batch of 32 on GPU: Better amortization

### Q: Can I use this in production?

A: Not recommended without modification. It's designed for learning. For production, use optimized libraries (Hugging Face Transformers, vLLM, etc.).

## Comparison with Other Projects

### Q: How does this compare to Hugging Face Transformers?

A: This is ~300 lines of clear, educational code. Hugging Face is a production library with 100K+ lines. Different goals:

| | This project | Hugging Face |
|--|---|---|
| **Goal** | Learn how it works | Use pre-built models |
| **Code** | 300 lines, clear | 100K+ lines, optimized |
| **Speed** | Slow (for learning) | Fast (optimized) |
| **Features** | Minimal | Comprehensive |

### Q: How does this compare to Pythia?

A: Pythia is a suite of models for interpretability research. This is a single, simple model for learning. Pythia is much more complex.

### Q: Can I use this code as a baseline?

A: Absolutely! It's designed for that. Modify, extend, and learn from it.

## Common Errors

### Q: "RuntimeError: Expected all tensors to be on the same device"

A: Your model and data are on different devices. Fix:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

### Q: "RuntimeError: CUDA out of memory"

A: Your model doesn't fit on GPU. Options:
- Use smaller model
- Reduce batch size
- Use CPU instead

### Q: "AssertionError: d_out must be divisible by num_heads"

A: Your `emb_dim` must be divisible by `n_heads`. For example:
- `emb_dim=768, n_heads=12` → 768/12=64 ✓
- `emb_dim=100, n_heads=12` → 100/12 ≠ integer ✗

### Q: Output is all zeros or NaNs

A: Usually a bug in forward pass. Check:
- Input shape assumptions
- Gradient flow (if training)
- Numerical stability (very large/small numbers)

## Learning Questions

### Q: How long should I spend on this?

A: Depends on learning path:
- Quick overview: 2-3 hours
- Deep understanding: 1-2 weeks
- Master implementation: ongoing

### Q: What should I read first?

A: See [LEARNING_PATH.md](LEARNING_PATH.md) for three options.

### Q: What if I'm stuck?

A: Try:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md)
2. Check [CONCEPTS.md](CONCEPTS.md)
3. Read code comments
4. Modify code and test hypotheses
5. Ask on GitHub Discussions

### Q: Where can I learn more?

A: Resources:
- Read papers: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Watch videos: 3Blue1Brown's Transformers series
- Study: Hugging Face course (huggingface.co/course)

### Q: What are next steps after this project?

A: Options:
- Add training loop
- Implement other architectures (BERT, T5)
- Work on Hugging Face models
- Read research papers
- Contribute to open source

## Project Questions

### Q: Can I use this code commercially?

A: Check the [LICENSE](../LICENSE) file. Currently MIT, so yes with attribution.

### Q: Can I contribute?

A: Yes! See [CONTRIBUTING.md](CONTRIBUTING.md).

### Q: Is this actively maintained?

A: Yes, contributions and issues are welcomed.

### Q: Can I distribute this?

A: Under MIT license, yes. Include the license and attribution.

---

## Didn't find your answer?

- Open an [issue](https://github.com/yourusername/LLM-From-Scratch/issues)
- Start a [discussion](https://github.com/yourusername/LLM-From-Scratch/discussions)
- Check other docs: [ARCHITECTURE.md](ARCHITECTURE.md), [CONCEPTS.md](CONCEPTS.md)

**Happy learning!** 🚀
