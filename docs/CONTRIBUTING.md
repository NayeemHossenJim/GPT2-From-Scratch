# Contributing

Thank you for your interest in contributing! This guide explains how to get started.

## Code of Conduct

Be respectful and inclusive. We value all contributions regardless of experience level.

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/LLM-From-Scratch.git
cd LLM-From-Scratch
```

### 2. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/my-improvement
```

### 3. Set Up Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 4. Make Changes

Edit code in `src/3_compact_gpt_architecture.py` or create new files in `docs/`.

### 5. Test Your Changes

Run the demo to verify nothing is broken:

```bash
python src/3_compact_gpt_architecture.py
```

Test with modified hyperparameters:

```python
# Create a small test script
model = GPTModel({
    "vocab_size": 200_000,
    "context_length": 256,
    "emb_dim": 128,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
})
# Quick forward pass
x = torch.randint(0, 200_000, (1, 100))
out = model(x)
assert out.shape == (1, 100, 200_000), "Shape mismatch"
print("✓ Test passed")
```

### 6. Commit Changes

```bash
# Stage changes
git add src/3_compact_gpt_architecture.py docs/NEWFILE.md

# Commit with clear message
git commit -m "Add feature: clearer attention documentation"
```

**Commit message tips**:
- Start with verb: "Add", "Fix", "Improve", "Refactor"
- Be specific: "Add positional encoding visualization" (not "update code")
- Reference issues if applicable: "Fixes #123"

### 7. Push and Create PR

```bash
git push origin feature/my-improvement
```

Visit GitHub and create a Pull Request. Fill in the template.

## Types of Contributions

### Documentation

Improve explanations, fix typos, add examples, create new guides.

**Good first contribution**: Fix unclear wording in [ARCHITECTURE.md](ARCHITECTURE.md) or [CONCEPTS.md](CONCEPTS.md).

Example:
```markdown
# Before
"The attention mechanism uses Q, K, V matrices."

# After
"The attention mechanism projects input to Query (Q), Key (K), and Value (V) matrices,
then computes weighted averages of V based on similarity between Q and K."
```

### Code Improvements

- Add docstrings to undocumented functions
- Improve code readability
- Add helpful comments at complex parts
- Optimize performance

**Example**: Add more comments to the attention forward pass.

### Bug Fixes

Found a bug? Create an issue first describing:
1. What's broken
2. How to reproduce it
3. Expected vs actual behavior

Then create a PR that fixes it.

### New Features

**Before implementing**: Open an issue to discuss if it fits the project scope.

Examples of good features:
- ✅ Add sampling mode for text generation
- ✅ Add gradient checkpointing for memory efficiency
- ✅ Add evaluation metrics (perplexity, etc.)
- ✅ Refactor into modular file structure

Examples of out-of-scope:
- ❌ Implement entire training loop (not educational focus)
- ❌ Add distributed training
- ❌ Support for transformers other than GPT-style

### Educational Content

- Interactive notebooks explaining concepts
- Visualization of attention weights
- Comparison with other implementations
- Performance benchmarks

## Code Style

### Python

Keep code simple and readable:

```python
# Good: clear variable names
queries = self.W_query(x)

# Avoid: cryptic names
q = self.w_q(x)

# Good: self-documenting
attn_scores = queries @ keys.transpose(-2, -1)

# Avoid: unclear
s = q @ k.T
```

### Docstrings

Use Google-style docstrings:

```python
def forward(self, x):
    """Process input through attention.

    Args:
        x: Input tensor of shape (batch, seq_len, d_model)

    Returns:
        Attention output of shape (batch, seq_len, d_model)
    """
    ...
```

### Comments

- **Explain why, not what**: Code shows what, comments explain why
- **Keep brief**: One line max unless essential
- **No obvious comments**: Skip obvious statements

```python
# Good: explains non-obvious choice
self.register_buffer("mask", ...)  # Causal mask prevents attending to future tokens

# Avoid: obvious
x = x + 1  # Add 1 to x

# Good: explains tricky line
attn_scores.masked_fill_(mask_bool, -torch.inf)  # Set future positions to -inf for softmax
```

## Testing Your Changes

### Manual Testing

```bash
python src/3_compact_gpt_architecture.py
```

Verify:
- ✅ Script runs without errors
- ✅ Output looks reasonable
- ✅ No warnings or deprecations

### Testing Modifications

If you modify model code:

```python
import torch
from src.model import GPTModel

cfg = {"vocab_size": 200_000, "context_length": 512, ...}
model = GPTModel(cfg)
model.eval()

# Test forward pass
x = torch.randint(0, 200_000, (2, 256))
output = model(x)
assert output.shape == (2, 256, 200_000), f"Shape mismatch: {output.shape}"

# Test generation
from src.model import generate_text_simple
generated = generate_text_simple(model, x, max_new_tokens=10, context_size=512)
assert generated.shape[1] == x.shape[1] + 10, "Generation length mismatch"

print("✓ All tests passed")
```

## PR Review Process

When you open a PR:

1. **GitHub Actions** run (if configured) — checks formatting, basic tests
2. **Maintainers review** — provide feedback
3. **Iterate** — make requested changes
4. **Merge** — once approved!

### PR Checklist

Before submitting:

- [ ] Code follows style guidelines
- [ ] Docstrings added/updated
- [ ] Changes don't break existing functionality
- [ ] Commit messages are clear
- [ ] Related issues/PRs linked (if applicable)
- [ ] No large unrelated changes bundled in one PR

## Big Picture

### Philosophy

This project prioritizes:
1. **Clarity** — Code should be understandable to learners
2. **Correctness** — Implementation should be accurate
3. **Completeness** — Docs should explain concepts

### Scope

**In scope**:
- Educational improvements (clearer explanations, better comments)
- Bug fixes
- Code clarity refactors
- Model variants (different sizes, architectures)
- Evaluation metrics
- Training utilities

**Out of scope**:
- Full training loops (not the focus)
- Distributed training
- Production optimizations
- Support for other model architectures (we're teaching GPT specifically)

## Questions?

- Check [FAQ.md](FAQ.md) for common questions
- Open an issue to discuss
- Comment on related PRs/issues

## Recognition

Contributors are recognized in:
- ✅ Project README (for significant contributions)
- ✅ Commit history (everyone)
- ✅ Pull request (acknowledged)

---

**Thank you for contributing!** 🎉

Whether it's a typo fix or major feature, all contributions make this better for learners.
