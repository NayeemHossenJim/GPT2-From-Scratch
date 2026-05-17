# Architecture

This guide explains the high-level design of the GPT model and how all the pieces fit together.

## Model Overview

The model transforms tokens (words/subwords) into text predictions through several stages:

```
Raw Text
   ↓ (tokenize)
Token IDs: [15496, 11, 314, 716, 28186]
   ↓ (embedding + position encoding)
Dense Vectors: [[0.2, -0.5, ...], [...], ...]
   ↓ (12 Transformer Blocks)
   ├─ Block 1: Attention → Feed-Forward
   ├─ Block 2: Attention → Feed-Forward
   ├─ ...
   └─ Block 12: Attention → Feed-Forward
   ↓ (normalize)
Dense Vectors: [[...], [...], ...]
   ↓ (project to vocab)
Logits: [0.1, 2.3, -0.5, ..., 1.2] (200K scores)
   ↓ (argmax or sampling)
Next Token: 2481 ("beautiful")
   ↓ (decode)
Output Text: "Hello, I am beautiful"
```

## Configuration

The current model uses these settings:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `vocab_size` | 200,000 | Number of unique tokens |
| `context_length` | 1024 | Max sequence length |
| `emb_dim` (d_model) | 768 | Hidden dimension throughout model |
| `n_heads` | 12 | Number of attention heads |
| `n_layers` | 12 | Number of transformer blocks |
| `dropout` | 0.1 | Regularization dropout rate |

These are hyperparameters — you can modify them to create larger/smaller models.

## Component Breakdown

### 1. Tokenization

**Purpose**: Convert raw text into integer sequences

```
"Hello world" → [15496, 1917]
```

**How it works**:
- Uses tiktoken's `o200k_base` encoding (OpenAI's tokenizer)
- Maps bytes → tokens → embeddings
- Handles unknown characters gracefully

**Why**: Neural networks work with numbers, not text. Tokenization creates the right level of granularity (subwords).

### 2. Embeddings

**Token Embeddings** (`emb_dim` × `vocab_size`)
- Learnable lookup table: token ID → dense vector
- Initialized randomly, learned during training
- Example: token 42 → `[0.5, -0.2, 1.1, ...]` (768 dimensions)

**Positional Embeddings** (`emb_dim` × `context_length`)
- Encodes token position in sequence
- Without it, "cat dog" and "dog cat" would look identical
- Learnable or fixed (this implementation uses learnable)

**Combined**:
```python
# For each token
token_vector = token_embed(token_id)      # shape: (emb_dim,)
position_vector = pos_embed(position)     # shape: (emb_dim,)
combined = token_vector + position_vector # shape: (emb_dim,)
```

### 3. Transformer Block

Each block has two sub-layers:

```
Input x
  ↓
[Layer Norm]
  ↓
[Multi-Head Attention]
  ↓
[Residual Add] — adds original input x back
  ↓
[Layer Norm]
  ↓
[Feed-Forward Network] — two linear layers with GELU activation
  ↓
[Residual Add]
  ↓
Output
```

**Pre-norm architecture**: Layer norm applied *before* each sub-layer (not after). This is more stable for deep networks.

**Residual connections**: `output = sublayer(norm(x)) + x`. Allows gradient flow through deep networks.

### 4. Multi-Head Attention

The core innovation that makes transformers powerful.

**Single Head Attention** (simplified):

```
Query Q = x @ W_q    # What we're looking for
Key K = x @ W_k      # What we have
Value V = x @ W_v    # What we retrieve

Scores = Q @ K.T / sqrt(d)  # How similar queries and keys
Weights = softmax(Scores)    # Normalize to probabilities
Output = Weights @ V         # Weighted average of values
```

**Multi-Head**: Run 12 parallel attention operations, each with different weight matrices:

```
Head 1 output: (batch, seq_len, 64)
Head 2 output: (batch, seq_len, 64)
...
Head 12 output: (batch, seq_len, 64)
          ↓
      Concatenate
          ↓
Final output: (batch, seq_len, 768)
```

Why multiple heads?
- Each head learns different relationship types (grammar, semantics, long-range dependencies, etc.)
- Parallel computation is efficient
- Ensemble effect improves robustness

**Causal Masking**: Critical for language modeling. Prevents the model from "cheating" by looking at future tokens:

```
Position:  0 1 2 3 4
Can see:   ✓ ✗ ✗ ✗ ✗  (position 0 sees only itself)
           ✓ ✓ ✗ ✗ ✗  (position 1 sees 0-1)
           ✓ ✓ ✓ ✗ ✗  (position 2 sees 0-2)
           ✓ ✓ ✓ ✓ ✗  (position 3 sees 0-3)
           ✓ ✓ ✓ ✓ ✓  (position 4 sees 0-4)
```

Implemented by setting impossible attention scores to `-inf` before softmax.

### 5. Feed-Forward Network

Position-wise, fully-connected network:

```python
FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
         ├─ First layer expands: 768 → 3072
         ├─ ReLU activation
         └─ Second layer contracts: 3072 → 768
```

Applies same MLP to each position independently. Each position gets its own "processing step."

### 6. Layer Normalization

Normalizes activations across the embedding dimension:

```python
# For each sample/position
normalized = (x - mean(x)) / sqrt(var(x) + epsilon)

# Then scale and shift (learnable parameters)
output = gamma * normalized + beta
```

**Why**: Stabilizes training, reduces internal covariate shift, helps gradients flow.

### 7. Output Layer

After 12 blocks, project hidden states to vocabulary scores:

```python
logits = x @ W_out    # (batch, seq_len, vocab_size)
         # shape: (batch, 1024, 200000)
```

During generation, take argmax (or sample) to get next token:

```python
next_token = argmax(logits[-1, :])  # Greedy: take most likely
# or
next_token = sample(logits[-1, :])  # Sampling: probabilistic
```

## Data Flow in Detail

### Inference (Generating Text)

Given prompt "Hello, I am":

```
1. Tokenize:
   "Hello, I am" → [15496, 11, 314, 716]

2. Embed:
   For each token, get dense vector + position encoding
   Shape: (1, 4, 768)  # batch=1, seq_len=4, emb_dim=768

3. Forward through 12 blocks:
   Each block: (1, 4, 768) → (1, 4, 768)
   Connections are learned, attention focuses on relevant tokens

4. Layer norm:
   (1, 4, 768) → (1, 4, 768)

5. Project to vocab:
   (1, 4, 768) @ W → (1, 4, 200000)

6. Select last position (most recent token context):
   (1, 200000,) → take argmax

7. Decode:
   Token 2481 → "beautiful"

8. Append and repeat:
   Tokens now: [15496, 11, 314, 716, 2481]
   Repeat from step 2 for next token
```

## Key Design Decisions

### Pre-norm vs Post-norm

**Pre-norm** (this implementation):
- `output = sublayer(norm(x)) + x`
- More stable for deep networks
- Better gradient flow
- Standard in modern transformers

**Post-norm** (older):
- `output = norm(sublayer(x) + x)`
- Sometimes easier to train initially
- But struggles at 12+ layers

### Causal Masking

**Why not bidirectional attention?**
- Language models must predict next token without seeing future
- Bidirectional (BERT-style) is for encoding/understanding
- Causal (GPT-style) is for generation

### Learnable vs Fixed Embeddings

**Learnable positional embeddings** (this implementation):
- Trained with the model
- Can learn position-specific patterns
- No absolute bound on sequence length
- Slightly more parameters

**Fixed (sinusoidal)** (original transformer):
- No parameters
- Works for any sequence length
- Can extrapolate to longer sequences

## Hyperparameters to Experiment With

Want to modify the model? Try adjusting:

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `n_layers` | Depth, expressiveness | Training time, memory |
| `emb_dim` | Capacity, model size | Memory, computation |
| `n_heads` | Attention diversity | Computation per head |
| `context_length` | Long-range dependencies | Memory, computation |
| `vocab_size` | Token coverage | Memory, softmax size |
| `dropout` | Regularization | Underfitting if too high |

**Example**: Create a tiny model for fast testing

```python
model = GPTModel(
    vocab_size=200_000,
    context_length=256,     # was 1024
    emb_dim=256,            # was 768
    n_heads=4,              # was 12
    n_layers=2,             # was 12
    dropout=0.1
)
# Much faster to train and run!
```

## Scaling Laws

The general trend:

- **2x more parameters** → ~1% better performance (on eval)
- **2x more data** → ~2-3% better performance
- **2x more compute** → Diminishing returns apply

Most gains come from data and compute scale, not architectural changes.

## Typical Model Sizes

| Name | Parameters | emb_dim | n_layers | n_heads |
|------|------------|---------|----------|---------|
| Tiny | 3M | 128 | 3 | 2 |
| Small | 125M | 768 | 12 | 12 |
| Medium | 350M | 1024 | 24 | 16 |
| Large | 1.3B | 2048 | 24 | 32 |
| GPT-3 | 175B | 12,288 | 96 | 96 |

This implementation is roughly "Small" size (125M parameters).

## Next Steps

- **Modify the model**: See [Examples](EXAMPLES.md)
- **Deep dive on components**: See [Components](COMPONENTS.md)
- **Understand concepts**: See [Concepts](CONCEPTS.md)
- **See code**: Open `src/3_compact_gpt_architecture.py`
