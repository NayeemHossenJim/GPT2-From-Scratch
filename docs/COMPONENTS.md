# Components

Detailed walkthrough of each class and function in the GPT model.

## Overview

The entire implementation lives in `src/3_compact_gpt_architecture.py` and consists of:

1. **GPTDataset** — Tokenizes text into training samples
2. **create_dataloader()** — Wraps GPTDataset in PyTorch DataLoader
3. **MultiHeadAttention** — Scaled dot-product attention with multiple heads
4. **LayerNorm** — Normalizes activations with learnable parameters
5. **GELU** — Smooth activation function (approximate)
6. **FeedForward** — Position-wise MLP (two linear layers)
7. **TransformerBlock** — Combines attention + FFN with residuals
8. **GPTModel** — Full model (embeddings + transformer blocks)
9. **generate_text_simple()** — Text generation via greedy decoding
10. **main()** — Demo entry point

---

## 1. GPTDataset

**Purpose**: Convert raw text into training samples with sliding window.

**File**: `src/3_compact_gpt_architecture.py:7-24`

### How It Works

```python
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # txt: raw text string
        # tokenizer: tiktoken encoder
        # max_length: context window size (1024)
        # stride: sliding window step (1 = every position)

        # 1. Tokenize entire text
        token_ids = tokenizer.encode(txt)

        # 2. Slide window to create samples
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]         # "Hello, I am"
            target_chunk = token_ids[i + 1 : i + max_length + 1]  # "I am Elon"
            # Target is input shifted by 1 (next token prediction)
```

### Shape Transformations

```
Raw text: "Hello world, how are you?"
         ↓ tokenize
Token IDs: [15496, 1917, 11, 1128, 527, 345]
         ↓ create samples (max_length=3, stride=1)
Sample 1: input=[15496, 1917, 11]      target=[1917, 11, 1128]
Sample 2: input=[1917, 11, 1128]       target=[11, 1128, 527]
Sample 3: input=[11, 1128, 527]        target=[1128, 527, 345]
```

### Usage

```python
dataset = GPTDataset(
    txt="Hello world",
    tokenizer=tiktoken.get_encoding("o200k_base"),
    max_length=1024,
    stride=1
)
print(len(dataset))  # Number of samples created
x, y = dataset[0]   # Get first sample
print(x.shape)      # torch.Size([1024])
```

### Key Points

- **Stride parameter**: Controls overlap
  - `stride=1` (every position) → dense sampling, many samples
  - `stride=max_length` (no overlap) → sparse sampling, fewer samples
- **Target is shifted input**: Enables next-token prediction
- **Padding not implemented**: Text must be longer than max_length

---

## 2. create_dataloader()

**Purpose**: Convenience function to create tokenizer, dataset, and dataloader.

**File**: `src/3_compact_gpt_architecture.py:40-62`

### How It Works

```python
def create_dataloader(txt, batch_size, max_length, stride,
                      shuffle=True, drop_last=True, num_workers=0):
    # 1. Get tokenizer
    tokenizer = tiktoken.get_encoding("o200k_base")

    # 2. Create dataset
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    # 3. Wrap in DataLoader (batching + shuffling)
    dataloader = DataLoader(dataset, batch_size=batch_size, ...)
    return dataloader
```

### Usage

```python
with open("training_data.txt") as f:
    text = f.read()

loader = create_dataloader(
    txt=text,
    batch_size=64,
    max_length=1024,
    stride=128  # Non-overlapping windows
)

for batch_x, batch_y in loader:
    print(batch_x.shape)  # (64, 1024) — batch of 64 sequences
    print(batch_y.shape)  # (64, 1024) — corresponding targets
    break
```

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `shuffle` | True | Randomize batch order |
| `drop_last` | True | Drop incomplete final batch |
| `num_workers` | 0 | Parallel data loading (0 = main process) |

---

## 3. MultiHeadAttention

**Purpose**: Core transformer mechanism. Computes attention across 12 parallel "heads."

**File**: `src/3_compact_gpt_architecture.py:76-138`

### What Is Attention?

Simple question: "Given Query, what parts of Key/Value are relevant?"

```
Query = "What"
Key = ["The", "quick", "brown", "fox"]
Value = [embedding_for_the, embedding_for_quick, ...]

Attention = softmax(Query @ Key^T) @ Value
           ↑                    ↑
         similarity scores   weighted sum
```

Result: A context vector that blends all values, weighted by relevance to query.

### Multi-Head: Why?

Instead of one attention operation, run 12 in parallel with different weight matrices:

```
Head 1: Focuses on "subject-verb" relations
Head 2: Focuses on "adjective-noun" relations
Head 3: Focuses on "temporal" relations
...
Head 12: Focuses on "reference resolution"

All 12 outputs concatenated + projected back
```

### Causal Masking: Why?

During generation, we can't look at future tokens. Set future attention scores to `-inf`:

```
Position: 0 1 2 3 4
Can see:  ✓ ✗ ✗ ✗ ✗  (position 0 can only see itself)
          ✓ ✓ ✗ ✗ ✗  (position 1 can see 0-1)
          ✓ ✓ ✓ ✗ ✗  (position 2 can see 0-2)
```

### Code Walkthrough

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # d_in = 768 (embedding dim)
        # d_out = 768 (output dim, must be divisible by num_heads)
        # num_heads = 12
        # head_dim = 768 / 12 = 64

        self.W_query = nn.Linear(d_in, d_out)      # Transform to Q
        self.W_key = nn.Linear(d_in, d_out)        # Transform to K
        self.W_value = nn.Linear(d_in, d_out)      # Transform to V
        self.out_proj = nn.Linear(d_out, d_out)    # Final projection

        # Pre-compute causal mask (lower triangular = can see, upper = cannot)
        self.register_buffer("mask", torch.triu(torch.ones(...), diagonal=1).bool())

    def forward(self, x):
        # x shape: (batch=2, seq_len=4, d_in=768)

        # 1. Project to Q, K, V
        queries = self.W_query(x)  # (2, 4, 768)
        keys = self.W_key(x)       # (2, 4, 768)
        values = self.W_value(x)   # (2, 4, 768)

        # 2. Reshape for multi-head
        # (2, 4, 768) → (2, 4, 12, 64) → (2, 12, 4, 64)
        #               batch, seq, heads, head_dim
        queries = queries.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        attn_scores = queries @ keys.transpose(-2, -1)  # (2, 12, 4, 4)
        attn_scores = attn_scores / sqrt(head_dim)       # Scale by sqrt(d_k)

        # 4. Apply causal mask
        attn_scores.masked_fill_(mask, -torch.inf)       # Set future to -inf

        # 5. Softmax to get weights
        attn_weights = softmax(attn_scores, dim=-1)      # (2, 12, 4, 4)

        # 6. Weighted average of values
        context = attn_weights @ values                  # (2, 12, 4, 64)

        # 7. Concatenate heads and project
        context = context.view(b, seq_len, d_out)        # (2, 4, 768)
        output = self.out_proj(context)                  # (2, 4, 768)
        return output
```

### Usage

```python
attn = MultiHeadAttention(
    d_in=768,
    d_out=768,
    context_length=1024,
    dropout=0.1,
    num_heads=12
)

x = torch.randn(2, 10, 768)  # batch=2, seq_len=10, dim=768
output = attn(x)
print(output.shape)  # torch.Size([2, 10, 768])
```

---

## 4. LayerNorm

**Purpose**: Normalize activations to improve training stability.

**File**: `src/3_compact_gpt_architecture.py:140-153`

### What It Does

```
Normalize: mean = 0, std = 1
Scale and shift: y = gamma * norm_x + beta  (learnable)
```

### Why It Matters

- **Prevents exploding/vanishing gradients** by keeping activation magnitudes stable
- **Enables faster learning** by reducing internal covariate shift
- **Improves generalization** in deep networks

### Code

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma: learnable
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # beta: learnable

    def forward(self, x):
        # x shape: (batch, seq_len, emb_dim)

        # Normalize across the last dimension (embedding dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / sqrt(var + eps)

        # Scale and shift
        return self.scale * norm_x + self.shift
```

### Example

```python
layer_norm = LayerNorm(emb_dim=768)
x = torch.randn(2, 1024, 768)  # batch activations
y = layer_norm(x)
print(y.mean(dim=-1))          # Should be ~0
print(y.std(dim=-1))           # Should be ~1
```

---

## 5. GELU & FeedForward

**Purpose**: Non-linearity and position-wise processing.

**File**: `src/3_compact_gpt_architecture.py:155-177`

### GELU (Gaussian Error Linear Unit)

Smooth, non-linear activation that approximates: `0.5 * x * (1 + erf(x / sqrt(2)))`

```python
# Smooth activation, not harsh like ReLU
# Allows gradients for both positive and negative inputs
```

### FeedForward

Position-wise MLP applied to each token independently:

```
Input: (batch, seq_len, emb_dim=768)
  ↓ Linear(768 → 3072)
  ↓ GELU activation
  ↓ Linear(3072 → 768)
Output: (batch, seq_len, 768)
```

Why expand-then-contract?
- Expansion allows more complex transformations
- Contraction brings back to model dimension
- Standard architecture from "Attention Is All You Need"

### Code

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)
```

---

## 6. TransformerBlock

**Purpose**: Core building block combining attention + feedforward.

**File**: `src/3_compact_gpt_architecture.py:179-212`

### Architecture

```
Input x
  ↓
[Layer Norm]
  ↓
[Multi-Head Attention]
  ↓
[Dropout]
  ↓
x + above ← Residual connection
  ↓ (now normalized)
[Layer Norm]
  ↓
[Feed-Forward]
  ↓
[Dropout]
  ↓
x + above ← Residual connection
  ↓
Output
```

### Pre-norm vs Post-norm

This implementation uses **pre-norm** (LayerNorm before each sub-layer):

```
Pre-norm:  x + SubLayer(Norm(x))
Post-norm: Norm(x + SubLayer(x))
```

Pre-norm is more stable for deep networks (12 layers here).

### Code Walkthrough

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        self.att = MultiHeadAttention(...)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention sub-layer with residual
        shortcut = x
        x = self.norm1(x)               # Pre-norm
        x = self.att(x)
        x = self.drop_shortcut(x)       # Regularization
        x = x + shortcut                # Residual

        # Feed-forward sub-layer with residual
        shortcut = x
        x = self.norm2(x)               # Pre-norm
        x = self.ff(x)
        x = self.drop_shortcut(x)       # Regularization
        x = x + shortcut                # Residual

        return x
```

### Key Insight

Residual connections allow information to "bypass" the non-linear transformations, making gradients flow more easily through deep networks.

---

## 7. GPTModel

**Purpose**: Full model orchestration.

**File**: `src/3_compact_gpt_architecture.py:214-245`

### Architecture Diagram

```
Token IDs: [15496, 11, 314]  (input)
  ↓ Token Embedding (lookup)
Dense vectors: [[0.2, -0.5, ...], ...]
  ↓ Add Position Embeddings
Position-aware vectors
  ↓ Dropout (regularization)
  ↓ [TransformerBlock 1] ← 12 of these
  ↓ [TransformerBlock 2]
  ↓ ...
  ↓ [TransformerBlock 12]
  ↓ Final Layer Norm
Normalized activations
  ↓ Linear to vocab (200K)
Logits: [0.1, 2.3, -0.5, ..., 1.2]  (200,000 scores)
  ↓ argmax or sample
Next token ID: 2481
```

### Code

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        # Token → embedding lookup table
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)

        # Position → embedding lookup table
        self.pos_emb = nn.Embedding(context_length, emb_dim)

        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])

        # Final normalization and projection
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, in_idx):
        # in_idx shape: (batch=2, seq_len=1024)

        # Embed and add positions
        tok_embeds = self.tok_emb(in_idx)      # (2, 1024, 768)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=...)   # (1024, 768)
        )
        x = tok_embeds + pos_embeds             # (2, 1024, 768)

        # Pass through 12 transformer blocks
        x = self.trf_blocks(x)                  # (2, 1024, 768)

        # Final normalization and projection to vocabulary
        x = self.final_norm(x)                  # (2, 1024, 768)
        logits = self.out_head(x)               # (2, 1024, 200000)

        return logits
```

### Usage

```python
cfg = {
    "vocab_size": 200000,
    "emb_dim": 768,
    "context_length": 1024,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model = GPTModel(cfg)
token_ids = torch.randint(0, 200000, (2, 512))  # Random tokens
logits = model(token_ids)
print(logits.shape)  # torch.Size([2, 512, 200000])
```

---

## 8. generate_text_simple()

**Purpose**: Generate new text token-by-token using greedy decoding.

**File**: `src/3_compact_gpt_architecture.py:247-277`

### Algorithm

```
Input: starting tokens = [15496, 11, 314]
Max new tokens: 10

for i in range(10):
    1. Keep only last context_size tokens (e.g., 1024)
    2. Forward pass through model → logits
    3. Take logits of last position only
    4. Greedy: argmax(logits) → next token
    5. Append to sequence

Output: [15496, 11, 314, 2481, 1829, ...]  (extended by 10 tokens)
```

### Code

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Keep only the last context_size tokens
        idx_cond = idx[:, -context_size:]

        # Forward pass (no gradients needed)
        with torch.no_grad():
            logits = model(idx_cond)              # (batch, context_size, vocab)

        # Take logits for the last position
        logits = logits[:, -1, :]                 # (batch, vocab)

        # Greedy: take the most likely token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### Usage

```python
model = GPTModel(cfg)
model.eval()

# Start with a prompt
tokenizer = tiktoken.get_encoding("o200k_base")
prompt = "Hello, I am"
encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

# Generate 50 new tokens
output_tokens = generate_text_simple(
    model=model,
    idx=encoded,
    max_new_tokens=50,
    context_size=cfg["context_length"]
)

# Decode back to text
output_text = tokenizer.decode(output_tokens.squeeze(0).tolist())
print(output_text)
```

### Greedy vs. Sampling

This function uses **greedy decoding** (always pick argmax). Alternative:

**Sampling**:
```python
# Instead of argmax:
probs = torch.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Greedy = deterministic, often boring. Sampling = diverse, sometimes nonsensical.

---

## 9. main()

**Purpose**: Demonstration entry point.

**File**: `src/3_compact_gpt_architecture.py:279-322`

### What It Does

1. Create tokenizer
2. Define model configuration
3. Initialize GPTModel
4. Encode a prompt
5. Generate text
6. Decode and print output

### Run It

```bash
python src/3_compact_gpt_architecture.py
```

### Modify It

Change the prompt:

```python
start_context = "Once upon a time"
```

Or the number of tokens:

```python
max_new_tokens=50  # Generate more tokens
```

Or the model size:

```python
GPT_CONFIG = {
    ...
    "emb_dim": 256,      # Smaller
    "n_heads": 4,        # Fewer heads
    "n_layers": 2,       # Fewer layers
}
```

---

## Summary

| Component | Purpose | Key Parameters |
|-----------|---------|-----------------|
| GPTDataset | Tokenize text → training samples | max_length, stride |
| MultiHeadAttention | Weighted attention across multiple "heads" | num_heads, d_out |
| LayerNorm | Normalize + learn scale/shift | emb_dim |
| GELU | Smooth activation | None |
| FeedForward | Position-wise MLP | expansion factor (4x) |
| TransformerBlock | Attention + FFN + residuals + norms | emb_dim, n_heads |
| GPTModel | Full model orchestration | all config |
| generate_text_simple | Greedy text generation | max_new_tokens |

---

## Next Steps

- **See code**: Open `src/3_compact_gpt_architecture.py`
- **Understand concepts**: Read [Concepts](CONCEPTS.md)
- **Try examples**: Check [Examples](EXAMPLES.md)
- **Modify the model**: [Architecture](ARCHITECTURE.md) has tuning tips
