# API Reference

Formal documentation of classes, functions, and their parameters.

## GPTDataset

```python
class GPTDataset(Dataset)
```

PyTorch Dataset for language modeling with sliding window tokenization.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `txt` | str | Raw text to tokenize |
| `tokenizer` | tiktoken.Encoding | Tokenizer (use `tiktoken.get_encoding("o200k_base")`) |
| `max_length` | int | Context window size (recommend: 512-1024) |
| `stride` | int | Sliding window step (1 = dense, max_length = no overlap) |

### Methods

**`__len__() → int`**
- Returns number of samples in dataset

**`__getitem__(idx: int) → Tuple[Tensor, Tensor]`**
- Returns: (input_tokens, target_tokens)
- Shape: Both are `(max_length,)` tensors of dtype `torch.long`

### Example

```python
import tiktoken
from src.models import GPTDataset

dataset = GPTDataset(
    txt=open("data.txt").read(),
    tokenizer=tiktoken.get_encoding("o200k_base"),
    max_length=1024,
    stride=256
)
x, y = dataset[0]
print(x.shape, y.shape)  # torch.Size([1024]) torch.Size([1024])
```

---

## create_dataloader()

```python
def create_dataloader(
    txt: str,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) → DataLoader
```

Creates a DataLoader from raw text.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `txt` | str | — | Raw text to tokenize |
| `batch_size` | int | — | Samples per batch |
| `max_length` | int | — | Context length |
| `stride` | int | — | Sliding window step |
| `shuffle` | bool | True | Randomize batch order |
| `drop_last` | bool | True | Drop incomplete final batch |
| `num_workers` | int | 0 | Parallel workers (0=main process) |

### Returns

`torch.utils.data.DataLoader` with tokenized sequences

### Example

```python
loader = create_dataloader(
    txt=open("train.txt").read(),
    batch_size=32,
    max_length=512,
    stride=128
)

for batch_x, batch_y in loader:
    print(batch_x.shape)  # (32, 512)
    print(batch_y.shape)  # (32, 512)
```

---

## MultiHeadAttention

```python
class MultiHeadAttention(nn.Module)
```

Multi-head self-attention with causal masking.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_in` | int | — | Input dimension |
| `d_out` | int | — | Output dimension (must divisible by num_heads) |
| `context_length` | int | — | Max sequence length |
| `dropout` | float | — | Dropout rate (0.0-1.0) |
| `num_heads` | int | — | Number of attention heads |
| `qkv_bias` | bool | False | Use bias in Q,K,V projections |

### Methods

**`forward(x: Tensor) → Tensor`**
- Input: `(batch, seq_len, d_in)`
- Output: `(batch, seq_len, d_out)`

### Example

```python
attn = MultiHeadAttention(
    d_in=768,
    d_out=768,
    context_length=1024,
    dropout=0.1,
    num_heads=12
)

x = torch.randn(2, 512, 768)
output = attn(x)
print(output.shape)  # torch.Size([2, 512, 768])
```

---

## LayerNorm

```python
class LayerNorm(nn.Module)
```

Layer normalization with learnable scale and shift.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `emb_dim` | int | Dimension to normalize across |

### Methods

**`forward(x: Tensor) → Tensor`**
- Normalizes last dimension to mean=0, std=1
- Applies learnable scale and shift
- Input/output shape: Same as input

### Example

```python
norm = LayerNorm(emb_dim=768)
x = torch.randn(2, 1024, 768)
y = norm(x)
print(y.shape)  # torch.Size([2, 1024, 768])
```

---

## GELU

```python
class GELU(nn.Module)
```

Gaussian Error Linear Unit activation (approximate).

### Methods

**`forward(x: Tensor) → Tensor`**
- Smooth activation approximating Gaussian Error Linear Unit
- Preserves input shape

### Example

```python
gelu = GELU()
x = torch.randn(2, 512, 768)
y = gelu(x)
print(y.shape)  # torch.Size([2, 512, 768])
```

---

## FeedForward

```python
class FeedForward(nn.Module)
```

Position-wise feed-forward network: Linear → GELU → Linear.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg` | dict | Config dict with key `emb_dim` |

### Methods

**`forward(x: Tensor) → Tensor`**
- First linear: `emb_dim → 4*emb_dim`
- GELU activation
- Second linear: `4*emb_dim → emb_dim`

### Example

```python
cfg = {"emb_dim": 768}
ff = FeedForward(cfg)
x = torch.randn(2, 512, 768)
y = ff(x)
print(y.shape)  # torch.Size([2, 512, 768])
```

---

## TransformerBlock

```python
class TransformerBlock(nn.Module)
```

Transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg` | dict | Config dict with keys: emb_dim, n_heads, context_length, drop_rate, qkv_bias |

### Methods

**`forward(x: Tensor) → Tensor`**
- Input: `(batch, seq_len, emb_dim)`
- Output: `(batch, seq_len, emb_dim)`

### Example

```python
cfg = {
    "emb_dim": 768,
    "n_heads": 12,
    "context_length": 1024,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
block = TransformerBlock(cfg)
x = torch.randn(2, 512, 768)
y = block(x)
print(y.shape)  # torch.Size([2, 512, 768])
```

---

## GPTModel

```python
class GPTModel(nn.Module)
```

GPT-style language model with stacked transformer blocks.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg` | dict | Config dict with all required keys (see below) |

### Required Config Keys

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `vocab_size` | int | 200000 | Vocabulary size |
| `context_length` | int | 1024 | Max sequence length |
| `emb_dim` | int | 768 | Embedding/hidden dimension |
| `n_heads` | int | 12 | Attention heads per block |
| `n_layers` | int | 12 | Number of transformer blocks |
| `drop_rate` | float | 0.1 | Dropout rate |
| `qkv_bias` | bool | False | Use bias in attention |

### Methods

**`forward(in_idx: Tensor) → Tensor`**
- Input: `(batch, seq_len)` token IDs
- Output: `(batch, seq_len, vocab_size)` logits

### Example

```python
cfg = {
    "vocab_size": 200000,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model = GPTModel(cfg)
token_ids = torch.randint(0, 200000, (2, 512))
logits = model(token_ids)
print(logits.shape)  # torch.Size([2, 512, 200000])

# Get probabilities for next token
probs = torch.softmax(logits[:, -1, :], dim=-1)
print(probs.shape)  # torch.Size([2, 200000])
```

---

## generate_text_simple()

```python
def generate_text_simple(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int
) → Tensor
```

Generate new tokens using greedy decoding.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | GPTModel | Model in evaluation mode |
| `idx` | Tensor | Initial token IDs, shape `(batch, seq_len)` |
| `max_new_tokens` | int | Number of tokens to generate |
| `context_size` | int | Model context length (from config) |

### Returns

`Tensor` of shape `(batch, seq_len + max_new_tokens)` with generated tokens appended.

### Example

```python
model = GPTModel(cfg)
model.eval()

# Start with a prompt
tokenizer = tiktoken.get_encoding("o200k_base")
prompt = "Hello, I am"
encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

# Generate 50 tokens
output_ids = generate_text_simple(
    model=model,
    idx=encoded,
    max_new_tokens=50,
    context_size=cfg["context_length"]
)

# Decode to text
output_text = tokenizer.decode(output_ids[0].tolist())
print(output_text)
```

---

## main()

```python
def main() → None
```

Demonstration script. Initializes model, encodes prompt, generates text.

### What It Does

1. Creates tokenizer (tiktoken)
2. Defines GPT configuration
3. Initializes GPTModel
4. Encodes prompt "Hello, I am Elon Musk"
5. Generates 10 new tokens
6. Prints input encoding and decoded output

### Run It

```bash
python src/3_compact_gpt_architecture.py
```

### Modify It

```python
# Change prompt
start_context = "The future of AI"

# Change number of tokens
max_new_tokens = 50

# Change model size
GPT_CONFIG = {
    ...
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 2,
}
```

---

## Configuration Template

```python
GPT_CONFIG = {
    # Vocabulary
    "vocab_size": 200_000,  # Number of tokens
    
    # Sequence length
    "context_length": 1024,  # Max tokens per sequence
    
    # Model dimensions
    "emb_dim": 768,          # Hidden dimension throughout
    "n_heads": 12,           # Attention heads
    "n_layers": 12,          # Transformer blocks
    
    # Regularization
    "drop_rate": 0.1,        # Dropout rate
    
    # Attention details
    "qkv_bias": False,       # Bias in Q,K,V projections
}

model = GPTModel(GPT_CONFIG)
```

### Typical Configurations

**Tiny (3M parameters)**
```python
{"emb_dim": 128, "n_heads": 2, "n_layers": 3, "context_length": 256}
```

**Small (125M parameters)**
```python
{"emb_dim": 768, "n_heads": 12, "n_layers": 12, "context_length": 1024}
```

**Large (1.3B parameters)**
```python
{"emb_dim": 2048, "n_heads": 32, "n_layers": 24, "context_length": 2048}
```

---

## Data Types

All tensors use `torch.long` for token IDs and `torch.float32` for activations (default).

To use `torch.float16` (GPU only):

```python
model = GPTModel(cfg)
model = model.half()  # Convert to float16
```

---

## Device Handling

By default, model uses CPU. To use GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(cfg).to(device)
token_ids = token_ids.to(device)
```

---

## Next Steps

- **Examples**: See [Examples](EXAMPLES.md)
- **Concepts**: See [CONCEPTS](CONCEPTS.md)
- **Architecture**: See [ARCHITECTURE](ARCHITECTURE.md)
