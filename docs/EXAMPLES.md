# Examples

Practical code samples demonstrating common usage patterns.

## Example 1: Basic Text Generation

Generate text from a prompt.

```python
import torch
import tiktoken
from src.model import GPTModel, generate_text_simple

# Setup
cfg = {
    "vocab_size": 200_000,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model = GPTModel(cfg)
model.eval()  # Evaluation mode

# Tokenize prompt
tokenizer = tiktoken.get_encoding("o200k_base")
prompt = "The meaning of life is"
token_ids = tokenizer.encode(prompt)
token_tensor = torch.tensor(token_ids).unsqueeze(0)

# Generate
with torch.no_grad():
    output = generate_text_simple(
        model=model,
        idx=token_tensor,
        max_new_tokens=20,
        context_size=cfg["context_length"]
    )

# Decode
output_text = tokenizer.decode(output[0].tolist())
print(output_text)
```

## Example 2: Batch Generation

Generate multiple texts in parallel.

```python
prompts = [
    "Once upon a time",
    "The future of AI",
    "In a galaxy far away"
]

tokenizer = tiktoken.get_encoding("o200k_base")

# Encode all prompts to max length for batching
max_prompt_len = max(len(tokenizer.encode(p)) for p in prompts)
token_tensors = []

for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    # Pad or truncate to max length
    tokens = tokens[:max_prompt_len] + [0] * (max_prompt_len - len(tokens))
    token_tensors.append(torch.tensor(tokens))

batch = torch.stack(token_tensors)  # (3, max_prompt_len)

# Generate for all in batch
output = generate_text_simple(model, batch, max_new_tokens=30, context_size=1024)

# Decode all
for i, out in enumerate(output):
    text = tokenizer.decode(out.tolist())
    print(f"Prompt {i+1}: {text}\n")
```

## Example 3: Custom Model Size

Create a smaller model for faster experimentation.

```python
# Tiny model: ~3M parameters
tiny_cfg = {
    "vocab_size": 200_000,
    "context_length": 256,    # Shorter context
    "emb_dim": 128,           # Small hidden dim
    "n_heads": 2,             # Few heads
    "n_layers": 3,            # Few layers
    "drop_rate": 0.1,
    "qkv_bias": False,
}

tiny_model = GPTModel(tiny_cfg)

# Quick test
x = torch.randint(0, 200_000, (1, 128))
logits = tiny_model(x)
print(logits.shape)  # torch.Size([1, 128, 200000])

# This runs much faster!
```

## Example 4: Get Model Parameters

Count parameters and inspect weights.

```python
model = GPTModel(cfg)

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Parameters per layer
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} params, shape {param.shape}")

# Find specific layer
embedding_params = model.tok_emb.weight
print(f"Token embeddings: {embedding_params.shape}")
# Output: torch.Size([200000, 768])

# Verify it's trainable
print(f"Requires grad: {embedding_params.requires_grad}")
```

## Example 5: Create Training Batch

Prepare data for training.

```python
from torch.utils.data import DataLoader
from src.dataset import create_dataloader

# Load training data
with open("training_data.txt") as f:
    text = f.read()

# Create DataLoader
train_loader = create_dataloader(
    txt=text,
    batch_size=32,
    max_length=1024,
    stride=256  # Some overlap
)

# Iterate through batches
for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Target shape: {target_ids.shape}")
    
    if batch_idx == 0:
        # Forward pass
        model.train()
        logits = model(input_ids)
        print(f"  Logits shape: {logits.shape}")
        
        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss()
        # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
        loss = loss_fn(
            logits.view(-1, cfg["vocab_size"]),
            target_ids.view(-1)
        )
        print(f"  Loss: {loss.item():.4f}")
        break
```

## Example 6: Move Model to GPU

Use CUDA for faster inference.

```python
import torch

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
model = GPTModel(cfg).to(device)

# Move data to device
token_ids = torch.randint(0, 200_000, (2, 512)).to(device)

# Forward pass
logits = model(token_ids)
print(logits.device)  # cuda:0 or cpu
```

## Example 7: Modify Attention Heads

Inspect attention weights (requires hooking into model).

```python
import torch
from torch import nn

# Register forward hook to capture attention weights
attention_weights = []

def hook_fn(module, input, output):
    # output is attention output, we'd need to hook inside MultiHeadAttention
    pass

# This requires modifying the model to expose intermediate activations
# For now, we can access logits
logits = model(token_ids)

# Top-k next tokens
top_k = 10
top_logits, top_indices = torch.topk(logits[0, -1, :], k=top_k)
top_probs = torch.softmax(top_logits, dim=0)

for idx, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
    token_str = tiktoken.get_encoding("o200k_base").decode([token_id.item()])
    print(f"{idx+1}. {token_str:10s} ({prob.item():.4f})")
```

## Example 8: Sampling Instead of Greedy

Generate more diverse text using sampling.

```python
def generate_text_sampling(model, idx, max_new_tokens, context_size, temperature=1.0):
    """Generate using sampling instead of greedy argmax."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature  # Temperature scaling
        probs = torch.softmax(logits, dim=-1)
        
        # Sample from distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

# Use it
tokenizer = tiktoken.get_encoding("o200k_base")
prompt_ids = torch.tensor(tokenizer.encode("Hello")).unsqueeze(0)

# Sample with temperature=0.7 (less random)
output_cool = generate_text_sampling(model, prompt_ids, max_new_tokens=20, 
                                      context_size=1024, temperature=0.7)

# Sample with temperature=2.0 (more random)
output_warm = generate_text_sampling(model, prompt_ids, max_new_tokens=20,
                                      context_size=1024, temperature=2.0)

print("Cool (temp=0.7):", tokenizer.decode(output_cool[0].tolist()))
print("Warm (temp=2.0):", tokenizer.decode(output_warm[0].tolist()))
```

## Example 9: Save and Load Model

Persist model weights.

```python
import torch

# Save
save_path = "checkpoints/model_v1.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Load
model_new = GPTModel(cfg)
model_new.load_state_dict(torch.load(save_path))
model_new.eval()
print("Model loaded")

# Verify they produce same output
x = torch.randint(0, 200_000, (1, 10))
out1 = model(x)
out2 = model_new(x)
print(f"Outputs match: {torch.allclose(out1, out2)}")
```

## Example 10: Profile Performance

Measure inference speed.

```python
import time
import torch

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Warmup (GPU needs this)
x = torch.randint(0, 200_000, (1, 512)).to(device)
_ = model(x)

# Timing
num_runs = 100
start = time.time()

with torch.no_grad():
    for _ in range(num_runs):
        x = torch.randint(0, 200_000, (1, 512)).to(device)
        _ = model(x)

elapsed = time.time() - start
avg_time = (elapsed / num_runs) * 1000  # ms

print(f"Average inference time: {avg_time:.2f} ms")
print(f"Throughput: {512 / avg_time * 1000:.0f} tokens/second")
```

## Tips for Experimentation

### 1. Start Small
Test with tiny models before full-size:
```python
cfg = {"emb_dim": 128, "n_heads": 2, "n_layers": 2, ...}
```

### 2. Use torch.no_grad()
For inference, always disable gradients:
```python
with torch.no_grad():
    output = model(x)
```

### 3. Set Seed for Reproducibility
```python
torch.manual_seed(42)
```

### 4. Monitor Memory
```python
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
```

### 5. Profile Your Code
```python
import cProfile
cProfile.run("generate_text_simple(...)")
```

---

## Common Patterns

**Pattern: Generate from any starting point**
```python
prompt = "In the beginning"
tokens = tokenizer.encode(prompt)
output = generate_text_simple(model, torch.tensor(tokens).unsqueeze(0), ...)
```

**Pattern: Batch different sequence lengths**
```python
# Pad to max length in batch
max_len = max(len(seq) for seq in sequences)
padded = [seq + [0]*(max_len-len(seq)) for seq in sequences]
batch = torch.tensor(padded)
```

**Pattern: Monitor generation progress**
```python
for i in range(max_new_tokens):
    # Generate one token
    output = generate_text_simple(model, idx, max_new_tokens=1, ...)
    idx = output
    if i % 10 == 0:
        print(f"Generated {i} tokens...")
```

---

## Next Steps

- **Modify the code**: Try changing hyperparameters
- **Read the docs**: [Components](COMPONENTS.md), [API Reference](API_REFERENCE.md)
- **Contribute**: See [Contributing](CONTRIBUTING.md)
