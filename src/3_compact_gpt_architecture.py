import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """PyTorch Dataset for language modeling with sliding window tokenization.

    Converts raw text into overlapping token sequences for training. Each sample
    contains an input sequence and a target sequence (input shifted by 1 token).

    Args:
        txt: Raw text string to tokenize
        tokenizer: tiktoken encoder (e.g., from tiktoken.get_encoding())
        max_length: Context length for each sample
        stride: Step size for sliding window (1 = every position, max_length = no overlap)
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>", "<|endofprompt|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size, max_length, stride,
                      shuffle=True, drop_last=True, num_workers=0):
    """Creates a DataLoader for GPT training from raw text.

    Args:
        txt: Raw text string to tokenize
        batch_size: Number of samples per batch
        max_length: Context length for each sample
        stride: Sliding window step size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop incomplete final batch
        num_workers: Number of worker processes for data loading

    Returns:
        torch.utils.data.DataLoader with tokenized sequences
    """
    tokenizer = tiktoken.get_encoding("o200k_base")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer with causal masking.

    Implements scaled dot-product attention across multiple representation subspaces.
    Causal masking prevents attending to future tokens during training and generation.

    Args:
        d_in: Input dimension (embedding dimension)
        d_out: Output dimension (must be divisible by num_heads)
        context_length: Maximum sequence length for causal mask
        dropout: Dropout rate applied to attention weights
        num_heads: Number of parallel attention heads
        qkv_bias: Whether to use bias in Q, K, V projections

    Shape:
        Input: (batch_size, seq_len, d_in)
        Output: (batch_size, seq_len, d_out)
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Register causal mask (upper triangle = future positions to mask)
        self.register_buffer("mask",torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Project to Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape for multi-head: (batch, seq_len, d_out) → (batch, num_heads, seq_len, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask: prevent attending to future tokens
        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Combine attention-weighted values and project back
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift parameters.

    Normalizes input to zero mean and unit variance across the embedding dimension,
    then applies learnable affine transformation (scale and shift).

    Args:
        emb_dim: Embedding dimension to normalize across
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """Gaussian Error Linear Unit activation (approximation).

    Uses tanh approximation of GELU which is faster than torch.nn.GELU
    while maintaining numerical stability.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        coeff = torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype))
        return 0.5 * x * (1 + torch.tanh(coeff * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Linear.

    Expands hidden dimension by 4x in first layer, contracts back in second.
    Applied independently to each token position.

    Args:
        cfg: Configuration dict with 'emb_dim' key
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),)

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual.

    Uses pre-norm architecture with residual connections. Attention and feed-forward
    layers both have dropout applied to their outputs before adding residuals.

    Args:
        cfg: Configuration dict with keys: emb_dim, n_heads, context_length, drop_rate, qkv_bias
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention with residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward with residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    """GPT-style language model with stacked transformer blocks.

    Composes token + position embeddings, passes through N transformer blocks,
    applies final layer normalization, and projects to vocabulary logits.

    Args:
        cfg: Configuration dict with keys:
            - vocab_size: Number of tokens in vocabulary
            - context_length: Maximum sequence length
            - emb_dim: Embedding and hidden dimension
            - n_layers: Number of transformer blocks
            - n_heads: Number of attention heads per block
            - drop_rate: Dropout rate
            - qkv_bias: Whether to use bias in attention projections

    Shape:
        Input: (batch_size, seq_len) — token IDs
        Output: (batch_size, seq_len, vocab_size) — logits
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine token and position embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        # Pass through all transformer blocks
        x = self.trf_blocks(x)
        # Final normalization and projection to vocabulary
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generate new tokens using greedy decoding (argmax).

    Iteratively predicts the next most likely token and appends it to the sequence.
    Uses only the most recent context_size tokens to stay within model's context window.

    Args:
        model: GPTModel instance in evaluation mode
        idx: Initial token IDs, shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        context_size: Model's context length (from config)

    Returns:
        Extended token sequence including initial and generated tokens,
        shape (batch_size, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # Keep only the last context_size tokens to stay within context window
        idx_cond = idx[:, -context_size:]

        # Generate logits for next token (no gradient computation needed)
        with torch.no_grad():
            logits = model(idx_cond)

        # Take logits for the last position only
        logits = logits[:, -1, :]
        # Greedy: select token with highest probability
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    """Demo: Initialize model and generate text from a prompt.

    Shows complete pipeline: tokenization → embedding → forward pass → decoding.
    """
    tokenizer = tiktoken.get_encoding("o200k_base")

    GPT_CONFIG = {
        "vocab_size": tokenizer.n_vocab,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG)
    model.eval()

    start_context = "Hello, I am Elon Musk"

    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG["context_length"],
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

if __name__ == "__main__":
    main()