# Concepts

Fundamental concepts for understanding LLMs and transformers.

## Tokenization

### What Is It?

Converting raw text into discrete units (tokens) that a neural network can process.

### Why Do It?

- Neural networks work with numbers, not text
- Tokens are the right granularity for language (between characters and words)
- Enables handling of rare words and subword units

### Methods

**Character-level**: Each character is a token
- Pro: Small vocabulary (256 possible characters)
- Con: Long sequences, harder to learn

**Word-level**: Each word is a token
- Pro: Reasonable sequence length
- Con: Huge vocabulary (100K+ unique words), doesn't handle new words well

**Subword (BPE)**: Frequent subword units
- Pro: Balance between vocabulary size and sequence length
- Con: Requires training the tokenizer
- Used by: GPT-2, GPT-3, GPT-4 (tiktoken)

### Example

```
Text: "Hello, I'm learning!"

Character-level: [72, 101, 108, 108, 111, 44, ...]  (more tokens)
Word-level: [123, 456, 789, ...]  (fewer tokens, but unknown word "learning" → [unknown])
Subword (BPE): [15496, 11, 314, 716, 2319, 0]  (balanced)
```

This project uses BPE via `tiktoken.get_encoding("o200k_base")`.

---

## Embeddings

### What Are They?

Dense vectors in continuous space that represent tokens semantically.

### Why Do We Need Them?

- Tokens are just integers (arbitrary labels)
- Embeddings capture semantic meaning
- Similar tokens have similar embeddings
- Neural networks learn from continuous values, not discrete labels

### Token Embeddings

Learnable lookup table: token ID → dense vector

```python
token_embed = nn.Embedding(vocab_size=200_000, embedding_dim=768)
token_embed(torch.tensor([15496]))  # → (1, 768) dense vector
```

Words with similar meaning eventually learn similar embeddings during training.

### Positional Embeddings

Encodes the position of a token in the sequence.

```
Sequence: "The cat sat"
Positions: 0   1    2

Without position embeddings:
- "cat" at position 1 and "sat" at position 2 have nothing distinguishing their position

With position embeddings:
- "cat" sees pos_embed[1]
- "sat" sees pos_embed[2]
- Model learns that position matters
```

**Two approaches**:

1. **Learnable** (this project): Embedding lookup table for each position
   - Pros: Learned by the model, specific to this task
   - Cons: Limited to max_context_length

2. **Sinusoidal** (original transformer): Fixed mathematical formulas
   - Pros: Works for any sequence length
   - Cons: No learning, mathematical overhead

---

## Self-Attention Mechanism

### The Core Idea

Given a query "What's important here?", attention answers "Look at these specific parts."

### Scaled Dot-Product Attention

```
Query Q:  "What's important?"
Key K:    "Here are descriptions of different parts"
Value V:  "Here's the actual content"

Attention(Q, K, V) = softmax(Q·K^T / √d) · V
                     └─ similarity scores ─┘   └ weighted values ┘
```

### Step-by-Step

```
1. Compute similarity between query and each key:
   scores = Q @ K^T  → (seq_len, seq_len) matrix

2. Scale by sqrt(dimension) for numerical stability:
   scores = scores / sqrt(d_k)

3. Softmax to get weights (probabilities):
   weights = softmax(scores)  → each row sums to 1

4. Weighted average of values:
   output = weights @ V  → blend values based on attention weights
```

### Intuition

Attention learns which positions are relevant to each other:

```
Input: "The cat sat on the mat"
Position 0 ("The") might attend to:
- Position 1 ("cat") with weight 0.8
- Position 2 ("sat") with weight 0.1
- Position 3 ("on") with weight 0.05
- etc.

Result: Context vector for "The" is mostly about "cat"
```

### Why It Works

- **Learns dependencies**: Can capture long-range relationships
- **Flexible**: Automatically learns what's relevant
- **Parallel**: All positions attend simultaneously (unlike RNNs)

---

## Multi-Head Attention

### The Problem with Single-Head Attention

One attention mechanism might focus on one type of relationship (e.g., grammar) and miss others (e.g., long-range semantics).

### The Solution

Run multiple attention mechanisms in parallel:

```
Head 1: Focuses on syntax ("verb modifying subject")
Head 2: Focuses on semantics ("related concepts")
Head 3: Focuses on coreference ("which noun does 'it' refer to?")
...
Head 12: Learns something else

All 12 outputs → concatenate → project → single output
```

Each head has its own Q, K, V projections with different learned parameters.

### Benefit

Ensemble effect: different "experts" each learn different aspects of language.

### Computation

```
Input: (batch, seq_len, emb_dim=768)

12 parallel heads, each with dimension 768/12=64:
Head 1: (batch, seq_len, 64)
Head 2: (batch, seq_len, 64)
...
Head 12: (batch, seq_len, 64)

Concatenate: (batch, seq_len, 768)
Project: (batch, seq_len, 768)
```

---

## Transformer Blocks

### Architecture

```
Input x
  ↓
[Layer Norm] — normalize activations
  ↓
[Multi-Head Attention] — attend across all positions
  ↓
[Residual Connection] — add original input back
  ↓
[Layer Norm] — normalize again
  ↓
[Feed-Forward Network] — position-wise MLP
  ↓
[Residual Connection] — add residual again
  ↓
Output
```

### Why Residual Connections?

```
Without residuals:
x → [Attention] → [Norm] → [FFN] → [Norm] → output

With residuals:
x ────────────────────────────────┐
  ↓                               ↓
[Attention] + ← adds original x
  ↓
[Norm]
  ↓
y ────────────────────────────────┐
  ↓                               ↓
[FFN] + ← adds y back
  ↓
output
```

**Benefits**:
- **Gradient flow**: Gradients can flow directly through residuals
- **Enables depth**: Can stack many layers (12+) without vanishing gradients
- **Information preservation**: Original input is always added back

### Why Pre-Norm?

**Pre-norm** (this implementation):
```
x → [Norm] → [Sublayer] → + → output
```

**Post-norm** (original transformer):
```
x → [Sublayer] → [Norm] → + → output
```

Pre-norm is more stable for deep networks because:
- Normalization happens within gradient flow
- Prevents activation explosion before residuals
- Better gradient propagation

---

## Layer Normalization

### What It Does

Normalizes activations across the feature dimension:

```
For each sample/position:
  mean = average of all features
  std = standard deviation of features
  normalized = (features - mean) / std
  output = gamma * normalized + beta  (gamma, beta are learned)
```

### Why Normalize?

- **Stabilizes training**: Prevents extreme activation values
- **Reduces internal covariate shift**: Downstream layers see stable distributions
- **Enables faster learning**: Gradients are more predictable

### Example

```
Activations: [5.2, -100.5, 3.1, 200.0, 0.5]
Mean: 21.66, Std: 115.7

Normalized: [-0.14, -1.05, -0.16, 1.54, -0.19]
Much more stable range!
```

---

## Feed-Forward Networks

### Architecture

```
Input: (batch, seq_len, emb_dim=768)
  ↓
[Linear] → 4 * emb_dim (3072)
  ↓
[GELU activation] → smooth non-linearity
  ↓
[Linear] → emb_dim (768)
  ↓
Output: (batch, seq_len, 768)
```

### Why Expand-Then-Contract?

- **Expansion**: Allows learning of non-linear combinations
- **4x scaling**: Empirically works well, from original transformer paper
- **Contraction**: Brings back to model dimension for residual connection

### Applied Position-Wise

The same FFN is applied independently to each token position:

```
Position 0: [0.2, -0.5, ...] → FFN → [0.1, 0.3, ...]
Position 1: [0.1, 0.2, ...] → FFN → [0.2, -0.1, ...]
...
```

All positions use the same learned weights, but see different input.

---

## Causal Masking

### The Problem

During training, we want to predict "word at position t given words before position t."

Without masking, the model could "cheat" by looking at the answer (word at position t).

### The Solution

Block attention to future positions:

```
Position:  0 1 2 3 4
Can see:   • ✗ ✗ ✗ ✗  (position 0 sees only itself)
           • • ✗ ✗ ✗  (position 1 sees 0-1)
           • • • ✗ ✗  (position 2 sees 0-2)
           • • • • ✗  (position 3 sees 0-3)
           • • • • •  (position 4 sees all)
```

### Implementation

Set attention scores for future positions to `-∞` before softmax:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attn_scores.masked_fill_(mask, -torch.inf)
weights = softmax(attn_scores)  # -∞ → 0 in softmax
```

After softmax, `-∞` becomes 0 probability (ignored).

---

## Language Modeling

### Task

Predict the next token given previous tokens:

```
Input: "The quick brown"
Target: "fox"

Model learns: P(fox | The quick brown) = 0.93
```

### Loss Function

Cross-entropy loss between predicted logits and true next token:

```
logits = model(input_tokens)           # shape: (batch, seq_len, vocab_size)
loss = cross_entropy(logits, targets)  # scalar
```

### Training Procedure

1. Feed sequence through model
2. Compute loss (next-token prediction)
3. Backpropagate gradients
4. Update parameters with optimizer
5. Repeat on next batch

Thousands of iterations on massive datasets → model learns language!

---

## Inference and Generation

### Greedy Decoding

```
1. Start with prompt: "Hello, I am"
2. Forward pass → logits for next token
3. Select most likely token (argmax)
4. Append to sequence
5. Repeat from step 2 with new sequence
```

**Pros**: Deterministic, fast
**Cons**: Often repetitive, unnatural

### Sampling

```
1. Forward pass → logits
2. Convert to probabilities
3. Sample from distribution
4. Append to sequence
5. Repeat
```

**Pros**: Diverse, more natural
**Cons**: Non-deterministic, can be incoherent

### Temperature

Controls randomness:

```
logits_scaled = logits / temperature

temperature=0.1  → sharp distribution (argmax-like)
temperature=1.0  → normal softmax
temperature=2.0  → flatter distribution (more random)
```

---

## Scaling Laws

Empirical observations about LLM behavior:

- **More parameters** → better performance (scaling law)
- **More data** → better performance (more than architecture changes)
- **More compute** → better performance (training for more steps)

### Chinchilla Scaling

Optimal ratio: compute ∝ data size

```
If doubling compute budget:
- Split half to more data
- Split half to larger model
```

### Implications

- Data >> architecture for improving performance
- Bigger isn't always better (compute-optimal models)
- Training for longer helps more than bigger models

---

## Context Length

### What It Is

Maximum number of tokens the model sees at once.

### Trade-offs

**Longer context (1024 vs 512)**:
- ✓ Better for long-range dependencies
- ✗ More computation (quadratic in attention)
- ✗ More memory

**Shorter context**:
- ✓ Faster and lighter
- ✗ Can't see distant context
- ✗ Limited reasoning on long documents

---

## Next Steps

- **Read papers**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **See code**: [COMPONENTS.md](COMPONENTS.md)
- **Run examples**: [EXAMPLES.md](EXAMPLES.md)
