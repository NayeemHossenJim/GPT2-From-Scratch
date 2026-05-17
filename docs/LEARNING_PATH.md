# Learning Path

Different people learn best in different ways. This guide provides three structured paths through the material. Pick the one that matches your learning style, or mix and match.

## Quick Decision Tree

- **"Just show me how to run it"** → Go to [Getting Started](GETTING_STARTED.md)
- **"I'm visual, show me diagrams"** → Start with **Visual Path**
- **"I like to read and understand theory"** → Start with **Conceptual Path**
- **"I learn by doing"** → Start with **Code-First Path**
- **"I want the whole picture"** → Do **Comprehensive Path** (all of above)

---

## Path 1: Visual Learner 🎨

**Time: 2-3 hours**

Best if: You prefer diagrams, visualizations, and seeing how things connect.

### Stage 1: High-Level Picture (15 min)

1. Read [Architecture Overview](ARCHITECTURE.md#model-overview)
2. Look at the data flow diagram (skip the detailed explanations for now)
3. Scan the component breakdown section (just headings)

**Takeaway**: "A transformer is tokens → embeddings → 12 blocks of attention/FFN → output"

### Stage 2: Interactive Notebooks (1 hour)

Start Jupyter and explore:

```bash
jupyter notebook
```

Open notebooks in this order:

1. **`extra/Appendix_A.ipynb`** (30 min)
   - Focus on visualizations of tensor shapes
   - Pay attention to reshape and transpose sections
   - Skip math proofs if they feel heavy

2. **`src/1_dataloader.ipynb`** (15 min)
   - See how "Hello world" becomes [15496, 1917]
   - Watch shape transformations

3. **`src/2_dataloader_multihead-attention.ipynb`** (15 min)
   - **Most important**: Watch the attention visualization
   - See how attention "focuses" on relevant tokens
   - Don't worry if the math feels abstract

### Stage 3: Architecture Components (1 hour)

Read these sections from [Architecture](ARCHITECTURE.md):

1. **Embeddings** (10 min)
   - Look at the + symbol: why add embeddings?

2. **Transformer Block** (15 min)
   - Trace the flow diagram
   - Notice pre-norm before each layer

3. **Multi-Head Attention** (15 min)
   - Focus on the "Why multiple heads?" section
   - Look at causal masking diagram

4. **Output Layer** (10 min)
   - Quick section, tie it all together

### Stage 4: See the Code (30 min)

Open `src/3_compact_gpt_architecture.py` and find:

1. Search for `class MultiHeadAttention` (line ~65)
   - Find the attention visualization comment
   - Understand what Q, K, V mean

2. Search for `class TransformerBlock` (line ~100)
   - Trace attention → FFN → add pattern

3. Search for `class GPTModel` (line ~130)
   - See how blocks are stacked

4. Don't need to understand every line — get the flow

### Stage 5: Run It Yourself (15 min)

```bash
python src/3_compact_gpt_architecture.py
```

Modify the prompt and run again:

```python
# In main(), change this line:
prompt_tokens = model.generate(starting_tokens, max_new_tokens=20)

# Try different prompts:
# "What is"
# "The future"
# "In the beginning"
```

**Takeaway**: You can now explain attention and why transformers work to someone else!

---

## Path 2: Conceptual Learner 📚

**Time: 3-4 hours**

Best if: You like understanding the "why" and "how," reading explanations and papers.

### Stage 1: Big Picture (30 min)

Read [Concepts](CONCEPTS.md):

1. **Tokenization** section
   - Why do we tokenize?
   - How does BPE work?

2. **Embeddings** section
   - Why are embeddings necessary?
   - Why sum token + position embeddings?

### Stage 2: Core Mechanisms (1.5 hours)

Read [Concepts](CONCEPTS.md):

1. **Self-Attention Mechanism** (45 min)
   - Read the explanation + equations
   - Work through the numerical example
   - Understand Query, Key, Value

2. **Multi-Head Attention** (30 min)
   - Why use multiple heads?
   - How are they combined?

### Stage 3: Architecture Deep Dive (1 hour)

Read [Architecture](ARCHITECTURE.md):

1. **Component Breakdown** (30 min)
   - Focus on Attention, Transformer Block, FFN sections
   - Read the "Why" explanations

2. **Key Design Decisions** (30 min)
   - Pre-norm vs post-norm
   - Causal masking explained
   - Learnable vs fixed embeddings

### Stage 4: Connect to Code (1 hour)

Read [Components](COMPONENTS.md):

For each major class (in order):

1. **MultiHeadAttention**
   - Implementation notes
   - How does it match the theory?

2. **TransformerBlock**
   - How are layers arranged?

3. **GPTModel**
   - How does training vs inference differ?

4. **generate_text_simple**
   - Greedy vs. sampling

### Stage 5: Reference Papers (Optional, self-paced)

Read these in order:

1. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Original transformer)
   - Abstract + Introduction
   - Sections 3.1-3.3 (Attention, Multi-Head Attention)
   - Sections 4-5 (Model, training)

2. **[GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**
   - Sections 1-2 (Intro, why causal language modeling)

**Note**: Don't get stuck on math. The intuition is in the text, equations are for reference.

### Stage 6: Hands-On Experiment (30 min)

Run and modify:

```bash
python src/3_compact_gpt_architecture.py
```

Try changing hyperparameters in `main()`:

```python
model = GPTModel(
    vocab_size=200_000,
    context_length=512,    # Make it smaller
    emb_dim=256,          # Make it smaller
    n_heads=4,            # Reduce heads
    n_layers=2,           # Reduce layers
)
```

Run it — does output quality change? How fast?

**Takeaway**: You understand the theory and can explain design choices!

---

## Path 3: Code-First Learner 💻

**Time: 2-3 hours**

Best if: You want to run code, modify it, and learn by experimenting.

### Stage 1: Get It Running (15 min)

1. Follow [Getting Started](GETTING_STARTED.md)
2. Run the demo successfully
3. Verify: prompt → output

### Stage 2: Interactive Exploration (45 min)

Open Python interpreter in project directory:

```bash
python
```

Try this:

```python
import torch
from src.model import GPTModel

# Create model
model = GPTModel()
print(model)  # See architecture

# Check sizes
print(f"Vocab size: {model.vocab_size}")
print(f"Context length: {model.context_length}")

# Forward pass
x = torch.randint(0, 100, (1, 50))  # Dummy input
output = model(x)
print(output.shape)  # Should be (1, 50, 200000)
```

Explore:
- What happens if you make `context_length` smaller?
- What happens if you make `emb_dim` smaller?

### Stage 3: Modify and Test (1 hour)

Create a test file `my_test.py`:

```python
import torch
from src.model import GPTModel

# Experiment 1: Tiny model
tiny = GPTModel(
    vocab_size=200_000,
    context_length=128,  # Small
    emb_dim=64,          # Tiny
    n_heads=2,
    n_layers=1,
)

# Experiment 2: Generate text
prompt = "Hello"
output = tiny.generate(prompt, max_new_tokens=10)
print(output)

# Experiment 3: Check parameter count
total_params = sum(p.numel() for p in tiny.parameters())
print(f"Parameters: {total_params:,}")
```

Run it and observe:
- How many parameters does a tiny model have?
- How much faster does it run?
- Does output quality matter for small models?

### Stage 4: Read Code (1 hour)

Open `src/3_compact_gpt_architecture.py` and read in this order:

1. **Imports and device setup** (lines 1-15)
   - What libraries does it use?

2. **GPTDataset class** (lines 17-40)
   - How does it convert text to tensors?

3. **MultiHeadAttention class** (lines 60-95)
   - What are Q, K, V?
   - Where is causal masking applied?

4. **TransformerBlock class** (lines 115-140)
   - Why are there two layer norms?

5. **GPTModel class** (lines 150-190)
   - How are parameters initialized?
   - Why are there two embedding layers?

6. **generate_text_simple** (lines 195-210)
   - Step through the logic
   - What does argmax do?

### Stage 5: Modify and Break Things (45 min)

Intentionally modify code to understand it:

**Experiment A**: Remove causal masking
- In `MultiHeadAttention`, comment out the mask
- What breaks? Why?

**Experiment B**: Change embedding dimension
- Set `emb_dim=128` in main
- What changes? Size? Speed?

**Experiment C**: Change number of heads
- Set `n_heads=1`
- Does output change?

**Experiment D**: Try different prompts
- See how changing prompt affects generation
- Try multi-word vs single word

### Stage 6: Read Explanations (30 min)

Now that you've played with code, read:

1. [Architecture](ARCHITECTURE.md) — sections on Attention and Transformer Block
2. [Components](COMPONENTS.md) — the classes you've been modifying

Explanations will make more sense now!

**Takeaway**: You can modify and debug the code. You own this codebase!

---

## Path 4: Comprehensive Path 🎓

**Time: 5-6 hours**

Want it all? Do all three paths in sequence for deepest understanding:

1. **Visual Path** (2-3 hours) — Build intuition
2. **Conceptual Path** (3-4 hours) — Understand theory
3. **Code-First Path** (2-3 hours) — Master implementation

The paths overlap intentionally — repetition from different angles locks in understanding.

**Optional extras**:
- Read the papers
- Experiment with different architectures
- Create your own simple attention implementation

---

## After the Learning Path

### Level: Junior Practitioner

You can now:
- ✅ Run and understand the model
- ✅ Modify hyperparameters
- ✅ Explain transformers to others
- ✅ Read research papers

### Next steps:

**Understand more components**
- Read [Components](COMPONENTS.md) for code walkthroughs
- See [API Reference](API_REFERENCE.md) for formal documentation

**Modify the model**
- See [Examples](EXAMPLES.md) for code patterns
- Try the experiments in "Code-First" path

**Go deeper on concepts**
- See [Concepts](CONCEPTS.md) for theoretical deep-dives

**Learn to contribute**
- See [Contributing](CONTRIBUTING.md)

**Have questions?**
- Check [FAQ](FAQ.md)

---

## Tips for Learning

### 1. **Take Notes**

Write a 1-page summary after each major section. Forces you to consolidate.

### 2. **Run the Code**

Don't just read — actually execute, modify, break it. Mistakes teach fast.

### 3. **Draw Diagrams**

Sketch out the data flow. Show a friend. Explaining cements understanding.

### 4. **Experiment**

- Change hyperparameters → see what breaks
- Read a section → try implementing it yourself
- Combine paths → visual + code together

### 5. **Ask Questions**

- See [FAQ](FAQ.md) for common ones
- Modify code to test hypotheses
- Post issues/discussions if stuck

### 6. **Don't Memorize**

You don't need to memorize matrix operations. Focus on intuition and knowing *where* to look.

---

## Learning Resources Beyond This Project

### Interactive Tutorials

- **[3Blue1Brown's Attention Video](https://www.youtube.com/watch?v=eMlx5aFJsQM)** — Brilliant visual explanation
- **[Transformer Circuits](https://transformer-circuits.pub/)** — Deep dive with visuals

### Beginner-Friendly Papers

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — Original transformer (dense but foundational)
- **[A Gentle Introduction to Transformers](https://machinelearningmastery.com/the-transformer-model/)** — Blog post, very approachable

### Production Implementations

- **[Hugging Face Transformers](https://huggingface.co/transformers/)** — See how it's done at scale
- **[PyTorch Transformers](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)** — Reference implementation

---

## FAQ

**Q: How long should each path take?**
A: Visual = 2-3h, Conceptual = 3-4h, Code-First = 2-3h. Comprehensive = 5-6h if you do all three.

**Q: Can I skip a path?**
A: Yes! Pick the one that matches your style. They overlap enough that you'll fill gaps.

**Q: What if I get stuck?**
A: Check [FAQ](FAQ.md). Try modifying code. Come back to it later — sometimes breaks click overnight.

**Q: Should I memorize the formulas?**
A: No. Understand the intuition. You can always look up the formula.

---

**Ready to start? Pick your path above and dive in!** 🚀
