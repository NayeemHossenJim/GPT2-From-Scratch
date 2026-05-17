# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite (8 docs files)
- Docstrings for all classes and functions
- Multiple learning paths for different learners
- API reference with complete parameter documentation
- Practical examples (10 different usage patterns)
- Contributing guidelines
- FAQ with 30+ common questions

### Changed
- Updated README to focus on quick start
- Reorganized documentation for better navigation

## [1.0.0] - 2026-05-18

### Added
- Core GPT model implementation (12-layer, 12-head transformer)
- Multi-head self-attention with causal masking
- Layer normalization with learnable parameters
- Position-wise feed-forward networks
- Greedy text generation
- GPTDataset class for tokenization and batching
- Support for tiktoken BPE tokenization (200K vocabulary)
- Educational notebooks demonstrating concepts:
  - Tensor operations fundamentals
  - Dataloader mechanics
  - Attention mechanism visualization
- Appendix materials for tensor basics
- Configuration-driven model initialization
- Support for both CPU and GPU inference

### Documentation
- README with project overview
- Architecture guide explaining model design
- Getting started guide for first-time users
- Components guide with detailed code walkthroughs
- Concepts guide covering fundamental topics
- Learning paths for visual, conceptual, and code-first learners
- API reference with complete function signatures
- Practical examples with 10 usage patterns
- Contributing guidelines
- Comprehensive FAQ
- This changelog

### Features
- Model configurations from tiny (3M) to large (1.3B) parameters
- Pre-norm residual connections (modern design)
- Causal masking for language modeling
- Dropout for regularization
- Configurable context length (512-2048 tokens)
- Batch generation support
- Temperature-controlled sampling ready

---

## Vision for Future Versions

### Planned (v1.1)
- Training loop with next-token prediction loss
- Optimizer implementations (SGD, Adam)
- Validation metrics (perplexity, loss)
- Checkpoint saving/loading

### Considered (v2.0)
- Distributed training utilities
- Different attention patterns (local, sparse)
- Model quantization (int8, float16)
- Inference optimizations (KV cache)
- Different model architectures (BERT-style bidirectional)

### Out of Scope
- Production deployment
- Distributed multi-GPU/TPU training
- State-of-the-art performance optimization
- Support for other languages (focus on English education)

---

## Migration Guide

### From 0.x to 1.0

No breaking changes. This is the first stable release.

---

## Notes

This project prioritizes educational clarity over production optimization. Performance improvements are welcome, but not at the expense of readability or learning value.
