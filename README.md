# Transformer Workshop

A personal PyTorch workshop for understanding modern LLMs and building a handwritten code reference, starting from Simon J. D. Prince's [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Chapter 12.

## Current progress

The [attention notebook](attention_mechanism/transformer.ipynb) implements single-head scaled dot-product attention and gradient inspection. Input shape is `(D, I)`: tokens are columns. Batching, masks, multiple heads, and a complete language model are still planned.

To run it, open the notebook in Jupyter with PyTorch installed. The current experiment runs on CPU.

## Plan

1. Build a tiny decoder with training and generation.
2. Learn modern blocks: RoPE, RMSNorm, and SwiGLU.
3. Study efficient inference: KV caching, GQA, FlashAttention, paging, and MLA.
4. Read DeepSeek-V2/V3 to connect attention, MoE, and training efficiency.
5. Study post-training and reasoning, then read DeepSeek-R1.
6. Explore sparse attention, hybrid models, and other directions of interest.

Learn individual mechanisms before tackling papers that combine them. BERT and translation models remain optional comparison projects. All steps above are future work.

## Study method and references

**Read → derive → code a small version → test → write a cheat sheet.** Explain tensor shapes, check against an independent reference, and distinguish measured results from paper claims.

- [Technology roadmap](docs/TECHNOLOGY_ROADMAP.md): topics, purpose, and checkpoints.
- [Paper list](docs/PAPERS.md): original sources in study order.
- [Note template](docs/PAPER_NOTE_TEMPLATE.md): a short record for each mechanism.
