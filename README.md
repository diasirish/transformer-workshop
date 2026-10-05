# Transformer Workshop

A personal library of attention mechanisms and Transformer architectures, implemented by hand in PyTorch.

I am building this project to understand language models from their fundamental operations: tensor shapes, learned projections, attention scores, masks, gradients, and training objectives. The goal is to make that understanding visible through readable implementations, worked experiments, and explanations of the choices behind the code.

PyTorch provides tensor operations and automatic differentiation. I write the attention calculations and assemble the architectures explicitly, progressing from a single attention head to small models that can be trained and evaluated.

## Current Progress

The starting point is [the self-attention notebook](attention_mechanism/transformer.ipynb). It implements a single head of scaled dot-product self-attention, including a forward pass and parameter-gradient inspection.

The notebook stores tokens in columns: an input with shape `(D, I)` contains `I` tokens, each with `D` embedding features. This follows the convention used in *Understanding Deep Learning*.

## Roadmap

- [x] Single-head scaled dot-product self-attention.
- [ ] Extract the attention implementation into a reusable Python module.
- [ ] Support batches of sequences and verify tensor shapes and gradients.
- [ ] Add padding masks and causal masks.
- [ ] Implement multi-head self-attention.
- [ ] Implement cross-attention between two sequences.
- [ ] Build Transformer blocks with residual connections, normalization, and feed-forward networks.
- [ ] Add token embeddings and positional information.
- [ ] Build a BERT-like encoder with a masked language modeling objective.
- [ ] Build a small GPT-like decoder with autoregressive training and generation.
- [ ] Build an encoder-decoder Transformer for translation.
- [ ] Explore further attention and architecture variations, recording their tradeoffs.

## Repository Layout

```text
transformer-workshop/
    attention_mechanism/
        README.md
        transformer.ipynb
    AGENTS.md
    README.md
    .gitignore
```

`attention_mechanism` is the first area of the workshop and lives in the same repository. Further modules and model directories will be added as their implementations are developed.

## How I Work

For each mechanism or architecture, I aim to:

1. Explain the equations and assign a meaning to each tensor dimension.
2. Implement the operations explicitly and inspect a small example.
3. Check the result against a manual calculation or an independent reference.
4. Verify gradients, masking behavior, and relevant edge cases.
5. Train a small model when the architecture is ready and record what it learns.

Learning notes, intermediate experiments, and unfinished milestones are part of the project. The roadmap describes intended work; completed implementations are linked above.

## Running the Starting Notebook

Use a Python environment with PyTorch and Jupyter installed, select that environment as the notebook kernel, and open `attention_mechanism/transformer.ipynb` in Jupyter or a notebook-capable editor. The current experiment runs on CPU.

## Reference

The initial attention implementation follows the mathematical presentation in Simon J. D. Prince's [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Chapter 12.
