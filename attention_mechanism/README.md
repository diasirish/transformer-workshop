# Attention Mechanisms

Handwritten PyTorch components for studying Transformers.

- [transformer.ipynb](transformer.ipynb): explanations, paper diagrams, and CPU/MPS experiments.
- [classic_transformer.py](classic_transformer.py): the notebook's seven reusable classes, without example execution or device selection on import.

Hidden states use `(B, I, D)`; token IDs use `(B, I)`. Single-head attention is unmasked. Multi-head attention and the post-LayerNorm Transformer block are causal. Sinusoidal positions are fixed buffers; the character tokenizer supports only its known vocabulary. The original initialization and class names are preserved.

From a notebook running at the repository root:

```python
from attention_mechanism.classic_transformer import (
    Embedding_Block,
    PositionalEncoding_Block,
    Transformer_Block,
)
```

From a notebook running inside this directory, use `from classic_transformer import ...`. Check `Path.cwd()` if an import fails.

Next: train a subword tokenizer, assemble a small GPT with a vocabulary prediction head, and implement next-token training and generation. Padding masks and cross-attention remain future exercises. Notebook definitions and the extracted module are separate copies; edits do not synchronize automatically.
