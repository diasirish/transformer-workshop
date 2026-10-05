# Attention Mechanisms

This directory is the starting point of Transformer Workshop: attention mechanisms implemented explicitly in PyTorch, with small experiments to understand their behavior.

## Current Implementation

[transformer.ipynb](transformer.ipynb) contains a single-head scaled dot-product self-attention experiment and a backward pass for inspecting learned parameter gradients.

The input convention is `(D, I)`: each column is a token, and each row is one embedding feature. With `num_att_heads=1`, the query, key, and value projection matrices have shape `(D, D)`.

The score matrix is computed as `Q.t() @ K`, so its rows represent query tokens and its columns represent key tokens. Softmax normalizes each row, and the output is a weighted sum of value vectors for each query token.

## Next Steps

- Extract a reusable Python module and support batches of sequences.
- Add padding and causal masks.
- Implement multi-head attention and cross-attention.
- Check each implementation against small examples and independent references.
