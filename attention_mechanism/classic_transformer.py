"""Handwritten Transformer components extracted from transformer.ipynb.

Hidden states use (B, I, D): batch, tokens, features. Token IDs use (B, I).
SelfAttention_Block is unmasked; MultiHead_SelfAttention_Block and
Transformer_Block use causal attention. Transformer_Block uses post-LayerNorm.
Initialization and class names are preserved from the learning notebook.
This module contains components, not a complete GPT or training loop.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "SelfAttention_Block",
    "MultiHead_SelfAttention_Block",
    "FCN_Layer",
    "Transformer_Block",
    "CharacterTokenizer",
    "Embedding_Block",
    "PositionalEncoding_Block",
]


class SelfAttention_Block(nn.Module):
    def __init__(self, D_dimension: int):
        super().__init__()
        self.D = D_dimension
        # linear projections  W_*  ∈ ℝ[D × D]
        self.Q_w = nn.Parameter(torch.randn(self.D, self.D))
        self.K_w = nn.Parameter(torch.randn(self.D, self.D))
        self.V_w = nn.Parameter(torch.randn(self.D, self.D))

        self.Q_b = nn.Parameter(torch.randn(self.D, 1))
        self.K_b = nn.Parameter(torch.randn(self.D, 1))
        self.V_b = nn.Parameter(torch.randn(self.D, 1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Shared projections: (B, I, D) @ (D, D) -> (B, I, D).
        Q = X @ self.Q_w.T + self.Q_b.T
        K = X @ self.K_w.T + self.K_b.T
        V = X @ self.V_w.T + self.V_b.T

        # One attention matrix per sequence; normalize over key tokens.
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.D)
        attn = self.softmax(scores)  # (B, I, I)
        context = attn @ V           # (B, I, D)
        return context


class MultiHead_SelfAttention_Block(nn.Module):
    def __init__(self, num_att_heads: int, D_dimension: int):
        super().__init__()
        assert num_att_heads > 0 and D_dimension % num_att_heads == 0, "D must be divisible by the number of heads"
        self.heads_n = num_att_heads
        self.D = D_dimension
        self.d_k = D_dimension // self.heads_n            # head size

        self.Q_w = nn.ParameterList()
        self.K_w = nn.ParameterList()
        self.V_w = nn.ParameterList()

        self.Q_b = nn.ParameterList()
        self.K_b = nn.ParameterList()
        self.V_b = nn.ParameterList()

        self.O_w = nn.Parameter(torch.randn(self.D, self.D))
        # linear projections  W_*  ∈ ℝ[d_k × D]
        for i in range(num_att_heads):
            self.Q_w.append(nn.Parameter(torch.randn(self.d_k, self.D)))
            self.K_w.append(nn.Parameter(torch.randn(self.d_k, self.D)))
            self.V_w.append(nn.Parameter(torch.randn(self.d_k, self.D)))

            self.Q_b.append(nn.Parameter(torch.randn(self.d_k, 1)))
            self.K_b.append(nn.Parameter(torch.randn(self.d_k, 1)))
            self.V_b.append(nn.Parameter(torch.randn(self.d_k, 1)))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        head_contexts = []
        for i in range(self.heads_n):
            # (B, I, D) @ (D, d_k) + (1, d_k) -> (B, I, d_k).
            # No batch dimension on weights: every sequence shares them.
            Q = X @ self.Q_w[i].T + self.Q_b[i].T
            K = X @ self.K_w[i].T + self.K_b[i].T
            V = X @ self.V_w[i].T + self.V_b[i].T

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

            I = X.shape[1]
            allowed = torch.tril(
                torch.ones(I, I, device=X.device, dtype=torch.bool)
                )
            scores = scores.masked_fill(~allowed, float("-inf"))

            attn = self.softmax(scores)  # (B, I, I); softmax over keys
            context = attn @ V           # (B, I, d_k)
            head_contexts.append(context)

        context = torch.cat(head_contexts, dim=-1)  # (B, I, D)
        output = context @ self.O_w.T               # (B, I, D)
        return output


class FCN_Layer(nn.Module):
    def __init__(self, D, hidden_dim):
        super().__init__()
        self.linear_1 = nn.Linear(D, hidden_dim)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, D)

    def forward(self, X):
        # Apply the three operations in order.
        hidden = self.linear_1(X)
        hidden = self.activation(hidden)
        output = self.linear_2(hidden)
        return output


class Transformer_Block(nn.Module):
    def __init__(self, D_dimension: int, num_att_heads: int, hidden_dim: int):
        super().__init__()
        self.multi_head_att = MultiHead_SelfAttention_Block(num_att_heads, D_dimension)
        self.layer_norm_1 = nn.LayerNorm(D_dimension)
        self.layer_norm_2 = nn.LayerNorm(D_dimension)
        self.fcn_layer = FCN_Layer(D_dimension, hidden_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Post-LayerNorm structure; normalize each token's last dimension.
        X = self.layer_norm_1(X + self.multi_head_att(X))
        # Linear layers share weights across all tokens and all sequences.
        X = self.layer_norm_2(X + self.fcn_layer(X))
        return X  # (B, I, D)


class CharacterTokenizer:
    def __init__(self, vocabulary_text: str):
        self.id_to_token = sorted(set(vocabulary_text))
        self.token_to_id = {
            token: i for i, token in enumerate(self.id_to_token)
        }

    def encode(self, text: str) -> list[int]:
        # Only characters in the vocabulary are supported for now.
        return [self.token_to_id[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_token[i] for i in token_ids)


class Embedding_Block(nn.Module):
    def __init__(self, vocab_size: int, D: int):
        super().__init__()
        # A trainable table: one D-dimensional vector per vocabulary entry.
        self.token_embeddings = nn.Embedding(vocab_size, D)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embeddings(token_ids)  # (B, I) -> (B, I, D)
        return embeddings  # Features are last; no transpose.


class PositionalEncoding_Block(nn.Module):
    def __init__(self, D: int, max_length: int = 128):
        super().__init__()
        if D <= 0 or max_length <= 0:
            raise ValueError("D and max_length must be positive")

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        # (max_length, 1): one row per position.
        pair_indices = torch.arange(0, D, 2, dtype=torch.float32)
        denominators = 10000 ** (pair_indices / D)
        angles = positions / denominators  # (max_length, ceil(D / 2))

        Pi = torch.zeros(max_length, D)
        Pi[:, 0::2] = torch.sin(angles)
        # If D is odd, there is one fewer cosine column.
        Pi[:, 1::2] = torch.cos(angles[:, :D // 2])
        # A buffer is saved and moves with .to(device), but is not learned.
        self.register_buffer("Pi", Pi.unsqueeze(0))  # (1, max_length, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3 or X.shape[-1] != self.Pi.shape[-1]:
            raise ValueError("Expected X with shape (B, I, D) matching the encoding")
        length = X.shape[1]
        if length > self.Pi.shape[1]:
            raise ValueError("Sequence exceeds the configured max_length")
        # Same position vectors for every sequence; preserve the input dtype.
        return X + self.Pi[:, :length, :].to(dtype=X.dtype)
