"""Reusable neural network building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """Instantiate an activation module by name.

    Args:
        name: One of ``relu``, ``gelu``, or ``silu``.

    Returns:
        An :class:`nn.Module` activation.

    Raises:
        ValueError: If ``name`` is unrecognized.
    """
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside transformer blocks.

    Args:
        d_model: Input / output embedding dimension.
        dim_feedforward: Inner hidden dimension.
        dropout: Dropout probability.
        activation: Activation function name.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            Transformed tensor of the same shape as ``x``.
        """
        return self.net(x)


class ResidualNorm(nn.Module):
    """Pre-norm residual wrapper around a submodule.

    Args:
        dim: Feature dimension for layer normalization.
        dropout: Residual dropout probability.
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Apply pre-norm → sublayer → residual add.

        Args:
            x: Input tensor.
            sublayer: Module consuming the normalized input.

        Returns:
            Residual output tensor.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadSelfAttention(nn.Module):
    """Scaled multi-head self-attention.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input embeddings of shape ``(batch, seq, d_model)``.
            attn_mask: Optional boolean / additive mask broadcastable to
                ``(batch, n_heads, seq, seq)``. ``True`` / ``-inf`` means masked.

        Returns:
            Contextualized embeddings of shape ``(batch, seq, d_model)``.
        """
        batch, seq, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).contiguous().reshape(batch, seq, self.d_model)
        return self.out_proj(context)


class TransformerEncoderBlock(nn.Module):
    """Single pre-norm transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.attn_residual = ResidualNorm(d_model, dropout=dropout)
        self.ff_residual = ResidualNorm(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply attention and feed-forward sublayers with residuals.

        Args:
            x: Input embeddings ``(batch, seq, d_model)``.
            attn_mask: Optional attention mask.

        Returns:
            Encoded embeddings of the same shape.
        """
        x = self.attn_residual(x, lambda h: self.attn(h, attn_mask=attn_mask))
        x = self.ff_residual(x, self.ff)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_seq_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to token embeddings.

        Args:
            x: Token embeddings of shape ``(batch, seq, d_model)``.

        Returns:
            Position-aware embeddings.
        """
        seq = x.size(1)
        x = x + self.encoding[:, :seq]
        return self.dropout(x)
