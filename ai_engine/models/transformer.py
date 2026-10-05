"""Transformer-based sequence classifier."""

from __future__ import annotations

import torch
from torch import nn

from ai_engine.config.schema import ModelConfig
from ai_engine.models.base import BaseModel
from ai_engine.models.layers import SinusoidalPositionalEncoding, TransformerEncoderBlock


class TransformerClassifier(BaseModel):
    """Transformer encoder with mean-pooling classification head.

    Args:
        vocab_size: Token vocabulary size.
        num_classes: Number of output classes.
        d_model: Embedding / model dimension.
        n_heads: Attention heads per layer.
        n_layers: Number of encoder blocks.
        dim_feedforward: FFN inner dimension.
        max_seq_len: Maximum supported sequence length.
        dropout: Dropout probability.
        activation: FFN activation name.
        pad_token_id: Padding token id used to build key masks.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Encode token ids and produce class logits.

        Args:
            features: Long tensor of token ids ``(batch, seq)``.
            attention_mask: Optional bool mask ``(batch, seq)`` where ``True``
                marks valid (non-pad) tokens. If omitted, pads are inferred from
                ``pad_token_id``.
            **kwargs: Ignored.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        del kwargs
        if features.dtype != torch.long:
            features = features.long()

        if attention_mask is None:
            attention_mask = features.ne(self.pad_token_id)

        x = self.embedding(features) * (self.embedding.embedding_dim**0.5)
        x = self.positional(x)

        # Build additive attention mask: True means "mask out".
        # Shape broadcast to (batch, 1, 1, seq) for key masking.
        key_padding = ~attention_mask.bool()
        attn_mask = key_padding[:, None, None, :]

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)

        mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)

    @classmethod
    def from_config(cls, config: ModelConfig) -> TransformerClassifier:
        """Construct a :class:`TransformerClassifier` from :class:`ModelConfig`."""
        return cls(
            vocab_size=config.vocab_size,
            num_classes=config.num_classes,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            activation=config.activation,
        )
