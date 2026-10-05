"""Typed configuration schemas for the AI Engine framework.

All runtime settings are validated with Pydantic models so misconfiguration
fails early with clear error messages rather than at training time.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Precision(str, Enum):
    """Supported mixed-precision modes."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class OptimizerName(str, Enum):
    """Supported optimizer backends."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"


class SchedulerName(str, Enum):
    """Supported learning-rate scheduler backends."""

    COSINE = "cosine"
    STEP = "step"
    NONE = "none"


class DataConfig(BaseModel):
    """Configuration for dataset loading and preprocessing.

    Attributes:
        train_path: Path to training data (CSV, NPY, or directory of tensors).
        val_path: Optional path to validation data.
        test_path: Optional path to test / hold-out data.
        batch_size: Mini-batch size used by DataLoaders.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to pin host memory for faster GPU transfer.
        shuffle: Whether to shuffle the training set each epoch.
        max_length: Maximum sequence length for sequence datasets.
        normalize: Whether to apply feature-wise standardization.
        feature_columns: Optional explicit feature column names for tabular CSV.
        label_column: Label column name for tabular CSV.
        cache_preprocessed: Cache transformed samples in memory when feasible.
    """

    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    shuffle: bool = True
    max_length: int = Field(default=512, ge=1)
    normalize: bool = True
    feature_columns: list[str] | None = None
    label_column: str = "label"
    cache_preprocessed: bool = False


class ModelConfig(BaseModel):
    """Configuration for model architecture construction.

    Attributes:
        name: Registered model name (e.g. ``mlp``, ``transformer_classifier``).
        input_dim: Input feature dimension for feed-forward models.
        num_classes: Number of output classes (classification).
        hidden_dims: Hidden layer sizes for MLP-style models.
        dropout: Dropout probability applied in residual / FFN blocks.
        vocab_size: Vocabulary size for sequence / embedding models.
        d_model: Transformer model / embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        dim_feedforward: Inner dimension of the transformer FFN.
        max_seq_len: Maximum positional encoding length.
        activation: Activation function name.
        pretrained_path: Optional path to pretrained weights.
    """

    name: str = "mlp"
    input_dim: int = Field(default=128, ge=1)
    num_classes: int = Field(default=2, ge=1)
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    vocab_size: int = Field(default=10000, ge=1)
    d_model: int = Field(default=256, ge=1)
    n_heads: int = Field(default=8, ge=1)
    n_layers: int = Field(default=4, ge=1)
    dim_feedforward: int = Field(default=1024, ge=1)
    max_seq_len: int = Field(default=512, ge=1)
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    pretrained_path: Path | None = None

    @model_validator(mode="after")
    def validate_attention_heads(self) -> ModelConfig:
        """Ensure ``d_model`` is divisible by ``n_heads`` for multi-head attention."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self


class OptimizerConfig(BaseModel):
    """Optimizer hyperparameters.

    Attributes:
        name: Optimizer backend identifier.
        lr: Peak learning rate.
        weight_decay: L2 / decoupled weight decay coefficient.
        betas: Adam / AdamW beta coefficients.
        momentum: SGD momentum (ignored by Adam variants).
        eps: Numerical stability epsilon for Adam variants.
    """

    name: OptimizerName = OptimizerName.ADAMW
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    eps: float = Field(default=1e-8, gt=0.0)


class SchedulerConfig(BaseModel):
    """Learning-rate scheduler hyperparameters.

    Attributes:
        name: Scheduler backend identifier.
        warmup_steps: Linear warmup steps before the main schedule.
        min_lr: Minimum learning rate for cosine annealing.
        step_size: Epoch / step interval for step decay.
        gamma: Multiplicative decay factor for step decay.
    """

    name: SchedulerName = SchedulerName.COSINE
    warmup_steps: int = Field(default=0, ge=0)
    min_lr: float = Field(default=1e-6, ge=0.0)
    step_size: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.1, gt=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """End-to-end training loop configuration.

    Attributes:
        output_dir: Directory for checkpoints, logs, and metrics.
        epochs: Maximum number of training epochs.
        max_steps: Optional hard cap on optimizer steps (overrides epochs when set).
        grad_clip_norm: Global gradient-norm clip threshold; ``None`` disables clipping.
        precision: Mixed-precision mode.
        seed: Global RNG seed for reproducibility.
        log_every: Log training metrics every N steps.
        eval_every: Run validation every N steps (in addition to epoch-end eval).
        save_every: Persist a checkpoint every N epochs.
        early_stopping_patience: Stop if validation metric does not improve for N evals.
        early_stopping_metric: Validation metric name monitored for early stopping.
        early_stopping_mode: Whether higher or lower metric values are better.
        accumulate_grad_batches: Gradient accumulation steps for effective larger batches.
        compile_model: Use ``torch.compile`` when available (PyTorch 2+).
        device: Preferred device string (``cpu``, ``cuda``, ``cuda:0``, ``mps``).
    """

    output_dir: Path = Path("runs/default")
    epochs: int = Field(default=10, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    grad_clip_norm: float | None = Field(default=1.0, gt=0.0)
    precision: Precision = Precision.FP32
    seed: int = 42
    log_every: int = Field(default=10, ge=1)
    eval_every: int | None = Field(default=None, ge=1)
    save_every: int = Field(default=1, ge=1)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: Literal["min", "max"] = "min"
    accumulate_grad_batches: int = Field(default=1, ge=1)
    compile_model: bool = False
    device: str = "auto"


class InferenceConfig(BaseModel):
    """Configuration for efficient batched inference.

    Attributes:
        checkpoint_path: Path to a trained checkpoint directory or ``.pt`` file.
        batch_size: Inference mini-batch size.
        device: Preferred device string or ``auto``.
        precision: Inference precision mode.
        max_length: Maximum sequence length for sequence models.
        num_workers: DataLoader workers for offline inference.
        use_amp: Enable autocast during inference when precision is not FP32.
        temperature: Softmax temperature for probabilistic outputs.
        top_k: Optional top-k filtering for generative decoding.
        top_p: Optional nucleus (top-p) filtering for generative decoding.
    """

    checkpoint_path: Path | None = None
    batch_size: int = Field(default=64, ge=1)
    device: str = "auto"
    precision: Precision = Precision.FP32
    max_length: int = Field(default=512, ge=1)
    num_workers: int = Field(default=0, ge=0)
    use_amp: bool = True
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class EngineConfig(BaseModel):
    """Top-level aggregate configuration for the AI Engine.

    Attributes:
        data: Dataset / dataloader settings.
        model: Architecture settings.
        optimizer: Optimizer settings.
        scheduler: LR schedule settings.
        training: Trainer loop settings.
        inference: Inference engine settings.
        experiment_name: Human-readable experiment identifier.
        tags: Optional free-form tags for experiment tracking.
        extra: Escape hatch for custom user keys.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    experiment_name: str = "default"
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_name")
    @classmethod
    def non_empty_name(cls, value: str) -> str:
        """Reject blank experiment names."""
        if not value.strip():
            raise ValueError("experiment_name must be a non-empty string")
        return value.strip()
