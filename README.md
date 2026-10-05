# AI Engine

A modular, scalable **Python deep-learning framework** built on PyTorch. It provides production-oriented building blocks for:

- **Data preprocessing** — transforms, CSV/NPY loaders, train/val splits, DataLoaders
- **Model architecture** — MLP and Transformer classifiers, reusable layers, registry
- **Training pipelines** — AMP, grad accumulation, clipping, callbacks, early stopping
- **Efficient inference** — micro-batching, probability outputs, TorchScript / ONNX export

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with the pinned requirements file:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
from ai_engine.config import load_config
from ai_engine.data import DataPreprocessor, create_dataloader, dataset_from_bundle, train_val_split, PreprocessedBundle
from ai_engine.models import build_model
from ai_engine.training import Trainer
from ai_engine.inference import InferenceEngine
import numpy as np

config = load_config("configs/mlp_example.yaml")

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, config.model.input_dim)).astype("float32")
y = rng.integers(0, config.model.num_classes, size=1000)

pre = DataPreprocessor(config.data)
bundle = pre.fit_transform(X, y)
x_tr, x_va, y_tr, y_va = train_val_split(bundle.features, bundle.labels, seed=0)

train_loader = create_dataloader(dataset_from_bundle(PreprocessedBundle(x_tr, y_tr)), config.data)
val_loader = create_dataloader(dataset_from_bundle(PreprocessedBundle(x_va, y_va)), config.data, shuffle=False)

model = build_model(config.model)
trainer = Trainer(model, config, train_loader, val_loader=val_loader)
trainer.fit()

engine = InferenceEngine(model, config)
preds = engine.predict(x_va[:32])
```

Run the bundled demo:

```bash
python examples/train_classifier.py
```

## Project layout

```
ai_engine/
  config/       # Pydantic schemas + YAML/JSON loading
  data/         # Transforms, preprocessor, datasets
  models/       # Layers, MLP, Transformer, registry
  training/     # Trainer, optim, schedulers, callbacks, metrics
  inference/    # InferenceEngine, batching, export
  utils/        # Logging, seeding, devices, checkpoints
configs/        # Example YAML configs
examples/       # End-to-end scripts
tests/          # Unit + integration tests
```

## Configuration

All runtime settings are validated with Pydantic (`EngineConfig`). Example files live in `configs/`.

| Section | Purpose |
|--------|---------|
| `data` | Paths, batch size, normalization, sequence length |
| `model` | Architecture name + hyperparameters |
| `optimizer` / `scheduler` | AdamW/SGD, cosine/step schedules |
| `training` | Epochs, AMP, clipping, early stopping, device |
| `inference` | Batch size, temperature, checkpoint path |

Load with overrides:

```python
from ai_engine.config import load_config
cfg = load_config("configs/mlp_example.yaml", overrides={"training": {"epochs": 3}})
```

## Models

Register custom architectures:

```python
from ai_engine.models import register_model, BaseModel
from ai_engine.config import ModelConfig

@register_model("my_model")
def build_my_model(config: ModelConfig) -> BaseModel:
    ...
```

Built-ins: `mlp`, `transformer_classifier`.

## Training features

- Device auto-selection (`cpu` / `cuda` / `mps`)
- Mixed precision (`fp16`, `bf16`) via `torch.autocast`
- Gradient accumulation + global-norm clipping
- Warmup + cosine LR schedule
- Checkpointing (periodic + best) and early stopping callbacks
- Optional `torch.compile`

## Inference features

- `InferenceEngine.predict` with automatic micro-batching
- Probability outputs with temperature scaling
- Streaming iterator for large collections
- `export_torchscript` / `export_onnx` helpers

```python
from ai_engine.inference import InferenceEngine
engine = InferenceEngine.from_checkpoint("runs/mlp_demo/best.pt", config=cfg)
proba = engine.predict(X, return_proba=True)
```

## CLI

```bash
ai-engine-train --config configs/mlp_example.yaml
ai-engine-infer --config configs/mlp_example.yaml --checkpoint runs/mlp_demo/best.pt --input features.npy
```

## Tests

```bash
pytest -q
```

## Design notes

- **Typed configs** fail fast on invalid hyperparameters.
- **Registry pattern** keeps architecture selection declarative.
- **Callback hooks** separate logging / checkpointing from the core loop.
- **Preprocessor state** (scalers, label maps) can be reused at inference time.

## License

MIT
