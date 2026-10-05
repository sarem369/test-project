"""Configuration loading helpers (YAML / JSON / dict)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ai_engine.config.schema import EngineConfig


def _load_raw(path: Path) -> dict[str, Any]:
    """Load a raw configuration dictionary from disk.

    Args:
        path: Path to a ``.yaml``, ``.yml``, or ``.json`` file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported or content is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text) or {}
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {suffix} (use .yaml, .yml, or .json)")

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    return raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> EngineConfig:
    """Load and validate an :class:`EngineConfig` from a file.

    Args:
        path: Path to a YAML or JSON configuration file.
        overrides: Optional nested dictionary of values that take precedence
            over file contents (useful for CLI flags).

    Returns:
        A validated :class:`EngineConfig` instance.
    """
    raw = _load_raw(Path(path))
    if overrides:
        raw = deep_merge(raw, overrides)
    return EngineConfig.model_validate(raw)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Args:
        base: Base dictionary.
        override: Values that win on key conflicts. Nested dicts are merged.

    Returns:
        A new merged dictionary (inputs are not mutated).
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: EngineConfig, path: str | Path) -> Path:
    """Serialize an :class:`EngineConfig` to YAML on disk.

    Args:
        config: Configuration instance to persist.
        path: Destination file path (parent directories are created as needed).

    Returns:
        The resolved destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination.resolve()
