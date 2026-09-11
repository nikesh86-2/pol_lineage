"""Configuration loading with dotted-path access and CLI-style overrides.

The canonical configuration is ``config/config.yaml``.  Values may be overridden
from the command line using ``dotted.path=value`` pairs, where ``value`` is
parsed as YAML so that numbers, booleans and lists round-trip correctly::

    from hbvpol.config import load_config
    cfg = load_config("config/config.yaml", overrides=["selection.window=50"])
    cfg["selection"]["window"]   # -> 50
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

__all__ = ["load_config", "merge", "get", "parse_overrides", "resolve_path"]


def parse_overrides(overrides: Iterable[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=true"]`` into ``{"a": {"b": 1}, "c": True}``."""
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins)."""
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: str | os.PathLike[str], overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load YAML config, apply dotted overrides, and expand ``${ENV}`` values."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if overrides:
        merge(cfg, parse_overrides(overrides))
    return _expand_env(cfg)


def _expand_env(node: Any) -> Any:
    if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
        return os.environ.get(node[2:-1], "")
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    return node


def get(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch a nested value by dotted path, returning ``default`` if absent."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(config: Mapping[str, Any], dotted: str, root: str | os.PathLike[str] = ".") -> Path:
    """Resolve a configured path relative to the project root."""
    value = get(config, dotted)
    if value is None:
        raise KeyError(f"missing config path {dotted!r}")
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path
