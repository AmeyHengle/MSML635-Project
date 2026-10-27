"""Content-addressed caching for processed arrays.

Featurising 36k queries is slow enough that repeating it on every lambda value
and every seed would dominate runtime. Keying the cache on a hash of the
config means a changed setting silently produces a new cache entry instead of
silently reusing a stale one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["cache_path", "config_hash", "load_arrays", "save_arrays"]

DEFAULT_CACHE_DIR = Path("data/processed")


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    """Short deterministic hash of a JSON-serialisable config."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def cache_path(name: str, config: dict[str, Any], cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{name}-{config_hash(config)}.npz"


def save_arrays(path: str | Path, arrays: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    if config is not None:
        payload["__config__"] = np.array(json.dumps(config, sort_keys=True, default=str))
    np.savez_compressed(path, **payload)
    logger.info("Cached %d arrays to %s", len(arrays), path)
    return path


def load_arrays(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any] | None] | None:
    """Return ``(arrays, config)`` or ``None`` if the cache entry is absent."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files if k != "__config__"}
        config = json.loads(str(handle["__config__"])) if "__config__" in handle.files else None
    logger.info("Loaded cache %s", path)
    return arrays, config
