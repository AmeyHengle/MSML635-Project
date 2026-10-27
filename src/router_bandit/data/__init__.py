"""Dataset layer: loading, featurising, splitting, and replaying RouterBench."""

from .cache import cache_path, config_hash, load_arrays, save_arrays
from .features import (
    FeatureBackend,
    SentenceTransformerBackend,
    TfidfSvdBackend,
    build_features,
)
from .loaders import RouterBenchData, build, clean, load, load_raw, parse_prompt
from .registry import (
    MODELS,
    N_ARMS,
    PREMIUM_MODELS,
    SHORT_NAMES,
    arm_index,
    cost_col,
    eval_to_family,
    quality_col,
    short,
    short_names,
)
from .splits import Splits, make_splits
from .stream import RouterBenchStream, Stream, build_stream
from .synthetic import SyntheticStream

__all__ = [
    "MODELS",
    "N_ARMS",
    "PREMIUM_MODELS",
    "SHORT_NAMES",
    "FeatureBackend",
    "RouterBenchData",
    "RouterBenchStream",
    "SentenceTransformerBackend",
    "Splits",
    "Stream",
    "SyntheticStream",
    "TfidfSvdBackend",
    "arm_index",
    "build",
    "build_features",
    "build_stream",
    "cache_path",
    "clean",
    "config_hash",
    "cost_col",
    "eval_to_family",
    "load",
    "load_arrays",
    "load_raw",
    "make_splits",
    "parse_prompt",
    "quality_col",
    "save_arrays",
    "short",
    "short_names",
]
