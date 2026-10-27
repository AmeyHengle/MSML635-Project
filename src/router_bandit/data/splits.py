"""Stratified, disjoint splits.

Three roles, deliberately separated:

``calibration``
    The only split the feature reducer and the full-information supervised
    router may see. Keeping it separate is what makes the supervised baseline
    an honest comparison rather than an oracle in disguise.

``stream``
    Replayed as the online bandit stream.

``heldout``
    Frozen final evaluation, never touched during development.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["Splits", "make_splits"]


@dataclass(frozen=True)
class Splits:
    calibration: np.ndarray
    stream: np.ndarray
    heldout: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "calibration": self.calibration,
            "stream": self.stream,
            "heldout": self.heldout,
        }

    def check_disjoint(self, n_total: int) -> None:
        """Raise if the splits overlap or fail to cover the dataset."""
        parts = list(self.as_dict().values())
        concat = np.concatenate(parts)
        if concat.size != n_total:
            raise ValueError(f"splits cover {concat.size} rows, expected {n_total}")
        if np.unique(concat).size != n_total:
            raise ValueError("splits overlap: an index appears in more than one split")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Splits(calibration={self.calibration.size}, "
            f"stream={self.stream.size}, heldout={self.heldout.size})"
        )


def make_splits(
    families: pd.Series | np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.15, 0.70, 0.15),
    seed: int = 0,
) -> Splits:
    """Split indices into calibration/stream/heldout, stratified by family.

    Stratification matters because family proportions are very uneven (MMLU is
    ~39% of rows, MBPP ~1%), so an unstratified split can leave a family almost
    absent from calibration.

    Parameters
    ----------
    families:
        Per-query family label, length ``n``.
    fractions:
        ``(calibration, stream, heldout)``; must sum to 1.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions} -> {sum(fractions)}")
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative, got {fractions}")

    labels = np.asarray(pd.Series(families).to_numpy())
    n = labels.shape[0]
    rng = np.random.default_rng(seed)

    buckets: list[list[np.ndarray]] = [[], [], []]
    f_cal, f_str, _ = fractions

    for fam in pd.unique(labels):
        idx = np.flatnonzero(labels == fam)
        rng.shuffle(idx)
        n_fam = idx.size
        cut_a = round(f_cal * n_fam)
        cut_b = cut_a + round(f_str * n_fam)
        # Tiny families: guarantee the stream is non-empty before anything else.
        if n_fam < 3:
            cut_a, cut_b = 0, n_fam
        for bucket, part in zip(buckets, (idx[:cut_a], idx[cut_a:cut_b], idx[cut_b:])):
            if part.size:
                bucket.append(part)

    def _join(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(parts)).astype(np.int64)

    splits = Splits(_join(buckets[0]), _join(buckets[1]), _join(buckets[2]))
    splits.check_disjoint(n)
    logger.info("Split %d queries -> %r", n, splits)
    return splits
