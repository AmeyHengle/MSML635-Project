"""Time-varying prices.

RouterBench stores a per-query dollar cost for every model, so cost is already
``cost(query, arm)`` rather than a per-arm constant. Scenario S2 adds a second
dependency -- on the round -- by re-pricing an arm part-way through the stream.
This module keeps that dependency in one place so nothing downstream caches a
stale price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CostModel", "PriceChange"]


@dataclass(frozen=True)
class PriceChange:
    """A multiplicative re-pricing of one arm, effective from ``round``."""

    round: int
    arm: int
    multiplier: float

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")
        if self.round < 0:
            raise ValueError(f"round must be non-negative, got {self.round}")


@dataclass
class CostModel:
    """Applies scheduled price changes to a base cost vector.

    Changes compose multiplicatively: two changes to the same arm both apply
    once their rounds have passed.
    """

    n_arms: int
    changes: tuple[PriceChange, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for change in self.changes:
            if not 0 <= change.arm < self.n_arms:
                raise ValueError(f"price change for arm {change.arm} outside [0, {self.n_arms})")

    def multipliers(self, t: int) -> np.ndarray:
        """``(n_arms,)`` price multipliers in force at round ``t``."""
        mult = np.ones(self.n_arms, dtype=np.float64)
        for change in self.changes:
            if t >= change.round:
                mult[change.arm] *= change.multiplier
        return mult

    def apply(self, t: int, base_cost: np.ndarray) -> np.ndarray:
        """Effective cost vector at round ``t``."""
        return base_cost * self.multipliers(t)

    @property
    def is_static(self) -> bool:
        return not self.changes

    def change_points(self) -> list[int]:
        return sorted({c.round for c in self.changes if c.round > 0})
