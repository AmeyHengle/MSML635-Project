"""Arm availability over time.

Scenario S1 adds a model to the pool part-way through the stream, so the action
set is not fixed. Most bandit implementations assume a constant ``n_arms``;
routing the availability question through a registry keeps that assumption out
of the policies, which only ever see the arms they are allowed to pull.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ArmRegistry"]


@dataclass
class ArmRegistry:
    """Tracks which arms are selectable at each round.

    Parameters
    ----------
    n_arms:
        Total number of arms in the pool, including ones not yet released.
    activation:
        ``{arm_index: round}``. An arm listed here is unavailable before that
        round and available from it onwards. Arms absent from the mapping are
        available from round 0.
    retirement:
        ``{arm_index: round}``. An arm listed here becomes unavailable from that
        round onwards.
    """

    n_arms: int
    activation: dict[int, int] = field(default_factory=dict)
    retirement: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for mapping, label in ((self.activation, "activation"), (self.retirement, "retirement")):
            for arm, round_ in mapping.items():
                if not 0 <= arm < self.n_arms:
                    raise ValueError(f"{label}: arm {arm} outside [0, {self.n_arms})")
                if round_ < 0:
                    raise ValueError(f"{label}: round {round_} must be non-negative")

    def mask(self, t: int) -> np.ndarray:
        """Boolean ``(n_arms,)`` mask of arms selectable at round ``t``."""
        m = np.ones(self.n_arms, dtype=bool)
        for arm, start in self.activation.items():
            if t < start:
                m[arm] = False
        for arm, stop in self.retirement.items():
            if t >= stop:
                m[arm] = False
        if not m.any():
            raise RuntimeError(f"no arms available at round {t}")
        return m

    def available(self, t: int) -> np.ndarray:
        """Indices of arms selectable at round ``t``."""
        return np.flatnonzero(self.mask(t))

    def is_available(self, t: int, arm: int) -> bool:
        return bool(self.mask(t)[arm])

    @property
    def is_static(self) -> bool:
        return not self.activation and not self.retirement

    def change_points(self) -> list[int]:
        """Rounds at which the action set changes."""
        return sorted({r for r in (*self.activation.values(), *self.retirement.values()) if r > 0})
