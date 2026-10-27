"""Policies with closed-form propensities.

These are the ones usable as *logging* policies: their action distribution is
known exactly, so inverse propensity scoring is well defined. LinUCB cannot
play that role, which is why RQ3 needs its own run rather than reusing RQ1/RQ2
logs.
"""

from __future__ import annotations

import numpy as np

from .base import Policy
from .linear import _LinearBase

__all__ = ["EpsilonGreedy", "FixedArm", "UniformRandom"]


class UniformRandom(Policy):
    """Uniform over the available arms. The canonical logging policy."""

    loggable = True

    def __init__(self, n_arms: int, *, seed: int = 0) -> None:
        self.n_arms = int(n_arms)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.name = "Random"

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        return int(self._rng.choice(available)), 1.0 / available.size

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)


class EpsilonGreedy(Policy):
    """Linear reward model, uniform exploration with probability ``epsilon``.

    Propensity is exact: ``eps/|A| + (1 - eps) * 1[a is greedy]``.
    """

    loggable = True

    def __init__(
        self,
        n_arms: int,
        dim: int,
        *,
        epsilon: float = 0.1,
        ridge: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)
        self._core = _LinearBase(n_arms, dim, ridge=ridge, seed=seed)
        self._seed = seed
        self._rng = np.random.default_rng(seed + 9973)
        self.name = f"eps-greedy(e={epsilon:g})"

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        x = np.asarray(context, dtype=np.float64)
        scores = self._core._theta[available] @ x
        greedy_local = int(np.argmax(scores))
        k = available.size

        if self._rng.random() < self.epsilon:
            chosen_local = int(self._rng.integers(k))
        else:
            chosen_local = greedy_local

        propensity = self.epsilon / k + (1.0 - self.epsilon) * float(chosen_local == greedy_local)
        return int(available[chosen_local]), propensity

    def update(self, t: int, context: np.ndarray, action: int, reward: float) -> None:
        self._core.update(t, context, action, reward)

    def reset(self) -> None:
        self._core.reset()
        self._rng = np.random.default_rng(self._seed + 9973)

    @property
    def theta(self) -> np.ndarray:
        return self._core.theta


class FixedArm(Policy):
    """Always pull the same arm, falling back when it is unavailable.

    Instantiated as the always-cheapest and always-most-expensive baselines.
    """

    loggable = True

    def __init__(self, arm: int, *, name: str | None = None) -> None:
        self.arm = int(arm)
        self.name = name or f"always[{arm}]"

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        if self.arm in available:
            return self.arm, 1.0
        return int(available[0]), 1.0
