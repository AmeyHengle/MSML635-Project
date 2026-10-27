"""Linear contextual bandits: LinUCB, discounted LinUCB, linear Thompson sampling.

All three maintain per-arm ridge regressions of reward on context. They differ
only in how they turn the posterior into a decision: an optimism bonus
(LinUCB), the same with geometric forgetting (discounted LinUCB), or a
posterior sample (LinTS).

The inverse design matrices are maintained incrementally with the
Sherman-Morrison identity, so an update is ``O(d^2)`` rather than ``O(d^3)``.
"""

from __future__ import annotations

import numpy as np

from .base import Policy

__all__ = ["DiscountedLinUCB", "LinTS", "LinUCB"]


class _LinearBase:
    """Shared per-arm ridge machinery (a helper, not a policy)."""

    def __init__(self, n_arms: int, dim: int, *, ridge: float = 1.0, seed: int = 0) -> None:
        if ridge <= 0:
            raise ValueError(f"ridge must be positive, got {ridge}")
        self.n_arms = int(n_arms)
        self.dim = int(dim)
        self.ridge = float(ridge)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        d, k = self.dim, self.n_arms
        self._A_inv = np.repeat((np.eye(d) / self.ridge)[None], k, axis=0)
        self._b = np.zeros((k, d))
        self._theta = np.zeros((k, d))
        self._counts = np.zeros(k, dtype=np.int64)
        self._rng = np.random.default_rng(self._seed)

    def _refresh_theta(self, arm: int) -> None:
        self._theta[arm] = self._A_inv[arm] @ self._b[arm]

    def update(self, t: int, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, dtype=np.float64)
        A_inv = self._A_inv[action]
        Ax = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self._A_inv[action] = A_inv - np.outer(Ax, Ax) / denom
        self._b[action] += reward * x
        self._counts[action] += 1
        self._refresh_theta(action)

    def _mean(self, context: np.ndarray, available: np.ndarray) -> np.ndarray:
        return self._theta[available] @ context

    @property
    def theta(self) -> np.ndarray:
        """``(n_arms, dim)`` current point estimates."""
        return self._theta

    @property
    def counts(self) -> np.ndarray:
        return self._counts


class LinUCB(Policy):
    """Disjoint LinUCB (Li et al., 2010).

    Selects ``argmax_a  theta_a . x + alpha * sqrt(x' A_a^-1 x)``.
    Deterministic, so it is not usable as a logging policy.
    """

    loggable = False

    def __init__(
        self, n_arms: int, dim: int, *, alpha: float = 1.0, ridge: float = 1.0, seed: int = 0
    ) -> None:
        self.alpha = float(alpha)
        self._core = _LinearBase(n_arms, dim, ridge=ridge, seed=seed)
        self.name = f"LinUCB(a={alpha:g})"

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        x = np.asarray(context, dtype=np.float64)
        core = self._core
        mean = core._theta[available] @ x
        var = np.einsum("i,aij,j->a", x, core._A_inv[available], x)
        scores = mean + self.alpha * np.sqrt(np.maximum(var, 0.0))
        return int(available[int(np.argmax(scores))]), 1.0

    def update(self, t: int, context: np.ndarray, action: int, reward: float) -> None:
        self._core.update(t, context, action, reward)

    def reset(self) -> None:
        self._core.reset()

    @property
    def theta(self) -> np.ndarray:
        return self._core.theta


class DiscountedLinUCB(LinUCB):
    """LinUCB with geometric forgetting, after Russac et al. (2019).

    Down-weights older observations by ``gamma`` each round, so the posterior
    tracks a drifting reward function instead of averaging over it. This is the
    same mechanism ParetoBandit calls "geometric forgetting"; we use it as the
    non-stationary variant for RQ2 rather than as a contribution.

    ``gamma = 1`` recovers plain LinUCB.
    """

    def __init__(
        self,
        n_arms: int,
        dim: int,
        *,
        alpha: float = 1.0,
        gamma: float = 0.999,
        ridge: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
        super().__init__(n_arms, dim, alpha=alpha, ridge=ridge, seed=seed)
        self.gamma = float(gamma)
        self.name = f"D-LinUCB(a={alpha:g},g={gamma:g})"

    def update(self, t: int, context: np.ndarray, action: int, reward: float) -> None:
        core = self._core
        if self.gamma < 1.0:
            # Discounting A means inflating A^-1 by the same factor.
            core._A_inv[action] /= self.gamma
            core._b[action] *= self.gamma
        super().update(t, context, action, reward)


class LinTS(Policy):
    """Linear Thompson sampling (Agrawal & Goyal, 2013).

    Draws ``theta_a ~ N(theta_hat_a, v^2 A_a^-1)`` and acts greedily on the
    draw. Stochastic, but its action distribution has no closed form, so
    propensities are estimated by Monte Carlo when ``mc_propensity`` is set --
    needed only if it is used as a logging policy.
    """

    loggable = False

    def __init__(
        self,
        n_arms: int,
        dim: int,
        *,
        v: float = 0.25,
        ridge: float = 1.0,
        seed: int = 0,
        mc_propensity: int = 0,
    ) -> None:
        self.v = float(v)
        self.mc_propensity = int(mc_propensity)
        self._core = _LinearBase(n_arms, dim, ridge=ridge, seed=seed)
        self.name = f"LinTS(v={v:g})"
        self.loggable = self.mc_propensity > 0

    def _sample_scores(self, x: np.ndarray, available: np.ndarray, n: int = 1) -> np.ndarray:
        core = self._core
        mean = core._theta[available] @ x
        var = np.einsum("i,aij,j->a", x, core._A_inv[available], x)
        sd = self.v * np.sqrt(np.maximum(var, 0.0))
        return mean + sd * core._rng.normal(size=(n, available.size))

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        x = np.asarray(context, dtype=np.float64)
        action_local = int(np.argmax(self._sample_scores(x, available)[0]))
        propensity = 1.0
        if self.mc_propensity > 0:
            draws = self._sample_scores(x, available, self.mc_propensity)
            picks = np.argmax(draws, axis=1)
            propensity = float(max((picks == action_local).mean(), 1.0 / self.mc_propensity))
        return int(available[action_local]), propensity

    def update(self, t: int, context: np.ndarray, action: int, reward: float) -> None:
        self._core.update(t, context, action, reward)

    def reset(self) -> None:
        self._core.reset()

    @property
    def theta(self) -> np.ndarray:
        return self._core.theta
