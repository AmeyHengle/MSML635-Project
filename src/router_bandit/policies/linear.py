"""Linear contextual bandits: LinUCB, discounted LinUCB, linear Thompson sampling.

All three maintain per-arm ridge regressions of reward on context. They differ
only in how they turn the posterior into a decision: an optimism bonus
(LinUCB), the same with geometric forgetting (discounted LinUCB), or a
posterior sample (LinTS). :class:`PilotStyle` is the same LinUCB update with a
calibration ridge prior and a paced spend cap, the two pieces of PILOT
(Panda et al., 2025) this study compares against.

The inverse design matrices are maintained incrementally with the
Sherman-Morrison identity, so an update is ``O(d^2)`` rather than ``O(d^3)``.
"""

from __future__ import annotations

import numpy as np

from .base import Policy

__all__ = ["DiscountedLinUCB", "LinTS", "LinUCB", "PilotStyle"]


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


class PilotStyle(LinUCB):
    """LinUCB with a calibration prior and a cumulative spend cap.

    Two adaptations of PILOT (Panda et al., 2025). The posterior starts at a
    ridge fit on the calibration split, held with ``prior_strength``
    pseudo-observations, so the mean begins at that fit while a small strength
    leaves the optimism bonus nearly as wide as a cold start. ``select`` then
    refuses any arm whose mean cost would push cumulative spend above
    ``pace * (t + 1)``. ``pace`` is the oracle's mean spend at this lambda.
    When every arm would break the pace, the cheapest available arm is taken,
    so a query still receives a model.

    The policy never sees a query's cost vector. It prices arms by the
    per-arm means it was given, which are known before the stream starts.
    Round 0 begins a new episode: a frozen replay keeps the posterior and
    still paces its own spend from zero.
    """

    def __init__(
        self,
        n_arms: int,
        dim: int,
        *,
        alpha: float = 1.0,
        ridge: float = 1.0,
        seed: int = 0,
        arm_costs: np.ndarray | None = None,
        pace: float = np.inf,
        prior_strength: float = 1.0,
    ) -> None:
        super().__init__(n_arms, dim, alpha=alpha, ridge=ridge, seed=seed)
        self.pace = float(pace)
        if self.pace < 0.0 or np.isnan(self.pace):
            raise ValueError(f"pace must be non-negative or inf, got {pace}")
        self.prior_strength = float(prior_strength)
        if self.prior_strength < 0.0:
            raise ValueError(f"prior_strength must be non-negative, got {prior_strength}")
        if arm_costs is None:
            self.arm_costs = None
        else:
            self.arm_costs = np.asarray(arm_costs, dtype=np.float64)
            if self.arm_costs.shape != (n_arms,):
                raise ValueError(
                    f"arm_costs must have shape ({n_arms},), got {self.arm_costs.shape}"
                )
        if np.isfinite(self.pace) and self.arm_costs is None:
            raise ValueError("a finite pace needs one mean cost per arm")
        self._spent = 0.0
        self._prior_A_inv: np.ndarray | None = None
        self._prior_b: np.ndarray | None = None
        self.name = f"PILOT(a={alpha:g})"

    def set_prior(self, X: np.ndarray, rewards: np.ndarray) -> PilotStyle:
        """Centre the posterior on a ridge fit of the calibration split.

        ``X`` is ``(n, dim)`` and ``rewards`` is ``(n, n_arms)``, the reward
        of every arm. The fit is the uncentered ridge ``(X'X + λI)^{-1} X'r``,
        the same hypothesis the online update uses. ``b`` is chosen so the
        prior mean stays at that fit, and ``prior_strength / n`` of the
        empirical Gram is added to the precision. That strength is one
        pseudo-count for the whole calibration split.
        """
        X = np.asarray(X, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        core = self._core
        if X.ndim != 2 or X.shape[1] != core.dim:
            raise ValueError(f"X must have shape (n, {core.dim}), got {X.shape}")
        if rewards.shape != (X.shape[0], core.n_arms):
            raise ValueError(
                f"rewards must have shape ({X.shape[0]}, {core.n_arms}), got {rewards.shape}"
            )
        if X.shape[0] < 1:
            raise ValueError("the calibration prior needs at least one row")

        n, d = X.shape
        gram = X.T @ X
        penalty = core.ridge * np.eye(d)
        theta_hat = np.linalg.solve(gram + penalty, X.T @ rewards).T
        precision = penalty + (self.prior_strength / n) * gram
        A_inv = np.linalg.inv(precision)
        # precision @ theta_hat keeps the mean on the fit; strength only
        # tightens the covariance. precision is symmetric.
        self._prior_A_inv = np.repeat(A_inv[None, :, :], core.n_arms, axis=0)
        self._prior_b = theta_hat @ precision
        self.reset()
        return self

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        available = np.asarray(available, dtype=np.int64)
        if t == 0:
            self._spent = 0.0
        allowed = self._affordable(t, available)
        action, propensity = super().select(t, context, allowed)
        if self.arm_costs is not None and np.isfinite(self.pace):
            self._spent += float(self.arm_costs[action])
        return action, propensity

    def reset(self) -> None:
        super().reset()
        self._spent = 0.0
        if self._prior_A_inv is not None and self._prior_b is not None:
            self._core._A_inv = self._prior_A_inv.copy()
            self._core._b = self._prior_b.copy()
            for arm in range(self._core.n_arms):
                self._core._refresh_theta(arm)

    def _affordable(self, t: int, available: np.ndarray) -> np.ndarray:
        """Arms whose mean cost still fits under the cumulative pace."""
        if self.arm_costs is None or not np.isfinite(self.pace):
            return available
        room = self.pace * (t + 1) - self._spent
        costs = self.arm_costs[available]
        allowed = available[costs <= room + 1e-12]
        if allowed.size == 0:
            cheapest = int(available[int(np.argmin(costs))])
            return np.asarray([cheapest], dtype=np.int64)
        return allowed
