"""The bandit environment: where the feedback restriction is enforced.

This is the central abstraction of the project. :meth:`BanditEnv.step` returns
a **scalar** -- the reward of the arm that was actually pulled -- and nothing
else. The full per-arm reward vector exists, because the oracle, regret, and
off-policy ground truth all require it, but it is reachable only through
:attr:`BanditEnv.truth`, a separate object handed to the evaluator and never to
a policy.

Making that a type boundary rather than a comment is deliberate: the single
most damaging bug available in this project is a policy that accidentally sees
rewards it was not supposed to, and such a bug produces beautiful results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.stream import Stream
from .arms import ArmRegistry
from .costs import CostModel
from .shifts import ShiftSchedule

__all__ = ["BanditEnv", "EnvTruth", "StepResult"]


@dataclass(frozen=True)
class StepResult:
    """What a policy learns from one pull."""

    reward: float
    quality: float
    cost: float


class EnvTruth:
    """Evaluator-only view of an environment.

    Holds everything a policy must not see: the full reward matrix, the
    per-round oracle, and the realised shift effects.
    """

    def __init__(self, env: BanditEnv) -> None:
        self._env = env

    def rewards(self, t: int) -> np.ndarray:
        """``(n_arms,)`` reward for every arm at round ``t`` (unavailable arms
        are ``-inf`` so they can never be selected as the oracle)."""
        return self._env._reward_vector(t, mask_unavailable=True)

    def reward_matrix(self) -> np.ndarray:
        return np.vstack([self.rewards(t) for t in range(self._env.n_rounds)])

    def quality(self, t: int) -> np.ndarray:
        return self._env._effective_quality(t)

    def cost(self, t: int) -> np.ndarray:
        return self._env._effective_cost(t)

    def oracle_action(self, t: int) -> int:
        return int(np.argmax(self.rewards(t)))

    def oracle_reward(self, t: int) -> float:
        return float(np.max(self.rewards(t)))

    def oracle_actions(self) -> np.ndarray:
        return np.array([self.oracle_action(t) for t in range(self._env.n_rounds)], dtype=np.int64)

    def oracle_rewards(self) -> np.ndarray:
        return np.array([self.oracle_reward(t) for t in range(self._env.n_rounds)])


class BanditEnv:
    """Replays a :class:`~router_bandit.data.stream.Stream` under bandit feedback.

    Parameters
    ----------
    stream:
        The query stream. Supplies contexts and the per-arm quality/cost vectors.
    lam:
        Cost penalty in ``reward = quality - lam * cost``. Calibrated on
        RouterBench to roughly ``[2, 30]``; see the data-exploration notebook.
    schedule:
        Distribution shifts to inject. Omit for the stationary setting.
    seed:
        Seeds the S4 degradation draw only; policies carry their own RNGs.
    """

    def __init__(
        self,
        stream: Stream,
        lam: float,
        schedule: ShiftSchedule | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self._stream = stream
        self._lam = float(lam)
        self._schedule = schedule or ShiftSchedule()
        self._rng = np.random.default_rng(seed)

        n_arms = stream.n_arms
        self._arms: ArmRegistry = self._schedule.arm_registry(n_arms)
        self._costs: CostModel = self._schedule.cost_model(n_arms)
        self._degraded = self._schedule.degradation_mask(stream.n_rounds, n_arms, self._rng)

        self._pulls = 0

    # -- geometry ---------------------------------------------------------- #
    @property
    def n_rounds(self) -> int:
        return self._stream.n_rounds

    @property
    def n_arms(self) -> int:
        return self._stream.n_arms

    @property
    def dim(self) -> int:
        return self._stream.dim

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def schedule(self) -> ShiftSchedule:
        return self._schedule

    @property
    def n_pulls(self) -> int:
        return self._pulls

    # -- policy-facing ----------------------------------------------------- #
    def context(self, t: int) -> np.ndarray:
        return self._stream.context(t)

    def available_actions(self, t: int) -> np.ndarray:
        return self._arms.available(t)

    def action_mask(self, t: int) -> np.ndarray:
        return self._arms.mask(t)

    def step(self, t: int, action: int) -> StepResult:
        """Pull ``action`` at round ``t`` and observe **only its** outcome."""
        if not self._arms.is_available(t, action):
            raise ValueError(f"arm {action} is not available at round {t}")
        quality = float(self._effective_quality(t)[action])
        cost = float(self._effective_cost(t)[action])
        self._pulls += 1
        return StepResult(reward=quality - self._lam * cost, quality=quality, cost=cost)

    # -- evaluator-facing -------------------------------------------------- #
    @property
    def truth(self) -> EnvTruth:
        """Full-information view. Never hand this to a policy."""
        return EnvTruth(self)

    # -- internals --------------------------------------------------------- #
    def _effective_quality(self, t: int) -> np.ndarray:
        quality = self._stream.quality_vec(t).astype(np.float64, copy=True)
        if self._degraded is not None:
            quality = np.where(self._degraded[t], 0.0, quality)
        return quality

    def _effective_cost(self, t: int) -> np.ndarray:
        return self._costs.apply(t, self._stream.cost_vec(t).astype(np.float64, copy=False))

    def _reward_vector(self, t: int, *, mask_unavailable: bool = False) -> np.ndarray:
        rewards = self._effective_quality(t) - self._lam * self._effective_cost(t)
        if mask_unavailable:
            rewards = np.where(self._arms.mask(t), rewards, -np.inf)
        return rewards

    def change_points(self) -> list[int]:
        return self._schedule.change_points()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "stationary" if self._schedule.is_stationary else "shifted"
        return (
            f"BanditEnv(n_rounds={self.n_rounds}, n_arms={self.n_arms}, "
            f"lam={self._lam:g}, {kind})"
        )
