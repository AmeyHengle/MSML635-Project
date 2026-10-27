"""Decision logging with propensities.

Everything RQ3 needs: the context, the action taken, the probability the
logging policy assigned to it, and the realised reward. Without the propensity
column none of the off-policy estimators are defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DecisionLog", "DecisionLogger"]


@dataclass
class DecisionLog:
    """Immutable record of one replay."""

    contexts: np.ndarray
    actions: np.ndarray
    propensities: np.ndarray
    rewards: np.ndarray
    qualities: np.ndarray
    costs: np.ndarray
    rounds: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.size)

    @property
    def min_propensity(self) -> float:
        return float(self.propensities.min())

    def check_loggable(self) -> None:
        """Raise if the log cannot support inverse propensity scoring."""
        if np.allclose(self.propensities, 1.0):
            raise ValueError(
                "All propensities are 1: the logging policy was deterministic, so "
                "IPS is undefined. Re-run logging with a stochastic policy "
                "(UniformRandom or EpsilonGreedy)."
            )
        if self.min_propensity <= 0:
            raise ValueError("non-positive propensity in log")


@dataclass
class DecisionLogger:
    """Accumulates decisions during a replay."""

    _contexts: list = field(default_factory=list)
    _actions: list = field(default_factory=list)
    _propensities: list = field(default_factory=list)
    _rewards: list = field(default_factory=list)
    _qualities: list = field(default_factory=list)
    _costs: list = field(default_factory=list)
    _rounds: list = field(default_factory=list)

    def record(
        self,
        t: int,
        context: np.ndarray,
        action: int,
        propensity: float,
        reward: float,
        quality: float,
        cost: float,
    ) -> None:
        self._rounds.append(t)
        self._contexts.append(np.asarray(context, dtype=np.float64))
        self._actions.append(int(action))
        self._propensities.append(float(propensity))
        self._rewards.append(float(reward))
        self._qualities.append(float(quality))
        self._costs.append(float(cost))

    def build(self) -> DecisionLog:
        return DecisionLog(
            contexts=np.asarray(self._contexts, dtype=np.float64),
            actions=np.asarray(self._actions, dtype=np.int64),
            propensities=np.asarray(self._propensities, dtype=np.float64),
            rewards=np.asarray(self._rewards, dtype=np.float64),
            qualities=np.asarray(self._qualities, dtype=np.float64),
            costs=np.asarray(self._costs, dtype=np.float64),
            rounds=np.asarray(self._rounds, dtype=np.int64),
        )
