"""Policy interface.

Every policy exposes the same two methods, so the experiment runner does not
know or care whether it is driving a bandit, a trained classifier, or a fixed
baseline.

``select`` returns ``(action, propensity)``. The propensity -- the probability
the policy assigned to the action it took -- is what makes off-policy
evaluation possible, and it is the reason RQ3 cannot reuse logs from RQ1/RQ2:
LinUCB is a deterministic argmax, so its propensity is always 1 and inverse
propensity scoring degenerates. Logging runs must use a genuinely stochastic
policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Policy"]


class Policy(ABC):
    """A routing policy under bandit feedback."""

    #: Short label used in results tables and figures.
    name: str = "policy"

    #: True if ``select`` is stochastic with a computable propensity, i.e. the
    #: policy is usable as a logging policy for off-policy evaluation.
    loggable: bool = False

    @abstractmethod
    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        """Choose an arm from ``available``; return ``(action, propensity)``."""

    def update(self, t: int, context: np.ndarray, action: int, reward: float) -> None:
        """Incorporate the observed reward. Stateless policies may ignore this."""

    def reset(self) -> None:
        """Clear learned state so the policy can be replayed from scratch."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
