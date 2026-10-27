"""Per-query oracle and fixed-arm reference policies."""

from __future__ import annotations

import numpy as np

from ..envs.bandit_env import EnvTruth
from ..policies.base import Policy

__all__ = ["OraclePolicy"]


class OraclePolicy(Policy):
    """Selects the best available arm every round.

    An upper bound, not a learner: it is constructed from
    :class:`~router_bandit.envs.bandit_env.EnvTruth`, the full-information view
    that real policies are structurally prevented from touching. Regret is
    measured against it.
    """

    loggable = False

    def __init__(self, truth: EnvTruth) -> None:
        self._truth = truth
        self.name = "Oracle"

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        return self._truth.oracle_action(t), 1.0
