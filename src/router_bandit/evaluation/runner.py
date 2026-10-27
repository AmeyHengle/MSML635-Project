"""Replay a policy against an environment.

The loop is deliberately tiny, because it is the place where the feedback
restriction either holds or silently breaks. The policy sees ``context`` and
``available``; it receives back one scalar. Everything else -- the oracle, the
full reward vector, the regret -- is read from ``env.truth`` *after* the policy
has committed to its action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.bandit_env import BanditEnv
from ..logs.decision_logger import DecisionLog, DecisionLogger
from ..policies.base import Policy

__all__ = ["RunResult", "run_policy"]


@dataclass
class RunResult:
    """Outcome of one replay."""

    policy: str
    rewards: np.ndarray
    qualities: np.ndarray
    costs: np.ndarray
    actions: np.ndarray
    propensities: np.ndarray
    oracle_rewards: np.ndarray
    oracle_actions: np.ndarray
    log: DecisionLog | None = None

    @property
    def instant_regret(self) -> np.ndarray:
        return self.oracle_rewards - self.rewards

    @property
    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.instant_regret)

    @property
    def total_regret(self) -> float:
        return float(self.instant_regret.sum())

    @property
    def mean_reward(self) -> float:
        return float(self.rewards.mean())

    @property
    def mean_quality(self) -> float:
        return float(self.qualities.mean())

    @property
    def mean_cost(self) -> float:
        return float(self.costs.mean())

    @property
    def oracle_match(self) -> float:
        """Fraction of rounds the policy picked the oracle arm."""
        return float((self.actions == self.oracle_actions).mean())

    def summary(self) -> dict[str, float | str]:
        return {
            "policy": self.policy,
            "mean_reward": self.mean_reward,
            "mean_quality": self.mean_quality,
            "mean_cost": self.mean_cost,
            "total_regret": self.total_regret,
            "oracle_match": self.oracle_match,
        }


def run_policy(
    env: BanditEnv,
    policy: Policy,
    *,
    log: bool = False,
    reset: bool = True,
) -> RunResult:
    """Replay ``policy`` over the whole of ``env``."""
    if reset:
        policy.reset()

    T = env.n_rounds
    rewards = np.empty(T)
    qualities = np.empty(T)
    costs = np.empty(T)
    actions = np.empty(T, dtype=np.int64)
    propensities = np.empty(T)
    logger = DecisionLogger() if log else None

    truth = env.truth
    oracle_rewards = np.empty(T)
    oracle_actions = np.empty(T, dtype=np.int64)

    for t in range(T):
        context = env.context(t)
        available = env.available_actions(t)

        action, propensity = policy.select(t, context, available)
        outcome = env.step(t, action)
        policy.update(t, context, action, outcome.reward)

        rewards[t] = outcome.reward
        qualities[t] = outcome.quality
        costs[t] = outcome.cost
        actions[t] = action
        propensities[t] = propensity

        # Evaluator-side, read only after the policy has committed.
        oracle_rewards[t] = truth.oracle_reward(t)
        oracle_actions[t] = truth.oracle_action(t)

        if logger is not None:
            logger.record(t, context, action, propensity, outcome.reward,
                          outcome.quality, outcome.cost)

    return RunResult(
        policy=policy.name,
        rewards=rewards,
        qualities=qualities,
        costs=costs,
        actions=actions,
        propensities=propensities,
        oracle_rewards=oracle_rewards,
        oracle_actions=oracle_actions,
        log=logger.build() if logger is not None else None,
    )
