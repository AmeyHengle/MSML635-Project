"""Reward, oracle, and the episode loop.

Shifts are a later task. An episode shows the policy the context and the
chosen reward only. The full reward matrix stays here for the oracle.
"""

from typing import NamedTuple

import numpy as np


class Stream(NamedTuple):
    """One episode. `cost` is raw, and `train_max` is computed before the loop."""

    context: np.ndarray  # (T, d)
    quality: np.ndarray  # (T, K), already mapped to [0, 1]
    cost: np.ndarray  # (T, K), raw cost of the arms in play
    lam: float
    train_max: float


class EpisodeResult(NamedTuple):
    actions: np.ndarray
    rewards: np.ndarray
    oracle_actions: np.ndarray
    gaps: np.ndarray
    regret: float


def train_cost_max(cost: np.ndarray) -> float:
    """Max raw cost over the four arms. A fifth column is held out and ignored."""
    values = np.asarray(cost, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    n_arms = min(4, values.shape[1])
    return float(np.max(values[:, :n_arms]))


def normalized_cost(cost: np.ndarray, train_max: float) -> np.ndarray:
    """Divide by the train maximum and clip to [0, 1]. A zero maximum yields zeros."""
    raw = np.asarray(cost, dtype=float)
    if train_max == 0:
        return np.zeros_like(raw, dtype=float)
    return np.clip(raw / train_max, 0.0, 1.0)


def arm_rewards(quality: np.ndarray, cost: np.ndarray, lam: float, train_max: float) -> np.ndarray:
    """r = q - λ * normalized cost.

    quality_* is already mapped into [0, 1] by pivot_wide; do not clip it again.
    """
    q = np.asarray(quality, dtype=float)
    return q - float(lam) * normalized_cost(cost, train_max)


def oracle_action(quality: np.ndarray, cost: np.ndarray, lam: float, train_max: float) -> int:
    """Best arm for one query. Ties break to the lower raw cost, then the lower index."""
    return _oracle_arm(arm_rewards(quality, cost, lam, train_max), cost)


def regret(gaps: np.ndarray) -> float:
    """Sum of per-step gaps, r(oracle) - r(chosen)."""
    return float(np.asarray(gaps, dtype=float).sum())


def run_episode(policy, stream: Stream, update: bool) -> EpisodeResult:
    """Play `stream` in order. `update=False` does not call `policy.update`."""
    context = np.asarray(stream.context, dtype=float)
    quality = np.asarray(stream.quality, dtype=float)
    cost = np.asarray(stream.cost, dtype=float)
    reward_matrix = arm_rewards(quality, cost, stream.lam, stream.train_max)
    steps = context.shape[0]
    actions = np.empty(steps, dtype=int)
    chosen = np.empty(steps, dtype=float)
    oracle = np.empty(steps, dtype=int)
    gaps = np.empty(steps, dtype=float)
    for t in range(steps):
        x = context[t]
        action = int(policy.act(x))
        reward = float(reward_matrix[t, action])
        best = _oracle_arm(reward_matrix[t], cost[t])
        actions[t] = action
        chosen[t] = reward
        oracle[t] = best
        gaps[t] = float(reward_matrix[t, best] - reward)
        if update:
            policy.update(x, action, reward)
    return EpisodeResult(actions, chosen, oracle, gaps, regret(gaps))


def _oracle_arm(reward_row: np.ndarray, cost_row: np.ndarray) -> int:
    """Highest reward. Equal rewards take the cheaper raw cost, then the smaller index."""
    rewards = np.asarray(reward_row, dtype=float)
    costs = np.asarray(cost_row, dtype=float)
    best = 0
    for arm in range(1, rewards.shape[0]):
        if rewards[arm] > rewards[best]:
            best = arm
        elif rewards[arm] == rewards[best] and costs[arm] < costs[best]:
            best = arm
    return best
