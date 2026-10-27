"""Validation metrics, each tied to a research question."""

from __future__ import annotations

import numpy as np

__all__ = [
    "bootstrap_ci",
    "cumulative_regret",
    "recovery_time",
    "routing_errors",
]


def cumulative_regret(rewards: np.ndarray, oracle_rewards: np.ndarray) -> np.ndarray:
    """RQ2: learning efficiency against the per-query oracle."""
    return np.cumsum(np.asarray(oracle_rewards) - np.asarray(rewards))


def recovery_time(
    rewards: np.ndarray,
    shift_round: int,
    *,
    window: int = 200,
    tolerance: float = 0.05,
) -> int | None:
    """RQ2: rounds after ``shift_round`` until reward returns near its pre-shift level.

    Pre-shift level is the mean over the ``window`` rounds before the shift.
    Returns ``None`` if the policy never recovers within the stream.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    lo = max(0, shift_round - window)
    if shift_round <= lo or shift_round >= rewards.size:
        return None
    baseline = float(rewards[lo:shift_round].mean())
    target = baseline - abs(baseline) * tolerance

    post = rewards[shift_round:]
    if post.size < window:
        return None
    kernel = np.ones(window) / window
    smoothed = np.convolve(post, kernel, mode="valid")
    hits = np.flatnonzero(smoothed >= target)
    return int(hits[0]) if hits.size else None


def routing_errors(
    actions: np.ndarray,
    oracle_actions: np.ndarray,
    mean_cost: np.ndarray,
) -> dict[str, float]:
    """RQ1/RQ2: decompose mistakes into over- and under-spending.

    An *escalation* is choosing a more expensive arm than the oracle would; a
    *downgrade* is choosing a cheaper one. Both are errors, but they fail in
    opposite directions, and a router that only ever escalates is just an
    expensive fixed policy wearing a hat.
    """
    actions = np.asarray(actions)
    oracle_actions = np.asarray(oracle_actions)
    mean_cost = np.asarray(mean_cost, dtype=np.float64)

    chosen, target = mean_cost[actions], mean_cost[oracle_actions]
    wrong = actions != oracle_actions
    return {
        "error_rate": float(wrong.mean()),
        "escalation_rate": float((wrong & (chosen > target)).mean()),
        "downgrade_rate": float((wrong & (chosen < target)).mean()),
        "mean_excess_cost": float((chosen - target).mean()),
    }


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap ``(mean, lower, upper)``."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )
