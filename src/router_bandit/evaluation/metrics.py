"""Validation metrics, each tied to a research question."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Recovery",
    "average_performance_gap_recovered",
    "bootstrap_ci",
    "cost_performance_threshold",
    "cumulative_regret",
    "heldout_summary",
    "performance_gap_recovered",
    "pool_reference_arms",
    "recovery_time",
    "routing_errors",
]


def cumulative_regret(rewards: np.ndarray, oracle_rewards: np.ndarray) -> np.ndarray:
    """RQ2: learning efficiency against the per-query oracle."""
    return np.cumsum(np.asarray(oracle_rewards) - np.asarray(rewards))


@dataclass(frozen=True)
class Recovery:
    """When post-shift reward leaves its pre-shift level, and how long it stays away.

    ``departed_at`` is the start of the first post-shift window whose mean falls
    more than the tolerance below the pre-shift mean, counted in rounds after
    the shift. ``rounds_to_recover`` is the number of rounds from that window
    until a later window is back within tolerance.

    Both are ``None`` when the policy never departs. ``rounds_to_recover`` alone
    is ``None`` when it departs and does not return. Callers store ``None`` as
    a missing value rather than 0. A departure at 0 means the first window is
    already below the band, and a recovery of 0 cannot occur, because that
    same window is the one that fell below the band.
    """

    departed_at: int | None
    rounds_to_recover: int | None


def recovery_time(
    rewards: np.ndarray,
    shift_round: int,
    *,
    window: int = 50,
    tolerance: float = 0.05,
) -> Recovery:
    """RQ2: departure from, and return to, the pre-shift reward level.

    The pre-shift level is the mean over the ``window`` rounds before the
    shift. Post-shift reward is smoothed with the same window. A window has
    departed when its mean is strictly more than ``tolerance`` (fractionally)
    below that level, and has recovered when a later window is back at or above
    it. Post-shift regret, not this pair, is the headline adaptation number:
    a policy that never leaves the band is not the same event as one that
    returns immediately.
    """
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    rewards = np.asarray(rewards, dtype=np.float64)
    lo = max(0, shift_round - window)
    if shift_round <= lo or shift_round >= rewards.size:
        return Recovery(None, None)
    baseline = float(rewards[lo:shift_round].mean())
    floor = baseline - abs(baseline) * tolerance

    post = rewards[shift_round:]
    if post.size < window:
        return Recovery(None, None)
    smoothed = np.convolve(post, np.ones(window) / window, mode="valid")
    below = smoothed < floor
    departures = np.flatnonzero(below)
    if departures.size == 0:
        return Recovery(None, None)
    departed_at = int(departures[0])
    back = np.flatnonzero((~below) & (np.arange(smoothed.size) > departed_at))
    if back.size == 0:
        return Recovery(departed_at, None)
    return Recovery(departed_at, int(back[0] - departed_at))


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


def pool_reference_arms(mean_cost: np.ndarray, mean_quality: np.ndarray) -> tuple[int, int]:
    """Weak and strong arms of a pool: cheapest, then highest mean quality."""
    mean_cost = np.asarray(mean_cost, dtype=np.float64)
    mean_quality = np.asarray(mean_quality, dtype=np.float64)
    if mean_cost.shape != mean_quality.shape or mean_cost.ndim != 1 or mean_cost.size == 0:
        raise ValueError("mean_cost and mean_quality must be non-empty 1-d arrays of equal length")
    return int(np.argmin(mean_cost)), int(np.argmax(mean_quality))


def performance_gap_recovered(q_router: float, q_weak: float, q_strong: float) -> float:
    """Fraction of the weak-to-strong quality gap that ``q_router`` closes.

    ``(Q_router - Q_weak) / (Q_strong - Q_weak)``. One point on the lambda
    sweep; APGR averages these. Returns NaN when the two reference arms have
    the same quality, because there is no gap to recover.
    """
    gap = float(q_strong) - float(q_weak)
    if gap == 0.0:
        return float("nan")
    return (float(q_router) - float(q_weak)) / gap


def average_performance_gap_recovered(pgr: np.ndarray) -> float:
    """APGR: mean performance-gap recovered over the lambda grid."""
    pgr = np.asarray(pgr, dtype=np.float64)
    if pgr.size == 0:
        return float("nan")
    return float(np.mean(pgr))


def cost_performance_threshold(
    mean_cost: np.ndarray,
    pgr: np.ndarray,
    *,
    level: float = 0.5,
) -> float:
    """CPT: smallest mean cost on the frontier whose gap recovered reaches ``level``.

    ``mean_cost`` and ``pgr`` are paired, one entry per lambda. Returns NaN
    when no point on the grid reaches ``level``.
    """
    mean_cost = np.asarray(mean_cost, dtype=np.float64)
    pgr = np.asarray(pgr, dtype=np.float64)
    if mean_cost.shape != pgr.shape or mean_cost.ndim != 1:
        raise ValueError("mean_cost and pgr must be 1-d arrays of equal length")
    hit = pgr >= level
    if not np.any(hit):
        return float("nan")
    return float(mean_cost[hit].min())


def heldout_summary(
    rewards: np.ndarray,
    qualities: np.ndarray,
    dollar_costs: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimates and 95% bootstrap intervals for one frozen held-out replay.

    The interval is over queries, via :func:`bootstrap_ci`. Seed-to-seed
    dispersion is a different quantity and is left to the table that averages
    these rows.
    """
    fields = {
        "heldout_reward": rewards,
        "heldout_quality": qualities,
        "heldout_dollar_cost": dollar_costs,
    }
    out: dict[str, float] = {}
    for name, values in fields.items():
        mean, lo, hi = bootstrap_ci(values, n_boot=n_boot, alpha=0.05, seed=seed)
        out[name] = mean
        out[f"{name}_lo"] = lo
        out[f"{name}_hi"] = hi
    return out


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
