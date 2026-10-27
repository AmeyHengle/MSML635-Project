"""Confidence-based cascade (FrugalGPT-style).

A cascade queries models in increasing order of price and stops when a judge
accepts the answer. It therefore pulls *several* arms per query, which does not
fit the single-pull contract of :class:`~router_bandit.envs.bandit_env.BanditEnv`.
It is evaluated directly over the full matrices instead, exactly as RouterBench
evaluated it: with a judge given perfect knowledge of the score, then corrupted
by an explicit error rate.

RouterBench reports that cascades beat the Zero Router while the judge's error
rate stays below about 0.1 and degrade sharply past 0.2, so ``judge_error`` is
the parameter that matters.
"""

from __future__ import annotations

import numpy as np

__all__ = ["run_cascade"]


def run_cascade(
    quality: np.ndarray,
    cost: np.ndarray,
    *,
    order: np.ndarray | None = None,
    threshold: float = 0.5,
    judge_error: float = 0.0,
    budget: float = np.inf,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run a cascade over every query.

    Parameters
    ----------
    quality, cost:
        ``(n_queries, n_arms)`` full matrices.
    order:
        Arm order to try. Defaults to ascending mean cost.
    threshold:
        Quality at or above which the judge would accept.
    judge_error:
        Probability the judge's verdict is flipped.
    budget:
        Maximum cumulative cost per query before the cascade stops and returns
        whatever it has.

    Returns
    -------
    dict with ``quality``, ``cost`` (cumulative, including rejected attempts),
    ``n_calls``, and ``final_arm``.
    """
    quality = np.asarray(quality, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n, k = quality.shape
    if order is None:
        order = np.argsort(cost.mean(axis=0))
    order = np.asarray(order, dtype=np.int64)
    rng = np.random.default_rng(seed)

    flips = rng.random((n, k)) < judge_error
    out_q = np.zeros(n)
    out_c = np.zeros(n)
    out_calls = np.zeros(n, dtype=np.int64)
    out_arm = np.zeros(n, dtype=np.int64)

    for i in range(n):
        spent = 0.0
        arm = int(order[0])
        for step, a in enumerate(order):
            spent += cost[i, a]
            arm = int(a)
            out_calls[i] = step + 1
            accept = quality[i, a] >= threshold
            if flips[i, step]:
                accept = not accept
            if accept or spent >= budget:
                break
        out_q[i] = quality[i, arm]
        out_c[i] = spent
        out_arm[i] = arm

    return {"quality": out_q, "cost": out_c, "n_calls": out_calls, "final_arm": out_arm}
