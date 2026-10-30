"""The Zero Router: RouterBench's query-agnostic baseline.

Given the per-arm mean cost and quality, the Zero Router mixes probabilistically
between the arms on the *non-decreasing convex hull* of the cost-quality plane
to hit any requested cost level. It uses no information about the query at all.

This matters because RouterBench's authors found that their own KNN and MLP
routers failed to consistently beat it. A contextual router that does not clear
this bar has not demonstrated anything, so it is the single most important
baseline in the study -- and the one the original proposal omitted.
"""

from __future__ import annotations

import numpy as np

from ..policies.base import Policy

__all__ = [
    "ZeroRouter",
    "aiq",
    "hull_target_costs",
    "non_decreasing_convex_hull",
    "zero_frontier_policies",
]


def non_decreasing_convex_hull(costs: np.ndarray, qualities: np.ndarray) -> np.ndarray:
    """Indices of the non-decreasing upper convex hull, ordered by cost.

    Two stages, following RouterBench: discard any arm that a cheaper arm
    already matches or beats, then take the upper convex hull of the survivors.
    """
    costs = np.asarray(costs, dtype=np.float64)
    qualities = np.asarray(qualities, dtype=np.float64)

    kept: list[int] = []
    best = -np.inf
    for i in np.argsort(costs):
        if qualities[i] > best:
            kept.append(int(i))
            best = qualities[i]

    hull: list[int] = []
    for i in kept:
        while len(hull) >= 2:
            x1, y1 = costs[hull[-2]], qualities[hull[-2]]
            x2, y2 = costs[hull[-1]], qualities[hull[-1]]
            if (y2 - y1) * (costs[i] - x1) <= (qualities[i] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull, dtype=np.int64)


def hull_target_costs(
    costs: np.ndarray, qualities: np.ndarray, n_grid: int = 9
) -> np.ndarray:
    """Costs that trace the non-decreasing convex hull.

    Every hull vertex is included, along with ``n_grid`` points from the
    cheapest hull arm to the dearest. A Zero Router at each of those costs
    is the mixture curve Figure 4a draws.
    """
    hull = non_decreasing_convex_hull(costs, qualities)
    vertex = np.asarray(costs, dtype=np.float64)[hull]
    if vertex.size <= 1 or n_grid <= 1:
        return vertex.copy()
    grid = np.linspace(float(vertex[0]), float(vertex[-1]), int(n_grid))
    return np.unique(np.concatenate([vertex, grid]))


def zero_frontier_policies(
    mean_cost: np.ndarray,
    mean_quality: np.ndarray,
    *,
    seed: int = 0,
    n_grid: int = 9,
) -> list[ZeroRouter]:
    """One Zero Router per :func:`hull_target_costs` point."""
    targets = hull_target_costs(mean_cost, mean_quality, n_grid=n_grid)
    return [
        ZeroRouter(
            mean_cost,
            mean_quality,
            float(target),
            seed=seed,
            name=f"ZeroRouter-hull[{i}]",
        )
        for i, target in enumerate(targets)
    ]


def aiq(costs: np.ndarray, qualities: np.ndarray, hull: np.ndarray, n_grid: int = 512) -> float:
    """Average Improvement in Quality: area under the hull, normalised by its width."""
    hc, hq = np.asarray(costs)[hull], np.asarray(qualities)[hull]
    if hc.max() <= hc.min():
        return float(hq.mean())
    grid = np.linspace(hc.min(), hc.max(), n_grid)
    return float(np.trapezoid(np.interp(grid, hc, hq), grid) / (hc.max() - hc.min()))


class ZeroRouter(Policy):
    """Query-agnostic probabilistic mixture over hull arms at a target cost."""

    loggable = True

    def __init__(
        self,
        mean_cost: np.ndarray,
        mean_quality: np.ndarray,
        target_cost: float,
        *,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        self.mean_cost = np.asarray(mean_cost, dtype=np.float64)
        self.mean_quality = np.asarray(mean_quality, dtype=np.float64)
        self.hull = non_decreasing_convex_hull(self.mean_cost, self.mean_quality)
        self.target_cost = float(target_cost)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lo, self._hi, self._p_hi = self._bracket(self.target_cost)
        self.name = f"ZeroRouter(c={target_cost:.2e})" if name is None else name

    def _bracket(self, target: float) -> tuple[int, int, float]:
        hc = self.mean_cost[self.hull]
        if target <= hc[0]:
            return int(self.hull[0]), int(self.hull[0]), 0.0
        if target >= hc[-1]:
            return int(self.hull[-1]), int(self.hull[-1]), 1.0
        j = int(np.searchsorted(hc, target))
        lo, hi = int(self.hull[j - 1]), int(self.hull[j])
        span = hc[j] - hc[j - 1]
        p_hi = 0.0 if span <= 0 else float((target - hc[j - 1]) / span)
        return lo, hi, p_hi

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        take_hi = self._rng.random() < self._p_hi
        arm = self._hi if take_hi else self._lo
        prob = self._p_hi if take_hi else 1.0 - self._p_hi
        if arm not in available:
            return int(available[0]), 1.0
        if self._lo == self._hi:
            prob = 1.0
        return int(arm), float(max(prob, 1e-12))

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    @property
    def expected_quality(self) -> float:
        return float(
            (1 - self._p_hi) * self.mean_quality[self._lo] + self._p_hi * self.mean_quality[self._hi]
        )
