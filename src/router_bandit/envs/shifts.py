"""Scripted distribution shifts (S1-S4).

The four scenarios from the proposal, each injected at a chosen round ``t0``:

====  ==========================  ===========================================
S1    a new model joins the pool  :class:`~router_bandit.envs.arms.ArmRegistry`
S2    a model's price changes     :class:`~router_bandit.envs.costs.CostModel`
S3    the query mix shifts        stream *ordering* (applied before the replay)
S4    a model degrades silently   per-round quality mask
====  ==========================  ===========================================

S1-S3 come essentially free from RouterBench's structure: arms are columns,
prices are a column family, and ``eval_name`` already labels the query mix. S4
has no natural counterpart in the data -- quality is fixed per (query, model)
pair -- so it is *synthesised* by converting a fraction of an arm's correct
answers into failures after ``t0``. That is a modelling choice, not a
measurement, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .arms import ArmRegistry
from .costs import CostModel, PriceChange

__all__ = ["MixShift", "QualityDegradation", "ShiftSchedule"]


@dataclass(frozen=True)
class QualityDegradation:
    """S4: from ``round``, arm ``arm`` fails on a ``severity`` fraction of the
    queries it would otherwise have answered correctly."""

    round: int
    arm: int
    severity: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must lie in [0, 1], got {self.severity}")


@dataclass(frozen=True)
class MixShift:
    """S3: from ``round``, draw queries using ``weights`` over families.

    Weights need not sum to 1; they are normalised. Families absent from the
    mapping get weight 0 after the shift.
    """

    round: int
    weights: dict[str, float]


@dataclass
class ShiftSchedule:
    """Composite schedule. An empty schedule is the stationary setting."""

    new_arms: dict[int, int] = field(default_factory=dict)
    price_changes: tuple[PriceChange, ...] = field(default_factory=tuple)
    mix_shift: MixShift | None = None
    degradations: tuple[QualityDegradation, ...] = field(default_factory=tuple)

    # -- component builders ------------------------------------------------ #
    def arm_registry(self, n_arms: int) -> ArmRegistry:
        return ArmRegistry(n_arms=n_arms, activation=dict(self.new_arms))

    def cost_model(self, n_arms: int) -> CostModel:
        return CostModel(n_arms=n_arms, changes=tuple(self.price_changes))

    @property
    def is_stationary(self) -> bool:
        return not (self.new_arms or self.price_changes or self.mix_shift or self.degradations)

    def change_points(self) -> list[int]:
        """Every round at which something changes -- used to segment the plots."""
        points = set(self.new_arms.values())
        points |= {c.round for c in self.price_changes}
        points |= {d.round for d in self.degradations}
        if self.mix_shift is not None:
            points.add(self.mix_shift.round)
        return sorted(p for p in points if p > 0)

    # -- S4 ---------------------------------------------------------------- #
    def degradation_mask(
        self, n_rounds: int, n_arms: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        """``(n_rounds, n_arms)`` boolean mask; True means "force a failure".

        Pre-computed once so a replay is reproducible and a policy cannot
        influence the realisation by choosing differently.
        """
        if not self.degradations:
            return None
        mask = np.zeros((n_rounds, n_arms), dtype=bool)
        for deg in self.degradations:
            if deg.round >= n_rounds:
                continue
            draws = rng.random(n_rounds - deg.round) < deg.severity
            mask[deg.round :, deg.arm] = draws
        return mask

    # -- S3 ---------------------------------------------------------------- #
    def build_order(
        self,
        families: pd.Series | np.ndarray,
        index: np.ndarray,
        rng: np.random.Generator,
        *,
        n_rounds: int | None = None,
    ) -> np.ndarray:
        """Stream order implementing the query-mix shift.

        Without a :class:`MixShift` this is a plain shuffle. With one, queries
        before ``round`` are drawn from the empirical family distribution and
        queries after it from ``weights``, both without replacement.
        """
        index = np.asarray(index, dtype=np.int64)
        if self.mix_shift is None:
            order = index.copy()
            rng.shuffle(order)
            return order if n_rounds is None else order[:n_rounds]

        labels = np.asarray(pd.Series(families).to_numpy())[index]
        pools: dict[str, list[int]] = {}
        for fam in pd.unique(labels):
            members = index[labels == fam].tolist()
            rng.shuffle(members)
            pools[fam] = members

        total = index.size if n_rounds is None else min(n_rounds, index.size)
        t0 = min(self.mix_shift.round, total)

        pre_counts = {fam: len(v) for fam, v in pools.items()}
        pre_total = sum(pre_counts.values())
        pre_w = {fam: c / pre_total for fam, c in pre_counts.items()}

        post_raw = {fam: max(float(self.mix_shift.weights.get(fam, 0.0)), 0.0) for fam in pools}
        post_sum = sum(post_raw.values())
        if post_sum <= 0:
            raise ValueError("MixShift.weights gives zero mass to every present family")
        post_w = {fam: v / post_sum for fam, v in post_raw.items()}

        def draw(n: int, weights: dict[str, float]) -> list[int]:
            """Take ``n`` queries, allocating them across families by ``weights``.

            Allocation is computed once as a count per family rather than by
            sampling a family per draw, which keeps this O(families) instead of
            O(n) and makes the realised mix match the requested weights closely
            even for short segments.
            """
            if n <= 0:
                return []
            live = {f: w for f, w in weights.items() if w > 0 and pools.get(f)}
            if not live:
                live = {f: 1.0 for f in pools if pools[f]}
            total_w = sum(live.values())
            if total_w <= 0:
                live = {f: 1.0 for f in pools if pools[f]}
                total_w = float(len(live)) or 1.0

            picked: list[int] = []
            for fam, w in live.items():
                take = min(round(n * w / total_w), len(pools[fam]))
                for _ in range(take):
                    picked.append(pools[fam].pop())

            # Top up (or trim) to exactly n from whatever remains.
            if len(picked) < n:
                leftovers = [f for f in pools if pools[f]]
                rng.shuffle(leftovers)
                for fam in leftovers:
                    while pools[fam] and len(picked) < n:
                        picked.append(pools[fam].pop())
                    if len(picked) >= n:
                        break
            picked = picked[:n]
            rng.shuffle(picked)
            return picked

        order = draw(t0, pre_w) + draw(total - t0, post_w)
        return np.asarray(order, dtype=np.int64)
