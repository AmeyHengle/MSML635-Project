"""Choose S2 and S4 strengths that move the supervised router's decision.

The deployed router scores arms from context alone, so a later price change
or a silent failure does not alter ``select``. Calibration applies the shock
to those predicted rewards and re-reads the argmax. That is the decision the
shock is being sized to change: the smallest multiplier or severity at which
the router's own ranking switches on at least 25% of post-shift queries.

A published price is a known change to the reward the router was trained on,
``quality - λ · cost``, so the price probe subtracts ``λ · cost · (m - 1)``.
Silent degradation zeros quality; the probe subtracts that quality, which is
the failed answer's reward when the prediction was exact. Failures are
independent across queries, so a severity ``s`` is scored as ``s`` times the
flip rate of failing on every post-shift query.

If the strongest value on the grid still falls short of 25%, that value is
kept and the flip rate it actually produced is reported with it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..envs.costs import PriceChange
from ..envs.shifts import MixShift, QualityDegradation, ShiftSchedule

__all__ = [
    "DEGRADE_SEVERITIES",
    "FLIP_TARGET",
    "PRICE_MULTIPLIERS",
    "ChosenStrength",
    "apply_degradation_shock",
    "apply_price_shock",
    "calibrate_degradation",
    "calibrate_price",
    "choose_strength",
    "degradation_flip_rate",
    "flip_rate",
    "most_selected_arm",
    "price_flip_rate",
    "pull_rate",
    "scenario_schedules",
]

FLIP_TARGET = 0.25
# Increasing treatment strength. The search keeps the smallest entry whose
# flip rate reaches FLIP_TARGET, or the last entry when none of them do.
PRICE_MULTIPLIERS = (2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0)
DEGRADE_SEVERITIES = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class ChosenStrength:
    """A grid point kept by the shift search.

    ``reached_target`` is false when even the strongest shock changes the
    router's arm on fewer than the target fraction of post-shift queries.
    ``strength`` is still that strongest shock, and ``flip_rate`` is the rate
    it actually produced.
    """

    strength: float
    flip_rate: float
    reached_target: bool


def flip_rate(before: np.ndarray, after: np.ndarray, t0: int) -> float:
    """Fraction of rounds at or after ``t0`` whose chosen arm changed."""
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("before and after must be 1-d arrays of equal length")
    if before.size == 0 or not 0 <= t0 < before.size:
        raise ValueError(f"t0 must lie in [0, {before.size}), got {t0}")
    return float((before[t0:] != after[t0:]).mean())


def pull_rate(actions: np.ndarray, arm: int, start: int, end: int) -> float:
    """Fraction of ``actions[start:end]`` equal to ``arm``."""
    actions = np.asarray(actions)
    if actions.ndim != 1 or start < 0 or end > actions.size or start >= end:
        raise ValueError(f"slice [{start}, {end}) is outside actions of length {actions.size}")
    return float((actions[start:end] == arm).mean())


def most_selected_arm(actions: np.ndarray, n_arms: int) -> int:
    """Arm chosen most often. Ties go to the lower index."""
    actions = np.asarray(actions, dtype=np.int64)
    if actions.ndim != 1 or actions.size == 0:
        raise ValueError("actions must be a non-empty 1-d array")
    if n_arms < 1 or np.any((actions < 0) | (actions >= n_arms)):
        raise ValueError(f"actions fall outside 0..{n_arms - 1}")
    return int(np.argmax(np.bincount(actions, minlength=n_arms)))


def choose_strength(
    strengths: tuple[float, ...] | list[float],
    rates: tuple[float, ...] | list[float],
    *,
    target: float = FLIP_TARGET,
) -> ChosenStrength:
    """Smallest strength whose flip rate is at least ``target``.

    ``strengths`` must be sorted in increasing order, one rate each. When no
    rate reaches ``target``, the last strength is kept together with the rate
    it produced.
    """
    strengths = [float(s) for s in strengths]
    rates = [float(r) for r in rates]
    if not strengths or len(strengths) != len(rates):
        raise ValueError("strengths and rates must be non-empty and the same length")
    for earlier, later in itertools.pairwise(strengths):
        if later < earlier:
            raise ValueError("strengths must be sorted in increasing order")
    for strength, rate in zip(strengths, rates):
        if rate >= target:
            return ChosenStrength(strength, rate, True)
    return ChosenStrength(strengths[-1], rates[-1], False)


def _panel(
    preds: np.ndarray, other: np.ndarray, arm: int, t0: int
) -> tuple[np.ndarray, np.ndarray]:
    preds = np.asarray(preds, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if preds.ndim != 2 or preds.shape != other.shape:
        raise ValueError("predictions and the shock panel must have the same shape")
    n_rounds, n_arms = preds.shape
    if not 0 <= arm < n_arms:
        raise ValueError(f"arm {arm} outside [0, {n_arms})")
    if n_rounds == 0 or not 0 <= t0 < n_rounds:
        raise ValueError(f"t0 must lie in [0, {n_rounds}), got {t0}")
    return preds, other


def apply_price_shock(
    preds: np.ndarray,
    cost: np.ndarray,
    arm: int,
    t0: int,
    lam: float,
    multiplier: float,
) -> np.ndarray:
    """Predicted rewards after arm ``arm`` costs ``multiplier`` times as much.

    Predictions are left untouched before ``t0``. The model was trained on
    ``quality - λ cost``, so multiplying the published price by ``m`` subtracts
    ``λ cost (m - 1)`` from that arm.
    """
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    preds, cost = _panel(preds, cost, arm, t0)
    adjusted = preds.copy()
    adjusted[t0:, arm] -= float(lam) * cost[t0:, arm] * (float(multiplier) - 1.0)
    return adjusted


def apply_degradation_shock(
    preds: np.ndarray, quality: np.ndarray, arm: int, t0: int
) -> np.ndarray:
    """Predicted rewards as if ``arm`` failed on every query from ``t0`` on.

    Zeroing quality subtracts it from a prediction of ``quality - λ cost``.
    """
    preds, quality = _panel(preds, quality, arm, t0)
    adjusted = preds.copy()
    adjusted[t0:, arm] -= quality[t0:, arm]
    return adjusted


def price_flip_rate(
    preds: np.ndarray,
    cost: np.ndarray,
    arm: int,
    t0: int,
    lam: float,
    multiplier: float,
) -> float:
    """Share of post-shift queries whose routed arm changes at this price."""
    base = np.asarray(preds).argmax(axis=1)
    adjusted = apply_price_shock(preds, cost, arm, t0, lam, multiplier).argmax(axis=1)
    return flip_rate(base, adjusted, t0)


def degradation_flip_rate(
    preds: np.ndarray,
    quality: np.ndarray,
    arm: int,
    t0: int,
    severity: float,
) -> float:
    """Expected share of post-shift queries whose routed arm changes.

    Each post-shift query fails independently with probability ``severity``,
    and only a failure can change the ranking, so the expected flip rate is
    ``severity`` times the rate from failing on every post-shift query.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must lie in [0, 1], got {severity}")
    base = np.asarray(preds).argmax(axis=1)
    failed = apply_degradation_shock(preds, quality, arm, t0).argmax(axis=1)
    return float(severity * flip_rate(base, failed, t0))


def calibrate_price(
    preds: np.ndarray,
    cost: np.ndarray,
    arm: int,
    t0: int,
    lam: float,
    multipliers: tuple[float, ...] = PRICE_MULTIPLIERS,
    *,
    target: float = FLIP_TARGET,
) -> ChosenStrength:
    """Smallest price multiplier that moves the router's arm on ``target`` of queries."""
    rates = [price_flip_rate(preds, cost, arm, t0, lam, multiplier) for multiplier in multipliers]
    return choose_strength(multipliers, rates, target=target)


def calibrate_degradation(
    preds: np.ndarray,
    quality: np.ndarray,
    arm: int,
    t0: int,
    severities: tuple[float, ...] = DEGRADE_SEVERITIES,
    *,
    target: float = FLIP_TARGET,
) -> ChosenStrength:
    """Smallest failure rate that moves the router's arm on ``target`` of queries."""
    rates = [degradation_flip_rate(preds, quality, arm, t0, severity) for severity in severities]
    return choose_strength(severities, rates, target=target)


def scenario_schedules(
    t0: int,
    *,
    oracle_arm: int,
    shocked_arm: int,
    multiplier: float,
    severity: float,
) -> dict[str, ShiftSchedule]:
    """S1–S4 plus the stationary control.

    S1 still withholds the oracle's most-used arm until ``t0``. S3 still
    reweights the same three families. S2 and S4 shock ``shocked_arm`` only,
    at the calibrated multiplier and severity.
    """
    return {
        "S0-stationary": ShiftSchedule(),
        "S1-new-model": ShiftSchedule(new_arms={oracle_arm: t0}),
        "S2-price-change": ShiftSchedule(
            price_changes=(PriceChange(round=t0, arm=shocked_arm, multiplier=multiplier),)
        ),
        "S3-query-mix": ShiftSchedule(
            mix_shift=MixShift(round=t0, weights={"mbpp": 3.0, "gsm8k": 3.0, "arc": 1.0})
        ),
        "S4-silent-degrade": ShiftSchedule(
            degradations=(QualityDegradation(round=t0, arm=shocked_arm, severity=severity),)
        ),
    }
