"""S2/S4 calibration: smallest shock that moves the router, and the logged rates."""

from __future__ import annotations

import numpy as np
import pytest

from router_bandit.baselines import SupervisedRouter
from router_bandit.envs import BanditEnv, MixShift, PriceChange, QualityDegradation, ShiftSchedule
from router_bandit.evaluation.shift_calibration import (
    FLIP_TARGET,
    ChosenStrength,
    apply_degradation_shock,
    apply_price_shock,
    calibrate_degradation,
    calibrate_price,
    choose_strength,
    degradation_flip_rate,
    flip_rate,
    most_selected_arm,
    price_flip_rate,
    pull_rate,
    scenario_schedules,
)


def _rewards() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Arm 0 leads by 0.1. A 2x price, or a total failure, hands the query to arm 1."""
    quality = np.array([[0.90, 0.40], [0.90, 0.40]])
    cost = np.array([[0.05, 0.01], [0.05, 0.01]])
    lam = 10.0
    preds = quality - lam * cost
    return preds, quality, cost, lam


def test_choose_strength_keeps_the_smallest_that_clears_the_target():
    chosen = choose_strength((1.2, 2.0, 10.0), (0.0, 0.25, 1.0))
    assert chosen == ChosenStrength(2.0, 0.25, True)


def test_choose_strength_keeps_the_strongest_shock_and_its_shortfall():
    chosen = choose_strength((2.0, 10.0, 100.0), (0.0, 0.10, 0.20), target=FLIP_TARGET)
    assert chosen == ChosenStrength(100.0, 0.20, False)
    assert chosen.flip_rate < FLIP_TARGET


def test_choose_strength_rejects_an_unsorted_grid():
    with pytest.raises(ValueError, match="increasing"):
        choose_strength((10.0, 2.0), (1.0, 1.0))


def test_price_search_stops_at_the_first_multiplier_that_flips():
    preds, _, cost, lam = _rewards()
    # 1.2x subtracts exactly the 0.1 margin, so the tie stays with arm 0.
    assert price_flip_rate(preds, cost, arm=0, t0=1, lam=lam, multiplier=1.2) == 0.0
    assert price_flip_rate(preds, cost, arm=0, t0=1, lam=lam, multiplier=2.0) == 1.0

    chosen = calibrate_price(preds, cost, 0, 1, lam, multipliers=(1.2, 2.0, 10.0))
    assert chosen == ChosenStrength(2.0, 1.0, True)


def test_degradation_search_scales_with_severity_and_stops_at_the_target():
    preds, quality, _, _ = _rewards()
    assert degradation_flip_rate(preds, quality, arm=0, t0=1, severity=1.0) == 1.0
    assert degradation_flip_rate(preds, quality, arm=0, t0=1, severity=0.2) == pytest.approx(0.2)

    chosen = calibrate_degradation(preds, quality, 0, 1, severities=(0.2, 0.4, 1.0))
    assert chosen == ChosenStrength(0.4, pytest.approx(0.4), True)


def test_degradation_that_cannot_move_the_arm_keeps_severity_one():
    preds = np.array([[0.2, 0.9], [0.2, 0.9]])
    quality = np.array([[0.0, 0.9], [0.0, 0.9]])
    chosen = calibrate_degradation(preds, quality, arm=0, t0=1, severities=(0.5, 1.0))
    assert chosen == ChosenStrength(1.0, 0.0, False)


def test_price_and_degradation_probes_match_the_environment_oracle():
    """When the predictions are the true rewards, re-ranking is the new oracle."""
    quality = np.array(
        [
            [0.90, 0.40],
            [0.90, 0.40],
            [0.90, 0.40],
            [0.20, 0.90],
        ]
    )
    cost = np.tile(np.array([0.05, 0.01]), (quality.shape[0], 1))
    lam = 10.0
    t0 = 2
    preds = quality - lam * cost

    class _Stream:
        n_rounds = quality.shape[0]
        n_arms = quality.shape[1]
        dim = 1

        def context(self, t):
            return np.zeros(1)

        def quality_vec(self, t):
            return quality[t]

        def cost_vec(self, t):
            return cost[t]

    price = ShiftSchedule(price_changes=(PriceChange(round=t0, arm=0, multiplier=2.0),))
    price_env = BanditEnv(_Stream(), lam, price, seed=0)
    reranked = apply_price_shock(preds, cost, arm=0, t0=t0, lam=lam, multiplier=2.0)
    assert np.array_equal(reranked.argmax(axis=1), price_env.truth.oracle_actions())
    assert flip_rate(preds.argmax(1), reranked.argmax(1), t0) == 0.5

    failed = ShiftSchedule(degradations=(QualityDegradation(round=t0, arm=0, severity=1.0),))
    failed_env = BanditEnv(_Stream(), lam, failed, seed=0)
    degraded = apply_degradation_shock(preds, quality, arm=0, t0=t0)
    assert np.array_equal(degraded.argmax(axis=1), failed_env.truth.oracle_actions())


def test_pull_rate_splits_on_the_shift_and_flip_rate_ignores_the_prefix():
    actions = np.array([0, 0, 1, 1])
    assert pull_rate(actions, 0, 0, 2) == 1.0
    assert pull_rate(actions, 1, 2, 4) == 1.0
    assert flip_rate(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 2) == 0.5


def test_most_selected_arm_breaks_ties_toward_the_lower_index():
    assert most_selected_arm(np.array([1, 0, 1, 0]), n_arms=3) == 0


def test_predict_rewards_matches_select():
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(30, 4))
    rewards = rng.normal(size=(30, 3))
    router = SupervisedRouter().fit(contexts, rewards)
    preds = router.predict_rewards(contexts)
    for i in range(len(contexts)):
        action, propensity = router.select(i, contexts[i], np.arange(3))
        assert action == int(preds[i].argmax())
        assert propensity == 1.0


def test_s1_and_s3_are_unchanged_while_s2_and_s4_use_the_calibrated_arm():
    schedules = scenario_schedules(10, oracle_arm=1, shocked_arm=4, multiplier=5.0, severity=0.9)
    assert schedules["S0-stationary"].is_stationary
    assert schedules["S1-new-model"].new_arms == {1: 10}
    assert schedules["S1-new-model"].price_changes == ()
    assert schedules["S3-query-mix"].mix_shift == MixShift(
        round=10, weights={"mbpp": 3.0, "gsm8k": 3.0, "arc": 1.0}
    )
    assert schedules["S2-price-change"].price_changes == (
        PriceChange(round=10, arm=4, multiplier=5.0),
    )
    assert schedules["S4-silent-degrade"].degradations == (
        QualityDegradation(round=10, arm=4, severity=0.9),
    )
