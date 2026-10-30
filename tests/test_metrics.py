"""Held-out intervals, recovery, APGR/CPT, and select latency."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router_bandit.baselines import OraclePolicy
from router_bandit.data import SyntheticStream
from router_bandit.envs import BanditEnv
from router_bandit.evaluation import (
    Recovery,
    average_performance_gap_recovered,
    bootstrap_ci,
    cost_performance_threshold,
    heldout_summary,
    performance_gap_recovered,
    pool_reference_arms,
    recovery_time,
    run_policy,
)
from router_bandit.policies import LinUCB


def test_bootstrap_defaults_match_the_heldout_interval():
    assert inspect.signature(bootstrap_ci).parameters["n_boot"].default == 2000
    assert inspect.signature(heldout_summary).parameters["n_boot"].default == 2000
    assert inspect.signature(recovery_time).parameters["window"].default == 50


def test_heldout_summary_intervals_bracket_the_query_mean():
    rng = np.random.default_rng(0)
    rewards = rng.normal(loc=0.4, scale=0.1, size=400)
    qualities = rng.uniform(0.2, 0.9, size=400)
    costs = rng.uniform(1e-5, 1e-3, size=400)
    out = heldout_summary(rewards, qualities, costs, n_boot=300, seed=1)

    assert out["heldout_reward"] == pytest.approx(float(rewards.mean()))
    assert out["heldout_quality"] == pytest.approx(float(qualities.mean()))
    assert out["heldout_dollar_cost"] == pytest.approx(float(costs.mean()))
    assert out["heldout_reward_lo"] < out["heldout_reward"] < out["heldout_reward_hi"]
    assert out["heldout_quality_lo"] < out["heldout_quality"] < out["heldout_quality_hi"]
    assert out["heldout_dollar_cost_lo"] < out["heldout_dollar_cost"]
    assert out["heldout_dollar_cost"] < out["heldout_dollar_cost_hi"]


def test_constant_heldout_interval_collapses():
    out = heldout_summary(np.full(50, 0.4), np.full(50, 0.7), np.full(50, 0.01), n_boot=20)
    assert out["heldout_reward_lo"] == pytest.approx(0.4)
    assert out["heldout_reward_hi"] == pytest.approx(0.4)


def test_recovery_stays_missing_when_the_policy_never_drops():
    """The old metric returned 0 here, because the first window was already inside 5%."""
    rewards = np.ones(400)
    rewards[200:] = 0.96  # 4% below the pre-shift mean: inside the band
    assert recovery_time(rewards, 200) == Recovery(None, None)


def test_recovery_reports_a_delayed_drop_that_never_returns():
    rewards = np.concatenate([np.ones(200), np.ones(60), np.zeros(200)])
    rec = recovery_time(rewards, 200, window=50)
    assert rec.departed_at == 13
    assert rec.rounds_to_recover is None


def test_recovery_counts_rounds_from_departure_back_into_band():
    rewards = np.concatenate([np.ones(200), np.zeros(80), np.ones(200)])
    rec = recovery_time(rewards, 200, window=50)
    assert rec.departed_at == 0
    assert rec.rounds_to_recover == 78


def test_missing_recovery_is_blank_rather_than_zero(tmp_path: Path):
    rec = recovery_time(np.ones(400), 200)
    row = {
        "departed_at": np.nan if rec.departed_at is None else rec.departed_at,
        "rounds_to_recover": np.nan if rec.rounds_to_recover is None else rec.rounds_to_recover,
    }
    path = tmp_path / "rq2.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    body = path.read_text().splitlines()[1]
    assert "0" not in body
    reloaded = pd.read_csv(path)
    assert pd.isna(reloaded.loc[0, "departed_at"])
    assert pd.isna(reloaded.loc[0, "rounds_to_recover"])


def test_pool_reference_arms_are_cheapest_and_highest_quality():
    weak, strong = pool_reference_arms(
        np.array([3.0, 1.0, 2.0]), np.array([0.2, 0.4, 0.9])
    )
    assert (weak, strong) == (1, 2)


def test_apgr_is_the_mean_gap_and_cpt_is_the_cheapest_half_recovery():
    assert performance_gap_recovered(0.75, 0.5, 1.0) == pytest.approx(0.5)
    assert np.isnan(performance_gap_recovered(0.5, 0.5, 0.5))

    pgr = np.array([0.2, 0.8, 0.5])
    cost = np.array([1.0, 4.0, 2.0])
    assert average_performance_gap_recovered(pgr) == pytest.approx(0.5)
    # 0.8 costs 4 and 0.5 costs 2; the smallest cost that clears half the gap is 2.
    assert cost_performance_threshold(cost, pgr) == pytest.approx(2.0)
    assert np.isnan(cost_performance_threshold(np.array([1.0, 2.0]), np.array([0.1, 0.4])))


def test_frozen_replay_does_not_learn_and_times_only_select():
    stream = SyntheticStream(n_rounds=40, n_arms=3, dim=4, noise=0.0, seed=0)
    env = BanditEnv(stream, lam=0.0, seed=0)
    policy = LinUCB(stream.n_arms, stream.dim, alpha=0.3, seed=0)

    first = run_policy(env, policy)
    assert first.select_times.shape == (stream.n_rounds,)
    assert np.isfinite(first.select_times).all()
    assert first.select_times.min() >= 0.0
    assert first.select_p50 == pytest.approx(float(np.percentile(first.select_times, 50)))
    assert first.select_p99 == pytest.approx(float(np.percentile(first.select_times, 99)))

    learned = policy.theta.copy()
    counts = policy._core.counts.copy()
    frozen = run_policy(env, policy, reset=False, update=False)
    assert np.array_equal(policy.theta, learned)
    assert np.array_equal(policy._core.counts, counts)
    assert frozen.rewards.shape == (stream.n_rounds,)
    assert np.isfinite(frozen.select_times).all()

    run_policy(env, policy, reset=False, update=True)
    assert not np.array_equal(policy._core.counts, counts)


def test_heldout_oracle_must_be_rebound_to_the_new_stream():
    stream_a = SyntheticStream(n_rounds=30, n_arms=4, dim=4, noise=0.0, seed=0)
    stream_b = SyntheticStream(n_rounds=30, n_arms=4, dim=4, noise=0.0, seed=1)
    env_a = BanditEnv(stream_a, lam=0.0, seed=0)
    env_b = BanditEnv(stream_b, lam=0.0, seed=0)
    trained = OraclePolicy(env_a.truth)
    run_policy(env_a, trained)

    stale = run_policy(env_b, trained, reset=False, update=False)
    rebound = run_policy(env_b, OraclePolicy(env_b.truth), reset=False, update=False)
    assert stale.oracle_match < 1.0
    assert rebound.oracle_match == pytest.approx(1.0)
