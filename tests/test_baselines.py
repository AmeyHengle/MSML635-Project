"""Cascade scoring, the Zero Router hull sweep, and the PILOT-style policy."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router_bandit.baselines import (
    CASCADE_JUDGE_ERRORS,
    ZeroRouter,
    hull_target_costs,
    summarize_cascade,
    zero_frontier_policies,
)
from router_bandit.data import SyntheticStream
from router_bandit.envs import BanditEnv
from router_bandit.evaluation import run_policy
from router_bandit.policies import LinUCB, PilotStyle

FIXTURE = Path("data/fixtures/rb0_sample5k.parquet")
needs_fixture = pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture {FIXTURE} not present")
ROOT = Path(__file__).resolve().parents[1]


def _ridge_theta(X: np.ndarray, rewards: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + penalty, X.T @ rewards).T


def test_cascade_judge_errors_match_the_routerbench_breakpoints():
    assert CASCADE_JUDGE_ERRORS == (0.0, 0.1, 0.2)


def test_cascade_reports_quality_cumulative_cost_and_reward():
    quality = np.tile(np.array([0.0, 1.0]), (20, 1))
    cost_norm = np.tile(np.array([0.1, 0.4]), (20, 1))
    perfect = summarize_cascade(quality, cost_norm, lam=2.0, cost_scale=10.0, judge_error=0.0)
    blind = summarize_cascade(quality, cost_norm, lam=2.0, cost_scale=10.0, judge_error=1.0)

    assert perfect["mean_quality"] == pytest.approx(1.0)
    assert perfect["mean_cost"] == pytest.approx(0.5)
    assert perfect["mean_dollar_cost"] == pytest.approx(5.0)
    assert perfect["mean_reward"] == pytest.approx(1.0 - 2.0 * 0.5)
    assert np.array_equal(perfect["final_arm"], np.ones(20, dtype=np.int64))

    assert blind["mean_quality"] == pytest.approx(0.0)
    assert blind["mean_cost"] == pytest.approx(0.1)
    assert blind["mean_reward"] == pytest.approx(-0.2)
    assert perfect["mean_cost"] > blind["mean_cost"]


def test_hull_targets_include_the_vertices_and_increase():
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    qualities = np.array([0.2, 0.5, 0.4, 0.8])
    targets = hull_target_costs(costs, qualities, n_grid=5)

    assert targets[0] == pytest.approx(1.0)
    assert targets[-1] == pytest.approx(4.0)
    assert np.any(np.isclose(targets, 2.0))
    assert np.all(np.diff(targets) > 0)

    routers = zero_frontier_policies(costs, qualities, seed=0, n_grid=5)
    assert [router.target_cost for router in routers] == pytest.approx(targets)
    assert len({router.name for router in routers}) == len(routers)
    assert all(router.name.startswith("ZeroRouter-hull[") for router in routers)

    named = ZeroRouter(costs, qualities, 2.5, name="ZeroRouter")
    assert named.name == "ZeroRouter"
    assert named.target_cost == pytest.approx(2.5)


def test_pilot_prior_matches_ridge_and_survives_reset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    rewards = X @ rng.normal(size=(4, 2))
    ridge = 1.0
    fitted = _ridge_theta(X, rewards, ridge)

    policy = PilotStyle(2, 4, ridge=ridge, prior_strength=1.0)
    assert inspect.signature(PilotStyle).parameters["prior_strength"].default == 1.0
    policy.set_prior(X, rewards)
    assert np.allclose(policy.theta, fitted)

    cold = np.eye(4) / ridge
    full = np.linalg.inv(X.T @ X + ridge * np.eye(4))
    # One pseudo-count keeps the covariance near the cold start.
    assert np.linalg.norm(policy._core._A_inv[0] - cold) < np.linalg.norm(full - cold)

    other = policy._core._A_inv[1].copy()
    policy.update(0, X[0], 0, 1.0)
    assert np.allclose(policy._core._A_inv[1], other)
    assert not np.allclose(policy._core._A_inv[0], other)

    moved = policy.theta.copy()
    policy.reset()
    assert np.allclose(policy.theta, fitted)
    assert not np.allclose(policy.theta, moved)
    assert policy._spent == 0.0

    weak = PilotStyle(2, 4, ridge=ridge, prior_strength=0.0)
    weak.set_prior(X, rewards)
    assert np.allclose(weak.theta, fitted)
    assert np.allclose(weak._core._A_inv[0], cold)


def test_pilot_masks_arms_that_would_break_the_pace():
    costs = np.array([0.1, 0.5])
    paced = PilotStyle(2, 3, alpha=0.0, arm_costs=costs, pace=0.3)
    paced._core._theta[1] = np.ones(3)
    context = np.ones(3)
    available = np.arange(2)

    actions = [paced.select(t, context, available)[0] for t in range(6)]
    assert actions[0] == 0
    assert actions[1] == 1
    assert 1 in actions
    assert paced._spent <= 0.3 * 6 + 1e-8

    open_budget = PilotStyle(2, 3, alpha=0.0, arm_costs=costs, pace=np.inf)
    open_budget._core._theta[1] = np.ones(3)
    assert open_budget.select(0, context, available)[0] == 1

    plain = LinUCB(2, 3, alpha=0.0)
    plain._core._theta[1] = np.ones(3)
    assert plain.select(0, context, available)[0] == 1

    with pytest.raises(ValueError):
        PilotStyle(2, 3, pace=-1.0, arm_costs=costs)
    with pytest.raises(ValueError):
        PilotStyle(2, 3, pace=1.0)


def test_frozen_replay_keeps_the_posterior_and_restarts_the_pace():
    stream = SyntheticStream(n_rounds=15, n_arms=2, dim=3, noise=0.0, seed=1)
    env = BanditEnv(stream, lam=0.0, seed=0)
    policy = PilotStyle(
        stream.n_arms, stream.dim, alpha=0.0, arm_costs=np.array([0.1, 5.0]), pace=0.1
    )
    policy._core._theta[:] = 0.0
    policy._core._theta[1, -1] = 100.0

    plain = LinUCB(stream.n_arms, stream.dim, alpha=0.0)
    plain._core._theta[:] = policy._core._theta
    assert plain.select(0, env.context(0), np.arange(2))[0] == 1

    first = run_policy(env, policy, reset=False)
    assert np.all(first.actions == 0)
    learned = policy.theta.copy()
    second = run_policy(env, policy, reset=False, update=False)
    assert np.allclose(policy.theta, learned)
    assert np.all(second.actions == 0)
    assert policy._spent == pytest.approx(0.1 * stream.n_rounds)


def _load_experiments():
    os.environ["RB_SOURCE"] = "fixture"
    exp = str(ROOT / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import importlib

    import common
    import run_all

    return importlib.reload(common), importlib.reload(run_all)


def test_cascade_rows_cover_the_three_judge_errors():
    common, run_all = _load_experiments()
    stream = SyntheticStream(n_rounds=25, n_arms=3, dim=4, noise=0.0, seed=0)
    env = BanditEnv(stream, lam=1.0, seed=0)

    class _Data:
        cost = stream.cost_matrix()
        cost_scale = 1.0

    class _Setup:
        data = _Data()

    rows = run_all._cascade_rows(
        _Setup(), stream, stream, env, env, 1.0, 0, 0.2, 0.9
    )
    assert [row["policy"] for row in rows] == ["Cascade(e=0)", "Cascade(e=0.1)", "Cascade(e=0.2)"]
    assert all(row["frontier"] is False for row in rows)
    for row in rows:
        assert row["mean_reward"] == pytest.approx(row["mean_quality"] - 1.0 * row["mean_cost"])
        assert np.isfinite(row["mean_dollar_cost"])
        assert np.isnan(row["select_p50_s"])
        assert np.isnan(row["select_p99_s"])
    assert "frontier=True" in inspect.getsource(run_all.rq1)
    assert "zero_frontier_policies" in inspect.getsource(run_all.rq1)

    frame = pd.DataFrame({
        "policy": ["ZeroRouter", "ZeroRouter-hull[0]", "Cascade(e=0)"],
        "frontier": [False, True, False],
    })
    assert common.is_frontier(frame).tolist() == [False, True, False]


@needs_fixture
def test_build_policies_matches_zero_router_and_pilot_to_the_oracle_spend():
    common, _run_all = _load_experiments()
    setup = common.build_setup(seed=0, dim=8)
    env, stream = common.make_env(setup, common.LAMBDA_REF, 0)
    policies = common.build_policies(setup, stream, env, common.LAMBDA_REF, 0)
    by_name = {policy.name: policy for policy in policies}

    router = by_name["ZeroRouter"]
    pilot = by_name["PILOT(a=0.3)"]
    expected = common.oracle_mean_cost(setup, stream.order, common.LAMBDA_REF)
    mean_cost, _mean_quality = common.calibration_stats(setup)

    assert router.target_cost == pytest.approx(expected)
    assert pilot.pace == pytest.approx(expected)
    assert np.allclose(pilot.arm_costs, mean_cost)
    assert np.linalg.norm(pilot.theta) > 0.0
    theta = pilot.theta.copy()
    pilot.reset()
    assert np.allclose(pilot.theta, theta)

    frontier = zero_frontier_policies(mean_cost, _mean_quality, seed=0)
    assert len(frontier) >= 2
    assert frontier[0].target_cost < frontier[-1].target_cost
