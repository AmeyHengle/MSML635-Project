"""Verification tests (iii), (iv), and the hull/AIQ half of (v).

(i), (ii), and the split half of (v) live in ``test_data.py``. (iii) and (iv)
need a linear bandit whose parameters are known, which RouterBench cannot
provide, so they run on :class:`~router_bandit.data.synthetic.SyntheticStream`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from router_bandit.baselines import aiq, non_decreasing_convex_hull
from router_bandit.data import SyntheticStream, build, clean, load_raw
from router_bandit.envs import BanditEnv
from router_bandit.evaluation import doubly_robust, fit_reward_model, ips, run_policy, snips
from router_bandit.policies import LinUCB, UniformRandom

FIXTURE = Path("data/fixtures/rb0_sample5k.parquet")
needs_fixture = pytest.mark.skipif(
    not FIXTURE.exists(), reason=f"fixture {FIXTURE} not present"
)

def _linear_stream(seed: int, n_rounds: int) -> SyntheticStream:
    # Noise-free, and skill_scale small enough that clipping is rare, so the
    # reward stays inside the linear class LinUCB is built for.
    return SyntheticStream(
        n_rounds=n_rounds,
        n_arms=5,
        dim=8,
        noise=0.0,
        bernoulli=False,
        skill_scale=0.25,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# (iii) LinUCB attains sublinear regret on a linear bandit
# --------------------------------------------------------------------------- #
def test_linucb_regret_is_sublinear_on_linear_bandit():
    """Second-half regret stays under half the first half, and regret/t falls."""
    n_rounds = 2000
    mid = n_rounds // 2
    for seed in (0, 1, 2, 3):
        stream = _linear_stream(seed, n_rounds)
        env = BanditEnv(stream, lam=0.0, seed=seed)
        policy = LinUCB(stream.n_arms, stream.dim, alpha=1.0, seed=seed)
        regret = run_policy(env, policy).instant_regret

        first = float(regret[:mid].sum())
        second = float(regret[mid:].sum())
        assert first > 0.0, f"seed {seed}: first half has no regret to compare"
        ratio = second / first
        assert ratio < 0.5, f"seed {seed}: second/first regret ratio {ratio:.3f}"
        assert regret.sum() / n_rounds < first / mid, (
            f"seed {seed}: regret per round did not fall after the midpoint"
        )


# --------------------------------------------------------------------------- #
# (iv) IPS, SNIPS, and DR are unbiased on uniform logs
# --------------------------------------------------------------------------- #
def test_ope_matches_known_value_on_uniform_logs():
    """Mean estimate over seeds is within 0.05 of the known policy value."""
    n_rounds = 2000
    lam = 1.0
    n_arms = 4
    errors = {"IPS": [], "SNIPS": [], "DR": []}

    for seed in range(8):
        stream = SyntheticStream(
            n_rounds=n_rounds,
            n_arms=n_arms,
            dim=8,
            noise=0.0,
            skill_scale=0.2,
            seed=seed,
        )
        env = BanditEnv(stream, lam=lam, seed=seed)
        log = run_policy(env, UniformRandom(n_arms, seed=seed), log=True).log
        assert np.allclose(log.propensities, 1.0 / n_arms)

        # Target is the noise-free optimal arm on each round: its value is
        # known exactly, and the log never reveals that choice.
        target = stream.expected_rewards(lam).argmax(axis=1)
        truth = stream.true_policy_value(target, lam)
        q_hat = fit_reward_model(log.contexts, log.actions, log.rewards, n_arms)
        estimates = {
            "IPS": ips(log.rewards, log.actions, log.propensities, target).value,
            "SNIPS": snips(log.rewards, log.actions, log.propensities, target).value,
            "DR": doubly_robust(
                log.rewards, log.actions, log.propensities, target, q_hat
            ).value,
        }
        for name, value in estimates.items():
            errors[name].append(value - truth)

    for name, errs in errors.items():
        bias = float(np.mean(errs))
        assert abs(bias) < 0.05, f"{name} bias {bias:+.4f} is outside 0.05"


# --------------------------------------------------------------------------- #
# (v) Non-decreasing convex hull and AIQ, on the fixture
# --------------------------------------------------------------------------- #
def _pool_frontier(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = build(clean(load_raw(path)))
    costs = data.cost.mean(axis=0)
    qualities = data.quality.mean(axis=0)
    hull = non_decreasing_convex_hull(costs, qualities)
    return costs, qualities, hull, float(aiq(costs, qualities, hull))


@needs_fixture
def test_convex_hull_increases_in_cost_and_aiq_is_stable():
    """Hull cost is strictly increasing, and a second run reproduces AIQ."""
    costs, qualities, hull, score = _pool_frontier(FIXTURE)
    costs_b, qualities_b, hull_b, score_b = _pool_frontier(FIXTURE)

    assert hull.size >= 2
    assert np.all(np.diff(costs[hull]) > 0)
    assert np.all(np.diff(qualities[hull]) > 0)
    if hull.size >= 3:
        slopes = np.diff(qualities[hull]) / np.diff(costs[hull])
        assert np.all(np.diff(slopes) < 0)

    assert np.allclose(costs, costs_b)
    assert np.allclose(qualities, qualities_b)
    assert np.array_equal(hull, hull_b)
    assert score == pytest.approx(score_b)
    assert 0.0 < score <= 1.0
