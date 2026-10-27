"""Dataset-layer tests.

These cover verification items (i), (ii), and part of (v) from the proposal:
reward/cost bookkeeping is correct, the recomputed oracle agrees with the
oracle column RouterBench ships, and the feature pipeline cannot see data it
is not entitled to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router_bandit.data import (
    MODELS,
    N_ARMS,
    RouterBenchStream,
    Stream,
    SyntheticStream,
    TfidfSvdBackend,
    build,
    build_features,
    build_stream,
    clean,
    eval_to_family,
    load_raw,
    make_splits,
    parse_prompt,
    short_names,
)
from router_bandit.data import registry as R

FIXTURE = Path("data/fixtures/rb0_sample5k.parquet")
needs_fixture = pytest.mark.skipif(
    not FIXTURE.exists(), reason=f"fixture {FIXTURE} not present"
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_is_consistent():
    assert len(MODELS) == N_ARMS == 11
    assert len(set(MODELS)) == N_ARMS, "duplicate model names"
    assert len(set(short_names())) == N_ARMS, "short names collide"
    for i, m in enumerate(MODELS):
        assert R.arm_index(m) == i
        assert R.cost_col(m).endswith("|total_cost")
        assert R.quality_col(m) == m


@pytest.mark.parametrize(
    ("eval_name", "expected"),
    [
        ("mmlu_high_school_physics", "mmlu"),
        ("hellaswag", "hellaswag"),
        ("grade-school-math", "gsm8k"),
        ("gsm8k", "gsm8k"),
        ("winogrande", "winogrande"),
        ("arc", "arc"),
        ("mbpp", "mbpp"),
        ("mt-bench", "mtbench"),
        ("Chinese_character_riddles", "other"),
        ("abstract2title", "other"),
    ],
)
def test_eval_to_family(eval_name, expected):
    assert eval_to_family(eval_name) == expected


def test_undocumented_evals_are_tagged_not_dropped():
    fam = eval_to_family("chinese_tang_poetries")
    assert fam == R.OTHER_FAMILY
    assert not R.is_documented(fam)


def test_parse_prompt_handles_stringified_lists():
    assert parse_prompt("['a', 'b']") == "a\nb"
    assert parse_prompt(["a", "b"]) == "a\nb"
    assert parse_prompt("plain text") == "plain text"
    assert parse_prompt("[unparseable") == "[unparseable"
    assert parse_prompt(None) == ""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data():
    if not FIXTURE.exists():
        pytest.skip(f"fixture {FIXTURE} not present")
    return build(clean(load_raw(FIXTURE)))


@needs_fixture
def test_matrices_are_well_formed(data):
    assert data.quality.shape == data.cost.shape
    assert data.n_arms == N_ARMS
    assert data.quality.dtype == np.float64
    assert np.isfinite(data.quality).all()
    assert np.isfinite(data.cost).all()


@needs_fixture
def test_quality_in_unit_interval_and_cost_positive(data):
    assert data.quality.min() >= 0.0
    assert data.quality.max() <= 1.0
    assert data.cost.min() > 0.0


@needs_fixture
def test_cost_normalisation_is_unit_scaled(data):
    assert np.isclose(data.cost_norm.max(), 1.0)
    assert np.allclose(data.cost_norm * data.cost_scale, data.cost)


@needs_fixture
def test_mtbench_is_excluded_by_default(data):
    assert "mtbench" not in set(data.meta["family"])


@needs_fixture
def test_recomputed_oracle_matches_shipped_column(data):
    """Verification test (ii): our oracle must reproduce RouterBench's."""
    if R.ORACLE_COL not in data.meta.columns:
        pytest.skip("fixture has no shipped oracle column")
    # lambda -> 0+ reproduces "best quality, cheapest on ties", which is the
    # rule RouterBench used to build its oracle column.
    ours = np.array(MODELS, dtype=object)[data.oracle(1e-9)]
    shipped = data.meta[R.ORACLE_COL].to_numpy()
    answerable = shipped != R.NO_MODEL_CORRECT
    assert answerable.sum() > 0
    agreement = float((ours[answerable] == shipped[answerable]).mean())
    assert agreement == pytest.approx(1.0), f"oracle disagreement: {agreement:.4f}"


@needs_fixture
def test_no_policy_can_beat_the_oracle(data):
    """Verification test (ii), second half."""
    for lam in (0.0, 1.0, 10.0, 30.0):
        rewards = data.rewards(lam)
        oracle_reward = rewards.max(axis=1)
        for arm in range(data.n_arms):
            assert (rewards[:, arm] <= oracle_reward + 1e-12).all()


@needs_fixture
def test_lambda_increases_cheap_arm_preference(data):
    """Higher lambda must never make the oracle pick a more expensive arm on average."""
    mean_cost_low = data.cost[np.arange(data.n_queries), data.oracle(1.0)].mean()
    mean_cost_high = data.cost[np.arange(data.n_queries), data.oracle(30.0)].mean()
    assert mean_cost_high <= mean_cost_low + 1e-12


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@needs_fixture
def test_splits_are_disjoint_and_complete(data):
    splits = make_splits(data.meta["family"], seed=0)
    splits.check_disjoint(data.n_queries)
    assert splits.stream.size > splits.calibration.size


@needs_fixture
def test_splits_are_deterministic(data):
    a = make_splits(data.meta["family"], seed=7)
    b = make_splits(data.meta["family"], seed=7)
    assert np.array_equal(a.stream, b.stream)
    c = make_splits(data.meta["family"], seed=8)
    assert not np.array_equal(a.stream, c.stream)


def test_splits_reject_bad_fractions():
    fams = pd.Series(["a"] * 50 + ["b"] * 50)
    with pytest.raises(ValueError):
        make_splits(fams, fractions=(0.5, 0.4, 0.4))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@needs_fixture
def test_features_fit_only_on_calibration(data):
    texts = data.meta[R.PROMPT_COL].astype(str).tolist()
    splits = make_splits(data.meta["family"], seed=0)
    backend = TfidfSvdBackend(n_components=16, min_df=1)
    features, fitted = build_features(texts, splits.calibration, backend)

    assert features.shape == (len(texts), fitted.n_components)
    assert np.isfinite(features).all()

    # Refitting on the stream split must give different features: if it did
    # not, the reducer was not actually conditioned on its fit set.
    other, _ = build_features(texts, splits.stream, TfidfSvdBackend(n_components=16, min_df=1))
    assert not np.allclose(features, other)


def test_build_features_rejects_empty_fit_index():
    with pytest.raises(ValueError):
        build_features(["a", "b"], np.array([], dtype=int), TfidfSvdBackend(4, min_df=1))


def test_build_features_rejects_out_of_bounds_index():
    with pytest.raises(IndexError):
        build_features(["a", "b"], np.array([5]), TfidfSvdBackend(4, min_df=1))


# --------------------------------------------------------------------------- #
# Stream
# --------------------------------------------------------------------------- #
@needs_fixture
def test_stream_round_trips(data):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(data.n_queries, 8))
    splits = make_splits(data.meta["family"], seed=0)
    stream = build_stream(data, features, splits.stream, seed=1)

    assert isinstance(stream, Stream)
    assert stream.n_rounds == splits.stream.size
    assert stream.n_arms == N_ARMS
    assert stream.dim == 9  # 8 features + bias
    assert stream.context(0).shape == (9,)
    assert stream.quality_vec(0).shape == (N_ARMS,)
    assert np.isclose(stream.context(0)[-1], 1.0)
    assert stream.quality_matrix().shape == (stream.n_rounds, N_ARMS)


@needs_fixture
def test_stream_shuffles_away_file_order(data):
    features = np.zeros((data.n_queries, 4))
    splits = make_splits(data.meta["family"], seed=0)
    ordered = build_stream(data, features, splits.stream, shuffle=False)
    shuffled = build_stream(data, features, splits.stream, shuffle=True, seed=3)
    assert not np.array_equal(ordered.order, shuffled.order)
    assert sorted(ordered.order.tolist()) == sorted(shuffled.order.tolist())


@needs_fixture
def test_stream_rejects_misaligned_features(data):
    with pytest.raises(ValueError):
        RouterBenchStream(data, np.zeros((data.n_queries - 1, 4)), np.arange(10))


# --------------------------------------------------------------------------- #
# Synthetic stream
# --------------------------------------------------------------------------- #
def test_synthetic_satisfies_stream_protocol():
    s = SyntheticStream(n_rounds=200, n_arms=5, dim=6, seed=0)
    assert isinstance(s, Stream)
    assert s.n_rounds == 200
    assert s.n_arms == 5
    assert s.dim == 7  # 6 + bias
    assert s.context(0).shape == (7,)
    assert s.quality_vec(0).shape == (5,)


def test_synthetic_exposes_ground_truth():
    s = SyntheticStream(n_rounds=300, n_arms=4, dim=5, seed=1)
    assert s.true_theta.shape == (4, 6)
    assert s.expected_quality().shape == (300, 4)
    assert 0.0 <= s.expected_quality().min() and s.expected_quality().max() <= 1.0

    uniform = s.true_uniform_value(lam=1.0)
    per_arm = [s.true_policy_value(np.full(300, a), lam=1.0) for a in range(4)]
    assert uniform == pytest.approx(float(np.mean(per_arm)), rel=1e-9)


def test_synthetic_costs_are_ordered_and_normalised():
    s = SyntheticStream(n_rounds=50, n_arms=6, dim=4, cost_spread=72.0, seed=0)
    costs = s.cost_vec(0)
    assert np.all(np.diff(costs) > 0), "arms should be ordered cheap -> expensive"
    assert np.isclose(costs.max(), 1.0)
    assert costs.max() / costs.min() == pytest.approx(72.0, rel=1e-6)


def test_synthetic_bernoulli_variant_is_binary():
    s = SyntheticStream(n_rounds=200, n_arms=3, dim=4, bernoulli=True, seed=0)
    assert set(np.unique(s.quality_matrix())).issubset({0.0, 1.0})


def test_synthetic_is_reproducible():
    a = SyntheticStream(n_rounds=100, n_arms=3, dim=4, seed=42)
    b = SyntheticStream(n_rounds=100, n_arms=3, dim=4, seed=42)
    assert np.array_equal(a.quality_matrix(), b.quality_matrix())
