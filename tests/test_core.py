def test_linucb_import():
    from obp.policy import LinUCB

    assert LinUCB is not None


def test_pivot_split_and_pca(tmp_path):
    """Wide pickle to long, drop a query missing an arm, disjoint family split, train-only PCA.

    No download and no RouterBench file.
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    from src.data import filter_evals, fit_contexts, load_long, pivot_wide, split_ids

    raw = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "prompt": ["a", "b"],
            "eval_name": ["mmlu-anatomy", "hellaswag"],
            "open-cheap": [0.2, 0.2],
            "open-cheap|total_cost": [1.0, 1.0],
            "open-cheap|model_response": ["x", "x"],
            "GPT-4-test": [0.9, 0.9],
            "GPT-4-test|total_cost": [10.0, 10.0],
        }
    )
    raw.to_pickle(tmp_path / "bench.pkl")
    long = load_long(tmp_path / "bench.pkl")
    
    # Check that the long DataFrame has the expected model names and sample IDs after filtering evaluations.
    assert set(long["model_name"]) == {"open-cheap", "GPT-4-test"}
    assert list(filter_evals(long)["sample_id"]) == ["s1", "s1"]

    # Create a wide DataFrame with four arms and their corresponding costs and performances for various queries.
    arms = ["open-cheap", "open-mid", "open-strong", "GPT-4-test"]
    specs = {
        "open-cheap": (1.0, 0.2),
        "open-mid": (3.0, 0.5),
        "open-strong": (5.0, 0.8),
        "GPT-4-test": (10.0, 0.9),
        "claude-held": (7.0, 0.85),
    }
    families = ["mmlu", "grade-school-math", "mbpp", "mtbench"]
    queries = [(f"{family}-{i}", family) for family in families for i in range(8)]
    queries.append(("gap", "mbpp"))
    rows = []
    for sample_id, eval_name in queries:
        for model, (cost, score) in specs.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "prompt": sample_id,
                    "eval_name": eval_name,
                    "model_name": model,
                    "cost": cost,
                    "performance": score,
                }
            )
    frame = pd.DataFrame(rows)
    frame = frame[~((frame.sample_id == "gap") & (frame.model_name == "GPT-4-test"))]

    wide = pivot_wide(frame, arms, "claude-held")
    # Check that the wide DataFrame has the expected sample IDs, costs, and quality frontier values after pivoting.
    assert "gap" not in set(wide["sample_id"])
    assert len(wide) == len(queries) - 1
    assert set(wide["cost_cheap"]) == {1.0}
    assert set(wide["quality_frontier"]) == {0.9}

    split = split_ids(wide)
    train, test = set(split["train"]), set(split["test"])
    # Check that the training and test sets are disjoint and cover all sample IDs in the wide DataFrame.
    assert train.isdisjoint(test)
    assert train | test == set(wide["sample_id"].astype(str))
    labels = dict(zip(families, ("mmlu", "gsm8k", "mbpp", "mt-bench")))
    by_id = dict(zip(wide["sample_id"].astype(str), wide["eval_name"].map(labels)))
    # Check that each family is represented in both the training and test sets.
    for family in ("mmlu", "gsm8k", "mbpp", "mt-bench"):
        assert any(by_id[sample_id] == family for sample_id in train)
        assert any(by_id[sample_id] == family for sample_id in test)

    pca_dim = 3
    embeddings = np.random.default_rng(0).normal(size=(len(wide), 8))
    context_train, context_test, pca = fit_contexts(embeddings, wide, split, pca_dim, tmp_path)
    # Check that the shapes of the training and test contexts are as expected, and that the PCA model has been fitted correctly.
    assert context_train.shape == (len(split["train"]), pca_dim + 1)
    assert context_test.shape == (len(split["test"]), pca_dim + 1)
    assert pca.n_samples_ == len(split["train"])
    assert np.all(context_train[:, -1] == 1.0)
    assert np.all(context_test[:, -1] == 1.0)

    positions = {sample_id: index for index, sample_id in enumerate(wide["sample_id"].astype(str))}
    train_x = embeddings[[positions[sample_id] for sample_id in split["train"]]]
    reference = PCA(n_components=pca_dim, random_state=0).fit(train_x)
    # Check that the transformed training contexts match the reference PCA transformation of the training embeddings.
    assert np.allclose(context_train[:, :-1], reference.transform(train_x))


def test_check_i_reward_normalization_and_oracle():
    """Check (i): Hand-written expected numbers for normalization, reward, regret, and tie-breaking.

    Qualities (1.0, 0.0), normalized costs (0.25, 1.0), lambda = 1 give rewards (0.75, -1.0)
    and regret 1.75 for arm 1. A raw cost of 1.5 with train max 1.0 normalizes to 1.0.
    Tied rewards select the cheaper arm.
    """
    import numpy as np

    from src.sim import compute_reward, find_oracle_action, normalize_cost

    # Raw cost of 1.5 with train max 1.0 normalizes to 1.0
    norm_c = normalize_cost(1.5, train_max_cost=1.0)
    assert np.isclose(norm_c, 1.0)

    # Qualities (1.0, 0.0), normalized costs (0.25, 1.0), lambda = 1
    qualities = np.array([1.0, 0.0])
    norm_costs = np.array([0.25, 1.0])
    rewards = compute_reward(qualities, norm_costs, lambda_weight=1.0)

    # Give rewards (0.75, -1.0)
    assert np.allclose(rewards, np.array([0.75, -1.0]))

    # Oracle action and regret 1.75 for arm 1
    best_arm, best_rew = find_oracle_action(rewards, norm_costs)
    assert best_arm == 0
    assert np.isclose(best_rew, 0.75)
    regret_arm_1 = best_rew - rewards[1]
    assert np.isclose(regret_arm_1, 1.75)

    # Tied rewards select the cheaper arm
    tied_rewards = np.array([0.5, 0.5])
    costs_diff = np.array([0.8, 0.2])  # arm 1 is cheaper
    chosen_arm, _ = find_oracle_action(tied_rewards, costs_diff)
    assert chosen_arm == 1

    tied_rewards_2 = np.array([0.5, 0.5])
    costs_diff_2 = np.array([0.1, 0.4])  # arm 0 is cheaper
    chosen_arm_2, _ = find_oracle_action(tied_rewards_2, costs_diff_2)
    assert chosen_arm_2 == 0


def test_check_ii_oracle_upper_bound_and_uniform_regret():
    """Check (ii): On a fixed synthetic matrix of length 2000, every arm's reward is at most
    the oracle reward. A uniform policy's regret(t)/t at t = 2000 and at t = 1000 differ by
    less than 10% relative.
    """
    import numpy as np

    from src.sim import QueryStream, compute_oracle_matrix, run_episode

    rng = np.random.default_rng(0)
    T = 2000
    K = 4
    d = 10

    # Synthetic contexts, qualities, and normalized costs
    contexts = rng.normal(size=(T, d))
    # Contexts have constant 1 appended
    contexts = np.concatenate([contexts, np.ones((T, 1))], axis=1)

    qualities = rng.uniform(0.1, 0.9, size=(T, K))
    costs = rng.uniform(0.05, 0.95, size=(T, K))
    rewards = qualities - 1.0 * costs

    stream = QueryStream(contexts=contexts, rewards=rewards, costs=costs, qualities=qualities)

    # Every arm's reward is at most the oracle reward
    oracle_actions, oracle_rewards = compute_oracle_matrix(rewards, costs)
    for a in range(K):
        assert np.all(rewards[:, a] <= oracle_rewards + 1e-12)

    # Uniform random policy
    class UniformPolicy:
        def __init__(self, n_actions: int, seed: int = 0):
            self.rng = np.random.default_rng(seed)
            self.n_actions = n_actions

        def act(self, context: np.ndarray) -> int:
            return int(self.rng.integers(0, self.n_actions))

        def update(self, context: np.ndarray, action: int, reward: float) -> None:
            pass

    policy = UniformPolicy(n_actions=K, seed=42)
    result = run_episode(policy, stream, update=True)

    # Regret at t = 1000 and t = 2000
    regret_1000 = result.cumulative_regrets[999] / 1000.0
    regret_2000 = result.cumulative_regrets[1999] / 2000.0

    # Differ by less than 10% relative: |r2 - r1| / r1 < 0.10
    rel_diff = abs(regret_2000 - regret_1000) / regret_1000
    assert rel_diff < 0.10, f"Relative difference {rel_diff:.4f} >= 0.10 (regret@1000={regret_1000:.4f}, regret@2000={regret_2000:.4f})"


def test_check_iii_synthetic_linear_bandit():
    """Check (iii): synthetic linear bandit, d=10, 4 arms, T=5000, noise N(0, 0.1^2), one seed.

    LinUCB's regret(T)/T is below regret(T/2)/(T/2).
    Uniform random's two ratios differ by less than 10% relative.
    Discounted LinUCB runs on the same draws and returns actions in {0,1,2,3}.
    """
    import numpy as np

    from src.policies import DiscountedLinUCB, LinUCBPolicy, UniformRandomPolicy
    from src.sim import QueryStream, run_episode

    rng = np.random.default_rng(0)
    d = 10
    K = 4
    T = 5000

    # Synthetic linear bandit true parameters
    theta_star = rng.normal(scale=1.0, size=(K, d))

    contexts = rng.normal(size=(T, d))
    norms = np.linalg.norm(contexts, axis=1, keepdims=True)
    contexts = contexts / np.maximum(norms, 1e-6)

    # Expected reward: theta_star @ x
    expected_rewards = contexts @ theta_star.T  # (T, K)
    noise = rng.normal(loc=0.0, scale=0.1, size=(T, K))
    noisy_rewards = expected_rewards + noise

    # Cost is 0 so reward equals quality
    costs = np.zeros((T, K))
    stream = QueryStream(contexts=contexts, rewards=noisy_rewards, costs=costs, qualities=noisy_rewards)

    # LinUCB
    linucb = LinUCBPolicy(dim=d, n_actions=K, alpha=1.0, seed=0)
    result_ucb = run_episode(linucb, stream, update=True)

    ratio_half_ucb = result_ucb.cumulative_regrets[T // 2 - 1] / (T / 2)
    ratio_full_ucb = result_ucb.cumulative_regrets[T - 1] / float(T)
    assert ratio_full_ucb < ratio_half_ucb, f"LinUCB average regret did not decrease: {ratio_full_ucb} >= {ratio_half_ucb}"

    # Uniform random
    uniform = UniformRandomPolicy(n_actions=K, random_state=0)
    result_uni = run_episode(uniform, stream, update=True)
    ratio_half_uni = result_uni.cumulative_regrets[T // 2 - 1] / (T / 2)
    ratio_full_uni = result_uni.cumulative_regrets[T - 1] / float(T)
    rel_diff_uni = abs(ratio_full_uni - ratio_half_uni) / ratio_half_uni
    assert rel_diff_uni < 0.10, f"Uniform random ratios differ by {rel_diff_uni:.4f} >= 0.10"

    # Discounted LinUCB
    dlinucb = DiscountedLinUCB(dim=d, n_actions=K, gamma=0.99, alpha=1.0, ridge=1.0, random_state=0)
    result_ducb = run_episode(dlinucb, stream, update=True)
    assert set(result_ducb.actions).issubset({0, 1, 2, 3})
    assert len(result_ducb.actions) == T


def test_all_policies_interface():
    """Verify all policy classes implement act, update, and propensity."""
    import numpy as np

    from src.policies import (
        AlwaysCheapPolicy,
        AlwaysFrontierPolicy,
        CascadePolicy,
        DiscountedLinUCB,
        LinEpsilonGreedyPolicy,
        LinTSPolicy,
        LinUCBPolicy,
        SupervisedRidgePolicy,
        UniformRandomPolicy,
    )

    d = 4
    K = 4
    x = np.ones(d)

    cheap = AlwaysCheapPolicy(cheap_arm=0, n_actions=K)
    frontier = AlwaysFrontierPolicy(frontier_arm=3, n_actions=K)
    uni = UniformRandomPolicy(n_actions=K, random_state=0)
    ucb = LinUCBPolicy(dim=d, n_actions=K, alpha=1.0, seed=0)
    ts = LinTSPolicy(dim=d, n_actions=K, seed=0)
    eps = LinEpsilonGreedyPolicy(dim=d, n_actions=K, epsilon=0.1, seed=0)
    ducb = DiscountedLinUCB(dim=d, n_actions=K, gamma=0.99, alpha=1.0, ridge=1.0, random_state=0)

    # Supervised and Cascade
    X_train = np.random.default_rng(0).normal(size=(20, d))
    q_train = np.random.default_rng(0).uniform(size=(20, K))
    costs = [0.1, 0.3, 0.6, 1.0]
    sup = SupervisedRidgePolicy.fit(X_train, q_train, costs, ridge_alpha=1.0, lambda_weight=1.0)
    cascade = CascadePolicy(sup, tau=0.5)

    policies = [cheap, frontier, uni, ucb, ts, eps, ducb, sup, cascade]
    for p in policies:
        a = p.act(x)
        assert a in range(K)
        p.update(x, a, 0.5)
        prop = p.propensity(x, a)
        assert 0.0 <= prop <= 1.0
