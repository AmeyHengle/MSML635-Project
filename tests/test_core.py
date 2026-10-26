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


def test_reward_cost_clip_and_oracle_ties():
    """Check (i): hand-written rewards, cost clipping, the held-out column, and tie-breaks."""
    import numpy as np

    from src.sim import Stream, arm_rewards, normalized_cost, oracle_action, run_episode, train_cost_max

    quality = np.array([1.0, 0.0])
    cost = np.array([0.25, 1.0])
    rewards = arm_rewards(quality, cost, lam=1.0, train_max=1.0)
    assert np.allclose(rewards, [0.75, -1.0])

    policy = _FixedArm(1)
    result = run_episode(
        policy,
        Stream(
            context=np.zeros((1, 2)),
            quality=quality.reshape(1, -1),
            cost=cost.reshape(1, -1),
            lam=1.0,
            train_max=1.0,
        ),
        update=False,
    )
    assert result.oracle_actions[0] == 0
    assert result.actions[0] == 1
    assert result.gaps[0] == 1.75
    assert result.regret == 1.75
    assert policy.updates == []

    assert float(normalized_cost(np.array([1.5]), 1.0)[0]) == 1.0
    four = np.array([[0.2, 0.4, 0.6, 1.0], [0.3, 0.5, 0.7, 0.9]])
    five = np.concatenate([four, np.full((2, 1), 50.0)], axis=1)
    assert train_cost_max(four) == 1.0
    assert train_cost_max(five) == train_cost_max(four)

    # A train maximum of 0 contributes no cost penalty.
    assert np.all(normalized_cost(np.array([3.0, 4.0]), 0.0) == 0.0)
    # pivot_wide already clipped quality, so a value outside [0, 1] is left as written.
    assert arm_rewards(np.array([1.5]), np.array([0.0]), 1.0, 1.0)[0] == 1.5

    tied = arm_rewards(np.array([0.8, 0.8]), np.array([1.5, 2.0]), lam=1.0, train_max=1.0)
    assert np.allclose(tied, [-0.2, -0.2])
    assert oracle_action(np.array([0.8, 0.8]), np.array([1.5, 2.0]), 1.0, 1.0) == 0
    assert oracle_action(np.array([0.8, 0.8]), np.array([2.0, 2.0]), 1.0, 1.0) == 0


def test_uniform_regret_rate_is_stable():
    """Check (ii): the oracle dominates, and round-robin regret(t) / t settles down."""
    import numpy as np

    from src.sim import Stream, arm_rewards, oracle_action, regret, run_episode

    rng = np.random.default_rng(0)
    steps, n_arms = 2000, 4
    quality = rng.uniform(0.0, 1.0, size=(steps, n_arms))
    cost = rng.uniform(0.1, 1.0, size=(steps, n_arms))
    lam = 1.0
    train_max = float(cost.max())
    matrix = arm_rewards(quality, cost, lam, train_max)
    for t in range(steps):
        best = oracle_action(quality[t], cost[t], lam, train_max)
        assert matrix[t, best] == matrix[t].max()
        assert np.all(matrix[t] <= matrix[t, best])

    result = run_episode(
        _RoundRobin(n_arms),
        Stream(context=np.zeros((steps, 1)), quality=quality, cost=cost, lam=lam, train_max=train_max),
        update=False,
    )
    assert np.array_equal(result.actions, np.arange(steps) % n_arms)
    rate_1000 = regret(result.gaps[:1000]) / 1000
    rate_2000 = regret(result.gaps[:2000]) / 2000
    assert abs(rate_2000 - rate_1000) / abs(rate_1000) < 0.10


def test_episode_updates_only_when_asked():
    """The policy sees the context and the chosen reward, and only when update is on."""
    import numpy as np

    from src.sim import Stream, arm_rewards, run_episode

    quality = np.array([[1.0, 0.0], [0.4, 0.9], [0.2, 0.2]])
    cost = np.array([[0.25, 1.0], [0.0, 0.5], [1.5, 2.0]])
    context = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    stream = Stream(context=context, quality=quality, cost=cost, lam=1.0, train_max=1.0)
    actions = [1, 0, 1]
    learning = _FixedArm(actions)
    result = run_episode(learning, stream, update=True)
    frozen = _FixedArm(actions)
    run_episode(frozen, stream, update=False)

    assert frozen.updates == []
    assert len(learning.updates) == len(actions)
    expected = arm_rewards(quality, cost, stream.lam, stream.train_max)
    for t, (seen, action, reward) in enumerate(learning.updates):
        assert np.array_equal(seen, context[t])
        assert action == actions[t]
        assert type(reward) is float
        assert reward == float(expected[t, actions[t]])
        assert result.rewards[t] == reward


class _FixedArm:
    """Returns the given arm, or the next arm in a list, and records updates."""

    def __init__(self, arm):
        self.arm = arm
        self.t = 0
        self.updates = []

    def act(self, context):
        if isinstance(self.arm, list):
            choice = self.arm[self.t]
        else:
            choice = self.arm
        self.t += 1
        return choice

    def update(self, context, action, reward):
        self.updates.append((context, action, reward))


class _RoundRobin:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0

    def act(self, context):
        choice = self.t % self.n_arms
        self.t += 1
        return choice

    def update(self, context, action, reward):
        raise AssertionError("update should not be called")
