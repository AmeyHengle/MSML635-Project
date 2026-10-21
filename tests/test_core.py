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
