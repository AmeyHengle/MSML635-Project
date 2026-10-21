def test_linucb_import():
    from obp.policy import LinUCB

    assert LinUCB is not None


def _long(specs, queries):
    """specs maps model -> (cost, performance). queries are (sample_id, eval_name)."""
    import pandas as pd

    rows = []
    # Create a row for each combination of sample_id, eval_name, and model
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
    return pd.DataFrame(rows)


# Midpoint of cheap (1) and strong (5) is 3. claude-mid sits on it; open-mid is close.
_SPECS = {
    "open-cheap": (1.0, 0.2),
    "open-strong": (5.0, 0.8),
    "open-mid": (3.1, 0.5),
    "claude-mid": (3.0, 0.55),
    "GPT-4-test": (10.0, 0.99),
    "claude-held": (7.0, 0.9),
}


def test_filter_evals_keeps_routerbench_names():
    import pandas as pd

    from src.data import filter_evals

    frame = pd.DataFrame(
        {
            "eval_name": [
                "mmlu-anatomy",
                "grade-school-math",
                "mbpp",
                "mtbench",
                "mtbench-math",
                "hellaswag",
                "arc-challenge",
                "GSM-8K-extra",
            ]
        }
    )
    # assert that the result matches the expected set of names
    assert set(filter_evals(frame)["eval_name"]) == {
        "mmlu-anatomy",
        "grade-school-math",
        "mbpp",
        "mtbench",
        "mtbench-math",
        "GSM-8K-extra",
    }


def test_select_arms_prefers_a_close_open_model():
    """Test that select_arms chooses the correct arms and held-out arm."""
    from src.data import choose_mix_eval, select_arms

    specs = dict(_SPECS)    # Create a copy of the _SPECS dictionary to modify for this test
    specs["claude-sparse"] = (4.0, 0.98)    # Add a new model to the specs for testing
    queries = [("s1", "mmlu"), ("s2", "grade-school-math"), ("s3", "mtbench")]  
    frame = _long(specs, queries)   # Create a long-format DataFrame using the modified specs and queries
    frame = frame[~((frame.sample_id == "s3") & (frame.model_name == "claude-sparse"))]
    selection = select_arms(frame)  # Select the arms and held-out arm based on the DataFrame
    
    # Assert that the selected arms and held-out arm match the expected values
    assert selection["arms"] == ["open-cheap", "open-mid", "open-strong", "GPT-4-test"]
    assert selection["held_out_arm"] == "claude-held"
    assert choose_mix_eval(frame, "open-cheap") == "mtbench"


def test_mix_eval_falls_back_to_the_weakest_cheap_eval():
    """Test that choose_mix_eval selects the weakest cheap-model evaluation."""
    import pandas as pd

    from src.data import choose_mix_eval

    frame = pd.DataFrame(
        [
            {"eval_name": "mmlu", "model_name": "open-cheap", "performance": 0.9, "cost": 1},
            {"eval_name": "mbpp", "model_name": "open-cheap", "performance": 0.1, "cost": 1},
        ]
    )
    # Assert that the mix_eval selected is "mbpp", which is the weakest cheap-model evaluation
    assert choose_mix_eval(frame, "open-cheap") == "mbpp"


def test_bad_cost_order_does_not_write_the_config(tmp_path):
    """Test that lock_evals_and_arms raises a ValueError and does not write to the config file if the cost order is bad."""
    import pytest
    import yaml

    from src.data import lock_evals_and_arms, require_locked_arms

    config = tmp_path / "default.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 0",
                "arms: []                       # task 2: four model_name strings, role order",
                "arm_roles: [cheap, mid, strong_open, frontier]",
                "held_out_arm: null",
                "eval_names: []",
                "mix_eval: null                 # MT-Bench, or the weakest cheap-model eval",
                "pca_dim: 64",
                "",
            ]
        ),
        encoding="utf-8",
    )
    bad = _long(
        {
            "open-cheap": (1.0, 0.2),
            "open-strong": (3.0, 0.8),
            "claude-costly": (9.0, 0.5),
            "gpt-4-preview": (10.0, 0.99),
        },
        [("s1", "mmlu"), ("s2", "mbpp")],
    )
    # Assert that lock_evals_and_arms raises a ValueError and does not modify the config file
    original = config.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="config was not written"):
        lock_evals_and_arms(bad, config)
    assert config.read_text(encoding="utf-8") == original

    # Assert that lock_evals_and_arms writes the config file correctly with the expected values
    locked = lock_evals_and_arms(_long(_SPECS, [("s1", "mmlu"), ("s2", "mtbench"), ("s3", "hellaswag")]), config)
    text = config.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert "arm_roles: [cheap, mid, strong_open, frontier]" in text
    assert "pca_dim: 64" in text
    assert "# task 2: four model_name strings, role order" in text
    assert "# MT-Bench, or the weakest cheap-model eval" in text
    assert require_locked_arms(loaded["arms"], locked["arms"]) == [
        "open-cheap",
        "open-mid",
        "open-strong",
        "GPT-4-test",
    ]
    assert loaded["held_out_arm"] == "claude-held"
    assert loaded["mix_eval"] == "mtbench"
    assert loaded["eval_names"] == ["mmlu", "mtbench"]
    assert loaded["seed"] == 0
    with pytest.raises(ValueError, match="wide table"):
        require_locked_arms(["a", "b", "c", "d"], locked["arms"])


def test_performance_to_quality_maps_bool_and_clips(tmp_path):
    """Bool becomes 0/1, a unit interval is kept, and an outside range is clipped and noted."""
    import pandas as pd

    from src.data import performance_to_quality

    bools = performance_to_quality(pd.Series([True, False, True]))
    assert list(bools) == [1.0, 0.0, 1.0]

    unit = performance_to_quality(pd.Series([0.0, 0.4, 1.0]))
    assert list(unit) == [0.0, 0.4, 1.0]

    sim = tmp_path / "sim.py"
    sim.write_text('"""Reward, oracle, episode loop, shifts."""\n', encoding="utf-8")
    clipped = performance_to_quality(pd.Series([0.0, 1.5, None]), sim_path=sim)
    assert clipped.iloc[0] == 0.0
    assert clipped.iloc[1] == 1.0
    assert pd.isna(clipped.iloc[2])
    text = sim.read_text(encoding="utf-8")
    assert text.startswith('"""Reward, oracle, episode loop, shifts."""')
    assert "clip(performance, 0, 1)" in text

    same = performance_to_quality(pd.Series([0.2, 0.8]), sim_path=sim)
    assert list(same) == [0.2, 0.8]
    assert "clip(performance, 0, 1)" not in sim.read_text(encoding="utf-8")


def test_pivot_wide_drops_incomplete_arms_and_splits(tmp_path, capsys):
    """Pivot, drop a query missing an arm, and split 70/30 with seed 0.

    No download and no RouterBench file. Train and test ids are disjoint and
    cover every kept id.
    """
    import json

    import numpy as np
    import pandas as pd
    import pytest

    from src.data import pivot_wide, print_wide_check, save_wide_and_split, split_ids

    queries = [(f"m{i}", "mmlu") for i in range(6)]
    queries += [(f"g{i}", "grade-school-math") for i in range(4)]
    queries += [(f"b{i}", "mbpp") for i in range(4)]
    queries += [(f"t{i}", "mtbench") for i in range(4)]
    queries.append(("rare", "mmlu-anatomy"))
    queries.append(("gap", "mbpp"))
    frame = _long(_SPECS, queries)
    frame = frame[~((frame.sample_id == "gap") & (frame.model_name == "GPT-4-test"))]
    arms = ["open-cheap", "open-mid", "open-strong", "GPT-4-test"]

    partial = frame.copy()
    partial.loc[
        partial.sample_id.eq("m0") & partial.model_name.eq("claude-held"), "cost"
    ] = np.nan
    partial_wide = pivot_wide(partial, arms, "claude-held")
    assert "gap" not in set(partial_wide.sample_id)
    assert "m0" in set(partial_wide.sample_id)
    held_cost = partial_wide.loc[partial_wide.sample_id.eq("m0"), "cost_held_out"]
    assert not np.isfinite(held_cost.to_numpy(dtype=float)).all()

    wide = pivot_wide(frame, arms, "claude-held")
    assert "gap" not in set(wide.sample_id)
    assert set(wide.sample_id) == {sample_id for sample_id, _ in queries if sample_id != "gap"}
    assert list(wide.columns) == [
        "sample_id",
        "prompt",
        "eval_name",
        "quality_cheap",
        "cost_cheap",
        "quality_mid",
        "cost_mid",
        "quality_strong_open",
        "cost_strong_open",
        "quality_frontier",
        "cost_frontier",
        "quality_held_out",
        "cost_held_out",
    ]
    assert set(wide["cost_cheap"]) == {1.0}
    assert set(wide["cost_frontier"]) == {10.0}
    assert set(wide["quality_cheap"]) == {0.2}
    for role in ("cheap", "mid", "strong_open", "frontier", "held_out"):
        assert np.isfinite(wide[f"quality_{role}"].to_numpy(dtype=float)).all()
        assert np.isfinite(wide[f"cost_{role}"].to_numpy(dtype=float)).all()

    split = split_ids(wide)
    assert split_ids(wide) == split
    train, test = set(split["train"]), set(split["test"])
    assert train.isdisjoint(test)
    assert train | test == set(wide["sample_id"])
    assert len(split["train"]) + len(split["test"]) == len(wide)
    assert abs(len(split["test"]) / len(wide) - 0.3) < 0.05

    wide_path, split_path = save_wide_and_split(wide, split, tmp_path)
    loaded = pd.read_pickle(wide_path)
    assert list(loaded["sample_id"]) == list(wide["sample_id"])
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert payload == split

    print_wide_check(wide, split)
    output = capsys.readouterr().out
    assert "dropped 1 queries missing a finite value for an arm" in output
    assert "held-out values missing: 1" in output
    assert "held-out values missing: 0" in output
    assert "stratifying on family labels" in output
    assert "mmlu-anatomy" in output
    assert "5 wide-table rows:" in output
    assert "finite quality and cost for every arm: True" in output
    assert "train/test id intersection: []" in output
    with pytest.raises(ValueError, match="wide table"):
        pivot_wide(frame, arms[:3], "claude-held")


def test_offline_pivot_split_and_train_only_pca(tmp_path, monkeypatch, capsys):
    """Pivot a tiny long frame, drop a query missing an arm, split with seed 0, and fit PCA on train.

    No download and no RouterBench file. Train and test ids are disjoint, and each
    context row has width pca_dim + 1.
    """
    import sys
    import types

    import joblib
    import numpy as np
    from sklearn.decomposition import PCA

    from src.data import embed_prompts, fit_pca_contexts, pivot_wide, split_ids

    queries = [(f"m{i}", "mmlu") for i in range(10)]
    queries += [(f"g{i}", "grade-school-math") for i in range(10)]
    queries += [(f"b{i}", "mbpp") for i in range(10)]
    queries += [(f"t{i}", "mtbench") for i in range(10)]
    queries.append(("gap", "mbpp"))
    frame = _long(_SPECS, queries)
    frame = frame[~((frame.sample_id == "gap") & (frame.model_name == "open-strong"))]
    arms = ["open-cheap", "open-mid", "open-strong", "GPT-4-test"]

    wide = pivot_wide(frame, arms, "claude-held")
    assert "gap" not in set(wide["sample_id"])
    assert set(wide["sample_id"]) == {sample_id for sample_id, _ in queries if sample_id != "gap"}

    split = split_ids(wide, random_state=0)
    train_ids, test_ids = set(split["train"]), set(split["test"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(wide["sample_id"])

    pca_dim = 4
    embeddings = np.random.default_rng(0).normal(size=(len(wide), 16))
    context_train, context_test, pca = fit_pca_contexts(embeddings, wide, split, pca_dim, tmp_path)
    assert context_train.shape == (len(split["train"]), pca_dim + 1)
    assert context_test.shape == (len(split["test"]), pca_dim + 1)
    assert pca.n_samples_ == len(split["train"])
    assert pca.n_samples_ != len(wide)
    assert np.all(context_train[:, -1] == 1.0)
    assert np.all(context_test[:, -1] == 1.0)

    positions = {sample_id: index for index, sample_id in enumerate(wide["sample_id"].tolist())}
    train_x = embeddings[[positions[sample_id] for sample_id in split["train"]]]
    test_x = embeddings[[positions[sample_id] for sample_id in split["test"]]]
    reference = PCA(n_components=pca_dim, random_state=0).fit(train_x)
    assert np.allclose(pca.components_, reference.components_)
    assert np.allclose(pca.mean_, reference.mean_)
    assert np.allclose(context_train[:, :-1], reference.transform(train_x))
    assert np.allclose(context_test[:, :-1], reference.transform(test_x))

    saved = joblib.load(tmp_path / "pca.joblib")
    assert saved.n_samples_ == len(split["train"])
    assert np.load(tmp_path / "context_train.npy").shape == context_train.shape
    assert np.load(tmp_path / "context_test.npy").shape[1] == pca_dim + 1

    fake_models = types.ModuleType("sentence_transformers")

    class _MustNotEncode:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("cached embeddings must not load MiniLM")

    fake_models.SentenceTransformer = _MustNotEncode
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_models)
    cached = np.ones((len(wide), 384), dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", cached)
    loaded = embed_prompts(wide, "sentence-transformers/all-MiniLM-L6-v2", tmp_path)
    assert loaded.shape == (len(wide), 384)
    assert np.array_equal(loaded, cached)
    assert "loaded cached embeddings" in capsys.readouterr().out
