"""Frozen feature configs, and the ablation rule that picks them."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from router_bandit.data import MODELS

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "fixture" / "feature_ablation_scores.csv"
CALIBRATION = ROOT / "results" / "fixture" / "shift_calibration.json"


def _experiments():
    os.environ["RB_SOURCE"] = "fixture"
    exp = str(ROOT / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import common

    return importlib.reload(common)


def _ablation():
    os.environ["RB_SOURCE"] = "fixture"
    exp = str(ROOT / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import ablate_features

    return importlib.reload(ablate_features)


def _winner_flag(values: pd.Series) -> pd.Series:
    return values.map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"})


def test_normalize_backend_accepts_the_two_study_names():
    common = _experiments()
    assert common.normalize_backend("TfidfSvdBackend") == "tfidf"
    assert common.normalize_backend("SentenceTransformerBackend") == "minilm"
    with pytest.raises(ValueError):
        common.normalize_backend("bow")
    with pytest.raises(ValueError):
        common.build_setup(backend="bow")


def test_choose_winner_takes_the_higher_mean_heldout_reward():
    ablate = _ablation()
    frame = pd.DataFrame([
        {"backend": "tfidf", "dim": 32, "heldout_reward": 0.0},
        {"backend": "tfidf", "dim": 32, "heldout_reward": 0.0},
        {"backend": "minilm", "dim": 128, "heldout_reward": 1.0},
        {"backend": "minilm", "dim": 128, "heldout_reward": 0.0},
    ])
    winner = ablate.choose_winner(frame)
    assert winner["backend"] == "minilm"
    assert int(winner["dim"]) == 128
    assert winner["heldout_reward"] == pytest.approx(0.5)


def test_choose_winner_breaks_ties_toward_the_smaller_tfidf():
    ablate = _ablation()
    tied = pd.DataFrame([
        {"backend": "minilm", "dim": 128, "heldout_reward": 0.5},
        {"backend": "tfidf", "dim": 64, "heldout_reward": 0.5},
        {"backend": "minilm", "dim": 32, "heldout_reward": 0.5},
        {"backend": "tfidf", "dim": 32, "heldout_reward": 0.4},
    ])
    smaller = ablate.choose_winner(tied)
    assert smaller["backend"] == "minilm"
    assert int(smaller["dim"]) == 32

    same_dim = pd.DataFrame([
        {"backend": "minilm", "dim": 64, "heldout_reward": 0.2},
        {"backend": "tfidf", "dim": 64, "heldout_reward": 0.2},
    ])
    tfidf = ablate.choose_winner(same_dim)
    assert tfidf["backend"] == "tfidf"
    assert int(tfidf["dim"]) == 64

    with pytest.raises(ValueError):
        ablate.choose_winner(pd.DataFrame())


def test_configs_agree_and_common_reads_them():
    common = _experiments()
    docs = {
        name: yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        for name in ("rq1", "rq2", "rq3")
    }
    assert common.SEEDS == [int(seed) for seed in docs["rq1"]["seeds"]]
    assert common.LAMBDA_GRID == [float(value) for value in docs["rq1"]["lambda_grid"]]
    assert common.LAMBDA_REF == pytest.approx(float(docs["rq1"]["lambda_ref"]))
    assert common.LAMBDA_REF in common.LAMBDA_GRID
    for doc in docs.values():
        assert [int(seed) for seed in doc["seeds"]] == common.SEEDS
        assert float(doc["lambda_ref"]) == pytest.approx(common.LAMBDA_REF)
        assert doc["features"]["backend"] == common.FEATURE_BACKEND
        assert int(doc["features"]["dim"]) == common.DIM
        assert doc["features"]["model"] == common.FEATURE_MODEL
    shifts = docs["rq2"]["shifts"]
    assert common.SHIFTS.shocked_arm == int(shifts["shocked_arm"])
    assert common.SHIFTS.shocked_model == shifts["shocked_model"]
    assert MODELS[common.SHIFTS.shocked_arm] == common.SHIFTS.shocked_model
    assert common.SHIFTS.s2_multiplier == pytest.approx(float(shifts["s2_multiplier"]))
    assert common.SHIFTS.s4_severity == pytest.approx(float(shifts["s4_severity"]))
    assert common.SHIFTS.s2_flip_rate == pytest.approx(float(shifts["s2_flip_rate"]))
    assert common.SHIFTS.s4_flip_rate == pytest.approx(float(shifts["s4_flip_rate"]))
    assert common.SHIFTS.s2_reached_target is bool(shifts["s2_reached_target"])
    assert common.SHIFTS.s4_reached_target is bool(shifts["s4_reached_target"])
    assert common.SHIFTS.flip_target == pytest.approx(float(shifts["flip_target"]))
    assert common.FEATURE_BACKEND in {"tfidf", "minilm"}
    assert common.DIM in {32, 64, 128}


def test_rq2_uses_the_frozen_shift_strengths():
    os.environ["RB_SOURCE"] = "fixture"
    exp = str(ROOT / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import run_all

    run_all = importlib.reload(run_all)
    source = run_all.rq2.__code__.co_names
    # The search stays available for the ablation. The experiment itself reads
    # the frozen arm and strengths, then only measures the flip rate.
    assert "frozen_router_shocks" in source
    assert "calibrate_router_shocks" not in source
    frozen = run_all.frozen_router_shocks.__code__.co_names
    assert "SHIFTS" in frozen
    assert "price_flip_rate" in frozen
    assert "degradation_flip_rate" in frozen


def test_frozen_winner_matches_the_ablation_table():
    if not SCORES.exists() or not CALIBRATION.exists():
        pytest.fail("run experiments/ablate_features.py before checking the frozen configs")
    common = _experiments()
    scores = pd.read_csv(SCORES)
    flagged = scores.loc[_winner_flag(scores["winner"])]
    assert len(flagged) == 1
    winner = flagged.iloc[0]
    assert common.FEATURE_BACKEND == winner["backend"]
    assert common.DIM == int(winner["dim"])

    recorded = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert recorded["feature_backend"] == common.FEATURE_BACKEND
    assert int(recorded["feature_dim"]) == common.DIM
    assert recorded["shocked_model"] == common.SHIFTS.shocked_model
    assert int(recorded["shocked_arm"]) == common.SHIFTS.shocked_arm
    assert recorded["s2_multiplier"] == pytest.approx(common.SHIFTS.s2_multiplier)
    assert recorded["s4_severity"] == pytest.approx(common.SHIFTS.s4_severity)
    # The configs keep six decimal places. The json keeps the unrounded rate.
    assert recorded["s2_flip_rate"] == pytest.approx(common.SHIFTS.s2_flip_rate, abs=1e-6)
    assert recorded["s4_flip_rate"] == pytest.approx(common.SHIFTS.s4_flip_rate, abs=1e-6)
    assert recorded["s2_reached_target"] is common.SHIFTS.s2_reached_target
    assert recorded["s4_reached_target"] is common.SHIFTS.s4_reached_target
    assert float(winner["heldout_reward"]) == pytest.approx(recorded["selection_heldout_reward"])
