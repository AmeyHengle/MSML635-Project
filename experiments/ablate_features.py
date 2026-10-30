"""Fixture ablation of TF-IDF against MiniLM, then freeze the winner.

Scores the RQ1 policies whose decision depends on the query vector: the
supervised router and the contextual bandits. Oracle, fixed arms, random,
Zero Router, and the cascade ignore that vector, so they cannot change which
representation wins and they are not replayed here.

The selection score is the mean held-out reward at lambda 10, with equal
weight on each policy and each study seed. The feature reducer is fit once,
on the seed-0 calibration split, which is how ``run_all`` featurises a source.
Ties break toward the smaller dimension, then TF-IDF.

    uv run python experiments/ablate_features.py
    uv run python experiments/ablate_features.py --force

The runner always uses the 5k fixture. It writes
``results/fixture/feature_ablation.csv``, a per-policy summary, the config
scores, ``shift_calibration.json``, and ``configs/rq1.yaml``, ``rq2.yaml``,
and ``rq3.yaml``.
"""

from __future__ import annotations

import json
import os
import time

os.environ["RB_SOURCE"] = "fixture"

import common
import pandas as pd
from common import (
    EMBED_CACHE,
    FEATURE_MODEL,
    LAMBDA_GRID,
    LAMBDA_REF,
    REPO,
    RESULTS,
    SEEDS,
    build_policies,
    build_setup,
    fit_supervised,
    make_env,
    make_heldout_env,
)

from router_bandit.data import MODELS
from router_bandit.data.features import SentenceTransformerBackend
from router_bandit.evaluation import run_policy

BACKENDS = ("tfidf", "minilm")
DIMS = (32, 64, 128)
CELLS = RESULTS / "cells" / "ablation"
DETAIL = RESULTS / "feature_ablation.csv"
SUMMARY = RESULTS / "feature_ablation_summary.csv"
SCORES = RESULTS / "feature_ablation_scores.csv"
CALIBRATION = RESULTS / "shift_calibration.json"


def choose_winner(frame: pd.DataFrame) -> pd.Series:
    """Backend and dimension with the highest mean held-out reward.

    The mean gives every policy and every seed the same weight. An exact tie
    goes to the smaller dimension, and then to TF-IDF, which does not need
    the embedding extra for the full study.
    """
    required = {"backend", "dim", "heldout_reward"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ablation frame is missing {sorted(missing)}")
    if frame.empty:
        raise ValueError("ablation frame is empty")
    scores = frame.groupby(["backend", "dim"], as_index=False)["heldout_reward"].mean()
    scores["backend_rank"] = scores["backend"].map({"tfidf": 0, "minilm": 1}).fillna(2)
    scores = scores.sort_values(
        ["heldout_reward", "dim", "backend_rank"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return scores.iloc[0]


def policy_winners(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per policy: the backend and dimension with its best mean reward."""
    ranked = summary.copy()
    ranked["backend_rank"] = ranked["backend"].map({"tfidf": 0, "minilm": 1}).fillna(2)
    ranked = ranked.sort_values(
        ["policy", "heldout_reward", "dim", "backend_rank"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return ranked.groupby("policy", as_index=False).head(1).reset_index(drop=True)


def _feature_policies(setup, stream, env, lam: float, seed: int):
    """Supervised router plus the contextual bandits RQ1 actually trains."""
    policies = build_policies(setup, stream, env, lam, seed, which="bandits")
    policies.append(fit_supervised(setup, stream, lam))
    return policies


def _score_seed(setup, backend: str, dim: int, seed: int) -> pd.DataFrame:
    env, stream = make_env(setup, LAMBDA_REF, seed)
    held_env, held_stream = make_heldout_env(setup, LAMBDA_REF, seed)
    rows = []
    for policy in _feature_policies(setup, stream, env, LAMBDA_REF, seed):
        run_policy(env, policy)
        held = run_policy(held_env, policy, reset=False, update=False)
        dollar = setup.data.cost[held_stream.order, held.actions]
        rows.append({
            "backend": backend,
            "dim": int(dim),
            "seed": int(seed),
            "policy": held.policy,
            "heldout_reward": held.mean_reward,
            "heldout_quality": held.mean_quality,
            "heldout_cost": held.mean_cost,
            "heldout_dollar_cost": float(dollar.mean()),
        })
    return pd.DataFrame(rows)


def _cell_path(backend: str, dim: int, seed: int):
    return CELLS / f"{backend}_d{dim}_seed{seed}.csv"


def _load_or_run(setup, backend: str, dim: int, seed: int, *, force: bool) -> pd.DataFrame:
    path = _cell_path(backend, dim, seed)
    if path.exists() and not force:
        print(f"  [{backend} d={dim} seed={seed}] cached, skipped", flush=True)
        return pd.read_csv(path)
    started = time.perf_counter()
    frame = _score_seed(setup, backend, dim, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    print(
        f"  [{backend} d={dim} seed={seed}] done in {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    return frame


def _summarise(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    counts = detail.groupby(["backend", "dim"]).size()
    if counts.nunique() != 1:
        raise SystemExit(f"ablation cells are unbalanced: {counts.to_dict()}")
    policies = detail.groupby(["backend", "dim"])["policy"].nunique()
    if int(policies.min()) < 1 or policies.nunique() != 1:
        raise SystemExit(f"ablation policies differ by setting: {policies.to_dict()}")
    summary = (
        detail.groupby(["backend", "dim", "policy"], as_index=False)
        .agg(
            heldout_reward=("heldout_reward", "mean"),
            heldout_reward_std=("heldout_reward", "std"),
            heldout_quality=("heldout_quality", "mean"),
            heldout_dollar_cost=("heldout_dollar_cost", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    scores = (
        detail.groupby(["backend", "dim"], as_index=False)
        .agg(
            heldout_reward=("heldout_reward", "mean"),
            heldout_quality=("heldout_quality", "mean"),
            heldout_dollar_cost=("heldout_dollar_cost", "mean"),
            n_rows=("heldout_reward", "size"),
        )
    )
    winner = choose_winner(detail)
    scores["winner"] = (
        (scores["backend"] == winner["backend"]) & (scores["dim"] == winner["dim"])
    )
    return summary, scores, winner


def _yaml_float(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _agreement_note(summary: pd.DataFrame, backend: str, dim: int) -> str:
    disagree = []
    for row in policy_winners(summary).itertuples(index=False):
        if row.backend == backend and int(row.dim) == int(dim):
            continue
        disagree.append(f"# {row.policy} prefers {row.backend} at d={int(row.dim)}.")
    if not disagree:
        return "# Every feature-using RQ1 policy prefers this backend and dimension.\n"
    return (
        "# These features win on mean held-out reward. Other per-policy winners:\n"
        + "\n".join(disagree)
        + "\n"
    )


def write_configs(
    backend: str,
    dim: int,
    score: float,
    summary: pd.DataFrame,
    arm: int,
    price,
    degrade,
) -> None:
    """Rewrite the three study configs so they share the frozen winner and shifts."""
    model = SentenceTransformerBackend.DEFAULT_MODEL
    if model != FEATURE_MODEL:
        raise SystemExit(
            f"configs name {FEATURE_MODEL}, but the encoder default is {model}"
        )
    seeds = "[" + ", ".join(str(int(seed)) for seed in SEEDS) + "]"
    grid = "[" + ", ".join(_yaml_float(value) for value in LAMBDA_GRID) + "]"
    features = (
        "features:\n"
        f"  backend: {backend}\n"
        f"  dim: {int(dim)}\n"
        f'  model: "{model}"\n'
    )
    shared_head = (
        "# common.py requires seeds, lambda_ref, and features to match the other "
        "research-question configs.\n\n"
        f"seeds: {seeds}\n"
        f"lambda_ref: {_yaml_float(LAMBDA_REF)}\n"
    )
    note = _agreement_note(summary, backend, dim)
    from router_bandit.evaluation.shift_calibration import FLIP_TARGET

    shifts = (
        "shifts:\n"
        "  # Fixture calibration on the seed-0 stationary stream. shocked_arm is the\n"
        "  # arm the supervised router selects most often. The multiplier and severity\n"
        "  # are the smallest shocks that change its arm on at least flip_target of\n"
        "  # post-shift queries, or the strongest grid point when none of them do.\n"
        "  # S1 and S3 stay as coded. RQ2 applies these strengths and recomputes the\n"
        "  # flip rate on the source it is running.\n"
        f"  shocked_arm: {int(arm)}\n"
        f'  shocked_model: "{MODELS[int(arm)]}"\n'
        f"  s2_multiplier: {_yaml_float(price.strength)}\n"
        f"  s2_flip_rate: {_yaml_float(price.flip_rate)}\n"
        f"  s2_reached_target: {'true' if price.reached_target else 'false'}\n"
        f"  s4_severity: {_yaml_float(degrade.strength)}\n"
        f"  s4_flip_rate: {_yaml_float(degrade.flip_rate)}\n"
        f"  s4_reached_target: {'true' if degrade.reached_target else 'false'}\n"
        f"  flip_target: {_yaml_float(FLIP_TARGET)}\n"
    )
    rq1 = (
        "# Frozen by experiments/ablate_features.py on the 5k fixture.\n"
        f"# Winner: {backend} at d={int(dim)}, mean held-out reward {_yaml_float(score)}\n"
        "# over the feature-using RQ1 policies (supervised router and the contextual\n"
        f"# bandits) at lambda {_yaml_float(LAMBDA_REF)}. Table: "
        "results/fixture/feature_ablation_scores.csv\n"
        f"{note}"
        f"{shared_head}"
        f"lambda_grid: {grid}\n"
        f"{features}"
    )
    rq2 = (
        "# Frozen by experiments/ablate_features.py on the 5k fixture.\n"
        "# S1 and S3 are not stored here.\n"
        f"{note}"
        f"{shared_head}"
        f"{features}"
        f"{shifts}"
    )
    rq3 = (
        "# Frozen by experiments/ablate_features.py on the 5k fixture.\n"
        f"{note}"
        f"{shared_head}"
        f"{features}"
    )
    for name, text in (("rq1.yaml", rq1), ("rq2.yaml", rq2), ("rq3.yaml", rq3)):
        path = REPO / "configs" / name
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}", flush=True)


def _write_calibration(backend: str, dim: int, score: float, t0: int, arm: int, price, degrade) -> None:
    payload = {
        "source": "fixture",
        "lambda": LAMBDA_REF,
        "seed": 0,
        "feature_backend": backend,
        "feature_dim": int(dim),
        "selection_heldout_reward": float(score),
        "t0": int(t0),
        "shocked_arm": int(arm),
        "shocked_model": MODELS[int(arm)],
        "s2_multiplier": float(price.strength),
        "s2_flip_rate": float(price.flip_rate),
        "s2_reached_target": bool(price.reached_target),
        "s4_severity": float(degrade.strength),
        "s4_flip_rate": float(degrade.flip_rate),
        "s4_reached_target": bool(degrade.reached_target),
        "flip_target": 0.25,
    }
    from router_bandit.evaluation.shift_calibration import FLIP_TARGET

    payload["flip_target"] = float(FLIP_TARGET)
    CALIBRATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {CALIBRATION}", flush=True)


def run(force: bool = False) -> pd.Series:
    if common.SOURCE != "fixture":
        raise SystemExit("feature ablation is fixture-only")
    if SentenceTransformerBackend.DEFAULT_MODEL != FEATURE_MODEL:
        raise SystemExit(
            f"configs name {FEATURE_MODEL}, but the encoder default is "
            f"{SentenceTransformerBackend.DEFAULT_MODEL}"
        )
    frames = []
    for backend in BACKENDS:
        if backend == "minilm":
            print(f"MiniLM embeddings cache: {EMBED_CACHE}", flush=True)
        for dim in DIMS:
            print(f"features {backend} d={dim}", flush=True)
            setup = build_setup(seed=0, dim=dim, backend=backend)
            print(f"  matrix {setup.features.shape}", flush=True)
            for seed in SEEDS:
                frames.append(_load_or_run(setup, backend, dim, seed, force=force))
    detail = pd.concat(frames, ignore_index=True)
    summary, scores, winner = _summarise(detail)
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL, index=False)
    summary.to_csv(SUMMARY, index=False)
    scores.to_csv(SCORES, index=False)
    print(f"wrote {DETAIL}", flush=True)
    print(scores.sort_values("heldout_reward", ascending=False).to_string(index=False), flush=True)

    backend = str(winner["backend"])
    dim = int(winner["dim"])
    score = float(winner["heldout_reward"])
    print(f"winner: {backend} d={dim} heldout_reward={score:.6f}", flush=True)

    from run_all import calibrate_router_shocks

    setup = build_setup(seed=0, dim=dim, backend=backend)
    probe_env, probe_stream = make_env(setup, LAMBDA_REF, 0)
    t0 = probe_env.n_rounds // 2
    arm, price, degrade = calibrate_router_shocks(setup, probe_stream, t0)
    print(
        f"calibration t0={t0}: arm {arm} ({MODELS[arm]}), "
        f"S2 x{price.strength:g} flip={price.flip_rate:.3f} "
        f"reached={price.reached_target}, "
        f"S4 severity {degrade.strength:g} flip={degrade.flip_rate:.3f} "
        f"reached={degrade.reached_target}",
        flush=True,
    )
    _write_calibration(backend, dim, score, t0, arm, price, degrade)
    write_configs(backend, dim, score, summary, arm, price, degrade)
    return winner


def main() -> None:
    import sys

    force = "--force" in sys.argv[1:]
    unknown = [arg for arg in sys.argv[1:] if arg != "--force"]
    if unknown:
        raise SystemExit(f"Unknown argument(s): {unknown}")
    run(force=force)


if __name__ == "__main__":
    main()
