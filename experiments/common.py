"""Shared experiment setup: data, features, splits, policy construction."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from router_bandit.baselines import (
    OraclePolicy,
    SupervisedRouter,
    ZeroRouter,
)
from router_bandit.data import (
    TfidfSvdBackend,
    build_features,
    build_stream,
    load,
    make_splits,
)
from router_bandit.data import registry as R
from router_bandit.envs import BanditEnv, ShiftSchedule
from router_bandit.policies import (
    DiscountedLinUCB,
    EpsilonGreedy,
    FixedArm,
    LinTS,
    LinUCB,
    PilotStyle,
    UniformRandom,
)

LAMBDA_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
LAMBDA_REF = 10.0
SEEDS = [0, 1, 2, 3, 4]
DIM = 32

# --------------------------------------------------------------------------- #
# Data source and results location
#
# The source is explicit. "full" (default) is the 36k-query RouterBench release
# and fails loudly if the file is missing. "fixture" is the 5k subsample for
# quick tests: RB_SOURCE=fixture uv run python experiments/run_all.py
# Each source writes to its own folder, so the two can never be mixed up.
# --------------------------------------------------------------------------- #
SOURCE = os.environ.get("RB_SOURCE", "full").strip().lower()
if SOURCE not in {"full", "fixture"}:
    raise SystemExit(f"RB_SOURCE must be 'full' or 'fixture', got {SOURCE!r}")

RESULTS = REPO / "results" / SOURCE
FIGDIR = RESULTS / "figures"
CELLS = RESULTS / "cells"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

MIN_FULL_QUERIES = 30_000  # the full release has 36,497 rows


def data_path() -> pathlib.Path:
    if SOURCE == "fixture":
        fixture = REPO / "data" / "fixtures" / "rb0_sample5k.parquet"
        if not fixture.exists():
            raise FileNotFoundError(f"Fixture not found: {fixture}")
        return fixture
    full = sorted(REPO.glob("data/raw/**/routerbench_0shot.pkl"))
    if not full:
        raise FileNotFoundError(
            "Full RouterBench file not found (looked for "
            "data/raw/**/routerbench_0shot.pkl). Download it first, or set "
            "RB_SOURCE=fixture for a quick 5k-subsample test run."
        )
    return full[0]


@dataclass
class Setup:
    data: object
    splits: object
    features: np.ndarray
    scale_note: str
    path: pathlib.Path

    @property
    def n_arms(self) -> int:
        return self.data.n_arms


def build_setup(seed: int = 0, dim: int = DIM) -> Setup:
    path = data_path()
    note = "full 0-shot release" if SOURCE == "full" else "5k subsample"
    data = load(path)
    if SOURCE == "full" and data.n_queries < MIN_FULL_QUERIES:
        raise SystemExit(
            f"RB_SOURCE=full but {path.name} has only {data.n_queries:,} usable "
            f"queries (expected ~36k). Wrong file?"
        )
    splits = make_splits(data.meta["family"], seed=seed)
    texts = data.meta[R.PROMPT_COL].astype(str).tolist()
    features, _ = build_features(
        texts, splits.calibration, TfidfSvdBackend(dim, min_df=2, seed=seed)
    )
    return Setup(data=data, splits=splits, features=features, scale_note=note, path=path)


# --------------------------------------------------------------------------- #
# Run metadata and resumable cells
# --------------------------------------------------------------------------- #
def check_or_write_meta(setup: Setup, *, force: bool = False) -> dict:
    """Record what produced this folder; refuse to mix incompatible runs.

    Cached cells are only valid for the configuration that produced them. If the
    data size, seeds, lambda grid, feature dimension, or the recorded metric
    columns changed, reusing them would silently mix results, so we stop and
    say how to proceed.
    """
    current = {
        "source": SOURCE,
        "data_file": str(setup.path.relative_to(REPO)),
        "n_queries": int(setup.data.n_queries),
        "n_arms": int(setup.n_arms),
        "n_stream": len(setup.splits.stream),
        "seeds": SEEDS,
        "lambda_grid": LAMBDA_GRID,
        "lambda_ref": LAMBDA_REF,
        "feature_dim": DIM,
        # Bump when a cell gains columns or policies. Old cells lack the cascade,
        # the PILOT-style router, the lambda-matched Zero Router, and the hull sweep.
        "metrics": "heldout-ci,recovery-v2,apgr,select-latency,shift-calibration,baselines-v1",
    }
    path = RESULTS / "meta.json"
    if path.exists() and not force:
        old = json.loads(path.read_text())
        diffs = {k: (old.get(k), v) for k, v in current.items() if old.get(k) != v}
        if diffs:
            raise SystemExit(
                f"Config changed since cached cells were written: {diffs}\n"
                f"Delete {CELLS} (or pass --force) to recompute."
            )
        return old
    current["created_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(current, indent=2))
    return current


def cached_cell(
    rq: str, key: str, compute: Callable[[], pd.DataFrame], *, force: bool = False
) -> pd.DataFrame:
    """Return the cell's rows from disk if present, else compute and save them.

    Each (experiment, setting, seed) cell is its own CSV, written atomically, so
    a crash or Ctrl-C loses at most the cell in progress and a rerun skips
    everything already finished.
    """
    path = CELLS / rq / f"{key}.csv"
    if path.exists() and not force:
        print(f"  [{rq}] {key}: cached, skipped", flush=True)
        return pd.read_csv(path)
    start = time.time()
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"  [{rq}] {key}: done in {time.time() - start:.0f}s", flush=True)
    return df


def make_env(setup: Setup, lam: float, seed: int, schedule: ShiftSchedule | None = None):
    """Stream + environment for one (lambda, seed) cell."""
    order_rng = np.random.default_rng(1000 + seed)
    schedule = schedule or ShiftSchedule()
    order = schedule.build_order(setup.data.meta["family"], setup.splits.stream, order_rng)
    stream = build_stream(setup.data, setup.features, order, shuffle=False)
    return BanditEnv(stream, lam=lam, schedule=schedule, seed=seed), stream


def make_heldout_env(setup: Setup, lam: float, seed: int):
    """Stationary replay of the held-out split.

    The split is never used for learning. Shuffling uses a different seed from
    the stream so the frozen evaluation is not tied to the training order.
    """
    stream = build_stream(
        setup.data, setup.features, setup.splits.heldout, seed=2000 + seed, shuffle=True
    )
    return BanditEnv(stream, lam=lam, seed=seed), stream


def fit_supervised(setup: Setup, stream, lam: float, model: str = "ridge") -> SupervisedRouter:
    X = stream.feature_matrix()[setup.splits.calibration]
    rewards = setup.data.rewards(lam)[setup.splits.calibration]
    return SupervisedRouter(model=model).fit(X, rewards)


def calibration_stats(setup: Setup) -> tuple[np.ndarray, np.ndarray]:
    idx = setup.splits.calibration
    return setup.data.cost_norm[idx].mean(0), setup.data.quality[idx].mean(0)


def oracle_mean_cost(setup: Setup, index: np.ndarray, lam: float) -> float:
    """Mean normalised cost of the per-query oracle on ``index``.

    One scalar, fixed before the run. The lambda-matched Zero Router and the
    PILOT-style pacer both spend toward the oracle's average cost on these
    queries.
    """
    index = np.asarray(index, dtype=np.int64)
    if index.size == 0:
        raise ValueError("oracle mean cost needs at least one query")
    actions = setup.data.oracle(float(lam))[index]
    return float(setup.data.cost_norm[index, actions].mean())


def is_frontier(frame: pd.DataFrame) -> pd.Series:
    """True on Zero Router hull-sweep rows.

    Those rows trace the mixture curve. The comparison tables keep the single
    lambda-matched Zero Router, whose ``frontier`` flag is false.
    """
    if "frontier" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["frontier"].map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"})


def _pilot(setup: Setup, stream, lam: float, seed: int, mean_cost: np.ndarray, pace: float):
    """LinUCB with a calibration prior and the oracle's mean spend as its pace."""
    k, d = setup.n_arms, stream.dim
    policy = PilotStyle(k, d, alpha=0.3, seed=seed, arm_costs=mean_cost, pace=pace)
    calibration = setup.splits.calibration
    policy.set_prior(
        stream.feature_matrix()[calibration],
        setup.data.rewards(lam)[calibration],
    )
    return policy


def build_policies(setup: Setup, stream, env, lam: float, seed: int, *, which: str = "all"):
    """Instantiate the policy set for a run."""
    k, d = setup.n_arms, stream.dim
    mean_cost, mean_quality = calibration_stats(setup)
    pace = oracle_mean_cost(setup, stream.order, lam)
    cheapest = int(np.argmin(setup.data.cost.mean(0)))
    priciest = int(np.argmax(setup.data.cost.mean(0)))

    bandits = [
        EpsilonGreedy(k, d, epsilon=0.1, seed=seed),
        LinUCB(k, d, alpha=0.3, seed=seed),
        _pilot(setup, stream, lam, seed, mean_cost, pace),
        LinTS(k, d, v=0.2, seed=seed),
        DiscountedLinUCB(k, d, alpha=0.3, gamma=0.9995, seed=seed),
    ]
    references = [
        OraclePolicy(env.truth),
        FixedArm(cheapest, name="always-cheapest"),
        FixedArm(priciest, name="always-premium"),
        UniformRandom(k, seed=seed),
        # Target the oracle's mean spend at this lambda so the row sits at the
        # same cost the other policies are judged against. The hull sweep in
        # RQ1 traces the rest of the curve.
        ZeroRouter(mean_cost, mean_quality, pace, seed=seed, name="ZeroRouter"),
        fit_supervised(setup, stream, lam),
    ]
    if which == "bandits":
        return bandits
    if which == "references":
        return references
    return references + bandits
