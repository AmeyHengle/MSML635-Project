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
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from router_bandit.baselines import (
    OraclePolicy,
    SupervisedRouter,
    ZeroRouter,
)
from router_bandit.data import (
    SentenceTransformerBackend,
    TfidfSvdBackend,
    build_features,
    build_stream,
    cached_sentence_embeddings,
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

# --------------------------------------------------------------------------- #
# Frozen study settings
#
# configs/rq1.yaml, rq2.yaml, and rq3.yaml are written by the fixture feature
# ablation. Shared fields have to agree. S2/S4 strengths are the fixture
# calibration, not a search that is repeated on the full release.
# --------------------------------------------------------------------------- #
_BACKEND_ALIASES = {
    "tfidf": "tfidf",
    "tfidfsvdbackend": "tfidf",
    "minilm": "minilm",
    "sentencetransformerbackend": "minilm",
}
EMBED_CACHE = REPO / "data" / "cache"


def _read_config(name: str) -> dict:
    path = REPO / "configs" / name
    if not path.exists():
        raise SystemExit(f"Missing experiment config: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a mapping")
    return loaded


def _match(field: str, values: dict[str, object]) -> object:
    items = list(values.items())
    first_label, first = items[0]
    for label, value in items[1:]:
        if value != first:
            raise SystemExit(
                f"{field} disagrees across configs: {first_label} has {first!r}, "
                f"{label} has {value!r}"
            )
    return first


def _feature_block(doc: dict, name: str) -> tuple[str, int, str]:
    features = doc.get("features")
    if not isinstance(features, dict):
        raise SystemExit(f"{name} is missing a features: block")
    backend = _BACKEND_ALIASES.get(str(features.get("backend", "")).strip().lower())
    if backend is None:
        raise SystemExit(
            f"{name} features.backend must be 'tfidf' or 'minilm', "
            f"got {features.get('backend')!r}"
        )
    try:
        dim = int(features["dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{name} features.dim must be an integer") from exc
    if dim not in {32, 64, 128}:
        raise SystemExit(f"{name} features.dim must be 32, 64, or 128, got {dim}")
    model = features.get("model")
    if not isinstance(model, str) or not model.strip():
        raise SystemExit(f"{name} features.model must name the MiniLM checkpoint")
    return backend, dim, model.strip()


def _number_list(doc: dict, key: str, name: str, *, as_int: bool) -> list[float] | list[int]:
    raw = doc.get(key)
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{name} {key} must be a non-empty list")
    try:
        return [int(v) if as_int else float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{name} {key} must be numeric") from exc


def _as_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SystemExit(f"{label} must be true or false, got {value!r}")


def _as_float(doc: dict, key: str) -> float:
    try:
        return float(doc[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"configs/rq2.yaml shifts.{key} must be a number") from exc


@dataclass(frozen=True)
class ShiftSettings:
    """S2/S4 shock frozen from the fixture calibration.

    ``shocked_arm`` is the arm the static supervised router selects most often
    on the seed-0 stationary stream. The multiplier and severity are the
    smallest grid points that change that router's arm on at least
    ``flip_target`` of the post-shift queries, or the strongest grid point
    when none of them do. S1 and S3 are not in here: they stay as coded.
    """

    shocked_arm: int
    shocked_model: str
    s2_multiplier: float
    s2_flip_rate: float
    s2_reached_target: bool
    s4_severity: float
    s4_flip_rate: float
    s4_reached_target: bool
    flip_target: float


def _load_shifts(doc: dict) -> ShiftSettings:
    shifts = doc.get("shifts")
    if not isinstance(shifts, dict):
        raise SystemExit("configs/rq2.yaml is missing a shifts: block")
    try:
        arm = int(shifts["shocked_arm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("configs/rq2.yaml shifts.shocked_arm must be an integer") from exc
    model = shifts.get("shocked_model")
    if not isinstance(model, str) or not model.strip():
        raise SystemExit("configs/rq2.yaml shifts.shocked_model must name the shocked arm")
    if arm < 0 or arm >= len(R.MODELS) or R.MODELS[arm] != model.strip():
        raise SystemExit(
            f"shifts.shocked_arm {arm} does not match shifts.shocked_model {model!r}"
        )
    multiplier = _as_float(shifts, "s2_multiplier")
    severity = _as_float(shifts, "s4_severity")
    s2_rate = _as_float(shifts, "s2_flip_rate")
    s4_rate = _as_float(shifts, "s4_flip_rate")
    target = _as_float(shifts, "flip_target")
    if multiplier <= 0:
        raise SystemExit(f"s2_multiplier must be positive, got {multiplier}")
    if not 0.0 <= severity <= 1.0:
        raise SystemExit(f"s4_severity must lie in [0, 1], got {severity}")
    if not 0.0 <= s2_rate <= 1.0 or not 0.0 <= s4_rate <= 1.0:
        raise SystemExit("recorded flip rates must lie in [0, 1]")
    if not 0.0 < target <= 1.0:
        raise SystemExit(f"flip_target must lie in (0, 1], got {target}")
    return ShiftSettings(
        shocked_arm=arm,
        shocked_model=model.strip(),
        s2_multiplier=multiplier,
        s2_flip_rate=s2_rate,
        s2_reached_target=_as_bool(shifts.get("s2_reached_target"), "s2_reached_target"),
        s4_severity=severity,
        s4_flip_rate=s4_rate,
        s4_reached_target=_as_bool(shifts.get("s4_reached_target"), "s4_reached_target"),
        flip_target=target,
    )


def load_study_config() -> tuple[list[int], list[float], float, str, int, str, ShiftSettings]:
    """Read the three research-question configs and return the shared study settings."""
    docs = {name: _read_config(f"{name}.yaml") for name in ("rq1", "rq2", "rq3")}
    seeds = _match(
        "seeds",
        {name: _number_list(doc, "seeds", name, as_int=True) for name, doc in docs.items()},
    )
    lambda_ref = _match(
        "lambda_ref",
        {name: float(doc["lambda_ref"]) for name, doc in docs.items()},
    )
    features = _match(
        "features",
        {name: _feature_block(doc, name) for name, doc in docs.items()},
    )
    grid = _number_list(docs["rq1"], "lambda_grid", "rq1", as_int=False)
    if float(lambda_ref) not in {float(v) for v in grid}:
        raise SystemExit(f"lambda_ref {lambda_ref} is not on the rq1 lambda grid {grid}")
    backend, dim, model = features
    return seeds, grid, float(lambda_ref), backend, dim, model, _load_shifts(docs["rq2"])


def normalize_backend(name: str) -> str:
    """Map a config or CLI backend name to ``tfidf`` or ``minilm``."""
    key = _BACKEND_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"feature backend must be 'tfidf' or 'minilm', got {name!r}")
    return key


SEEDS, LAMBDA_GRID, LAMBDA_REF, FEATURE_BACKEND, DIM, FEATURE_MODEL, SHIFTS = load_study_config()

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


def featurise(
    texts: list[str],
    fit_index: np.ndarray,
    backend: str,
    dim: int,
    seed: int,
) -> np.ndarray:
    """Fit ``backend`` on the calibration rows and transform the whole corpus.

    ``tfidf`` fits character n-grams and truncated SVD. ``minilm`` encodes
    with the frozen checkpoint (cached under ``data/cache/``) and fits PCA on
    the calibration rows only.
    """
    backend = normalize_backend(backend)
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"feature dimension must be positive, got {dim}")
    if backend == "tfidf":
        features, _fitted = build_features(
            texts, fit_index, TfidfSvdBackend(dim, min_df=2, seed=seed)
        )
        return features
    model = SentenceTransformerBackend(dim, model_name=FEATURE_MODEL, seed=seed)
    embeddings = cached_sentence_embeddings(texts, model, EMBED_CACHE)
    features = model.fit_transform_embeddings(embeddings, fit_index)
    if features.shape[0] != len(texts):
        raise ValueError(f"backend returned {features.shape[0]} rows for {len(texts)} texts")
    if not np.isfinite(features).all():
        raise ValueError("feature matrix contains NaN or inf")
    return features


def build_setup(seed: int = 0, dim: int | None = None, *, backend: str | None = None) -> Setup:
    """Load the active source and featurise it.

    ``dim`` and ``backend`` default to the frozen config. The ablation passes
    them explicitly so it can score every candidate before that config exists
    as the winner. The reducer is fit on this seed's calibration split.
    """
    backend_name = normalize_backend(FEATURE_BACKEND if backend is None else backend)
    width = DIM if dim is None else int(dim)
    if width < 1:
        raise ValueError(f"feature dimension must be positive, got {width}")
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
    features = featurise(texts, splits.calibration, backend_name, width, seed)
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
        "feature_backend": FEATURE_BACKEND,
        "feature_dim": DIM,
        "feature_model": FEATURE_MODEL,
        "shocked_arm": SHIFTS.shocked_arm,
        "s2_multiplier": SHIFTS.s2_multiplier,
        "s4_severity": SHIFTS.s4_severity,
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
