"""Load and clean the RouterBench release into dense reward matrices.

The raw file is one row per query with eleven ``(quality, cost, response)``
triplets. We keep the full reward matrix on purpose: the router is only ever
shown the entry it selected, but the evaluator needs every entry to compute the
per-query oracle, regret, and off-policy-evaluation ground truth.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry as R

logger = logging.getLogger(__name__)

__all__ = ["RouterBenchData", "build", "clean", "load_raw", "parse_prompt"]


@dataclass(frozen=True)
class RouterBenchData:
    """Dense, validated view of RouterBench.

    Attributes
    ----------
    quality:
        ``(n_queries, n_arms)`` float array in ``[0, 1]``.
    cost:
        ``(n_queries, n_arms)`` dollar cost per query/model pair.
    cost_norm:
        ``cost`` divided by :attr:`cost_scale`, so lambda is on a calibrated
        scale where ``reward = quality - lambda * cost_norm``.
    cost_scale:
        The maximum per-query dollar cost over the whole matrix.
    meta:
        Per-query metadata: ``sample_id``, ``eval_name``, ``family``,
        ``is_documented``, ``prompt``, and the shipped oracle label.
    models:
        Arm names, aligned with the matrix columns.
    """

    quality: np.ndarray
    cost: np.ndarray
    cost_norm: np.ndarray
    cost_scale: float
    meta: pd.DataFrame
    models: tuple[str, ...] = R.MODELS

    @property
    def n_queries(self) -> int:
        return int(self.quality.shape[0])

    @property
    def n_arms(self) -> int:
        return int(self.quality.shape[1])

    def rewards(self, lam: float) -> np.ndarray:
        """Full reward matrix ``quality - lam * cost_norm`` for a given lambda.

        Evaluator-facing. The bandit environment exposes only one entry of this
        per round; see :mod:`router_bandit.envs`.
        """
        return self.quality - float(lam) * self.cost_norm

    def oracle(self, lam: float) -> np.ndarray:
        """Per-query oracle arm index under ``lam``."""
        return np.asarray(self.rewards(lam).argmax(axis=1), dtype=np.int64)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RouterBenchData(n_queries={self.n_queries}, n_arms={self.n_arms}, "
            f"cost_scale=${self.cost_scale:.5f})"
        )


def parse_prompt(value: object) -> str:
    """Normalise RouterBench's prompt field to a plain string.

    Prompts are stored as the *repr* of a list of strings, e.g.
    ``"['turn one', 'turn two']"``. We parse that and join the turns. Anything
    that fails to parse is returned as-is, so a truncated fixture still works.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, (list, tuple)):
            return "\n".join(str(v) for v in parsed)
        return str(parsed)
    return text


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the RouterBench release.

    Supports the shipped ``.pkl`` and the committed ``.parquet`` test fixture.

    Notes
    -----
    ``pandas.read_pickle`` executes code during unpickling. Only point this at
    the official ``withmartian/routerbench`` artefact.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            "  uv run hf download withmartian/routerbench --repo-type dataset "
            '--include "routerbench_0shot.pkl" --local-dir data/raw'
        )
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        df = pd.read_pickle(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported file type {suffix!r} for {path}")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a DataFrame (got {type(df)!r})")
    logger.info("Loaded %s rows x %s cols from %s", *df.shape, path)
    return df


def clean(
    df: pd.DataFrame,
    *,
    drop_responses: bool = True,
    excluded_families: frozenset[str] | None = None,
    keep_undocumented: bool = True,
) -> pd.DataFrame:
    """Coerce dtypes, attach family tags, and apply the filtering policy.

    Parameters
    ----------
    drop_responses:
        Drop the eleven ``|model_response`` text columns. They are unused for
        routing and dominate the file size.
    excluded_families:
        Families to remove. Defaults to
        :data:`~router_bandit.data.registry.DEFAULT_EXCLUDED_FAMILIES`
        (MT-Bench, which is far too small to split).
    keep_undocumented:
        Keep rows whose ``eval_name`` is not one of the families named in the
        paper. They stay tagged via ``is_documented`` so results can be reported
        both ways.
    """
    if excluded_families is None:
        excluded_families = R.DEFAULT_EXCLUDED_FAMILIES

    missing = [c for c in (R.ID_COL, R.EVAL_COL, *R.quality_cols(), *R.cost_cols()) if c not in df.columns]
    if missing:
        raise KeyError(f"RouterBench frame is missing expected columns: {missing[:6]}")

    out = df.copy()

    if drop_responses:
        out = out.drop(columns=[c for c in R.response_cols() if c in out.columns])

    # Quality ships as ``object`` dtype even though the values are numeric.
    for col in R.quality_cols():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in R.cost_cols():
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["family"] = out[R.EVAL_COL].map(R.eval_to_family)
    out["is_documented"] = out["family"].map(R.is_documented)

    before = len(out)
    if excluded_families:
        out = out[~out["family"].isin(excluded_families)]
    if not keep_undocumented:
        out = out[out["is_documented"]]
    if len(out) != before:
        logger.info("Filtering removed %d of %d rows", before - len(out), before)

    n_bad = int(out[R.quality_cols()].isna().sum().sum() + out[R.cost_cols()].isna().sum().sum())
    if n_bad:
        logger.warning("Dropping %d rows with non-numeric quality/cost entries", n_bad)
        out = out.dropna(subset=[*R.quality_cols(), *R.cost_cols()])

    if R.PROMPT_COL in out.columns:
        out[R.PROMPT_COL] = out[R.PROMPT_COL].map(parse_prompt)

    return out.reset_index(drop=True)


def build(
    df: pd.DataFrame,
    *,
    cost_scale: float | None = None,
    validate: bool = True,
) -> RouterBenchData:
    """Materialise dense ``quality`` / ``cost`` matrices from a cleaned frame.

    Parameters
    ----------
    cost_scale:
        Divisor used to normalise cost. Defaults to the maximum per-query cost
        in ``df``. Pass the training value explicitly when building a held-out
        split so both share one scale.
    """
    quality = df[R.quality_cols()].to_numpy(dtype=np.float64)
    cost = df[R.cost_cols()].to_numpy(dtype=np.float64)

    scale = float(cost.max()) if cost_scale is None else float(cost_scale)
    if scale <= 0:
        raise ValueError(f"cost_scale must be positive, got {scale}")

    meta_cols = [c for c in (R.ID_COL, R.EVAL_COL, "family", "is_documented", R.PROMPT_COL, R.ORACLE_COL) if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)

    data = RouterBenchData(
        quality=quality,
        cost=cost,
        cost_norm=cost / scale,
        cost_scale=scale,
        meta=meta,
    )
    if validate:
        validate_data(data)
    return data


def validate_data(data: RouterBenchData) -> None:
    """Fail loudly on malformed data rather than producing quiet nonsense."""
    q, c = data.quality, data.cost
    if q.shape != c.shape:
        raise ValueError(f"quality {q.shape} and cost {c.shape} disagree")
    if q.shape[1] != R.N_ARMS:
        raise ValueError(f"expected {R.N_ARMS} arms, got {q.shape[1]}")
    if len(data.meta) != q.shape[0]:
        raise ValueError(f"meta has {len(data.meta)} rows, matrices have {q.shape[0]}")
    if not np.isfinite(q).all() or not np.isfinite(c).all():
        raise ValueError("quality/cost contain NaN or inf after cleaning")
    if q.min() < 0.0 or q.max() > 1.0:
        raise ValueError(f"quality outside [0, 1]: [{q.min()}, {q.max()}]")
    if c.min() <= 0.0:
        raise ValueError(f"non-positive cost encountered: {c.min()}")


def load(
    path: str | Path,
    *,
    drop_responses: bool = True,
    excluded_families: frozenset[str] | None = None,
    keep_undocumented: bool = True,
    cost_scale: float | None = None,
) -> RouterBenchData:
    """Convenience wrapper: :func:`load_raw` then :func:`clean` then :func:`build`."""
    raw = load_raw(path)
    cleaned = clean(
        raw,
        drop_responses=drop_responses,
        excluded_families=excluded_families,
        keep_undocumented=keep_undocumented,
    )
    return build(cleaned, cost_scale=cost_scale)
