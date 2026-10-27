"""The ``Stream`` protocol -- the single interface everything downstream uses.

A stream emits, for each round ``t``, a context vector and the *full* vectors
of per-arm quality and normalised cost. It deliberately does **not** apply
lambda and does **not** enforce bandit feedback:

* lambda stays out so one replay serves an entire lambda sweep, instead of
  re-streaming the data for every point on the cost-quality frontier;
* the bandit-feedback restriction is enforced one layer up, in
  :mod:`router_bandit.envs`, which hands the policy a single scalar and keeps
  the full vector for the evaluator.

Both :class:`RouterBenchStream` and
:class:`~router_bandit.data.synthetic.SyntheticStream` satisfy this protocol,
so experiments run unchanged against real or synthetic data.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .loaders import RouterBenchData

__all__ = ["RouterBenchStream", "Stream"]


@runtime_checkable
class Stream(Protocol):
    """Minimal contract for a replayable query stream."""

    @property
    def n_rounds(self) -> int: ...

    @property
    def n_arms(self) -> int: ...

    @property
    def dim(self) -> int:
        """Context dimensionality."""
        ...

    def context(self, t: int) -> np.ndarray:
        """``(dim,)`` feature vector for round ``t``."""
        ...

    def quality_vec(self, t: int) -> np.ndarray:
        """``(n_arms,)`` quality for every arm at round ``t``."""
        ...

    def cost_vec(self, t: int) -> np.ndarray:
        """``(n_arms,)`` normalised cost for every arm at round ``t``."""
        ...


class RouterBenchStream:
    """Replay RouterBench rows as an online stream.

    Parameters
    ----------
    data:
        Loaded RouterBench matrices.
    features:
        ``(n_queries, dim)`` context matrix aligned with ``data``.
    order:
        Row indices, in stream order. Typically the ``stream`` split, shuffled.
        Supplying this explicitly is what lets the shift scheduler re-weight the
        query mix (scenario S3) without touching the underlying data.
    add_bias:
        Append a constant 1 feature, giving linear policies an intercept.
    """

    def __init__(
        self,
        data: RouterBenchData,
        features: np.ndarray,
        order: np.ndarray,
        *,
        add_bias: bool = True,
    ) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.shape[0] != data.n_queries:
            raise ValueError(
                f"features has {features.shape[0]} rows but data has {data.n_queries}"
            )
        order = np.asarray(order, dtype=np.int64)
        if order.size and (order.max() >= data.n_queries or order.min() < 0):
            raise IndexError("order contains indices outside the dataset")

        self._data = data
        self._order = order
        self._add_bias = bool(add_bias)
        self._features = (
            np.hstack([features, np.ones((features.shape[0], 1))]) if add_bias else features
        )

    # -- protocol ---------------------------------------------------------- #
    @property
    def n_rounds(self) -> int:
        return int(self._order.size)

    @property
    def n_arms(self) -> int:
        return self._data.n_arms

    @property
    def dim(self) -> int:
        return int(self._features.shape[1])

    def context(self, t: int) -> np.ndarray:
        return self._features[self._order[t]]

    def quality_vec(self, t: int) -> np.ndarray:
        return self._data.quality[self._order[t]]

    def cost_vec(self, t: int) -> np.ndarray:
        return self._data.cost_norm[self._order[t]]

    # -- convenience ------------------------------------------------------- #
    @property
    def data(self) -> RouterBenchData:
        return self._data

    @property
    def order(self) -> np.ndarray:
        return self._order

    def feature_matrix(self) -> np.ndarray:
        """``(n_queries, dim)`` augmented features for *every* query, not just
        the streamed ones.

        Needed by the full-information supervised baseline, which trains on the
        calibration split: it must see exactly the same feature space the online
        policies see, bias column included.
        """
        return self._features

    def contexts(self) -> np.ndarray:
        """``(n_rounds, dim)`` all contexts in stream order."""
        return self._features[self._order]

    def quality_matrix(self) -> np.ndarray:
        """``(n_rounds, n_arms)`` -- evaluator-facing."""
        return self._data.quality[self._order]

    def cost_matrix(self) -> np.ndarray:
        """``(n_rounds, n_arms)`` -- evaluator-facing."""
        return self._data.cost_norm[self._order]

    def dollar_cost_matrix(self) -> np.ndarray:
        """``(n_rounds, n_arms)`` un-normalised dollar costs, for reporting."""
        return self._data.cost[self._order]

    def meta(self):
        """Per-round metadata rows, in stream order."""
        return self._data.meta.iloc[self._order].reset_index(drop=True)

    def families(self) -> np.ndarray:
        return self.meta()["family"].to_numpy()

    def reorder(self, order: np.ndarray) -> RouterBenchStream:
        """A new stream over the same data with a different round order."""
        raw = self._features[:, :-1] if self._add_bias else self._features
        return RouterBenchStream(self._data, raw, order, add_bias=self._add_bias)

    def __len__(self) -> int:
        return self.n_rounds

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RouterBenchStream(n_rounds={self.n_rounds}, n_arms={self.n_arms}, "
            f"dim={self.dim})"
        )


def build_stream(
    data: RouterBenchData,
    features: np.ndarray,
    index: np.ndarray,
    *,
    seed: int = 0,
    shuffle: bool = True,
    add_bias: bool = True,
) -> RouterBenchStream:
    """Build a stream over ``index``, optionally shuffled.

    Shuffling is on by default: RouterBench rows arrive grouped by ``eval_name``,
    so replaying them in file order would inject a huge unintended distribution
    shift and make the stationary experiments meaningless.
    """
    order = np.asarray(index, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    return RouterBenchStream(data, features, order, add_bias=add_bias)
