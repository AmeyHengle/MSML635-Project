"""Full-information supervised router.

Trained on the calibration split with the reward of *every* arm for *every*
query -- the labelling regime most academic routers assume and the one that
would be prohibitively expensive in deployment, since obtaining it means
running all eleven models on all queries.

This is the reference point for RQ1: how much does a bandit give up by
learning from one outcome per query instead of eleven?
"""

from __future__ import annotations

import numpy as np

from ..policies.base import Policy

__all__ = ["SupervisedRouter"]


class SupervisedRouter(Policy):
    """Per-arm reward regression fitted with full information.

    Parameters
    ----------
    model:
        ``"ridge"`` (default) matches LinUCB's hypothesis class, so RQ1
        isolates the effect of *feedback* rather than of model capacity.
        ``"gbdt"`` fits gradient-boosted trees instead, as a capacity ablation.
    """

    loggable = False

    def __init__(self, *, model: str = "ridge", alpha: float = 1.0, seed: int = 0) -> None:
        self.model = model
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._models: list = []
        self.name = f"Supervised({model})"

    def _make(self):
        if self.model == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=self.alpha)
        if self.model == "gbdt":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(random_state=self.seed, max_iter=150)
        raise ValueError(f"unknown model {self.model!r}")

    def fit(self, X: np.ndarray, rewards: np.ndarray) -> SupervisedRouter:
        """Fit one regressor per arm.

        Parameters
        ----------
        X:
            ``(n, dim)`` calibration contexts.
        rewards:
            ``(n, n_arms)`` reward of every arm on every calibration query.
        """
        X = np.asarray(X, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        if X.shape[0] != rewards.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, rewards has {rewards.shape[0]}")
        self._models = []
        for arm in range(rewards.shape[1]):
            est = self._make()
            est.fit(X, rewards[:, arm])
            self._models.append(est)
        return self

    def select(self, t: int, context: np.ndarray, available: np.ndarray) -> tuple[int, float]:
        if not self._models:
            raise RuntimeError("SupervisedRouter.fit must be called before select")
        x = np.asarray(context, dtype=np.float64).reshape(1, -1)
        preds = np.array([self._models[a].predict(x)[0] for a in available])
        return int(available[int(np.argmax(preds))]), 1.0
