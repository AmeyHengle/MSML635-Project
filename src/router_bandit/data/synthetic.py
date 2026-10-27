"""Synthetic streams with known ground truth.

RouterBench cannot support two of the verification tests in the proposal:

* test (iii) -- LinUCB attains sublinear regret on a *linear* bandit with known
  parameters. RouterBench's reward surface is unknown, so there is nothing to
  be correct about.
* test (iv) -- IPS is unbiased on uniformly random logs. Checking unbiasedness
  needs the true policy value, which only a generative model provides.

:class:`SyntheticStream` satisfies the same :class:`~router_bandit.data.stream.Stream`
protocol as :class:`~router_bandit.data.stream.RouterBenchStream`, so the same
policies and evaluators run against it unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SyntheticStream"]


class SyntheticStream:
    """Linear contextual bandit with a known reward model.

    Quality is ``clip(theta_a . x + offset, 0, 1)``, which is linear in the
    context (up to clipping), so a well-specified linear policy can in
    principle reach zero regret. Costs are log-spaced across arms to mirror the
    roughly 72x cheap-to-premium spread measured on RouterBench.

    Parameters
    ----------
    n_rounds, n_arms, dim:
        Stream geometry. ``dim`` excludes the bias term, which is appended when
        ``add_bias`` is set.
    noise:
        Standard deviation of Gaussian observation noise added to quality.
    bernoulli:
        If set, quality is drawn as ``Bernoulli(p)`` with ``p`` the clipped
        linear score. This gives a deliberately mis-specified, high-variance
        variant for stress-testing linear policies.
    cost_spread:
        Ratio between the most and least expensive arm.
    skill_scale:
        Magnitude of the per-arm parameter vectors; larger values make arms more
        distinguishable and the routing problem easier.
    """

    def __init__(
        self,
        n_rounds: int = 5000,
        n_arms: int = 11,
        dim: int = 16,
        *,
        noise: float = 0.05,
        bernoulli: bool = False,
        cost_spread: float = 72.0,
        skill_scale: float = 0.35,
        offset: float = 0.5,
        add_bias: bool = True,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._rng = rng
        self._n_rounds = int(n_rounds)
        self._n_arms = int(n_arms)
        self._noise = float(noise)
        self._bernoulli = bool(bernoulli)
        self._add_bias = bool(add_bias)

        raw = rng.normal(size=(self._n_rounds, int(dim)))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
        self._raw_contexts = raw
        self._contexts = np.hstack([raw, np.ones((self._n_rounds, 1))]) if add_bias else raw

        # True parameters, defined over the *augmented* context so that the
        # bias column carries each arm's baseline competence.
        theta = rng.normal(scale=skill_scale, size=(self._n_arms, int(dim)))
        if add_bias:
            baseline = np.linspace(-0.15, 0.25, self._n_arms)
            theta = np.hstack([theta, (offset + baseline)[:, None]])
        self._theta = theta

        # Cheap -> expensive, log-spaced, normalised to [~0, 1].
        costs = np.logspace(0.0, np.log10(cost_spread), self._n_arms)
        self._arm_cost = costs / costs.max()

        scores = self._contexts @ self._theta.T
        if not add_bias:
            scores = scores + offset
        self._mean_quality = np.clip(scores, 0.0, 1.0)

        if self._bernoulli:
            self._quality = rng.binomial(1, self._mean_quality).astype(np.float64)
        else:
            noisy = self._mean_quality + rng.normal(scale=self._noise, size=self._mean_quality.shape)
            self._quality = np.clip(noisy, 0.0, 1.0)

        self._cost = np.tile(self._arm_cost, (self._n_rounds, 1))

    # -- Stream protocol --------------------------------------------------- #
    @property
    def n_rounds(self) -> int:
        return self._n_rounds

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def dim(self) -> int:
        return int(self._contexts.shape[1])

    def context(self, t: int) -> np.ndarray:
        return self._contexts[t]

    def quality_vec(self, t: int) -> np.ndarray:
        return self._quality[t]

    def cost_vec(self, t: int) -> np.ndarray:
        return self._cost[t]

    # -- evaluator-facing / ground truth ----------------------------------- #
    def contexts(self) -> np.ndarray:
        return self._contexts

    def quality_matrix(self) -> np.ndarray:
        return self._quality

    def cost_matrix(self) -> np.ndarray:
        return self._cost

    @property
    def true_theta(self) -> np.ndarray:
        """``(n_arms, dim)`` true parameters -- the ground truth for test (iii)."""
        return self._theta

    def expected_quality(self) -> np.ndarray:
        """Noise-free ``(n_rounds, n_arms)`` quality."""
        return self._mean_quality

    def expected_rewards(self, lam: float) -> np.ndarray:
        """Noise-free reward matrix under ``lam``."""
        return self._mean_quality - float(lam) * self._cost

    def true_policy_value(self, actions: np.ndarray, lam: float) -> float:
        """Exact value of a deterministic action sequence.

        This is the ground truth that off-policy estimators are scored against
        in verification test (iv).
        """
        actions = np.asarray(actions, dtype=np.int64)
        rewards = self.expected_rewards(lam)
        return float(rewards[np.arange(actions.size), actions].mean())

    def true_uniform_value(self, lam: float) -> float:
        """Exact value of the uniformly random policy."""
        return float(self.expected_rewards(lam).mean())

    def __len__(self) -> int:
        return self._n_rounds

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "bernoulli" if self._bernoulli else "gaussian"
        return (
            f"SyntheticStream(n_rounds={self._n_rounds}, n_arms={self._n_arms}, "
            f"dim={self.dim}, {kind})"
        )
