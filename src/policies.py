"""Policy implementations for LLM routing.

Baselines (Always-Cheap, Always-Frontier, Uniform, Oracle),
Supervised Ridge router, Confidence-based Cascade,
OBP contextual bandits (LinUCB, LinTS, LinEpsilonGreedy),
and Discounted LinUCB for non-stationary environments.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from sklearn.linear_model import Ridge


class Policy(Protocol):
    """Protocol that all routing policies must implement."""

    def act(self, context: np.ndarray) -> int:
        ...

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        ...

    def propensity(self, context: np.ndarray, action: int) -> float:
        ...


class AlwaysCheapPolicy:
    """Always routes to the arm with the lowest training-set raw cost."""

    def __init__(self, cheap_arm: int = 0, n_actions: int = 4) -> None:
        self.cheap_arm = int(cheap_arm)
        self.n_actions = int(n_actions)

    def act(self, context: np.ndarray) -> int:
        return self.cheap_arm

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass  # Frozen

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 if action == self.cheap_arm else 0.0


class AlwaysFrontierPolicy:
    """Always routes to the most expensive frontier model arm."""

    def __init__(self, frontier_arm: int = 3, n_actions: int = 4) -> None:
        self.frontier_arm = int(frontier_arm)
        self.n_actions = int(n_actions)

    def act(self, context: np.ndarray) -> int:
        return self.frontier_arm

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass  # Frozen

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 if action == self.frontier_arm else 0.0


class UniformRandomPolicy:
    """Routes queries uniformly at random across all available arms."""

    def __init__(self, n_actions: int = 4, random_state: int | None = None) -> None:
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(random_state)

    def act(self, context: np.ndarray) -> int:
        return int(self.rng.integers(0, self.n_actions))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass  # Frozen

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 / float(self.n_actions)


class OraclePolicy:
    """Oracle router that selects the arm maximizing reward using ground truth."""

    def __init__(self, n_actions: int = 4) -> None:
        self.n_actions = int(n_actions)

    def act(self, context: np.ndarray) -> int:
        raise NotImplementedError("Oracle requires the ground truth reward vector.")

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0


class SupervisedRidgePolicy:
    """Supervised router: fits one Ridge regression per arm to predict quality.

    Chooses a_t = argmax_a (q_hat(x, a) - lambda * c_tilde(a)) using pre-shift costs.
    Remains completely frozen on the test stream.
    """

    def __init__(
        self,
        models: list[Ridge],
        costs: np.ndarray | Sequence[float],
        lambda_weight: float = 1.0,
    ) -> None:
        self.models = models
        self.costs = np.asarray(costs, dtype=float)
        self.lambda_weight = float(lambda_weight)
        self.n_actions = len(models)

    @classmethod
    def fit(
        cls,
        contexts_train: np.ndarray,
        qualities_train: np.ndarray,
        normalized_costs: np.ndarray | Sequence[float],
        ridge_alpha: float = 1.0,
        lambda_weight: float = 1.0,
    ) -> SupervisedRidgePolicy:
        """Fit one Ridge regressor per arm on training contexts and qualities."""
        models: list[Ridge] = []
        n_arms = qualities_train.shape[1]
        for a in range(n_arms):
            model = Ridge(alpha=ridge_alpha, fit_intercept=False)
            model.fit(contexts_train, qualities_train[:, a])
            models.append(model)
        return cls(models=models, costs=normalized_costs, lambda_weight=lambda_weight)

    def predict_qualities(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=float).reshape(1, -1)
        return np.array([float(model.predict(x)[0]) for model in self.models])

    def act(self, context: np.ndarray) -> int:
        q_hat = self.predict_qualities(context)
        scores = q_hat - self.lambda_weight * self.costs
        max_score = float(np.max(scores))
        candidates = np.where(np.isclose(scores, max_score, atol=1e-12))[0]
        if len(candidates) == 1:
            return int(candidates[0])
        # Tie break in favor of cheaper arm
        cand_costs = self.costs[candidates]
        best_cand_idx = int(np.argmin(cand_costs))
        return int(candidates[best_cand_idx])

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass  # Frozen

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 if action == self.act(context) else 0.0


class CascadePolicy:
    """Confidence-based cascade router (FrugalGPT style).

    Walks Cheap -> Mid -> Strong Open -> Frontier.
    Stops at the first arm whose predicted quality is at least tau (default 0.5).
    Otherwise picks Frontier.
    """

    def __init__(
        self,
        supervised_policy: SupervisedRidgePolicy,
        tau: float = 0.5,
    ) -> None:
        self.supervised_policy = supervised_policy
        self.tau = float(tau)
        self.n_actions = supervised_policy.n_actions

    def act(self, context: np.ndarray) -> int:
        q_hat = self.supervised_policy.predict_qualities(context)
        for a in range(self.n_actions - 1):
            if q_hat[a] >= self.tau:
                return a
        return self.n_actions - 1

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass  # Frozen

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 if action == self.act(context) else 0.0


class LinUCBPolicy:
    """Wrapper around OBP's LinUCB algorithm."""

    def __init__(
        self,
        dim: int = 65,
        n_actions: int = 4,
        alpha: float = 1.0,
        seed: int = 0,
    ) -> None:
        from obp.policy import LinUCB

        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.policy = LinUCB(
            dim=self.dim,
            n_actions=self.n_actions,
            len_list=1,
            epsilon=self.alpha,
            random_state=seed,
        )

    def act(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(1, -1)
        return int(self.policy.select_action(x)[0])

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, dtype=float).reshape(1, -1)
        self.policy.update_params(action=int(action), reward=float(reward), context=x)

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 if action == self.act(context) else 0.0


class LinTSPolicy:
    """Wrapper around OBP's Linear Thompson Sampling algorithm."""

    def __init__(
        self,
        dim: int = 65,
        n_actions: int = 4,
        seed: int = 0,
    ) -> None:
        from obp.policy import LinTS

        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.policy = LinTS(
            dim=self.dim,
            n_actions=self.n_actions,
            len_list=1,
            random_state=seed,
        )

    def act(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(1, -1)
        return int(self.policy.select_action(x)[0])

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, dtype=float).reshape(1, -1)
        self.policy.update_params(action=int(action), reward=float(reward), context=x)

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 if action == self.act(context) else 0.0


class LinEpsilonGreedyPolicy:
    """Wrapper around OBP's Linear Epsilon Greedy algorithm."""

    def __init__(
        self,
        dim: int = 65,
        n_actions: int = 4,
        epsilon: float = 0.1,
        seed: int = 0,
    ) -> None:
        from obp.policy import LinEpsilonGreedy

        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self.policy = LinEpsilonGreedy(
            dim=self.dim,
            n_actions=self.n_actions,
            len_list=1,
            epsilon=self.epsilon,
            random_state=seed,
        )

    def act(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(1, -1)
        return int(self.policy.select_action(x)[0])

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, dtype=float).reshape(1, -1)
        self.policy.update_params(action=int(action), reward=float(reward), context=x)

    def propensity(self, context: np.ndarray, action: int) -> float:
        """Propensity is (1 - eps) + eps / K on greedy arm and eps / K otherwise."""
        x = np.asarray(context, dtype=float).reshape(1, -1)
        theta_hat = np.concatenate(
            [self.policy.A_inv[i] @ self.policy.b[:, i][:, np.newaxis] for i in range(self.n_actions)],
            axis=1,
        )
        predicted_rewards = (x @ theta_hat).flatten()
        greedy_arm = int(np.argmax(predicted_rewards))
        eps = self.epsilon
        K = self.n_actions
        if action == greedy_arm:
            return float((1.0 - eps) + eps / K)
        return float(eps / K)


class DiscountedLinUCB:
    """Discounted LinUCB for non-stationary environments (Russac et al., 2019).

    Maintains covariance A and reward vector b per arm with exponential discount factor gamma:
        On chosen arm: A_a <- gamma * A_a + x x^T,  b_a <- gamma * b_a + r * x
    """

    def __init__(
        self,
        dim: int = 65,
        n_actions: int = 4,
        gamma: float = 0.99,
        alpha: float = 1.0,
        ridge: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.rng = np.random.default_rng(random_state)

        self.A = [self.ridge * np.identity(self.dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.dim) for _ in range(self.n_actions)]

    def act(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).flatten()
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.pinv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            variance = float(x @ A_inv @ x)
            scores[a] = float(theta_hat @ x) + self.alpha * np.sqrt(max(variance, 0.0))

        max_score = float(np.max(scores))
        candidates = np.where(np.isclose(scores, max_score, atol=1e-12))[0]
        return int(candidates[0])

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, dtype=float).flatten()
        r = float(reward)
        a = int(action)
        self.A[a] = self.gamma * self.A[a] + np.outer(x, x)
        self.b[a] = self.gamma * self.b[a] + r * x

    def propensity(self, context: np.ndarray, action: int) -> float:
        return 1.0 if action == self.act(context) else 0.0

    def add_arm(self) -> None:
        """Append a fresh arm with initial (A, b)."""
        self.n_actions += 1
        self.A.append(self.ridge * np.identity(self.dim))
        self.b.append(np.zeros(self.dim))
