"""Off-policy evaluation: IPS, SNIPS, and the doubly robust estimator.

Implemented directly rather than via the Open Bandit Pipeline. OBP declares
``requires-python >=3.8,<3.11`` and transitively pulls PyTorch and the CUDA
toolchain -- several gigabytes -- for estimators that are closed-form
expressions of a few lines. Keeping them here removes a hard Python ceiling
from the project; OBP is retained only in a separate pinned environment to
cross-check these implementations.

RQ3 asks whether these estimators can be trusted for routing. Because the
simulator retains every arm's reward, the true value of any evaluation policy
is computable, so each estimator's bias and RMSE are measurable rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OPEResult", "doubly_robust", "fit_reward_model", "ips", "on_policy_value", "snips"]


@dataclass(frozen=True)
class OPEResult:
    estimator: str
    value: float
    std_error: float

    def error_against(self, truth: float) -> float:
        return self.value - truth


def on_policy_value(rewards: np.ndarray) -> float:
    """Ground-truth value when the evaluation policy was actually run."""
    return float(np.mean(rewards))


def _indicator(actions: np.ndarray, target_actions: np.ndarray) -> np.ndarray:
    return (np.asarray(actions) == np.asarray(target_actions)).astype(np.float64)


def ips(
    rewards: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
) -> OPEResult:
    """Inverse propensity scoring.

    Unbiased when the logging policy has full support, but high variance when
    propensities get small -- which is exactly what happens as a logging policy
    becomes more greedy.
    """
    propensities = np.asarray(propensities, dtype=np.float64)
    if np.any(propensities <= 0):
        raise ValueError("propensities must be strictly positive")
    terms = _indicator(actions, target_actions) / propensities * np.asarray(rewards)
    n = terms.size
    return OPEResult("IPS", float(terms.mean()), float(terms.std(ddof=1) / np.sqrt(n)))


def snips(
    rewards: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
) -> OPEResult:
    """Self-normalised IPS: trades a little bias for much lower variance."""
    propensities = np.asarray(propensities, dtype=np.float64)
    weights = _indicator(actions, target_actions) / propensities
    denom = weights.sum()
    if denom <= 0:
        return OPEResult("SNIPS", float("nan"), float("nan"))
    value = float((weights * np.asarray(rewards)).sum() / denom)
    resid = weights * (np.asarray(rewards) - value) / (denom / weights.size)
    return OPEResult("SNIPS", value, float(resid.std(ddof=1) / np.sqrt(weights.size)))


def doubly_robust(
    rewards: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
    q_hat: np.ndarray,
) -> OPEResult:
    """Doubly robust estimator (Dudik et al., 2011).

    Consistent if *either* the propensities or the reward model ``q_hat`` are
    correct, and typically far lower variance than IPS.

    Parameters
    ----------
    q_hat:
        ``(n, n_arms)`` predicted reward for every arm on every logged query.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    propensities = np.asarray(propensities, dtype=np.float64)
    target_actions = np.asarray(target_actions, dtype=np.int64)
    q_hat = np.asarray(q_hat, dtype=np.float64)

    rows = np.arange(rewards.size)
    direct = q_hat[rows, target_actions]
    correction = _indicator(actions, target_actions) / propensities * (
        rewards - q_hat[rows, actions]
    )
    terms = direct + correction
    return OPEResult("DR", float(terms.mean()), float(terms.std(ddof=1) / np.sqrt(terms.size)))


def fit_reward_model(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_arms: int,
    *,
    ridge: float = 1.0,
) -> np.ndarray:
    """Per-arm ridge reward model fitted on logged data only.

    Trained strictly on observed ``(context, action, reward)`` triples -- the
    model never sees counterfactual rewards, which is what makes the resulting
    DR estimate an honest off-policy estimate rather than a disguised oracle.

    Returns ``(n, n_arms)`` predictions for every logged context.
    """
    from sklearn.linear_model import Ridge

    contexts = np.asarray(contexts, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)

    q_hat = np.zeros((contexts.shape[0], n_arms))
    global_mean = float(rewards.mean()) if rewards.size else 0.0
    for arm in range(n_arms):
        mask = actions == arm
        if mask.sum() < 2:
            q_hat[:, arm] = global_mean
            continue
        est = Ridge(alpha=ridge).fit(contexts[mask], rewards[mask])
        q_hat[:, arm] = est.predict(contexts)
    return q_hat
