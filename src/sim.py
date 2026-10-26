"""Simulator for contextual bandit LLM routing.

Reward, train-max normalization, oracle computation with tie-breaking,
episode loop, and production shift injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

ROLES = ("cheap", "mid", "strong_open", "frontier")


class Policy(Protocol):
    """Protocol for routing policies."""

    def act(self, context: np.ndarray) -> int:
        ...

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        ...


@dataclass
class EpisodeResult:
    """Stores the full trajectory and metrics of an episode."""

    actions: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)
    oracle_actions: np.ndarray  # (T,)
    oracle_rewards: np.ndarray  # (T,)
    regrets: np.ndarray  # (T,)
    cumulative_regrets: np.ndarray  # (T,)
    qualities: np.ndarray | None = None  # (T,)
    costs: np.ndarray | None = None  # (T,)


def normalize_cost(
    cost: np.ndarray | float,
    train_max_cost: float,
) -> np.ndarray | float:
    """Normalize raw cost by train-split maximum and clip to [0, 1].

    c_tilde(a) = clip(c(a) / max_train(c), 0, 1)
    """
    if train_max_cost <= 0:
        raise ValueError(f"train_max_cost must be positive, got {train_max_cost}")
    return np.clip(np.asarray(cost, dtype=float) / float(train_max_cost), 0.0, 1.0)


def compute_reward(
    quality: np.ndarray | float,
    normalized_cost: np.ndarray | float,
    lambda_weight: float = 1.0,
) -> np.ndarray | float:
    """Calculate reward: r = clip(quality, 0, 1) - lambda * clip(normalized_cost, 0, 1)."""
    q = np.clip(np.asarray(quality, dtype=float), 0.0, 1.0)
    c = np.clip(np.asarray(normalized_cost, dtype=float), 0.0, 1.0)
    return q - float(lambda_weight) * c


def find_oracle_action(
    rewards: np.ndarray | Sequence[float],
    costs: np.ndarray | Sequence[float],
) -> tuple[int, float]:
    """Find a* = argmax_a r(x, a), breaking ties in favor of the cheaper arm."""
    r = np.asarray(rewards, dtype=float)
    c = np.asarray(costs, dtype=float)
    if len(r) != len(c):
        raise ValueError(f"rewards length {len(r)} must match costs length {len(c)}")

    max_r = np.max(r)
    # Find all arms with max reward within float precision
    candidates = np.where(np.isclose(r, max_r, atol=1e-12, rtol=1e-10))[0]
    if len(candidates) == 1:
        best_arm = int(candidates[0])
    else:
        # Break ties in favor of cheaper arm (lowest cost)
        cand_costs = c[candidates]
        best_cand_idx = int(np.argmin(cand_costs))
        best_arm = int(candidates[best_cand_idx])
    return best_arm, float(r[best_arm])


def compute_oracle_matrix(
    rewards: np.ndarray,
    costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute oracle action and oracle reward for each row in a 2D matrix."""
    r_mat = np.asarray(rewards, dtype=float)
    c_mat = np.asarray(costs, dtype=float)
    if r_mat.shape != c_mat.shape:
        raise ValueError(f"rewards shape {r_mat.shape} must match costs shape {c_mat.shape}")

    n_samples = r_mat.shape[0]
    oracle_actions = np.zeros(n_samples, dtype=int)
    oracle_rewards = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        oracle_actions[i], oracle_rewards[i] = find_oracle_action(r_mat[i], c_mat[i])
    return oracle_actions, oracle_rewards


@dataclass
class QueryStream:
    """Container for streaming contexts, full reward matrix, and costs."""

    contexts: np.ndarray  # (T, d)
    rewards: np.ndarray  # (T, K)
    costs: np.ndarray  # (T, K) - used for tie breaking and cost tracking
    qualities: np.ndarray | None = None  # (T, K) - optional quality matrix
    eval_names: list[str] | None = None

    def __post_init__(self) -> None:
        if len(self.contexts) != len(self.rewards) or len(self.rewards) != len(self.costs):
            raise ValueError(
                f"length mismatch: contexts {len(self.contexts)}, "
                f"rewards {len(self.rewards)}, costs {len(self.costs)}"
            )

    def __len__(self) -> int:
        return len(self.contexts)


def run_episode(
    policy: Any,
    stream: QueryStream | dict[str, Any] | Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    update: bool = True,
) -> EpisodeResult:
    """Run an episode over a query stream.

    Calls policy.act(x) -> int and, when update is True, policy.update(x, a, r)
    with the chosen reward only. The full reward matrix remains inside the
    simulator for computing oracle actions, oracle rewards, and regret.
    """
    if isinstance(stream, dict):
        contexts = np.asarray(stream["contexts"])
        rewards = np.asarray(stream["rewards"])
        costs = np.asarray(stream["costs"])
        qualities = np.asarray(stream["qualities"]) if "qualities" in stream and stream["qualities"] is not None else None
        stream = QueryStream(contexts=contexts, rewards=rewards, costs=costs, qualities=qualities)

    if isinstance(stream, QueryStream):
        t_steps = len(stream)
        actions = np.zeros(t_steps, dtype=int)
        chosen_rewards = np.zeros(t_steps, dtype=float)
        chosen_qualities = np.zeros(t_steps, dtype=float) if stream.qualities is not None else None
        chosen_costs = np.zeros(t_steps, dtype=float)

        oracle_actions, oracle_rewards = compute_oracle_matrix(stream.rewards, stream.costs)

        for t in range(t_steps):
            x_t = stream.contexts[t]
            a_t = int(policy.act(x_t))
            r_t = float(stream.rewards[t, a_t])

            if update:
                policy.update(x_t, a_t, r_t)

            actions[t] = a_t
            chosen_rewards[t] = r_t
            chosen_costs[t] = stream.costs[t, a_t]
            if stream.qualities is not None:
                chosen_qualities[t] = stream.qualities[t, a_t]

        regrets = oracle_rewards - chosen_rewards
        # Verification: halt if any chosen reward exceeds the oracle
        if np.any(regrets < -1e-12):
            max_violation = float(np.min(regrets))
            raise ValueError(f"Halt: method reward exceeded oracle reward by {-max_violation} at one or more steps.")

        # Ensure no negative numerical jitter in regrets
        regrets = np.maximum(regrets, 0.0)
        cumulative_regrets = np.cumsum(regrets)

        return EpisodeResult(
            actions=actions,
            rewards=chosen_rewards,
            oracle_actions=oracle_actions,
            oracle_rewards=oracle_rewards,
            regrets=regrets,
            cumulative_regrets=cumulative_regrets,
            qualities=chosen_qualities,
            costs=chosen_costs,
        )

    # If stream is an arbitrary generator/iterator yielding (x, rewards_row, costs_row)
    act_list: list[int] = []
    rew_list: list[float] = []
    orc_act_list: list[int] = []
    orc_rew_list: list[float] = []
    cost_list: list[float] = []

    for item in stream:
        x_t = item[0]
        r_row = item[1]
        c_row = item[2]

        o_act, o_rew = find_oracle_action(r_row, c_row)
        a_t = int(policy.act(x_t))
        r_t = float(r_row[a_t])

        if update:
            policy.update(x_t, a_t, r_t)

        act_list.append(a_t)
        rew_list.append(r_t)
        orc_act_list.append(o_act)
        orc_rew_list.append(o_rew)
        cost_list.append(float(c_row[a_t]))

    actions_arr = np.array(act_list, dtype=int)
    rewards_arr = np.array(rew_list, dtype=float)
    oracle_act_arr = np.array(orc_act_list, dtype=int)
    oracle_rew_arr = np.array(orc_rew_list, dtype=float)
    regrets_arr = np.maximum(oracle_rew_arr - rewards_arr, 0.0)

    return EpisodeResult(
        actions=actions_arr,
        rewards=rewards_arr,
        oracle_actions=oracle_act_arr,
        oracle_rewards=oracle_rew_arr,
        regrets=regrets_arr,
        cumulative_regrets=np.cumsum(regrets_arr),
        costs=np.array(cost_list, dtype=float),
    )


def compute_train_max_cost(wide: pd.DataFrame, split: dict[str, list[str]]) -> float:
    """Compute the maximum raw cost observed in the train split across the four arms."""
    train_ids = set(split["train"])
    train_rows = wide[wide["sample_id"].astype(str).isin(train_ids)]
    cost_cols = [f"cost_{role}" for role in ROLES]
    max_cost = float(train_rows[cost_cols].to_numpy().max())
    if max_cost <= 0:
        raise ValueError(f"train max cost must be positive, found {max_cost}")
    return max_cost


def build_query_stream(
    wide: pd.DataFrame,
    contexts: np.ndarray,
    split_ids: list[str],
    train_max_cost: float,
    lambda_weight: float = 1.0,
    include_held_out: bool = False,
    split_reference_ids: list[str] | None = None,
) -> QueryStream:
    """Build a QueryStream for a specified split (e.g. test set)."""
    id_to_idx = {str(sid): i for i, sid in enumerate(wide["sample_id"].astype(str))}
    row_indices = [id_to_idx[str(sid)] for sid in split_ids]

    roles = list(ROLES)
    if include_held_out:
        roles.append("held_out")

    sub_df = wide.iloc[row_indices]

    if len(contexts) == len(wide):
        sub_contexts = contexts[row_indices]
    elif len(contexts) == len(split_ids):
        sub_contexts = contexts
    elif split_reference_ids is not None:
        ref_to_idx = {str(sid): i for i, sid in enumerate(split_reference_ids)}
        context_indices = [ref_to_idx[str(sid)] for sid in split_ids]
        sub_contexts = contexts[context_indices]
    else:
        raise ValueError(
            f"contexts shape {contexts.shape} does not match wide ({len(wide)}) "
            f"or split_ids ({len(split_ids)}) and no split_reference_ids provided"
        )

    qualities = sub_df[[f"quality_{role}" for role in roles]].to_numpy(dtype=float)
    raw_costs = sub_df[[f"cost_{role}" for role in roles]].to_numpy(dtype=float)

    normalized_costs = normalize_cost(raw_costs, train_max_cost)
    rewards = compute_reward(qualities, normalized_costs, lambda_weight=lambda_weight)

    return QueryStream(
        contexts=sub_contexts,
        rewards=rewards,
        costs=normalized_costs,
        qualities=qualities,
        eval_names=sub_df["eval_name"].astype(str).tolist(),
    )
