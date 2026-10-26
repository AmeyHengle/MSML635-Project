# Simulator

Ticket 003. `src/sim.py` turns a table of quality and raw cost into rewards, picks the best arm, and plays one episode. The policy sees the context and the reward of the arm it chose. The full reward matrix stays inside the simulator so the oracle and the regret can be computed.

`src/metrics.py` is still empty. Regret lives here so a later task can call it. Shifts, policies, and off-policy evaluation are later tasks. Quality is already clipped to `[0, 1]` in `pivot_wide` (`src/data.py`). This module uses those values as quality and does not clip them again.

## Reward

For each arm,

\[
\tilde{c} = \mathrm{clip}(c / \max_{\text{train}} c,\, 0, 1), \qquad r = q - \lambda \tilde{c}
\]

`train_cost_max` is the maximum raw cost over the four arms on the train rows. It reads only the first four columns, so a fifth held-out column does not change the number. The episode takes that scalar as `train_max` and does not recompute it from the stream. A later shift can reuse the same maximum. If the maximum is 0, every normalized cost is 0.

Example. Qualities `(1.0, 0.0)`, raw costs `(0.25, 1.0)`, train max `1.0`, and \(\lambda = 1\) give normalized costs `(0.25, 1.0)` and rewards `(0.75, -1.0)`. A raw cost of `1.5` with train max `1.0` clips to `1.0`.

## Oracle and regret

The oracle is the arm with the highest reward. If two arms tie, it picks the lower raw cost. If the raw costs also tie, it picks the lower arm index.

Qualities `(0.8, 0.8)`, raw costs `(1.5, 2.0)`, train max `1.0`, and \(\lambda = 1\) both clip to a normalized cost of `1`, so both rewards are `-0.2`. The oracle selects arm 0 because `1.5` is cheaper. Equal raw costs also select arm 0.

A gap is the oracle reward minus the chosen reward. `regret` is the sum of those gaps. Choosing arm 1 in the example above leaves a gap of `0.75 - (-1.0) = 1.75`.

## Episode

`run_episode(policy, stream, update)` walks the stream in order. `Stream` holds

| Field | Shape | Meaning |
|---|---|---|
| `context` | `(T, d)` | What `policy.act` receives, one row per step |
| `quality` | `(T, K)` | Already mapped quality |
| `cost` | `(T, K)` | Raw cost of the arms in play |
| `lam` | scalar | Cost penalty |
| `train_max` | scalar | Train-split maximum, fixed for the episode |

The policy needs `act(x) -> int` and `update(x, a, r)`. Each step calls `act` with that row of context only, then reads the chosen arm's reward as a Python float. `update=True` calls `update` once per step with that same context, arm, and reward. `update=False` does not call `update`.

The result is an `EpisodeResult`:

| Field | Meaning |
|---|---|
| `actions` | Arm chosen at each step |
| `rewards` | Reward of that arm |
| `oracle_actions` | Oracle arm at each step |
| `gaps` | Oracle reward minus chosen reward |
| `regret` | Sum of `gaps` |

`regret(t) / t` is `gaps[:t].sum() / t`. The cumulative value is the `regret` field.

## What each function does

- `train_cost_max(cost)` — one scalar, max of the first four cost columns.
- `normalized_cost(cost, train_max)` — divide by that scalar and clip to `[0, 1]`.
- `arm_rewards(quality, cost, lam, train_max)` — `q - λ * normalized cost`.
- `oracle_action(...)` — best arm for one query, with the tie-break above. `_oracle_arm` is the same rule on a reward row that is already computed.
- `regret(gaps)` — sum of the per-step gaps.
- `run_episode(...)` — the loop described above.
