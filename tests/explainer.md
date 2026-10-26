# Tests

`tests/test_core.py` checks the project without downloading RouterBench. Run it from the project root:

```
pytest tests/test_core.py
```

The file has five tests. The first two are from setup and the data task. The last three are the simulator checks. Two small policy classes at the bottom exist only so the simulator tests can call `act` and `update`.

## `test_linucb_import`

Imports `LinUCB` from `obp.policy` and checks that the import succeeded. This locks the Open Bandit Pipeline dependency before any policy code uses it.

## `test_pivot_split_and_pca`

Builds a tiny wide table in a temp directory and runs the data pipeline on it.

- `load_long` turns wide `model` and `model|total_cost` columns into one row per sample and model.
- `filter_evals` keeps MMLU, GSM8K, MBPP, and MT-Bench, and drops a row whose eval name matches none of those.
- `pivot_wide` makes one row per query with quality and cost for the four arms and the held-out model. A query missing a finite value for any of the four arms is dropped. Performance is clipped to `[0, 1]` here.
- `split_ids` makes a 70/30 split, seed 0, stratified by eval family. The two id sets are disjoint and cover every kept id, and each family appears in both sets.
- `fit_contexts` fits PCA on the train rows only, appends a column of ones, and matches a PCA fit on those same rows.

## `test_reward_cost_clip_and_oracle_ties`

Check (i). Hand-written numbers, no random draw.

- Qualities `(1.0, 0.0)`, costs `(0.25, 1.0)`, train max `1`, and \(\lambda = 1\) produce rewards `(0.75, -1.0)`.
- A one-step episode that picks arm 1 records oracle arm 0, gap `1.75`, and regret `1.75`. `update=False`, so the policy's update list stays empty.
- Raw cost `1.5` with train max `1.0` normalizes to `1.0`.
- A cost matrix whose fifth column is `50` has the same train maximum as the first four columns alone (`1.0`).
- Train max `0` makes every normalized cost `0`.
- Quality `1.5` is left as `1.5`. Clipping already happened in `pivot_wide`.
- Qualities `(0.8, 0.8)` and raw costs `(1.5, 2.0)` both reward `-0.2`. The oracle picks arm 0, the cheaper raw cost. If both raw costs are `2.0`, it still picks arm 0, the lower index.

## `test_uniform_regret_rate_is_stable`

Check (ii). A fixed synthetic stream: `np.random.default_rng(0)`, `T = 2000`, `K = 4`, quality uniform on `[0, 1]`, raw cost uniform on `[0.1, 1]`, \(\lambda = 1\), and `train_max` equal to that cost matrix's maximum.

For every step, the oracle arm's reward equals the max reward on that row, and every other arm is at most that high.

A round-robin policy plays arm `t % 4`. The test checks that the action list is exactly that sequence. It then compares `regret(1000) / 1000` with `regret(2000) / 2000`. The relative gap `|rate_2000 - rate_1000| / |rate_1000|` must be under 10%. On this draw the rates are about `0.406` and `0.402`, a gap of about `1.1%`. `update` stays off.

## `test_episode_updates_only_when_asked`

Three hand-written steps. One policy is run with `update=True`, another with the same actions and `update=False`.

With `update=True`, `update` is called once per step. Each call receives that step's context row, the chosen arm, and that arm's reward. The reward is a Python `float` and matches `arm_rewards` for the chosen arm. The result's `rewards` array stores the same values.

With `update=False`, the update list stays empty.

## Helpers

`_FixedArm` returns one arm every step, or the next arm in a list. It appends `(context, action, reward)` on `update`.

`_RoundRobin` returns `t % n_arms` and raises if `update` is called. Check (ii) uses it with `update=False`, so that raise means the episode called `update` when it should not have.
