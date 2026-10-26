# `tests/test_core.py`

Offline checks for the data pipeline, the simulator, and the routers. No test downloads RouterBench or reads `data/`. From the repo root:

```
pytest tests/test_core.py
```

## `test_linucb_import`

Imports `obp.policy.LinUCB` and asserts the class exists. This is the step 1 proof that Open Bandit Pipeline is installed before any policy wrapper uses it.

## `test_pivot_split_and_pca`

Builds a tiny table in a temp directory.

- A wide pickle with `model` and `model|total_cost` columns becomes one long row per sample and model. `filter_evals` keeps the MMLU row and drops HellaSwag.
- `pivot_wide` maps four fake model names onto the four roles and keeps the held-out model. The query `gap` is missing the frontier arm and is dropped. Remaining rows have `cost_cheap == 1.0` and `quality_frontier == 0.9`.
- The 70/30 split is disjoint, covers every kept id, and puts each of the four families on both sides.
- PCA is fit on train rows only. Both context matrices end with a column of ones, and the train transform matches a fresh PCA fit on those same rows. `pca.n_samples_` equals the train count.

## `test_check_i_reward_normalization_and_oracle`

Hand-written numbers, no random draw.

- Raw cost `1.5` with train max `1.0` normalizes to `1.0`.
- Qualities `(1.0, 0.0)`, normalized costs `(0.25, 1.0)`, and \(\lambda = 1\) give rewards `(0.75, -1.0)`.
- The oracle picks arm 0. Choosing arm 1 leaves regret `0.75 - (-1.0) = 1.75`.
- Equal rewards `(0.5, 0.5)` select the cheaper cost. Cost `(0.8, 0.2)` selects arm 1. Cost `(0.1, 0.4)` selects arm 0.

## `test_check_ii_oracle_upper_bound_and_uniform_regret`

A fixed synthetic stream: `np.random.default_rng(0)`, `T = 2000`, `K = 4`, quality uniform on `[0.1, 0.9]`, cost uniform on `[0.05, 0.95]`, rewards `quality - cost`. Contexts are 10-dimensional noise plus a column of ones. The uniform policy ignores them.

For every arm and every step, that arm’s reward is at most the oracle reward.

The behavior policy draws uniformly with its own seed (`42`). Its `regret(t) / t` at `t = 2000` and at `t = 1000` must differ by less than 10% relative, using `|rate_2000 - rate_1000| / rate_1000`. A uniform policy on a stationary stream should have a flat average gap. This policy is a small class inside the test. Check (iii) uses `UniformRandomPolicy` from `src/policies.py` instead.

## `test_check_iii_synthetic_linear_bandit`

A pure linear bandit, separate from the RouterBench context. `np.random.default_rng(0)`, dimension 10, 4 arms, \(T = 5000\). Each arm has a true \(\theta\). Each context is a normal vector, then divided by its length. The reward is \(\theta^\top x\) plus noise \(\mathcal{N}(0, 0.1^2)\). Costs are all zero, so the oracle is just the best noisy reward and there is no cost penalty. There is no column of ones. `LinUCB` is built with `dim=10`, not 65.

`LinUCB` (`alpha=1`, seed 0) must drive `regret(T) / T` below `regret(T/2) / (T/2)`. That is the learning check: average regret falls on a stationary linear problem.

`UniformRandomPolicy` with seed 0 runs on the same reward matrix. Its two ratios differ by less than 10% relative. The uniform policy is not supposed to learn.

`DiscountedLinUCB` (`gamma=0.99`, `alpha=1`, `ridge=1`) runs on those same draws. Every action is in `{0, 1, 2, 3}` and the episode has length 5000. The test does not require its regret to fall. It checks that the custom update and the UCB score return legal arms.

## `test_all_policies_interface`

Builds one instance of each router except `OraclePolicy`, including a ridge model fit on 20 synthetic rows and a cascade around it. For a single context of ones, `act` returns an arm in `{0, 1, 2, 3}`, `update` accepts reward `0.5`, and `propensity` of the chosen arm is between 0 and 1. `OraclePolicy` is left out because `act` raises.
