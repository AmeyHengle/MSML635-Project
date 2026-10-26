# `src/sim.py`

Plays one query at a time. The policy receives the context and, if it is learning, the reward of the arm it picked. Quality and cost for every arm stay inside the simulator, which is what makes the oracle and the regret well defined. Shifts are not implemented here. The train maximum is computed once so a later shift can reuse it.

## Reward

Quality is clipped to `[0, 1]` again at reward time. `pivot_wide` already clipped `performance`, and this second clip keeps a hand-built matrix inside the same range.

Cost uses one train-split maximum over the four role columns (`cost_cheap` through `cost_frontier`). `cost_held_out` is excluded, so a model that is not yet an arm cannot inflate the scale. A non-positive maximum raises. The maximum is an argument to the episode. The stream does not recompute it.

\[
\tilde{c}(a) = \mathrm{clip}\!\left(\frac{c(a)}{\max_{\text{train}} c}, 0, 1\right), \qquad
r(x, a) = q(x, a) - \lambda \, \tilde{c}(a)
\]

`build_query_stream` slices the wide table in `split_ids` order, normalizes with that maximum, and stores rewards, normalized costs, and qualities on `QueryStream`. `include_held_out=True` appends the fifth model and still divides by the original maximum. That is the catalog-shift hook. The shift itself is a later task.

Contexts are not always stored in wide-table order, so the function lines them up in one of three ways:

- One row per wide-table row, as in `embeddings.npy`: take the rows for `split_ids`.
- One row per id in `split_ids`, already in that order, as in `context_test.npy`: use the matrix as it stands.
- Some other id order: pass that order as `split_reference_ids`, and the rows are gathered to match `split_ids`.

Anything else raises. A short debug episode can pass a prefix of the test ids together with the full test context and the full test id list as the reference.

## Oracle and regret

The oracle is \(\arg\max_a r(x, a)\). Near-ties use `np.isclose`. Among tied rewards, the cheaper stored cost wins. If those costs also match, the lower arm index wins. After clipping, two raw costs that both sat above the train maximum share normalized cost `1`, so the tie-break falls through to the index.

A per-step regret is the oracle reward minus the chosen reward. `EpisodeResult.cumulative_regrets` is the running sum. `regret(t) / t` is `cumulative_regrets[t - 1] / t`.

If any chosen reward exceeds the oracle by more than `1e-12`, the episode stops. That is the shared-run rule that no method may beat the oracle. Tiny negative gaps from float error are then clipped to zero.

## Episode

`run_episode(policy, stream, update)` accepts a `QueryStream`, a dict with the same arrays, or an iterator of `(context, reward_row, cost_row)`.

Each step calls `policy.act(x)` with that row of context only. The chosen reward is `rewards[t, a]` as a Python float. `update=True` calls `policy.update(x, a, r)` with that triple. `update=False` does not call `update`. The result records the chosen action and reward, the oracle action and reward, the per-step gaps, and the cumulative sum. Chosen quality and cost are filled when the stream carries them.

The episode calls `act` and, when learning, `update`. It does not call `propensity`. The concrete routers live in `src/policies.py`.
