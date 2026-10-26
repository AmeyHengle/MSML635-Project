# `src/policies.py`

Every router implements `act(x) -> int`, `update(x, a, r)`, and `propensity(x, a)`. The episode loop only calls `act` and `update`. Propensity is there for the later off-policy evaluation. A frozen policy leaves `update` empty, so the test stream cannot move its parameters.

Bandits learn the scalar reward of the arm they chose, \(r = q - \lambda\tilde{c}\). They never see the other arms' rewards. The supervised router is different: it is fit on train qualities, and \(\lambda\) is applied only when it picks an arm.

Arm indexes follow role order. Index 0 is cheap, 1 is mid, 2 is strong open, and 3 is frontier. The classes do not search the train table for the cheapest model. That choice is already the config's arm order.

## Frozen baselines

`AlwaysCheapPolicy` and `AlwaysFrontierPolicy` return a fixed index (0 and 3 by default). Propensity is 1 on that arm and 0 elsewhere.

`UniformRandomPolicy` draws with its own NumPy generator. Propensity is \(1/K\) on every arm, which is the behavior policy the off-policy evaluation will log. The draw does not depend on the context.

`OraclePolicy.act` raises. A policy only receives the context, and the oracle needs the full reward row. Scoring still goes through `find_oracle_action` in `src/sim.py`. This class exists so the oracle has the same three methods when a later runner lists policies.

## Supervised router and cascade

`SupervisedRidgePolicy.fit` trains one `Ridge` per arm, `fit_intercept=False`, `alpha` equal to the config's `ridge`. The intercept is already the column of ones on the context, so a second intercept would be unidentified. The target is quality, not reward. One set of models can be paired with any \(\lambda\).

`act` scores \(\hat{q}(x, a) - \lambda\tilde{c}(a)\) with the normalized costs stored on the policy. Those costs are the pre-shift table. Tied scores pick the cheaper stored cost, then the lower index. `update` does nothing. On a shift, this router keeps choosing with the old costs and the old quality fit. That freeze is what makes the price shift visible later.

`CascadePolicy` reuses those quality predictions. It walks cheap, then mid, then strong open, and stops at the first arm whose predicted quality is at least `tau` (0.5). If none qualify, it picks the last arm, frontier. It does not look at cost. It is frozen too.

## Open Bandit Pipeline wrappers

`LinUCBPolicy`, `LinTSPolicy`, and `LinEpsilonGreedyPolicy` construct the matching `obp.policy` class with `len_list=1`. `act` passes `context.reshape(1, -1)` and reads action index 0. `update` forwards the chosen arm and its reward to `update_params`. Defaults are dimension 65, 4 actions, and the config seeds.

OBP names LinUCB's exploration width `epsilon`. The wrapper passes the config's `alpha` into that argument. The config's `epsilon` (0.1) is a different knob: it belongs only to `LinEpsilonGreedy`. LinUCB's score inside OBP is \(\hat{\theta}^\top x + \alpha\sqrt{x^\top A^{-1} x}\).

`LinEpsilonGreedyPolicy.propensity` does not draw. It rebuilds \(\hat{\theta}\) from OBP's \(A^{-1}\) and \(b\), takes the greedy arm, and returns \((1-\varepsilon)+\varepsilon/K\) on that arm and \(\varepsilon/K\) on the others.

`LinUCBPolicy.propensity` returns 1 when `act` would pick that arm and 0 otherwise. LinUCB's choice is deterministic given its parameters. `LinTSPolicy.propensity` uses the same 1-or-0 rule, but `act` draws a new posterior sample, so the call is another Thompson draw rather than the probability of the action that was passed in.

## Discounted LinUCB

`DiscountedLinUCB` is the one policy that is not an OBP wrapper. The docstring cites Russac, Vernade, and Cappé (2019). Each arm starts at \(A = \mathrm{ridge}\, I\) and \(b = 0\). Only the chosen arm is updated:

\[
A \leftarrow \gamma A + xx^\top, \qquad b \leftarrow \gamma b + rx
\]

Selection is the usual linear UCB score, with `alpha` as the width. Tied scores take the lowest index. `add_arm` appends a fresh \((A, b)\) at the same dimension. The catalog shift can call that when the held-out model appears. The shift itself is not in this file. `gamma` defaults to 0.99. The stored random generator is unused. Selection does not draw.
