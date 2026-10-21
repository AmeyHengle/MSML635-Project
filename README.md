# Adaptive Model Selection for LLMs Using Contextual Bandits

Build guide for the MSML635 proposal ([PDF](../MSML635%20Project%20Proposal.pdf)). Check a box when that step is done. Check the matching box under [Progress](#progress) when the task’s last box is done. Anyone can take the next open task. No task is assigned to a named person.

Replay [RouterBench](https://huggingface.co/datasets/withmartian/routerbench). The simulator stores every model’s quality and cost, and the policy sees only the arm it chose. Results go in `results/` and `report/REPORT.md`. The notes in [`ideas/`](ideas/README.md) stay as they are.

The project is finished when the four boxes in [Done](#done) are checked. The demo is optional and sits outside that line.

## Progress

- [x] 1. Setup
- [x] 2. Data and features
- [ ] 3. Simulator and checks (i) and (ii)
- [ ] 4. Policies, sanity check (v), synthetic check (iii)
- [ ] 5. Stationary comparison, RQ1
- [ ] 6. Shifts, RQ2
- [ ] 7. Off-policy evaluation, RQ3, check (iv)
- [ ] 8. Figures, latency, and report
- [ ] 9. Optional demo

Suggested pace, still unassigned: tasks 1–3 by November 7, tasks 4–5 by November 21, tasks 6–7 by December 2, task 8 by December 11.

## Goal and reward

Route each query to one of four models and maximize quality minus a cost penalty.

Map RouterBench `performance` to quality in \([0, 1]\). If the printed range is already inside that interval, quality equals `performance`. Otherwise use `clip(performance, 0, 1)` and record that choice in `src/sim.py`.

Normalize cost with the **train-split** maximum over the four arms, then clip:

\[
\tilde{c}(a) = \mathrm{clip}\!\left(\frac{c(a)}{\max_{\text{train}} c}, 0, 1\right), \qquad
r(x, a) = q(x, a) - \lambda \, \tilde{c}(a)
\]

The oracle is \(a^\star = \arg\max_a r(x, a)\). Equal rewards break toward the cheaper arm. Regret is the sum of \(r(x_t, a^\star_t) - r(x_t, a_t)\).

The supervised model predicts **quality**. Multiply by \(\lambda\) only when choosing an arm, so one fit covers every \(\lambda\).

## Research questions

- **RQ1.** With full-information train labels and a stationary test stream, how close are the bandits to the supervised router?
- **RQ2.** After each scripted shift, do the bandits recover faster and accumulate less regret than the frozen supervised router?
- **RQ3.** Do IPS and DR, given logged propensities, match the policy value computed from the full reward matrix?

## Rules

These are the proposal. Changing them changes the question.

1. Four arms, chosen after printing the real `model_name` values: cheap open, mid-tier, strong open, frontier (the name that contains `GPT-4` when one exists). One more model is held out for shift S1.
2. Eval sets: MMLU, GSM8K, MBPP, and MT-Bench when that `eval_name` is present.
3. Split `sample_id` 70/30, stratified by eval family (mmlu, gsm8k, mbpp, mt-bench), seed 0. Fit PCA and the cost maximum on train prompts only. Cache embeddings to disk.
4. Context: `all-MiniLM-L6-v2`, PCA width 64 by default, plus a constant 1. Bandit dimension is 65. Widths 32 and 128 appear only in the small ablation in task 5.
5. Bandits are OBP `LinEpsilonGreedy`, `LinUCB`, and `LinTS` (`select_action`, `update_params`). The only custom policy is discounted LinUCB: on the chosen arm, \(A \leftarrow \gamma A + xx^\top\) and \(b \leftarrow \gamma b + rx\) (Russac et al., 2019).
6. The supervised router trains on the train split and stays frozen on the test stream, including its cost table and its arm set. Bandits start cold on the test stream and update from the chosen reward only. That freeze is what makes the price shift visible.
7. The OPE logger is uniform over the four arms, propensity \(1/4\). LinUCB is an evaluation policy.
8. Five seeds: `0, 1, 2, 3, 4`. They move policy randomness only. They do not redraw the split.

Main grid: four \(\lambda\) values \(\{0, 0.5, 1, 2\}\) at PCA width 64. Shifts and OPE use \(\lambda = 1\) and width 64. A full cross of every \(\lambda\), every width, and every method is out of the plan.

## Code layout

Five modules and three scripts. Add a function to the module that already owns the job.

```
configs/default.yaml
src/data.py        # download, wide table, split, embeddings, PCA
src/sim.py         # reward, oracle, episode loop, shifts
src/policies.py    # baselines, OBP wrappers, discounted LinUCB
src/metrics.py     # regret, APGR, CPT, recovery, routing errors, latency
src/ope.py         # uniform log, IPS, DR
scripts/prepare.py
scripts/run.py     # --exp rq1 | shifts | ope | all
scripts/plot.py
tests/test_core.py # checks (i)–(iv), no download
```

`data/` and `results/` are generated and gitignored. `pytest` must pass with no RouterBench file on disk.

```mermaid
flowchart LR
  Q[Query stream] --> F[MiniLM plus PCA]
  F --> P[Policy]
  P --> S[Simulator]
  S -->|chosen reward| P
  S --> E[Oracle, regret, OPE truth]
```

## Shared run rules

Tasks 5–8 use the same artifacts and the same config.

- [ ] One command, `python scripts/run.py --exp all`, rebuilds every reported number from `configs/default.yaml`.
- [ ] Score every method against the same oracle. Halt if any method’s reward on a query exceeds the oracle.
- [ ] Write `results/summary.csv` with one row per method, seed, \(\lambda\), PCA width, and scenario.
- [ ] Scalar intervals: average the five seeds’ per-query values, resample test queries 1000 times, and take the 2.5 and 97.5 percentiles. Regret and recovery plots show the mean across seeds.

---

## 1. Setup

- [x] Create the tree in [Code layout](#code-layout), plus `requirements.txt` and `.gitignore` (`.venv/`, `__pycache__/`, `*.pkl`, `data/`, `results/`).
- [x] Pin Python 3.11 and install `obp`, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `pyyaml`, `huggingface_hub`, `pytest`, `matplotlib`. Record `name==version` for each.
- [x] Paste the config below into `configs/default.yaml`.
- [x] `tests/test_core.py` starts with one test that imports `obp.policy.LinUCB`.
- [x] Done when `python -c "from obp.policy import LinUCB"` works and `pytest` passes.

```yaml
seed: 0
seeds: [0, 1, 2, 3, 4]
arms: []                       # task 2: four model_name strings, role order
arm_roles: [cheap, mid, strong_open, frontier]
held_out_arm: null
eval_names: []
mix_eval: null                 # MT-Bench, or the weakest cheap-model eval
embedding: sentence-transformers/all-MiniLM-L6-v2
pca_dim: 64
pca_dims_ablation: [32, 64, 128]
lambdas: [0.0, 0.5, 1.0, 2.0]
lambda_main: 1.0
epsilon: 0.1
alpha: 1.0
gamma: 0.99
ridge: 1.0
tau: 0.5
t0_fraction: 0.5
price_multiplier: 4.0
quality_scale: 0.5
mix_fraction: 0.8
recovery_window: 100
debug_rows: null               # set an int for a smoke run; reported runs leave this null
```

## 2. Data and features

`src/data.py`, `scripts/prepare.py`. Outputs under `data/processed/`: the wide table, `split.json`, cached embeddings, `pca.joblib`, and context matrices.

- [x] Download `routerbench_0shot.pkl` from `withmartian/routerbench` (~100 MB) with `huggingface_hub.hf_hub_download`.
- [x] The Hub pickle is wide (`model` and `model|total_cost`). `load_long` turns it into `sample_id`, `prompt`, `eval_name`, `model_name`, `performance`, `cost`.
- [x] Four arms and `held_out_arm` are read from the config, in role order. The loader requires exactly those four names. `mix_eval` is `mtbench`.
- [x] Keep rows whose `eval_name` matches MMLU, GSM8K (including `grade-school-math`), MBPP, or MT-Bench.
- [x] Pivot to one row per `sample_id` with quality and cost for each role and for the held-out model. Drop a query that is missing a finite value for any of the four arms.
- [x] Split ids 70/30, `random_state=0`, stratified by those four families. The two id sets are disjoint.
- [x] Embed prompts with MiniLM, save the arrays, fit PCA on train embeddings only, append a column of ones.
- [x] Done when a five-row print shows finite quality and cost for every arm, the id intersection is empty, and a test context has length 65. `pca.n_samples_` equals the train prompt count.

## 3. Simulator and checks (i) and (ii)

`src/sim.py`. An episode yields the context and, after `act`, the chosen reward. The full reward matrix stays inside the simulator for the oracle.

- [ ] Implement quality mapping, train-max normalization, \(r = q - \lambda\tilde{c}\), the oracle, and regret. The held-out arm’s cost stays out of the training maximum. Later shifts reuse that maximum.
- [ ] `run_episode(policy, stream, update)` calls `policy.act(x) -> int` and, when `update` is true, `policy.update(x, a, r)` with the chosen reward only.
- [ ] Check (i) in `tests/test_core.py`, with hand-written expected numbers. Qualities `(1.0, 0.0)`, normalized costs `(0.25, 1.0)`, \(\lambda = 1\) give rewards `(0.75, -1.0)` and regret `1.75` for arm 1. A raw cost of `1.5` with train max `1.0` normalizes to `1.0`. Tied rewards select the cheaper arm.
- [ ] Check (ii): on a fixed synthetic matrix of length 2000, every arm’s reward is at most the oracle reward. A uniform policy’s `regret(t)/t` at \(t = 2000\) and at \(t = 1000\) differ by less than 10% relative.
- [ ] Done when `pytest tests/test_core.py` passes offline.

## 4. Policies, sanity check (v), synthetic check (iii)

All of this lives in `src/policies.py`. Every policy implements `act`, `update`, and `propensity`. Frozen policies leave `update` empty.

- [ ] Always-cheap and always-expensive: lowest and highest train-set mean raw cost. Uniform: propensity `1/4`. Oracle: for scoring.
- [ ] Supervised: one `Ridge` per arm, `fit_intercept=False`, `alpha=ridge`, fit on train qualities. Choose \(\arg\max_a \hat{q}(x,a) - \lambda\tilde{c}(a)\) with the pre-shift cost.
- [ ] Cascade: walk cheap → mid → strong → frontier and stop at the first arm whose predicted quality is at least `tau` (0.5). Otherwise pick frontier.
- [ ] Wrap OBP `LinEpsilonGreedy` (this is the proposal’s \(\varepsilon\)-greedy), `LinUCB`, and `LinTS` with `dim=65`, `n_actions=4`, `len_list=1`. Pass `context.reshape(1, -1)` and read action index 0. `LinEpsilonGreedy` propensity is \((1-\varepsilon)+\varepsilon/K\) on the greedy arm and \(\varepsilon/K\) otherwise.
- [ ] Discounted LinUCB, cited in the docstring to Russac, Vernade, and Cappé (2019). Initialize \(A = \mathrm{ridge}\cdot I\), \(b = 0\). Update the chosen arm with \(A \leftarrow \gamma A + xx^\top\), \(b \leftarrow \gamma b + rx\). Select with the usual linear UCB score.
- [ ] RouteLLM, optional: the pretrained controller on the cheap-vs-frontier pair only, threshold chosen on the train split, then frozen. Follow [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM). If it is not running after one sitting, write the error in `results/sanity.md` and continue.
- [ ] Sanity check (v): on each locked eval, mean quality and mean raw cost of always-cheap and always-expensive. `results/sanity.md` places those pairs next to Hu et al. (2024): the expensive arm costs more and scores higher. If that ordering is reversed on a locked eval, fix the arm roles before task 5.
- [ ] Check (iii): synthetic linear bandit, \(d=10\), 4 arms, \(T=5000\), noise \(\mathcal{N}(0, 0.1^2)\), one seed. LinUCB’s `regret(T)/T` is below `regret(T/2)/(T/2)`. Uniform random’s two ratios differ by less than 10% relative. Discounted LinUCB runs on the same draws and returns actions in `{0,1,2,3}`.
- [ ] Run one short RouterBench episode (`debug_rows: 200`) with LinUCB and with discounted LinUCB, then set `debug_rows` back to null.
- [ ] Done when `pytest` passes, `results/sanity.md` records the ordering, and the smoke episode finishes.

## 5. Stationary comparison, RQ1

`scripts/run.py --exp rq1` appends to `results/summary.csv`. Bandits are new at each seed and learn only on the test stream. The supervised router and the cascade stay at their train-split parameters.

- [ ] Sweep `lambdas` at `pca_dim` 64 for always-cheap, always-expensive, uniform, oracle, supervised, cascade, LinEpsilonGreedy, LinUCB, LinTS, and discounted LinUCB. Include RouteLLM when task 4 loaded it.
- [ ] Ablation, separate and small: supervised and LinUCB only, \(\lambda = 1\), widths 32, 64, and 128, five seeds. Fit each PCA on train embeddings of that run.
- [ ] Record mean quality, mean normalized cost, cumulative regret, APGR, CPT(50%), unnecessary-escalation rate, and failed-downgrade rate. Definitions are in [Metrics](#metrics).
- [ ] Save per-step regret at \(\lambda = 1\), width 64, into `results/curves.npz` for the regret plot.
- [ ] Done when every method, seed, and \(\lambda\) in the main grid has a row, the ablation has its 2 × 3 × 5 rows, and no non-oracle mean reward exceeds the oracle.

## 6. Shifts, RQ2

`src/sim.py` applies one shift at \(t_0 = \lfloor 0.5\, T \rfloor\). `scripts/run.py --exp shifts`. Same episode loop as task 5. \(\lambda = 1\), width 64, five seeds.

Methods: frozen supervised, cascade, LinEpsilonGreedy, LinUCB, LinTS, discounted LinUCB. Plots in task 8 highlight the frozen supervised router and discounted LinUCB. The csv keeps the rest.

- [ ] **S1.** At \(t_0\), append `held_out_arm` with a fresh \((A, b)\). Normalize its cost with the original train maximum. Frozen methods keep the four-arm set.
- [ ] **S2.** Multiply the mid arm’s raw cost by `price_multiplier`, then normalize with the original train maximum and clip. The frozen router keeps choosing with the pre-shift \(\tilde{c}\). The reward uses the new \(\tilde{c}\). If clipping leaves the mid arm’s mean normalized cost nearly unchanged, multiply the cheap arm instead and write the substitution in `summary.csv`.
- [ ] **S3.** Before \(t_0\), play the test stream in order. After \(t_0\), draw queries so about `mix_fraction` (0.8) come from `mix_eval`.
- [ ] **S4.** After \(t_0\), multiply the strong-open arm’s quality by `quality_scale` and clip to \([0, 1]\). Frozen quality models keep their old predictions.
- [ ] Record cumulative regret and recovery time (see [Metrics](#metrics)). Save the per-step reward series for the supervised router and discounted LinUCB into `results/curves.npz`.
- [ ] Done when each of S1–S4 has rows for the frozen supervised router and discounted LinUCB, for every seed, with a recovery time and a `recovered` flag.

## 7. Off-policy evaluation, RQ3, check (iv)

`src/ope.py`. One \(\lambda\), the default. Uniform behavior, propensity `0.25` on every row. Store action, propensity, quality, and normalized cost so the reward can be rebuilt.

- [ ] Targets: the frozen supervised router, and the action LinUCB takes after its task-5 run. Ground truth is the mean of \(r(x, \pi(x))\) on the full reward matrix.
- [ ] Call OBP `InverseProbabilityWeighting` and `DoublyRobust`. `action_dist` has shape `(n, n_actions, 1)`. Fit the DR reward model on logged rows only.
- [ ] Check (iv) in `tests/test_core.py`: on synthetic uniform logs, the mean IPS estimate is within sampling error of the known policy value.
- [ ] Write `results/ope.csv` with bias and RMSE for IPS, DR, and IPS again with every propensity replaced by 1.
- [ ] Done when check (iv) passes and `ope.csv` has those three estimators for both targets.

## 8. Figures, latency, and report

- [ ] `scripts/plot.py` writes four figures under `results/figures/`: cost–quality frontier, RQ1 regret curves, one recovery panel per shift, and OPE error bars. Point estimates are means over the five seeds, with the intervals from [Shared run rules](#shared-run-rules).
- [ ] Latency in `src/metrics.py`: time `act` for 100 calls after 10 warmup calls. Record p50 and p99 in milliseconds for the frozen supervised router and for LinUCB. Time embedding plus PCA on the same prompts in a separate column.
- [ ] `results/numbers.md`: one page with the frontier, APGR, CPT(50%), regret, recovery, routing-error rates, OPE errors, the PCA ablation, and latency. Task 8’s report quotes this page.
- [ ] `report/REPORT.md` answers RQ1, RQ2, and RQ3 from `numbers.md`, and states the three proposal risks: the supervised router can win while labels are free; RouterBench prices are from 2024, so compare methods relatively; linear models can degrade as width grows, which the 32/64/128 ablation checks.
- [ ] Add a Reproduce section: dataset `withmartian/routerbench`, file `routerbench_0shot.pkl`, Python 3.11, and `python scripts/run.py --exp all`.
- [ ] Draft `report/slides.md` from the same page: problem, reward, pipeline, RQ1, RQ2, RQ3, risks.
- [ ] Done when the figures, `numbers.md`, the report, and the slide outline exist, and `pytest` still passes.

## 9. Optional demo

Start only after `report/REPORT.md` has a draft. Skip this if December 11 is close. The [Done](#done) boxes do not include it.

- [ ] `demo/app.py` with FastAPI. `POST /route` embeds the prompt with the saved PCA, calls the frozen supervised router, and returns the arm, its mean train cost, the running cost, and the running cost of always using the frontier.
- [ ] The process calls no model API. Counters reset on restart.
- [ ] Done when a local request returns one of the four arms and both totals.

---

## Metrics

Implement these once, in `src/metrics.py`.

| Metric | Definition |
|---|---|
| Quality, cost | Mean quality and mean normalized cost on the test stream. |
| Regret | Sum of oracle gaps, above. |
| PGR | \((r - r_{\text{cheap}}) / (r_{\text{frontier}} - r_{\text{cheap}})\), using mean quality. |
| Call rate \(c\) | Fraction of queries sent to the frontier. |
| APGR | Sweep \(\lambda\). Interpolate PGR onto call rates \(0.05, 0.15, \ldots, 0.95\). APGR is the mean of those ten values (Ong et al., 2025). Leave APGR blank for a method whose call rate cannot move. |
| CPT(50%) | Smallest frontier call rate on that curve whose PGR is at least 0.5. Blank when the curve stays below 0.5. |
| Recovery | Smallest time \(\ge t_0 + W\) at which the mean reward of the last \(W=100\) steps is within 5% of that method’s pre-shift mean. If it never happens, `recovered=false` and the time is the end of the stream. |
| Unnecessary escalation | A strictly cheaper arm matches or beats the chosen arm’s quality. |
| Failed downgrade | A strictly costlier arm beats the chosen arm’s quality. |
| OPE bias, RMSE | Estimate minus ground truth, across seeds. |
| Latency | p50 and p99 of `act`, plus embedding time in its own column. |

## Done

- [ ] `pytest` passes, including checks (i), (ii), (iii), and (iv).
- [ ] `results/sanity.md` records check (v), and `results/numbers.md` plus the four figures exist.
- [ ] `report/REPORT.md` answers RQ1–RQ3, states the three risks, and gives the download path and `python scripts/run.py --exp all`.
- [ ] `report/slides.md` exists.

## Scope

This repo is a RouterBench replay with OBP bandits, four scripted shifts, and an OPE check against simulator ground truth. Training an LLM and calling a live model API are outside the proposal.

## References

- Hu et al. (2024). RouterBench. [arXiv:2403.12031](https://arxiv.org/abs/2403.12031). Data: [withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench).
- Saito et al. (2021). Open Bandit Pipeline. [github.com/st-tech/zr-obp](https://github.com/st-tech/zr-obp).
- Ong et al. (2025). RouteLLM. ICLR 2025. [arXiv:2406.18665](https://arxiv.org/abs/2406.18665). APGR and CPT.
- Li et al. (2010). LinUCB. Agrawal and Goyal (2013). Linear Thompson sampling. Russac et al. (2019). Discounted linear bandits.
- Dudík, Langford, and Li (2011). Doubly robust evaluation. Chen, Zaharia, and Zou (2023). FrugalGPT cascade.
