# `configs/default.yaml`

Shared settings for every later run. The data loader and the simulator read this file. They do not choose arms or hyperparameters themselves.

## Locked arms

Four models, in role order. A fifth model is held out for the later price-and-catalog shifts. It is not an arm in the stationary comparison.

| Role | `model_name` |
|---|---|
| cheap | `mistralai/mistral-7b-chat` |
| mid | `mistralai/mixtral-8x7b-chat` |
| strong_open | `zero-one-ai/Yi-34B-Chat` |
| frontier | `gpt-4-1106-preview` |
| held out | `claude-v1` |

`mix_eval` is `mtbench`. The shift that remixes the query stream will draw from that family.

## What is fixed now, and what later tasks use

- `embedding` is `sentence-transformers/all-MiniLM-L6-v2`. `pca_dim` is 64. With the appended column of ones, each context has length 65. Widths 32 and 128 are only for the later ablation.
- `lambdas` is `0, 0.5, 1, 2`. Shifts and off-policy evaluation use `lambda_main` (`1.0`). One quality model can serve every λ, because λ is applied when an arm is chosen, not when quality is fit.
- `seeds` is `0, 1, 2, 3, 4`. Those seeds move policy randomness only. The 70/30 split stays at seed 0.
- `epsilon` (0.1) is the explore rate for linear \(\varepsilon\)-greedy. `alpha` (1.0) is the LinUCB width. Open Bandit Pipeline calls that width `epsilon`, and `LinUCBPolicy` passes `alpha` into it. `gamma` (0.99) discounts `DiscountedLinUCB`. `ridge` (1.0) is the supervised penalty and the discounted bandit's initial \(A\). `tau` (0.5) is the cascade's quality threshold. The policy classes take these as arguments. They do not open this file.
- `debug_rows` stays `null` for reported runs. A smoke episode can set it to a small integer and then put it back. `scripts/run.py` does not read it yet.
