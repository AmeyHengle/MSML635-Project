# `src/data.py`

Turns the RouterBench pickle into one row per query, a frozen 70/30 split, and a context vector the bandit can score. Arms are read from the config. This module does not pick them.

## Wide pickle to long rows

The Hub file is wide: a performance column per model, plus `model|total_cost`. `load_long` melts that into `sample_id`, `prompt`, `eval_name`, `model_name`, `performance`, and `cost`, one row per query and model. A duplicate `(sample_id, model_name)` is an error.

`filter_evals` keeps a row when `eval_name` matches MMLU, GSM8K, MBPP, or MT-Bench. RouterBench spells GSM8K as `grade-school-math` and MT-Bench as `mtbench`, so the filter is a substring regex, not an exact name. Anything else, including the HellaSwag row in the unit test, is dropped.

## One row per query

`pivot_wide` expects the four config names in role order (`cheap`, `mid`, `strong_open`, `frontier`) and a held-out name that is not one of those four. Performance is clipped to `[0, 1]` here and stored as `quality_*`. Cost stays raw.

A query is dropped when any of the four arms lacks a finite quality or cost. The held-out columns may be missing. That model is only needed when a later shift adds a fifth arm, so it does not decide which queries enter the split.

## Split and context

`split_ids` holds out 30% of `sample_id` with `random_state=0`, stratified by eval family. MMLU subjects stay in the table, but they all count as the family `mmlu`. Stratifying on each subject would break on rare subjects and would not match the four evals the proposal compares. The two id sets are disjoint and cover every kept id. A family with fewer than two rows raises, because a stratified split cannot place it on both sides.

`embed_prompts` encodes prompts with MiniLM (384 dimensions) in wide-table order and caches `embeddings.npy`. The cache is reused only when the shape matches the current table. Embeddings are not L2-normalized. PCA is what compresses them.

`fit_contexts` fits PCA on the train rows only, with `random_state=0`, then appends a column of ones. The ones are the intercept, so a linear bandit of width 64 has dimension 65. Test rows are transformed with that same PCA and never enter the fit. `pca.n_samples_` must equal the train prompt count. Saved contexts follow split-id order, not wide-table order. `save_table` writes `wide.pkl` and `split.json` beside them.

The five experiment seeds do not redraw this split. Every method sees the same prompts and the same features.
