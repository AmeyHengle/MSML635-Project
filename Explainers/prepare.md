# `scripts/prepare.py`

One pass from `configs/default.yaml` to the files under `data/processed/`. Those files are gitignored. Later runs read them instead of downloading and embedding again.

The script inserts the repo root on `sys.path`, loads the config, and requires exactly four distinct arm names. It then:

1. Downloads `routerbench_0shot.pkl` into `data/raw` with `huggingface_hub`.
2. Converts it to long rows, keeps the four eval families, and pivots with the config arms and `held_out_arm`.
3. Splits ids 70/30 and writes `wide.pkl` and `split.json`.
4. Embeds every kept prompt, fits PCA on the train rows, and writes `embeddings.npy`, `pca.joblib`, `context_train.npy`, and `context_test.npy`.

The printed check is the step 2 exit condition: a five-row preview with finite quality and cost, an empty train/test id intersection, a test context of length `pca_dim + 1` (65 at the default), and `pca.n_samples_` equal to the train count.

The script does not build rewards or call a policy. λ and the train-split cost maximum are applied later, in `src/sim.py`, so the same cached table can be scored at every λ.
