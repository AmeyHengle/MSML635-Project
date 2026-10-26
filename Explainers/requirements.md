# `requirements.txt`

Step 1 pins the environment so the LinUCB import check and the later Open Bandit Pipeline wrappers see the same library.

Python is 3.11 (`.python-version`). The pins that this stage actually uses:

- `obp==0.4.1` supplies `LinUCB`, `LinTS`, and `LinEpsilonGreedy`. `src/policies.py` wraps those three. Discounted LinUCB is written in that file and does not come from the package.
- `huggingface_hub` downloads `routerbench_0shot.pkl` from `withmartian/routerbench`.
- `sentence-transformers` encodes prompts with MiniLM. `scikit-learn` supplies the stratified split, PCA, and the per-arm `Ridge` quality models.
- `pandas`, `numpy`, `joblib`, and `pyyaml` carry the table, the cached arrays, and `configs/default.yaml`.
- `pytest` runs the offline checks. `matplotlib` is installed for the figure task and is unused so far.

`data/`, `results/`, `*.pkl`, the virtualenv, and `__pycache__` are gitignored. A fresh clone has no RouterBench file, and `pytest` still has to pass.
