"""Load configs/default.yaml and write the wide table, split, embeddings, and PCA contexts."""

import sys
from pathlib import Path

import yaml

# Add the root of the project to the Python path so that we can import src.data
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # import src.data

from src.data import (
    download_routerbench,
    embed_prompts,
    fit_pca_contexts,
    load_long,
    lock_evals_and_arms,
    pivot_wide,
    print_context_check,
    print_inventory,
    print_wide_check,
    require_locked_arms,
    save_wide_and_split,
    split_ids,
)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    # Load the config file
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Check that the config is a mapping
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    # Inventory prints before any arm is written.
    raw_path = download_routerbench()   # Download RouterBench and return the path to the raw data
    frame = load_long(raw_path)  # Load the raw data into a long-format DataFrame
    print_inventory(frame)  # Print the inventory of the data
    locked = lock_evals_and_arms(frame, config_path)
    with config_path.open(encoding="utf-8") as handle:
        written = yaml.safe_load(handle)
    arms = require_locked_arms(written.get("arms"), locked["arms"])
    held_out = written.get("held_out_arm")
    eval_names = written.get("eval_names")
    if not isinstance(held_out, str) or not held_out:
        raise ValueError("held_out_arm must be a model name before the wide table is built")
    if not isinstance(eval_names, list) or not eval_names:
        raise ValueError("eval_names must be locked before the wide table is built")
    kept = frame.loc[frame["eval_name"].astype(str).isin({str(name) for name in eval_names})].copy()
    wide = pivot_wide(kept, arms, held_out, sim_path=ROOT / "src" / "sim.py")
    split = split_ids(wide)
    save_wide_and_split(wide, split)
    model_name = config.get("embedding")
    pca_dim = config.get("pca_dim")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("config embedding must be a model name")
    if type(pca_dim) is not int or pca_dim < 1:
        raise ValueError(f"config pca_dim must be a positive int, found {pca_dim!r}")
    embeddings = embed_prompts(wide, model_name)
    _context_train, context_test, pca = fit_pca_contexts(embeddings, wide, split, pca_dim)
    print_wide_check(wide, split)
    print_context_check(context_test, pca, len(split["train"]))


if __name__ == "__main__":
    main()
