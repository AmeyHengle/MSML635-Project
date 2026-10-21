"""Read configs/default.yaml and write the wide table, split, embeddings, and PCA contexts."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    download_routerbench,
    embed_prompts,
    filter_evals,
    fit_contexts,
    load_long,
    pivot_wide,
    save_table,
    split_ids,
)


def main() -> None:
    # Read the default configuration file to get the list of arms and the held-out arm.
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    arms = [str(name) for name in config["arms"]]
    held_out = str(config["held_out_arm"])
    if len(arms) != 4 or len(set(arms)) != 4:
        raise ValueError(f"config arms must be four model names, found {config['arms']}")

    # Load the RouterBench data, filter evaluations, pivot to wide format, and split into training and test sets.
    wide = pivot_wide(filter_evals(load_long(download_routerbench())), arms, held_out)
    split = split_ids(wide)
    save_table(wide, split)
    _train, test, pca = fit_contexts(
        embed_prompts(wide, str(config["embedding"])),
        wide,
        split,
        int(config["pca_dim"]),
    )
    print(f"rows: {len(wide)}  train: {len(split['train'])}  test: {len(split['test'])}")
    print(f"train/test intersection: {set(split['train']) & set(split['test'])}")
    print(wide.head().to_string(index=False))
    print(f"test context length: {test.shape[1]}")
    print(f"pca.n_samples_={pca.n_samples_} train count={len(split['train'])}")


if __name__ == "__main__":
    main()
