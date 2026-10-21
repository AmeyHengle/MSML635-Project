"""Load configs/default.yaml, download RouterBench, and print the inventory."""

import sys
from pathlib import Path

import yaml

# Add the root of the project to the Python path so that we can import src.data
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # import src.data

from src.data import download_routerbench, load_long, print_inventory


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    # Load the config file
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Check that the config is a mapping
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    # Extract the raw data path from the config
    raw_path = download_routerbench()   # Download RouterBench and return the path to the raw data
    frame = load_long(raw_path)  # Load the raw data into a long-format DataFrame
    print_inventory(frame)  # Print the inventory of the data


if __name__ == "__main__":
    main()
