"""Download, wide table, split, embeddings, PCA."""

from pathlib import Path

import numpy as np
import pandas as pd


ROUTERBENCH_REPO = "withmartian/routerbench"    # Hugging Face dataset repo for RouterBench
ROUTERBENCH_FILENAME = "routerbench_0shot.pkl"  # Hugging Face dataset filename for RouterBench

# The long schema for RouterBench, as described in the paper. 
# The Hub pickle is wide: one performance column per model, plus ``{model}|total_cost``.
LONG_COLUMNS = (
    "sample_id",
    "prompt",
    "eval_name",
    "model_name",
    "performance",
    "cost",
)

# Paper schema already uses LONG_COLUMNS. These aliases apply only when a
# required header is missing.
_COLUMN_ALIASES = {
    "sample_id": ("sample_id", "sample id"),
    "prompt": ("prompt", "query"),
    "eval_name": ("eval_name", "eval name"),
    "model_name": ("model_name", "model name", "model"),
    "performance": ("performance", "score"),
    "cost": ("cost", "total_cost"),
}


def repo_root() -> Path:
    """Returns the Repo Root directory, which is the parent of the src directory."""
    return Path(__file__).resolve().parents[1]


def raw_data_dir() -> Path:
    """Returns the path to the raw data directory, which is data/raw."""
    path = repo_root() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_routerbench(raw_dir: Path | None = None) -> Path:
    """Download routerbench_0shot.pkl into data/raw and return its path."""
    from huggingface_hub import hf_hub_download

    destination = raw_data_dir() if raw_dir is None else Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # Download the file from Hugging Face Hub into the destination directory
    downloaded = hf_hub_download(
        repo_id=ROUTERBENCH_REPO,
        filename=ROUTERBENCH_FILENAME,
        repo_type="dataset",
        local_dir=destination,
    )
    return Path(downloaded)


def load_long(path: Path | str) -> pd.DataFrame:
    """Read the pickle and return one row per (sample_id, model_name).

    The paper lists a long schema. The Hub pickle is wide: one performance
    column per model, plus ``{model}|total_cost``. Both layouts become
    ``LONG_COLUMNS``.
    """
    loaded = pd.read_pickle(path)
    # Convert to DataFrame if not already
    frame = loaded if isinstance(loaded, pd.DataFrame) else pd.DataFrame(loaded)
    # Check that the DataFrame has the required columns, and convert to long format if necessary
    if _looks_wide(frame) and not _has_long_columns(frame):
        frame = _rename_columns(frame, columns=("sample_id", "prompt", "eval_name"))
        long = _wide_to_long(frame)
    else:
        frame = _rename_columns(frame)
        missing = [name for name in LONG_COLUMNS if name not in frame.columns]  # Check for missing columns
        if missing:   # Raise an error if any required columns are missing
            found = [str(column) for column in frame.columns]
            raise ValueError(
                f"RouterBench file is missing columns {missing}; found {found}"
            )
        long = frame.loc[:, list(LONG_COLUMNS)].copy()  # Select only the required columns
    # Check for duplicate rows based on (sample_id, model_name)
    duplicate_count = int(long.duplicated(["sample_id", "model_name"]).sum())   
    assert duplicate_count == 0, (
        "expected one row per (sample_id, model_name), "
        f"found {duplicate_count} duplicate rows"
    )
    return long


def print_inventory(frame: pd.DataFrame) -> None:
    """Print columns, models, evals, and per-model cost and performance."""
    
    # Print the columns of the DataFrame, the unique model names, and the unique evaluation names
    print("columns:", ", ".join(str(column) for column in frame.columns))
    print("model_name:")
    for name in _sorted_labels(frame["model_name"]):
        print(f"  {name}")
    print("eval_name:")
    for name in _sorted_labels(frame["eval_name"]):
        print(f"  {name}")
        
    # Convert cost and performance to numeric, coercing errors to NaN
    numeric = pd.DataFrame(
        {
            "model_name": frame["model_name"],
            "cost": pd.to_numeric(frame["cost"], errors="coerce"),
            "performance": _numeric_performance(frame["performance"]),
        }
    )
    # Group by model_name and calculate min, max, and mean for cost and performance
    stats = numeric.groupby("model_name", sort=True)[["cost", "performance"]].agg(
        ["min", "max", "mean"]
    )
    print("per-model cost and performance (min, max, mean):")
    print(stats.to_string())


def _has_long_columns(frame: pd.DataFrame) -> bool:
    """ Check if the DataFrame has all the required long columns."""
    return all(name in frame.columns for name in LONG_COLUMNS)


def _looks_wide(frame: pd.DataFrame) -> bool:
    """ Check if the DataFrame looks like a wide-format RouterBench file"""
    return any(str(column).endswith("|total_cost") for column in frame.columns)


def _wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide-format RouterBench DataFrame to long format."""
    missing_ids = [ # Check for missing required columns in the wide DataFrame
        name for name in ("sample_id", "prompt", "eval_name") if name not in frame.columns
    ]
    if missing_ids: # Raise an error if any required columns are missing
        found = [str(column) for column in frame.columns]
        raise ValueError(
            f"RouterBench file is missing columns {missing_ids}; found {found}"
        )
    models = [  # Extract model names from the wide DataFrame by removing the "|total_cost" suffix
        str(column)[: -len("|total_cost")]
        for column in frame.columns
        if str(column).endswith("|total_cost")
    ]
    
    # Check for models that have cost columns but no corresponding performance columns
    missing_performance = [model for model in models if model not in frame.columns]
    if missing_performance:
        raise ValueError(
            "RouterBench file has cost columns without performance columns: "
            f"{missing_performance}"
        )
        
    # Check for models that have performance columns but no corresponding cost columns
    if not models:
        found = [str(column) for column in frame.columns]
        missing = [name for name in LONG_COLUMNS if name not in frame.columns]
        raise ValueError(
            f"RouterBench file is missing columns {missing}; found {found}"
        )
        
    # Drop the model response columns (those ending with "|model_response") 
    # melt the DataFrame to long format for performance and cost
    response_columns = [
        column for column in frame.columns if str(column).endswith("|model_response")
    ]
    slim = frame.drop(columns=response_columns)
    performance = slim.melt(
        id_vars=["sample_id", "prompt", "eval_name"],
        value_vars=models,
        var_name="model_name",
        value_name="performance",
    )
    cost = slim.melt(
        id_vars=["sample_id"],
        value_vars=[f"{model}|total_cost" for model in models],
        var_name="cost_column",
        value_name="cost",
    )
    # Add a model_name column to the cost DataFrame by removing the "|total_cost" suffix from the cost_column
    cost["model_name"] = cost["cost_column"].str.removesuffix("|total_cost")
    long = performance.merge(   # Merge the performance and cost DataFrames on sample_id and model_name
        cost[["sample_id", "model_name", "cost"]],
        on=["sample_id", "model_name"],
        how="left",
        validate="one_to_one",
    )
    print(
        f"normalized {len(models)} wide model columns into "
        "model_name, performance, and cost"
    )
    return long.loc[:, list(LONG_COLUMNS)].copy()


def _rename_columns(
    frame: pd.DataFrame, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Rename columns in the DataFrame to their canonical names."""
    aliases = (
        _COLUMN_ALIASES
        if columns is None
        else {name: _COLUMN_ALIASES[name] for name in columns}
    )
    lower_to_original: dict[str, object] = {}
    for column in frame.columns:
        lower_to_original.setdefault(str(column).strip().lower(), column)
    rename: dict[object, str] = {}
    for canonical, names in aliases.items():
        if canonical in frame.columns:
            continue
        for alias in names:
            original = lower_to_original.get(alias.strip().lower())
            if original is None or original in rename:
                continue
            rename[original] = canonical
            break
    if not rename:
        return frame
    renamed = ", ".join(f"{source} -> {target}" for source, target in rename.items())
    print(f"renamed columns: {renamed}")
    return frame.rename(columns=rename)


def _numeric_performance(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() or series.isna().all():
        return numeric
    values = series.dropna()
    if values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        return series.map(
            lambda value: float(value) if isinstance(value, (bool, np.bool_)) else np.nan
        )
    return numeric


def _sorted_labels(series: pd.Series) -> list[object]:
    """Return a sorted list of unique non-null values from a Series."""
    labels = [label for label in series.dropna().unique().tolist()]
    return sorted(labels, key=str)
