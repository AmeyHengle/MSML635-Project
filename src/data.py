"""Download RouterBench and build the wide table, split, and contexts.

Arms, the held-out model, and mix_eval are read from the config. This module
does not choose them. Evals are kept by substring: MMLU, GSM8K (RouterBench
calls it grade-school-math), MBPP, and MT-Bench.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ROUTERBENCH_REPO = "withmartian/routerbench"
ROUTERBENCH_FILENAME = "routerbench_0shot.pkl"
ROLES = ("cheap", "mid", "strong_open", "frontier")
# RouterBench names GSM8K "grade-school-math" and MT-Bench "mtbench".
EVAL_RE = "mmlu|gsm8k|gsm-8k|grade-school-math|mbpp|mt-?bench"
EMBED_DIM = 384  # all-MiniLM-L6-v2
LONG_COLUMNS = ["sample_id", "prompt", "eval_name", "model_name", "performance", "cost"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def processed_dir() -> Path:
    path = repo_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_routerbench() -> Path:
    """Download routerbench_0shot.pkl into data/raw."""
    from huggingface_hub import hf_hub_download

    destination = repo_root() / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=ROUTERBENCH_REPO,
            filename=ROUTERBENCH_FILENAME,
            repo_type="dataset",
            local_dir=destination,
        )
    )


def load_long(path: Path | str) -> pd.DataFrame:
    """One row per (sample_id, model_name). The Hub pickle is wide."""
    loaded = pd.read_pickle(path)
    # If the Hub pickle is wide, convert it to long. The wide table has columns like
    # "open-cheap|total_cost" and "open-cheap|model_response".
    frame = loaded if isinstance(loaded, pd.DataFrame) else pd.DataFrame(loaded)
    if any(str(column).endswith("|total_cost") for column in frame.columns):
        frame = _wide_to_long(frame)
    missing = [name for name in LONG_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {list(map(str, frame.columns))}")
    long = frame.loc[:, LONG_COLUMNS].copy()
    if long.duplicated(["sample_id", "model_name"]).any():
        raise ValueError("expected one row per (sample_id, model_name)")
    return long


def filter_evals(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep MMLU, GSM8K, MBPP, and MT-Bench rows."""
    mask = frame["eval_name"].astype(str).str.lower().str.contains(EVAL_RE, regex=True)
    kept = frame.loc[mask].copy()
    if kept.empty:
        raise ValueError("no rows matched MMLU, GSM8K, MBPP, or MT-Bench")
    return kept


def pivot_wide(frame: pd.DataFrame, arms: list[str], held_out_arm: str) -> pd.DataFrame:
    """One row per sample_id. Drop a query missing a finite value for any of the four arms."""
    names = [str(name) for name in arms]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError(f"arms must be four names in role order, found {arms}")
    held = str(held_out_arm)
    if held in names:
        raise ValueError(f"held_out_arm {held} is already one of the four arms")

    # Map the four arms to roles and the held-out arm to "held_out". The wide table has columns like
    # "quality_cheap", "cost_mid", "quality_strong_open", "cost_frontier", and "quality_held_out", "cost_held_out".
    roles = dict(zip(names, ROLES))
    roles[held] = "held_out" # type: ignore
    work = frame.copy()
    work["model_name"] = work["model_name"].astype(str)
    work["quality"] = pd.to_numeric(work["performance"], errors="coerce").clip(0, 1)
    work["cost"] = pd.to_numeric(work["cost"], errors="coerce")
    base = work.groupby("sample_id", sort=False)[["prompt", "eval_name"]].first()
    part = work.loc[work["model_name"].isin(roles)].copy()
    part["role"] = part["model_name"].map(roles)
    quality = part.pivot(index="sample_id", columns="role", values="quality").add_prefix("quality_")
    cost = part.pivot(index="sample_id", columns="role", values="cost").add_prefix("cost_")
    wide = base.join(quality).join(cost).reset_index()

    # Keep only the columns we expect, in a consistent order. Drop any row missing a finite value for any of the four arms.
    columns = ["sample_id", "prompt", "eval_name"]
    for role in list(ROLES) + ["held_out"]:
        columns += [f"quality_{role}", f"cost_{role}"]
    for column in columns:
        if column not in wide.columns:
            wide[column] = np.nan
    wide = wide.loc[:, columns]
    arm_columns = [f"{kind}_{role}" for role in ROLES for kind in ("quality", "cost")]
    keep = np.isfinite(wide.loc[:, arm_columns].to_numpy(dtype=float)).all(axis=1)
    return wide.loc[keep].reset_index(drop=True) # type: ignore


def split_ids(wide: pd.DataFrame, test_size: float = 0.3, random_state: int = 0) -> dict[str, list[str]]:
    """70/30 split of sample_id, stratified by eval family. The two sets are disjoint."""
    # Map eval_name to one of four families: mmlu, gsm8k, mbpp, or mt-bench. Raise if any eval_name is not one of those four.
    labels = wide["eval_name"].map(_family)
    counts = labels.value_counts()
    if counts.min() < 2:
        raise ValueError(f"cannot stratify; a family has fewer than two rows: {counts.to_dict()}")
    ids = wide["sample_id"].astype(str).tolist()
    # split
    train, test = train_test_split(ids, test_size=test_size, random_state=random_state, stratify=labels.tolist())
    train, test = list(train), list(test)
    if set(train) & set(test) or set(train) | set(test) != set(ids):
        raise ValueError("train and test id sets must be disjoint and cover every kept id")
    return {"train": train, "test": test}


def embed_prompts(wide: pd.DataFrame, model_name: str, directory: Path | None = None) -> np.ndarray:
    """384-d MiniLM vectors in wide-table row order. Reuse embeddings.npy when the shape matches."""
    directory = processed_dir() if directory is None else Path(directory)   # default to data/processed if no directory is provided
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "embeddings.npy"
    expected = (len(wide), EMBED_DIM)
    if path.exists():
        cached = np.load(path)
        if cached.shape == expected:
            return cached
    from sentence_transformers import SentenceTransformer

    # Encode prompts with the specified model. Use batch_size=64 and show a progress bar. 
    # Convert to numpy array and ensure it's contiguous and of type float32. 
    # Save the embeddings to embeddings.npy in the specified directory.
    vectors = np.asarray(
        SentenceTransformer(model_name).encode(
            wide["prompt"].astype(str).tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.shape != expected:
        raise ValueError(f"expected embeddings shape {expected}, got {vectors.shape}")
    np.save(path, vectors)
    return vectors


def fit_contexts(
    embeddings: np.ndarray,
    wide: pd.DataFrame,
    split: dict[str, list[str]],
    pca_dim: int,
    directory: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on train rows only, append a column of ones, and save both matrices."""
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[0] != len(wide):
        raise ValueError(f"embeddings must have one row per prompt, found {matrix.shape} for {len(wide)} rows")
    positions = _positions(wide)    # Create a mapping of sample_id to its index in the wide DataFrame for easy access to the corresponding rows in the embeddings matrix.
    # get indices for train and test sample_ids based on the split dictionary. 
    train_index = np.array([positions[sample_id] for sample_id in split["train"]])
    test_index = np.array([positions[sample_id] for sample_id in split["test"]])
    # init PCA with the specified number of components and a fixed random state for reproducibility.
    pca = PCA(n_components=pca_dim, random_state=0) 
    # Fit PCA on the training embeddings and transform both the training and test embeddings.
    context_train = _with_ones(pca.fit_transform(np.asarray(matrix[train_index], dtype=np.float64)))
    context_test = _with_ones(pca.transform(np.asarray(matrix[test_index], dtype=np.float64)))
    
    if int(pca.n_samples_) != len(train_index):
        raise ValueError(f"pca.n_samples_={pca.n_samples_} does not equal the train count {len(train_index)}")
    directory = processed_dir() if directory is None else Path(directory)   # default to data/processed if no directory is provided
    directory.mkdir(parents=True, exist_ok=True)
    # Save the fitted PCA model to pca.joblib in the specified directory for later use.
    joblib.dump(pca, directory / "pca.joblib")
    np.save(directory / "context_train.npy", context_train)
    np.save(directory / "context_test.npy", context_test)
    return context_train, context_test, pca


def save_table(wide: pd.DataFrame, split: dict[str, list[str]], directory: Path | None = None) -> None:
    directory = processed_dir() if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wide.to_pickle(directory / "wide.pkl")
    (directory / "split.json").write_text(json.dumps(split) + "\n", encoding="utf-8")


def _wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide table to long format. The wide table has columns like"""
    models = [str(column)[: -len("|total_cost")] for column in frame.columns if str(column).endswith("|total_cost")]
    missing_models = [model for model in models if model not in frame.columns]
    if not models or missing_models:
        raise ValueError(f"cost columns without performance columns: {missing_models or 'none found'}")
    performance = frame.melt(
        id_vars=["sample_id", "prompt", "eval_name"],
        value_vars=models,
        var_name="model_name",
        value_name="performance",
    )
    cost = frame.melt(
        id_vars=["sample_id"],
        value_vars=[f"{model}|total_cost" for model in models],
        var_name="cost_column",
        value_name="cost",
    )
    cost["model_name"] = cost["cost_column"].str.removesuffix("|total_cost")
    return performance.merge(cost[["sample_id", "model_name", "cost"]], on=["sample_id", "model_name"])


def _family(name: object) -> str:
    """ Map an eval_name to one of four families: mmlu, gsm8k, mbpp, or mt-bench. Raise if the name is not one of those four."""
    text = str(name).lower()
    if "mmlu" in text:
        return "mmlu"
    if "gsm" in text or "grade-school-math" in text or "grade school math" in text:
        return "gsm8k"
    if "mbpp" in text:
        return "mbpp"
    if "mt" in text.replace("-", "") and "bench" in text.replace("-", ""):
        return "mt-bench"
    raise ValueError(f"eval_name {name} is not mmlu, gsm8k, mbpp, or mt-bench")


def _positions(wide: pd.DataFrame) -> dict[str, int]:
    """ Create a mapping of sample_id to its index in the wide DataFrame for easy access to the corresponding rows in the embeddings matrix."""
    ids = wide["sample_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("wide table sample_id values are not unique")
    return {sample_id: index for index, sample_id in enumerate(ids.tolist())}


def _with_ones(transformed: np.ndarray) -> np.ndarray:
    """Append a column of ones to the transformed embeddings for bias term in linear models."""
    ones = np.ones((transformed.shape[0], 1), dtype=transformed.dtype)
    return np.concatenate([transformed, ones], axis=1)
