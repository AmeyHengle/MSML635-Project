"""Download RouterBench, lock arms, and build the wide table, split, and contexts."""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


ROUTERBENCH_REPO = "withmartian/routerbench"
ROUTERBENCH_FILENAME = "routerbench_0shot.pkl"
LONG_COLUMNS = ("sample_id", "prompt", "eval_name", "model_name", "performance", "cost")
_ALIASES = {
    "sample_id": ("sample_id", "sample id"),
    "prompt": ("prompt", "query"),
    "eval_name": ("eval_name", "eval name"),
    "model_name": ("model_name", "model name", "model"),
    "performance": ("performance", "score"),
    "cost": ("cost", "total_cost"),
}
# RouterBench calls GSM8K "grade-school-math".
_EVAL_RE = "mmlu|gsm8k|gsm-8k|grade-school-math|grade school math|mbpp|mt-?bench"
_CLOSED = ("gpt", "claude", "gemini", "sonar", "you.com")
_ROLES = ("cheap", "mid", "strong_open", "frontier")
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
_MID_CLOSE = 0.05  # costs this close, relative to the cheap/strong gap, count as a tie
_QUALITY_NOTE = (
    "# Quality is clip(performance, 0, 1) because the printed performance range was outside [0, 1]."
)
_WIDE_COLUMNS = ["sample_id", "prompt", "eval_name"] + [
    f"{kind}_{role}" for role in list(_ROLES) + ["held_out"] for kind in ("quality", "cost")
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_data_dir() -> Path:
    path = repo_root() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_data_dir() -> Path:
    path = repo_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_routerbench(raw_dir: Path | None = None) -> Path:
    """Download routerbench_0shot.pkl into data/raw."""
    from huggingface_hub import hf_hub_download

    destination = raw_data_dir() if raw_dir is None else Path(raw_dir)
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
    """One row per (sample_id, model_name). The Hub pickle is wide; the paper schema is long."""
    loaded = pd.read_pickle(path)
    frame = loaded if isinstance(loaded, pd.DataFrame) else pd.DataFrame(loaded)
    # Check for wide columns and normalize to long format if needed. 
    # Then rename columns and check for missing LONG_COLUMNS.
    if _looks_wide(frame) and not set(LONG_COLUMNS).issubset(frame.columns):
        long = _wide_to_long(_rename_columns(frame, ("sample_id", "prompt", "eval_name")))
    else:
        frame = _rename_columns(frame)
        missing = [name for name in LONG_COLUMNS if name not in frame.columns]
        if missing:
            found = list(map(str, frame.columns))
            raise ValueError(f"RouterBench file is missing columns {missing}; found {found}")
        long = frame.loc[:, list(LONG_COLUMNS)].copy()
    duplicates = int(long.duplicated(["sample_id", "model_name"]).sum())
    # Check that there is exactly one row per (sample_id, model_name) pair.
    assert duplicates == 0, (
        f"expected one row per (sample_id, model_name), found {duplicates} duplicate rows"
    )
    return long


def print_inventory(frame: pd.DataFrame) -> None:
    """Print columns, models, evals, and per-model cost and performance."""
    print("columns:", ", ".join(map(str, frame.columns)))
    for column in ("model_name", "eval_name"):
        print(f"{column}:")
        for name in _labels(frame[column]):
            print(f"  {name}")
    numeric = pd.DataFrame(
        {
            "model_name": frame["model_name"],
            "cost": pd.to_numeric(frame["cost"], errors="coerce"),
            "performance": _as_float(frame["performance"]),
        }
    )
    stats = numeric.groupby("model_name", sort=True)[["cost", "performance"]].agg(["min", "max", "mean"])
    print("per-model cost and performance (min, max, mean):")
    print(stats.to_string())


def filter_evals(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep MMLU, GSM8K, MBPP, and MT-Bench rows."""
    mask = frame["eval_name"].astype(str).str.lower().str.contains(_EVAL_RE, regex=True)
    kept = frame.loc[mask].copy()
    if kept.empty:
        raise ValueError("no rows matched MMLU, GSM8K, MBPP, or MT-Bench")
    names = [str(name) for name in _labels(kept["eval_name"])]
    print(f"locked eval_names ({len(names)}):")
    for name in names:
        print(f"  {name}")
    return kept


def select_arms(frame: pd.DataFrame) -> dict[str, object]:
    """Pick cheap, mid, strong_open, frontier, and one held-out model.

    Raises before any config write if mean cost does not rise along the four
    roles, or if frontier scores below cheap.
    """
    means, covered = _means(frame)  # get mean cost and performance per model
    if means.empty:
        raise ValueError("no model rows to select arms from")
    cost, score = means["cost"], means["performance"]
    names = [str(name) for name in means.index]

    # Pick the four arms and one held-out model. The frontier must contain "gpt-4".
    frontier = [name for name in names if "gpt-4" in name.lower()]
    if len(frontier) != 1:
        raise ValueError(f"expected one model_name containing GPT-4, found {frontier or 'none'}")
    frontier = frontier[0]
    open_models = [
        name
        for name in names
        if _open(name) and np.isfinite(cost[name]) and np.isfinite(score[name])
    ]
    if len(open_models) < 2:
        raise ValueError("need at least two open models to choose cheap and strong_open")
    # cheap: the open model with the lowest mean cost
    cheap = min(open_models, key=lambda name: (float(cost[name]), name))
    # strong_open: the open model with the highest mean performance, excluding cheap
    strong = min(
        (name for name in open_models if name != cheap),
        key=lambda name: (-float(score[name]), name),
    )
    if float(cost[strong]) < float(cost[cheap]):
        cheap, strong = strong, cheap

    # Pick mid: the open model with mean cost between cheap and frontier, closest to the midpoint.
    low, high = float(cost[cheap]), float(cost[frontier])
    target = (low + float(cost[strong])) / 2
    band = _MID_CLOSE * abs(float(cost[strong]) - low)
    between = [
        name
        for name in names
        if name not in {cheap, strong, frontier} and low < float(cost[name]) < high
    ]
    if not between:
        raise ValueError(
            "no unused model has a mean cost between cheap and frontier; config was not written"
        )
    best = min(abs(float(cost[name]) - target) for name in between)
    close = [name for name in between if abs(float(cost[name]) - target) <= best + band]
    mid = min(
        [name for name in close if _open(name)] or close,
        key=lambda name: (abs(float(cost[name]) - target), name),
    )

    # Pick held-out: any unused model with full coverage on the kept queries, lowest mean cost, and highest mean performance.
    held_pool = [name for name in names if name not in {cheap, mid, strong, frontier} and name in covered]
    if not held_pool:
        raise ValueError(
            "no unused model has full coverage on the kept queries; config was not written"
        )
    held = min(held_pool, key=lambda name: (-float(score[name]), name))
    arms = [cheap, mid, strong, frontier]
    print("selected arms:") # Print the selected arms and their costs and scores.
    # loop through the roles and arms, printing their costs and scores
    for role, name in zip(_ROLES, arms):
        print(f"  {role}: {name}  cost={float(cost[name]):.6g}  score={float(score[name]):.4f}")
    print(f"held_out_arm: {held}  cost={float(cost[held]):.6g}  score={float(score[held]):.4f}")
    ladder = [float(cost[name]) for name in arms]
    if any(ladder[i] > ladder[i + 1] for i in range(3)) or not (
        float(score[frontier]) >= float(score[cheap])
    ):
        raise ValueError(
            f"refusing to lock {list(zip(_ROLES, arms))} with costs {ladder}; config was not written"
        )
    return {"arms": arms, "held_out_arm": held}


def choose_mix_eval(frame: pd.DataFrame, cheap_model: str) -> str:
    """Exact MT-Bench name, or the locked eval where the cheap arm is weakest."""
    names = [str(name) for name in _labels(frame["eval_name"])]
    exact = [
        name
        for name in names
        if name.lower().replace("-", "").replace("_", "").replace(" ", "") == "mtbench"
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"more than one exact MT-Bench eval_name: {exact}")
    rows = frame.loc[frame["model_name"].astype(str) == str(cheap_model)]
    if rows.empty:
        raise ValueError(f"cheap arm {cheap_model} has no rows on the locked evals")
    
    # Compute the mean quality for each eval_name, and pick the one with the lowest mean quality.
    quality = _as_float(rows["performance"]).clip(0, 1)
    scores = rows.assign(_quality=quality.to_numpy()).groupby("eval_name", sort=False)["_quality"].mean()
    scores = scores[np.isfinite(scores.to_numpy())]
    if scores.empty:
        raise ValueError("cheap arm has no finite quality on the locked evals")
    return min((str(name) for name in scores.index), key=lambda name: (float(scores.loc[name]), name))


def lock_evals_and_arms(frame: pd.DataFrame, config_path: Path | str) -> dict[str, object]:
    """Filter, choose arms, and write four config fields. A failed check writes nothing."""
    locked = filter_evals(frame)    # Keep only the evals that match MMLU, GSM8K, MBPP, or MT-Bench.
    selection = select_arms(locked) # Select the four arms and one held-out model based on mean cost and performance.
    # Choose the mix_eval based on the cheap arm's performance on the locked evals.
    mix_eval = choose_mix_eval(locked, str(selection["arms"][0])) # type: ignore
    eval_names = [str(name) for name in _labels(locked["eval_name"])]
    # Patch the config file with the selected arms, held-out arm, eval names, and mix_eval.
    print(f"mix_eval: {mix_eval}")
    _patch_config(
        config_path,
        {
            "arms": list(selection["arms"]), # type: ignore
            "held_out_arm": selection["held_out_arm"],
            "eval_names": eval_names,
            "mix_eval": mix_eval,
        },
    )
    selection["eval_names"] = eval_names
    selection["mix_eval"] = mix_eval
    print(f"wrote arms, held_out_arm, eval_names, and mix_eval to {config_path}")
    return selection


def require_locked_arms(arms: object, expected: object) -> list[str]:
    """Refuse the wide table unless arms is exactly the four locked names, in role order."""
    expected_names = [str(name) for name in expected] if isinstance(expected, (list, tuple)) else []
    found = [str(name) for name in arms] if isinstance(arms, (list, tuple)) else None
    if found != expected_names or len(expected_names) != 4 or len(set(expected_names)) != 4:
        raise ValueError(
            "refusing to build the wide table: arms must be exactly "
            f"{expected_names} in role order (cheap, mid, strong_open, frontier), found {arms}"
        )
    return expected_names


def performance_to_quality(performance: pd.Series, sim_path: Path | str | None = None) -> pd.Series:
    """Bool becomes 0/1. Clip to [0, 1] only when the printed range falls outside it."""
    numeric = _as_float(performance).astype("float64")
    finite = numeric.to_numpy()
    finite = finite[np.isfinite(finite)]
    clipped = finite.size > 0 and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0)
    if finite.size == 0:
        print("no finite performance values; quality is missing")
    elif clipped:
        low, high = float(finite.min()), float(finite.max())
        print(f"printed performance range [{low:.6g}, {high:.6g}] is outside [0, 1]; quality = clip(performance, 0, 1)")
    else:
        low, high = float(finite.min()), float(finite.max())
        print(f"printed performance range [{low:.6g}, {high:.6g}] is inside [0, 1]; quality equals performance")
    if sim_path is not None:
        _sync_quality_note(Path(sim_path), clipped)
    return numeric.clip(0.0, 1.0) if clipped else numeric


def pivot_wide(
    frame: pd.DataFrame,
    arms: list[str],
    held_out_arm: str,
    sim_path: Path | str | None = None,
) -> pd.DataFrame:
    """One row per sample_id. Drop a query missing a finite value for any of the four arms."""
    names = [str(name) for name in arms]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError(
            "refusing to build the wide table: arms must be exactly four names "
            f"in role order (cheap, mid, strong_open, frontier), found {arms}"
        )
    held = str(held_out_arm)
    if held in names:
        raise ValueError(f"held_out_arm {held} is already one of the four arms")
    roles = dict(zip(names, _ROLES))
    roles[held] = "held_out" # type: ignore
    work = frame.copy()
    work["model_name"] = work["model_name"].astype(str)
    work["quality"] = performance_to_quality(work["performance"], sim_path)
    work["cost"] = pd.to_numeric(work["cost"], errors="coerce")
    base = work.groupby("sample_id", sort=False)[["prompt", "eval_name"]].first()
    part = work.loc[work["model_name"].isin(roles)].copy()
    part["role"] = part["model_name"].map(roles)
    quality = part.pivot(index="sample_id", columns="role", values="quality").add_prefix("quality_")
    cost = part.pivot(index="sample_id", columns="role", values="cost").add_prefix("cost_")
    wide = base.join(quality).join(cost).reset_index()
    for column in _WIDE_COLUMNS:
        if column not in wide.columns:
            wide[column] = np.nan
    wide = wide.loc[:, _WIDE_COLUMNS]
    arm_cols = [f"{kind}_{role}" for role in _ROLES for kind in ("quality", "cost")]
    keep = np.isfinite(wide.loc[:, arm_cols].to_numpy(dtype=float)).all(axis=1)
    print(f"dropped {int((~keep).sum())} queries missing a finite value for an arm")
    wide = wide.loc[keep].reset_index(drop=True) # type: ignore
    held_cols = ["quality_held_out", "cost_held_out"]
    missing = int((~np.isfinite(wide.loc[:, held_cols].to_numpy(dtype=float))).sum())
    print(f"held-out values missing: {missing}")
    return wide


def split_ids(wide: pd.DataFrame, test_size: float = 0.3, random_state: int = 0) -> dict[str, list]:
    """70/30 split of sample_id, stratified by eval_name (or by family if an eval is rare)."""
    labels = wide["eval_name"].astype(str)
    rare = sorted(labels.value_counts().loc[lambda counts: counts < 2].index)
    if rare:
        labels = labels.map(_family)
        print(
            "stratifying on family labels (mmlu, gsm8k, mbpp, mt-bench) because these "
            "eval_name values have fewer than two rows: " + ", ".join(rare)
        )
        if labels.value_counts().min() < 2:
            raise ValueError("cannot stratify; a family has fewer than two rows")
    else:
        print("stratifying by eval_name")
    train, test = train_test_split(
        [_plain(value) for value in wide["sample_id"].tolist()],
        test_size=test_size,
        random_state=random_state,
        stratify=labels.tolist(),
    )
    train, test = list(train), list(test)
    if set(train) & set(test) or set(train) | set(test) != set(wide["sample_id"]):
        raise ValueError("train and test id sets must be disjoint and cover every kept id")
    print(f"split: {len(train)} train, {len(test)} test")
    return {"train": train, "test": test}


def save_wide_and_split(
    wide: pd.DataFrame, split: dict[str, list], processed_dir: Path | str | None = None
) -> tuple[Path, Path]:
    """Write data/processed/wide.pkl and split.json."""
    directory = processed_data_dir() if processed_dir is None else Path(processed_dir)
    directory.mkdir(parents=True, exist_ok=True)
    wide_path, split_path = directory / "wide.pkl", directory / "split.json"
    wide.to_pickle(wide_path)
    split_path.write_text(json.dumps({"train": split["train"], "test": split["test"]}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {wide_path}")
    print(f"wrote {split_path}")
    return wide_path, split_path


def print_wide_check(wide: pd.DataFrame, split: dict[str, list], n: int = 5) -> None:
    """Print a few wide rows and the train/test id intersection."""
    columns = ["sample_id", "eval_name"] + [f"{kind}_{role}" for role in _ROLES for kind in ("quality", "cost")]
    preview = wide.loc[:, columns].head(n)
    print(f"{len(preview)} wide-table rows:")
    print(preview.to_string(index=False))
    values = preview.drop(columns=["sample_id", "eval_name"]).to_numpy(dtype=float)
    print(f"finite quality and cost for every arm: {bool(values.size) and bool(np.isfinite(values).all())}")
    print(f"train/test id intersection: {sorted(set(split['train']) & set(split['test']), key=str)}")


def embed_prompts(
    wide: pd.DataFrame,
    model_name: str,
    processed_dir: Path | str | None = None,
) -> np.ndarray:
    """Cache 384-d MiniLM vectors aligned to the wide-table row order.

    An existing embeddings.npy with one row per wide-table prompt is reused, so a
    rerun does not re-encode.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"embedding model name must be a non-empty string, found {model_name!r}")
    if "prompt" not in wide.columns:
        raise ValueError("wide table is missing a prompt column")
    directory = processed_data_dir() if processed_dir is None else Path(processed_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "embeddings.npy"
    expected = (len(wide), _EMBEDDING_DIM)
    if path.exists():
        cached = np.load(path)
        if cached.shape == expected:
            print(f"loaded cached embeddings {cached.shape} from {path}")
            return cached
        print(f"cached embeddings shape {getattr(cached, 'shape', None)} does not match {expected}; re-encoding")
    from sentence_transformers import SentenceTransformer

    prompts = wide["prompt"].astype(str).tolist()
    print(f"encoding {len(prompts)} prompts with {model_name}")
    model = SentenceTransformer(model_name)
    vectors = np.asarray(
        model.encode(
            prompts,
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
    print(f"wrote {path} with shape {vectors.shape}")
    return vectors


def fit_pca_contexts(
    embeddings: np.ndarray,
    wide: pd.DataFrame,
    split: dict[str, list],
    pca_dim: int,
    processed_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on train rows only, append a column of ones, and save the contexts.

    embeddings follows wide-table row order. context_train.npy and context_test.npy
    follow the train and test id lists. Each context row has length pca_dim + 1.
    """
    if type(pca_dim) is not int or pca_dim < 1:
        raise ValueError(f"pca_dim must be a positive int, found {pca_dim!r}")
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[0] != len(wide):
        raise ValueError(
            f"embeddings must have one row per wide-table prompt, found shape {matrix.shape} for {len(wide)} rows"
        )
    train_index = _positions(wide, split["train"])
    test_index = _positions(wide, split["test"])
    train_ids = {_plain(sample_id) for sample_id in split["train"]}
    test_ids = {_plain(sample_id) for sample_id in split["test"]}
    covered = {_plain(sample_id) for sample_id in wide["sample_id"].tolist()}
    if train_ids & test_ids:
        raise ValueError("train and test ids overlap")
    if (train_ids | test_ids) != covered:
        raise ValueError("train and test ids must cover every wide-table row")
    limit = min(matrix.shape[1], len(train_index))
    if pca_dim > limit:
        raise ValueError(f"pca_dim {pca_dim} exceeds min(train rows, embedding dim)={limit}")
    train_x = np.asarray(matrix[train_index], dtype=np.float64)
    test_x = np.asarray(matrix[test_index], dtype=np.float64)
    pca = PCA(n_components=pca_dim, random_state=0)
    context_train = _with_ones(pca.fit_transform(train_x))
    context_test = _with_ones(pca.transform(test_x))
    if int(pca.n_samples_) != len(train_index):
        raise ValueError(f"pca.n_samples_={pca.n_samples_} does not equal the train count {len(train_index)}")
    if context_train.shape != (len(train_index), pca_dim + 1) or context_test.shape != (
        len(test_index),
        pca_dim + 1,
    ):
        raise ValueError(
            f"expected context width {pca_dim + 1}, got train {context_train.shape} and test {context_test.shape}"
        )
    if not np.isfinite(context_train).all() or not np.isfinite(context_test).all():
        raise ValueError("context matrices contain non-finite values")
    directory = processed_data_dir() if processed_dir is None else Path(processed_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pca_path = directory / "pca.joblib"
    train_path = directory / "context_train.npy"
    test_path = directory / "context_test.npy"
    joblib.dump(pca, pca_path)
    np.save(train_path, context_train)
    np.save(test_path, context_test)
    print(f"fit PCA n_components={pca_dim} on {len(train_index)} train embeddings")
    print(f"wrote {pca_path}")
    print(f"wrote {train_path} shape {context_train.shape}")
    print(f"wrote {test_path} shape {context_test.shape}")
    return context_train, context_test, pca


def print_context_check(context_test: np.ndarray, pca: PCA, n_train: int) -> None:
    """Print one test context length and pca.n_samples_ beside the train count."""
    length = int(context_test.shape[1]) if context_test.ndim == 2 else None
    print(f"one test context length: {length}")
    print(f"pca.n_samples_={int(pca.n_samples_)} train count={n_train}")


def _looks_wide(frame: pd.DataFrame) -> bool:
    return any(str(column).endswith("|total_cost") for column in frame.columns)


def _wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in ("sample_id", "prompt", "eval_name") if name not in frame.columns]
    if missing:
        raise ValueError(f"RouterBench file is missing columns {missing}; found {list(map(str, frame.columns))}")
    models = [str(column)[: -len("|total_cost")] for column in frame.columns if str(column).endswith("|total_cost")]
    absent = [model for model in models if model not in frame.columns]
    if not models or absent:
        raise ValueError(
            "RouterBench file has cost columns without performance columns: "
            f"{absent or list(map(str, frame.columns))}"
        )
    slim = frame.drop(columns=[column for column in frame.columns if str(column).endswith("|model_response")])
    performance = slim.melt(
        id_vars=["sample_id", "prompt", "eval_name"], value_vars=models,
        var_name="model_name", value_name="performance",
    )
    cost = slim.melt(
        id_vars=["sample_id"], value_vars=[f"{model}|total_cost" for model in models],
        var_name="cost_column", value_name="cost",
    )
    cost["model_name"] = cost["cost_column"].str.removesuffix("|total_cost")
    long = performance.merge(
        cost[["sample_id", "model_name", "cost"]], on=["sample_id", "model_name"],
        how="left", validate="one_to_one",
    )
    print(f"normalized {len(models)} wide model columns into model_name, performance, and cost")
    return long.loc[:, list(LONG_COLUMNS)].copy()


def _rename_columns(frame: pd.DataFrame, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    aliases = _ALIASES if columns is None else {name: _ALIASES[name] for name in columns}
    by_lower = {}
    for column in frame.columns:
        by_lower.setdefault(str(column).strip().lower(), column)
    rename = {}
    for canonical, names in aliases.items():
        if canonical in frame.columns:
            continue
        for alias in names:
            original = by_lower.get(alias.strip().lower())
            if original is not None and original not in rename:
                rename[original] = canonical
                break
    if not rename:
        return frame
    print("renamed columns: " + ", ".join(f"{src} -> {dst}" for src, dst in rename.items()))
    return frame.rename(columns=rename)


def _as_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() or series.dropna().empty:
        return numeric
    return series.map(lambda value: float(value) if isinstance(value, (bool, np.bool_)) else np.nan)


def _labels(series: pd.Series) -> list[object]:
    return sorted((label for label in series.dropna().unique().tolist()), key=str)


def _open(name: str) -> bool:
    text = name.lower()
    return not any(marker in text for marker in _CLOSED)


def _means(frame: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    work = pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "model_name": frame["model_name"].astype(str),
            "cost": pd.to_numeric(frame["cost"], errors="coerce"),
            "performance": _as_float(frame["performance"]),
        }
    )
    means = work.groupby("model_name", sort=False)[["cost", "performance"]].mean()
    finite = work[np.isfinite(work["cost"]) & np.isfinite(work["performance"])]
    counts = finite.groupby("model_name")["sample_id"].nunique()
    covered = set(counts[counts == work["sample_id"].nunique()].index)
    return means, covered


def _family(name: str) -> str:
    text = name.lower()
    if "mmlu" in text:
        return "mmlu"
    if "gsm" in text or "grade-school-math" in text or "grade school math" in text:
        return "gsm8k"
    if "mbpp" in text:
        return "mbpp"
    if "mtbench" in text.replace("-", ""):
        return "mt-bench"
    return text


def _positions(wide: pd.DataFrame, ids: list) -> np.ndarray:
    """Row positions of ids, using the wide table's current order."""
    positions: dict[object, int] = {}
    for index, sample_id in enumerate(wide["sample_id"].tolist()):
        key = _plain(sample_id)
        if key in positions:
            raise ValueError(f"wide table sample_id values are not unique: {key}")
        positions[key] = index
    missing = [sample_id for sample_id in ids if _plain(sample_id) not in positions]
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"{len(missing)} split ids are missing from the wide table: {preview}")
    return np.fromiter((positions[_plain(sample_id)] for sample_id in ids), dtype=int, count=len(ids))


def _with_ones(transformed: np.ndarray) -> np.ndarray:
    ones = np.ones((transformed.shape[0], 1), dtype=transformed.dtype)
    return np.concatenate([transformed, ones], axis=1)


def _plain(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _sync_quality_note(path: Path, clipped: bool) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if (_QUALITY_NOTE in lines) == clipped:
        return
    lines = [line for line in lines if line != _QUALITY_NOTE]
    if clipped:
        lines.append(_QUALITY_NOTE)
    updated = "\n".join(lines)
    path.write_text(updated + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def _yaml_value(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_value(item) for item in value) + "]"
    text = str(value)
    return text if re.fullmatch(r"[A-Za-z0-9_./+-]+", text) else json.dumps(text)


def _patch_config(path: Path | str, values: dict[str, object]) -> None:
    """Replace four keys and leave every other line, including comments, alone."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    for key, value in values.items():
        rendered = _yaml_value(value)
        for index, line in enumerate(lines):
            if not line.startswith(f"{key}:"):
                continue
            comment = f"  {line[line.find('#'):]}" if "#" in line else ""
            lines[index] = f"{key}: {rendered}{comment}"
            break
        else:
            raise ValueError(f"{config_path} is missing {key}")
    config_path.write_text(newline.join(lines) + newline, encoding="utf-8", newline="")
