"""Canonical arm registry and column schema for RouterBench.

RouterBench ships one row per query and three columns per model:

* ``<model>``                 -- quality score in ``[0, 1]`` (``object`` dtype in the raw file)
* ``<model>|total_cost``      -- dollar cost of that model's response to that query
* ``<model>|model_response``  -- the generated text (unused for routing)

Everything downstream refers to arms by their index in :data:`MODELS`, which is
ordered cheapest to most expensive so that plots and confusion matrices read
left-to-right as "cheap to premium".
"""

from __future__ import annotations

import re
from typing import Final

# --------------------------------------------------------------------------- #
# Non-model columns
# --------------------------------------------------------------------------- #
ID_COL: Final[str] = "sample_id"
PROMPT_COL: Final[str] = "prompt"
EVAL_COL: Final[str] = "eval_name"
ORACLE_COL: Final[str] = "oracle_model_to_route_to"

#: Sentinel used by RouterBench when no model answered a query correctly.
NO_MODEL_CORRECT: Final[str] = "no_model_correct"

# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
#: The 11 RouterBench arms, ordered by mean per-query cost (ascending).
MODELS: Final[tuple[str, ...]] = (
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "meta/code-llama-instruct-34b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "meta/llama-2-70b-chat",
    "claude-instant-v1",
    "gpt-3.5-turbo-1106",
    "claude-v1",
    "claude-v2",
    "gpt-4-1106-preview",
)

N_ARMS: Final[int] = len(MODELS)

#: Short labels for figures and tables.
SHORT_NAMES: Final[dict[str, str]] = {
    "mistralai/mistral-7b-chat": "mistral-7b",
    "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b",
    "mistralai/mixtral-8x7b-chat": "mixtral-8x7b",
    "meta/code-llama-instruct-34b-chat": "codellama-34b",
    "zero-one-ai/Yi-34B-Chat": "yi-34b",
    "meta/llama-2-70b-chat": "llama2-70b",
    "claude-instant-v1": "claude-instant",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
    "claude-v1": "claude-v1",
    "claude-v2": "claude-v2",
    "gpt-4-1106-preview": "gpt-4",
}

#: Models usually treated as the "premium" tier when reporting escalation rates.
PREMIUM_MODELS: Final[tuple[str, ...]] = (
    "claude-v1",
    "claude-v2",
    "gpt-4-1106-preview",
)


def quality_col(model: str) -> str:
    """Column holding the quality score for ``model``."""
    return model


def cost_col(model: str) -> str:
    """Column holding the dollar cost for ``model``."""
    return f"{model}|total_cost"


def response_col(model: str) -> str:
    """Column holding the generated text for ``model``."""
    return f"{model}|model_response"


def quality_cols() -> list[str]:
    return [quality_col(m) for m in MODELS]


def cost_cols() -> list[str]:
    return [cost_col(m) for m in MODELS]


def response_cols() -> list[str]:
    return [response_col(m) for m in MODELS]


def arm_index(model: str) -> int:
    """Index of ``model`` in :data:`MODELS`."""
    return MODELS.index(model)


def short(model: str) -> str:
    return SHORT_NAMES.get(model, model)


def short_names() -> list[str]:
    return [short(m) for m in MODELS]


PREMIUM_INDICES: Final[tuple[int, ...]] = tuple(arm_index(m) for m in PREMIUM_MODELS)

# --------------------------------------------------------------------------- #
# Benchmark families
# --------------------------------------------------------------------------- #
# ``eval_name`` takes 86 distinct values in the 0-shot split: the 57 MMLU
# subjects plus a long tail. We fold them into the families described in the
# RouterBench paper and tag everything else as ``other`` rather than dropping
# it, so results can be reported with and without the undocumented tail.
_FAMILY_RULES: Final[tuple[tuple[str, str], ...]] = (
    (r"mmlu", "mmlu"),
    (r"hellaswag", "hellaswag"),
    (r"gsm8k|grade[\s_-]*school[\s_-]*math", "gsm8k"),
    (r"winogrande", "winogrande"),
    (r"(?:^|[^a-z])arc(?:[^a-z]|$)|arc[\s_-]*challenge", "arc"),
    (r"mbpp", "mbpp"),
    (r"mt[\s_-]*bench", "mtbench"),
    (r"(?:^|[^a-z])rag(?:[^a-z]|$)", "rag"),
)

_COMPILED_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(pat), name) for pat, name in _FAMILY_RULES
)

OTHER_FAMILY: Final[str] = "other"

#: Families explicitly described in the RouterBench paper.
DOCUMENTED_FAMILIES: Final[frozenset[str]] = frozenset(
    {"mmlu", "hellaswag", "gsm8k", "winogrande", "arc", "mbpp", "mtbench", "rag"}
)

#: Families excluded by default. MT-Bench is roughly 0.4% of rows -- too small
#: to split -- which is why RouterBench's own authors excluded it from their
#: predictive-router experiments.
DEFAULT_EXCLUDED_FAMILIES: Final[frozenset[str]] = frozenset({"mtbench"})


def eval_to_family(eval_name: str) -> str:
    """Map a raw ``eval_name`` to a benchmark family.

    Unrecognised evals return :data:`OTHER_FAMILY` -- they are kept and tagged,
    never silently dropped.
    """
    key = str(eval_name).lower()
    for pattern, family in _COMPILED_RULES:
        if pattern.search(key):
            return family
    return OTHER_FAMILY


def is_documented(family: str) -> bool:
    """True if ``family`` is one of the families named in the paper."""
    return family in DOCUMENTED_FAMILIES
