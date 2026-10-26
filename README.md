# Adaptive Model Selection for LLMs Using Contextual Bandits

MSML635 Group 13 - University of Maryland, College Park
Amey Hengle, Sawyer Byrd, Sujeet Madihalli

Routing as a contextual bandit: the router observes only the reward of the
model it chose, while models, prices, and query traffic keep changing.
This repo provides a RouterBench-based stream simulator with scripted
distribution shifts, bandit and supervised routers, and an off-policy
evaluation study validated against simulator ground truth.

## Setup

Install uv once:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then from the repo root:

    uv sync --extra dev
    uv run pytest

That is the entire setup. uv installs the pinned interpreter (3.12) and
resolves all dependencies from uv.lock, so every collaborator gets an
identical environment.

## Optional dependency groups

The core install is deliberately light: no PyTorch, no CUDA. Heavy
dependencies are only needed for one-time offline jobs.

    uv sync --extra embed    # compute query embeddings once, then cached
    uv sync --extra data     # download RouterBench from HuggingFace
    uv sync --extra dev      # tests and linting

Run the embedding job once, commit the cached vectors, and no other
collaborator ever installs PyTorch.

## Layout

    src/router_bandit/
      data/         stream protocol: RouterBench loader + synthetic generator
      envs/         bandit environment, shift scheduler, arm registry, costs
      policies/     epsilon-greedy, LinUCB, LinTS, discounted LinUCB
      baselines/    oracle, Zero Router, supervised, cascade
      evaluation/   regret, cost-quality frontier, OPE (IPS / SNIPS / DR)
      logs/         decision logger with propensities
    configs/        YAML experiment configs, one per research question
    tests/          verification tests
    data/           gitignored; small fixture committed under data/fixtures
    results/        gitignored except results/figures

## Data

RouterBench (~1.47 GB) is not committed. Tests run against a small
committed fixture, so a fresh clone needs no download.

## Note on OBP

Open Bandit Pipeline is not a runtime dependency: it is capped at
Python < 3.11 and transitively requires PyTorch and the CUDA toolchain.
We implement the OPE estimators directly and use OBP only in a separate
pinned environment to cross-validate them.
