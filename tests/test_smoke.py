"""Smoke test: the package imports and the environment is sane."""

import numpy as np

import router_bandit


def test_version():
    assert router_bandit.__version__ == "0.1.0"


def test_subpackages_import():
    from router_bandit import baselines, data, envs, evaluation, logs, policies

    assert all(m is not None for m in (data, envs, policies, baselines, evaluation, logs))


def test_numpy_is_v2():
    assert int(np.__version__.split(".")[0]) >= 2
