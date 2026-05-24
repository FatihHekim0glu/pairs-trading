"""Tests for the stationary-bootstrap confidence interval."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pairs.evaluation import stationary_bootstrap_ci


def _mean(arr: np.ndarray) -> float:
    return float(arr.mean())


@given(
    n=st.integers(min_value=30, max_value=500),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=15, deadline=None)
def test_bootstrap_block_positive(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=n)
    result = stationary_bootstrap_ci(arr, _mean, alpha=0.1, n_boot=100, rng=rng)
    assert result.expected_block >= 2


@pytest.mark.slow
def test_bootstrap_iid_mean_coverage() -> None:
    rng = np.random.default_rng(42)
    coverage = 0
    trials = 100
    for _ in range(trials):
        x = rng.normal(loc=0.0, scale=1.0, size=500)
        ci = stationary_bootstrap_ci(x, _mean, alpha=0.05, n_boot=200, rng=rng)
        if ci.ci_low <= 0.0 <= ci.ci_high:
            coverage += 1
    assert coverage / trials >= 0.85


def test_bootstrap_seed_reproducible() -> None:
    rng_a = np.random.default_rng(2024)
    rng_b = np.random.default_rng(2024)
    arr = np.linspace(-1, 1, 200)
    ci_a = stationary_bootstrap_ci(arr, _mean, alpha=0.1, n_boot=200, rng=rng_a)
    ci_b = stationary_bootstrap_ci(arr, _mean, alpha=0.1, n_boot=200, rng=rng_b)
    assert ci_a.ci_low == ci_b.ci_low
    assert ci_a.ci_high == ci_b.ci_high
