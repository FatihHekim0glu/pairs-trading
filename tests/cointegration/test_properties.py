"""Hypothesis-based property tests for the Engle-Granger test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pairs.cointegration import engle_granger


def _make_random_walk(seed: int, t: int) -> tuple[pd.Series, pd.Series]:
    gen = np.random.default_rng(seed)
    x = 100 + gen.standard_normal(t).cumsum()
    y = 100 + gen.standard_normal(t).cumsum()
    return (
        pd.Series(x, name="x"),
        pd.Series(y, name="y"),
    )


def _make_coint_pair(seed: int, t: int, rho: float) -> tuple[pd.Series, pd.Series]:
    gen = np.random.default_rng(seed)
    x = 100 + gen.standard_normal(t).cumsum()
    eps = gen.standard_normal(t)
    resid = np.empty(t)
    resid[0] = eps[0]
    for i in range(1, t):
        resid[i] = rho * resid[i - 1] + eps[i]
    y = x + resid
    return pd.Series(x, name="x"), pd.Series(y, name="y")


@pytest.mark.property
@settings(
    deadline=None,
    max_examples=1,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(start_seed=st.integers(min_value=1, max_value=10_000))
def test_eg_false_positive_rate_random_walks(start_seed):
    n_trials = 50
    rejected = 0
    for k in range(n_trials):
        x, y = _make_random_walk(start_seed + k, t=300)
        res = engle_granger(x, y, use_log=False)
        if res.pvalue < 0.05:
            rejected += 1
    fpr = rejected / n_trials
    assert fpr <= 0.15, f"empirical false-positive rate {fpr:.2f} exceeded 0.15"


@pytest.mark.property
@settings(
    deadline=None,
    max_examples=1,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(start_seed=st.integers(min_value=1, max_value=10_000))
def test_eg_detection_rate_synthetic_coint_pair(start_seed):
    n_trials = 50
    detected = 0
    for k in range(n_trials):
        x, y = _make_coint_pair(start_seed + k, t=400, rho=0.3)
        res = engle_granger(x, y, use_log=False)
        if res.pvalue < 0.05:
            detected += 1
    rate = detected / n_trials
    assert rate >= 0.7, f"empirical detection rate {rate:.2f} below 0.70"


@pytest.mark.property
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    scale=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    shift=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=1, max_value=1_000),
)
def test_eg_pvalue_invariant_under_affine(scale, shift, seed):
    x, y = _make_coint_pair(seed, t=300, rho=0.3)
    base = engle_granger(x, y, use_log=False)
    transformed = engle_granger(scale * x + shift, y, use_log=False)
    assert abs(base.pvalue - transformed.pvalue) <= 0.15
