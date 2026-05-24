"""Tests for the CSCV Probability of Backtest Overfitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pairs._exceptions import InputError
from pairs.evaluation import pbo_cscv


@given(
    t=st.integers(min_value=80, max_value=200),
    n=st.integers(min_value=3, max_value=8),
)
@settings(max_examples=15, deadline=None)
def test_pbo_bounded_unit_interval(t: int, n: int) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(t, n)))
    result = pbo_cscv(df, s=8)
    assert 0.0 <= result.pbo <= 1.0


@pytest.mark.slow
def test_pbo_null_iid_returns_half(iid_returns_matrix: pd.DataFrame) -> None:
    subset = iid_returns_matrix.iloc[:1000, :20]
    result = pbo_cscv(subset, s=10)
    assert 0.30 <= result.pbo <= 0.70


def test_pbo_s_must_be_even() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 5)))
    with pytest.raises(InputError):
        pbo_cscv(df, s=7)
