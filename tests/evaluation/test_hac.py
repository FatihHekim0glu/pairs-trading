"""Tests for the Newey-West HAC standard error."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pairs.evaluation import andrews_lag, newey_west_se


@given(
    t1=st.integers(min_value=10, max_value=1000),
    t2=st.integers(min_value=10, max_value=2000),
)
@settings(max_examples=30, deadline=None)
def test_andrews_lag_monotone(t1: int, t2: int) -> None:
    a, b = min(t1, t2), max(t1, t2)
    assert andrews_lag(a) <= andrews_lag(b)


def test_newey_west_iid_matches_ols_se_approx() -> None:
    rng = np.random.default_rng(11)
    t = 5000
    x = rng.normal(0.0, 1.0, size=t)
    nw = newey_west_se(x)
    ols = float(x.std(ddof=1) / np.sqrt(t))
    assert nw == pytest.approx(ols, rel=0.25)
