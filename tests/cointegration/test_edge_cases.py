"""Edge-case tests for the cointegration sub-package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import (
    DegenerateSeriesError,
    InputError,
    InsufficientDataError,
)
from pairs.cointegration import engle_granger


@pytest.mark.unit
def test_constant_series_raises_degenerate(rng):
    x = pd.Series(np.ones(200), name="x")
    y = pd.Series(rng.standard_normal(200).cumsum() + 100, name="y")
    with pytest.raises(DegenerateSeriesError):
        engle_granger(x, y, use_log=False)


@pytest.mark.unit
def test_nan_handling_via_inner_join(rng):
    n = 200
    x = pd.Series(rng.standard_normal(n).cumsum() + 100, name="x")
    y = pd.Series(rng.standard_normal(n).cumsum() + 100, name="y")
    y.iloc[10:20] = np.nan
    res = engle_granger(x, y, use_log=False)
    assert res.n_obs == n - 10


@pytest.mark.unit
def test_mismatched_index_raises_input_error(rng):
    n = 200
    a = pd.Series(rng.standard_normal(n).cumsum() + 100, index=pd.RangeIndex(0, n), name="a")
    b = pd.Series(
        rng.standard_normal(n).cumsum() + 100,
        index=pd.RangeIndex(10_000, 10_000 + n),
        name="b",
    )
    # No overlapping index labels at all -> no aligned observations.
    with pytest.raises((InputError, InsufficientDataError)):
        engle_granger(a, b, use_log=False)


@pytest.mark.unit
def test_short_series_warns_or_raises(rng):
    x = pd.Series(rng.standard_normal(10).cumsum(), name="x")
    y = pd.Series(rng.standard_normal(10).cumsum(), name="y")
    with pytest.raises(InsufficientDataError):
        engle_granger(x, y, use_log=False)


@pytest.mark.unit
def test_one_leg_flat_raises_degenerate(rng):
    x = pd.Series(np.full(200, 5.0), name="x")
    y = pd.Series(rng.standard_normal(200).cumsum() + 100, name="y")
    with pytest.raises(DegenerateSeriesError):
        engle_granger(x, y, use_log=False)
