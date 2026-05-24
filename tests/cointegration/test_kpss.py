"""Unit tests for :func:`pairs.cointegration.kpss_spread`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.cointegration import kpss_spread


@pytest.mark.unit
def test_kpss_fails_to_reject_on_stationary(stationary_series):
    s = stationary_series(t=400, rho=0.3)
    res = kpss_spread(s)
    assert res.is_stationary
    assert res.pvalue > 0.05


@pytest.mark.unit
def test_kpss_rejects_on_random_walk(rng):
    rw = pd.Series(rng.standard_normal(500).cumsum(), name="rw")
    res = kpss_spread(rw)
    assert not res.is_stationary
    assert res.pvalue < 0.05


@pytest.mark.unit
def test_kpss_interpolation_flag_at_extremes(rng):
    rw = pd.Series(rng.standard_normal(500).cumsum(), name="rw")
    res = kpss_spread(rw)
    # Random walk returns p-value clipped to 0.01 by statsmodels.
    assert res.pvalue_interpolated


@pytest.mark.unit
def test_kpss_rejects_bad_regression(stationary_series):
    from pairs._exceptions import InputError

    s = stationary_series()
    with pytest.raises(InputError):
        kpss_spread(s, regression="zz")  # type: ignore[arg-type]


@pytest.mark.unit
def test_kpss_constant_series_raises(rng):
    from pairs._exceptions import DegenerateSeriesError

    with pytest.raises(DegenerateSeriesError):
        kpss_spread(np.zeros(200))
