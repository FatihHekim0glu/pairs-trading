"""Tests for :mod:`pairs.selection._hurst`."""

from __future__ import annotations

import numpy as np
import pytest

from pairs._exceptions import InputError, InsufficientDataError
from pairs._rng import default_rng
from pairs.selection._hurst import hurst_exponent


def _ar1(rng: np.random.Generator, rho: float, T: int) -> np.ndarray:
    eps = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = eps[0]
    for t in range(1, T):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def test_hurst_ar1_mean_reverting() -> None:
    """Strong mean reversion should give Hurst values below the RW boundary."""
    rng = default_rng(seed=20260524)
    values: list[float] = []
    for _ in range(50):
        series = _ar1(rng, rho=0.2, T=2000)
        h = hurst_exponent(series)
        values.append(h)
    mean_h = float(np.mean(values))
    assert mean_h < 0.65, f"expected mean-reverting AR(1) H < 0.65 (RS bias band), got {mean_h:.3f}"


def test_hurst_random_walk_near_half() -> None:
    rng = default_rng(seed=20260525)
    values: list[float] = []
    for _ in range(30):
        series = np.cumsum(rng.standard_normal(2000))
        values.append(hurst_exponent(series, is_increments=False))
    mean_h = float(np.mean(values))
    assert 0.35 < mean_h < 0.75, (
        f"random walk Hurst centre should be near 0.5; got mean {mean_h:.3f}"
    )


def test_hurst_raises_on_short_series() -> None:
    with pytest.raises(InsufficientDataError):
        hurst_exponent(np.ones(50))


def test_hurst_invalid_min_lag() -> None:
    rng = default_rng(seed=1)
    with pytest.raises(InputError):
        hurst_exponent(rng.standard_normal(500), min_lag=1)


def test_hurst_invalid_window_relationship() -> None:
    rng = default_rng(seed=2)
    with pytest.raises(InputError):
        hurst_exponent(rng.standard_normal(500), min_lag=20, max_lag=15)


def test_hurst_returns_nan_on_constant_input() -> None:
    """Constant series have zero variance in every window."""
    result = hurst_exponent(np.zeros(500))
    assert np.isnan(result)


def test_hurst_filters_non_finite_values() -> None:
    rng = default_rng(seed=3)
    series = rng.standard_normal(500)
    series[::25] = np.nan
    h = hurst_exponent(series)
    assert np.isfinite(h)
