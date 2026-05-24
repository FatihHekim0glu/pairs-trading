"""Tests for :func:`pairs.spread.build_spread`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs._rng import default_rng
from pairs.spread.hedge import ols_hedge
from pairs.spread.spread import build_spread


def test_build_spread_zero_mean_on_perfect_fit() -> None:
    rng = default_rng(7)
    n = 400
    x_log = 4.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    beta_true = 1.5
    y_log = beta_true * x_log  # no noise, no intercept
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    y = pd.Series(np.exp(y_log), index=idx, name="y")
    x = pd.Series(np.exp(x_log), index=idx, name="x")
    fit = ols_hedge(y, x)
    spread = build_spread(y, x, beta=fit.beta, alpha=fit.alpha)
    assert abs(float(spread.mean())) < 1e-6


def test_spread_named_correctly() -> None:
    rng = default_rng(8)
    n = 200
    x = pd.Series(
        np.exp(np.cumsum(rng.standard_normal(n) * 0.01)) + 1.0,
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
        name="legX",
    )
    y = pd.Series(
        np.exp(np.cumsum(rng.standard_normal(n) * 0.01)) + 1.0,
        index=x.index,
        name="legY",
    )
    spread = build_spread(y, x, beta=1.0)
    assert spread.name == "spread(legY,legX)"


def test_use_log_false_uses_raw_prices() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0], index=idx, name="y")
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="x")
    log_spread = build_spread(y, x, beta=2.0, alpha=0.0, use_log=True)
    raw_spread = build_spread(y, x, beta=2.0, alpha=0.0, use_log=False)
    np.testing.assert_allclose(raw_spread.to_numpy(), np.zeros(5), atol=1e-12)
    np.testing.assert_allclose(
        log_spread.to_numpy(), np.log(y.to_numpy()) - 2.0 * np.log(x.to_numpy())
    )


def test_input_must_be_series() -> None:
    with pytest.raises(InputError):
        build_spread([1.0, 2.0], pd.Series([1.0, 2.0]), beta=1.0)  # type: ignore[arg-type]


def test_log_rejects_non_positive() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    y = pd.Series([1.0, 2.0, 3.0], index=idx, name="y")
    x = pd.Series([-1.0, 2.0, 3.0], index=idx, name="x")
    with pytest.raises(InputError):
        build_spread(y, x, beta=1.0, use_log=True)
