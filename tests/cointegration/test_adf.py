"""Unit tests for :func:`pairs.cointegration.unit_root_check`."""

from __future__ import annotations

import numpy as np
import pytest

from pairs.cointegration import unit_root_check


@pytest.mark.unit
def test_unit_root_guard_raises_on_returns_input(rng):
    from pairs._exceptions import InputError

    prices = 100 + rng.standard_normal(500).cumsum()
    returns = np.diff(prices)
    with pytest.raises(InputError, match="already stationary"):
        unit_root_check(returns)


@pytest.mark.unit
def test_unit_root_passes_on_log_prices(rng):
    prices = 100 + rng.standard_normal(500).cumsum()
    log_prices = np.log(prices)
    res = unit_root_check(log_prices)
    assert res.is_i1
    assert res.levels_pvalue >= 0.05
    assert res.diff_pvalue < 0.05


@pytest.mark.unit
def test_unit_root_short_sample_uses_dfgls_or_fallback(rng):
    prices = 100 + rng.standard_normal(60).cumsum()
    res = unit_root_check(prices, method="auto", min_obs_for_dfgls=100)
    assert res.method in {"dfgls", "adf"}


@pytest.mark.unit
def test_unit_root_rejects_bad_method(rng):
    from pairs._exceptions import InputError

    prices = 100 + rng.standard_normal(200).cumsum()
    with pytest.raises(InputError):
        unit_root_check(prices, method="zzz")  # type: ignore[arg-type]


@pytest.mark.unit
def test_unit_root_records_leg_name(rng):
    prices = 100 + rng.standard_normal(300).cumsum()
    res = unit_root_check(prices, leg_name="apple")
    assert res.leg_name == "apple"
