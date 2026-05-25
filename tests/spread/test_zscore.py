"""Tests for the z-score transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pairs._exceptions import InputError
from pairs._rng import default_rng
from pairs.spread.ou import fit_ou
from pairs.spread.zscore import zscore


@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    k=st.integers(min_value=1, max_value=20),
    window=st.integers(min_value=5, max_value=50),
)
def test_zscore_shift_invariance(k: int, window: int) -> None:
    rng = default_rng(900)
    s = pd.Series(
        rng.standard_normal(500),
        index=pd.date_range("2020-01-01", periods=500, freq="D"),
        name="s",
    )
    z_orig = zscore(s, window=window).dropna()
    z_shifted = zscore(s.shift(k), window=window).shift(-k).dropna()
    # Align on overlap and compare
    common = z_orig.index.intersection(z_shifted.index)
    np.testing.assert_allclose(
        z_orig.loc[common].to_numpy(),
        z_shifted.loc[common].to_numpy(),
        atol=1e-9,
    )


@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    a=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    window=st.integers(min_value=5, max_value=40),
)
def test_zscore_scale_invariance(a: float, b: float, window: int) -> None:
    rng = default_rng(901)
    s = pd.Series(
        rng.standard_normal(300),
        index=pd.date_range("2020-01-01", periods=300, freq="D"),
        name="s",
    )
    z1 = zscore(s, window=window).dropna()
    z2 = zscore(a * s + b, window=window).dropna()
    np.testing.assert_allclose(z1.to_numpy(), z2.to_numpy(), atol=1e-9)


def test_window_auto_pick_from_half_life(simulated_ou) -> None:
    rng = default_rng(902)
    spread = simulated_ou(rng, theta=0.05, n=800)
    ou = fit_ou(spread)
    expected_window = max(2, int(round(2.0 * ou.half_life)))
    z_auto = zscore(spread, window=None, ou_result=ou)
    z_manual = zscore(spread, window=expected_window)
    pd.testing.assert_series_equal(z_auto.dropna(), z_manual.dropna(), check_names=False)


def test_window_minimum_floor_two(simulated_ou) -> None:
    rng = default_rng(903)
    spread = simulated_ou(rng, theta=0.49, mu=0.0, sigma=1.0, n=600)
    ou = fit_ou(spread)
    # If half_life is tiny, the floor of 2 must apply.
    z = zscore(spread, window=None, ou_result=ou)
    # Should not raise and should produce finite values eventually.
    assert z.dropna().shape[0] > 0


def test_ou_mode_uses_stationary_std(simulated_ou) -> None:
    rng = default_rng(904)
    spread = simulated_ou(rng, theta=0.05, mu=0.5, sigma=1.0, n=800)
    ou = fit_ou(spread)
    z = zscore(spread, use_ou=True, ou_result=ou)
    expected = (spread - ou.mu) / ou.sigma_eq
    pd.testing.assert_series_equal(z, expected, check_names=False)


def test_raises_when_both_window_and_ou_missing() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(InputError):
        zscore(s)


def test_use_ou_without_ou_result_raises() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(InputError):
        zscore(s, use_ou=True)


def test_invalid_window_raises() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(InputError):
        zscore(s, window=1)


def test_input_must_be_series() -> None:
    with pytest.raises(InputError):
        zscore([1.0, 2.0, 3.0], window=2)  # type: ignore[arg-type]


def test_no_lookahead() -> None:
    rng = default_rng(905)
    n = 200
    s = pd.Series(
        rng.standard_normal(n),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
        name="s",
    )
    z_full = zscore(s, window=20)
    t = 100  # any index >= window - 1
    s_perturbed = s.copy()
    s_perturbed.iloc[t + 1 :] = rng.standard_normal(n - t - 1) * 100.0
    z_perturbed = zscore(s_perturbed, window=20)
    np.testing.assert_allclose(
        z_full.iloc[: t + 1].dropna().to_numpy(),
        z_perturbed.iloc[: t + 1].dropna().to_numpy(),
        atol=1e-12,
    )
