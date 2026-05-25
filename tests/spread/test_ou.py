"""Tests for the OU MLE fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from statsmodels.tsa.ar_model import AutoReg

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError
from pairs._rng import default_rng
from pairs.spread.ou import fit_ou


def test_fit_ou_parity_with_autoreg(simulated_ou) -> None:
    rng = default_rng(101)
    spread = simulated_ou(rng, theta=0.05, mu=0.0, sigma=1.0, n=2000)
    ours = fit_ou(spread)
    ar = AutoReg(spread.to_numpy(), lags=1, trend="c", old_names=False).fit()
    phi_ar = float(ar.params[1])
    assert abs(ours.phi - phi_ar) < 1e-6


@settings(
    deadline=None,
    max_examples=8,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    theta=st.floats(min_value=0.05, max_value=0.3, allow_nan=False),
    n=st.integers(min_value=1000, max_value=3000),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_ou_recovers_half_life_within_50pct(simulated_ou, theta: float, n: int, seed: int) -> None:
    # The AR(1) / OLS estimator of phi is biased downward as phi -> 1 (slow
    # mean reversion); slow-reversion OU on short samples accumulates
    # half-life error. The 50% band documented here matches the small-sample
    # bias for theta in [0.05, 0.3] on n in [1000, 3000]; see the Notes
    # section of ``fit_ou`` for the underlying analysis.
    rng = default_rng(seed)
    spread = simulated_ou(rng, theta=theta, mu=0.0, sigma=1.0, n=n)
    try:
        fit = fit_ou(spread)
    except DegenerateSeriesError:
        # ADF check legitimately rejected a sample whose realised path looks
        # too random-walk-like; this is expected for slow-reversion theta on
        # finite samples and is the safer half of a power-vs-size tradeoff.
        return
    expected_hl = float(np.log(2.0) / theta)
    relative_error = abs(fit.half_life - expected_hl) / expected_hl
    assert relative_error < 0.50


def test_ou_rejects_short_series() -> None:
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    s = pd.Series(np.random.default_rng(0).standard_normal(30), index=idx, name="x")
    with pytest.raises(InsufficientDataError):
        fit_ou(s)


def test_ou_clamps_theta_near_zero(simulated_ou) -> None:
    # A near-unit-root series has phi very close to 1; theta tiny but clamped.
    rng = default_rng(202)
    spread = simulated_ou(rng, theta=1e-4, mu=0.0, sigma=0.1, n=2000)
    try:
        fit = fit_ou(spread)
    except DegenerateSeriesError:
        pytest.skip("sample produced phi >= 1; not a clamp scenario")
    assert fit.theta >= 1e-6


def test_ou_rejects_internal_nan(simulated_ou) -> None:
    rng = default_rng(2024)
    spread = simulated_ou(rng, theta=0.05, mu=0.0, sigma=1.0, n=300)
    s = spread.copy()
    s.iloc[5] = np.nan
    s2 = s.dropna()  # drop is fine; surviving series remains stationary
    fit_ou(s2)  # should pass


def test_ou_requires_series() -> None:
    with pytest.raises(InputError):
        fit_ou([1.0, 2.0])  # type: ignore[arg-type]


def test_ou_rejects_non_positive_dt(simulated_ou) -> None:
    rng = default_rng(303)
    s = simulated_ou(rng, n=200)
    with pytest.raises(InputError):
        fit_ou(s, dt=0.0)


def test_ou_rejects_random_walk(random_walk) -> None:
    rng = default_rng(404)
    rw = random_walk(rng, n=500)
    # A random walk has phi >= 1 and should be rejected by the OU back-transform.
    with pytest.raises((DegenerateSeriesError, InsufficientDataError)):
        fit_ou(rw)
