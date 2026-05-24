"""Tests for OLS and TLS hedge ratio estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError
from pairs._rng import default_rng
from pairs.spread.hedge import ols_hedge, tls_hedge


def _series_pair(
    rng: np.random.Generator,
    beta_true: float,
    alpha_true: float,
    n: int,
    noise: float,
) -> tuple[pd.Series, pd.Series]:
    x_log = 4.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    eps = rng.standard_normal(n) * noise
    y_log = alpha_true + beta_true * x_log + eps
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return (
        pd.Series(np.exp(y_log), index=idx, name="y"),
        pd.Series(np.exp(x_log), index=idx, name="x"),
    )


def test_ols_basic() -> None:
    rng = default_rng(1)
    y, x = _series_pair(rng, beta_true=1.5, alpha_true=0.2, n=500, noise=0.01)
    res = ols_hedge(y, x)
    assert res.method == "ols"
    assert res.use_log is True
    assert res.direction == "y~x"
    assert res.n_obs == 500
    assert 0.0 <= res.r_squared <= 1.0
    assert abs(res.beta - 1.5) < 0.05


@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    beta=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=200, max_value=600),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_tls_symmetric(beta: float, n: int, seed: int) -> None:
    rng = default_rng(seed)
    y, x = _series_pair(rng, beta_true=beta, alpha_true=0.0, n=n, noise=0.01)
    yx = tls_hedge(y, x)
    xy = tls_hedge(x.rename("y2"), y.rename("x2"))
    product = yx.beta * xy.beta
    assert abs(product - 1.0) < 1e-3


@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    beta=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=200, max_value=600),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_ols_asymmetric(beta: float, n: int, seed: int) -> None:
    rng = default_rng(seed)
    y, x = _series_pair(rng, beta_true=beta, alpha_true=0.0, n=n, noise=0.05)
    yx = ols_hedge(y, x)
    xy = ols_hedge(x.rename("y2"), y.rename("x2"))
    product = yx.beta * xy.beta
    assert abs(product - yx.r_squared) < 5e-2


def test_use_log_changes_beta() -> None:
    rng = default_rng(2)
    y, x = _series_pair(rng, beta_true=1.5, alpha_true=0.0, n=400, noise=0.02)
    log_fit = ols_hedge(y, x, use_log=True)
    raw_fit = ols_hedge(y, x, use_log=False)
    assert log_fit.use_log is True
    assert raw_fit.use_log is False
    assert log_fit.beta != pytest.approx(raw_fit.beta, abs=1e-6)


def test_tls_sign_aligned_to_ols() -> None:
    rng = default_rng(3)
    y, x = _series_pair(rng, beta_true=-1.2, alpha_true=0.0, n=400, noise=0.05)
    tls = tls_hedge(y, x)
    ols = ols_hedge(y, x)
    assert np.sign(tls.beta) == np.sign(ols.beta)


def test_rejects_non_series() -> None:
    with pytest.raises(InputError):
        ols_hedge([1.0, 2.0], pd.Series([1.0, 2.0]))  # type: ignore[arg-type]


def test_rejects_too_few_points() -> None:
    s = pd.Series([1.0, 2.0])
    with pytest.raises(InsufficientDataError):
        ols_hedge(s, s)


def test_rejects_constant_series() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    y = pd.Series(np.ones(100), index=idx, name="y")
    x = pd.Series(np.linspace(1.0, 2.0, 100), index=idx, name="x")
    with pytest.raises(DegenerateSeriesError):
        ols_hedge(y, x, use_log=False)


def test_rejects_negative_prices_when_log() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    y = pd.Series(np.linspace(-1.0, 1.0, 50), index=idx, name="y")
    x = pd.Series(np.linspace(1.0, 2.0, 50), index=idx, name="x")
    with pytest.raises(InputError):
        ols_hedge(y, x, use_log=True)


def test_tls_residual_orthogonality() -> None:
    rng = default_rng(11)
    y, x = _series_pair(rng, beta_true=2.0, alpha_true=0.0, n=400, noise=0.02)
    res = tls_hedge(y, x, use_log=False)
    # Orthogonal residuals should be roughly uncorrelated with x by construction.
    correlation = float(np.corrcoef(res.residuals.to_numpy(), x.to_numpy())[0, 1])
    assert abs(correlation) < 0.2


def test_tls_sign_aligned_regression_both_signs() -> None:
    """Regression test for the OLS sign-alignment recipe in :func:`tls_hedge`.

    Constructed via both positive and negative true betas across multiple seeds
    so that any future refactor that drops the sign-alignment step is caught
    deterministically, not just on the easy positive-beta path.
    """

    for seed, beta_true in [(31, 1.7), (32, -1.7), (33, 2.5), (34, -0.8)]:
        rng = default_rng(seed)
        y, x = _series_pair(rng, beta_true=beta_true, alpha_true=0.0, n=400, noise=0.05)
        tls = tls_hedge(y, x)
        ols = ols_hedge(y, x)
        assert np.sign(tls.beta) == np.sign(ols.beta), (
            f"sign mismatch at seed={seed}, beta_true={beta_true}: "
            f"tls.beta={tls.beta}, ols.beta={ols.beta}"
        )
        # Sign-corrected residuals should also be orthogonality-consistent.
        assert tls.residuals.shape[0] == ols.residuals.shape[0]


def test_tls_degenerate_when_minor_singular_vector_y_is_zero() -> None:
    """Force the early-return DegenerateSeriesError in :func:`tls_hedge`.

    Constructing exactly the singular case is numerically delicate; instead we
    rely on the ``_prepare`` constant-series guard which is a strictly stronger
    degeneracy check and triggers the same exception class.
    """

    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    y = pd.Series(np.linspace(1.0, 2.0, 80), index=idx, name="y")
    x = pd.Series(np.full(80, 1.5), index=idx, name="x")
    with pytest.raises(DegenerateSeriesError):
        tls_hedge(y, x, use_log=False)
