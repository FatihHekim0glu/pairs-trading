"""Tests for :class:`pairs.spread.KalmanHedge`."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError, InsufficientDataError
from pairs._rng import default_rng
from pairs.spread.hedge import ols_hedge
from pairs.spread.kalman import KalmanHedge


def _drifting_pair(rng: np.random.Generator, n: int = 800) -> tuple[pd.Series, pd.Series]:
    x_log = 4.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    beta_t = 1.0 + 0.001 * np.arange(n)  # slow linear drift
    eps = rng.standard_normal(n) * 0.01
    y_log = beta_t * x_log + eps
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return (
        pd.Series(np.exp(y_log), index=idx, name="y"),
        pd.Series(np.exp(x_log), index=idx, name="x"),
    )


def test_kalman_beats_ols_on_drift() -> None:
    os.environ["KALMAN_BACKEND"] = "numpy"
    rng = default_rng(11)
    y, x = _drifting_pair(rng, n=800)
    n = y.shape[0]
    beta_t = 1.0 + 0.001 * np.arange(n)
    kh = KalmanHedge().fit(y, x, delta=5e-3)
    ols = ols_hedge(y, x)
    # Compare hedge ratio tracking MSE.
    mse_kalman = float(np.mean((kh.beta_series.to_numpy()[200:] - beta_t[200:]) ** 2))
    mse_ols = float(np.mean((ols.beta - beta_t[200:]) ** 2))
    assert mse_kalman < mse_ols


def test_kalman_backend_parity() -> None:
    """Both backends should produce qualitatively similar hedge-ratio
    trajectories on the same input.

    pykalman and the hand-rolled numpy fallback do not use identical
    state-space conventions (transition-covariance scaling, initial-state
    diffuse-prior handling, etc.), so per-bar equality is unrealistic.
    The honest contract is: both backends converge to similar dynamics
    after the burn-in period — high Pearson correlation on the steady-
    state slice, and final values within a few percent.
    """
    pytest.importorskip("pykalman")
    rng = default_rng(12)
    y, x = _drifting_pair(rng, n=400)
    os.environ["KALMAN_BACKEND"] = "numpy"
    np_res = KalmanHedge().fit(y, x, delta=1e-3)
    os.environ["KALMAN_BACKEND"] = "pykalman"
    pk_res = KalmanHedge().fit(y, x, delta=1e-3)
    os.environ["KALMAN_BACKEND"] = "numpy"

    np_beta = np_res.beta_series.to_numpy()
    pk_beta = pk_res.beta_series.to_numpy()
    # Skip the burn-in (first 100 bars) where diffuse-prior conventions diverge.
    tail = slice(100, None)
    corr = float(np.corrcoef(np_beta[tail], pk_beta[tail])[0, 1])
    assert corr > 0.85, (
        f"backends should track each other qualitatively after burn-in; corr={corr:.3f}"
    )
    # Final-state values should be within ~10%.
    np.testing.assert_allclose(np_beta[-1], pk_beta[-1], rtol=0.10, atol=0.10)


def test_kalman_state_shapes() -> None:
    os.environ["KALMAN_BACKEND"] = "numpy"
    rng = default_rng(13)
    y, x = _drifting_pair(rng, n=300)
    res = KalmanHedge().fit(y, x, delta=1e-4)
    assert res.beta_series.shape == (300,)
    assert res.alpha_series.shape == (300,)
    assert res.dynamic_spread.shape == (300,)
    assert res.dynamic_zscore.shape == (300,)
    assert res.innovations.shape == (300,)
    assert res.backend == "numpy"


def test_kalman_input_validation() -> None:
    os.environ["KALMAN_BACKEND"] = "numpy"
    rng = default_rng(14)
    y, x = _drifting_pair(rng, n=200)
    with pytest.raises(InputError):
        KalmanHedge().fit([1.0, 2.0], x)  # type: ignore[arg-type]
    with pytest.raises(InputError):
        KalmanHedge().fit(y, x, delta=0.0)
    with pytest.raises(InputError):
        KalmanHedge().fit(y, x, delta=1.0)


def test_kalman_requires_min_obs() -> None:
    os.environ["KALMAN_BACKEND"] = "numpy"
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    y = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx, name="y")
    x = pd.Series([1.0, 1.05, 1.1, 1.15, 1.2], index=idx, name="x")
    with pytest.raises(InsufficientDataError):
        KalmanHedge().fit(y, x)


def test_kalman_rejects_non_positive_with_log() -> None:
    os.environ["KALMAN_BACKEND"] = "numpy"
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    y = pd.Series(np.linspace(-1.0, 1.0, 20), index=idx, name="y")
    x = pd.Series(np.linspace(1.0, 2.0, 20), index=idx, name="x")
    with pytest.raises(InputError):
        KalmanHedge().fit(y, x, use_log=True)


def test_kalman_use_log_false_works() -> None:
    os.environ["KALMAN_BACKEND"] = "numpy"
    rng = default_rng(15)
    y, x = _drifting_pair(rng, n=200)
    res = KalmanHedge().fit(y, x, use_log=False, delta=1e-4)
    assert np.isfinite(res.log_likelihood)


def test_kalman_numpy_rejects_constant_x() -> None:
    """A constant x makes the observation matrix singular for noise estimation."""

    from pairs._exceptions import DegenerateSeriesError

    os.environ["KALMAN_BACKEND"] = "numpy"
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    x = pd.Series(np.full(50, 5.0), index=idx, name="x")
    y = pd.Series(np.linspace(1.0, 2.0, 50), index=idx, name="y")
    with pytest.raises(DegenerateSeriesError):
        KalmanHedge().fit(y, x, use_log=False, delta=1e-3)


def test_kalman_numpy_backend_selected_when_env_forces_numpy() -> None:
    """When ``KALMAN_BACKEND=numpy``, the numpy backend wins regardless of pykalman."""

    os.environ["KALMAN_BACKEND"] = "numpy"
    rng = default_rng(16)
    y, x = _drifting_pair(rng, n=200)
    res = KalmanHedge().fit(y, x, delta=1e-3)
    assert res.backend == "numpy"


def test_kalman_backend_unknown_env_falls_back_to_numpy() -> None:
    """Unrecognised ``KALMAN_BACKEND`` values resolve to the numpy fallback."""

    os.environ["KALMAN_BACKEND"] = "made-up-backend"
    try:
        rng = default_rng(17)
        y, x = _drifting_pair(rng, n=150)
        res = KalmanHedge().fit(y, x, delta=1e-3)
        assert res.backend == "numpy"
    finally:
        os.environ["KALMAN_BACKEND"] = "numpy"
