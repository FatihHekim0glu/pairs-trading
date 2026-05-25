"""Shared fixtures for the spread test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._rng import default_rng


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator for every test that asks for it."""

    return default_rng(20260523)


@pytest.fixture
def simulated_ou():
    """Return a factory that simulates an OU process as a pandas Series."""

    def _make(
        rng: np.random.Generator,
        theta: float = 0.05,
        mu: float = 0.0,
        sigma: float = 1.0,
        n: int = 1000,
        dt: float = 1.0,
        s0: float | None = None,
    ) -> pd.Series:
        phi = float(np.exp(-theta * dt))
        sigma_eq = sigma / np.sqrt(2.0 * theta)
        s = np.empty(n, dtype=np.float64)
        s[0] = mu if s0 is None else float(s0)
        eps_scale = sigma_eq * np.sqrt(1.0 - phi * phi)
        eps = rng.standard_normal(n - 1) * eps_scale
        for t in range(1, n):
            s[t] = mu + phi * (s[t - 1] - mu) + eps[t - 1]
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.Series(s, index=idx, name="ou_spread")

    return _make


@pytest.fixture
def random_walk():
    """Return a factory producing a unit-root random walk."""

    def _make(rng: np.random.Generator, n: int = 1000) -> pd.Series:
        steps = rng.standard_normal(n)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.Series(np.cumsum(steps), index=idx, name="rw")

    return _make


@pytest.fixture
def cointegrated_prices(simulated_ou):
    """Return a factory producing a cointegrated pair of strictly positive prices."""

    def _make(
        rng: np.random.Generator,
        beta_true: float = 1.5,
        n: int = 1000,
        alpha_true: float = 0.0,
    ) -> tuple[pd.Series, pd.Series]:
        # x is a log-random-walk; spread = log(y) - beta*log(x) - alpha is OU.
        x_log_steps = rng.standard_normal(n) * 0.01
        x_log = 4.0 + np.cumsum(x_log_steps)
        spread = simulated_ou(rng, theta=0.05, mu=0.0, sigma=0.05, n=n, dt=1.0, s0=0.0)
        y_log = alpha_true + beta_true * x_log + spread.to_numpy()
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        y = pd.Series(np.exp(y_log), index=idx, name="y_px")
        x = pd.Series(np.exp(x_log), index=idx, name="x_px")
        return y, x

    return _make
