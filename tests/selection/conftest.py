"""Shared fixtures for the selection tests.

Synthetic series are generated with NumPy's deterministic ``Generator``
spawned from the library's :func:`pairs._rng.default_rng` helper so test
behaviour is identical across machines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._rng import default_rng


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic RNG seeded from the library default."""
    return default_rng(seed=20260523)


@pytest.fixture
def ar1_series(rng: np.random.Generator):
    """Factory: produce a length-``T`` AR(1) series with autoregressive coef ``rho``."""

    def _factory(rho: float = 0.5, T: int = 2000) -> np.ndarray:
        eps = rng.standard_normal(T)
        x = np.empty(T, dtype=np.float64)
        x[0] = eps[0]
        for t in range(1, T):
            x[t] = rho * x[t - 1] + eps[t]
        return x

    return _factory


@pytest.fixture
def random_walk(rng: np.random.Generator):
    """Factory: produce a length-``T`` cumulative sum of iid Normal shocks."""

    def _factory(T: int = 2000) -> np.ndarray:
        return np.cumsum(rng.standard_normal(T)).astype(np.float64)

    return _factory


@pytest.fixture
def synthetic_universe(rng: np.random.Generator):
    """Factory: a list of fake tickers ``["T00", "T01", ...]``."""

    def _factory(n_tickers: int = 10) -> list[str]:
        return [f"T{ix:02d}" for ix in range(n_tickers)]

    _ = rng  # consume to satisfy the fixture contract while staying deterministic
    return _factory


@pytest.fixture
def synthetic_prices_panel(rng: np.random.Generator):
    """Factory: wide ``(T, n_tickers)`` price panel of correlated random walks.

    Returned values are positive prices so they can be log-transformed
    safely by downstream estimators.
    """

    def _factory(n_tickers: int = 10, T: int = 2000) -> pd.DataFrame:
        index = pd.date_range("2010-01-01", periods=T, freq="B")
        latent = rng.standard_normal((T, 1))
        idio = rng.standard_normal((T, n_tickers))
        shocks = 0.6 * latent + 0.4 * idio
        log_prices = 4.0 + np.cumsum(shocks * 0.01, axis=0)
        prices = np.exp(log_prices)
        tickers = [f"T{ix:02d}" for ix in range(n_tickers)]
        return pd.DataFrame(prices, index=index, columns=tickers)

    return _factory
