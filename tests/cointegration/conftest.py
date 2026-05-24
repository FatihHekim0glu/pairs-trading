"""Shared fixtures for the cointegration test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic seed for cross-test reproducibility."""
    return np.random.default_rng(seed=20260523)


def _ar1(n: int, rho: float, sigma: float, gen: np.random.Generator) -> np.ndarray:
    eps = gen.standard_normal(n) * sigma
    out = np.empty(n, dtype=float)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + eps[i]
    return out


@pytest.fixture
def synthetic_coint_pair(rng: np.random.Generator):
    """Return ``(X, Y)`` where ``Y = beta * X + AR(1) residual``."""

    def _make(
        t: int = 500,
        beta_true: float = 1.0,
        rho_residual: float = 0.5,
        sigma_x: float = 1.0,
        sigma_eps: float = 0.5,
        x0: float = 100.0,
    ) -> tuple[pd.Series, pd.Series]:
        eta = rng.standard_normal(t) * sigma_x
        x = x0 + np.cumsum(eta)
        resid = _ar1(t, rho_residual, sigma_eps, rng)
        y = beta_true * x + resid
        idx = pd.RangeIndex(t)
        return pd.Series(x, index=idx, name="X"), pd.Series(y, index=idx, name="Y")

    return _make


@pytest.fixture
def two_random_walks(rng: np.random.Generator):
    """Return two independent Gaussian random walks of equal length."""

    def _make(t: int = 500, x0: float = 100.0, y0: float = 100.0) -> tuple[pd.Series, pd.Series]:
        x = x0 + np.cumsum(rng.standard_normal(t))
        y = y0 + np.cumsum(rng.standard_normal(t))
        idx = pd.RangeIndex(t)
        return pd.Series(x, index=idx, name="X"), pd.Series(y, index=idx, name="Y")

    return _make


@pytest.fixture
def stationary_series(rng: np.random.Generator):
    """Return a single mean-zero AR(1) series."""

    def _make(t: int = 500, rho: float = 0.5, sigma: float = 1.0) -> pd.Series:
        return pd.Series(_ar1(t, rho, sigma, rng), name="S")

    return _make
