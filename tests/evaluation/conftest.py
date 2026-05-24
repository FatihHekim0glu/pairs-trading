"""Fixtures for the evaluation test suite."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Five-year daily price panel of three cointegrated-ish series."""
    t = 1260
    index = pd.bdate_range("2018-01-01", periods=t)
    common = np.cumsum(rng.normal(0.0, 1.0, size=t))
    noise = rng.normal(0.0, 0.5, size=(t, 3))
    levels = 100.0 + common[:, None] + np.cumsum(noise, axis=0) * 0.1
    return pd.DataFrame(levels, index=index, columns=["A", "B", "C"])


@pytest.fixture
def iid_returns_matrix(rng: np.random.Generator) -> pd.DataFrame:
    """``(2000, 50)`` matrix of IID Gaussian returns."""
    arr = rng.normal(0.0, 0.01, size=(2000, 50))
    return pd.DataFrame(arr, columns=[f"s{i}" for i in range(50)])


@pytest.fixture
def synthetic_strategy_returns(rng: np.random.Generator) -> pd.DataFrame:
    """``(500, 20)`` strategy panel with one slightly better column."""
    arr = rng.normal(0.0, 0.01, size=(500, 20))
    arr[:, 0] += 0.0005
    index = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame(arr, index=index, columns=[f"strat_{i}" for i in range(20)])


@pytest.fixture
def mock_pair_selector() -> Callable[[pd.DataFrame], Any]:
    def _selector(train_prices: pd.DataFrame) -> dict[str, str]:
        cols = list(train_prices.columns)
        return {"long": cols[0], "short": cols[1]}

    return _selector


@pytest.fixture
def mock_pair_backtester(
    rng: np.random.Generator,
) -> Callable[[pd.DataFrame, Any], pd.Series]:
    def _backtester(test_prices: pd.DataFrame, _selection: Any) -> pd.Series:
        # Deterministic-ish returns of small magnitude; uses test_prices index.
        sigma = 0.001
        # Use a separate generator seeded from the panel hash for determinism
        # across calls with the same input.
        seed = int(abs(hash(tuple(test_prices.index.astype("int64").to_numpy()[:5].tolist()))) % (2**32))
        local = np.random.default_rng(seed + int(rng.integers(0, 1_000_000)))
        ret = local.normal(0.0002, sigma, size=test_prices.shape[0])
        return pd.Series(ret, index=test_prices.index)

    return _backtester
