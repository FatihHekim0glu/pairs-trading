"""Shared fixtures for the portfolio test suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pairs.portfolio import PairLifecycle


@dataclass
class MockBacktestResult:
    """Minimal stand-in for `pairs.backtest.BacktestResult`."""

    returns: pd.Series
    equity: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)


def _make_returns_panel(rng: np.random.Generator, n_days: int, n_pairs: int) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    data = rng.normal(loc=0.0003, scale=0.01, size=(n_days, n_pairs))
    cols = [f"P{i}" for i in range(n_pairs)]
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def toy_5pair_returns(rng: np.random.Generator) -> pd.DataFrame:
    return _make_returns_panel(rng, n_days=500, n_pairs=5)


@pytest.fixture
def correlated_spread_pnls(rng: np.random.Generator) -> pd.DataFrame:
    n_days, n_pairs, rho = 500, 10, 0.6
    common = rng.normal(size=n_days)
    idiosyncratic = rng.normal(size=(n_days, n_pairs))
    a = float(np.sqrt(rho))
    b = float(np.sqrt(1.0 - rho))
    data = a * common[:, None] + b * idiosyncratic
    data *= 0.01
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    cols = [f"P{i}" for i in range(n_pairs)]
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def mock_backtest_factory():
    def _make(returns: pd.Series) -> MockBacktestResult:
        equity = (1.0 + returns.fillna(0.0)).cumprod()
        return MockBacktestResult(returns=returns, equity=equity)

    return _make


@pytest.fixture
def mock_lifecycle() -> PairLifecycle:
    return PairLifecycle(
        cointegration_retest=lambda *_args, **_kwargs: SimpleNamespace(cointegrated=True),
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=10,
    )


@pytest.fixture
def sector_and_asset_maps():
    sector_map = {f"P{i}": f"SEC{i % 2}" for i in range(20)}
    asset_legs_map = {f"P{i}": (f"A{i}", f"A{i + 100}") for i in range(20)}
    return sector_map, asset_legs_map
